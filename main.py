"""프로젝트 실행 진입점이다.

직접 `python main.py`로 실행하면
1. 실행할 프로파일
2. 평가 데이터 생성 방식
3. 저장된 시드 또는 랜덤 시드
를 차례로 선택한다.

다른 스크립트에서 `import main; main.main(...)`으로 호출하면
interactive 프롬프트 없이 override 값만 적용해 사용할 수 있다.
"""

import copy
from datetime import datetime
import os
import random
import re

import torch

from config import CFG
from experiment_runner import build_profile_runtime_configs, run_profile_seed
from plotting import plot_policy_ablation, plot_summary
from results_io import (
    build_experiment_summary,
    build_latency_summary,
    save_experiment_summary_csv,
    save_latency_summary_csv,
)


def _select_eval_mode():
    """평가 데이터 생성 방식을 선택한다."""

    print("평가 방식을 선택하세요.")
    print("  1. SNR마다 독립 realization")
    print("  2. 같은 channel state를 SNR 전 구간에서 재사용")

    while True:
        selected = input("입력 (1-2): ").strip()
        if selected == "1":
            print("-> SNR마다 독립 realization으로 평가합니다.")
            return False
        if selected == "2":
            print("-> 같은 channel state를 SNR 전 구간에서 재사용합니다.")
            return True
        print("1 또는 2를 입력해야 합니다.")


def _select_profiles(receiver_profiles):
    """실행할 프로파일 범위를 선택한다."""

    print("실행할 프로파일을 선택하세요.")
    for idx, profile in enumerate(receiver_profiles, start=1):
        print(f"  {idx}. {profile['name']}")
    all_option = len(receiver_profiles) + 1
    print(f"  {all_option}. 전체")

    while True:
        selected = input(f"입력 (1-{all_option}): ").strip()
        if not selected.isdigit():
            print("숫자를 입력해야 합니다.")
            continue

        selected_idx = int(selected)
        if 1 <= selected_idx <= len(receiver_profiles):
            chosen_profile = receiver_profiles[selected_idx - 1]
            print(f"-> {chosen_profile['name']}만 실행합니다.")
            return [chosen_profile]
        if selected_idx == all_option:
            print("-> 모든 프로파일을 실행합니다.")
            return receiver_profiles

        print(f"1부터 {all_option} 사이 숫자를 입력해야 합니다.")


def _discover_saved_artifacts(receiver_profiles, artifact_cfg):
    """선택한 프로파일들에 대해 저장된 artifact와 공통 저장 시드를 찾는다."""

    profile_names = {profile["name"] for profile in receiver_profiles}
    weights_dir = artifact_cfg.get("weights_dir", "artifacts/weights")
    checkpoints_dir = artifact_cfg.get("checkpoints_dir", "artifacts/checkpoints")

    weights_pattern = re.compile(r"^(?P<profile>.+)_seed(?P<seed>\d+)_best_weights\.pth$")
    checkpoint_pattern = re.compile(r"^(?P<profile>.+)_seed(?P<seed>\d+)_best_checkpoint\.pt$")

    artifacts_by_profile = {name: {} for name in profile_names}

    if os.path.isdir(weights_dir):
        for filename in os.listdir(weights_dir):
            match = weights_pattern.match(filename)
            if not match:
                continue
            profile_name = match.group("profile")
            if profile_name not in profile_names:
                continue
            seed = int(match.group("seed"))
            artifacts_by_profile[profile_name][seed] = os.path.abspath(os.path.join(weights_dir, filename))

    if os.path.isdir(checkpoints_dir):
        for filename in os.listdir(checkpoints_dir):
            match = checkpoint_pattern.match(filename)
            if not match:
                continue
            profile_name = match.group("profile")
            if profile_name not in profile_names:
                continue
            seed = int(match.group("seed"))
            artifacts_by_profile[profile_name].setdefault(
                seed,
                os.path.abspath(os.path.join(checkpoints_dir, filename)),
            )

    if artifacts_by_profile:
        seed_sets = [set(profile_artifacts.keys()) for profile_artifacts in artifacts_by_profile.values()]
        common_saved_seeds = sorted(set.intersection(*seed_sets)) if seed_sets else []
    else:
        common_saved_seeds = []

    return artifacts_by_profile, common_saved_seeds


def _select_seed(saved_seeds):
    """저장된 시드 또는 랜덤 시드 중 하나를 선택한다."""

    if not saved_seeds:
        print("선택한 프로파일 조합에 공통으로 저장된 가중치가 없습니다.")
        print("  1. 랜덤 시드로 새 학습 시작")
        print("  2. 종료")

        while True:
            selected = input("입력 (1-2): ").strip()
            if selected == "1":
                generated_seed = random.SystemRandom().randint(1000, 999999)
                print(f"-> 랜덤 시드 {generated_seed}로 새 학습을 시작합니다.")
                return generated_seed, False
            if selected == "2":
                print("-> 실행을 종료합니다.")
                return None, None
            print("1 또는 2를 입력해야 합니다.")

    print("저장된 가중치 시드를 선택하세요.")
    for idx, seed in enumerate(saved_seeds, start=1):
        print(f"  {idx}. {seed}")
    random_option = len(saved_seeds) + 1
    print(f"  {random_option}. 랜덤 시드")

    while True:
        selected = input(f"입력 (1-{random_option}): ").strip()
        if not selected.isdigit():
            print("숫자를 입력해야 합니다.")
            continue

        selected_idx = int(selected)
        if 1 <= selected_idx <= len(saved_seeds):
            selected_seed = saved_seeds[selected_idx - 1]
            print(f"-> 저장된 시드 {selected_seed}를 사용합니다.")
            return selected_seed, True
        if selected_idx == random_option:
            generated_seed = random.SystemRandom().randint(1000, 999999)
            print(f"-> 랜덤 시드 {generated_seed}를 사용합니다.")
            return generated_seed, False

        print(f"1부터 {random_option} 사이 숫자를 입력해야 합니다.")


def _profile_output_dir(profile_name, shared_channel_state):
    """프로파일 이름과 평가 방식으로 결과 저장 폴더를 만든다."""

    sf_dir = profile_name.split("_")[0]
    mode_dir = "shared_channel_state" if shared_channel_state else "independent_realization"
    return os.path.join(sf_dir, mode_dir)


def _save_outputs_by_profile(summary, latency_summary, receiver_profiles, shared_channel_state):
    """프로파일별 폴더에 CSV와 그래프를 각각 저장한다."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_dirs = []

    for receiver_profile in receiver_profiles:
        profile_name = receiver_profile["name"]
        profile_summary = summary[summary["profile"] == profile_name]
        if profile_summary.empty:
            continue

        profile_latency = latency_summary[latency_summary["profile"] == profile_name]
        output_dir = _profile_output_dir(profile_name, shared_channel_state)

        save_experiment_summary_csv(
            profile_summary,
            csv_dir=output_dir,
            filename=f"experiment_summary_{timestamp}.csv",
        )
        if not profile_latency.empty:
            save_latency_summary_csv(
                profile_latency,
                csv_dir=output_dir,
                filename=f"latency_summary_{timestamp}.csv",
            )

        plot_summary(
            profile_summary,
            graph_dir=output_dir,
            filename_suffix=f"_{timestamp}",
        )
        plot_policy_ablation(
            profile_summary,
            graph_dir=output_dir,
            filename_suffix=f"_{timestamp}",
        )
        saved_dirs.append(output_dir)

    return timestamp, saved_dirs


def main(
    shared_channel_state_override=None,
    seed_override=None,
    interactive=False,
    allow_training_without_saved_artifact=False,
    same_realization_override=None,
):
    """설정된 수신기 프로파일들에 대해 학습/평가를 수행한다."""

    # 이전 호출 코드 호환용 alias다.
    if shared_channel_state_override is None and same_realization_override is not None:
        shared_channel_state_override = same_realization_override

    base_train_cfg = CFG["training"]
    base_feature_cfg = CFG["feature_bank"]
    base_model_cfg = CFG.get("model", {})
    base_benchmark_cfg = CFG["benchmark"]
    artifact_cfg = CFG.get("artifacts", {})
    hybrid_cfg = CFG["hybrid"]
    channel_profiles = CFG["channel_profiles"]

    receiver_profiles = CFG["receiver_profiles"]
    selected_receiver_profiles = receiver_profiles

    if interactive and shared_channel_state_override is None:
        selected_receiver_profiles = _select_profiles(receiver_profiles)
        shared_channel_state_override = _select_eval_mode()
    elif interactive:
        selected_receiver_profiles = _select_profiles(receiver_profiles)

    artifacts_by_profile, saved_seeds = _discover_saved_artifacts(selected_receiver_profiles, artifact_cfg)

    selected_saved_seed = False
    if interactive and seed_override is None:
        seed_override, selected_saved_seed = _select_seed(saved_seeds)
        if seed_override is None and selected_saved_seed is None:
            return
    elif seed_override is not None:
        requested_seed = int(seed_override)
        selected_saved_seed = all(
            requested_seed in artifacts_by_profile.get(profile["name"], {})
            for profile in selected_receiver_profiles
        )
        if not selected_saved_seed and not allow_training_without_saved_artifact:
            raise RuntimeError(
                f"Seed {requested_seed} does not exist for every selected profile. "
                "Use an interactive random seed or set allow_training_without_saved_artifact=True."
            )

    base_experiment_cfg = dict(CFG["experiment"])
    if shared_channel_state_override is not None:
        base_experiment_cfg["shared_channel_state_across_snr"] = shared_channel_state_override
    if seed_override is not None:
        base_experiment_cfg["seeds"] = [int(seed_override)]

    pin_memory = torch.cuda.is_available()
    all_runs = []
    latency_rows = []

    for receiver_profile in selected_receiver_profiles:
        runtime_profile = copy.deepcopy(receiver_profile)

        if seed_override is not None:
            artifact_path = artifacts_by_profile.get(runtime_profile["name"], {}).get(int(seed_override))
            if artifact_path:
                runtime_profile["checkpoint_path"] = artifact_path
                print(f"-> Saved artifact found for {runtime_profile['name']} / seed {seed_override}: {artifact_path}")
            else:
                runtime_profile.pop("checkpoint_path", None)
                if selected_saved_seed:
                    raise RuntimeError(
                        f"Saved-seed mode was selected, but no artifact exists for "
                        f"{runtime_profile['name']} / seed {seed_override}."
                    )

        experiment_cfg, train_cfg, feature_cfg, model_cfg, benchmark_cfg = build_profile_runtime_configs(
            runtime_profile,
            base_experiment_cfg,
            base_train_cfg,
            base_feature_cfg,
            base_model_cfg,
            base_benchmark_cfg,
        )

        print(f"\n{'=' * 60}\n[PROFILE: {runtime_profile['name']}]\n{'=' * 60}")

        for seed in experiment_cfg["seeds"]:
            print(f"\n{'-' * 60}\n[SEED: {seed}]\n{'-' * 60}")

            run_rows, latency_row = run_profile_seed(
                runtime_profile,
                seed,
                artifact_cfg=artifact_cfg,
                hybrid_cfg=hybrid_cfg,
                channel_profiles=channel_profiles,
                experiment_cfg=experiment_cfg,
                train_cfg=train_cfg,
                feature_cfg=feature_cfg,
                model_cfg=model_cfg,
                benchmark_cfg=benchmark_cfg,
                pin_memory=pin_memory,
            )

            all_runs.extend(run_rows)
            latency_rows.append(latency_row)

    summary = build_experiment_summary(all_runs)
    latency_summary = build_latency_summary(latency_rows)

    effective_shared_channel_state = base_experiment_cfg.get("shared_channel_state_across_snr", False)
    timestamp, saved_dirs = _save_outputs_by_profile(
        summary,
        latency_summary,
        selected_receiver_profiles,
        effective_shared_channel_state,
    )

    unique_dirs = list(dict.fromkeys(saved_dirs))
    if unique_dirs:
        print("Saved CSV and plot files to:")
        for output_dir in unique_dirs:
            print(f"  .\\{output_dir}")
        print(f"Timestamp suffix: {timestamp}")
    else:
        print("No output files were saved because no summary rows were produced.")


if __name__ == "__main__":
    main(interactive=True)
