"""프로젝트 전체 실행의 시작점을 담당하는 파일이다.

이 파일의 역할은 "실험을 실제로 돌리는 계산" 자체보다는
"어떤 설정으로 어떤 프로파일을 어떻게 실행할지 정리하고 연결하는 것"에 가깝다.

즉 이 파일은 다음 순서를 관리한다.

1. 어떤 프로파일을 실행할지 결정한다.
2. 평가 데이터 생성 방식을 어떻게 할지 결정한다.
3. 저장된 시드를 재사용할지, 랜덤 시드로 새 학습을 할지 결정한다.
4. 각 프로파일/시드 조합을 experiment_runner에 넘겨 실제 학습/평가를 수행한다.
5. 실행 결과를 요약 CSV와 그래프로 저장한다.

실제 LoRa waveform 생성, CNN 학습, hybrid policy calibration 같은 무거운 계산은
다른 파일이 담당하고, 이 파일은 그 흐름을 조립한다.
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
    """평가 데이터 생성 방식을 사용자에게 입력받는다.

    이 함수가 반환하는 값은 bool 하나다.

    - False:
      SNR마다 독립적인 label/CFO/channel_state를 새로 뽑는다.
      즉 각 SNR 지점이 서로 다른 평가 샘플셋을 가진다.
    - True:
      label/CFO/channel_state를 SNR 전 구간에서 공유한다.
      즉 평가 조건을 더 통제한 상태로 SNR 변화만 보기 쉬워진다.
    """

    print("평가 방식을 선택하세요.")
    print("  1. SNR마다 독립 realization")
    print("  2. 같은 channel state를 SNR 전 구간에서 재사용")

    while True:
        # input()은 터미널에서 사용자의 문자열 입력을 읽는다.
        selected = input("입력 (1-2): ").strip()

        # 1번은 독립 realization이다.
        if selected == "1":
            print("-> SNR마다 독립 realization으로 평가합니다.")
            return False

        # 2번은 channel state를 공유하는 통제 평가다.
        if selected == "2":
            print("-> 같은 channel state를 SNR 전 구간에서 재사용합니다.")
            return True

        # 1, 2 외 값은 다시 입력받는다.
        print("1 또는 2를 입력해야 합니다.")


def _select_profiles(receiver_profiles):
    """실행할 receiver profile 범위를 선택한다.

    receiver_profiles는 config.py에 들어 있는 프로파일 목록이다.
    예를 들면
    - sf7_bw125
    - sf8_bw125
    - sf9_bw250
    같은 항목이 들어 있다.

    반환값은 항상 "선택된 프로파일 dict 목록"이다.
    즉 하나만 고르면 길이 1짜리 리스트,
    전체를 고르면 전체 리스트를 반환한다.
    """

    print("실행할 프로파일을 선택하세요.")

    # 각 프로파일에 번호를 붙여 출력한다.
    for idx, profile in enumerate(receiver_profiles, start=1):
        print(f"  {idx}. {profile['name']}")

    # 마지막 번호는 "전체 실행" 옵션이다.
    all_option = len(receiver_profiles) + 1
    print(f"  {all_option}. 전체")

    while True:
        selected = input(f"입력 (1-{all_option}): ").strip()

        # 숫자가 아니면 무효 입력이다.
        if not selected.isdigit():
            print("숫자를 입력해야 합니다.")
            continue

        selected_idx = int(selected)

        # 개별 프로파일 하나를 선택한 경우다.
        if 1 <= selected_idx <= len(receiver_profiles):
            chosen_profile = receiver_profiles[selected_idx - 1]
            print(f"-> {chosen_profile['name']}만 실행합니다.")
            return [chosen_profile]

        # 전체 실행 옵션을 선택한 경우다.
        if selected_idx == all_option:
            print("-> 모든 프로파일을 실행합니다.")
            return receiver_profiles

        # 범위를 벗어나면 다시 입력받는다.
        print(f"1부터 {all_option} 사이 숫자를 입력해야 합니다.")


def _discover_saved_artifacts(receiver_profiles, artifact_cfg):
    """선택한 프로파일들에 대해 저장된 weights/checkpoint를 스캔한다.

    이 함수는 두 가지를 반환한다.

    1. artifacts_by_profile
       - 프로파일 이름별로
       - 어떤 seed artifact가 저장돼 있는지
       - 그리고 그 파일 경로가 무엇인지
       를 담은 dict이다.

    2. common_saved_seeds
       - 선택한 모든 프로파일에 공통으로 존재하는 seed 목록이다.
       - 예를 들어 sf7에는 2080이 있고 sf8에는 없으면,
         sf7+sf8를 동시에 고를 때는 공통 saved seed가 없다고 본다.
    """

    # 선택된 프로파일 이름만 집합으로 뽑아 둔다.
    # 이후 파일명을 읽을 때 "선택하지 않은 프로파일" artifact는 무시하기 위해 쓴다.
    profile_names = {profile["name"] for profile in receiver_profiles}

    # weights_dir / checkpoints_dir는 config.py의 artifacts 설정에서 가져온다.
    weights_dir = artifact_cfg.get("weights_dir", "artifacts/weights")
    checkpoints_dir = artifact_cfg.get("checkpoints_dir", "artifacts/checkpoints")

    # 저장 파일 이름 규칙을 정규표현식으로 정의한다.
    # 예:
    # sf7_bw125_seed2080_best_weights.pth
    weights_pattern = re.compile(r"^(?P<profile>.+)_seed(?P<seed>\d+)_best_weights\.pth$")

    # 예:
    # sf7_bw125_seed2080_best_checkpoint.pt
    checkpoint_pattern = re.compile(r"^(?P<profile>.+)_seed(?P<seed>\d+)_best_checkpoint\.pt$")

    # 프로파일별 artifact 저장용 dict를 미리 만든다.
    artifacts_by_profile = {name: {} for name in profile_names}

    # weights 폴더가 존재하면 파일들을 훑는다.
    if os.path.isdir(weights_dir):
        for filename in os.listdir(weights_dir):
            match = weights_pattern.match(filename)

            # 이름 규칙이 맞지 않으면 무시한다.
            if not match:
                continue

            profile_name = match.group("profile")

            # 현재 선택한 프로파일에 해당하지 않으면 무시한다.
            if profile_name not in profile_names:
                continue

            # 파일명에서 seed를 정수로 뽑는다.
            seed = int(match.group("seed"))

            # 절대경로 형태로 저장한다.
            artifacts_by_profile[profile_name][seed] = os.path.abspath(
                os.path.join(weights_dir, filename)
            )

    # checkpoint 폴더도 같은 방식으로 훑는다.
    if os.path.isdir(checkpoints_dir):
        for filename in os.listdir(checkpoints_dir):
            match = checkpoint_pattern.match(filename)
            if not match:
                continue

            profile_name = match.group("profile")
            if profile_name not in profile_names:
                continue

            seed = int(match.group("seed"))

            # setdefault를 쓰는 이유는
            # 같은 seed에 weights 파일이 이미 있으면 그걸 우선 쓰고,
            # 없을 때만 checkpoint 경로를 넣기 위해서다.
            artifacts_by_profile[profile_name].setdefault(
                seed,
                os.path.abspath(os.path.join(checkpoints_dir, filename)),
            )

    # 선택한 프로파일들 각각의 저장 시드 집합을 구한다.
    if artifacts_by_profile:
        seed_sets = [
            set(profile_artifacts.keys())
            for profile_artifacts in artifacts_by_profile.values()
        ]

        # 모든 프로파일에 공통으로 있는 seed만 남긴다.
        common_saved_seeds = sorted(set.intersection(*seed_sets)) if seed_sets else []
    else:
        common_saved_seeds = []

    return artifacts_by_profile, common_saved_seeds


def _select_seed(saved_seeds):
    """저장된 시드를 재사용할지, 랜덤 시드를 새로 만들지 결정한다.

    반환값은 `(seed_value, selected_saved_seed)` 형태다.

    - seed_value:
      실제 사용할 정수 seed
    - selected_saved_seed:
      True면 저장 artifact를 재사용하는 평가-only 의미
      False면 새 랜덤 시드로 학습 허용 의미

    사용자가 종료를 선택하면 `(None, None)`을 반환한다.
    """

    # 공통 저장 시드가 하나도 없으면,
    # "랜덤 시드로 새 학습" 또는 "종료"만 고를 수 있다.
    if not saved_seeds:
        print("선택한 프로파일 조합에 공통으로 저장된 가중치가 없습니다.")
        print("  1. 랜덤 시드로 새 학습 시작")
        print("  2. 종료")

        while True:
            selected = input("입력 (1-2): ").strip()

            if selected == "1":
                # SystemRandom은 운영체제 난수원 기반으로 새 seed를 뽑는다.
                generated_seed = random.SystemRandom().randint(1000, 999999)
                print(f"-> 랜덤 시드 {generated_seed}로 새 학습을 시작합니다.")
                return generated_seed, False

            if selected == "2":
                print("-> 실행을 종료합니다.")
                return None, None

            print("1 또는 2를 입력해야 합니다.")

    # 저장된 시드가 있으면 목록을 보여주고 선택받는다.
    print("저장된 가중치 시드를 선택하세요.")
    for idx, seed in enumerate(saved_seeds, start=1):
        print(f"  {idx}. {seed}")

    # 마지막 옵션은 "저장된 것 말고 새 랜덤 시드"다.
    random_option = len(saved_seeds) + 1
    print(f"  {random_option}. 랜덤 시드")

    while True:
        selected = input(f"입력 (1-{random_option}): ").strip()

        if not selected.isdigit():
            print("숫자를 입력해야 합니다.")
            continue

        selected_idx = int(selected)

        # 저장된 시드를 고른 경우
        if 1 <= selected_idx <= len(saved_seeds):
            selected_seed = saved_seeds[selected_idx - 1]
            print(f"-> 저장된 시드 {selected_seed}를 사용합니다.")
            return selected_seed, True

        # 랜덤 시드를 새로 뽑는 경우
        if selected_idx == random_option:
            generated_seed = random.SystemRandom().randint(1000, 999999)
            print(f"-> 랜덤 시드 {generated_seed}를 사용합니다.")
            return generated_seed, False

        print(f"1부터 {random_option} 사이 숫자를 입력해야 합니다.")


def _profile_output_dir(profile_name, shared_channel_state):
    """프로파일 이름과 평가 방식으로 결과 저장 폴더를 만든다.

    예를 들어
    - profile_name = "sf7_bw125"
    - shared_channel_state = False
    라면
    보통은 `sf7/independent_realization`
    형태의 폴더를 반환한다.

    다만 나중에 같은 SF 아래에 서로 다른 BW 프로파일이 함께 생기면
    `sf7/sf7_bw125/independent_realization`
    처럼 profile 이름을 한 단계 더 넣어 파일이 섞이지 않게 한다.
    """

    # "sf7_bw125" -> "sf7"
    sf_dir = profile_name.split("_")[0]

    # 평가 조건에 따라 하위 폴더 이름을 다르게 준다.
    mode_dir = "shared_channel_state" if shared_channel_state else "independent_realization"

    # 같은 SF prefix를 가진 프로파일이 여러 개면 결과가 섞일 수 있으므로,
    # profile 이름 디렉터리를 한 단계 더 넣어 충돌을 막는다.
    same_sf_profiles = [
        profile["name"]
        for profile in CFG["receiver_profiles"]
        if profile["name"].split("_")[0] == sf_dir
    ]

    if len(same_sf_profiles) > 1:
        return os.path.join(sf_dir, profile_name, mode_dir)

    return os.path.join(sf_dir, mode_dir)


def _save_outputs_by_profile(summary, latency_summary, receiver_profiles, shared_channel_state):
    """프로파일별 폴더에 CSV와 그래프를 저장한다.

    summary / latency_summary는 모든 실행 결과가 한 DataFrame으로 합쳐진 상태다.
    이 함수는 그중에서 프로파일별 행만 잘라
    각 프로파일 폴더에 따로 저장한다.
    """

    # 파일명 충돌을 피하기 위해 현재 시각을 timestamp로 만든다.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 실제로 저장한 디렉터리 목록을 모아
    # 마지막에 사용자에게 한 번에 보여주기 위해 사용한다.
    saved_dirs = []

    for receiver_profile in receiver_profiles:
        profile_name = receiver_profile["name"]

        # summary DataFrame에서 현재 프로파일 결과만 자른다.
        profile_summary = summary[summary["profile"] == profile_name]

        # 이 프로파일 결과가 없으면 건너뛴다.
        if profile_summary.empty:
            continue

        # latency 요약도 동일하게 현재 프로파일만 자른다.
        profile_latency = latency_summary[latency_summary["profile"] == profile_name]

        # 저장 폴더를 결정한다.
        output_dir = _profile_output_dir(profile_name, shared_channel_state)

        # experiment summary CSV 저장
        save_experiment_summary_csv(
            profile_summary,
            csv_dir=output_dir,
            filename=f"experiment_summary_{timestamp}.csv",
        )

        # latency summary CSV 저장
        # latency는 없을 수도 있으므로 비어 있지 않을 때만 저장한다.
        if not profile_latency.empty:
            save_latency_summary_csv(
                profile_latency,
                csv_dir=output_dir,
                filename=f"latency_summary_{timestamp}.csv",
            )

        # summary 그래프 저장
        plot_summary(
            profile_summary,
            graph_dir=output_dir,
            filename_suffix=f"_{timestamp}",
        )

        # global vs bin ablation 그래프 저장
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
):
    """설정된 receiver profile들에 대해 학습/평가 전체를 수행한다.

    이 함수는 프로젝트의 진짜 실행 진입점이다.

    사용 방식은 두 가지다.

    1. interactive=True
       - 터미널에서 프로파일, 평가 방식, 시드를 직접 고른다.
    2. interactive=False
       - 다른 스크립트에서 `main(...)`으로 호출하면서 override 값을 직접 넣는다.

    주요 인자 의미:

    - shared_channel_state_override:
      평가 데이터 생성 시 packet 조건을 SNR 전 구간에서 공유할지 여부
    - seed_override:
      config.py의 기본 seed 목록 대신 특정 seed 하나만 쓸지 여부
    - allow_training_without_saved_artifact:
      저장된 가중치가 없는 seed를 줬을 때 새 학습을 허용할지 여부
    """

    # 아래 값들은 config.py의 전역 기본 설정이다.
    # 각 프로파일별 override가 있으면 이후 build_profile_runtime_configs에서 덮어쓴다.
    base_train_cfg = CFG["training"]
    base_feature_cfg = CFG["feature_bank"]
    base_model_cfg = CFG.get("model", {})
    base_benchmark_cfg = CFG["benchmark"]
    artifact_cfg = CFG.get("artifacts", {})
    hybrid_cfg = CFG["hybrid"]
    channel_profiles = CFG["channel_profiles"]

    # 기본적으로는 config.py에 들어 있는 모든 receiver profile을 대상으로 한다.
    receiver_profiles = CFG["receiver_profiles"]
    selected_receiver_profiles = receiver_profiles

    # interactive 모드면 사용자 선택을 받는다.
    if interactive and shared_channel_state_override is None:
        # 먼저 어떤 프로파일을 실행할지 고른다.
        selected_receiver_profiles = _select_profiles(receiver_profiles)

        # 그다음 평가 데이터 생성 방식을 고른다.
        shared_channel_state_override = _select_eval_mode()

    elif interactive:
        # 이미 override 값이 바깥에서 들어온 경우에는
        # 평가 방식은 묻지 않고 프로파일만 선택받는다.
        selected_receiver_profiles = _select_profiles(receiver_profiles)

    # 선택한 프로파일들에 대해 저장 artifact를 스캔한다.
    artifacts_by_profile, saved_seeds = _discover_saved_artifacts(
        selected_receiver_profiles,
        artifact_cfg,
    )

    # selected_saved_seed=True는 "저장 artifact 재사용" 모드라는 뜻이다.
    selected_saved_seed = False

    # interactive 모드에서 seed를 직접 고르지 않았다면 사용자 입력을 받는다.
    if interactive and seed_override is None:
        seed_override, selected_saved_seed = _select_seed(saved_seeds)

        # 사용자가 종료를 골랐으면 main을 바로 끝낸다.
        if seed_override is None and selected_saved_seed is None:
            return

    # 반대로 seed_override가 코드에서 직접 주어진 경우에는
    # 그 seed가 선택한 모든 프로파일에 공통으로 존재하는 artifact인지 확인한다.
    elif seed_override is not None:
        requested_seed = int(seed_override)

        selected_saved_seed = all(
            requested_seed in artifacts_by_profile.get(profile["name"], {})
            for profile in selected_receiver_profiles
        )

        # 저장 artifact가 모든 프로파일에 존재하지 않는데
        # 새 학습도 허용하지 않았다면 여기서 에러를 낸다.
        if not selected_saved_seed and not allow_training_without_saved_artifact:
            raise RuntimeError(
                f"Seed {requested_seed} does not exist for every selected profile. "
                "Use an interactive random seed or set allow_training_without_saved_artifact=True."
            )

    # experiment 설정은 dict(CFG["experiment"])로 얕은 복사를 만든다.
    # 이후 이 복사본에만 override를 적용해 원본 config를 직접 망가뜨리지 않는다.
    base_experiment_cfg = dict(CFG["experiment"])

    # 평가 방식 override가 있으면 여기서 덮어쓴다.
    if shared_channel_state_override is not None:
        base_experiment_cfg["shared_channel_state_across_snr"] = shared_channel_state_override

    # 특정 seed 하나만 강제로 쓰고 싶으면 여기서 seeds 리스트를 바꾼다.
    if seed_override is not None:
        base_experiment_cfg["seeds"] = [int(seed_override)]

    # CUDA가 가능하면 pin_memory를 켜 DataLoader host->device 전송 효율을 높인다.
    pin_memory = torch.cuda.is_available()

    # 모든 프로파일/seed 결과를 마지막에 합쳐 summary DataFrame으로 만들기 위해 모아 둔다.
    all_runs = []
    latency_rows = []

    # 선택된 프로파일을 하나씩 실행한다.
    for receiver_profile in selected_receiver_profiles:
        # 원본 profile dict를 직접 수정하지 않기 위해 deep copy를 만든다.
        runtime_profile = copy.deepcopy(receiver_profile)

        # 특정 seed가 지정된 경우,
        # 현재 프로파일에 해당 seed artifact가 있으면 checkpoint_path를 주입한다.
        if seed_override is not None:
            artifact_path = artifacts_by_profile.get(runtime_profile["name"], {}).get(int(seed_override))

            if artifact_path:
                runtime_profile["checkpoint_path"] = artifact_path
                print(
                    f"-> Saved artifact found for {runtime_profile['name']} / seed {seed_override}: {artifact_path}"
                )
            else:
                # 현재 프로파일에 artifact가 없으면 checkpoint_path를 제거한다.
                runtime_profile.pop("checkpoint_path", None)

                # 그런데 사용자가 "저장된 시드 평가"를 고른 경우라면
                # artifact가 반드시 있어야 하므로 여기서 에러를 낸다.
                if selected_saved_seed:
                    raise RuntimeError(
                        f"Saved-seed mode was selected, but no artifact exists for "
                        f"{runtime_profile['name']} / seed {seed_override}."
                    )

        # 기본 config와 profile별 override를 병합해
        # 현재 프로파일의 실제 runtime 설정을 만든다.
        experiment_cfg, train_cfg, feature_cfg, model_cfg, benchmark_cfg = (
            build_profile_runtime_configs(
                runtime_profile,
                base_experiment_cfg,
                base_train_cfg,
                base_feature_cfg,
                base_model_cfg,
                base_benchmark_cfg,
            )
        )

        # 수동 checkpoint 평가에서는
        # 같은 artifact를 여러 seed 결과처럼 집계하면 통계 해석이 틀어진다.
        # 그래서 evaluation-only 경로는 seed 하나만 허용한다.
        if runtime_profile.get("checkpoint_path") and len(experiment_cfg["seeds"]) != 1:
            raise RuntimeError(
                f"Manual checkpoint evaluation for {runtime_profile['name']} requires exactly one seed, "
                f"but got {experiment_cfg['seeds']}."
            )

        print(f"\n{'=' * 60}\n[PROFILE: {runtime_profile['name']}]\n{'=' * 60}")

        # 현재 프로파일에 대해 실행할 seed를 순회한다.
        for seed in experiment_cfg["seeds"]:
            print(f"\n{'-' * 60}\n[SEED: {seed}]\n{'-' * 60}")

            # 실제 학습/평가 로직은 experiment_runner가 수행한다.
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

            # profile/seed 실행 결과를 전역 목록에 이어붙인다.
            all_runs.extend(run_rows)
            latency_rows.append(latency_row)

    # 모든 raw row를 profile/type/snr 기준 summary DataFrame으로 만든다.
    summary = build_experiment_summary(all_runs)

    # latency도 별도 summary DataFrame으로 만든다.
    latency_summary = build_latency_summary(latency_rows)

    # 현재 실제로 적용된 평가 방식(bool)을 다시 읽어온다.
    effective_shared_channel_state = base_experiment_cfg.get("shared_channel_state_across_snr", False)

    # 프로파일별 폴더에 CSV와 그래프를 저장한다.
    timestamp, saved_dirs = _save_outputs_by_profile(
        summary,
        latency_summary,
        selected_receiver_profiles,
        effective_shared_channel_state,
    )

    # 같은 디렉터리가 중복 출력되지 않게 순서를 유지한 채 unique 처리한다.
    unique_dirs = list(dict.fromkeys(saved_dirs))

    if unique_dirs:
        print("Saved CSV and plot files to:")
        for output_dir in unique_dirs:
            print(f"  .\\{output_dir}")
        print(f"Timestamp suffix: {timestamp}")
    else:
        print("No output files were saved because no summary rows were produced.")


if __name__ == "__main__":
    # 직접 `python main.py`로 실행한 경우에는
    # 사용자 입력을 받는 interactive 모드로 들어간다.
    main(interactive=True)
