"""전체 실행 순서를 담당하는 진입점 파일이다.

이 파일의 역할은 "무엇을 어떤 순서로 호출할지"를 정리하는 것이다.
세부 구현은 다른 모듈로 분리되어 있으므로,
이 파일만 읽으면 프로그램 상위 흐름을 빠르게 파악할 수 있다.
"""

import torch

from config import CFG
from experiment_runner import build_profile_runtime_configs, run_profile_seed
from plotting import GRAPH_DIR, plot_policy_ablation, plot_summary
from results_io import (
    CSV_DIR,
    build_experiment_summary,
    build_latency_summary,
    save_experiment_summary_csv,
    save_latency_summary_csv,
)


def main():
    """설정된 모든 수신기 프로파일에 대해 학습과 평가를 수행한다."""

    # base_* 변수들은 config.py에 정의된 전역 기본 설정이다.
    # 각 receiver_profile은 필요할 때 이 기본값 위에 override를 덮어쓴다.
    base_experiment_cfg = CFG["experiment"]
    base_train_cfg = CFG["training"]
    base_feature_cfg = CFG["feature_bank"]
    base_model_cfg = CFG.get("model", {})
    base_benchmark_cfg = CFG["benchmark"]

    # artifact_cfg:
    # best weights / checkpoint 저장 위치와 저장 여부를 담는다.
    artifact_cfg = CFG.get("artifacts", {})

    # hybrid_cfg:
    # confidence 계산 방식과 tolerance 기준 등 하이브리드 정책 보정 설정이다.
    hybrid_cfg = CFG["hybrid"]

    # channel_profiles:
    # train / seen_eval / unseen_eval 채널 분포를 담는다.
    channel_profiles = CFG["channel_profiles"]

    # pin_memory:
    # GPU 환경이면 DataLoader에서 pinned memory를 써서 host-to-device 복사를 조금 더 빠르게 한다.
    pin_memory = torch.cuda.is_available()

    # all_runs:
    # 모든 profile/seed/SNR 조합의 raw 평가 결과를 누적한다.
    all_runs = []

    # latency_rows:
    # 각 profile/seed 조합의 benchmark 결과를 누적한다.
    latency_rows = []

    # receiver_profiles를 순회하면서 profile별 실행을 시작한다.
    for receiver_profile in CFG["receiver_profiles"]:
        # build_profile_runtime_configs:
        # 이 profile에 정의된 override를 기본 설정 위에 덮어 실제 실행 설정을 만든다.
        experiment_cfg, train_cfg, feature_cfg, model_cfg, benchmark_cfg = build_profile_runtime_configs(
            receiver_profile,
            base_experiment_cfg,
            base_train_cfg,
            base_feature_cfg,
            base_model_cfg,
            base_benchmark_cfg,
        )

        print(f"\n{'=' * 60}\n[PROFILE: {receiver_profile['name']}]\n{'=' * 60}")

        # 하나의 profile 안에서 여러 seed를 순회한다.
        for seed in experiment_cfg["seeds"]:
            print(f"\n{'-' * 60}\n[SEED: {seed}]\n{'-' * 60}")

            # run_profile_seed:
            # 학습 -> calibration -> seen/unseen 평가 -> benchmark까지 한 번에 수행한다.
            run_rows, latency_row = run_profile_seed(
                receiver_profile,
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

            # profile/seed 실행이 끝나면 raw 결과와 benchmark 결과를 누적한다.
            all_runs.extend(run_rows)
            latency_rows.append(latency_row)

    # build_experiment_summary:
    # raw 평가 결과를 평균/표준편차 요약표로 바꾼다.
    summary = build_experiment_summary(all_runs)

    # build_latency_summary:
    # benchmark raw 결과를 profile별 요약표로 바꾼다.
    latency_summary = build_latency_summary(latency_rows)

    # 내부 DataFrame은 기존 짧은 컬럼명을 쓰지만,
    # 저장용 CSV는 사람이 읽기 쉬운 컬럼명으로 바꿔 저장한다.
    save_experiment_summary_csv(summary)
    save_latency_summary_csv(latency_summary)

    # 마지막으로 summary / ablation 그래프를 생성한다.
    plot_summary(summary)
    plot_policy_ablation(summary)

    print(f"Saved CSV files in ./{CSV_DIR} and plot PNG files in ./{GRAPH_DIR}.")


if __name__ == "__main__":
    # 스크립트를 직접 실행했을 때만 main()을 호출한다.
    main()
