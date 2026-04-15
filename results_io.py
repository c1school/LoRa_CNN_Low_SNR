"""요약 DataFrame 생성과 CSV 저장을 담당하는 모듈이다."""

import os

import pandas as pd

from utils import flatten_summary_columns


# CSV_DIR:
# 요약 CSV 파일을 저장할 기본 폴더다.
CSV_DIR = "csv"

# CSV_RENAME_MAP:
# 내부 계산용 짧은 컬럼명을 사람이 읽기 쉬운 CSV 컬럼명으로 바꿀 때 사용하는 맵이다.
CSV_RENAME_MAP = {
    "ser_single_mean": "ser_default_lora_mean",
    "ser_single_std": "ser_default_lora_std",
    "ser_mh_mean": "ser_enhanced_lora_mean",
    "ser_mh_std": "ser_enhanced_lora_std",
    "ser_c_mean": "ser_full_cnn_mean",
    "ser_c_std": "ser_full_cnn_std",
    "ser_h_mean": "ser_hybrid_cnn_mean",
    "ser_h_std": "ser_hybrid_cnn_std",
    "per_single_mean": "per_default_lora_mean",
    "per_single_std": "per_default_lora_std",
    "per_mh_mean": "per_enhanced_lora_mean",
    "per_mh_std": "per_enhanced_lora_std",
    "per_c_mean": "per_full_cnn_mean",
    "per_c_std": "per_full_cnn_std",
    "per_h_mean": "per_hybrid_cnn_mean",
    "per_h_std": "per_hybrid_cnn_std",
    "util_mean": "cnn_utilization_mean",
    "util_std": "cnn_utilization_std",
    "single_ms_mean": "default_lora_ms_mean",
    "single_ms_std": "default_lora_ms_std",
    "mh_ms_mean": "enhanced_lora_ms_mean",
    "mh_ms_std": "enhanced_lora_ms_std",
    "cnn_ms_mean": "full_cnn_ms_mean",
    "cnn_ms_std": "full_cnn_ms_std",
    "hybrid_ms_mean": "hybrid_cnn_ms_mean",
    "hybrid_ms_std": "hybrid_cnn_ms_std",
}

# RESULT_METRIC_COLS:
# 실험 요약표에서 평균/표준편차를 낼 대상 metric 목록이다.
RESULT_METRIC_COLS = [
    "ser_single",
    "ser_mh",
    "ser_c",
    "ser_h",
    "per_single",
    "per_mh",
    "per_c",
    "per_h",
    "util",
]

# LATENCY_METRIC_COLS:
# benchmark 요약표에서 평균/표준편차를 낼 대상 metric 목록이다.
LATENCY_METRIC_COLS = [
    "params",
    "num_hypotheses",
    "feature_elements",
    "single_ms",
    "mh_ms",
    "cnn_ms",
    "hybrid_ms",
]


def build_experiment_summary(all_runs):
    """seed별 raw 결과 목록을 평균/표준편차 요약표로 바꾼다."""

    # results_df:
    # run_profile_seed가 만든 row 목록을 DataFrame으로 바꾼 것이다.
    results_df = pd.DataFrame(all_runs)

    # groupby 기준:
    # profile / policy type / snr가 같으면 같은 실험 조건으로 본다.
    summary = results_df.groupby(["profile", "type", "snr"])[RESULT_METRIC_COLS].agg(["mean", "std"]).reset_index()

    # multi-index 컬럼을 단일 문자열 컬럼으로 납작하게 편다.
    summary = flatten_summary_columns(summary)

    # n_runs:
    # 각 조건에서 실제로 몇 개 seed가 들어갔는지 기록한다.
    n_runs = results_df.groupby(["profile", "type", "snr"])["seed"].count().reset_index().rename(columns={"seed": "n_runs"})

    return pd.merge(summary, n_runs, on=["profile", "type", "snr"])


def build_latency_summary(latency_rows):
    """지연시간 raw 결과 목록을 프로파일별 평균/표준편차 요약표로 바꾼다."""

    latency_df = pd.DataFrame(latency_rows)
    latency_summary = latency_df.groupby(["profile"])[LATENCY_METRIC_COLS].agg(["mean", "std"]).reset_index()
    return flatten_summary_columns(latency_summary)


def save_experiment_summary_csv(summary, csv_dir: str = CSV_DIR):
    """실험 요약표를 CSV로 저장한다."""

    os.makedirs(csv_dir, exist_ok=True)
    summary_csv = summary.rename(columns=CSV_RENAME_MAP)
    output_path = os.path.join(csv_dir, "experiment_summary.csv")
    summary_csv.to_csv(output_path, index=False)
    return output_path


def save_latency_summary_csv(latency_summary, csv_dir: str = CSV_DIR):
    """지연시간 요약표를 CSV로 저장한다."""

    os.makedirs(csv_dir, exist_ok=True)
    latency_summary_csv = latency_summary.rename(columns=CSV_RENAME_MAP)
    output_path = os.path.join(csv_dir, "latency_summary.csv")
    latency_summary_csv.to_csv(output_path, index=False)
    return output_path
