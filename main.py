"""실험 전체를 orchestration하는 메인 실행 파일이다.

이 파일은 다음 순서로 동작한다.

1. 프로파일별 설정 병합
2. 학습/검증/평가 데이터셋 생성
3. CNN 학습
4. calibration으로 하이브리드 정책 보정
5. seen / unseen 채널 평가
6. CSV와 그래프 저장

즉, 사용자가 `python main.py`를 실행했을 때 실제로 돌아가는 실험 흐름은 대부분 이 파일에 있다.
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.data import DataLoader

from config import CFG
from dataset import (
    OnlineParametersDataset,
    create_fixed_waveform_range_dataset,
    create_fixed_waveform_dataset,
)
from evaluation import (
    benchmark_receivers,
    calibrate_confidence_bin_policy_from_outputs,
    calibrate_global_threshold_from_outputs,
    collect_receiver_outputs,
    summarize_outputs,
)
from models import Hypothesis2DCNN
from simulator import GPUOnlineSimulator
from training import train_online_model
from utils import count_trainable_parameters, flatten_summary_columns, merge_config, set_seed


GRAPH_DIR = "graph"
CSV_DIR = "csv"
SER_DISPLAY_FLOOR = 1e-6
LINE_ALPHA = {
    "default_lora": 0.75,
    "enhanced_lora": 0.75,
    "full_cnn": 0.75,
    "hybrid_cnn": 0.95,
    "cnn_utilization": 0.85,
}
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


def _configure_ser_axis(ax1, ax2):
    """SER 축과 CNN utilization 축의 공통 표시 형식을 맞춘다."""

    ax1.set_yscale("log")
    ax1.set_ylim([SER_DISPLAY_FLOOR, 1.1])

    xticks = np.arange(-21, 1, 1)
    ax1.set_xticks(xticks)
    ax1.set_xlim([-21.5, 0.5])

    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.set_ylim([-5, 105])


def plot_summary(summary_df, graph_dir: str = GRAPH_DIR):
    """프로파일별 최종 성능 비교 그래프를 저장한다.

    이 그래프는 다음 네 경로를 한 그림에 겹쳐 보여 준다.

    - Default LoRa
    - Enhanced LoRa
    - Full CNN
    - Hybrid CNN
    """

    def clip_ser(series):
        return np.clip(series, SER_DISPLAY_FLOOR, None)

    os.makedirs(graph_dir, exist_ok=True)

    for profile_name in summary_df["profile"].unique():
        for channel_type, title_suffix in [("seen", "Seen Channel"), ("unseen", "Unseen Channel")]:
            data = summary_df[
                (summary_df["profile"] == profile_name)
                & (summary_df["type"] == f"bin_{channel_type}")
            ]
            if data.empty:
                continue

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx()

            # SER 곡선은 왼쪽 y축, CNN utilization은 오른쪽 y축에 그린다.
            ax1.semilogy(
                data["snr"],
                clip_ser(data["ser_single_mean"]),
                label="Default LoRa",
                color="black",
                marker="x",
                alpha=LINE_ALPHA["default_lora"],
            )
            ax1.semilogy(
                data["snr"],
                clip_ser(data["ser_mh_mean"]),
                label="Enhanced LoRa",
                color="blue",
                marker="s",
                alpha=LINE_ALPHA["enhanced_lora"],
            )
            ax1.semilogy(
                data["snr"],
                clip_ser(data["ser_c_mean"]),
                label="Full CNN",
                color="orange",
                marker="v",
                alpha=LINE_ALPHA["full_cnn"],
            )
            ax1.semilogy(
                data["snr"],
                clip_ser(data["ser_h_mean"]),
                label="Hybrid CNN",
                color="red",
                marker="o",
                alpha=LINE_ALPHA["hybrid_cnn"],
            )
            ax2.plot(
                data["snr"],
                data["util_mean"],
                label="CNN Utilization",
                color="green",
                marker="*",
                alpha=LINE_ALPHA["cnn_utilization"],
            )

            ax1.fill_between(
                data["snr"],
                clip_ser(data["ser_h_mean"] - data["ser_h_std"]),
                clip_ser(data["ser_h_mean"] + data["ser_h_std"]),
                color="red",
                alpha=0.15,
            )
            ax2.fill_between(
                data["snr"],
                np.clip(data["util_mean"] - data["util_std"], 0, 100),
                np.clip(data["util_mean"] + data["util_std"], 0, 100),
                color="green",
                alpha=0.15,
            )

            ax1.set_xlabel("SNR [dB]")
            ax1.set_ylabel("SER")
            ax2.set_ylabel("CNN Utilization [%]")
            _configure_ser_axis(ax1, ax2)

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
            plt.title(f"{profile_name}: {title_suffix}")
            plt.savefig(
                os.path.join(graph_dir, f"{profile_name}_summary_{channel_type}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


def plot_policy_ablation(summary_df, graph_dir: str = GRAPH_DIR):
    """Global threshold policy와 confidence-bin policy를 비교하는 그래프를 저장한다."""

    def clip_ser(series):
        return np.clip(series, SER_DISPLAY_FLOOR, None)

    os.makedirs(graph_dir, exist_ok=True)

    for profile_name in summary_df["profile"].unique():
        for channel_type, title_suffix in [("seen", "Seen Channel"), ("unseen", "Unseen Channel")]:
            fixed_data = summary_df[
                (summary_df["profile"] == profile_name)
                & (summary_df["type"] == f"global_{channel_type}")
            ]
            adaptive_data = summary_df[
                (summary_df["profile"] == profile_name)
                & (summary_df["type"] == f"bin_{channel_type}")
            ]
            if fixed_data.empty or adaptive_data.empty:
                continue

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx()

            ax1.semilogy(
                fixed_data["snr"],
                clip_ser(fixed_data["ser_h_mean"]),
                label="Global Threshold Hybrid",
                color="blue",
                marker="s",
                alpha=0.80,
            )
            ax2.plot(
                fixed_data["snr"],
                fixed_data["util_mean"],
                label="Global Threshold Util",
                color="blue",
                linestyle="--",
                marker="s",
                alpha=0.80,
            )

            ax1.semilogy(
                adaptive_data["snr"],
                clip_ser(adaptive_data["ser_h_mean"]),
                label="Confidence-bin Hybrid",
                color="red",
                marker="o",
                alpha=0.95,
            )
            ax2.plot(
                adaptive_data["snr"],
                adaptive_data["util_mean"],
                label="Confidence-bin Util",
                color="red",
                linestyle="--",
                marker="o",
                alpha=0.95,
            )

            ax1.fill_between(
                adaptive_data["snr"],
                clip_ser(adaptive_data["ser_h_mean"] - adaptive_data["ser_h_std"]),
                clip_ser(adaptive_data["ser_h_mean"] + adaptive_data["ser_h_std"]),
                color="red",
                alpha=0.10,
            )
            ax2.fill_between(
                adaptive_data["snr"],
                np.clip(adaptive_data["util_mean"] - adaptive_data["util_std"], 0, 100),
                np.clip(adaptive_data["util_mean"] + adaptive_data["util_std"], 0, 100),
                color="red",
                alpha=0.10,
            )

            ax1.set_xlabel("SNR [dB]")
            ax1.set_ylabel("SER")
            ax2.set_ylabel("CNN Utilization [%]")
            _configure_ser_axis(ax1, ax2)

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
            plt.title(f"{profile_name}: Global vs Confidence-bin ({title_suffix})")
            plt.savefig(
                os.path.join(graph_dir, f"{profile_name}_ablation_{channel_type}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


def main():
    """설정된 모든 수신기 프로파일에 대해 학습과 평가를 수행한다."""

    base_experiment_cfg = CFG["experiment"]
    base_train_cfg = CFG["training"]
    base_feature_cfg = CFG["feature_bank"]
    base_model_cfg = CFG.get("model", {})
    base_benchmark_cfg = CFG["benchmark"]
    artifact_cfg = CFG.get("artifacts", {})
    hybrid_cfg = CFG["hybrid"]
    channel_profiles = CFG["channel_profiles"]
    pin_memory = torch.cuda.is_available()

    all_runs = []
    latency_rows = []
    os.makedirs(CSV_DIR, exist_ok=True)

    for receiver_profile in CFG["receiver_profiles"]:
        # 프로파일별 override를 기본 설정에 덮어써 실제 실행 설정을 만든다.
        experiment_cfg = merge_config(base_experiment_cfg, receiver_profile.get("experiment_overrides"))
        train_cfg = merge_config(base_train_cfg, receiver_profile.get("training_overrides"))
        feature_cfg = merge_config(base_feature_cfg, receiver_profile.get("feature_bank_overrides"))
        model_cfg = merge_config(base_model_cfg, receiver_profile.get("model_overrides"))
        benchmark_cfg = merge_config(base_benchmark_cfg, receiver_profile.get("benchmark_overrides"))

        print(f"\n{'=' * 60}\n[PROFILE: {receiver_profile['name']}]\n{'=' * 60}")
        for seed in experiment_cfg["seeds"]:
            print(f"\n{'-' * 60}\n[SEED: {seed}]\n{'-' * 60}")
            set_seed(seed)

            # 현재 프로파일에 맞는 LoRa 시뮬레이터와 CNN 모델을 만든다.
            simulator = GPUOnlineSimulator(
                sf=receiver_profile["sf"],
                bw=receiver_profile["bw"],
                fs=receiver_profile["fs"],
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            model = Hypothesis2DCNN(
                num_classes=simulator.M,
                num_hypotheses=feature_cfg["cfo_steps"] * feature_cfg["to_steps"],
                num_bins=simulator.M * feature_cfg["patch_size"],
                in_channels=2,
                stage_channels=model_cfg["stage_channels"],
                classifier_hidden=model_cfg["classifier_hidden"],
                dropout=model_cfg["dropout"],
                width_scale=model_cfg["width_scale"],
            ).to(simulator.device)

            # validation은 하나의 SNR 범위를 대표하는 고정 waveform 데이터셋을 사용한다.
            ds_val = create_fixed_waveform_range_dataset(
                simulator,
                num_packets=experiment_cfg["val_packets"],
                snr_range=experiment_cfg["train_snr_range"],
                channel_profile=channel_profiles["train"],
                seed=seed,
                experiment_cfg=experiment_cfg,
            )
            # calibration / test는 SNR별 고정 waveform dataset을 만든다.
            ds_calib = create_fixed_waveform_dataset(
                simulator,
                num_packets_per_snr=experiment_cfg["calib_packets"],
                snr_list=experiment_cfg["test_snrs"],
                channel_profile=channel_profiles["seen_eval"],
                seed=seed + 1,
                experiment_cfg=experiment_cfg,
            )
            ds_test_seen = create_fixed_waveform_dataset(
                simulator,
                num_packets_per_snr=experiment_cfg["test_packets"],
                snr_list=experiment_cfg["test_snrs"],
                channel_profile=channel_profiles["seen_eval"],
                seed=seed + 2,
                experiment_cfg=experiment_cfg,
            )
            ds_test_unseen = create_fixed_waveform_dataset(
                simulator,
                num_packets_per_snr=experiment_cfg["test_packets"],
                snr_list=experiment_cfg["test_snrs"],
                channel_profile=channel_profiles["unseen_eval"],
                seed=seed + 3,
                experiment_cfg=experiment_cfg,
            )

            # train loader는 label/SNR/CFO만 제공하고,
            # 실제 waveform은 training loop 안에서 온라인 생성한다.
            dl_train = DataLoader(
                OnlineParametersDataset(
                    simulator.M,
                    experiment_cfg["train_samples"],
                    experiment_cfg["train_snr_range"],
                    channel_profiles["train"]["max_cfo_bins"],
                    simulator.bw,
                ),
                batch_size=train_cfg["train_batch_size"],
                shuffle=True,
                pin_memory=pin_memory,
            )
            dl_val = DataLoader(
                ds_val,
                batch_size=train_cfg["eval_batch_size"],
                shuffle=False,
                pin_memory=pin_memory,
            )

            # 학습이 끝나면 가장 좋은 validation loss 기준 체크포인트가 복원된 상태의 모델을 받는다.
            model = train_online_model(
                model,
                simulator,
                dl_train,
                dl_val,
                channel_profile=channel_profiles["train"],
                train_cfg=train_cfg,
                feature_cfg=feature_cfg,
                artifact_cfg=artifact_cfg,
                run_name=f"{receiver_profile['name']}_seed{seed}",
                metadata={
                    "profile": receiver_profile["name"],
                    "seed": seed,
                    "sf": receiver_profile["sf"],
                    "bw": receiver_profile["bw"],
                    "fs": receiver_profile["fs"],
                    "train_cfg": train_cfg,
                    "experiment_cfg": experiment_cfg,
                    "feature_cfg": feature_cfg,
                    "model_cfg": model_cfg,
                    "hybrid_cfg": hybrid_cfg,
                    "train_channel_profile": simulator.resolve_channel_profile(channel_profiles["train"]),
                },
            )

            # held-out calibration 데이터로 두 가지 하이브리드 정책을 보정한다.
            calib_outputs = collect_receiver_outputs(
                model,
                simulator,
                ds_calib,
                channel_profiles["seen_eval"],
                feature_cfg=feature_cfg,
                eval_batch_size=train_cfg["eval_batch_size"],
                hybrid_cfg=hybrid_cfg,
            )
            global_policy = calibrate_global_threshold_from_outputs(
                calib_outputs,
                hybrid_cfg=hybrid_cfg,
                payload_symbols=experiment_cfg["payload_symbols"],
            )
            bin_policy = calibrate_confidence_bin_policy_from_outputs(
                calib_outputs,
                hybrid_cfg=hybrid_cfg,
                payload_symbols=experiment_cfg["payload_symbols"],
            )

            # 보정된 정책을 seen / unseen 평가 데이터에 적용한다.
            test_seen_outputs = collect_receiver_outputs(
                model,
                simulator,
                ds_test_seen,
                channel_profiles["seen_eval"],
                feature_cfg=feature_cfg,
                eval_batch_size=train_cfg["eval_batch_size"],
                hybrid_cfg=hybrid_cfg,
            )
            test_unseen_outputs = collect_receiver_outputs(
                model,
                simulator,
                ds_test_unseen,
                channel_profiles["unseen_eval"],
                feature_cfg=feature_cfg,
                eval_batch_size=train_cfg["eval_batch_size"],
                hybrid_cfg=hybrid_cfg,
            )

            results = {
                "global_seen": summarize_outputs(
                    test_seen_outputs,
                    global_policy,
                    hybrid_cfg=hybrid_cfg,
                    payload_symbols=experiment_cfg["payload_symbols"],
                ),
                "bin_seen": summarize_outputs(
                    test_seen_outputs,
                    bin_policy,
                    hybrid_cfg=hybrid_cfg,
                    payload_symbols=experiment_cfg["payload_symbols"],
                ),
                "global_unseen": summarize_outputs(
                    test_unseen_outputs,
                    global_policy,
                    hybrid_cfg=hybrid_cfg,
                    payload_symbols=experiment_cfg["payload_symbols"],
                ),
                "bin_unseen": summarize_outputs(
                    test_unseen_outputs,
                    bin_policy,
                    hybrid_cfg=hybrid_cfg,
                    payload_symbols=experiment_cfg["payload_symbols"],
                ),
            }

            # 중간 SNR 지점 하나를 골라 각 수신기 경로의 상대적 지연시간을 측정한다.
            benchmark_snr = experiment_cfg["test_snrs"][len(experiment_cfg["test_snrs"]) // 2]
            latency_rows.append(
                {
                    "profile": receiver_profile["name"],
                    "seed": seed,
                    "params": count_trainable_parameters(model),
                    "num_hypotheses": feature_cfg["cfo_steps"] * feature_cfg["to_steps"],
                    "feature_elements": feature_cfg["cfo_steps"] * feature_cfg["to_steps"] * simulator.M * feature_cfg["patch_size"],
                    **benchmark_receivers(
                        model,
                        simulator,
                        ds_test_seen[benchmark_snr],
                        channel_profiles["seen_eval"],
                        bin_policy,
                        benchmark_cfg=benchmark_cfg,
                        feature_cfg=feature_cfg,
                        hybrid_cfg=hybrid_cfg,
                    ),
                }
            )

            # SNR별 통계를 하나의 표 형태로 차곡차곡 누적한다.
            for eval_type, eval_stats in results.items():
                for snr in experiment_cfg["test_snrs"]:
                    all_runs.append(
                        {
                            "profile": receiver_profile["name"],
                            "seed": seed,
                            "snr": snr,
                            "type": eval_type,
                            **eval_stats[snr],
                        }
                    )

    # seed별 raw 결과를 평균/표준편차 요약표로 집계한다.
    results_df = pd.DataFrame(all_runs)
    metric_cols = [
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
    summary = results_df.groupby(["profile", "type", "snr"])[metric_cols].agg(["mean", "std"]).reset_index()
    summary = flatten_summary_columns(summary)
    n_runs = results_df.groupby(["profile", "type", "snr"])["seed"].count().reset_index().rename(columns={"seed": "n_runs"})
    summary = pd.merge(summary, n_runs, on=["profile", "type", "snr"])
    summary_csv = summary.rename(columns=CSV_RENAME_MAP)
    summary_csv.to_csv(os.path.join(CSV_DIR, "experiment_summary.csv"), index=False)

    # latency 요약표도 별도로 저장한다.
    latency_df = pd.DataFrame(latency_rows)
    latency_metric_cols = [
        "params",
        "num_hypotheses",
        "feature_elements",
        "single_ms",
        "mh_ms",
        "cnn_ms",
        "hybrid_ms",
    ]
    latency_summary = latency_df.groupby(["profile"])[latency_metric_cols].agg(["mean", "std"]).reset_index()
    latency_summary = flatten_summary_columns(latency_summary)
    latency_summary_csv = latency_summary.rename(columns=CSV_RENAME_MAP)
    latency_summary_csv.to_csv(os.path.join(CSV_DIR, "latency_summary.csv"), index=False)

    # 마지막으로 그래프를 생성한다.
    plot_summary(summary)
    plot_policy_ablation(summary)

    print("Saved CSV files in ./csv and plot PNG files in ./graph.")


if __name__ == "__main__":
    main()
