"""결과 그래프를 그리는 함수들을 모아 둔 모듈이다.

이 모듈은 학습이나 평가를 수행하지 않는다.
이미 계산된 요약 DataFrame을 받아 summary / ablation 그래프만 만든다.
"""

import os

import matplotlib.pyplot as plt
import numpy as np


# GRAPH_DIR:
# 그래프 PNG를 저장할 기본 폴더 이름이다.
GRAPH_DIR = "graph"

# SER_DISPLAY_FLOOR:
# 로그 축에서는 0을 그릴 수 없으므로,
# SER가 0이면 이 값으로 clip해 표시한다.
SER_DISPLAY_FLOOR = 1e-6

# LINE_ALPHA:
# 여러 선이 겹쳐도 완전히 묻히지 않도록 곡선별 투명도를 다르게 준다.
LINE_ALPHA = {
    "default_lora": 0.75,
    "full_cnn": 0.75,
    "hybrid_cnn": 0.95,
    "cnn_utilization": 0.85,
}


def _clip_ser(series):
    """SER을 로그축에 표시할 수 있도록 최소 표시값으로 clip한다."""

    return np.clip(series, SER_DISPLAY_FLOOR, None)


def _configure_ser_axis(ax1, ax2):
    """SER 축과 CNN utilization 축의 공통 표시 형식을 맞춘다."""

    # 왼쪽 y축은 SER용 로그축이다.
    ax1.set_yscale("log")
    ax1.set_ylim([SER_DISPLAY_FLOOR, 1.1])

    # x축은 -21 dB부터 0 dB까지 1 dB 간격으로 고정한다.
    xticks = np.arange(-21, 1, 1)
    ax1.set_xticks(xticks)
    ax1.set_xlim([-21.5, 0.5])

    # 읽기 쉽게 grid를 켠다.
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    # 오른쪽 y축은 CNN utilization[%]이다.
    ax2.set_ylim([-5, 105])


def plot_summary(summary_df, graph_dir: str = GRAPH_DIR, filename_suffix: str = ""):
    """프로파일별 최종 성능 비교 그래프를 저장한다."""

    # 저장 폴더가 없으면 먼저 만든다.
    os.makedirs(graph_dir, exist_ok=True)

    # profile별로 seen / unseen 채널 그래프를 각각 만든다.
    for profile_name in summary_df["profile"].unique():
        for channel_type, title_suffix in [("seen", "Seen Channel"), ("unseen", "Unseen Channel")]:
            # summary 그래프는 bin policy 결과를 기준으로 그린다.
            data = summary_df[
                (summary_df["profile"] == profile_name)
                & (summary_df["type"] == f"bin_{channel_type}")
            ]
            if data.empty:
                continue

            # ax1:
            # SER을 그릴 축이다.
            fig, ax1 = plt.subplots(figsize=(10, 6))
            # ax2:
            # 같은 x축 위에 CNN utilization을 추가로 그릴 축이다.
            ax2 = ax1.twinx()

            # Default LoRa SER 곡선
            ax1.semilogy(
                data["snr"],
                _clip_ser(data["ser_single_mean"]),
                label="Default LoRa",
                color="black",
                marker="x",
                alpha=LINE_ALPHA["default_lora"],
            )

            # Full CNN SER 곡선
            ax1.semilogy(
                data["snr"],
                _clip_ser(data["ser_c_mean"]),
                label="Full CNN",
                color="orange",
                marker="v",
                alpha=LINE_ALPHA["full_cnn"],
            )

            # Hybrid CNN SER 곡선
            ax1.semilogy(
                data["snr"],
                _clip_ser(data["ser_h_mean"]),
                label="Hybrid CNN",
                color="red",
                marker="o",
                alpha=LINE_ALPHA["hybrid_cnn"],
            )

            # CNN utilization 곡선
            ax2.plot(
                data["snr"],
                data["util_mean"],
                label="CNN Utilization",
                color="green",
                marker="*",
                alpha=LINE_ALPHA["cnn_utilization"],
            )

            # Hybrid CNN SER 표준편차 영역
            ax1.fill_between(
                data["snr"],
                _clip_ser(data["ser_h_mean"] - data["ser_h_std"]),
                _clip_ser(data["ser_h_mean"] + data["ser_h_std"]),
                color="red",
                alpha=0.15,
            )

            # CNN utilization 표준편차 영역
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

            # 왼쪽/오른쪽 축에 따로 달린 legend를 하나로 합친다.
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

            plt.title(f"{profile_name}: {title_suffix}")
            plt.savefig(
                os.path.join(
                    graph_dir,
                    f"{profile_name}_summary_{channel_type}{filename_suffix}.png",
                ),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


def plot_policy_ablation(summary_df, graph_dir: str = GRAPH_DIR, filename_suffix: str = ""):
    """Global threshold policy와 confidence-bin policy를 비교하는 그래프를 저장한다."""

    os.makedirs(graph_dir, exist_ok=True)

    for profile_name in summary_df["profile"].unique():
        for channel_type, title_suffix in [("seen", "Seen Channel"), ("unseen", "Unseen Channel")]:
            # fixed_data:
            # global threshold policy 결과다.
            fixed_data = summary_df[
                (summary_df["profile"] == profile_name)
                & (summary_df["type"] == f"global_{channel_type}")
            ]

            # adaptive_data:
            # confidence-bin policy 결과다.
            adaptive_data = summary_df[
                (summary_df["profile"] == profile_name)
                & (summary_df["type"] == f"bin_{channel_type}")
            ]

            if fixed_data.empty or adaptive_data.empty:
                continue

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx()

            # global threshold hybrid SER
            ax1.semilogy(
                fixed_data["snr"],
                _clip_ser(fixed_data["ser_h_mean"]),
                label="Global Threshold Hybrid",
                color="blue",
                marker="s",
                alpha=0.80,
            )

            # global threshold utilization
            ax2.plot(
                fixed_data["snr"],
                fixed_data["util_mean"],
                label="Global Threshold Util",
                color="blue",
                linestyle="--",
                marker="s",
                alpha=0.80,
            )

            # confidence-bin hybrid SER
            ax1.semilogy(
                adaptive_data["snr"],
                _clip_ser(adaptive_data["ser_h_mean"]),
                label="Confidence-bin Hybrid",
                color="red",
                marker="o",
                alpha=0.95,
            )

            # confidence-bin utilization
            ax2.plot(
                adaptive_data["snr"],
                adaptive_data["util_mean"],
                label="Confidence-bin Util",
                color="red",
                linestyle="--",
                marker="o",
                alpha=0.95,
            )

            # confidence-bin policy 표준편차 표시
            ax1.fill_between(
                adaptive_data["snr"],
                _clip_ser(adaptive_data["ser_h_mean"] - adaptive_data["ser_h_std"]),
                _clip_ser(adaptive_data["ser_h_mean"] + adaptive_data["ser_h_std"]),
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
                os.path.join(
                    graph_dir,
                    f"{profile_name}_ablation_{channel_type}{filename_suffix}.png",
                ),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
