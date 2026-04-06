import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import set_seed
from config import CFG
from simulator import GPUOnlineSimulator
from models import Hypothesis2DCNN
from training import train_online_model
from dataset import create_fixed_feature_dataset, create_fixed_waveform_dataset, OnlineParametersDataset
from evaluation import calibrate_adaptive_policy_joint, run_evaluation
from torch.utils.data import DataLoader



def plot_summary(summary_df):
    """
    메인 비교용 그래프를 그리는 함수이다.

    이 함수는 각 채널 조건(seen / unseen)에 대해 다음 네 가지를 함께 보여준다.
    1) conventional LoRa baseline SER
    2) full CNN SER
    3) hybrid SER
    4) CNN utilization

    또한 hybrid SER과 utilization에는 표준편차 영역(error band)을 추가하여
    반복 실험 간 변동성도 함께 확인할 수 있도록 하였다.
    """

    for channel_type, title, filename in zip(
        ['seen', 'unseen'],
        ['Seen Channel', 'Unseen Harsher Channel'],
        ['experiment_main_seen.png', 'experiment_main_unseen.png']
    ):
        adapt_type = f'adapt_{channel_type}'

        # 해당 타입의 데이터가 없으면 그래프를 건너뛴다.
        if adapt_type not in summary_df['type'].unique():
            continue

        data = summary_df[summary_df['type'] == adapt_type]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        # ------------------------------------------------------------
        # 평균 성능 곡선 그리기
        # ------------------------------------------------------------
        ax1.semilogy(
            data['snr'],
            data['ser_g_mean'],
            label='Conventional LoRa',
            color='black',
            marker='x',
            linestyle=':'
        )
        ax1.semilogy(
            data['snr'],
            data['ser_c_mean'],
            label='2D CNN (Full)',
            color='orange',
            marker='v',
            linestyle='-.'
        )
        ax1.semilogy(
            data['snr'],
            data['ser_h_mean'],
            label='Proposed Hybrid',
            color='red',
            marker='o',
            linestyle='-'
        )
        ax2.plot(
            data['snr'],
            data['util_mean'],
            label='CNN Utilization',
            color='green',
            marker='*',
            linestyle='--'
        )

        # ------------------------------------------------------------
        # 표준편차 영역 표시하기
        # ------------------------------------------------------------
        ax1.fill_between(
            data['snr'],
            np.clip(data['ser_h_mean'] - data['ser_h_std'], 1e-6, None),
            data['ser_h_mean'] + data['ser_h_std'],
            color='red',
            alpha=0.15
        )
        ax2.fill_between(
            data['snr'],
            np.clip(data['util_mean'] - data['util_std'], 0, 100),
            np.clip(data['util_mean'] + data['util_std'], 0, 100),
            color='green',
            alpha=0.15
        )

        # ------------------------------------------------------------
        # 축과 제목 설정
        # ------------------------------------------------------------
        ax1.set_xlabel('SNR [dB]', fontsize=12)
        ax1.set_ylabel('Symbol Error Rate (SER)', fontsize=12)
        ax2.set_ylabel('CNN Utilization (%)', fontsize=12, color='green')

        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        ax1.set_ylim([1e-4, 1.1])
        ax2.set_ylim([-5, 105])

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        plt.title(f'Overall Performance ({title})', fontsize=14)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()



def plot_ablation(summary_df):
    """
    fixed hybrid와 adaptive hybrid를 직접 비교하는 ablation 그래프를 그리는 함수이다.

    이 그래프는 다음 질문에 답하기 위해 사용한다.
    "adaptive policy가 fixed policy보다 얼마나 CNN 사용률을 줄였는가?"
    "그 대가로 SER은 얼마나 변하였는가?"
    """

    for channel_type, title, filename in zip(
        ['seen', 'unseen'],
        ['Seen Channel', 'Unseen Harsher Channel'],
        ['experiment_ablation_seen.png', 'experiment_ablation_unseen.png']
    ):
        fixed_type = f'fixed_{channel_type}'
        adapt_type = f'adapt_{channel_type}'

        if adapt_type not in summary_df['type'].unique() or fixed_type not in summary_df['type'].unique():
            continue

        fixed_data = summary_df[summary_df['type'] == fixed_type]
        adapt_data = summary_df[summary_df['type'] == adapt_type]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        # ------------------------------------------------------------
        # fixed policy 결과
        # ------------------------------------------------------------
        ax1.semilogy(
            fixed_data['snr'],
            fixed_data['ser_h_mean'],
            label='Fixed Hybrid SER',
            color='blue',
            marker='s',
            linestyle='-'
        )
        ax2.plot(
            fixed_data['snr'],
            fixed_data['util_mean'],
            label='Fixed CNN Util',
            color='blue',
            marker='s',
            linestyle='--'
        )

        # ------------------------------------------------------------
        # adaptive policy 결과
        # ------------------------------------------------------------
        ax1.semilogy(
            adapt_data['snr'],
            adapt_data['ser_h_mean'],
            label='Adaptive Hybrid SER',
            color='red',
            marker='o',
            linestyle='-'
        )
        ax2.plot(
            adapt_data['snr'],
            adapt_data['util_mean'],
            label='Adaptive CNN Util',
            color='red',
            marker='o',
            linestyle='--'
        )

        # adaptive 정책의 표준편차 영역을 함께 표시한다.
        ax1.fill_between(
            adapt_data['snr'],
            np.clip(adapt_data['ser_h_mean'] - adapt_data['ser_h_std'], 1e-6, None),
            adapt_data['ser_h_mean'] + adapt_data['ser_h_std'],
            color='red',
            alpha=0.1
        )
        ax2.fill_between(
            adapt_data['snr'],
            np.clip(adapt_data['util_mean'] - adapt_data['util_std'], 0, 100),
            np.clip(adapt_data['util_mean'] + adapt_data['util_std'], 0, 100),
            color='red',
            alpha=0.1
        )

        ax1.set_xlabel('SNR [dB]', fontsize=12)
        ax1.set_ylabel('Symbol Error Rate (SER)', fontsize=12)
        ax2.set_ylabel('CNN Utilization (%)', fontsize=12)

        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        ax1.set_ylim([1e-4, 1.1])
        ax2.set_ylim([-5, 105])

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        plt.title(f'Ablation: Fixed vs Adaptive Policy ({title})', fontsize=14)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()



def main():
    """
    전체 실험 파이프라인을 실행하는 메인 함수이다.

    전체 흐름은 다음과 같다.
    1) seed를 바꾸어 반복 실험한다.
    2) simulator와 model을 만든다.
    3) validation / calibration / test 데이터셋을 만든다.
    4) 모델을 학습한다.
    5) calibration 데이터로 adaptive threshold 정책을 찾는다.
    6) fixed / adaptive, seen / unseen 조건에서 평가한다.
    7) 모든 결과를 평균과 표준편차로 정리하여 저장한다.
    8) 요약 그래프와 ablation 그래프를 그린다.
    """

    all_runs = []

    for seed in CFG["seeds"]:
        print(f"\n{'=' * 45}\n[RUN: Seed {seed}]\n{'=' * 45}")

        # 실행마다 seed를 고정하여 재현성을 확보한다.
        set_seed(seed)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 시뮬레이터를 현재 설정값으로 생성한다.
        sim = GPUOnlineSimulator(
            sf=CFG["sf"],
            bw=CFG["bw"],
            fs=CFG["fs"],
            device=device,
        )

        # 모델을 생성한다.
        model = Hypothesis2DCNN(num_classes=sim.M, in_channels=2).to(device)

        # CFO 최대 범위를 Hz로 계산한다.
        max_cfo_hz = CFG["max_cfo_bins"] * (sim.bw / sim.M)

        # ------------------------------------------------------------
        # 데이터셋 생성
        # ------------------------------------------------------------
        ds_val = create_fixed_feature_dataset(
            sim,
            CFG["calib_samples"],
            (-20, 0),
            CFG["max_cfo_bins"],
            seed=seed,
        )
        ds_calib = create_fixed_waveform_dataset(
            sim,
            CFG["calib_samples"],
            CFG["test_snrs"],
            max_cfo_hz,
            seed=seed + 1,
        )
        ds_test_seen = create_fixed_waveform_dataset(
            sim,
            CFG["test_samples"],
            CFG["test_snrs"],
            max_cfo_hz,
            seed=seed + 2,
        )
        ds_test_unseen = create_fixed_waveform_dataset(
            sim,
            CFG["test_samples"],
            CFG["test_snrs"],
            max_cfo_hz * 1.5,
            seed=seed + 3,
        )

        # ------------------------------------------------------------
        # DataLoader 구성
        # ------------------------------------------------------------
        dl_train = DataLoader(
            OnlineParametersDataset(
                sim.M,
                CFG["train_samples"],
                (-20, 0),
                CFG["max_cfo_bins"],
                sim.bw,
            ),
            batch_size=CFG["train_batch_size"],
        )
        dl_val = DataLoader(ds_val, batch_size=CFG["eval_batch_size"])

        # ------------------------------------------------------------
        # 모델 학습
        # ------------------------------------------------------------
        model = train_online_model(
            model,
            sim,
            dl_train,
            dl_val,
            max_cfo_hz,
            num_epochs=CFG["num_epochs"],
            lr=CFG["learning_rate"],
        )

        # ------------------------------------------------------------
        # adaptive policy 보정
        # ------------------------------------------------------------
        policy = calibrate_adaptive_policy_joint(model, sim, ds_calib, max_cfo_hz)

        # ------------------------------------------------------------
        # 평가 수행
        # ------------------------------------------------------------
        res_fixed_seen = run_evaluation(model, sim, ds_test_seen, max_cfo_hz, 1.5)
        res_adapt_seen = run_evaluation(model, sim, ds_test_seen, max_cfo_hz, policy)
        res_fixed_unseen = run_evaluation(model, sim, ds_test_unseen, max_cfo_hz * 1.5, 1.5)
        res_adapt_unseen = run_evaluation(model, sim, ds_test_unseen, max_cfo_hz * 1.5, policy)

        # ------------------------------------------------------------
        # 결과 누적 저장
        # ------------------------------------------------------------
        for snr in CFG["test_snrs"]:
            all_runs.append({'seed': seed, 'snr': snr, 'type': 'fixed_seen', **res_fixed_seen[snr]})
            all_runs.append({'seed': seed, 'snr': snr, 'type': 'adapt_seen', **res_adapt_seen[snr]})
            all_runs.append({'seed': seed, 'snr': snr, 'type': 'fixed_unseen', **res_fixed_unseen[snr]})
            all_runs.append({'seed': seed, 'snr': snr, 'type': 'adapt_unseen', **res_adapt_unseen[snr]})

    # ------------------------------------------------------------
    # 결과 집계
    # ------------------------------------------------------------
    df = pd.DataFrame(all_runs)
    metric_cols = ['ser_g', 'ser_c', 'ser_h', 'per_g', 'per_c', 'per_h', 'util', 'th']

    # type과 snr별로 평균과 표준편차를 계산한다.
    summary = df.groupby(['type', 'snr'])[metric_cols].agg(['mean', 'std']).reset_index()

    # 다중 인덱스 컬럼을 평탄화한다.
    summary.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in summary.columns.values]

    # 반복 횟수도 함께 저장한다.
    n_runs_df = df.groupby(['type', 'snr'])['seed'].count().reset_index()
    n_runs_df.rename(columns={'seed': 'n_runs'}, inplace=True)
    summary = pd.merge(summary, n_runs_df, on=['type', 'snr'])

    # CSV 파일로 저장한다.
    summary.to_csv("experiment_summary.csv", index=False)

    # 그래프를 저장한다.
    plot_summary(summary)
    plot_ablation(summary)

    print(">> 구조 리팩토링과 시각화 정리 완료.")


if __name__ == "__main__":
    main()
