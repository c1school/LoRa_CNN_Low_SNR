import os, torch
import pandas as pd
import matplotlib.pyplot as plt
from utils import set_seed
from simulator import GPUOnlineSimulator
from models import Hypothesis2DCNN
from training import train_online_model
from dataset import create_fixed_feature_dataset, create_fixed_waveform_dataset, OnlineParametersDataset
from evaluation import calibrate_adaptive_policy_joint, run_evaluation
from torch.utils.data import DataLoader

def plot_summary(summary_df):
    """논문 메인 비교용 그래프: Conventional vs Full-CNN vs Proposed Hybrid"""
    snrs = sorted(summary_df['snr'].unique())
    
    for channel_type, title, filename in zip(
        ['seen', 'unseen'], 
        ['Seen Channel', 'Unseen Harsher Channel'], 
        ['experiment_v7_seen_plot.png', 'experiment_v7_unseen_plot.png']
    ):
        adapt_type = f'adapt_{channel_type}'
        if adapt_type not in summary_df['type'].unique():
            continue
            
        data = summary_df[summary_df['type'] == adapt_type]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        
        ax1.semilogy(data['snr'], data['ser_g_mean'], label='Conventional LoRa', color='black', marker='x', linestyle=':')
        ax1.semilogy(data['snr'], data['ser_c_mean'], label='2D CNN (Full)', color='orange', marker='v', linestyle='-.')
        ax1.semilogy(data['snr'], data['ser_h_mean'], label='Proposed Hybrid (V7.0)', color='red', marker='o', linestyle='-')
        ax2.plot(data['snr'], data['util_mean'], label='CNN Utilization', color='green', marker='*', linestyle='--')
        
        ax1.set_xlabel('SNR [dB]', fontsize=12)
        ax1.set_ylabel('Symbol Error Rate (SER)', fontsize=12)
        ax2.set_ylabel('CNN Utilization (%)', fontsize=12, color='green')
        
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        ax1.set_ylim([1e-4, 1.1]) 
        ax2.set_ylim([-5, 105])
        
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', bbox_to_anchor=(0.95, 0.95))
        
        plt.title(f'V7.0 Multi-Hypothesis Performance ({title})', fontsize=14)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # V7.0은 연산량이 어마어마하므로 테스트 속도를 위해 seed를 1~2개로 먼저 검증 권장
    seeds = [2026] 
    test_snrs = [-21, -19, -17, -15, -13]
    all_runs = []

    for seed in seeds:
        print(f"\n{'='*40}\n[RUN: Seed {seed} | V7.0 Multi-Hypothesis]\n{'='*40}")
        set_seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sim = GPUOnlineSimulator(sf=7, device=device)
        
        # V7.0: 2D CNN (153가설 적용)
        model = Hypothesis2DCNN(num_classes=sim.M, num_hypotheses=153, num_bins=128, in_channels=2).to(device)
        
        max_cfo_hz = 0.35 * (sim.bw/sim.M)
        
        # 데이터 크기를 V6 대비 조금 줄여 1차 런타임 최적화
        ds_val = create_fixed_feature_dataset(sim, 8000, (-20, 0), 0.35, seed=seed)
        ds_calib = create_fixed_waveform_dataset(sim, 8000, test_snrs, max_cfo_hz, seed=seed+1)
        ds_test_seen = create_fixed_waveform_dataset(sim, 10000, test_snrs, max_cfo_hz, seed=seed+2)
        ds_test_unseen = create_fixed_waveform_dataset(sim, 10000, test_snrs, max_cfo_hz * 1.5, seed=seed+3)

        # DataLoader Batch Size 하향 조정 (512 -> 64) OOM 방지
        dl_train = DataLoader(OnlineParametersDataset(sim.M, 40000, (-20, 0), 0.35, sim.bw), batch_size=64)
        dl_val = DataLoader(ds_val, batch_size=64)
        
        model = train_online_model(model, sim, dl_train, dl_val, max_cfo_hz, num_epochs=15)

        policy = calibrate_adaptive_policy_joint(model, sim, ds_calib, max_cfo_hz)
        
        res_fixed_seen = run_evaluation(model, sim, ds_test_seen, max_cfo_hz, 1.5)
        res_adapt_seen = run_evaluation(model, sim, ds_test_seen, max_cfo_hz, policy)
        res_fixed_unseen = run_evaluation(model, sim, ds_test_unseen, max_cfo_hz * 1.5, 1.5)
        res_adapt_unseen = run_evaluation(model, sim, ds_test_unseen, max_cfo_hz * 1.5, policy)

        for snr in test_snrs:
            all_runs.append({'seed': seed, 'snr': snr, 'type': 'fixed_seen', **res_fixed_seen[snr]})
            all_runs.append({'seed': seed, 'snr': snr, 'type': 'adapt_seen', **res_adapt_seen[snr]})
            all_runs.append({'seed': seed, 'snr': snr, 'type': 'fixed_unseen', **res_fixed_unseen[snr]})
            all_runs.append({'seed': seed, 'snr': snr, 'type': 'adapt_unseen', **res_adapt_unseen[snr]})

    df = pd.DataFrame(all_runs)
    metric_cols = ['ser_g', 'ser_c', 'ser_h', 'per_g', 'per_c', 'per_h', 'util', 'th']
    
    summary = df.groupby(['type', 'snr'])[metric_cols].agg(['mean', 'std']).reset_index()
    summary.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in summary.columns.values]
    
    n_runs_df = df.groupby(['type', 'snr'])['seed'].count().reset_index()
    n_runs_df.rename(columns={'seed': 'n_runs'}, inplace=True)
    summary = pd.merge(summary, n_runs_df, on=['type', 'snr'])

    summary.to_csv("experiment_v7_summary.csv", index=False)
    print("\n>> V7.0 실험 완료. 결과가 'experiment_v7_summary.csv'에 저장되었습니다.")
    
    plot_summary(summary)
    print(">> 시각화 그래프가 'experiment_v7_seen_plot.png' 등급으로 저장되었습니다.")

if __name__ == "__main__":
    main()