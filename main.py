import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from utils import set_seed
from simulator import GPUOnlineSimulator
from models import LoRaCNN
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
        ['experiment_v6_seen_plot.png', 'experiment_v6_unseen_plot.png']
    ):
        adapt_type = f'adapt_{channel_type}'
        if adapt_type not in summary_df['type'].unique():
            continue
            
        data = summary_df[summary_df['type'] == adapt_type]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        
        # 1. 기존 LoRa 방식 (Grouped Bin Baseline)
        ax1.semilogy(data['snr'], data['ser_g_mean'], label='Conventional LoRa (Grouped)', color='black', marker='x', linestyle=':')
        
        # 2. 순수 딥러닝 방식 (Full-CNN Upper Bound)
        ax1.semilogy(data['snr'], data['ser_c_mean'], label='Full CNN', color='orange', marker='v', linestyle='-.')
        
        # 3. 제안하는 방식 (Proposed Hybrid Adaptive)
        ax1.semilogy(data['snr'], data['ser_h_mean'], label='Proposed Hybrid', color='red', marker='o', linestyle='-')
        
        # 4. 제안하는 방식의 CNN 연산량 (Utilization)
        ax2.plot(data['snr'], data['util_mean'], label='CNN Utilization (Proposed)', color='green', marker='*', linestyle='--')
        
        ax1.set_xlabel('SNR [dB]', fontsize=12)
        ax1.set_ylabel('Symbol Error Rate (SER)', fontsize=12)
        ax2.set_ylabel('CNN Utilization (%)', fontsize=12, color='green')
        
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        ax1.set_ylim([1e-4, 1.1]) 
        ax2.set_ylim([-5, 105])
        
        # 범례 통합
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', bbox_to_anchor=(0.95, 0.95))
        
        plt.title(f'Performance Comparison ({title})', fontsize=14)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    seeds = [2024, 2025, 2026] 
    test_snrs = [-21, -19, -17, -15, -13]
    all_runs = []

    for seed in seeds:
        print(f"\n{'='*30}\n[RUN: Seed {seed}]\n{'='*30}")
        set_seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sim = GPUOnlineSimulator(sf=7, device=device)
        model = LoRaCNN(sim.M, sim.N).to(device)
        
        ds_val = create_fixed_feature_dataset(sim, 12000, (-20, 0), 0.35, seed=seed)
        ds_calib = create_fixed_waveform_dataset(sim, 10000, test_snrs, 0.35 * (sim.bw/sim.M), seed=seed+1)
        ds_test_seen = create_fixed_waveform_dataset(sim, 20000, test_snrs, 0.35 * (sim.bw/sim.M), seed=seed+2)
        ds_test_unseen = create_fixed_waveform_dataset(sim, 20000, test_snrs, 0.35 * (sim.bw/sim.M) * 1.5, seed=seed+3)

        dl_train = DataLoader(OnlineParametersDataset(sim.M, 68000, (-20, 0), 0.35, sim.bw), batch_size=512)
        dl_val = DataLoader(ds_val, batch_size=512)
        model = train_online_model(model, sim, dl_train, dl_val, num_epochs=20)

        policy = calibrate_adaptive_policy_joint(model, sim, ds_calib)
        
        res_fixed_seen = run_evaluation(model, sim, ds_test_seen, 1.5)
        res_adapt_seen = run_evaluation(model, sim, ds_test_seen, policy)
        res_fixed_unseen = run_evaluation(model, sim, ds_test_unseen, 1.5)
        res_adapt_unseen = run_evaluation(model, sim, ds_test_unseen, policy)

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

    summary.to_csv("experiment_v6_summary.csv", index=False)
    print("\n>> 실험 완료. 통계 결과가 'experiment_v6_summary.csv'에 저장되었습니다.")
    
    plot_summary(summary)
    print(">> 시각화 그래프가 'experiment_v6_seen_plot.png' 및 'experiment_v6_unseen_plot.png'로 저장되었습니다.")

if __name__ == "__main__":
    main()