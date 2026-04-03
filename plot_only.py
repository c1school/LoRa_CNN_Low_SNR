#CSV 데이터를 읽어와 그래프만 그림
import pandas as pd
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    try:
        df = pd.read_csv("experiment_v6_summary.csv")
        plot_summary(df)
        print(">> 기존 CSV 파일을 읽어 'Conventional vs Full-CNN vs Proposed Hybrid' 비교 그래프 재생성을 완료했습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")