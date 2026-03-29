import torch
import matplotlib.pyplot as plt

def evaluate_hybrid_packet_level(model, simulator, snr_list, max_cfo_hz, use_multipath, benchmark_name, packet_size=20, threshold=1.5):
    device = simulator.device
    model.to(device)
    model.eval()
    
    results = {"Grouped SER": [], "Hybrid SER": [], "Grouped PER": [], "Hybrid PER": [], "CNN Utilization": []}
    
    print(f"\n[하이브리드 패킷 평가: {benchmark_name}]")
    print("-" * 80)

    for snr in snr_list:
        num_packets = 2500 if snr >= -10 else (500 if snr >= -15 else 150)
        num_symbols = num_packets * packet_size
        batch_size = 2000
        num_batches = (num_symbols + batch_size - 1) // batch_size

        cor_g_sym, cor_h_sym = 0, 0
        err_g_pkt, err_h_pkt = 0, 0
        cnn_used_count = 0

        # 결과를 저장할 텐서 (패킷 에러 계산용)
        all_labels = []
        all_pred_g = []
        all_pred_h = []
        
        generated_symbols = 0

        with torch.no_grad():
            for _ in range(num_batches):
                current_batch = min(batch_size, num_symbols - generated_symbols)
                
                labels = torch.randint(0, simulator.M, (current_batch,), device=device)
                snrs = torch.full((current_batch,), snr, device=device)
                cfos = torch.empty(current_batch, device=device).uniform_(-max_cfo_hz, max_cfo_hz)

                rx_signals = simulator.generate_batch(labels, snrs, cfos, use_multipath)
                
                # 1. Classical Grouped-Bin 검출
                grouped_energy, _ = simulator.baseline_grouped_bin(rx_signals)
                pred_grouped = torch.argmax(grouped_energy, dim=1)
                
                # 2. 신뢰도(Confidence) 계산 (Peak1 / Peak2 비율)
                top2_values, _ = torch.topk(grouped_energy, 2, dim=1)
                confidence = top2_values[:, 0] / (top2_values[:, 1] + 1e-9)
                
                # 3. Hybrid 검출 (Threshold 미만인 경우만 CNN 호출)
                use_cnn_mask = confidence < threshold
                pred_hybrid = pred_grouped.clone()
                cnn_count = use_cnn_mask.sum().item()
                cnn_used_count += cnn_count
                
                if cnn_count > 0:
                    rx_signals_cnn = rx_signals[use_cnn_mask]
                    features = simulator.extract_features(rx_signals_cnn)
                    cnn_outputs = model(features)
                    pred_cnn = torch.argmax(cnn_outputs, dim=1)
                    pred_hybrid[use_cnn_mask] = pred_cnn

                all_labels.append(labels)
                all_pred_g.append(pred_grouped)
                all_pred_h.append(pred_hybrid)
                
                generated_symbols += current_batch

        labels_tensor = torch.cat(all_labels)
        pred_g_tensor = torch.cat(all_pred_g)
        pred_h_tensor = torch.cat(all_pred_h)

        # Symbol-level 계산
        cor_g_sym = (pred_g_tensor == labels_tensor).sum().item()
        cor_h_sym = (pred_h_tensor == labels_tensor).sum().item()

        # Packet-level 계산
        labels_pkt = labels_tensor.view(num_packets, packet_size)
        pred_g_pkt = pred_g_tensor.view(num_packets, packet_size)
        pred_h_pkt = pred_h_tensor.view(num_packets, packet_size)

        err_g_pkt = (torch.any(labels_pkt != pred_g_pkt, dim=1)).sum().item()
        err_h_pkt = (torch.any(labels_pkt != pred_h_pkt, dim=1)).sum().item()

        results["Grouped SER"].append(1.0 - (cor_g_sym / num_symbols))
        results["Hybrid SER"].append(1.0 - (cor_h_sym / num_symbols))
        results["Grouped PER"].append(err_g_pkt / num_packets)
        results["Hybrid PER"].append(err_h_pkt / num_packets)
        results["CNN Utilization"].append((cnn_used_count / num_symbols) * 100)

        print(f"SNR {snr:3d} dB | Util: {results['CNN Utilization'][-1]:5.1f}% | SER[Grp/Hyb]: {results['Grouped SER'][-1]:.4f} / {results['Hybrid SER'][-1]:.4f} | PER[Grp/Hyb]: {results['Grouped PER'][-1]:.4f} / {results['Hybrid PER'][-1]:.4f}")

    # 그래프 저장 로직
    filename = f"hybrid_per_ser_{benchmark_name.lower().replace(' ', '_')}.png"
    fig, ax1 = plt.subplots(figsize=(10, 7))

    ax1.set_xlabel('Signal-to-Noise Ratio (SNR) [dB]', fontsize=12)
    ax1.set_ylabel('Symbol / Packet Error Rate', fontsize=12)
    ax1.semilogy(snr_list, results["Grouped SER"], marker="x", linestyle=":", color="gray", label="Grouped SER")
    ax1.semilogy(snr_list, results["Hybrid SER"], marker="o", linestyle="-", color="red", label="Hybrid SER")
    ax1.semilogy(snr_list, results["Grouped PER"], marker="s", linestyle="--", color="blue", label="Grouped PER")
    ax1.semilogy(snr_list, results["Hybrid PER"], marker="^", linestyle="-.", color="purple", label="Hybrid PER")
    ax1.tick_params(axis='y')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.set_ylim([5e-5, 1.1])
    ax1.legend(loc="lower left")

    # 오른쪽 Y축 (CNN 사용률)
    ax2 = ax1.twinx()
    ax2.set_ylabel('CNN Utilization (%)', fontsize=12, color='green')
    ax2.plot(snr_list, results["CNN Utilization"], marker="d", linestyle="-", color="green", alpha=0.3, label="CNN Utilization")
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim([-5, 105])

    plt.title(f"Hybrid Receiver Performance: {benchmark_name}", fontsize=14)
    fig.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    return results