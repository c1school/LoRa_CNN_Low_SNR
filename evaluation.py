import torch
import matplotlib.pyplot as plt

def sweep_thresholds(model, simulator, target_snr, max_cfo_hz, use_multipath, thresholds, packet_size=20, num_packets=2000):
    device = simulator.device
    model.to(device)
    model.eval()
    
    num_symbols = num_packets * packet_size
    batch_size = 2000
    num_batches = (num_symbols + batch_size - 1) // batch_size
    
    print(f"\n[Threshold Sweep 분석: SNR {target_snr} dB]")
    print("=" * 90)
    print(f"{'Threshold':<10} | {'Grouped SER':<12} | {'Full-CNN SER':<14} | {'Hybrid SER':<12} | {'Hybrid PER':<12} | {'CNN Util (%)':<12}")
    print("-" * 90)
    
    all_grouped_conf = []
    all_pred_g = []
    all_pred_c = []
    all_labels = []
    
    generated_symbols = 0
    with torch.no_grad():
        for _ in range(num_batches):
            current_batch = min(batch_size, num_symbols - generated_symbols)
            labels = torch.randint(0, simulator.M, (current_batch,), device=device)
            snrs = torch.full((current_batch,), target_snr, device=device)
            cfos = torch.empty(current_batch, device=device).uniform_(-max_cfo_hz, max_cfo_hz)

            rx_signals = simulator.generate_batch(labels, snrs, cfos, use_multipath)
            
            # Grouped
            grouped_energy, _ = simulator.baseline_grouped_bin(rx_signals)
            pred_grouped = torch.argmax(grouped_energy, dim=1)
            top2_values, _ = torch.topk(grouped_energy, 2, dim=1)
            confidence = top2_values[:, 0] / (top2_values[:, 1] + 1e-9)
            
            # Full-CNN
            features = simulator.extract_features(rx_signals)
            cnn_outputs = model(features)
            pred_cnn = torch.argmax(cnn_outputs, dim=1)
            
            all_labels.append(labels)
            all_pred_g.append(pred_grouped)
            all_pred_c.append(pred_cnn)
            all_grouped_conf.append(confidence)
            
            generated_symbols += current_batch

    labels_tensor = torch.cat(all_labels)
    pred_g_tensor = torch.cat(all_pred_g)
    pred_c_tensor = torch.cat(all_pred_c)
    conf_tensor = torch.cat(all_grouped_conf)
    
    cor_g_sym = (pred_g_tensor == labels_tensor).sum().item()
    cor_c_sym = (pred_c_tensor == labels_tensor).sum().item()
    ser_g = 1.0 - (cor_g_sym / num_symbols)
    ser_c = 1.0 - (cor_c_sym / num_symbols)
    
    labels_pkt = labels_tensor.view(num_packets, packet_size)
    
    # 그래프를 그리기 위해 데이터 수집
    plot_ser_h = []
    plot_per_h = []
    plot_util = []

    for th in thresholds:
        use_cnn_mask = conf_tensor < th
        pred_hybrid = torch.where(use_cnn_mask, pred_c_tensor, pred_g_tensor)
        
        cor_h_sym = (pred_hybrid == labels_tensor).sum().item()
        ser_h = 1.0 - (cor_h_sym / num_symbols)
        
        pred_h_pkt = pred_hybrid.view(num_packets, packet_size)
        err_h_pkt = (torch.any(labels_pkt != pred_h_pkt, dim=1)).sum().item()
        per_h = err_h_pkt / num_packets
        
        utilization = (use_cnn_mask.sum().item() / num_symbols) * 100
        
        plot_ser_h.append(ser_h)
        plot_per_h.append(per_h)
        plot_util.append(utilization)
        
        print(f"{th:<10.2f} | {ser_g:<12.4f} | {ser_c:<14.4f} | {ser_h:<12.4f} | {per_h:<12.4f} | {utilization:<12.1f}")
    print("=" * 90)

    # --- 여기서부터 Threshold Sweep 그래프 시각화 코드 추가 ---
    filename = f"threshold_sweep_snr{target_snr}.png"
    fig, ax1 = plt.subplots(figsize=(9, 6))

    ax1.set_xlabel('Confidence Threshold (Top1 / Top2 ratio)', fontsize=12)
    ax1.set_ylabel('Error Rate', fontsize=12)
    
    # 하이브리드 성능 변화선
    ax1.plot(thresholds, plot_ser_h, marker='o', linestyle='-', color='red', label='Hybrid SER')
    ax1.plot(thresholds, plot_per_h, marker='d', linestyle='-.', color='purple', label='Hybrid PER')
    
    # 기준선 (가로선)
    ax1.axhline(y=ser_g, color='gray', linestyle=':', label='Grouped SER Baseline')
    ax1.axhline(y=ser_c, color='orange', linestyle='--', label='Full-CNN SER Baseline')

    ax1.tick_params(axis='y')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    # 추천 Threshold(1.5) 하이라이트
    ax1.axvline(x=1.5, color='blue', linestyle=':', alpha=0.7)
    ax1.text(1.52, max(plot_per_h)*0.8, 'Recommended\nOperating Point\n(Th=1.5)', color='blue', fontsize=10)

    ax1.legend(loc="center left")

    # 오른쪽 Y축 (CNN 사용률)
    ax2 = ax1.twinx()
    ax2.set_ylabel('CNN Utilization (%)', fontsize=12, color='green')
    ax2.plot(thresholds, plot_util, marker='*', linestyle='-', color='green', alpha=0.6, label='CNN Utilization')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim([-5, 105])
    
    ax2.legend(loc="center right")

    plt.title(f"Trade-off Analysis vs Threshold (SNR = {target_snr} dB)", fontsize=14)
    fig.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def evaluate_hybrid_packet_level(model, simulator, snr_list, max_cfo_hz, use_multipath, benchmark_name, packet_size=20, threshold=1.5):
    device = simulator.device
    model.to(device)
    model.eval()
    
    results = {"Grouped SER": [], "CNN SER": [], "Hybrid SER": [], 
               "Grouped PER": [], "CNN PER": [], "Hybrid PER": [], "CNN Utilization": []}
    
    print(f"\n[평가 시작: {benchmark_name}]")
    print("-" * 100)

    for snr in snr_list:
        num_packets = 2500 if snr >= -10 else (500 if snr >= -15 else 150)
        num_symbols = num_packets * packet_size
        batch_size = 2000
        num_batches = (num_symbols + batch_size - 1) // batch_size

        all_labels, all_pred_g, all_pred_c, all_pred_h = [], [], [], []
        cnn_used_count = 0
        generated_symbols = 0

        with torch.no_grad():
            for _ in range(num_batches):
                current_batch = min(batch_size, num_symbols - generated_symbols)
                
                labels = torch.randint(0, simulator.M, (current_batch,), device=device)
                snrs = torch.full((current_batch,), snr, device=device)
                cfos = torch.empty(current_batch, device=device).uniform_(-max_cfo_hz, max_cfo_hz)

                rx_signals = simulator.generate_batch(labels, snrs, cfos, use_multipath)
                
                # Grouped
                grouped_energy, _ = simulator.baseline_grouped_bin(rx_signals)
                pred_grouped = torch.argmax(grouped_energy, dim=1)
                top2_values, _ = torch.topk(grouped_energy, 2, dim=1)
                confidence = top2_values[:, 0] / (top2_values[:, 1] + 1e-9)
                
                # Full-CNN
                features = simulator.extract_features(rx_signals)
                cnn_outputs = model(features)
                pred_cnn = torch.argmax(cnn_outputs, dim=1)
                
                # Hybrid
                use_cnn_mask = confidence < threshold
                pred_hybrid = torch.where(use_cnn_mask, pred_cnn, pred_grouped)
                cnn_used_count += use_cnn_mask.sum().item()

                all_labels.append(labels)
                all_pred_g.append(pred_grouped)
                all_pred_c.append(pred_cnn)
                all_pred_h.append(pred_hybrid)
                
                generated_symbols += current_batch

        labels_tensor = torch.cat(all_labels)
        pred_g_tensor = torch.cat(all_pred_g)
        pred_c_tensor = torch.cat(all_pred_c)
        pred_h_tensor = torch.cat(all_pred_h)

        # Symbol-level 계산
        results["Grouped SER"].append(1.0 - (pred_g_tensor == labels_tensor).sum().item() / num_symbols)
        results["CNN SER"].append(1.0 - (pred_c_tensor == labels_tensor).sum().item() / num_symbols)
        results["Hybrid SER"].append(1.0 - (pred_h_tensor == labels_tensor).sum().item() / num_symbols)

        # Packet-level 계산
        labels_pkt = labels_tensor.view(num_packets, packet_size)
        pred_g_pkt = pred_g_tensor.view(num_packets, packet_size)
        pred_c_pkt = pred_c_tensor.view(num_packets, packet_size)
        pred_h_pkt = pred_h_tensor.view(num_packets, packet_size)

        results["Grouped PER"].append((torch.any(labels_pkt != pred_g_pkt, dim=1)).sum().item() / num_packets)
        results["CNN PER"].append((torch.any(labels_pkt != pred_c_pkt, dim=1)).sum().item() / num_packets)
        results["Hybrid PER"].append((torch.any(labels_pkt != pred_h_pkt, dim=1)).sum().item() / num_packets)
        results["CNN Utilization"].append((cnn_used_count / num_symbols) * 100)

        print(f"SNR {snr:3d} dB | Util: {results['CNN Utilization'][-1]:5.1f}% | "
              f"SER[Grp/Cnn/Hyb]: {results['Grouped SER'][-1]:.4f} / {results['CNN SER'][-1]:.4f} / {results['Hybrid SER'][-1]:.4f} | "
              f"PER[Grp/Hyb]: {results['Grouped PER'][-1]:.4f} / {results['Hybrid PER'][-1]:.4f}")

    # 시각화 (3파전)
    filename = f"hybrid_v3_{benchmark_name.lower().replace(' ', '_')}.png"
    fig, ax1 = plt.subplots(figsize=(10, 7))

    ax1.set_xlabel('Signal-to-Noise Ratio (SNR) [dB]', fontsize=12)
    ax1.set_ylabel('Error Rate', fontsize=12)
    
    ax1.semilogy(snr_list, results["Grouped SER"], marker="x", linestyle=":", color="gray", label="Grouped SER")
    ax1.semilogy(snr_list, results["CNN SER"], marker="v", linestyle=":", color="orange", label="Full-CNN SER")
    ax1.semilogy(snr_list, results["Hybrid SER"], marker="o", linestyle="-", color="red", label="Hybrid SER")
    
    ax1.semilogy(snr_list, results["Grouped PER"], marker="s", linestyle="--", color="blue", label="Grouped PER")
    ax1.semilogy(snr_list, results["CNN PER"], marker="^", linestyle="--", color="brown", label="Full-CNN PER")
    ax1.semilogy(snr_list, results["Hybrid PER"], marker="d", linestyle="-.", color="purple", label="Hybrid PER")
    
    ax1.tick_params(axis='y')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.set_ylim([5e-5, 1.1])
    ax1.legend(loc="lower left")

    ax2 = ax1.twinx()
    ax2.set_ylabel('CNN Utilization (%)', fontsize=12, color='green')
    ax2.plot(snr_list, results["CNN Utilization"], marker="*", linestyle="-", color="green", alpha=0.3, label="CNN Util")
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim([-5, 105])

    plt.title(f"Performance Comparison: {benchmark_name}", fontsize=14)
    fig.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    return results