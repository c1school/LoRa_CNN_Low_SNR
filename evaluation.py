import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def get_confidence(grouped_energy, conf_type='ratio'):
    top2_values, _ = torch.topk(grouped_energy, 2, dim=1)
    top1 = top2_values[:, 0]
    top2 = top2_values[:, 1]
    
    if conf_type == 'ratio':
        return top1 / (top2 + 1e-9)
    elif conf_type == 'norm_margin':
        return (top1 - top2) / (top1 + 1e-9)
    elif conf_type == 'entropy':
        probs = F.softmax(grouped_energy, dim=1)
        return -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
    else:
        raise ValueError("Invalid conf_type")

def calibrate_adaptive_policy(model, simulator, snr_list, max_cfo_hz, use_multipath, conf_type='ratio', calibration_samples=10000):
    """
    고정된 성능 저하 한계선(target_ser)을 만족하면서 Util을 최소화하는 적응형 임계값을 역산
    """
    device = simulator.device
    model.to(device)
    model.eval()
    policy = {}
    
    print(f"\n[알고리즘 기반 적응형 임계값 자동 추출 | Metric: {conf_type.upper()}]")
    
    for snr in snr_list:
        with torch.no_grad():
            labels = torch.randint(0, simulator.M, (calibration_samples,), device=device)
            snrs = torch.full((calibration_samples,), snr, device=device)
            cfos = torch.empty(calibration_samples, device=device).uniform_(-max_cfo_hz, max_cfo_hz)
            
            rx_signals = simulator.generate_batch(labels, snrs, cfos, use_multipath)
            grouped_energy, _ = simulator.baseline_grouped_bin(rx_signals)
            pred_grouped = torch.argmax(grouped_energy, dim=1)
            confidence = get_confidence(grouped_energy, conf_type)
            
            features = simulator.extract_features(rx_signals)
            pred_cnn = torch.argmax(model(features), dim=1)
            
            ser_g = 1.0 - (pred_grouped == labels).float().mean().item()
            ser_c = 1.0 - (pred_cnn == labels).float().mean().item()
            
            # 허용 에러율: 두 기본 모델 중 우수한 수치 대비 0.2%p 여유 부여
            target_ser = min(ser_g, ser_c) + 0.002 
            
            best_th = 1.5 if conf_type == 'ratio' else 0.5
            min_util = 100.0
            
            thresholds = torch.linspace(1.0, 3.0, 41) if conf_type == 'ratio' else torch.linspace(0.0, 1.0, 41)
            
            for th in thresholds:
                use_cnn = (confidence < th) if conf_type != 'entropy' else (confidence > th)
                pred_h = torch.where(use_cnn, pred_cnn, pred_grouped)
                ser_h = 1.0 - (pred_h == labels).float().mean().item()
                util = use_cnn.float().mean().item() * 100
                
                if ser_h <= target_ser and util < min_util:
                    best_th = th.item()
                    min_util = util
                    
            policy[snr] = round(best_th, 2)
            print(f"SNR {snr:3d} dB -> 도출된 Threshold: {policy[snr]:.2f} (예상 Util: {min_util:.1f}%)")
            
    return policy

def evaluate_hybrid_packet_level(model, simulator, snr_list, max_cfo_hz, use_multipath, benchmark_name, threshold_policy=1.5, conf_type='ratio'):
    device = simulator.device
    model.to(device)
    model.eval()
    packet_size = 20
    
    results = {"Grouped SER": [], "CNN SER": [], "Hybrid SER": [], 
               "Grouped PER": [], "CNN PER": [], "Hybrid PER": [], "CNN Utilization": []}
    
    print(f"\n[{benchmark_name} | Metric: {conf_type.upper()}]")
    print(f"{'SNR':>5} | {'Th':>4} | {'Util%':>6} | {'Grp SER':>7} | {'CNN SER':>7} | {'Hyb SER':>7} | {'Grp PER':>7} | {'CNN PER':>7} | {'Hyb PER':>7}")
    print("-" * 90)

    for snr in snr_list:
        num_packets = 2000 if snr >= -15 else 1000 
        num_symbols = num_packets * packet_size
        batch_size = 2000
        num_batches = (num_symbols + batch_size - 1) // batch_size

        all_labels, all_pred_g, all_pred_c, all_pred_h = [], [], [], []
        cnn_used_count = 0
        generated_symbols = 0

        th = threshold_policy.get(snr, 1.5) if isinstance(threshold_policy, dict) else threshold_policy

        with torch.no_grad():
            for _ in range(num_batches):
                current_batch = min(batch_size, num_symbols - generated_symbols)
                labels = torch.randint(0, simulator.M, (current_batch,), device=device)
                snrs = torch.full((current_batch,), snr, device=device)
                cfos = torch.empty(current_batch, device=device).uniform_(-max_cfo_hz, max_cfo_hz)

                rx_signals = simulator.generate_batch(labels, snrs, cfos, use_multipath)
                
                grouped_energy, _ = simulator.baseline_grouped_bin(rx_signals)
                pred_grouped = torch.argmax(grouped_energy, dim=1)
                confidence = get_confidence(grouped_energy, conf_type)
                
                features = simulator.extract_features(rx_signals)
                pred_cnn = torch.argmax(model(features), dim=1)
                
                if conf_type == 'entropy':
                    use_cnn_mask = confidence > th
                else:
                    use_cnn_mask = confidence < th
                    
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

        ser_g = 1.0 - (pred_g_tensor == labels_tensor).float().mean().item()
        ser_c = 1.0 - (pred_c_tensor == labels_tensor).float().mean().item()
        ser_h = 1.0 - (pred_h_tensor == labels_tensor).float().mean().item()
        util = (cnn_used_count / num_symbols) * 100

        labels_pkt = labels_tensor.view(num_packets, packet_size)
        per_g = (torch.any(labels_pkt != pred_g_tensor.view(num_packets, packet_size), dim=1)).float().mean().item()
        per_c = (torch.any(labels_pkt != pred_c_tensor.view(num_packets, packet_size), dim=1)).float().mean().item()
        per_h = (torch.any(labels_pkt != pred_h_tensor.view(num_packets, packet_size), dim=1)).float().mean().item()

        results["Grouped SER"].append(ser_g)
        results["CNN SER"].append(ser_c)
        results["Hybrid SER"].append(ser_h)
        results["Grouped PER"].append(per_g)
        results["CNN PER"].append(per_c)
        results["Hybrid PER"].append(per_h)
        results["CNN Utilization"].append(util)

        print(f"{snr:5d} | {th:4.2f} | {util:5.1f}% | {ser_g:7.4f} | {ser_c:7.4f} | {ser_h:7.4f} | {per_g:7.4f} | {per_c:7.4f} | {per_h:7.4f}")

    safe_name = benchmark_name.replace(' ', '_')
    filename = f"{safe_name}.png"
    fig, ax1 = plt.subplots(figsize=(10, 7))

    ax1.set_xlabel('SNR [dB]', fontsize=12)
    ax1.set_ylabel('Error Rate', fontsize=12)
    
    ax1.semilogy(snr_list, results["Grouped SER"], marker="x", linestyle=":", color="gray", label="Grp SER")
    ax1.semilogy(snr_list, results["CNN SER"], marker="v", linestyle=":", color="orange", label="CNN SER")
    ax1.semilogy(snr_list, results["Hybrid SER"], marker="o", linestyle="-", color="red", label="Hyb SER")
    
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.set_ylim([1e-4, 1.1])
    ax1.legend(loc="lower left")

    ax2 = ax1.twinx()
    ax2.set_ylabel('CNN Utilization (%)', fontsize=12, color='green')
    ax2.plot(snr_list, results["CNN Utilization"], marker="*", linestyle="-", color="green", alpha=0.3, label="CNN Util")
    ax2.set_ylim([-5, 105])

    plt.title(benchmark_name, fontsize=14)
    fig.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    return results