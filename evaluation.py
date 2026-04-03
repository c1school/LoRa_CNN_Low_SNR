import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def get_confidence(grouped_energy, conf_type='ratio'):
    """다양한 Confidence 지표를 계산하는 헬퍼 함수"""
    top2_values, _ = torch.topk(grouped_energy, 2, dim=1)
    
    if conf_type == 'ratio':
        # Ratio: 크면 확실, 작으면 불확실
        return top2_values[:, 0] / (top2_values[:, 1] + 1e-9)
    elif conf_type == 'margin':
        # Margin: 크면 확실, 작으면 불확실 (차이값)
        return top2_values[:, 0] - top2_values[:, 1]
    elif conf_type == 'entropy':
        # Entropy: 작으면 확실(뾰족함), 크면 불확실(평평함)
        probs = F.softmax(grouped_energy, dim=1)
        return -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
    else:
        raise ValueError("Invalid conf_type")

def sweep_thresholds(model, simulator, target_snr, max_cfo_hz, use_multipath, thresholds, packet_size=20, num_packets=2000, conf_type='ratio'):
    device = simulator.device
    model.to(device)
    model.eval()
    
    num_symbols = num_packets * packet_size
    batch_size = 2000
    num_batches = (num_symbols + batch_size - 1) // batch_size
    
    print(f"\n[Sweep 분석: SNR {target_snr} dB | Metric: {conf_type.upper()}]")
    print("=" * 110)
    print(f"{'Threshold':<10} | {'Grp SER':<10} | {'CNN SER':<10} | {'Hyb SER':<10} | {'Hyb PER':<10} | {'Util (%)':<10} | {'ΔSER/ΔUtil (x100)':<15}")
    print("-" * 110)
    
    all_grouped_conf, all_pred_g, all_pred_c, all_labels = [], [], [], []
    
    generated_symbols = 0
    with torch.no_grad():
        for _ in range(num_batches):
            current_batch = min(batch_size, num_symbols - generated_symbols)
            labels = torch.randint(0, simulator.M, (current_batch,), device=device)
            snrs = torch.full((current_batch,), target_snr, device=device)
            cfos = torch.empty(current_batch, device=device).uniform_(-max_cfo_hz, max_cfo_hz)

            rx_signals = simulator.generate_batch(labels, snrs, cfos, use_multipath)
            
            # Grouped & Confidence
            grouped_energy, _ = simulator.baseline_grouped_bin(rx_signals)
            pred_grouped = torch.argmax(grouped_energy, dim=1)
            confidence = get_confidence(grouped_energy, conf_type)
            
            # CNN
            features = simulator.extract_features(rx_signals)
            pred_cnn = torch.argmax(model(features), dim=1)
            
            all_labels.append(labels)
            all_pred_g.append(pred_grouped)
            all_pred_c.append(pred_cnn)
            all_grouped_conf.append(confidence)
            generated_symbols += current_batch

    labels_tensor = torch.cat(all_labels)
    pred_g_tensor = torch.cat(all_pred_g)
    pred_c_tensor = torch.cat(all_pred_c)
    conf_tensor = torch.cat(all_grouped_conf)
    
    ser_g = 1.0 - ((pred_g_tensor == labels_tensor).sum().item() / num_symbols)
    ser_c = 1.0 - ((pred_c_tensor == labels_tensor).sum().item() / num_symbols)
    labels_pkt = labels_tensor.view(num_packets, packet_size)
    
    prev_ser, prev_util = ser_g, 0.0

    for th in thresholds:
        # Entropy는 클수록 불확실하므로 부등호 방향이 반대임
        if conf_type == 'entropy':
            use_cnn_mask = conf_tensor > th
        else:
            use_cnn_mask = conf_tensor < th
            
        pred_hybrid = torch.where(use_cnn_mask, pred_c_tensor, pred_g_tensor)
        ser_h = 1.0 - ((pred_hybrid == labels_tensor).sum().item() / num_symbols)
        
        pred_h_pkt = pred_hybrid.view(num_packets, packet_size)
        per_h = (torch.any(labels_pkt != pred_h_pkt, dim=1)).sum().item() / num_packets
        
        utilization = (use_cnn_mask.sum().item() / num_symbols) * 100
        
        delta_ser = prev_ser - ser_h
        delta_util = utilization - prev_util
        efficiency = (delta_ser / delta_util * 100) if delta_util > 0 else 0.0
        
        print(f"{th:<10.4f} | {ser_g:<10.4f} | {ser_c:<10.4f} | {ser_h:<10.4f} | {per_h:<10.4f} | {utilization:<10.1f} | {efficiency:<15.4f}")
        prev_ser, prev_util = ser_h, utilization
        
    print("=" * 110)

def evaluate_hybrid_packet_level(model, simulator, snr_list, max_cfo_hz, use_multipath, benchmark_name, packet_size=20, threshold_policy=1.5, conf_type='ratio'):
    device = simulator.device
    model.to(device)
    model.eval()
    
    results = {"Grouped SER": [], "CNN SER": [], "Hybrid SER": [], 
               "Grouped PER": [], "CNN PER": [], "Hybrid PER": [], "CNN Utilization": []}
    
    print(f"\n[평가 시작: {benchmark_name} | Metric: {conf_type.upper()}]")
    print("-" * 100)

    for snr in snr_list:
        # 신뢰도를 위해 저 SNR 구간에서도 패킷 수를 1000개(2만 심볼) 이상으로 강력하게 보장
        num_packets = 2000 if snr >= -15 else 1000 
        num_symbols = num_packets * packet_size
        batch_size = 2000
        num_batches = (num_symbols + batch_size - 1) // batch_size

        all_labels, all_pred_g, all_pred_c, all_pred_h = [], [], [], []
        cnn_used_count = 0
        generated_symbols = 0

        # Adaptive Threshold 적용 (Dict면 SNR에 맞는 값 추출, 없거나 Float면 고정값 사용)
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

        results["Grouped SER"].append(1.0 - (pred_g_tensor == labels_tensor).sum().item() / num_symbols)
        results["CNN SER"].append(1.0 - (pred_c_tensor == labels_tensor).sum().item() / num_symbols)
        results["Hybrid SER"].append(1.0 - (pred_h_tensor == labels_tensor).sum().item() / num_symbols)

        labels_pkt = labels_tensor.view(num_packets, packet_size)
        results["Grouped PER"].append((torch.any(labels_pkt != pred_g_tensor.view(num_packets, packet_size), dim=1)).sum().item() / num_packets)
        results["CNN PER"].append((torch.any(labels_pkt != pred_c_tensor.view(num_packets, packet_size), dim=1)).sum().item() / num_packets)
        results["Hybrid PER"].append((torch.any(labels_pkt != pred_h_tensor.view(num_packets, packet_size), dim=1)).sum().item() / num_packets)
        results["CNN Utilization"].append((cnn_used_count / num_symbols) * 100)

        print(f"SNR {snr:3d} dB | Th: {th:.2f} | Util: {results['CNN Utilization'][-1]:5.1f}% | "
              f"SER[Grp/Cnn/Hyb]: {results['Grouped SER'][-1]:.4f} / {results['CNN SER'][-1]:.4f} / {results['Hybrid SER'][-1]:.4f}")

    return results