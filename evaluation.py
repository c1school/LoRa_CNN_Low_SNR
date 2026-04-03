import torch
import numpy as np
import torch.nn.functional as F

def get_confidence(grouped_energy, conf_type='ratio'):
    top2_values, _ = torch.topk(grouped_energy, 2, dim=1)
    t1, t2 = top2_values[:, 0], top2_values[:, 1]
    if conf_type == 'ratio': return t1 / (t2 + 1e-9)
    if conf_type == 'norm_margin': return (t1 - t2) / (t1 + 1e-9)
    if conf_type == 'entropy':
        p = F.softmax(grouped_energy, dim=1)
        return -torch.sum(p * torch.log(p + 1e-9), dim=1)
    return t1 / (t2 + 1e-9)

def calibrate_adaptive_policy_joint(model, simulator, calib_dict, max_cfo_hz, conf_type='ratio', ser_tol=0.002, per_tol=0.01):
    device = simulator.device
    model.eval()
    policy = {}
    packet_size = 20
    eval_batch_size = 128 # V7.0: GPU 메모리 오버플로우 방지를 위한 배치 사이즈
    
    # 훈련 시와 동일한 다중 가설 격자 생성
    cfo_grid, to_grid = simulator.generate_hypothesis_grid(max_cfo_hz, max_to_samples=4, cfo_steps=17, to_steps=9)
    
    for snr, dataset in calib_dict.items():
        labels, rx_signals = dataset.tensors
        labels, rx_signals = labels.to(device), rx_signals.to(device)
        num_samples = len(labels)
        num_packets = num_samples // packet_size
        
        pred_g_list, pred_c_list, conf_list = [], [], []
        
        print(f" -> [Calibration] SNR {snr:3d} dB 다중 가설 추론 중...")
        with torch.no_grad():
            # 메모리 보호를 위한 미니 배치 추론
            for i in range(0, num_samples, eval_batch_size):
                end = min(i + eval_batch_size, num_samples)
                rx_batch = rx_signals[i:end]
                
                # 1. 클래식 복조
                grouped_energy, _ = simulator.baseline_grouped_bin(rx_batch)
                pred_g_list.append(torch.argmax(grouped_energy, dim=1))
                conf_list.append(get_confidence(grouped_energy, conf_type))
                
                # 2. V7.0 딥러닝 다중 가설 복조
                features = simulator.extract_multi_hypothesis_bank(rx_batch, cfo_grid, to_grid)
                pred_c_list.append(torch.argmax(model(features), dim=1))
                
            pred_g = torch.cat(pred_g_list)
            pred_c = torch.cat(pred_c_list)
            conf = torch.cat(conf_list)
            
            # 메트릭 계산
            ser_g = 1.0 - (pred_g == labels).float().mean().item()
            ser_c = 1.0 - (pred_c == labels).float().mean().item()
            labels_pkt = labels.view(num_packets, packet_size)
            per_g = (torch.any(labels_pkt != pred_g.view(num_packets, packet_size), dim=1)).float().mean().item()
            per_c = (torch.any(labels_pkt != pred_c.view(num_packets, packet_size), dim=1)).float().mean().item()
            
            target_ser = min(ser_g, ser_c) + ser_tol
            target_per = min(per_g, per_c) + per_tol
            
            best_th, min_util = (3.0 if conf_type=='ratio' else 1.0), 100.0
            fallback_th, min_penalty = best_th, float('inf')
            valid_found = False
            
            search_space = torch.linspace(1.0, 3.0, 41) if conf_type=='ratio' else torch.linspace(0.0, 1.0, 41)
            
            for th in search_space:
                use_cnn = (conf < th) if conf_type != 'entropy' else (conf > th)
                pred_h = torch.where(use_cnn, pred_c, pred_g)
                s_h = 1.0 - (pred_h == labels).float().mean().item()
                p_h = (torch.any(labels_pkt != pred_h.view(num_packets, packet_size), dim=1)).float().mean().item()
                util = use_cnn.float().mean().item() * 100
                
                if s_h <= target_ser and p_h <= target_per and util < min_util:
                    best_th = th.item()
                    min_util = util
                    valid_found = True
                
                penalty = max(0, s_h - target_ser) * 10 + max(0, p_h - target_per) * 5 + (util / 100)
                if penalty < min_penalty:
                    min_penalty = penalty
                    fallback_th = th.item()
                    
            policy[snr] = round(best_th if valid_found else fallback_th, 2)
    return policy

def run_evaluation(model, simulator, test_dict, max_cfo_hz, policy, conf_type='ratio'):
    device = simulator.device
    model.eval()
    stats = {}
    packet_size = 20
    eval_batch_size = 128
    
    cfo_grid, to_grid = simulator.generate_hypothesis_grid(max_cfo_hz, max_to_samples=4, cfo_steps=17, to_steps=9)
    
    for snr, dataset in test_dict.items():
        labels, rx_signals = dataset.tensors
        labels, rx_signals = labels.to(device), rx_signals.to(device)
        num_samples = len(labels)
        num_packets = num_samples // packet_size
        th = policy.get(snr, 1.5) if isinstance(policy, dict) else policy
        
        pred_g_list, pred_c_list, conf_list = [], [], []
        
        with torch.no_grad():
            for i in range(0, num_samples, eval_batch_size):
                end = min(i + eval_batch_size, num_samples)
                rx_batch = rx_signals[i:end]
                
                grouped_energy, _ = simulator.baseline_grouped_bin(rx_batch)
                pred_g_list.append(torch.argmax(grouped_energy, dim=1))
                conf_list.append(get_confidence(grouped_energy, conf_type))
                
                features = simulator.extract_multi_hypothesis_bank(rx_batch, cfo_grid, to_grid)
                pred_c_list.append(torch.argmax(model(features), dim=1))
                
            pred_g = torch.cat(pred_g_list)
            pred_c = torch.cat(pred_c_list)
            conf = torch.cat(conf_list)
            
            use_cnn = (conf < th) if conf_type != 'entropy' else (conf > th)
            pred_h = torch.where(use_cnn, pred_c, pred_g)
            
            labels_pkt = labels.view(num_packets, packet_size)
            stats[snr] = {
                "ser_g": 1.0 - (pred_g == labels).float().mean().item(),
                "ser_c": 1.0 - (pred_c == labels).float().mean().item(),
                "ser_h": 1.0 - (pred_h == labels).float().mean().item(),
                "per_g": (torch.any(labels_pkt != pred_g.view(num_packets, packet_size), dim=1)).float().mean().item(),
                "per_c": (torch.any(labels_pkt != pred_c.view(num_packets, packet_size), dim=1)).float().mean().item(),
                "per_h": (torch.any(labels_pkt != pred_h.view(num_packets, packet_size), dim=1)).float().mean().item(),
                "util": use_cnn.float().mean().item() * 100,
                "th": th
            }
    return stats