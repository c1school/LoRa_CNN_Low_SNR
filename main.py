import os
import torch
from torch.utils.data import DataLoader

from utils import set_seed
from simulator import GPUOnlineSimulator
from dataset import OnlineParametersDataset, create_fixed_validation_set
from models import LoRaCNN
from training import train_online_model
from evaluation import evaluate_hybrid_packet_level, calibrate_adaptive_policy

def run_experiment(seed, metric_type='ratio'):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sim = GPUOnlineSimulator(sf=7, bw=125e3, fs=1e6, device=device)
    model = LoRaCNN(num_classes=sim.M, input_length=sim.N, in_channels=2)
    
    model_name = f"lora_hybrid_cnn_seed{seed}.pth"
    model_path = os.path.join("saved_models", model_name)
    max_cfo_hz = 0.35 * (sim.bw / sim.M)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    else:
        total_train_samples = 80000 
        ds_train = OnlineParametersDataset(sim.M, int(total_train_samples * 0.85), (-20, 0), 0.35, sim.bw)
        ds_val = create_fixed_validation_set(sim, int(total_train_samples * 0.15), (-20, 0), 0.35)
        
        dl_train = DataLoader(ds_train, batch_size=512, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=512, shuffle=False)
        
        model = train_online_model(model, sim, dl_train, dl_val, num_epochs=20, lr=0.0005)
        torch.save(model.state_dict(), model_path)

    test_snrs = [-21, -19, -17, -15, -13]
    
    # 1. 자동 최적화 알고리즘을 통한 Adaptive Policy 도출
    adaptive_policy = calibrate_adaptive_policy(
        model, sim, test_snrs, max_cfo_hz, True, conf_type=metric_type
    )

    # 2. Ablation 1: Seen Channel (Fixed vs Adaptive)
    evaluate_hybrid_packet_level(
        model, sim, test_snrs, max_cfo_hz, True, 
        benchmark_name=f"Seed_{seed}_Seen_Fixed_1_5", 
        threshold_policy=1.5 if metric_type == 'ratio' else 0.5, conf_type=metric_type
    )

    evaluate_hybrid_packet_level(
        model, sim, test_snrs, max_cfo_hz, True, 
        benchmark_name=f"Seed_{seed}_Seen_Adaptive", 
        threshold_policy=adaptive_policy, conf_type=metric_type
    )

    # 3. Ablation 2: Unseen Harsher Channel (Fixed vs Adaptive)
    evaluate_hybrid_packet_level(
        model, sim, test_snrs, max_cfo_hz * 1.5, True, 
        benchmark_name=f"Seed_{seed}_Unseen_Fixed_1_5", 
        threshold_policy=1.5 if metric_type == 'ratio' else 0.5, conf_type=metric_type
    )

    evaluate_hybrid_packet_level(
        model, sim, test_snrs, max_cfo_hz * 1.5, True, 
        benchmark_name=f"Seed_{seed}_Unseen_Adaptive", 
        threshold_policy=adaptive_policy, conf_type=metric_type
    )

def main():
    os.makedirs("saved_models", exist_ok=True)
    
    # 다중 시드 검증 구조 (평균 산출을 위해 시드를 확장할 수 있습니다)
    seeds = [2026] 
    
    for seed in seeds:
        print("\n" + "="*70 + f"\n[Experiment Run | Seed: {seed}]\n" + "="*70)
        run_experiment(seed, metric_type='ratio')

if __name__ == "__main__":
    main()