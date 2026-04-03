import os
import torch
from torch.utils.data import DataLoader

from utils import set_seed
from simulator import GPUOnlineSimulator
from dataset import OnlineParametersDataset, FixedParametersDataset
from models import LoRaCNN 
from training import train_online_model
from evaluation import evaluate_hybrid_packet_level, sweep_thresholds

def main():
    set_seed(2026)
    os.makedirs("saved_models", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sim = GPUOnlineSimulator(sf=7, bw=125e3, fs=1e6, device=device)
    model = LoRaCNN(num_classes=sim.M, input_length=sim.N, in_channels=2)
    
    # 모델 이름을 바꿔서 새롭게 학습된 가중치로 저장되도록 함
    model_name = "lora_resnet_hybrid_v4_seed2026.pth"
    model_path = os.path.join("saved_models", model_name)

    total_train_samples = 80000 
    max_cfo_hz = 0.35 * (sim.bw / sim.M)

    # 1. 모델 가중치 확인 및 훈련
    if os.path.exists(model_path):
        print(f">> 저장된 모델({model_name}) 가중치를 로드합니다.")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    else:
        print(f"\n>> 가중치 파일이 없습니다. {model_name}의 딥러닝 훈련을 새로 시작합니다")
        ds_train = OnlineParametersDataset(sim.M, int(total_train_samples * 0.85), (-20, 0), 0.35, sim.bw)
        ds_val = FixedParametersDataset(sim.M, int(total_train_samples * 0.15), (-20, 0), 0.35, sim.bw)
        
        dl_train = DataLoader(ds_train, batch_size=512, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=512, shuffle=False)
        
        model = train_online_model(model, sim, dl_train, dl_val, num_epochs=20, lr=0.0005)
        torch.save(model.state_dict(), model_path)

    # =========================================================================
    # Phase 1: Metric 비교 & 적응형 임계값 탐색 (Data Profiling)
    # =========================================================================
    print("\n" + "="*50 + "\n[Phase 1: Confidence Metric Profiling]\n" + "="*50)
    target_snr = -17
    
    # 1. Ratio 테스트
    sweep_thresholds(model, sim, target_snr, max_cfo_hz, True, 
                     thresholds=[1.1, 1.3, 1.5, 2.0], num_packets=3000, conf_type='ratio')
    
    # 2. Margin 테스트 (에너지 차이값은 스케일이 커서 threshold를 높게 잡아야 함)
    sweep_thresholds(model, sim, target_snr, max_cfo_hz, True, 
                     thresholds=[10, 50, 100, 300], num_packets=3000, conf_type='margin')

    # =========================================================================
    # Phase 2: SNR-Adaptive Policy 적용 평가 (V3.0 Fixed vs V4.0 Adaptive)
    # =========================================================================
    print("\n" + "="*50 + "\n[Phase 2: SNR-Adaptive Policy Evaluation]\n" + "="*50)
    test_snrs = [-21, -19, -17, -15, -13]
    
    adaptive_policy = {
        -25: 1.50, -23: 1.50, -21: 1.50, 
        -19: 1.45, -17: 1.35, -15: 1.25, 
        -13: 1.10, -11: 1.10
    }

    # 비교 1: 기존 V3.0 (Fixed Th=1.5)
    evaluate_hybrid_packet_level(
        model, sim, test_snrs, max_cfo_hz, True, 
        benchmark_name="Baseline V3.0 (Fixed Ratio=1.5)", 
        threshold_policy=1.5, conf_type='ratio'
    )

    # 비교 2: 새로운 V4.0 (SNR-Adaptive Ratio)
    evaluate_hybrid_packet_level(
        model, sim, test_snrs, max_cfo_hz, True, 
        benchmark_name="Proposed V4.0 (Adaptive Ratio Policy)", 
        threshold_policy=adaptive_policy, conf_type='ratio'
    )

if __name__ == "__main__":
    main()