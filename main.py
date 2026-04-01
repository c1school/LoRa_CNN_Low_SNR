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
    
    # 모델 파일명에 버전 및 실험 환경 명시
    model_name = "lora_hybrid_cnn_v3_seed2026_cfo0.35_mpTrue.pth"
    model_path = os.path.join("saved_models", model_name)

    total_train_samples = 80000 
    max_cfo_hz = 0.35 * (sim.bw / sim.M)

    if os.path.exists(model_path):
        print(f">> 저장된 모델({model_name}) 가중치를 로드합니다.")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    else:
        print(f">> {model_name} 학습을 시작합니다.")
        # Train은 Online (매 에포크 무작위), Val은 Fixed (고정)
        ds_train = OnlineParametersDataset(sim.M, int(total_train_samples * 0.85), (-20, 0), 0.35, sim.bw)
        ds_val = FixedParametersDataset(sim.M, int(total_train_samples * 0.15), (-20, 0), 0.35, sim.bw)
        
        dl_train = DataLoader(ds_train, batch_size=512, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=512, shuffle=False)
        
        model = train_online_model(model, sim, dl_train, dl_val, num_epochs=20, lr=0.0005)
        torch.save(model.state_dict(), model_path)

    # 1. Threshold Sweep 분석 (취약 구간인 -19dB, -17dB 측정)
    sweep_thresholds(
        model, sim, target_snr=-19, max_cfo_hz=max_cfo_hz, use_multipath=True, 
        thresholds=[1.1, 1.3, 1.5, 2.0, 3.0]
    )
    sweep_thresholds(
        model, sim, target_snr=-17, max_cfo_hz=max_cfo_hz, use_multipath=True, 
        thresholds=[1.1, 1.3, 1.5, 2.0, 3.0]
    )

    test_snrs = list(range(-25, 1, 2))

    # 2. Scenario A: Seen Channel
    evaluate_hybrid_packet_level(
        model, sim, test_snrs, 
        max_cfo_hz=max_cfo_hz, 
        use_multipath=True, 
        benchmark_name="Scenario A - Seen Channel",
        packet_size=20,
        threshold=1.5
    )

    # 3. Scenario B: Unseen Harsher Channel
    evaluate_hybrid_packet_level(
        model, sim, test_snrs, 
        max_cfo_hz=max_cfo_hz * 1.5, 
        use_multipath=True, 
        benchmark_name="Scenario B - Unseen Harsher Channel",
        packet_size=20,
        threshold=1.5
    )

if __name__ == "__main__":
    main()