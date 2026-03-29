import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from utils import set_seed
from simulator import GPUOnlineSimulator
from dataset import OnlineParametersDataset
from models import LoRaCNN
from training import train_online_model
from evaluation import evaluate_hybrid_packet_level

def main():
    set_seed()
    os.makedirs("saved_models", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sim = GPUOnlineSimulator(sf=7, bw=125e3, fs=1e6, device=device)
    model = LoRaCNN(num_classes=sim.M, input_length=sim.N, in_channels=2)
    model_path = "saved_models/lora_hybrid_cnn_v2.pth"

    total_train_samples = 80000 
    max_cfo_hz = 0.35 * (sim.bw / sim.M)

    if os.path.exists(model_path):
        print(">> 저장된 하이브리드 모델 가중치를 로드합니다.")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(">> GPU 기반 온라인 실시간 데이터 학습을 시작합니다.")
        ds_train = OnlineParametersDataset(sim.M, int(total_train_samples * 0.85), (-20, 0), 0.35, sim.bw)
        ds_val = OnlineParametersDataset(sim.M, int(total_train_samples * 0.15), (-20, 0), 0.35, sim.bw)
        
        dl_train = DataLoader(ds_train, batch_size=512, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=512, shuffle=False)
        
        model = train_online_model(model, sim, dl_train, dl_val, num_epochs=20, lr=0.0005)
        torch.save(model.state_dict(), model_path)

    test_snrs = list(range(-25, 1, 2))

    # Scenario A: 학습 환경과 유사한 조건
    evaluate_hybrid_packet_level(
        model, sim, test_snrs, 
        max_cfo_hz=max_cfo_hz, 
        use_multipath=True, 
        benchmark_name="Scenario A - Seen Channel Hybrid Test",
        packet_size=20,
        threshold=1.5
    )

    # Scenario B: Unseen 가혹 조건 (CFO 범위 1.5배 확장)
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