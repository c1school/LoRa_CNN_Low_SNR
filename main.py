import os
import torch
from torch.utils.data import DataLoader, random_split

from utils import set_seed
from simulator import LoRaResearchSimulator
from dataset import LoRaResearchDataset
from models import LoRaCNN
from training import train_research_model
from evaluation import evaluate_ablation_model


# ============================================================
# 7. 메인 실행 블록
# ------------------------------------------------------------
# 순서:
# 1) 시뮬레이터 생성
# 2) Complex CNN 초기화/학습(or 로드)
# 3) Mag CNN 초기화/학습(or 로드)
# 4) 3개 시나리오에서 benchmark 수행
#    - Scenario A: Pure AWGN
#    - Scenario B: Seen Impaired
#    - Scenario C: Unseen Impaired
# ============================================================
def main():
    set_seed()
    os.makedirs("saved_models", exist_ok=True)

    # LoRa-like simulator 생성
    sim = LoRaResearchSimulator(sf=7, bw=125e3, fs=1e6)

    # 두 모델 초기화
    # Complex CNN: Real/Imag 2채널 입력
    # Mag CNN: magnitude-only 1채널 입력
    model_comp = LoRaCNN(num_classes=sim.M, input_length=sim.N, in_channels=2)
    model_mag = LoRaCNN(num_classes=sim.M, input_length=sim.N, in_channels=1)

    path_comp = "saved_models/lora_comp_cnn_v2.pth"
    path_mag = "saved_models/lora_mag_cnn_v2.pth"

    # 학습 시 사용할 impaired 환경
    train_config = {"use_cfo": True, "max_cfo_bins": 0.35, "use_multipath": True}
    total_samples = 40000

    # 1. Complex CNN 훈련 또는 로드
    if os.path.exists(path_comp):
        # 이미 학습된 가중치가 있으면 재사용
        model_comp.load_state_dict(torch.load(path_comp, map_location=torch.device("cpu")))
    else:
        print(">> [학습 1/2] Complex CNN 훈련 중...")
        ds_comp = LoRaResearchDataset(
            sim,
            total_samples,
            (-20, 0),
            train_config,
            feature_type="complex",
        )
        dl_train, dl_val = random_split(
            ds_comp,
            [int(0.85 * total_samples), total_samples - int(0.85 * total_samples)],
        )
        model_comp = train_research_model(
            model_comp,
            DataLoader(dl_train, batch_size=256, shuffle=True),
            DataLoader(dl_val, batch_size=256),
            num_epochs=25,
        )
        torch.save(model_comp.state_dict(), path_comp)

    # 2. Mag CNN 훈련 또는 로드
    if os.path.exists(path_mag):
        model_mag.load_state_dict(torch.load(path_mag, map_location=torch.device("cpu")))
    else:
        print("\n>> [학습 2/2] Mag CNN (Ablation) 훈련 중...")
        ds_mag = LoRaResearchDataset(
            sim,
            total_samples,
            (-20, 0),
            train_config,
            feature_type="mag",
        )
        dl_train, dl_val = random_split(
            ds_mag,
            [int(0.85 * total_samples), total_samples - int(0.85 * total_samples)],
        )
        model_mag = train_research_model(
            model_mag,
            DataLoader(dl_train, batch_size=256, shuffle=True),
            DataLoader(dl_val, batch_size=256),
            num_epochs=25,
        )
        torch.save(model_mag.state_dict(), path_mag)

    # 평가할 SNR 점들
    test_snrs = list(range(-25, 1, 2))

    # Scenario A: Pure AWGN
    evaluate_ablation_model(
        model_comp,
        model_mag,
        sim,
        test_snrs,
        {"use_cfo": False, "use_multipath": False},
        "Scenario A - Pure AWGN",
    )

    # Scenario B: Seen Impaired
    evaluate_ablation_model(
        model_comp,
        model_mag,
        sim,
        test_snrs,
        train_config,
        "Scenario B - Seen Impaired",
    )

    # Scenario C: Unseen Impaired
    config_unseen = {
        "use_cfo": True,
        "max_cfo_bins": 0.45,          # CFO 범위 증가
        "use_multipath": True,
        "multipath_taps": [0.9, -0.6j, 0.4],  # 학습 때와 다른 tap 패턴
        "multipath_delays": [0, 4, 9],        # 더 긴 delay
    }
    evaluate_ablation_model(
        model_comp,
        model_mag,
        sim,
        test_snrs,
        config_unseen,
        "Scenario C - Unseen Impaired",
    )

    print("\n========== [완료] ==========")


if __name__ == "__main__":
    main()
