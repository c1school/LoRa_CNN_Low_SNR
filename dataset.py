import numpy as np
import torch
from torch.utils.data import Dataset


class LoRaResearchDataset(Dataset):
    """
    feature_type에 따라
      - complex 입력 (2채널)
      - mag 입력 (1채널)
    을 생성할 수 있도록 만든 데이터셋.

    같은 채널 환경에서 입력 표현만 다르게 하여
    complex CNN vs magnitude-only CNN을 공정하게 비교하기 위한 용도임.
    """

    def __init__(
        self,
        simulator,
        num_samples: int,
        snr_range,
        impairment_config: dict,
        mode: str = "train",
        feature_type: str = "complex",
    ):
        self.simulator = simulator
        self.num_samples = num_samples
        self.feature_type = feature_type

        print(f"[{mode.upper()} | {feature_type.upper()}] {num_samples}개의 데이터 생성 중...")
        self.data_x = []
        self.data_y = []

        for i in range(num_samples):
            # eval 모드에서는 내부 seed를 고정해 평가 표본을 재현 가능하게 함
            if mode == "eval":
                np.random.seed(2026 + i)

            # 랜덤 label 선택
            label = np.random.randint(0, self.simulator.M)

            # clean signal 생성
            clean_sig = self.simulator.generate_symbol(label)

            # snr_range가 tuple이면 구간 내 랜덤 SNR, 아니면 고정 SNR
            target_snr = (
                np.random.uniform(snr_range[0], snr_range[1])
                if isinstance(snr_range, tuple)
                else snr_range
            )

            # 채널 왜곡 적용
            noisy_sig = self.simulator.apply_impaired_channel(clean_sig, target_snr, impairment_config)

            # feature_type에 따라 특징 추출 방식 선택
            if self.feature_type == "complex":
                features = self.simulator.dechirp_and_fft_complex(noisy_sig)
            else:
                features = self.simulator.dechirp_and_fft_mag(noisy_sig)

            self.data_x.append(torch.tensor(features, dtype=torch.float32))
            self.data_y.append(torch.tensor(label, dtype=torch.long))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]
