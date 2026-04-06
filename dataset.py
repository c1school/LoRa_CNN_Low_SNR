import torch
from torch.utils.data import Dataset, TensorDataset
from config import CFG


class OnlineParametersDataset(Dataset):
    """
    학습용 파라미터 데이터셋을 정의한 클래스이다.

    이 클래스는 실제 수신 파형을 저장하지 않는다.
    대신 각 샘플마다 다음 세 가지를 즉석에서 만든다.
    1) 정답 심볼(label)
    2) SNR 값
    3) CFO 값

    즉, 이 데이터셋은 "파형 그 자체"를 담는 그릇이 아니라,
    "시뮬레이터가 파형을 만들기 위한 조건표" 역할을 한다.
    그래서 디스크에 거대한 IQ 파일을 저장하지 않고도
    매 배치마다 새로운 신호를 온라인으로 생성할 수 있게 한다.
    """

    def __init__(self, M: int, num_samples: int, snr_range: tuple, max_cfo_bins: float, bw: float):
        # M은 전체 심볼 개수이다.
        # LoRa에서 M은 보통 2 ** sf 형태로 계산된다.
        self.M = M

        # num_samples는 이 데이터셋이 몇 개의 샘플을 제공할지를 의미한다.
        self.num_samples = num_samples

        # snr_range는 SNR을 어느 범위에서 뽑을지 정한다.
        # 예를 들어 (-20, 0)이면 -20 dB부터 0 dB 사이에서 무작위로 선택한다.
        self.snr_range = snr_range

        # max_cfo_bins는 bin 단위의 CFO 한계값이다.
        # 실제 시뮬레이터는 Hz 단위를 사용하므로 bw / M을 곱해 Hz로 변환하였다.
        self.max_cfo_hz = max_cfo_bins * (bw / M)

    def __len__(self):
        # 데이터셋의 전체 길이를 반환한다.
        return self.num_samples

    def __getitem__(self, idx):
        # label은 0부터 M-1 사이의 정수 중 하나로 무작위 생성한다.
        # 이것이 현재 샘플의 정답 심볼이 된다.
        label = torch.randint(0, self.M, (1,)).item()

        # snr은 지정한 범위 안에서 연속값으로 무작위 생성한다.
        snr = torch.empty(1).uniform_(self.snr_range[0], self.snr_range[1]).item()

        # cfo는 -max_cfo_hz부터 +max_cfo_hz 사이에서 무작위 생성한다.
        cfo = torch.empty(1).uniform_(-self.max_cfo_hz, self.max_cfo_hz).item()

        # 학습 루프에서 바로 사용할 수 있도록 torch.tensor 형태로 반환한다.
        return (
            torch.tensor(label, dtype=torch.long),
            torch.tensor(snr, dtype=torch.float32),
            torch.tensor(cfo, dtype=torch.float32),
        )



def create_fixed_feature_dataset(simulator, num_samples, snr_range, max_cfo_bins, seed=None):
    """
    검증용 고정 특징 데이터셋을 생성하는 함수이다.

    이 함수는 매 epoch마다 검증용 데이터를 새로 만들지 않고,
    한 번 고정된 validation feature bank를 생성하여 저장한다.
    이렇게 해야 검증 손실과 검증 정확도를 epoch마다 공정하게 비교할 수 있다.

    반환 형식은 TensorDataset(labels, features)이다.
    여기서 features는 이미 다중 가설 특징맵으로 변환된 상태이므로,
    검증 단계에서는 시뮬레이터를 다시 돌릴 필요 없이 바로 모델에 넣으면 된다.
    """

    # simulator가 사용하는 장치를 그대로 따른다.
    device = simulator.device

    # 로컬 난수 생성기를 만든다.
    # 이 생성기는 고정 데이터셋을 만들 때만 사용하며,
    # 전역 난수 상태를 오염시키지 않도록 분리하였다.
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    # CFO 한계를 Hz 단위로 변환한다.
    max_cfo_hz = max_cfo_bins * (simulator.bw / simulator.M)

    # 다중 가설 특징맵을 만들기 위한 CFO / Timing 가설 격자를 생성한다.
    cfo_grid, to_grid = simulator.generate_hypothesis_grid(
        max_cfo_hz,
        CFG["max_to_samples"],
        CFG["cfo_steps"],
        CFG["to_steps"],
    )

    # 검증용 label, SNR, CFO를 로컬 generator로 고정 생성한다.
    labels = torch.randint(0, simulator.M, (num_samples,), generator=gen, device=device)
    snrs = torch.rand(num_samples, generator=gen, device=device) * (snr_range[1] - snr_range[0]) + snr_range[0]
    cfos = torch.rand(num_samples, generator=gen, device=device) * (2 * max_cfo_hz) - max_cfo_hz

    # 메모리에 순차적으로 쌓기 위한 리스트이다.
    features_list = []

    print(f"\n>> 검증용 2차원 특징 데이터셋 생성 중. (Seed: {seed})")

    with torch.no_grad():
        # 한 번에 너무 많은 샘플을 만들면 GPU 메모리가 부족할 수 있으므로
        # 적당한 크기로 나누어 생성하였다.
        batch_size = 500

        for i in range(0, num_samples, batch_size):
            end = min(i + batch_size, num_samples)

            # 현재 구간에 해당하는 파형을 시뮬레이터로 생성한다.
            # generator=gen을 넘겨 multipath와 noise도 고정된 난수 흐름을 따르게 하였다.
            rx = simulator.generate_batch(
                labels[i:end],
                snrs[i:end],
                cfos[i:end],
                use_multipath=True,
                generator=gen,
            )

            # 생성된 파형을 다중 가설 특징맵으로 변환한다.
            feat = simulator.extract_multi_hypothesis_bank(
                rx,
                cfo_grid,
                to_grid,
                CFG["patch_size"],
            )

            # CPU 메모리로 옮겨 저장한다.
            # 이렇게 해야 GPU 메모리를 오래 점유하지 않는다.
            features_list.append(feat.cpu())

    # label과 feature를 묶어 검증용 TensorDataset을 만든다.
    return TensorDataset(labels.cpu(), torch.cat(features_list))



def create_fixed_waveform_dataset(simulator, num_samples_per_snr, snr_list, max_cfo_hz, seed=None):
    """
    보정(calibration) 및 최종 평가(test)용 고정 파형 데이터셋을 생성하는 함수이다.

    이 함수는 feature가 아니라 rx_signals 자체를 저장한다.
    이유는 calibration/test 단계에서 baseline_grouped_bin과
    extract_multi_hypothesis_bank를 모두 다시 계산해야 하기 때문이다.

    반환 형식은 다음과 같은 딕셔너리이다.
    {
        snr1: TensorDataset(labels, rx_signals),
        snr2: TensorDataset(labels, rx_signals),
        ...
    }

    즉, SNR마다 별도의 고정 평가 세트를 만든다.
    """

    device = simulator.device

    # 로컬 난수 생성기를 만든다.
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    dataset_dict = {}

    # 각 SNR마다 독립적인 고정 데이터셋을 생성한다.
    for snr in snr_list:
        # 해당 SNR에서 사용할 label을 먼저 고정 생성한다.
        labels = torch.randint(0, simulator.M, (num_samples_per_snr,), generator=gen, device=device)

        # 현재 SNR은 고정값으로 채운다.
        snrs = torch.full((num_samples_per_snr,), snr, device=device)

        # CFO는 지정한 범위 안에서 무작위 생성한다.
        cfos = torch.rand(num_samples_per_snr, generator=gen, device=device) * (2 * max_cfo_hz) - max_cfo_hz

        rx_list = []

        with torch.no_grad():
            # 파형도 메모리와 GPU 부담을 고려해 나누어 생성한다.
            for i in range(0, num_samples_per_snr, 2000):
                end = min(i + 2000, num_samples_per_snr)

                rx = simulator.generate_batch(
                    labels[i:end],
                    snrs[i:end],
                    cfos[i:end],
                    use_multipath=True,
                    generator=gen,
                )

                rx_list.append(rx.cpu())

        # SNR별로 label과 rx_signals를 TensorDataset으로 저장한다.
        dataset_dict[snr] = TensorDataset(labels.cpu(), torch.cat(rx_list))

    return dataset_dict
