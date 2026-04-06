import torch
import torch.nn.functional as F
import numpy as np


class GPUOnlineSimulator:
    """
    LoRa 송수신 파형을 GPU에서 직접 생성하고,
    classical receiver와 neural receiver가 사용할 특징을 뽑아내는 시뮬레이터이다.

    이 클래스는 프로젝트 전체에서 매우 중요한 역할을 한다.
    크게 다음 기능을 담당한다.
    1) 정답 심볼로부터 송신 파형 생성
    2) multipath, CFO, AWGN 적용
    3) dechirp + FFT 기반 baseline 특징 추출
    4) 다중 가설 2D 특징맵 생성
    """

    def __init__(self, sf: int = 7, bw: float = 125e3, fs: float = 1e6, device: str = "cuda"):
        # spreading factor를 저장한다.
        self.sf = sf

        # 대역폭과 샘플링 주파수를 저장한다.
        self.bw = bw
        self.fs = fs

        # 전체 심볼 개수 M = 2 ** sf 이다.
        self.M = 2 ** sf

        # 심볼 길이 Ts를 계산한다.
        self.Ts = self.M / bw

        # 한 심볼을 샘플링한 총 샘플 수 N을 계산한다.
        self.N = int(self.Ts * fs)

        # oversampling ratio이다.
        # FFT 전체 길이 N 안에서 실제 LoRa 심볼 bin이 얼마나 벌어져 있는지를 나타낸다.
        self.osr = self.N // self.M

        # 사용 가능한 장치를 결정한다.
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 시간축 샘플을 미리 만든다.
        t = torch.arange(self.N, device=self.device, dtype=torch.float32) / self.fs

        # LoRa chirp의 기준 위상을 계산한다.
        self.base_phase = torch.pi * (self.bw ** 2 / self.M) * (t ** 2)

        # downchirp를 미리 만들어 둔다.
        # 수신 신호에 이것을 곱하면 dechirp가 된다.
        self.downchirp = torch.exp(-1j * self.base_phase)

        # 샘플 인덱스도 미리 저장한다.
        self.n_idx = torch.arange(self.N, device=self.device, dtype=torch.float32)

    def generate_batch(
        self,
        labels: torch.Tensor,
        snrs_db: torch.Tensor,
        cfos_hz: torch.Tensor,
        use_multipath: bool = True,
        generator=None,
    ) -> torch.Tensor:
        """
        주어진 label, SNR, CFO 조건에 따라 수신 파형 배치를 생성하는 함수이다.

        처리 순서는 다음과 같다.
        1) label에 대응하는 이상적인 LoRa 송신 심볼을 만든다.
        2) multipath가 활성화되어 있으면 echo를 더한다.
        3) CFO를 적용한다.
        4) 원하는 SNR에 맞는 복소 AWGN을 더한다.

        generator를 받도록 만든 이유는,
        고정 데이터셋 생성 시 난수 흐름을 완전히 통제하기 위함이다.
        """

        batch_size = labels.size(0)

        # ------------------------------------------------------------
        # 1. 이상적인 송신 심볼 생성
        # ------------------------------------------------------------
        # 각 label에 대해 해당 심볼 bin에 맞는 tone을 만든다.
        tone_freq = (labels * self.osr).unsqueeze(1) * self.n_idx.unsqueeze(0) / self.N
        tone = torch.exp(1j * 2 * torch.pi * tone_freq)

        # 기준 chirp에 tone을 곱하여 송신 심볼을 만든다.
        tx_signals = torch.exp(1j * self.base_phase.unsqueeze(0)) * tone

        # ------------------------------------------------------------
        # 2. multipath 적용
        # ------------------------------------------------------------
        if use_multipath:
            impaired_signals = tx_signals.clone()

            # 두 개의 반사 경로 세기를 무작위로 정한다.
            att1 = torch.empty(batch_size, 1, device=self.device).uniform_(0.3, 0.6, generator=generator)
            att2 = torch.empty(batch_size, 1, device=self.device).uniform_(0.1, 0.3, generator=generator)

            # 첫 번째 echo는 3샘플 지연 후 허수축 회전을 포함해 더한다.
            echo1 = torch.zeros_like(tx_signals)
            echo1[:, 3:] = tx_signals[:, :-3] * (att1 * 1j)

            # 두 번째 echo는 7샘플 지연 후 실수 계수로 더한다.
            echo2 = torch.zeros_like(tx_signals)
            echo2[:, 7:] = tx_signals[:, :-7] * att2

            impaired_signals = impaired_signals + echo1 + echo2
        else:
            impaired_signals = tx_signals

        # ------------------------------------------------------------
        # 3. CFO 적용
        # ------------------------------------------------------------
        t_matrix = self.n_idx.unsqueeze(0).repeat(batch_size, 1) / self.fs
        cfo_phase = 2 * torch.pi * cfos_hz.unsqueeze(1) * t_matrix
        impaired_signals = impaired_signals * torch.exp(1j * cfo_phase)

        # ------------------------------------------------------------
        # 4. AWGN 적용
        # ------------------------------------------------------------
        # 현재 신호 전력을 계산한다.
        signal_power = torch.mean(torch.abs(impaired_signals) ** 2, dim=1, keepdim=True)

        # dB 단위 SNR을 선형 스케일로 변환한다.
        snr_linear = 10 ** (snrs_db.unsqueeze(1) / 10)

        # noise power를 계산한다.
        noise_power = signal_power / snr_linear

        # 복소 가우시안 잡음을 만든다.
        noise_real = torch.randn(impaired_signals.shape, dtype=torch.float32, device=self.device, generator=generator)
        noise_imag = torch.randn(impaired_signals.shape, dtype=torch.float32, device=self.device, generator=generator)
        noise = torch.sqrt(noise_power / 2) * (noise_real + 1j * noise_imag)

        return impaired_signals + noise

    def extract_features(self, rx_signals: torch.Tensor) -> torch.Tensor:
        """
        단일 가설 dechirp + FFT 특징을 추출하는 함수이다.

        이 함수는 초기 단일 FFT 기반 모델이나 비교 실험에 사용된다.
        처리 결과는 [Batch, 2, N] 형태이며,
        2는 실수부와 허수부를 의미한다.
        """

        dechirped = rx_signals * self.downchirp.unsqueeze(0)
        fft_complex = torch.fft.fft(dechirped, dim=1)

        # 전체 스펙트럼을 샘플별 최대값으로 정규화한다.
        max_vals = torch.max(torch.abs(fft_complex), dim=1, keepdim=True).values + 1e-10
        fft_norm = fft_complex / max_vals

        return torch.stack((torch.real(fft_norm), torch.imag(fft_norm)), dim=1)

    def baseline_grouped_bin(self, rx_signals: torch.Tensor, window_size: int = 2):
        """
        classical grouped-bin baseline를 계산하는 함수이다.

        각 심볼 bin 중심 주변의 작은 구간 에너지를 합산하여
        해당 심볼의 점수로 사용한다.
        이렇게 하면 단일 bin peak보다 약간 더 robust하게 동작할 수 있다.
        """

        dechirped = rx_signals * self.downchirp.unsqueeze(0)
        fft_mag_sq = torch.abs(torch.fft.fft(dechirped, dim=1)) ** 2
        batch_size = rx_signals.size(0)

        grouped_energy = torch.zeros(batch_size, self.M, device=self.device)

        for k in range(self.M):
            # k번째 심볼 중심 bin 위치를 계산한다.
            center = int(np.round(k * self.osr))

            # 중심 주변 window_size만큼의 인덱스를 만든다.
            indices = torch.arange(center - window_size, center + window_size + 1, device=self.device) % self.N

            # 해당 구간 에너지를 모두 더한다.
            grouped_energy[:, k] = torch.sum(fft_mag_sq[:, indices], dim=1)

        return grouped_energy, fft_mag_sq

    def generate_hypothesis_grid(self, max_cfo_hz: float, max_to_samples: int, cfo_steps: int, to_steps: int):
        """
        CFO와 Timing Offset에 대한 가설 격자를 생성하는 함수이다.

        예를 들어 cfo_steps=17, to_steps=9이면
        총 17 * 9 = 153개의 가설 조합이 만들어진다.
        """

        cfo_grid = torch.linspace(-max_cfo_hz, max_cfo_hz, cfo_steps, device=self.device)
        to_grid = torch.linspace(-max_to_samples, max_to_samples, to_steps, device=self.device).long()
        return cfo_grid, to_grid

    def extract_multi_hypothesis_bank(self, rx_signals: torch.Tensor, cfo_grid: torch.Tensor, to_grid: torch.Tensor, patch_size: int) -> torch.Tensor:
        """
        다중 가설 2차원 특징맵을 생성하는 함수이다.

        처리 개념은 다음과 같다.
        1) timing offset 가설마다 수신 신호를 시프트한다.
        2) 각 timing 가설 안에서 CFO 가설마다 보정을 적용한다.
        3) 각 조합마다 dechirp + FFT를 수행한다.
        4) 각 심볼 중심 주변 patch를 추출한다.
        5) 모든 가설 결과를 쌓아 2차원 특징맵으로 만든다.

        최종 출력 형태는 대략 [Batch, 2, Num_Hypotheses, M * patch_size]이다.
        """

        batch_size = rx_signals.size(0)
        cfo_steps = len(cfo_grid)
        to_steps = len(to_grid)

        bank_list = []

        # 시간축 행렬이다.
        t_matrix = self.n_idx.unsqueeze(0).unsqueeze(0) / self.fs

        # 각 CFO 가설에 대해 보정 위상을 미리 계산한다.
        cfo_phase = -2 * torch.pi * cfo_grid.unsqueeze(0).unsqueeze(2) * t_matrix
        cfo_correction = torch.exp(1j * cfo_phase)

        # ------------------------------------------------------------
        # patch 인덱스 계산
        # ------------------------------------------------------------
        # 각 심볼 중심 bin 주변에서 patch_size만큼의 주파수 bin을 함께 가져오기 위함이다.
        half_patch = patch_size // 2
        offsets = torch.arange(-half_patch, half_patch + 1, device=self.device)
        base_indices = (torch.arange(self.M, device=self.device) * self.osr).long()

        patch_indices = (base_indices.unsqueeze(1) + offsets.unsqueeze(0)) % self.N
        patch_indices = patch_indices.view(-1)

        # ------------------------------------------------------------
        # timing 가설 반복
        # ------------------------------------------------------------
        for to in to_grid:
            # timing offset 보상을 위해 수신 신호를 순환 시프트한다.
            shifted_rx = torch.roll(rx_signals, shifts=to.item(), dims=1)

            # 모든 CFO 가설을 한 번에 적용한다.
            corrected_rx = shifted_rx.unsqueeze(1) * cfo_correction

            # dechirp를 수행한다.
            dechirped = corrected_rx * self.downchirp.unsqueeze(0).unsqueeze(0)

            # FFT를 수행한다.
            fft_complex = torch.fft.fft(dechirped, dim=2)

            # 각 심볼 중심 주변 patch만 추출한다.
            fft_m_patch = fft_complex[:, :, patch_indices]
            bank_list.append(fft_m_patch)

        # timing 가설별 결과를 하나의 텐서로 쌓는다.
        bank_tensor = torch.stack(bank_list, dim=1)

        # [Batch, to_steps, cfo_steps, M * patch] 형태를
        # [Batch, Num_Hypotheses, M * patch] 형태로 펼친다.
        num_hypotheses = to_steps * cfo_steps
        bank_tensor = bank_tensor.view(batch_size, num_hypotheses, self.M * patch_size)

        # 각 가설별 최대 크기로 정규화한다.
        max_vals = torch.max(torch.abs(bank_tensor), dim=2, keepdim=True).values + 1e-10
        bank_norm = bank_tensor / max_vals

        # 실수부와 허수부를 2채널로 분리하여 반환한다.
        features = torch.stack((torch.real(bank_norm), torch.imag(bank_norm)), dim=1)
        return features
