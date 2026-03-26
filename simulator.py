import numpy as np
from scipy.signal import lfilter


class LoRaResearchSimulator:
    """
    LoRa-like 심볼 생성, 채널 왜곡 적용, dechirp+FFT 특징 추출,
    classical baseline 복조를 담당하는 핵심 시뮬레이터.

    이 클래스 하나가 송신기 + 채널 + 수신기 전처리 역할을 동시에 수행함.
    """

    def __init__(self, sf: int = 7, bw: float = 125e3, fs: float = 1e6):
        self.sf = sf
        self.bw = bw
        self.fs = fs

        # LoRa 심볼 개수 M = 2^SF
        self.M = 2 ** sf

        # 심볼 길이 Ts = M / BW
        self.Ts = self.M / bw

        # 한 심볼을 샘플링한 총 샘플 수
        self.N = int(self.Ts * fs)

        # Oversampling ratio
        # dechirp 후 FFT peak는 symbol 자체가 아니라 symbol * osr 부근에 나타남.
        self.osr = self.N // self.M

        # base chirp 생성
        t = np.arange(self.N) / self.fs
        self.base_phase = np.pi * (self.bw**2 / self.M) * (t**2)

        # dechirp에 사용할 기준 downchirp
        self.downchirp = np.exp(-1j * self.base_phase)

    def generate_symbol(self, symbol: int) -> np.ndarray:
        """
        특정 symbol index에 대응되는 송신 심볼 생성.

        기본 chirp 위에 symbol * osr 위치에 대응되는 tone을 얹어,
        dechirp 후 FFT peak가 정해진 위치에 나타나게 설계함.
        """
        n = np.arange(self.N)
        tone = np.exp(1j * 2 * np.pi * (symbol * self.osr) * n / self.N)
        return np.exp(1j * self.base_phase) * tone

    def apply_impaired_channel(self, signal: np.ndarray, snr_db: float, impairment_config: dict) -> np.ndarray:
        """
        입력 clean signal에 채널 왜곡을 순서대로 적용함.

        1) Multipath
        2) CFO
        3) AWGN
        """
        impaired_signal = np.copy(signal)

        # Multipath 적용
        if impairment_config.get("use_multipath", False):
            taps = impairment_config.get("multipath_taps", [1.0, 0.4j, 0.2])
            delays = impairment_config.get("multipath_delays", [0, 2, 5])
            h = np.zeros(max(delays) + 1, dtype=np.complex128)
            h[delays] = taps
            impaired_signal = lfilter(h, 1, impaired_signal)

        # CFO 적용
        if impairment_config.get("use_cfo", False):
            bin_spacing = self.bw / self.M
            max_cfo_bins = impairment_config.get("max_cfo_bins", 0.35)
            cfo_hz = np.random.uniform(-max_cfo_bins, max_cfo_bins) * bin_spacing
            cfo_phase = 2 * np.pi * cfo_hz * (np.arange(self.N) / self.fs)
            impaired_signal = impaired_signal * np.exp(1j * cfo_phase)

        # AWGN 적용
        signal_power = np.mean(np.abs(impaired_signal) ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(self.N) + 1j * np.random.randn(self.N)
        )

        return impaired_signal + noise

    def dechirp_and_fft_complex(self, iq_signal: np.ndarray) -> np.ndarray:
        """
        dechirp 후 FFT를 수행하고, 그 복소수 결과를 [Real, Imag] 2채널로 반환함.
        amplitude뿐 아니라 phase 정보도 보존하는 특징 표현임.
        """
        dechirped = iq_signal * self.downchirp
        fft_complex = np.fft.fft(dechirped)
        max_val = np.max(np.abs(fft_complex)) + 1e-10
        fft_norm = fft_complex / max_val
        return np.vstack((np.real(fft_norm), np.imag(fft_norm)))

    def dechirp_and_fft_mag(self, iq_signal: np.ndarray) -> np.ndarray:
        """
        dechirp 후 FFT magnitude만 남기고 위상은 버린 특징 표현.
        ablation study에서 위상 정보의 기여를 비교하기 위한 입력임.
        """
        dechirped = iq_signal * self.downchirp
        fft_mag = np.abs(np.fft.fft(dechirped))
        max_val = np.max(fft_mag) + 1e-10
        fft_norm = fft_mag / max_val
        return np.expand_dims(fft_norm, axis=0)

    def baseline_demod_naive(self, rx_signal: np.ndarray) -> int:
        """
        가장 큰 FFT peak 하나만 찾고,
        그 위치를 osr로 나누어 symbol로 환산하는 고전적 복조 방식.
        """
        dechirped = rx_signal * self.downchirp
        fft_mag = np.abs(np.fft.fft(dechirped))
        peak_idx = int(np.argmax(fft_mag))
        return int(np.round(peak_idx / self.osr)) % self.M

    def baseline_demod_grouped_bin(self, rx_signal: np.ndarray, window_size: int = 2) -> int:
        """
        CFO나 spectral leakage로 인해 peak가 주변 bin으로 퍼질 수 있으므로,
        중심 bin 주변 에너지를 합산하여 판정하는 classical detector.
        """
        dechirped = rx_signal * self.downchirp
        fft_mag_sq = np.abs(np.fft.fft(dechirped)) ** 2

        grouped_energy = np.zeros(self.M)
        for k in range(self.M):
            center = int(np.round(k * self.osr))
            indices = np.mod(np.arange(center - window_size, center + window_size + 1), self.N)
            grouped_energy[k] = np.sum(fft_mag_sq[indices])

        return int(np.argmax(grouped_energy))
