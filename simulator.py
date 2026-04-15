"""LoRa 심볼 합성, 채널 impairment 주입, 복조용 feature 추출을 담당하는 파일이다.

이 파일은 실험 전체에서 가장 핵심적인 역할을 한다.

- SF / BW / sampling rate에 맞는 LoRa upchirp / downchirp를 만든다.
- multipath, timing offset, CFO, phase noise, tone interference, AWGN을 적용해 수신 신호를 합성한다.
- 기본 복조기(dechirp + FFT)와 다중 가설 feature bank를 계산한다.

즉, 학습과 평가가 사용하는 모든 synthetic IQ 신호는 이 파일을 통해 생성된다.
"""

from typing import Dict, Optional, Tuple

import torch


class GPUOnlineSimulator:
    """GPU 상에서 LoRa 신호 생성과 복조 전처리를 수행하는 시뮬레이터다.

    이 클래스 하나가 담당하는 역할은 크게 세 가지다.

    1. LoRa 물리 파라미터(SF, BW, sampling rate)로부터 기준 upchirp/downchirp를 만든다.
    2. clean LoRa 심볼에 여러 impairment를 적용해 synthetic 수신 신호를 만든다.
    3. 수신 신호를 classical 복조 score 또는 CNN 입력 feature bank로 변환한다.

    따라서 이 클래스의 함수들을 보면
    "실험에서 어떤 채널을 만들고, 그 신호를 어떤 형태로 복조기에 넣는가"가 거의 다 드러난다.
    """

    def __init__(self, sf: int = 7, bw: float = 125e3, fs: float = 1e6, device: str = "cuda"):
        # LoRa 기본 파라미터:
        # - sf: spreading factor
        # - bw: 대역폭 [Hz]
        # - fs: sampling rate [Hz]
        #
        # LoRa에서 한 심볼의 후보 개수는 2^SF개이므로 M = 2^SF로 둔다.
        # Ts는 심볼 시간이고, N은 그 심볼 하나를 몇 개 샘플로 표현할지를 뜻한다.
        #
        # 정리하면:
        # - M: 심볼 후보 수
        # - Ts: 심볼 지속 시간
        # - N: 심볼당 샘플 수
        # - osr: 한 LoRa bin이 샘플 축에서 차지하는 폭
        self.sf = sf
        self.bw = bw
        self.fs = fs
        self.M = 2 ** sf
        self.Ts = self.M / bw
        self.N = int(self.Ts * fs)
        self.osr = self.N // self.M
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # sample_times:
        # [0, 1/fs, 2/fs, ...] 형태의 시간축이다.
        # CFO 위상 회전, tone interference 위상 등을 만들 때 사용한다.
        self.sample_times = torch.arange(self.N, device=self.device, dtype=torch.float32) / self.fs

        # sample_indices:
        # timing shift를 인덱스 이동으로 구현할 때 사용하는 정수 샘플 축이다.
        self.sample_indices = torch.arange(self.N, device=self.device, dtype=torch.long)

        # base_phase:
        # LoRa chirp의 기준 위상 항이다.
        # 여기서는 복잡한 유도식 전체를 코드에 쓰기보다,
        # "시간에 따라 주파수가 선형으로 변하는 chirp"를 만들기 위한 위상 누적 항으로 이해하면 된다.
        self.base_phase = torch.pi * (self.bw ** 2 / self.M) * (self.sample_times ** 2)

        # upchirp / downchirp:
        # LoRa 복조는 수신 신호에 downchirp를 곱해 dechirp한 뒤 FFT를 보는 구조이므로,
        # 기준이 되는 upchirp와 downchirp를 미리 만들어 캐시해 둔다.
        self.upchirp = torch.exp(1j * self.base_phase)
        self.downchirp = torch.exp(-1j * self.base_phase)

        # grouped-bin / patch 추출에 필요한 인덱스는 반복해서 쓰이므로 캐시한다.
        self._group_index_cache = {}
        self._patch_index_cache = {}

    def resolve_channel_profile(self, profile: Dict) -> Dict:
        """심볼 길이 비율 기반 채널 설정을 현재 프로파일의 샘플 수 기준 값으로 변환한다.

        설정 파일에서는 profile마다
        - 절대 샘플 수 기준 설정
        - 심볼 길이 대비 비율 기준 설정
        이 둘을 섞어 쓸 수 있다.

        예를 들어 `max_to_symbol_fraction = 0.004`이면
        현재 심볼 길이 N에 맞춰 실제 샘플 수로 다시 바꿔 줘야 한다.

        이 함수를 거치면 이후 로직은 모두 "현재 프로파일에서 실제 몇 샘플인가" 기준으로 계산할 수 있다.
        """

        resolved = dict(profile)

        if "max_to_symbol_fraction" in resolved:
            # 최대 timing offset을 "심볼 길이의 몇 %" 형태로 줬다면,
            # 현재 프로파일의 샘플 수 N 기준 정수 샘플 수로 바꾼다.
            resolved["max_to_samples"] = max(
                0,
                int(round(self.N * float(resolved["max_to_symbol_fraction"]))),
            )

        if "max_delay_symbol_fraction" in resolved:
            # 최대 delay spread도 같은 방식으로 샘플 수 기준으로 변환한다.
            resolved["max_delay_samples"] = max(
                0,
                int(round(self.N * float(resolved["max_delay_symbol_fraction"]))),
            )

        # fractional timing offset을 아예 지정하지 않은 profile도 있으므로 기본값 0을 넣어 둔다.
        resolved.setdefault("max_fractional_to_samples", 0.0)
        return resolved

    def _sample_uniform(self, value_range: Tuple[float, float], batch_size: int, generator=None) -> torch.Tensor:
        """지정된 구간에서 균등분포 샘플을 생성한다."""

        low, high = value_range
        return torch.rand(batch_size, device=self.device, generator=generator) * (high - low) + low

    def _apply_sample_wise_shift(self, signals: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
        """정수 샘플 단위 timing shift를 적용한다.

        LoRa 심볼은 길이가 N인 순환 구조로 다루므로,
        인덱스는 modulo N으로 감아 준다.
        """

        # gather_index의 shape은 [batch, N]이다.
        # 각 샘플마다 "원래 어느 위치의 값을 읽어 올지"를 미리 계산한 뒤 gather로 한 번에 가져온다.
        gather_index = (self.sample_indices.unsqueeze(0) - shifts.long().unsqueeze(1)) % self.N
        return torch.gather(signals, 1, gather_index)

    def _apply_fractional_shift(self, signals: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
        """정수 shift와 선형 보간을 조합해 fractional timing shift를 근사한다.

        예를 들어 shift가 2.3 sample이면
        - 2 sample shift한 신호
        - 3 sample shift한 신호
        를 만들어 두고 둘을 0.7 : 0.3 비율처럼 선형 보간한다.

        엄밀한 bandlimited fractional delay 필터는 아니지만,
        본 실험에서는 residual timing mismatch를 넣는 용도로 충분한 근사다.
        """

        integer_shifts = torch.floor(shifts).long()
        alpha = (shifts - integer_shifts.float()).unsqueeze(1).to(signals.real.dtype)
        shifted_low = self._apply_sample_wise_shift(signals, integer_shifts)
        shifted_high = self._apply_sample_wise_shift(signals, integer_shifts + 1)
        return shifted_low * (1.0 - alpha) + shifted_high * alpha

    def _apply_multipath(
        self,
        tx_signals: torch.Tensor,
        path_delays: torch.Tensor,
        path_gains: torch.Tensor,
    ) -> torch.Tensor:
        """여러 경로의 지연/복소 이득을 합산해 multipath 수신 신호를 만든다.

        입력:
        - tx_signals: [batch, N]
        - path_delays: [batch, num_paths]
        - path_gains: [batch, num_paths]

        각 경로마다
        - 지연(delay)
        - 복소 이득(complex gain)
        을 적용한 뒤 모두 더해 최종 수신 신호를 만든다.
        """

        path_count = path_delays.size(1)
        # 각 경로별로 "지연을 반영했을 때 읽어 올 샘플 위치"를 미리 계산한다.
        shift_indices = (
            self.sample_indices.view(1, 1, -1) - path_delays.long().unsqueeze(-1)
        ) % self.N

        # tx_signals를 [batch, path_count, N]으로 확장한 뒤,
        # 각 경로가 참조해야 할 시간축 위치를 gather로 한 번에 가져온다.
        delayed = torch.gather(
            tx_signals.unsqueeze(1).expand(-1, path_count, -1),
            2,
            shift_indices,
        )

        # 마지막으로 경로별 복소 이득을 곱하고 경로 축으로 합산한다.
        return torch.sum(delayed * path_gains.unsqueeze(-1), dim=1)

    def _get_group_indices(self, window_size: int) -> torch.Tensor:
        """grouped-bin 에너지 계산에 필요한 FFT 인덱스를 캐시해 반환한다.

        기본 LoRa 복조에서는 dechirp 후 FFT를 한 뒤
        특정 bin 주변의 에너지를 묶어서 score로 쓰곤 한다.

        여기서는 각 LoRa bin 중심 주변으로
        `[-window_size, ..., 0, ..., +window_size]`
        범위의 FFT 인덱스를 미리 만들어 둔다.
        """

        if window_size not in self._group_index_cache:
            centers = torch.round(
                torch.arange(self.M, device=self.device, dtype=torch.float32) * self.osr
            ).long()
            offsets = torch.arange(-window_size, window_size + 1, device=self.device)
            self._group_index_cache[window_size] = (
                centers.unsqueeze(1) + offsets.unsqueeze(0)
            ) % self.N
        return self._group_index_cache[window_size]

    def sample_channel_state(self, batch_size: int, profile: Dict, generator=None) -> Dict[str, torch.Tensor]:
        """배치 단위 채널 상태를 샘플링한다.

        여기서 생성되는 값들은 waveform 생성 시 그대로 사용된다.
        따라서 이 함수가 사실상 채널 모델의 랜덤 파라미터를 정의한다고 보면 된다.
        """

        # profile 안의 비율 기반 설정을 현재 심볼 길이에 맞는 샘플 수로 바꾼다.
        profile = self.resolve_channel_profile(profile)
        max_paths = profile["max_paths"]
        max_delay_samples = profile["max_delay_samples"]

        # integer_timing_offsets:
        # 샘플 단위 정수 timing mismatch다.
        integer_timing_offsets = torch.randint(
            -profile["max_to_samples"],
            profile["max_to_samples"] + 1,
            (batch_size,),
            device=self.device,
            generator=generator,
        )

        # fractional_timing_offsets:
        # 정수 shift만으로는 표현되지 않는 sub-sample timing mismatch다.
        max_fractional_to = float(profile.get("max_fractional_to_samples", 0.0))
        if max_fractional_to > 0:
            fractional_timing_offsets = (
                torch.rand(batch_size, device=self.device, generator=generator) * 2.0 - 1.0
            ) * max_fractional_to
        else:
            fractional_timing_offsets = torch.zeros(batch_size, device=self.device)
        timing_offsets = integer_timing_offsets.float() + fractional_timing_offsets
        phase_offsets = torch.rand(batch_size, device=self.device, generator=generator) * (2 * torch.pi)

        # path_delays:
        # 각 경로의 지연 샘플 수다.
        #
        # 첫 번째 경로는 direct path 역할을 하도록 delay 0으로 고정한다.
        path_delays = torch.randint(
            0,
            max_delay_samples + 1,
            (batch_size, max_paths),
            device=self.device,
            generator=generator,
        )
        path_delays[:, 0] = 0
        path_delays, _ = torch.sort(path_delays, dim=1)

        # path_gains:
        # 각 경로의 복소 채널 계수다.
        # 실수부/허수부를 독립 가우시안으로 뽑아 복소 이득을 만든다.
        path_gains = (
            torch.randn((batch_size, max_paths), device=self.device, generator=generator)
            + 1j * torch.randn((batch_size, max_paths), device=self.device, generator=generator)
        )

        # delay가 긴 경로일수록 평균적으로 더 약해지도록 감쇠를 건다.
        delay_decay = torch.exp(-path_delays.float() / max(profile["delay_decay"], 1e-6))
        path_gains = path_gains * delay_decay

        # direct path는 완전히 사라지지 않도록 약간 더 강하게 만든다.
        path_gains[:, 0] = path_gains[:, 0] + 1.5

        # 추가 경로는 확률적으로 꺼서, 패킷마다 실제 경로 수가 달라지게 만든다.
        if max_paths > 1:
            active_mask = (
                torch.rand((batch_size, max_paths - 1), device=self.device, generator=generator)
                < profile["extra_path_prob"]
            )
            path_gains[:, 1:] = path_gains[:, 1:] * active_mask

        # 경로 이득의 총 전력을 1로 정규화해,
        # "경로 수가 많다고 원하는 신호 전력이 무조건 커지는" 현상을 막는다.
        gain_power = torch.sum(torch.abs(path_gains) ** 2, dim=1, keepdim=True).clamp_min(1e-10)
        path_gains = path_gains / torch.sqrt(gain_power)

        # phase noise는 샘플별 위상 랜덤 워크를 만들 때 사용할 표준편차다.
        phase_noise_std = self._sample_uniform(profile["phase_noise_std_range"], batch_size, generator=generator)

        # narrowband interference는 확률적으로만 활성화된다.
        tone_active = (
            torch.rand(batch_size, device=self.device, generator=generator)
            < profile["tone_interference_prob"]
        )
        tone_inr_db = self._sample_uniform(profile["tone_inr_db_range"], batch_size, generator=generator)
        tone_amplitudes = tone_active.float() * torch.sqrt(10 ** (tone_inr_db / 10.0))
        tone_freqs_hz = self._sample_uniform((-0.45 * self.bw, 0.45 * self.bw), batch_size, generator=generator)
        tone_phases = torch.rand(batch_size, device=self.device, generator=generator) * (2 * torch.pi)

        return {
            "timing_offsets": timing_offsets,
            "phase_offsets": phase_offsets,
            "path_delays": path_delays.long(),
            "path_gains": path_gains.to(torch.complex64),
            "phase_noise_std": phase_noise_std,
            "tone_amplitudes": tone_amplitudes,
            "tone_freqs_hz": tone_freqs_hz,
            "tone_phases": tone_phases,
        }

    def repeat_channel_state(self, channel_state: Dict[str, torch.Tensor], repeats: int) -> Dict[str, torch.Tensor]:
        """패킷 단위 채널 상태를 심볼 단위로 반복 확장한다.

        예를 들어 하나의 패킷이 16개 payload symbol로 구성돼 있으면,
        패킷마다 한 번 뽑은 채널 상태를 심볼 16개에 동일하게 복사해 사용한다.
        """

        repeated = {}
        for key, value in channel_state.items():
            repeated[key] = value.repeat_interleave(repeats, dim=0)
        return repeated

    def generate_batch(
        self,
        labels: torch.Tensor,
        snrs_db: torch.Tensor,
        cfos_hz: torch.Tensor,
        channel_state: Optional[Dict[str, torch.Tensor]] = None,
        profile: Optional[Dict] = None,
        generator=None,
    ) -> torch.Tensor:
        """라벨/SNR/CFO/채널 상태를 이용해 복소 수신 신호 배치를 생성한다.

        처리 순서는 대략 다음과 같다.

        1. clean LoRa symbol 생성
        2. multipath 적용
        3. timing shift 적용
        4. carrier phase 및 CFO 적용
        5. phase noise 적용
        6. tone interference 추가
        7. 마지막으로 AWGN 추가
        """

        # 입력은 어떤 shape로 와도 일단 1차원 배치 벡터로 정리한다.
        labels = labels.reshape(-1).to(self.device)
        snrs_db = snrs_db.reshape(-1).to(self.device)
        cfos_hz = cfos_hz.reshape(-1).to(self.device)
        batch_size = labels.numel()

        if snrs_db.numel() != batch_size or cfos_hz.numel() != batch_size:
            raise ValueError("labels, snrs_db, and cfos_hz must have the same number of samples.")

        if channel_state is None:
            if profile is None:
                raise ValueError("Either channel_state or profile must be provided.")
            # profile만 주어지면 여기서 즉석으로 채널 파라미터를 뽑는다.
            channel_state = self.sample_channel_state(batch_size, profile, generator=generator)
        else:
            # 이미 만들어 둔 channel_state를 쓰는 경우에도 batch 크기는 맞아야 한다.
            for key, value in channel_state.items():
                if value.size(0) != batch_size:
                    raise ValueError(f"Channel state `{key}` has batch size {value.size(0)} but expected {batch_size}.")

        # 라벨에 해당하는 baseband tone을 upchirp에 실어 LoRa 심볼을 만든다.
        #
        # labels는 "어느 심볼 bin을 보냈는가"를 뜻한다.
        # 이를 샘플 인덱스 축에서 회전하는 복소 tone으로 만든 뒤 upchirp에 곱하면
        # 해당 LoRa 심볼이 된다.
        tone_freq = (labels.float() * self.osr).unsqueeze(1) * self.sample_indices.float().unsqueeze(0) / self.N
        clean_symbols = self.upchirp.unsqueeze(0) * torch.exp(1j * 2 * torch.pi * tone_freq)

        # 원하는 신호 성분(desired signal)에 채널 왜곡을 차례로 적용한다.
        desired_signals = self._apply_multipath(
            clean_symbols,
            channel_state["path_delays"],
            channel_state["path_gains"],
        )
        desired_signals = self._apply_fractional_shift(desired_signals, channel_state["timing_offsets"])

        # 공통 carrier phase offset을 건다.
        desired_signals = desired_signals * torch.exp(1j * channel_state["phase_offsets"].unsqueeze(1))

        # CFO는 시간에 비례하는 위상 회전으로 구현한다.
        cfo_phase = 2 * torch.pi * cfos_hz.unsqueeze(1) * self.sample_times.unsqueeze(0)
        desired_signals = desired_signals * torch.exp(1j * cfo_phase)

        # phase noise는 샘플별 작은 위상 오차를 누적합 형태로 만들어 적용한다.
        # 누적합을 쓰는 이유는 "각 샘플마다 독립 위상"이 아니라 "조금씩 흔들리는 위상 드리프트"를 흉내 내기 위해서다.
        phase_noise = torch.randn(
            (batch_size, self.N),
            device=self.device,
            generator=generator,
        ) * channel_state["phase_noise_std"].unsqueeze(1)
        desired_signals = desired_signals * torch.exp(1j * torch.cumsum(phase_noise, dim=1))

        rx_signals = desired_signals
        tone_present = channel_state["tone_amplitudes"] > 0
        if torch.any(tone_present):
            # tone interference는 특정 좁은 주파수의 복소 sinusoid로 만든다.
            tone_phase = (
                2 * torch.pi * channel_state["tone_freqs_hz"].unsqueeze(1) * self.sample_times.unsqueeze(0)
                + channel_state["tone_phases"].unsqueeze(1)
            )
            interferer = channel_state["tone_amplitudes"].unsqueeze(1) * torch.exp(1j * tone_phase)
            rx_signals = rx_signals + interferer

        # AWGN 세기는 "원하는 신호 전력" 대비 목표 SNR이 되도록 계산한다.
        #
        # 중요한 점:
        # 여기서 기준으로 삼는 것은 interferer까지 포함한 전체 전력이 아니라
        # desired_signals의 전력이다.
        # 따라서 tone interference는 SNR 정의 밖의 별도 impairment처럼 작동한다.
        signal_power = torch.mean(torch.abs(desired_signals) ** 2, dim=1, keepdim=True).clamp_min(1e-10)
        snr_linear = 10 ** (snrs_db.unsqueeze(1) / 10.0)
        noise_power = signal_power / snr_linear

        noise_real = torch.randn(
            rx_signals.shape,
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )
        noise_imag = torch.randn(
            rx_signals.shape,
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )
        noise = torch.sqrt(noise_power / 2.0) * (noise_real + 1j * noise_imag)
        # 최종 반환값은
        # "desired signal + tone interference + complex AWGN" 형태의 복소 수신 신호다.
        return rx_signals + noise

    def _group_energy_from_fft(self, fft_mag_sq: torch.Tensor, window_size: int) -> torch.Tensor:
        """각 심볼 bin 주변의 에너지를 묶어서 grouped-bin score를 만든다.

        LoRa dechirp + FFT 결과는 이상적으로는 특정 bin 근처에 에너지가 모인다.
        하지만 CFO/TO/multipath가 있으면 에너지가 약간 퍼질 수 있으므로,
        중심 bin 하나만 보기보다 주변 몇 개를 더한 grouped energy를 score로 사용한다.
        """

        group_indices = self._get_group_indices(window_size)
        gathered = torch.gather(
            fft_mag_sq.unsqueeze(1).expand(-1, self.M, -1),
            2,
            group_indices.unsqueeze(0).expand(fft_mag_sq.size(0), -1, -1),
        )
        return gathered.sum(dim=2)

    def baseline_grouped_bin(self, rx_signals: torch.Tensor, window_size: int = 2):
        """기본 LoRa 복조기 경로를 계산한다.

        수신 신호를 downchirp로 dechirp한 뒤 FFT를 수행하고,
        grouped-bin 에너지로 심볼 score를 반환한다.
        """

        rx_signals = rx_signals.to(self.device)

        # dechirp:
        # 수신 신호에 기준 downchirp를 곱해,
        # 원래 chirp 위에 실려 있던 심볼 정보를 "정지한 tone"에 가깝게 만든다.
        dechirped = rx_signals * self.downchirp.unsqueeze(0)

        # FFT를 취하면 심볼에 해당하는 에너지가 특정 bin 근처에 나타난다.
        fft_complex = torch.fft.fft(dechirped, dim=1)
        fft_mag_sq = torch.abs(fft_complex) ** 2

        # grouped energy는 "기본 LoRa score"로 사용된다.
        grouped_energy = self._group_energy_from_fft(fft_mag_sq, window_size=window_size)
        return grouped_energy, fft_mag_sq

    def generate_hypothesis_grid(
        self,
        max_cfo_hz: float,
        max_to_samples: int,
        cfo_steps: int,
        to_steps: int,
    ):
        """CFO / timing offset 가설 grid를 균일 간격으로 생성한다.

        다중 가설 방식의 핵심 아이디어는
        "CFO와 timing offset이 정확히 얼마인지 모르니, 여러 후보를 놓고 다 시험해 보자"다.

        이 함수는 그 후보 목록을 균일 간격 grid 형태로 만든다.
        """

        cfo_grid = torch.linspace(-max_cfo_hz, max_cfo_hz, cfo_steps, device=self.device)
        to_grid = torch.linspace(-max_to_samples, max_to_samples, to_steps, device=self.device).long()
        return cfo_grid, to_grid

    def _build_patch_indices(self, patch_size: int) -> torch.Tensor:
        """각 LoRa bin 주변 patch를 뽑기 위한 FFT 인덱스를 만든다.

        CNN은 "bin 하나의 에너지"만 보는 것이 아니라
        각 bin 주변의 작은 patch까지 함께 보게 된다.
        예를 들어 patch_size=5이면 각 bin 주변 5개 FFT 지점을 본다.
        """

        if patch_size % 2 == 0:
            raise ValueError("patch_size must be odd.")

        if patch_size not in self._patch_index_cache:
            half_patch = patch_size // 2
            offsets = torch.arange(-half_patch, half_patch + 1, device=self.device)
            base_indices = (torch.arange(self.M, device=self.device) * self.osr).long()
            self._patch_index_cache[patch_size] = (
                base_indices.unsqueeze(1) + offsets.unsqueeze(0)
            ) % self.N
        return self._patch_index_cache[patch_size]

    def prepare_hypothesis_helper(
        self,
        cfo_grid: torch.Tensor,
        to_grid: torch.Tensor,
        patch_size: int,
        to_chunk_size: int = 3,
    ):
        """다중 가설 feature 추출에 필요한 보조 텐서를 미리 계산해 묶는다.

        extract_multi_hypothesis_bank를 매번 호출할 때마다
        - CFO 보정용 complex exponential
        - timing shift 인덱스
        - patch 추출 인덱스
        를 다시 만들면 비용이 크다.

        그래서 반복 사용되는 텐서를 미리 계산해 helper dict에 넣어 둔다.
        """

        patch_indices = self._build_patch_indices(patch_size)
        patch_indices_flat = patch_indices.reshape(-1)

        # cfo_correction:
        # 각 CFO 가설에 대해 "이만큼 주파수 오차가 있었다고 가정하면
        # 어떻게 보정할 것인가"를 나타내는 복소 보정 항이다.
        cfo_correction = torch.exp(
            -1j * 2 * torch.pi * cfo_grid.view(-1, 1) * self.sample_times.view(1, -1)
        )

        shift_index_chunks = []
        for start in range(0, len(to_grid), to_chunk_size):
            to_chunk = to_grid[start:start + to_chunk_size].long()

            # timing 가설마다 읽어 올 순환 인덱스를 미리 만든다.
            shift_indices = (
                self.sample_indices.view(1, -1) - to_chunk.view(-1, 1)
            ) % self.N
            shift_index_chunks.append(shift_indices)

        return {
            "cfo_grid": cfo_grid,
            "to_grid": to_grid,
            "patch_size": patch_size,
            "patch_indices_flat": patch_indices_flat,
            "cfo_correction": cfo_correction.to(torch.complex64),
            "shift_index_chunks": shift_index_chunks,
            "num_hypotheses": len(cfo_grid) * len(to_grid),
        }

    def extract_multi_hypothesis_bank(
        self,
        rx_signals: torch.Tensor,
        cfo_grid: torch.Tensor = None,
        to_grid: torch.Tensor = None,
        patch_size: int = None,
        return_energy: bool = False,
        helper: Optional[Dict] = None,
    ):
        """다중 CFO / timing 가설에 대한 feature bank를 추출한다.

        이 함수는 사실상 "CNN용 전처리기"다.

        처리 개념은 다음과 같다.

        1. timing offset 후보마다 수신 신호를 shift한다.
        2. 각 shifted 신호에 대해 여러 CFO 보정 가설을 적용한다.
        3. 각각을 dechirp + FFT 한다.
        4. 각 LoRa bin 주변 patch를 뽑아 펼친다.
        5. 실수부/허수부 2채널 feature로 만든다.

        반환:
        - features: [batch, 2, num_hypotheses, M * patch_size]
        - energy_bank(선택): [batch, num_hypotheses, M]
        """

        rx_signals = rx_signals.to(self.device)
        batch_size = rx_signals.size(0)
        if helper is None:
            helper = self.prepare_hypothesis_helper(cfo_grid, to_grid, patch_size)

        cfo_correction = helper["cfo_correction"]
        patch_indices_flat = helper["patch_indices_flat"]
        patch_size = helper["patch_size"]
        cfo_steps = cfo_correction.size(0)

        normalized_chunks = []
        energy_chunks = [] if return_energy else None
        for shift_indices in helper["shift_index_chunks"]:
            chunk_size = shift_indices.size(0)

            # 1. timing 가설별로 수신 신호를 shift한다.
            shifted = torch.gather(
                rx_signals.unsqueeze(1).expand(-1, chunk_size, -1),
                2,
                shift_indices.unsqueeze(0).expand(batch_size, -1, -1),
            )

            # 2. 각 timing 가설에 대해 여러 CFO 보정 가설을 곱한다.
            corrected = shifted.unsqueeze(2) * cfo_correction.view(1, 1, cfo_correction.size(0), -1)

            # 3. dechirp + FFT
            dechirped = corrected * self.downchirp.view(1, 1, 1, -1)
            fft_complex = torch.fft.fft(dechirped, dim=3)

            # 4. 각 LoRa bin 주변 patch를 모아 CNN이 볼 수 있는 로컬 주파수 패턴으로 만든다.
            fft_patch = fft_complex[..., patch_indices_flat].reshape(
                batch_size,
                chunk_size,
                cfo_steps,
                self.M,
                patch_size,
            )

            # chunk 안의 모든 가설을 [num_hypotheses_chunk, M * patch_size] 형태로 펼친다.
            flattened_chunk = fft_patch.reshape(batch_size, chunk_size * cfo_steps, self.M * patch_size)

            # 가설마다 magnitude 최대값으로 정규화해
            # 전체 진폭 스케일 차이보다 패턴 자체를 더 보게 한다.
            max_vals = torch.max(torch.abs(flattened_chunk), dim=2, keepdim=True).values.clamp_min(1e-10)
            normalized_chunks.append(flattened_chunk / max_vals)

            if return_energy:
                # energy_bank는 classical multi-hypothesis baseline을 만들 때 사용한다.
                energy_chunks.append(
                    torch.sum(torch.abs(fft_patch) ** 2, dim=-1).reshape(batch_size, chunk_size * cfo_steps, self.M)
                )

        normalized_bank = torch.cat(normalized_chunks, dim=1)

        # CNN 입력은 실수 텐서여야 하므로
        # 복소값 feature를 실수부/허수부 2채널로 분리한다.
        features = torch.stack((torch.real(normalized_bank), torch.imag(normalized_bank)), dim=1)

        if return_energy:
            return features, torch.cat(energy_chunks, dim=1)
        return features

    def multi_hypothesis_grouped_bin(
        self,
        rx_signals: torch.Tensor,
        cfo_grid: torch.Tensor = None,
        to_grid: torch.Tensor = None,
        window_size: int = None,
        helper: Optional[Dict] = None,
    ):
        """다중 가설 classical baseline의 score를 계산한다.

        이 경로에는 CNN이 없다.

        아이디어는 단순하다.
        - CFO/TO 가설을 여러 개 본다.
        - 각 가설에서 심볼별 에너지를 계산한다.
        - 가설 축에서 최댓값을 취해 "이 심볼은 어떤 가설에서는 잘 맞는다"를 score로 쓴다.

        따라서 이 함수는
        "동일한 hypothesis search를 neural 없이 돌리면 어디까지 가는가"
        를 보는 비교 기준으로 쓰인다.
        """

        patch_size = 2 * window_size + 1 if window_size is not None else helper["patch_size"]
        _, energy_bank = self.extract_multi_hypothesis_bank(
            rx_signals,
            cfo_grid,
            to_grid,
            patch_size=patch_size,
            return_energy=True,
            helper=helper,
        )
        collapsed_energy = torch.max(energy_bank, dim=1).values
        return collapsed_energy, energy_bank
