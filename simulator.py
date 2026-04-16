"""LoRa 심볼 생성, 채널 impairment 주입, 복조용 feature 추출을 담당하는 파일이다.

이 파일은 프로젝트 전체에서 가장 "신호 처리" 성격이 강한 부분이다.
학습과 평가에 필요한 synthetic IQ 신호를 만들고,
기본 LoRa 복조기에서 보는 score와 CNN이 볼 입력 feature를 함께 만든다.

이 파일을 읽을 때는 크게 세 단계로 이해하면 된다.

1. LoRa 기본 파형(upchirp / downchirp)을 만든다.
2. clean LoRa 심볼에 multipath, timing offset, CFO, phase noise, tone interference, AWGN을 넣는다.
3. 만들어진 수신 신호를 기본 복조기 score 또는 CNN 입력 feature bank로 바꾼다.

즉, "어떤 LoRa 신호를 보내고", "어떤 채널을 통과시키고", "수신기에는 어떤 형태로 넣는지"가
모두 이 파일 안에서 정의된다.
"""

from typing import Dict, Optional, Tuple

import torch


class GPUOnlineSimulator:
    """GPU 위에서 LoRa 신호 생성과 복조 전처리를 수행하는 시뮬레이터이다.

    이 클래스는 단순히 랜덤 신호를 만드는 도구가 아니다.
    프로젝트 전체에서 다음 세 역할을 동시에 맡는다.

    1. LoRa 물리 계층 파라미터를 바탕으로 기준 chirp를 만든다.
    2. 채널 impairment가 섞인 synthetic 수신 신호를 만든다.
    3. 만들어진 수신 신호를 classical 복조기 score 또는 CNN 입력 feature로 바꾼다.

    따라서 이 클래스의 각 함수는 "신호를 어떻게 만들고 어떻게 읽게 할 것인가"를
    단계별로 정의하는 부품이라고 보면 된다.
    """

    def __init__(self, sf: int = 7, bw: float = 125e3, fs: float = 1e6, device: str = "cuda"):
        """LoRa 기본 파라미터와 기준 chirp를 초기화한다.

        Args:
            sf: Spreading Factor. LoRa 심볼 개수 M = 2^SF를 결정한다.
            bw: LoRa 대역폭 [Hz]이다.
            fs: 시뮬레이션용 샘플링 주파수 [Hz]이다.
            device: 연산에 사용할 장치 이름이다. 보통 "cuda" 또는 "cpu"이다.
        """

        # Spreading Factor는 LoRa에서 한 심볼이 몇 개의 후보 중 하나인지를 결정한다.
        # 예를 들어 SF7이면 2^7 = 128개의 심볼 후보가 있다.
        self.sf = sf

        # LoRa 신호가 차지하는 주파수 폭이다.
        self.bw = bw

        # 우리가 파형을 몇 Hz로 샘플링해서 디지털 신호로 다룰지를 정한다.
        self.fs = fs

        # LoRa 심볼 후보 개수이다.
        self.M = 2 ** sf

        # 심볼 길이 [초]이다.
        # LoRa에서는 심볼 시간 Ts = M / BW 관계를 쓴다.
        self.Ts = self.M / bw

        # 심볼 하나를 몇 개의 샘플로 표현할지를 뜻한다.
        # 심볼 시간 Ts 동안 fs로 샘플링하므로 N = Ts * fs이다.
        self.N = int(self.Ts * fs)

        # oversampling ratio이다.
        # LoRa bin 하나가 샘플 축에서 대략 몇 칸 간격으로 배치되는지를 뜻한다.
        self.osr = self.N // self.M

        # GPU가 가능하면 GPU, 아니면 CPU를 사용한다.
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # [0, 1/fs, 2/fs, ...] 형태의 시간 축 벡터이다.
        # CFO, tone interference 같은 시간에 비례하는 위상 회전을 만들 때 사용한다.
        self.sample_times = torch.arange(self.N, device=self.device, dtype=torch.float32) / self.fs

        # [0, 1, 2, ..., N-1] 형태의 정수 인덱스 벡터이다.
        # timing shift를 "샘플 몇 칸 밀기" 형태로 구현할 때 사용한다.
        self.sample_indices = torch.arange(self.N, device=self.device, dtype=torch.long)

        # LoRa chirp의 기준 위상 누적량이다.
        # 여기서 바로 upchirp / downchirp를 만든다.
        # 비전공자 관점에서는 "시간이 갈수록 주파수가 바뀌는 chirp를 만들기 위한 위상 식" 정도로 이해하면 된다.
        self.base_phase = torch.pi * (self.bw ** 2 / self.M) * (self.sample_times ** 2)

        # 기준 upchirp이다.
        # 송신 심볼은 이 기준 upchirp에 심볼별 톤 성분을 얹어 만든다.
        self.upchirp = torch.exp(1j * self.base_phase)

        # 기준 downchirp이다.
        # 수신기에서는 dechirp를 위해 downchirp를 곱한다.
        self.downchirp = torch.exp(-1j * self.base_phase)

        # grouped-bin score 계산에 필요한 인덱스 캐시이다.
        # 같은 window_size가 반복되면 다시 계산하지 않고 재사용한다.
        self._group_index_cache = {}

        # FFT patch 추출에 필요한 인덱스 캐시이다.
        # patch_size가 같으면 동일한 인덱스가 반복되므로 캐시해 둔다.
        self._patch_index_cache = {}

    def resolve_channel_profile(self, profile: Dict) -> Dict:
        """비율 기반 채널 설정을 현재 심볼 길이에 맞는 샘플 수 설정으로 바꾼다.

        설정 파일에서는 종종
        - 심볼 길이 대비 timing offset 비율
        - 심볼 길이 대비 delay spread 비율
        처럼 상대적인 값으로 채널을 적어 둔다.

        하지만 실제 신호 생성 함수는 "몇 샘플 밀 것인가"를 알아야 하므로,
        여기서 현재 N 값에 맞춰 샘플 단위 값으로 바꿔 준다.
        """

        # 원본 profile을 직접 바꾸지 않기 위해 사본을 만든다.
        resolved = dict(profile)

        # timing offset가 "심볼 길이의 몇 %" 형태로 들어온 경우,
        # 현재 N에 맞는 샘플 수로 바꾼다.
        if "max_to_symbol_fraction" in resolved:
            resolved["max_to_samples"] = max(
                0,
                int(round(self.N * float(resolved["max_to_symbol_fraction"]))),
            )

        # delay spread도 같은 방식으로 샘플 단위 최대 지연으로 바꾼다.
        if "max_delay_symbol_fraction" in resolved:
            resolved["max_delay_samples"] = max(
                0,
                int(round(self.N * float(resolved["max_delay_symbol_fraction"]))),
            )

        # fractional timing offset를 지정하지 않은 profile도 있으므로 기본값 0을 넣는다.
        resolved.setdefault("max_fractional_to_samples", 0.0)
        return resolved

    def _sample_uniform(self, value_range: Tuple[float, float], batch_size: int, generator=None) -> torch.Tensor:
        """지정된 구간에서 균등분포 난수를 뽑는다.

        Args:
            value_range: (최소값, 최대값) 튜플이다.
            batch_size: 몇 개를 뽑을지 정한다.
            generator: PyTorch 난수 생성기를 직접 넘기고 싶을 때 사용한다.
        """

        # 범위의 양 끝값을 분리한다.
        low, high = value_range

        # [0, 1) 난수를 뽑은 뒤, 원하는 범위 [low, high]로 선형 변환한다.
        return torch.rand(batch_size, device=self.device, generator=generator) * (high - low) + low

    def _apply_sample_wise_shift(self, signals: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
        """정수 샘플 단위 timing shift를 적용한다.

        Args:
            signals: [batch, N] 복소 신호이다.
            shifts: 각 배치 항목에 적용할 정수 샘플 이동량이다.

        Returns:
            정수 shift가 적용된 [batch, N] 신호이다.
        """

        # LoRa 심볼은 길이 N의 순환 구조처럼 다루므로,
        # 인덱스 이동은 modulo N으로 감아 준다.
        gather_index = (self.sample_indices.unsqueeze(0) - shifts.long().unsqueeze(1)) % self.N

        # torch.gather를 이용해 "밀린 위치의 샘플을 읽는 방식"으로 shift를 적용한다.
        return torch.gather(signals, 1, gather_index)

    def _apply_fractional_shift(self, signals: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
        """분수 샘플 단위 timing shift를 선형 보간으로 근사한다.

        Args:
            signals: [batch, N] 복소 신호이다.
            shifts: 실수형 shift 값이다. 예를 들어 2.3 sample 같은 값이 들어올 수 있다.

        Returns:
            fractional shift가 적용된 [batch, N] 신호이다.

        설명:
            2.3 sample shift를 직접 구현하기는 어렵기 때문에,
            2 sample shift와 3 sample shift 결과를 각각 만든 뒤
            0.7 : 0.3 비율로 섞는 선형 보간을 사용한다.
        """

        # 소수점 아래를 버린 정수 부분이다.
        integer_shifts = torch.floor(shifts).long()

        # fractional part이다.
        # 예: 2.3이면 alpha = 0.3이다.
        alpha = (shifts - integer_shifts.float()).unsqueeze(1).to(signals.real.dtype)

        # 낮은 쪽 정수 shift 결과를 만든다.
        shifted_low = self._apply_sample_wise_shift(signals, integer_shifts)

        # 높은 쪽 정수 shift 결과를 만든다.
        shifted_high = self._apply_sample_wise_shift(signals, integer_shifts + 1)

        # 두 결과를 fractional part 비율로 선형 보간한다.
        return shifted_low * (1.0 - alpha) + shifted_high * alpha

    def _apply_multipath(
        self,
        tx_signals: torch.Tensor,
        path_delays: torch.Tensor,
        path_gains: torch.Tensor,
    ) -> torch.Tensor:
        """여러 경로의 delay와 복소 gain을 합쳐 multipath 수신 신호를 만든다.

        Args:
            tx_signals: [batch, N] 송신 신호이다.
            path_delays: [batch, num_paths] 각 경로의 지연 샘플 수이다.
            path_gains: [batch, num_paths] 각 경로의 복소 gain이다.

        Returns:
            multipath가 적용된 [batch, N] 수신 신호이다.
        """

        # 경로 개수이다.
        path_count = path_delays.size(1)

        # 각 경로마다 "몇 샘플 지연된 위치를 읽을 것인가"를 미리 계산한다.
        shift_indices = (
            self.sample_indices.view(1, 1, -1) - path_delays.long().unsqueeze(-1)
        ) % self.N

        # 송신 신호를 [batch, path_count, N]으로 확장한 뒤,
        # 각 경로에 맞는 지연 인덱스를 읽어 온다.
        delayed = torch.gather(
            tx_signals.unsqueeze(1).expand(-1, path_count, -1),
            2,
            shift_indices,
        )

        # 각 경로 gain을 곱하고 경로 축으로 합산하면 최종 multipath 신호가 된다.
        return torch.sum(delayed * path_gains.unsqueeze(-1), dim=1)

    def _get_group_indices(self, window_size: int) -> torch.Tensor:
        """grouped-bin 에너지 계산에 필요한 FFT 인덱스를 만든다.

        grouped-bin score는 "중심 bin 하나"만 보는 것이 아니라,
        중심 bin 주변의 몇 칸까지 함께 더해서 score를 만든다.

        Args:
            window_size: 중심 bin 양옆으로 몇 칸을 더 볼지 정한다.

        Returns:
            [M, 2 * window_size + 1] 형태의 FFT 인덱스 테이블이다.
        """

        # 같은 window_size는 반복 사용되므로 한 번 만든 뒤 캐시에 저장한다.
        if window_size not in self._group_index_cache:
            # LoRa 심볼 k가 FFT 축에서 대략 어느 중심 위치를 갖는지 계산한다.
            centers = torch.round(
                torch.arange(self.M, device=self.device, dtype=torch.float32) * self.osr
            ).long()

            # 중심 위치 좌우로 볼 오프셋을 만든다.
            offsets = torch.arange(-window_size, window_size + 1, device=self.device)

            # 각 심볼에 대해 주변 FFT 인덱스 범위를 만든다.
            self._group_index_cache[window_size] = (
                centers.unsqueeze(1) + offsets.unsqueeze(0)
            ) % self.N

        return self._group_index_cache[window_size]

    def sample_channel_state(self, batch_size: int, profile: Dict, generator=None) -> Dict[str, torch.Tensor]:
        """배치 단위 채널 상태를 샘플링한다.

        Args:
            batch_size: 몇 개의 신호에 대한 채널을 만들지 정한다.
            profile: 채널 프로파일 설정이다.
            generator: 난수 생성기를 직접 넘기고 싶을 때 사용한다.

        Returns:
            waveform 생성에 바로 쓸 수 있는 채널 상태 dict이다.

        설명:
            이 함수는 실제 채널 상태를 직접 만든다.
            즉 timing offset, path delay, path gain, phase noise 세기, tone interference 세기 등을
            한 번에 샘플링한다.
        """

        # profile 안의 비율 기반 설정을 현재 심볼 길이에 맞는 샘플 수 기준으로 바꾼다.
        profile = self.resolve_channel_profile(profile)

        # 최대 경로 수와 최대 지연 샘플 수를 읽는다.
        max_paths = profile["max_paths"]
        max_delay_samples = profile["max_delay_samples"]

        # 정수 샘플 단위 timing offset를 뽑는다.
        integer_timing_offsets = torch.randint(
            -profile["max_to_samples"],
            profile["max_to_samples"] + 1,
            (batch_size,),
            device=self.device,
            generator=generator,
        )

        # fractional timing offset를 따로 뽑는다.
        # 이 값은 정수 샘플 사이의 미세한 timing mismatch를 뜻한다.
        max_fractional_to = float(profile.get("max_fractional_to_samples", 0.0))
        if max_fractional_to > 0:
            fractional_timing_offsets = (
                torch.rand(batch_size, device=self.device, generator=generator) * 2.0 - 1.0
            ) * max_fractional_to
        else:
            fractional_timing_offsets = torch.zeros(batch_size, device=self.device)

        # 최종 timing offset는 정수 부분과 fractional 부분을 더한 값이다.
        timing_offsets = integer_timing_offsets.float() + fractional_timing_offsets

        # 공통 carrier phase offset를 샘플링한다.
        phase_offsets = torch.rand(batch_size, device=self.device, generator=generator) * (2 * torch.pi)

        # 각 경로의 delay를 샘플링한다.
        # 첫 번째 경로는 direct path 역할을 하도록 delay 0으로 고정한다.
        path_delays = torch.randint(
            0,
            max_delay_samples + 1,
            (batch_size, max_paths),
            device=self.device,
            generator=generator,
        )
        path_delays[:, 0] = 0

        # 경로 지연을 오름차순으로 정리해 해석 가능하게 만든다.
        path_delays, _ = torch.sort(path_delays, dim=1)

        # 각 경로의 복소 gain을 랜덤으로 만든다.
        # 실수부와 허수부를 각각 Gaussian으로 뽑아 복소수로 합친다.
        path_gains = (
            torch.randn((batch_size, max_paths), device=self.device, generator=generator)
            + 1j * torch.randn((batch_size, max_paths), device=self.device, generator=generator)
        )

        # 경로 지연이 클수록 평균적으로 더 약하게 만들기 위한 decay이다.
        delay_decay = torch.exp(-path_delays.float() / max(profile["delay_decay"], 1e-6))
        path_gains = path_gains * delay_decay

        # direct path가 완전히 사라지지 않도록 첫 경로를 더 강하게 만든다.
        path_gains[:, 0] = path_gains[:, 0] + 1.5

        # 추가 경로는 항상 존재하는 것이 아니라 확률적으로 켜고 끈다.
        # direct path를 제외한 나머지 경로에만 적용한다.
        if max_paths > 1:
            active_mask = (
                torch.rand((batch_size, max_paths - 1), device=self.device, generator=generator)
                < profile["extra_path_prob"]
            )
            path_gains[:, 1:] = path_gains[:, 1:] * active_mask

        # 전체 경로 gain의 총 전력을 1로 정규화한다.
        # 그렇지 않으면 경로가 많다는 이유만으로 전체 신호 전력이 지나치게 커질 수 있다.
        gain_power = torch.sum(torch.abs(path_gains) ** 2, dim=1, keepdim=True).clamp_min(1e-10)
        path_gains = path_gains / torch.sqrt(gain_power)

        # phase noise 세기를 샘플링한다.
        phase_noise_std = self._sample_uniform(profile["phase_noise_std_range"], batch_size, generator=generator)

        # narrowband tone interference가 존재할지 여부를 샘플링한다.
        tone_active = (
            torch.rand(batch_size, device=self.device, generator=generator)
            < profile["tone_interference_prob"]
        )

        # 간섭 세기를 INR[dB] 범위에서 뽑는다.
        tone_inr_db = self._sample_uniform(profile["tone_inr_db_range"], batch_size, generator=generator)

        # dB 값을 진폭 크기로 바꾼다.
        tone_amplitudes = tone_active.float() * torch.sqrt(10 ** (tone_inr_db / 10.0))

        # tone 간섭의 주파수를 대역폭 안에서 랜덤하게 고른다.
        tone_freqs_hz = self._sample_uniform((-0.45 * self.bw, 0.45 * self.bw), batch_size, generator=generator)

        # tone 간섭의 초기 위상을 랜덤하게 고른다.
        tone_phases = torch.rand(batch_size, device=self.device, generator=generator) * (2 * torch.pi)

        # 최종 채널 상태 dict를 반환한다.
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

        Args:
            channel_state: 패킷 단위 채널 상태 dict이다.
            repeats: 각 패킷 상태를 몇 번 반복할지 정한다.

        Returns:
            각 항목이 repeat_interleave된 채널 상태 dict이다.

        설명:
            예를 들어 패킷 하나가 16개의 payload symbol로 구성되어 있으면,
            그 패킷의 채널 상태를 심볼 16개에 동일하게 복사할 때 사용한다.
        """

        # 새 dict를 만든다.
        repeated = {}

        # 채널 상태의 각 항목을 심볼 수만큼 반복한다.
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
        """라벨, SNR, CFO, 채널 상태를 이용해 복소 수신 신호 배치를 만든다.

        Args:
            labels: 전송할 심볼 라벨 벡터이다.
            snrs_db: 각 샘플의 목표 SNR[dB] 벡터이다.
            cfos_hz: 각 샘플의 CFO[Hz] 벡터이다.
            channel_state: 이미 준비된 채널 상태 dict이다. 없으면 profile로부터 즉석 생성한다.
            profile: channel_state가 없을 때 채널을 샘플링할 프로파일이다.
            generator: 난수 생성기를 직접 넘기고 싶을 때 사용한다.

        Returns:
            [batch, N] 복소 수신 신호이다.

        처리 순서:
            1. clean LoRa symbol 생성
            2. multipath 적용
            3. timing shift 적용
            4. 공통 phase + CFO 적용
            5. phase noise 적용
            6. tone interference 추가
            7. 마지막에 complex AWGN 추가
        """

        # 입력을 모두 1차원 배치 벡터로 정리하고 device로 옮긴다.
        labels = labels.reshape(-1).to(self.device)
        snrs_db = snrs_db.reshape(-1).to(self.device)
        cfos_hz = cfos_hz.reshape(-1).to(self.device)

        # 배치 크기를 계산한다.
        batch_size = labels.numel()

        # 세 입력은 모두 같은 배치 크기여야 한다.
        if snrs_db.numel() != batch_size or cfos_hz.numel() != batch_size:
            raise ValueError("labels, snrs_db, and cfos_hz must have the same number of samples.")

        # channel_state가 없으면 profile에서 즉석으로 샘플링한다.
        if channel_state is None:
            if profile is None:
                raise ValueError("Either channel_state or profile must be provided.")
            channel_state = self.sample_channel_state(batch_size, profile, generator=generator)
        else:
            # channel_state를 직접 넘긴 경우, batch 크기가 맞는지 검증한다.
            for key, value in channel_state.items():
                if value.size(0) != batch_size:
                    raise ValueError(f"Channel state `{key}` has batch size {value.size(0)} but expected {batch_size}.")

        # 각 label에 해당하는 baseband tone을 만든다.
        # 이 tone을 upchirp에 얹으면 해당 LoRa 심볼이 된다.
        tone_freq = (labels.float() * self.osr).unsqueeze(1) * self.sample_indices.float().unsqueeze(0) / self.N

        # clean LoRa 심볼을 만든다.
        clean_symbols = self.upchirp.unsqueeze(0) * torch.exp(1j * 2 * torch.pi * tone_freq)

        # multipath를 적용해 desired signal을 만든다.
        desired_signals = self._apply_multipath(
            clean_symbols,
            channel_state["path_delays"],
            channel_state["path_gains"],
        )

        # timing offset를 적용한다.
        desired_signals = self._apply_fractional_shift(desired_signals, channel_state["timing_offsets"])

        # 공통 carrier phase offset를 적용한다.
        desired_signals = desired_signals * torch.exp(1j * channel_state["phase_offsets"].unsqueeze(1))

        # CFO에 해당하는 시간 비례 위상 회전을 적용한다.
        cfo_phase = 2 * torch.pi * cfos_hz.unsqueeze(1) * self.sample_times.unsqueeze(0)
        desired_signals = desired_signals * torch.exp(1j * cfo_phase)

        # phase noise를 누적 위상 형태로 적용한다.
        # 샘플마다 독립 위상 잡음이 아니라, 조금씩 흔들리는 위상 궤적을 만든다는 점이 중요하다.
        phase_noise = torch.randn(
            (batch_size, self.N),
            device=self.device,
            generator=generator,
        ) * channel_state["phase_noise_std"].unsqueeze(1)
        desired_signals = desired_signals * torch.exp(1j * torch.cumsum(phase_noise, dim=1))

        # 현재 수신 신호를 desired signal로 초기화한다.
        rx_signals = desired_signals

        # tone interference가 켜진 샘플이 있으면 간섭 tone을 더한다.
        tone_present = channel_state["tone_amplitudes"] > 0
        if torch.any(tone_present):
            # tone의 위상은 주파수 * 시간 + 초기 위상 형태이다.
            tone_phase = (
                2 * torch.pi * channel_state["tone_freqs_hz"].unsqueeze(1) * self.sample_times.unsqueeze(0)
                + channel_state["tone_phases"].unsqueeze(1)
            )

            # 복소 sinusoid 간섭 신호를 만든다.
            interferer = channel_state["tone_amplitudes"].unsqueeze(1) * torch.exp(1j * tone_phase)

            # 수신 신호에 간섭을 더한다.
            rx_signals = rx_signals + interferer

        # SNR 정의는 desired signal 전력 기준으로 잡는다.
        # 즉 tone interference는 SNR 정의 안에 포함된 것이 아니라 추가 impairment 역할을 한다.
        signal_power = torch.mean(torch.abs(desired_signals) ** 2, dim=1, keepdim=True).clamp_min(1e-10)

        # dB SNR을 선형 값으로 바꾼다.
        snr_linear = 10 ** (snrs_db.unsqueeze(1) / 10.0)

        # 목표 SNR을 만족하도록 noise power를 계산한다.
        noise_power = signal_power / snr_linear

        # complex AWGN의 실수부를 만든다.
        noise_real = torch.randn(
            rx_signals.shape,
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )

        # complex AWGN의 허수부를 만든다.
        noise_imag = torch.randn(
            rx_signals.shape,
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )

        # 복소 AWGN을 만든다.
        noise = torch.sqrt(noise_power / 2.0) * (noise_real + 1j * noise_imag)

        # 최종 수신 신호는 desired signal + interference + AWGN이다.
        return rx_signals + noise

    def _group_energy_from_fft(self, fft_mag_sq: torch.Tensor, window_size: int) -> torch.Tensor:
        """FFT magnitude 제곱으로부터 grouped-bin energy score를 만든다.

        Args:
            fft_mag_sq: [batch, N] FFT magnitude squared이다.
            window_size: 중심 bin 양옆으로 몇 칸까지 더할지 정한다.

        Returns:
            [batch, M] grouped energy score이다.

        설명:
            CFO, timing offset, multipath가 있으면 에너지가 한 bin에만 모이지 않고 주변으로 퍼질 수 있다.
            그래서 중심 bin 하나만 보기보다 주변 에너지를 합친 grouped score를 쓴다.
        """

        # 각 LoRa 심볼 bin 주변 FFT 인덱스를 읽어 온다.
        group_indices = self._get_group_indices(window_size)

        # 배치 전체에 대해 해당 인덱스를 gather한다.
        gathered = torch.gather(
            fft_mag_sq.unsqueeze(1).expand(-1, self.M, -1),
            2,
            group_indices.unsqueeze(0).expand(fft_mag_sq.size(0), -1, -1),
        )

        # 주변 에너지를 합해 심볼별 grouped score를 만든다.
        return gathered.sum(dim=2)

    def baseline_grouped_bin(self, rx_signals: torch.Tensor, window_size: int = 2):
        """기본 LoRa 복조기 점수를 계산한다.

        Args:
            rx_signals: [batch, N] 복소 수신 신호이다.
            window_size: grouped-bin score에 사용할 주변 bin 범위이다.

        Returns:
            grouped_energy: [batch, M] 기본 복조기 score이다.
            fft_mag_sq: [batch, N] FFT magnitude squared이다.
        """

        # 입력 신호를 현재 device로 옮긴다.
        rx_signals = rx_signals.to(self.device)

        # 수신 신호에 downchirp를 곱해 dechirp를 수행한다.
        dechirped = rx_signals * self.downchirp.unsqueeze(0)

        # dechirped 신호의 FFT를 본다.
        fft_complex = torch.fft.fft(dechirped, dim=1)

        # magnitude squared를 구한다.
        fft_mag_sq = torch.abs(fft_complex) ** 2

        # grouped-bin score를 계산한다.
        grouped_energy = self._group_energy_from_fft(fft_mag_sq, window_size=window_size)

        # 기본 복조기 점수와 원래 FFT 에너지를 함께 반환한다.
        return grouped_energy, fft_mag_sq

    def generate_hypothesis_grid(
        self,
        max_cfo_hz: float,
        max_to_samples: int,
        cfo_steps: int,
        to_steps: int,
    ):
        """CFO와 timing offset 가설 grid를 만든다.

        Args:
            max_cfo_hz: 고려할 CFO 범위의 절댓값 최대치이다.
            max_to_samples: 고려할 timing offset 범위의 절댓값 최대치이다.
            cfo_steps: CFO grid 개수이다.
            to_steps: timing offset grid 개수이다.

        Returns:
            cfo_grid, to_grid 튜플이다.

        설명:
            multi-hypothesis 방식은 "정확한 CFO와 timing offset를 모르니 여러 후보를 깔아 보자"는 생각이다.
            이 함수는 그 후보 목록을 균일 간격 grid로 만든다.
        """

        # CFO 후보들을 균일 간격으로 만든다.
        cfo_grid = torch.linspace(-max_cfo_hz, max_cfo_hz, cfo_steps, device=self.device)

        # timing offset 후보도 균일 간격으로 만든다.
        to_grid = torch.linspace(-max_to_samples, max_to_samples, to_steps, device=self.device).long()

        return cfo_grid, to_grid

    def _build_patch_indices(self, patch_size: int) -> torch.Tensor:
        """각 LoRa bin 주변 patch를 읽기 위한 FFT 인덱스를 만든다.

        Args:
            patch_size: 각 bin 주변에서 몇 개의 FFT 지점을 볼지 정한다.

        Returns:
            [M, patch_size] 형태의 FFT 인덱스 테이블이다.
        """

        # patch는 중심을 기준으로 좌우 대칭이 자연스러우므로 홀수 크기만 허용한다.
        if patch_size % 2 == 0:
            raise ValueError("patch_size must be odd.")

        # 같은 patch_size는 반복 사용되므로 캐시에 저장한다.
        if patch_size not in self._patch_index_cache:
            # 중심 기준 좌우 몇 칸까지 볼지 계산한다.
            half_patch = patch_size // 2

            # patch 안에서 사용할 상대 오프셋이다.
            offsets = torch.arange(-half_patch, half_patch + 1, device=self.device)

            # 각 LoRa 심볼에 대응하는 FFT 중심 위치이다.
            base_indices = (torch.arange(self.M, device=self.device) * self.osr).long()

            # 각 심볼 중심 주변 patch 인덱스를 만든다.
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
        """multi-hypothesis feature 추출에 필요한 보조 자료를 미리 계산한다.

        Args:
            cfo_grid: CFO 후보 grid이다.
            to_grid: timing offset 후보 grid이다.
            patch_size: 각 bin 주변 patch 크기이다.
            to_chunk_size: timing 후보를 몇 개씩 끊어 처리할지 정한다.

        Returns:
            feature 추출에 필요한 helper dict이다.

        설명:
            extract_multi_hypothesis_bank를 호출할 때마다
            - CFO 보정용 complex exponential
            - timing shift 인덱스
            - patch 인덱스
            를 다시 만들면 비효율적이다.
            그래서 반복 사용되는 자료를 여기서 한 번 계산한다.
        """

        # patch 추출에 쓸 인덱스를 만든다.
        patch_indices = self._build_patch_indices(patch_size)

        # 나중에 gather하기 좋도록 1차원으로 편다.
        patch_indices_flat = patch_indices.reshape(-1)

        # 각 CFO 가설에 대해 "이만큼 역회전하면 CFO가 보정된다"는 복소 보정 벡터를 만든다.
        cfo_correction = torch.exp(
            -1j * 2 * torch.pi * cfo_grid.view(-1, 1) * self.sample_times.view(1, -1)
        )

        # timing offset 후보가 많을 수 있으므로 chunk 단위로 묶어서 처리한다.
        shift_index_chunks = []
        for start in range(0, len(to_grid), to_chunk_size):
            # 현재 chunk에 해당하는 timing 후보들이다.
            to_chunk = to_grid[start:start + to_chunk_size].long()

            # 각 timing 후보에 대해 shift 후 읽어 올 샘플 인덱스를 만든다.
            shift_indices = (
                self.sample_indices.view(1, -1) - to_chunk.view(-1, 1)
            ) % self.N

            # chunk 목록에 추가한다.
            shift_index_chunks.append(shift_indices)

        # helper dict를 반환한다.
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
        """수신 신호에서 multi-hypothesis CNN 입력 feature bank를 만든다.

        Args:
            rx_signals: [batch, N] 복소 수신 신호이다.
            cfo_grid: CFO 후보 grid이다.
            to_grid: timing offset 후보 grid이다.
            patch_size: 각 bin 주변 patch 크기이다.
            return_energy: classical baseline용 energy bank도 반환할지 정한다.
            helper: prepare_hypothesis_helper로 미리 계산한 보조 자료이다.

        Returns:
            return_energy=False:
                features 하나만 반환한다.
            return_energy=True:
                (features, energy_bank) 튜플을 반환한다.

        features shape:
            [batch, 2, num_hypotheses, M * patch_size]

        energy_bank shape:
            [batch, num_hypotheses, M]

        설명:
            이 함수는 현재 프로젝트에서 CNN 입력을 만드는 핵심이다.
            여러 timing/CFO 가설을 차례로 적용해 본 뒤,
            각 가설 아래에서 dechirp + FFT 결과를 patch 형태로 뽑아
            하나의 큰 feature bank로 쌓는다.
        """

        # 입력 신호를 현재 device로 옮긴다.
        rx_signals = rx_signals.to(self.device)

        # 배치 크기를 읽는다.
        batch_size = rx_signals.size(0)

        # helper가 없으면 지금 입력 grid 기준으로 새로 만든다.
        if helper is None:
            helper = self.prepare_hypothesis_helper(cfo_grid, to_grid, patch_size)

        # helper 안에 있는 자료를 꺼낸다.
        cfo_correction = helper["cfo_correction"]
        patch_indices_flat = helper["patch_indices_flat"]
        patch_size = helper["patch_size"]
        cfo_steps = cfo_correction.size(0)

        # timing chunk별 정규화 feature를 담을 리스트이다.
        normalized_chunks = []

        # 필요할 때만 classical baseline용 energy bank도 모은다.
        energy_chunks = [] if return_energy else None

        # timing offset 후보 chunk를 하나씩 처리한다.
        for shift_indices in helper["shift_index_chunks"]:
            # 현재 chunk 안의 timing 후보 개수이다.
            chunk_size = shift_indices.size(0)

            # 1. 각 timing 가설마다 수신 신호를 shift한다.
            shifted = torch.gather(
                rx_signals.unsqueeze(1).expand(-1, chunk_size, -1),
                2,
                shift_indices.unsqueeze(0).expand(batch_size, -1, -1),
            )

            # 2. 각 shifted 신호에 대해 모든 CFO 보정 가설을 적용한다.
            corrected = shifted.unsqueeze(2) * cfo_correction.view(1, 1, cfo_correction.size(0), -1)

            # 3. dechirp를 수행한다.
            dechirped = corrected * self.downchirp.view(1, 1, 1, -1)

            # 4. 각 가설 조합에 대한 FFT를 본다.
            fft_complex = torch.fft.fft(dechirped, dim=3)

            # 5. 각 LoRa bin 주변 patch만 읽어 CNN이 볼 주파수 패턴으로 만든다.
            fft_patch = fft_complex[..., patch_indices_flat].reshape(
                batch_size,
                chunk_size,
                cfo_steps,
                self.M,
                patch_size,
            )

            # 현재 chunk의 모든 가설을 [num_hypotheses_chunk, M * patch_size] 형태로 편다.
            flattened_chunk = fft_patch.reshape(batch_size, chunk_size * cfo_steps, self.M * patch_size)

            # 각 가설 feature를 자기 자신의 최대 magnitude로 정규화한다.
            # 이렇게 하면 절대 진폭 차이보다 패턴 형태를 더 보게 된다.
            max_vals = torch.max(torch.abs(flattened_chunk), dim=2, keepdim=True).values.clamp_min(1e-10)
            normalized_chunks.append(flattened_chunk / max_vals)

            # energy bank가 필요하면 patch 단위가 아니라 bin 에너지 형태로 따로 만든다.
            if return_energy:
                energy_chunks.append(
                    torch.sum(torch.abs(fft_patch) ** 2, dim=-1).reshape(batch_size, chunk_size * cfo_steps, self.M)
                )

        # timing chunk별 feature를 hypothesis 축으로 이어 붙인다.
        normalized_bank = torch.cat(normalized_chunks, dim=1)

        # 복소수를 실수부/허수부 두 채널로 나눠 CNN 입력 형태로 만든다.
        features = torch.stack((torch.real(normalized_bank), torch.imag(normalized_bank)), dim=1)

        # energy bank도 요청했으면 함께 반환한다.
        if return_energy:
            return features, torch.cat(energy_chunks, dim=1)

        # 기본은 feature만 반환한다.
        return features
