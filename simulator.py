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
    """GPU 상에서 LoRa 신호 생성과 복조 전처리를 수행하는 시뮬레이터다."""

    def __init__(self, sf: int = 7, bw: float = 125e3, fs: float = 1e6, device: str = "cuda"):
        # LoRa 기본 파라미터를 정의한다.
        # M은 심볼 개수(= 2^SF), N은 한 심볼을 표현하는 샘플 수다.
        self.sf = sf
        self.bw = bw
        self.fs = fs
        self.M = 2 ** sf
        self.Ts = self.M / bw
        self.N = int(self.Ts * fs)
        self.osr = self.N // self.M
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.sample_times = torch.arange(self.N, device=self.device, dtype=torch.float32) / self.fs
        self.sample_indices = torch.arange(self.N, device=self.device, dtype=torch.long)

        self.base_phase = torch.pi * (self.bw ** 2 / self.M) * (self.sample_times ** 2)
        self.upchirp = torch.exp(1j * self.base_phase)
        self.downchirp = torch.exp(-1j * self.base_phase)
        self._group_index_cache = {}
        self._patch_index_cache = {}

    def resolve_channel_profile(self, profile: Dict) -> Dict:
        """심볼 길이 비율 기반 채널 설정을 현재 프로파일의 샘플 수 기준 값으로 변환한다."""

        resolved = dict(profile)

        if "max_to_symbol_fraction" in resolved:
            resolved["max_to_samples"] = max(
                0,
                int(round(self.N * float(resolved["max_to_symbol_fraction"]))),
            )

        if "max_delay_symbol_fraction" in resolved:
            resolved["max_delay_samples"] = max(
                0,
                int(round(self.N * float(resolved["max_delay_symbol_fraction"]))),
            )

        resolved.setdefault("max_fractional_to_samples", 0.0)
        return resolved

    def _sample_uniform(self, value_range: Tuple[float, float], batch_size: int, generator=None) -> torch.Tensor:
        """지정된 구간에서 균등분포 샘플을 생성한다."""

        low, high = value_range
        return torch.rand(batch_size, device=self.device, generator=generator) * (high - low) + low

    def _apply_sample_wise_shift(self, signals: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
        """정수 샘플 단위 timing shift를 적용한다."""

        gather_index = (self.sample_indices.unsqueeze(0) - shifts.long().unsqueeze(1)) % self.N
        return torch.gather(signals, 1, gather_index)

    def _apply_fractional_shift(self, signals: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
        """정수 shift와 선형 보간을 조합해 fractional timing shift를 근사한다."""

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
        """여러 경로의 지연/복소 이득을 합산해 multipath 수신 신호를 만든다."""

        path_count = path_delays.size(1)
        shift_indices = (
            self.sample_indices.view(1, 1, -1) - path_delays.long().unsqueeze(-1)
        ) % self.N
        delayed = torch.gather(
            tx_signals.unsqueeze(1).expand(-1, path_count, -1),
            2,
            shift_indices,
        )
        return torch.sum(delayed * path_gains.unsqueeze(-1), dim=1)

    def _get_group_indices(self, window_size: int) -> torch.Tensor:
        """grouped-bin 에너지 계산에 필요한 FFT 인덱스를 캐시해 반환한다."""

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

        profile = self.resolve_channel_profile(profile)
        max_paths = profile["max_paths"]
        max_delay_samples = profile["max_delay_samples"]

        integer_timing_offsets = torch.randint(
            -profile["max_to_samples"],
            profile["max_to_samples"] + 1,
            (batch_size,),
            device=self.device,
            generator=generator,
        )
        max_fractional_to = float(profile.get("max_fractional_to_samples", 0.0))
        if max_fractional_to > 0:
            fractional_timing_offsets = (
                torch.rand(batch_size, device=self.device, generator=generator) * 2.0 - 1.0
            ) * max_fractional_to
        else:
            fractional_timing_offsets = torch.zeros(batch_size, device=self.device)
        timing_offsets = integer_timing_offsets.float() + fractional_timing_offsets
        phase_offsets = torch.rand(batch_size, device=self.device, generator=generator) * (2 * torch.pi)

        # 첫 번째 경로는 항상 direct path가 되도록 delay 0을 강제한다.
        path_delays = torch.randint(
            0,
            max_delay_samples + 1,
            (batch_size, max_paths),
            device=self.device,
            generator=generator,
        )
        path_delays[:, 0] = 0
        path_delays, _ = torch.sort(path_delays, dim=1)

        path_gains = (
            torch.randn((batch_size, max_paths), device=self.device, generator=generator)
            + 1j * torch.randn((batch_size, max_paths), device=self.device, generator=generator)
        )
        delay_decay = torch.exp(-path_delays.float() / max(profile["delay_decay"], 1e-6))
        path_gains = path_gains * delay_decay
        path_gains[:, 0] = path_gains[:, 0] + 1.5

        # 추가 경로는 확률적으로 꺼서, 패킷마다 실제 경로 수가 달라지게 만든다.
        if max_paths > 1:
            active_mask = (
                torch.rand((batch_size, max_paths - 1), device=self.device, generator=generator)
                < profile["extra_path_prob"]
            )
            path_gains[:, 1:] = path_gains[:, 1:] * active_mask

        gain_power = torch.sum(torch.abs(path_gains) ** 2, dim=1, keepdim=True).clamp_min(1e-10)
        path_gains = path_gains / torch.sqrt(gain_power)

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
        """패킷 단위 채널 상태를 심볼 단위로 반복 확장한다."""

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

        labels = labels.reshape(-1).to(self.device)
        snrs_db = snrs_db.reshape(-1).to(self.device)
        cfos_hz = cfos_hz.reshape(-1).to(self.device)
        batch_size = labels.numel()

        if snrs_db.numel() != batch_size or cfos_hz.numel() != batch_size:
            raise ValueError("labels, snrs_db, and cfos_hz must have the same number of samples.")

        if channel_state is None:
            if profile is None:
                raise ValueError("Either channel_state or profile must be provided.")
            channel_state = self.sample_channel_state(batch_size, profile, generator=generator)
        else:
            for key, value in channel_state.items():
                if value.size(0) != batch_size:
                    raise ValueError(f"Channel state `{key}` has batch size {value.size(0)} but expected {batch_size}.")

        # 라벨에 해당하는 baseband tone을 upchirp에 실어 LoRa 심볼을 만든다.
        tone_freq = (labels.float() * self.osr).unsqueeze(1) * self.sample_indices.float().unsqueeze(0) / self.N
        clean_symbols = self.upchirp.unsqueeze(0) * torch.exp(1j * 2 * torch.pi * tone_freq)

        # 원하는 신호 성분(desired signal)에 채널 왜곡을 차례로 적용한다.
        desired_signals = self._apply_multipath(
            clean_symbols,
            channel_state["path_delays"],
            channel_state["path_gains"],
        )
        desired_signals = self._apply_fractional_shift(desired_signals, channel_state["timing_offsets"])

        desired_signals = desired_signals * torch.exp(1j * channel_state["phase_offsets"].unsqueeze(1))
        cfo_phase = 2 * torch.pi * cfos_hz.unsqueeze(1) * self.sample_times.unsqueeze(0)
        desired_signals = desired_signals * torch.exp(1j * cfo_phase)

        phase_noise = torch.randn(
            (batch_size, self.N),
            device=self.device,
            generator=generator,
        ) * channel_state["phase_noise_std"].unsqueeze(1)
        desired_signals = desired_signals * torch.exp(1j * torch.cumsum(phase_noise, dim=1))

        rx_signals = desired_signals
        tone_present = channel_state["tone_amplitudes"] > 0
        if torch.any(tone_present):
            tone_phase = (
                2 * torch.pi * channel_state["tone_freqs_hz"].unsqueeze(1) * self.sample_times.unsqueeze(0)
                + channel_state["tone_phases"].unsqueeze(1)
            )
            interferer = channel_state["tone_amplitudes"].unsqueeze(1) * torch.exp(1j * tone_phase)
            rx_signals = rx_signals + interferer

        # AWGN 세기는 "원하는 신호 전력" 대비 목표 SNR이 되도록 계산한다.
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
        return rx_signals + noise

    def _group_energy_from_fft(self, fft_mag_sq: torch.Tensor, window_size: int) -> torch.Tensor:
        """각 심볼 bin 주변의 에너지를 묶어서 grouped-bin score를 만든다."""

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
        dechirped = rx_signals * self.downchirp.unsqueeze(0)
        fft_complex = torch.fft.fft(dechirped, dim=1)
        fft_mag_sq = torch.abs(fft_complex) ** 2
        grouped_energy = self._group_energy_from_fft(fft_mag_sq, window_size=window_size)
        return grouped_energy, fft_mag_sq

    def generate_hypothesis_grid(
        self,
        max_cfo_hz: float,
        max_to_samples: int,
        cfo_steps: int,
        to_steps: int,
    ):
        """CFO / timing offset 가설 grid를 균일 간격으로 생성한다."""

        cfo_grid = torch.linspace(-max_cfo_hz, max_cfo_hz, cfo_steps, device=self.device)
        to_grid = torch.linspace(-max_to_samples, max_to_samples, to_steps, device=self.device).long()
        return cfo_grid, to_grid

    def _build_patch_indices(self, patch_size: int) -> torch.Tensor:
        """각 LoRa bin 주변 patch를 뽑기 위한 FFT 인덱스를 만든다."""

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

        이 helper는 반복 평가 시 중복 계산을 줄이기 위한 캐시 역할을 한다.
        """

        patch_indices = self._build_patch_indices(patch_size)
        patch_indices_flat = patch_indices.reshape(-1)
        cfo_correction = torch.exp(
            -1j * 2 * torch.pi * cfo_grid.view(-1, 1) * self.sample_times.view(1, -1)
        )

        shift_index_chunks = []
        for start in range(0, len(to_grid), to_chunk_size):
            to_chunk = to_grid[start:start + to_chunk_size].long()
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

        반환 특징은 CNN 입력용 2채널 텐서이며,
        필요하면 가설별 에너지 bank도 함께 반환한다.
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
            shifted = torch.gather(
                rx_signals.unsqueeze(1).expand(-1, chunk_size, -1),
                2,
                shift_indices.unsqueeze(0).expand(batch_size, -1, -1),
            )
            corrected = shifted.unsqueeze(2) * cfo_correction.view(1, 1, cfo_correction.size(0), -1)
            dechirped = corrected * self.downchirp.view(1, 1, 1, -1)
            fft_complex = torch.fft.fft(dechirped, dim=3)
            fft_patch = fft_complex[..., patch_indices_flat].reshape(
                batch_size,
                chunk_size,
                cfo_steps,
                self.M,
                patch_size,
            )
            flattened_chunk = fft_patch.reshape(batch_size, chunk_size * cfo_steps, self.M * patch_size)
            max_vals = torch.max(torch.abs(flattened_chunk), dim=2, keepdim=True).values.clamp_min(1e-10)
            normalized_chunks.append(flattened_chunk / max_vals)

            if return_energy:
                energy_chunks.append(
                    torch.sum(torch.abs(fft_patch) ** 2, dim=-1).reshape(batch_size, chunk_size * cfo_steps, self.M)
                )

        normalized_bank = torch.cat(normalized_chunks, dim=1)
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

        각 hypothesis에서 얻은 에너지 중 최댓값을 취해
        심볼별 collapsed energy를 만든다.
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
