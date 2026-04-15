"""학습, 검증, 평가에 사용하는 데이터셋 생성 유틸리티이다.

이 파일은 두 종류의 데이터셋을 만든다.

1. 온라인 학습용 파라미터 데이터셋
   - label, SNR, CFO만 샘플링한다.
   - 실제 waveform은 training loop 안에서 simulator가 실시간 생성한다.

2. 고정 waveform 데이터셋
   - validation, calibration, seen/unseen test에서 같은 입력을 반복 사용하기 위해
     IQ waveform 자체를 미리 생성해 TensorDataset으로 저장한다.
"""

from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

from config import CFG
from utils import get_max_cfo_hz


class OnlineParametersDataset(Dataset):
    """온라인 학습에 필요한 `(label, SNR, CFO)`만 샘플링하는 데이터셋이다."""

    def __init__(self, M: int, num_samples: int, snr_range: Tuple[float, float], max_cfo_bins: float, bw: float):
        self.M = M
        self.num_samples = num_samples
        self.snr_range = snr_range
        self.max_cfo_hz = max_cfo_bins * (bw / M)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # idx는 길이 계산용으로만 쓰고, 실제 샘플은 호출 시점마다 새로 뽑는다.
        label = torch.randint(0, self.M, (1,)).item()
        snr = torch.empty(1).uniform_(self.snr_range[0], self.snr_range[1]).item()
        cfo = torch.empty(1).uniform_(-self.max_cfo_hz, self.max_cfo_hz).item()
        return (
            torch.tensor(label, dtype=torch.long),
            torch.tensor(snr, dtype=torch.float32),
            torch.tensor(cfo, dtype=torch.float32),
        )


def _build_packet_parameters(
    simulator,
    num_packets: int,
    snr_range: Tuple[float, float],
    channel_profile: Dict,
    payload_symbols: int,
    generator,
):
    """패킷 단위로 공유되는 label, SNR, CFO, channel_state를 생성한다.

    반환 형식은 다음과 같다.

    - labels: [num_packets, payload_symbols]
    - snrs: [num_packets]
    - cfos: [num_packets]
    - channel_state_pkt: 패킷 단위 channel state

    중요한 점:
    channel_state는 여기서 심볼 단위로 반복하지 않는다.
    각 waveform 생성 함수가 현재 packet chunk 범위를 자른 뒤
    payload_symbols만큼 정확히 한 번만 반복해야 한다.
    """

    device = simulator.device
    max_cfo_hz = get_max_cfo_hz(simulator, channel_profile)

    labels = torch.randint(
        0,
        simulator.M,
        (num_packets, payload_symbols),
        generator=generator,
        device=device,
    )
    snrs = (
        torch.rand(num_packets, generator=generator, device=device)
        * (snr_range[1] - snr_range[0])
        + snr_range[0]
    )
    cfos = (
        torch.rand(num_packets, generator=generator, device=device)
        * (2 * max_cfo_hz)
        - max_cfo_hz
    )
    channel_state_pkt = simulator.sample_channel_state(num_packets, channel_profile, generator=generator)
    return labels, snrs, cfos, channel_state_pkt


def _expand_packet_channel_state(
    simulator,
    channel_state_pkt: Dict[str, torch.Tensor],
    pkt_start: int,
    pkt_end: int,
    payload_symbols: int,
):
    """현재 packet chunk에 해당하는 channel_state를 심볼 단위로 한 번만 확장한다."""

    packet_state_chunk = {
        key: value[pkt_start:pkt_end]
        for key, value in channel_state_pkt.items()
    }
    return simulator.repeat_channel_state(packet_state_chunk, payload_symbols)


def create_fixed_waveform_range_dataset(
    simulator,
    num_packets: int,
    snr_range: Tuple[float, float],
    channel_profile: Dict,
    seed: int,
    experiment_cfg: Dict = None,
):
    """하나의 SNR 범위 안에서 waveform을 고정 생성해 반환한다."""

    device = simulator.device
    experiment_cfg = CFG["experiment"] if experiment_cfg is None else experiment_cfg
    payload_symbols = experiment_cfg["payload_symbols"]

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    labels, snrs, cfos, channel_state_pkt = _build_packet_parameters(
        simulator,
        num_packets,
        snr_range,
        channel_profile,
        payload_symbols,
        generator=gen,
    )

    labels_flat = labels.reshape(-1).cpu()
    waveforms = torch.empty((num_packets * payload_symbols, simulator.N), dtype=torch.complex64)

    with torch.no_grad():
        chunk_packets = 32
        write_ptr = 0
        for pkt_start in range(0, num_packets, chunk_packets):
            pkt_end = min(pkt_start + chunk_packets, num_packets)
            labels_chunk = labels[pkt_start:pkt_end].reshape(-1)
            snrs_chunk = snrs[pkt_start:pkt_end].repeat_interleave(payload_symbols)
            cfos_chunk = cfos[pkt_start:pkt_end].repeat_interleave(payload_symbols)
            state_chunk = _expand_packet_channel_state(
                simulator,
                channel_state_pkt,
                pkt_start,
                pkt_end,
                payload_symbols,
            )
            rx_signals = simulator.generate_batch(
                labels_chunk,
                snrs_chunk,
                cfos_chunk,
                channel_state=state_chunk,
                generator=gen,
            )
            next_ptr = write_ptr + rx_signals.size(0)
            waveforms[write_ptr:next_ptr] = rx_signals.cpu()
            write_ptr = next_ptr

    if write_ptr != waveforms.size(0):
        raise RuntimeError("Waveform dataset generation wrote an unexpected number of samples.")

    return TensorDataset(labels_flat, waveforms)


def create_fixed_waveform_dataset(
    simulator,
    num_packets_per_snr: int,
    snr_list,
    channel_profile: Dict,
    seed: int,
    experiment_cfg: Dict = None,
    shared_channel_state_across_snr: bool = False,
):
    """SNR별 고정 waveform dataset을 생성한다.

    shared_channel_state_across_snr=True이면 첫 SNR에서 한 번 샘플링한
    label, CFO, channel_state를 이후 모든 SNR에서 재사용한다.
    즉 채널 상태와 라벨은 유지하고, SNR과 AWGN 샘플만 바뀐다.
    """

    device = simulator.device
    experiment_cfg = CFG["experiment"] if experiment_cfg is None else experiment_cfg
    payload_symbols = experiment_cfg["payload_symbols"]

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    datasets = {}
    shared_packet_parameters = None

    for snr in snr_list:
        if shared_channel_state_across_snr:
            if shared_packet_parameters is None:
                shared_packet_parameters = _build_packet_parameters(
                    simulator,
                    num_packets_per_snr,
                    (snr, snr),
                    channel_profile,
                    payload_symbols,
                    generator=gen,
                )
            labels, _, cfos, channel_state_pkt = shared_packet_parameters
        else:
            labels, _, cfos, channel_state_pkt = _build_packet_parameters(
                simulator,
                num_packets_per_snr,
                (snr, snr),
                channel_profile,
                payload_symbols,
                generator=gen,
            )

        snrs = torch.full_like(cfos, float(snr))
        labels_flat = labels.reshape(-1).cpu()
        waveforms = torch.empty((num_packets_per_snr * payload_symbols, simulator.N), dtype=torch.complex64)

        with torch.no_grad():
            chunk_packets = 32
            write_ptr = 0
            for pkt_start in range(0, num_packets_per_snr, chunk_packets):
                pkt_end = min(pkt_start + chunk_packets, num_packets_per_snr)
                labels_chunk = labels[pkt_start:pkt_end].reshape(-1)
                snrs_chunk = snrs[pkt_start:pkt_end].repeat_interleave(payload_symbols)
                cfos_chunk = cfos[pkt_start:pkt_end].repeat_interleave(payload_symbols)
                state_chunk = _expand_packet_channel_state(
                    simulator,
                    channel_state_pkt,
                    pkt_start,
                    pkt_end,
                    payload_symbols,
                )
                rx_signals = simulator.generate_batch(
                    labels_chunk,
                    snrs_chunk,
                    cfos_chunk,
                    channel_state=state_chunk,
                    generator=gen,
                )
                next_ptr = write_ptr + rx_signals.size(0)
                waveforms[write_ptr:next_ptr] = rx_signals.cpu()
                write_ptr = next_ptr

        if write_ptr != waveforms.size(0):
            raise RuntimeError(f"Waveform dataset generation failed for SNR {snr}.")

        datasets[snr] = TensorDataset(labels_flat, waveforms)

    return datasets


def load_recorded_waveform_dataset(npz_path: str, expected_num_samples: int = None):
    """외부에서 저장한 recorded IQ `.npz` 파일을 TensorDataset으로 읽는다."""

    data = np.load(npz_path)
    labels = torch.from_numpy(data["labels"]).long()

    if "rx" in data:
        rx = torch.from_numpy(data["rx"])
        if not torch.is_complex(rx):
            rx = rx[..., 0] + 1j * rx[..., 1]
    elif "rx_real" in data and "rx_imag" in data:
        rx = torch.from_numpy(data["rx_real"]) + 1j * torch.from_numpy(data["rx_imag"])
    else:
        raise ValueError("NPZ file must contain `rx` or (`rx_real`, `rx_imag`).")

    rx = rx.to(torch.complex64)
    if labels.ndim != 1:
        raise ValueError("`labels` must be a 1D tensor.")
    if rx.ndim != 2:
        raise ValueError("Recorded IQ data must have shape [num_symbols, num_samples].")
    if rx.size(0) != labels.numel():
        raise ValueError("The number of IQ symbols must match the number of labels.")
    if expected_num_samples is not None and rx.size(1) != expected_num_samples:
        raise ValueError(f"Expected {expected_num_samples} IQ samples per symbol, but found {rx.size(1)}.")

    return TensorDataset(labels, rx)
