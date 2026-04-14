"""학습/검증/평가에 사용하는 데이터셋 생성 함수를 모아 둔 파일이다.

이 파일은 크게 두 종류의 데이터를 만든다.

1. 온라인 학습용 파라미터 데이터셋
   - `label`, `SNR`, `CFO`만 미리 뽑아 둔다.
   - 실제 waveform은 학습 루프 안에서 채널 상태와 함께 생성한다.

2. 고정 waveform 데이터셋
   - validation / calibration / test 단계에서 동일한 입력을 반복 사용하기 위해
     IQ waveform 자체를 미리 생성해 저장한다.

또한 외부에서 저장한 recorded IQ `.npz`를 읽어오는 도우미도 포함되어 있다.
"""

from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

from config import CFG
from utils import get_max_cfo_hz


class OnlineParametersDataset(Dataset):
    """온라인 학습에 필요한 `(label, SNR, CFO)`만 샘플링하는 데이터셋이다.

    실제 waveform은 여기서 만들지 않는다.
    학습 루프에서 매 배치마다 채널 상태를 새로 샘플링해 waveform을 생성하므로,
    같은 label/SNR/CFO 조합이라도 채널 realization은 매번 달라질 수 있다.
    """

    def __init__(self, M: int, num_samples: int, snr_range: Tuple[float, float], max_cfo_bins: float, bw: float):
        self.M = M
        self.num_samples = num_samples
        self.snr_range = snr_range
        self.max_cfo_hz = max_cfo_bins * (bw / M)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # idx는 길이 계산용으로만 사용하고, 실제 샘플은 매 호출마다 랜덤하게 생성한다.
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
    """패킷 단위로 공유되는 파라미터와 채널 상태를 생성한다.

    이 함수의 핵심은 `한 패킷 안의 payload symbol들이 같은 채널 상태를 공유한다`는 점이다.
    따라서 packet 단위 consistency를 가진 실험을 구성할 수 있다.
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
    channel_state = simulator.repeat_channel_state(channel_state_pkt, payload_symbols)

    return labels, snrs, cfos, channel_state


def create_fixed_feature_dataset(
    simulator,
    num_packets: int,
    snr_range: Tuple[float, float],
    channel_profile: Dict,
    seed: int,
    experiment_cfg: Dict = None,
    feature_cfg: Dict = None,
):
    """고정된 feature tensor를 미리 만들어 TensorDataset으로 반환한다.

    이 함수는 feature 자체를 미리 저장하므로 빠른 반복 실험에는 편리하지만,
    메모리를 많이 사용한다. 현재 코드에서는 validation을 waveform 기반으로 두고 있어
    이 함수는 필요 시에만 보조적으로 사용할 수 있다.
    """

    device = simulator.device
    experiment_cfg = CFG["experiment"] if experiment_cfg is None else experiment_cfg
    feature_cfg = CFG["feature_bank"] if feature_cfg is None else feature_cfg
    payload_symbols = experiment_cfg["payload_symbols"]
    resolved_profile = simulator.resolve_channel_profile(channel_profile)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    labels, snrs, cfos, channel_state = _build_packet_parameters(
        simulator,
        num_packets,
        snr_range,
        channel_profile,
        payload_symbols,
        generator=gen,
    )

    max_cfo_hz = get_max_cfo_hz(simulator, channel_profile)
    cfo_grid, to_grid = simulator.generate_hypothesis_grid(
        max_cfo_hz,
        resolved_profile["max_to_samples"],
        feature_cfg["cfo_steps"],
        feature_cfg["to_steps"],
    )
    helper = simulator.prepare_hypothesis_helper(
        cfo_grid,
        to_grid,
        feature_cfg["patch_size"],
    )

    num_samples = num_packets * payload_symbols
    num_hypotheses = feature_cfg["cfo_steps"] * feature_cfg["to_steps"]
    num_bins = simulator.M * feature_cfg["patch_size"]

    features = torch.empty((num_samples, 2, num_hypotheses, num_bins), dtype=torch.float32)
    labels_flat = labels.reshape(-1).cpu()

    with torch.no_grad():
        chunk_packets = 32
        write_ptr = 0
        for pkt_start in range(0, num_packets, chunk_packets):
            pkt_end = min(pkt_start + chunk_packets, num_packets)
            labels_chunk = labels[pkt_start:pkt_end].reshape(-1)
            snrs_chunk = snrs[pkt_start:pkt_end].repeat_interleave(payload_symbols)
            cfos_chunk = cfos[pkt_start:pkt_end].repeat_interleave(payload_symbols)
            state_chunk = simulator.repeat_channel_state(
                {key: value[pkt_start:pkt_end] for key, value in channel_state.items()},
                payload_symbols,
            )
            rx_signals = simulator.generate_batch(
                labels_chunk,
                snrs_chunk,
                cfos_chunk,
                channel_state=state_chunk,
                generator=gen,
            )
            feature_bank = simulator.extract_multi_hypothesis_bank(
                rx_signals,
                helper=helper,
            )
            next_ptr = write_ptr + feature_bank.size(0)
            features[write_ptr:next_ptr] = feature_bank.cpu()
            write_ptr = next_ptr

    return TensorDataset(labels_flat, features)


def create_fixed_waveform_range_dataset(
    simulator,
    num_packets: int,
    snr_range: Tuple[float, float],
    channel_profile: Dict,
    seed: int,
    experiment_cfg: Dict = None,
):
    """하나의 SNR 범위 안에서 waveform을 고정 생성해 반환한다.

    validation처럼 `train_snr_range` 전체를 대표하는 고정 데이터셋이 필요할 때 사용한다.
    """

    device = simulator.device
    experiment_cfg = CFG["experiment"] if experiment_cfg is None else experiment_cfg
    payload_symbols = experiment_cfg["payload_symbols"]

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    labels, snrs, cfos, channel_state = _build_packet_parameters(
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
            state_chunk = simulator.repeat_channel_state(
                {key: value[pkt_start:pkt_end] for key, value in channel_state.items()},
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
):
    """SNR별로 독립적인 고정 waveform dataset을 생성한다.

    반환값은 `{snr: TensorDataset}` 형태의 딕셔너리다.
    calibration / seen test / unseen test에서 이 구조를 그대로 사용한다.
    """

    device = simulator.device
    experiment_cfg = CFG["experiment"] if experiment_cfg is None else experiment_cfg
    payload_symbols = experiment_cfg["payload_symbols"]

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    datasets = {}
    for snr in snr_list:
        labels, _, cfos, channel_state = _build_packet_parameters(
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
                state_chunk = simulator.repeat_channel_state(
                    {key: value[pkt_start:pkt_end] for key, value in channel_state.items()},
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
    """외부에서 저장한 recorded IQ `.npz` 파일을 TensorDataset으로 읽어온다."""

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
