"""학습, 검증, 보정, 평가에 사용할 데이터셋을 만드는 파일이다.

이 파일의 역할은 크게 두 가지다.

1. 온라인 학습용 파라미터 데이터셋을 만든다.
   - 여기서는 실제 IQ waveform을 저장하지 않는다.
   - 대신 `(label, SNR, CFO)`만 뽑아 training loop에 넘긴다.
   - training loop 안에서 simulator가 그때그때 waveform을 합성한다.

2. 고정 waveform 데이터셋을 만든다.
   - validation, calibration, seen/unseen test에서는
     매 epoch 또는 매 평가마다 완전히 같은 입력을 다시 써야 비교가 안정적이다.
   - 그래서 이 경우에는 IQ waveform 자체를 미리 만들어 `TensorDataset`으로 저장한다.

즉 이 파일은 "어떤 데이터를 어떤 형식으로 준비할 것인가"를 담당한다.
실제 LoRa 수신 신호 생성은 simulator가 맡고,
이 파일은 그 simulator를 어떤 규칙으로 호출할지를 정리한다.
"""

from typing import Dict, Tuple
import torch
from torch.utils.data import Dataset, TensorDataset

from config import CFG
from utils import get_max_cfo_hz


class OnlineParametersDataset(Dataset):
    """온라인 학습용 `(label, SNR, CFO)` 샘플을 제공하는 데이터셋이다.

    중요한 점은 이 클래스가 실제 수신 waveform을 들고 있지 않다는 것이다.
    `__getitem__`이 호출될 때마다
    - 정답 심볼 label
    - 해당 샘플에 적용할 SNR
    - 해당 샘플에 적용할 CFO
    를 새로 뽑아서 반환한다.

    왜 이렇게 하느냐면,
    학습용 waveform을 전부 미리 저장하면 메모리 사용량이 너무 커지기 때문이다.
    따라서 학습 단계에서는 "파라미터만 저장하고, waveform은 실시간 생성" 전략을 쓴다.
    """

    def __init__(
        self,
        M: int,
        num_samples: int,
        snr_range: Tuple[float, float],
        max_cfo_bins: float,
        bw: float,
    ):
        # M:
        # LoRa 심볼 후보 개수다.
        # 예를 들어 SF7이면 M = 2^7 = 128이다.
        self.M = M

        # num_samples:
        # 이 데이터셋이 한 epoch 동안 몇 개 샘플을 공급할지 정한다.
        self.num_samples = num_samples

        # snr_range:
        # 학습 중 SNR을 어느 범위에서 랜덤 샘플링할지 정한다.
        self.snr_range = snr_range

        # max_cfo_bins는 "LoRa bin 간격 기준 CFO 허용 범위"다.
        # 실제 waveform 생성에는 Hz 단위가 필요하므로 여기서 Hz로 변환해 둔다.
        self.max_cfo_hz = max_cfo_bins * (bw / M)

    def __len__(self):
        # DataLoader가 "이 데이터셋을 몇 번 꺼내 쓰면 되는가"를 알기 위한 길이다.
        return self.num_samples

    def __getitem__(self, idx):
        # idx는 Dataset 인터페이스 때문에 받는 값이다.
        # 하지만 이 클래스는 "미리 저장된 idx번째 샘플"을 꺼내는 구조가 아니라,
        # 호출될 때마다 새 샘플을 랜덤 생성하는 구조이므로
        # idx 자체는 실제 샘플 내용 계산에 사용하지 않는다.

        # label:
        # 0 ~ M-1 사이의 정답 심볼 인덱스를 하나 뽑는다.
        label = torch.randint(0, self.M, (1,)).item()

        # snr:
        # 지정된 SNR 범위 안에서 균일분포로 하나 뽑는다.
        snr = torch.empty(1).uniform_(self.snr_range[0], self.snr_range[1]).item()

        # cfo:
        # -max_cfo_hz ~ +max_cfo_hz 범위에서 CFO를 하나 뽑는다.
        cfo = torch.empty(1).uniform_(-self.max_cfo_hz, self.max_cfo_hz).item()

        # 반환 형식은 모두 torch tensor다.
        # training loop는 이 값을 그대로 GPU로 올려 simulator에 넣는다.
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
    """패킷 단위로 공유할 label, SNR, CFO, channel state를 생성한다.

    이 함수는 "고정 waveform 데이터셋"을 만들 때 쓰인다.
    여기서는 심볼 하나씩 독립적으로 뽑지 않고,
    먼저 패킷 단위로 큰 틀의 파라미터를 만든 뒤
    나중에 심볼 단위로 펼쳐서 waveform을 만든다.

    반환값 형식은 다음과 같다.

    - labels: [num_packets, payload_symbols]
      각 패킷 안의 payload 심볼 라벨들이다.
    - snrs: [num_packets]
      패킷 단위 SNR이다.
      이후 각 패킷 안의 payload_symbols개 심볼에 반복 적용한다.
    - cfos: [num_packets]
      패킷 단위 CFO다.
    - channel_state_pkt:
      패킷 단위 채널 상태다.

    여기서 특히 중요한 점은
    `channel_state_pkt`를 아직 심볼 단위로 반복하지 않는다는 것이다.
    과거 버전에서는 이 반복이 두 번 들어가 packet-symbol 정렬이 깨졌고,
    그 문제를 막기 위해 지금은 packet 단위 상태를 유지한 채 반환한다.
    """

    # simulator가 현재 CPU/GPU 중 어느 장치를 쓰는지 그대로 따라간다.
    device = simulator.device

    # channel_profile에 들어 있는 max_cfo_bins를 실제 Hz 단위로 바꾼다.
    max_cfo_hz = get_max_cfo_hz(simulator, channel_profile)

    # labels:
    # 각 패킷마다 payload_symbols개의 정답 심볼을 뽑는다.
    # shape은 [num_packets, payload_symbols]가 된다.
    labels = torch.randint(
        0,
        simulator.M,
        (num_packets, payload_symbols),
        generator=generator,
        device=device,
    )

    # snrs:
    # 각 패킷에 적용할 SNR 하나를 뽑는다.
    # shape은 [num_packets]다.
    snrs = (
        torch.rand(num_packets, generator=generator, device=device)
        * (snr_range[1] - snr_range[0])
        + snr_range[0]
    )

    # cfos:
    # 각 패킷에 적용할 CFO 하나를 뽑는다.
    # shape은 [num_packets]다.
    cfos = (
        torch.rand(num_packets, generator=generator, device=device)
        * (2 * max_cfo_hz)
        - max_cfo_hz
    )

    # channel_state_pkt:
    # multipath, timing offset, phase noise, interference 같은 채널 상태를
    # 패킷 단위로 한 번 샘플링한다.
    channel_state_pkt = simulator.sample_channel_state(
        num_packets,
        channel_profile,
        generator=generator,
    )

    return labels, snrs, cfos, channel_state_pkt


def _expand_packet_channel_state(
    simulator,
    channel_state_pkt: Dict[str, torch.Tensor],
    pkt_start: int,
    pkt_end: int,
    payload_symbols: int,
):
    """패킷 단위 channel state를 현재 chunk에 맞는 심볼 단위 상태로 펼친다.

    waveform을 만들 때는 packet 전체를 한 번에 만들지 않고
    메모리 사용량을 줄이기 위해 packet chunk 단위로 잘라서 처리한다.

    따라서 현재 chunk가
    `pkt_start ~ pkt_end-1` 패킷을 다룬다면,
    그 구간의 packet-level channel state만 잘라낸 뒤
    payload_symbols만큼 반복해 symbol-level channel state로 만들어야 한다.
    """

    # 현재 chunk에 해당하는 packet-level state만 자른다.
    packet_state_chunk = {
        key: value[pkt_start:pkt_end]
        for key, value in channel_state_pkt.items()
    }

    # packet state 하나가 payload_symbols개의 심볼에 공통 적용되도록 반복 확장한다.
    return simulator.repeat_channel_state(packet_state_chunk, payload_symbols)


def create_fixed_waveform_range_dataset(
    simulator,
    num_packets: int,
    snr_range: Tuple[float, float],
    channel_profile: Dict,
    seed: int,
    experiment_cfg: Dict = None,
):
    """하나의 SNR 범위에서 고정 waveform 데이터셋을 만든다.

    이 함수는 주로 validation에 사용된다.
    validation은 epoch마다 완전히 같은 입력으로 모델을 비교해야 하므로
    waveform을 미리 만들어 둔 `TensorDataset`을 반환한다.
    """

    # simulator가 정한 장치를 그대로 사용한다.
    device = simulator.device

    # experiment_cfg를 따로 넘기지 않으면 전역 기본 설정을 사용한다.
    experiment_cfg = CFG["experiment"] if experiment_cfg is None else experiment_cfg

    # packet 하나 안에 payload 심볼이 몇 개인지 가져온다.
    payload_symbols = experiment_cfg["payload_symbols"]

    # generator:
    # 이 데이터셋 생성 과정 전체에서 재현 가능한 난수 순서를 만들기 위한 torch 난수기다.
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # packet-level label/SNR/CFO/channel_state를 만든다.
    labels, snrs, cfos, channel_state_pkt = _build_packet_parameters(
        simulator,
        num_packets,
        snr_range,
        channel_profile,
        payload_symbols,
        generator=gen,
    )

    # labels_flat:
    # 최종 TensorDataset은 심볼 단위 샘플을 반환하므로
    # [num_packets, payload_symbols] 라벨을 1차원으로 평탄화한다.
    labels_flat = labels.reshape(-1).cpu()

    # waveforms:
    # 심볼 단위 복소 수신 신호를 저장할 버퍼다.
    # 총 샘플 수는 num_packets * payload_symbols개다.
    waveforms = torch.empty(
        (num_packets * payload_symbols, simulator.N),
        dtype=torch.complex64,
    )

    # 고정 dataset 생성 단계에서는 gradient가 전혀 필요 없다.
    with torch.no_grad():
        # chunk_packets:
        # packet을 한 번에 전부 생성하면 GPU 메모리가 커질 수 있으므로
        # 일정 packet 수씩 잘라서 처리한다.
        chunk_packets = 32

        # write_ptr:
        # waveforms 버퍼의 어디까지 썼는지 가리키는 포인터다.
        write_ptr = 0

        # packet chunk 단위로 반복한다.
        for pkt_start in range(0, num_packets, chunk_packets):
            pkt_end = min(pkt_start + chunk_packets, num_packets)

            # 현재 chunk 안의 label을 심볼 단위로 평탄화한다.
            labels_chunk = labels[pkt_start:pkt_end].reshape(-1)

            # packet 단위 snr/cfo를 payload_symbols만큼 반복해 심볼 단위로 맞춘다.
            snrs_chunk = snrs[pkt_start:pkt_end].repeat_interleave(payload_symbols)
            cfos_chunk = cfos[pkt_start:pkt_end].repeat_interleave(payload_symbols)

            # packet-level channel state도 현재 chunk 범위만 잘라
            # 심볼 단위로 한 번만 확장한다.
            state_chunk = _expand_packet_channel_state(
                simulator,
                channel_state_pkt,
                pkt_start,
                pkt_end,
                payload_symbols,
            )

            # generate_batch:
            # labels/SNR/CFO/channel_state를 받아 실제 복소 수신 waveform을 만든다.
            rx_signals = simulator.generate_batch(
                labels_chunk,
                snrs_chunk,
                cfos_chunk,
                channel_state=state_chunk,
                generator=gen,
            )

            # 현재 chunk 결과를 waveforms 버퍼에 이어붙인다.
            next_ptr = write_ptr + rx_signals.size(0)
            waveforms[write_ptr:next_ptr] = rx_signals.cpu()
            write_ptr = next_ptr

    # 버퍼를 예상 개수만큼 정확히 채웠는지 확인한다.
    if write_ptr != waveforms.size(0):
        raise RuntimeError("Waveform dataset generation wrote an unexpected number of samples.")

    # 최종 반환 형식은 (label, waveform) 쌍이다.
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
    """SNR별 고정 waveform 데이터셋 묶음을 만든다.

    반환값은 `{snr: TensorDataset}` 형태다.
    calibration, seen test, unseen test처럼
    여러 SNR 지점에 대해 따로 평가해야 하는 경우 이 함수를 사용한다.

    `shared_channel_state_across_snr=True`이면
    첫 번째 SNR에서 한 번 뽑은
    - label
    - CFO
    - channel state
    를 이후 모든 SNR에서 재사용한다.

    즉 완전히 같은 noisy waveform을 재사용하는 것은 아니고,
    packet 조건은 공유하되 각 SNR에서 AWGN 샘플은 새로 생성한다.
    """

    # simulator 장치를 그대로 사용한다.
    device = simulator.device

    # experiment_cfg가 없으면 기본 설정을 가져온다.
    experiment_cfg = CFG["experiment"] if experiment_cfg is None else experiment_cfg

    # 한 packet 안의 payload 심볼 개수다.
    payload_symbols = experiment_cfg["payload_symbols"]

    # 전체 데이터셋 생성 과정의 난수 재현성을 맞추기 위한 generator다.
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # datasets:
    # 각 SNR별 TensorDataset을 담을 dict다.
    datasets = {}

    # shared_packet_parameters:
    # shared_channel_state_across_snr=True일 때 첫 SNR에서 생성한 packet 조건을 저장한다.
    shared_packet_parameters = None

    # SNR 지점마다 별도 dataset을 만든다.
    for snr in snr_list:
        # 공유 모드면 첫 번째 SNR에서 만든 packet 조건을 계속 재사용한다.
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
            # 독립 모드면 SNR마다 새로운 label/CFO/channel_state를 뽑는다.
            labels, _, cfos, channel_state_pkt = _build_packet_parameters(
                simulator,
                num_packets_per_snr,
                (snr, snr),
                channel_profile,
                payload_symbols,
                generator=gen,
            )

        # snrs:
        # 현재 SNR 값을 packet 수만큼 채운 벡터다.
        snrs = torch.full_like(cfos, float(snr))

        # label은 최종적으로 심볼 단위로 평가하므로 1차원으로 편다.
        labels_flat = labels.reshape(-1).cpu()

        # 현재 SNR용 waveform 저장 버퍼를 만든다.
        waveforms = torch.empty(
            (num_packets_per_snr * payload_symbols, simulator.N),
            dtype=torch.complex64,
        )

        with torch.no_grad():
            # packet chunk 단위 처리로 메모리 사용량을 제한한다.
            chunk_packets = 32
            write_ptr = 0

            for pkt_start in range(0, num_packets_per_snr, chunk_packets):
                pkt_end = min(pkt_start + chunk_packets, num_packets_per_snr)

                # 현재 chunk 라벨을 심볼 단위 1차원으로 편다.
                labels_chunk = labels[pkt_start:pkt_end].reshape(-1)

                # packet 단위 snr/cfo를 심볼 단위로 펼친다.
                snrs_chunk = snrs[pkt_start:pkt_end].repeat_interleave(payload_symbols)
                cfos_chunk = cfos[pkt_start:pkt_end].repeat_interleave(payload_symbols)

                # packet-level channel state를 현재 chunk 범위에 맞춰 잘라
                # 심볼 단위로 한 번만 확장한다.
                state_chunk = _expand_packet_channel_state(
                    simulator,
                    channel_state_pkt,
                    pkt_start,
                    pkt_end,
                    payload_symbols,
                )

                # 현재 chunk의 수신 waveform을 생성한다.
                rx_signals = simulator.generate_batch(
                    labels_chunk,
                    snrs_chunk,
                    cfos_chunk,
                    channel_state=state_chunk,
                    generator=gen,
                )

                # 생성한 waveform을 버퍼에 순서대로 쓴다.
                next_ptr = write_ptr + rx_signals.size(0)
                waveforms[write_ptr:next_ptr] = rx_signals.cpu()
                write_ptr = next_ptr

        # 현재 SNR dataset도 예상 샘플 수를 정확히 채웠는지 확인한다.
        if write_ptr != waveforms.size(0):
            raise RuntimeError(f"Waveform dataset generation failed for SNR {snr}.")

        # 해당 SNR의 label/waveform dataset을 dict에 넣는다.
        datasets[snr] = TensorDataset(labels_flat, waveforms)

    return datasets

