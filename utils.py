"""여러 파일에서 공통으로 사용하는 유틸리티 함수 모음이다.

이 파일에는 다음과 같은 성격의 함수가 들어 있다.

- 재현성을 위한 seed 고정
- CFO 범위 계산
- 모델 파라미터 수 계산
- 중첩된 설정 딕셔너리 병합
- CPU/GPU 상태를 고려한 간단한 benchmark
"""

import copy
import os
import time
from typing import Callable, Dict

import numpy as np
import torch


def set_seed(seed: int = 2026) -> None:
    """NumPy와 PyTorch의 난수 시드를 고정한다.

    학습/평가 결과가 seed에 따라 달라질 수 있으므로,
    비교 실험을 할 때는 동일한 seed를 사용하는 것이 중요하다.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_max_cfo_hz(simulator, channel_profile: Dict) -> float:
    """채널 프로파일에 정의된 CFO 범위를 Hz 단위로 환산한다."""

    return channel_profile["max_cfo_bins"] * (simulator.bw / simulator.M)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """학습 가능한 파라미터 수만 합산해 반환한다."""

    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def move_to_cpu(value):
    """중첩된 자료구조 안의 텐서를 모두 CPU로 옮긴다.

    체크포인트 저장 시 optimizer state가 GPU에 남아 있으면 저장과 로드가 번거로워질 수 있다.
    따라서 저장 직전에 재귀적으로 CPU로 내린다.
    """

    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: move_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_cpu(item) for item in value)
    return value


def merge_config(base_config: Dict, overrides: Dict = None) -> Dict:
    """기본 설정 위에 override 설정을 재귀적으로 덮어쓴다."""

    merged = copy.deepcopy(base_config)
    if overrides is None:
        return merged

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def flatten_summary_columns(frame):
    """pandas의 다중 컬럼 이름을 단일 문자열 컬럼으로 평탄화한다."""

    frame.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in frame.columns.values
    ]
    return frame


def sync_device(device: torch.device) -> None:
    """CUDA 환경에서는 정확한 시간 측정을 위해 동기화를 수행한다."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_callable(
    fn: Callable[[], None],
    device: torch.device,
    warmup: int = 2,
    repeats: int = 6,
) -> float:
    """호출 가능한 객체의 평균 실행 시간을 ms 단위로 측정한다."""

    for _ in range(warmup):
        fn()
    sync_device(device)

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync_device(device)
        timings.append((time.perf_counter() - start) * 1000.0)

    return float(np.mean(timings))
