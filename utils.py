"""프로젝트 여러 파일에서 공통으로 쓰는 작은 보조 함수들을 모아 둔 파일이다."""

import copy
import random
import time
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int = 2026) -> None:
    """파이썬, NumPy, PyTorch 난수 시드를 함께 고정한다."""

    # random 모듈 시드 고정
    random.seed(seed)
    # NumPy 시드 고정
    np.random.seed(seed)
    # CPU용 PyTorch 시드 고정
    torch.manual_seed(seed)

    # CUDA가 있으면 GPU 시드도 함께 고정한다.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_max_cfo_hz(simulator, channel_profile: Dict) -> float:
    """채널 프로파일의 max_cfo_bins를 실제 Hz 단위 CFO 최대값으로 변환한다."""

    return channel_profile["max_cfo_bins"] * simulator.bw / simulator.M


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """학습 가능한 파라미터 수를 센다."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def move_to_cpu(value):
    """중첩된 dict/list/tuple 안의 텐서를 재귀적으로 CPU로 내린다."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: move_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_cpu(item) for item in value)
    return value


def merge_config(base_config: Dict, overrides: Dict = None) -> Dict:
    """기본 설정 위에 override를 덮어쓴 새 dict를 만든다."""

    # base_config를 직접 수정하지 않기 위해 deep copy를 만든다.
    merged = copy.deepcopy(base_config)
    if overrides is None:
        return merged

    # override dict를 순회하며 값을 덮어쓴다.
    for key, value in overrides.items():
        # 하위 dict끼리 병합해야 하면 재귀적으로 merge_config를 다시 호출한다.
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            # 그렇지 않으면 override 값을 그대로 대입한다.
            merged[key] = copy.deepcopy(value)
    return merged


def flatten_summary_columns(frame):
    """pandas multi-index 컬럼을 단일 문자열 컬럼으로 납작하게 편다."""

    flattened = []
    for column in frame.columns:
        if isinstance(column, tuple):
            # groupby-agg 결과는 ('ser_single', 'mean') 같은 튜플 컬럼을 만들므로
            # 이를 ser_single_mean처럼 이어 붙인다.
            flattened.append("_".join(str(item) for item in column if item))
        else:
            flattened.append(column)
    frame.columns = flattened
    return frame


def sync_device(device: torch.device) -> None:
    """CUDA 장치라면 비동기 연산이 끝날 때까지 동기화한다."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_callable(fn, repeats: int = 10, warmup: int = 2, device: torch.device = None):
    """함수를 여러 번 실행해 평균 실행 시간을 측정한다."""

    # warmup:
    # 캐시와 커널 초기화를 위해 먼저 몇 번 실행한다.
    for _ in range(warmup):
        fn()
        if device is not None:
            sync_device(device)

    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        if device is not None:
            sync_device(device)
        end = time.perf_counter()
        durations.append((end - start) * 1000.0)

    return float(np.mean(durations))
