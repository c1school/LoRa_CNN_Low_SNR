import os
import numpy as np
import torch


def set_seed(seed: int = 2026) -> None:
    """
    실험 재현성을 확보하기 위한 seed 고정 함수.

    같은 코드/같은 설정으로 다시 실행했을 때 결과가 크게 달라지지 않도록
    numpy, torch, CUDA, hash seed, cudnn 동작을 고정함.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
