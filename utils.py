import os
import numpy as np
import torch



def set_seed(seed: int = 2026) -> None:
    """
    실행 재현성을 높이기 위해 난수 시드를 고정하는 함수이다.

    이 함수는 다음 항목들을 함께 고정한다.
    1) NumPy 난수
    2) PyTorch CPU 난수
    3) PyTorch CUDA 난수
    4) Python hash seed
    5) cuDNN 동작 옵션

    이렇게 해야 같은 코드를 같은 seed로 실행했을 때
    가능한 한 비슷한 결과가 다시 나오도록 만들 수 있다.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    # deterministic=True로 설정하면 가능한 한 같은 결과를 재현하도록 유도한다.
    torch.backends.cudnn.deterministic = True

    # benchmark=False로 설정하면 입력 형태에 따라 매번 다른 최적 커널을 고르는 동작을 막는다.
    torch.backends.cudnn.benchmark = False
