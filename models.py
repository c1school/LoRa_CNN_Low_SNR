import torch
import torch.nn as nn
from config import CFG


class Hypothesis2DCNN(nn.Module):
    """
    다중 가설 2차원 특징맵을 입력으로 받아 LoRa 심볼을 분류하는 모델이다.

    입력 텐서의 기본 형태는 다음과 같다.
    [Batch, 2, Num_Hypotheses, Num_Bins]

    각 차원의 의미는 다음과 같다.
    - Batch          : 한 번에 처리하는 샘플 수이다.
    - 2              : 실수부와 허수부 채널이다.
    - Num_Hypotheses : CFO / Timing Offset 가설의 총개수이다.
    - Num_Bins       : 심볼 중심 주변 patch까지 포함한 주파수 방향 길이이다.

    이 모델은 2D CNN을 이용해
    "가설 축 방향 패턴"과 "주파수 축 방향 패턴"을 동시에 읽도록 설계하였다.
    """

    def __init__(
        self,
        num_classes: int,
        num_hypotheses: int = CFG["cfo_steps"] * CFG["to_steps"],
        num_bins: int = (2 ** CFG["sf"]) * CFG["patch_size"],
        in_channels: int = 2,
    ):
        super(Hypothesis2DCNN, self).__init__()

        # ------------------------------------------------------------
        # 특징 추출부
        # ------------------------------------------------------------
        # Conv2d를 사용하여
        # 1) 가설들 사이의 공간적 패턴
        # 2) 주파수 주변 patch의 지역적 패턴
        # 을 동시에 학습하도록 하였다.
        self.features = nn.Sequential(
            # 첫 번째 블록이다.
            # 가장 원시적인 지역 특징을 추출한다.
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 두 번째 블록이다.
            # 더 넓은 범위의 상관관계를 학습한다.
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 세 번째 블록이다.
            # 보다 추상적인 패턴을 추출한다.
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 네 번째 블록이다.
            # 마지막으로 특징을 압축하고 분류기 입력으로 넘긴다.
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 합성곱 결과를 일렬로 펴기 위한 계층이다.
        self.flatten = nn.Flatten()

        # ------------------------------------------------------------
        # 출력 차원 자동 계산
        # ------------------------------------------------------------
        # 입력 크기가 설정값에 따라 달라질 수 있으므로,
        # 더미 텐서를 한 번 통과시켜 flatten 차원을 자동 계산하였다.
        with torch.no_grad():
            dummy_x = torch.zeros(1, in_channels, num_hypotheses, num_bins)
            dummy_out = self.features(dummy_x)
            flattened_size = dummy_out.view(1, -1).size(1)

        # ------------------------------------------------------------
        # 분류기
        # ------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # 1) 2D CNN으로 특징을 추출한다.
        x = self.features(x)

        # 2) 선형 분류기로 넘기기 위해 펼친다.
        x = self.flatten(x)

        # 3) 최종 심볼 logits를 계산한다.
        x = self.classifier(x)
        return x
