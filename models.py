"""LoRa hypothesis feature bank를 입력으로 받아 심볼 클래스를 분류하는 CNN 정의 파일이다."""

import torch
import torch.nn as nn

from config import CFG


class Hypothesis2DCNN(nn.Module):
    """여러 CFO / timing hypothesis를 펼쳐 놓은 feature bank를 분류하는 CNN이다."""

    def __init__(
        self,
        num_classes: int,
        num_hypotheses: int = CFG["feature_bank"]["cfo_steps"] * CFG["feature_bank"]["to_steps"],
        num_bins: int = None,
        in_channels: int = 2,
        stage_channels=None,
        classifier_hidden: int = None,
        dropout: float = None,
        width_scale: float = None,
    ):
        super().__init__()

        # num_bins를 따로 주지 않으면
        # 첫 번째 receiver_profile의 기본 SF와 patch_size를 이용해 기본값을 만든다.
        if num_bins is None:
            num_bins = (2 ** CFG["receiver_profiles"][0]["sf"]) * CFG["feature_bank"]["patch_size"]

        # model_cfg:
        # config.py에 들어 있는 CNN 기본 설정이다.
        model_cfg = CFG.get("model", {})

        # width_scale / classifier_hidden / dropout은 함수 인자로 덮어쓸 수 있다.
        width_scale = model_cfg.get("width_scale", 1.0) if width_scale is None else width_scale
        classifier_hidden = model_cfg.get("classifier_hidden", 256) if classifier_hidden is None else classifier_hidden
        dropout = model_cfg.get("dropout", 0.3) if dropout is None else dropout

        # stage_channels를 직접 주지 않으면 config 기본 stage를 가져온다.
        if stage_channels is None:
            base_channels = model_cfg.get("stage_channels", [32, 64, 96, 128])

            # width_scale을 적용해 실제 합성곱 채널 수를 만든다.
            # 8의 배수에 맞춰 반올림하는 이유는 메모리 정렬과 구현 편의를 위한 것이다.
            stage_channels = [
                max(8, int(round((channel * width_scale) / 8.0) * 8))
                for channel in base_channels
            ]

        # 이 모델은 4개 stage를 전제로 설계했으므로 길이가 다르면 에러를 낸다.
        if len(stage_channels) != 4:
            raise ValueError("stage_channels must contain exactly four entries.")

        # c1 ~ c4는 각 stage의 출력 채널 수다.
        c1, c2, c3, c4 = stage_channels

        # features:
        # 입력 hypothesis bank에서 지역적인 패턴을 추출하는 feature extractor다.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=5, padding=2),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c3, c4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )

        # pool:
        # hypothesis 수와 patch 길이가 profile마다 달라도
        # classifier 입력 차원을 고정하기 위해 adaptive average pooling을 사용한다.
        self.pool = nn.AdaptiveAvgPool2d((4, 16))

        # 더미 입력을 한 번 통과시켜 classifier 입력 차원을 계산한다.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, num_hypotheses, num_bins)
            flattened_size = self.pool(self.features(dummy)).view(1, -1).size(1)

        # classifier:
        # convolution feature를 flatten한 뒤 최종 심볼 클래스를 출력한다.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x):
        """입력 feature bank를 받아 클래스 logits를 반환한다."""

        # 먼저 convolution feature extractor를 통과시킨다.
        x = self.features(x)
        # 그 다음 adaptive pooling으로 크기를 고정한다.
        x = self.pool(x)
        # 마지막으로 classifier를 통과시켜 logits를 만든다.
        return self.classifier(x)
