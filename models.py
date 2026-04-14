"""LoRa hypothesis bank를 입력으로 받아 심볼 클래스를 예측하는 2D CNN 정의 파일이다."""

import torch
import torch.nn as nn

from config import CFG


class Hypothesis2DCNN(nn.Module):
    """다중 CFO / timing hypothesis feature bank를 분류하는 CNN이다."""

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
        if num_bins is None:
            num_bins = (2 ** CFG["receiver_profiles"][0]["sf"]) * CFG["feature_bank"]["patch_size"]

        model_cfg = CFG.get("model", {})
        width_scale = model_cfg.get("width_scale", 1.0) if width_scale is None else width_scale
        classifier_hidden = model_cfg.get("classifier_hidden", 256) if classifier_hidden is None else classifier_hidden
        dropout = model_cfg.get("dropout", 0.3) if dropout is None else dropout

        if stage_channels is None:
            base_channels = model_cfg.get("stage_channels", [32, 64, 96, 128])
            # width_scale을 이용해 프로파일별로 CNN 폭을 조절한다.
            stage_channels = [
                max(8, int(round((channel * width_scale) / 8.0) * 8))
                for channel in base_channels
            ]

        if len(stage_channels) != 4:
            raise ValueError("stage_channels must contain exactly four entries.")

        c1, c2, c3, c4 = stage_channels
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

        # hypothesis 수와 patch 길이가 프로파일마다 달라지므로,
        # AdaptiveAvgPool로 classifier 입력 크기를 일정하게 맞춘다.
        self.pool = nn.AdaptiveAvgPool2d((4, 16))

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, num_hypotheses, num_bins)
            flattened_size = self.pool(self.features(dummy)).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x):
        """특징 추출기와 분류기를 순서대로 통과시켜 logits를 반환한다."""

        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
