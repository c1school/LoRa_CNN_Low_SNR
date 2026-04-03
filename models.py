import torch
import torch.nn as nn
import torch.nn.functional as F

class Hypothesis2DCNN(nn.Module):
    def __init__(self, num_classes: int, num_hypotheses: int = 153, num_bins: int = 128, in_channels: int = 2):
        super(Hypothesis2DCNN, self).__init__()
        
        # 입력 차원: [Batch, 2 (Real/Imag), 153 (Hypotheses), 128 (Bins)]
        
        self.features = nn.Sequential(
            # Block 1: 미세한 국소 특징(Local Phase/Amplitude) 추출
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 대략 [Batch, 32, 76, 64]
            
            # Block 2: 가설 간의 상관관계(Correlation) 학습
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 대략 [Batch, 64, 38, 32]
            
            # Block 3: 고차원 추상화 패턴 인식
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 대략 [Batch, 128, 19, 16]
            
            # Block 4: 최종 압축
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 대략 [Batch, 128, 9, 8]
        )
        
        self.flatten = nn.Flatten()
        
        # 동적 차원 계산 (가설 개수나 심볼 수가 바뀌어도 에러가 나지 않도록 방어적 설계)
        with torch.no_grad():
            dummy_x = torch.zeros(1, in_channels, num_hypotheses, num_bins)
            dummy_out = self.features(dummy_x)
            flattened_size = dummy_out.view(1, -1).size(1)
            
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # 과적합 방지
            nn.Linear(512, num_classes) # 최종 출력: 128개 심볼에 대한 Logit
        )

    def forward(self, x):
        # x 형태: [Batch, 2, 153, 128]
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x