import torch
import torch.nn as nn
import torch.nn.functional as F

class Hypothesis2DCNN(nn.Module):
    def __init__(self, num_classes: int, num_hypotheses: int = 153, num_bins: int = 640, in_channels: int = 2):
        super(Hypothesis2DCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )
        
        self.flatten = nn.Flatten()
        
        with torch.no_grad():
            dummy_x = torch.zeros(1, in_channels, num_hypotheses, num_bins)
            dummy_out = self.features(dummy_x)
            flattened_size = dummy_out.view(1, -1).size(1)
            
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), 
            nn.Linear(512, num_classes) 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x