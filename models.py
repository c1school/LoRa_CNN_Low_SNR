import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRaCNN(nn.Module):
    def __init__(self, num_classes: int, input_length: int, in_channels: int = 2):
        super(LoRaCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2, 2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2, 2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2, 2)

        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)

        with torch.no_grad():
            dummy_x = torch.zeros(1, in_channels, input_length)
            dummy_x = self.pool1(F.relu(self.bn1(self.conv1(dummy_x))))
            dummy_x = self.pool2(F.relu(self.bn2(self.conv2(dummy_x))))
            dummy_x = self.pool3(F.relu(self.bn3(self.conv3(dummy_x))))
            dummy_x = F.relu(self.bn4(self.conv4(dummy_x)))
            flattened_size = dummy_x.view(1, -1).size(1)

        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x