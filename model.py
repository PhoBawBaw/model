import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

# CNN 모델 정의
class CNNClassifier(nn.Module):
    def __init__(self, input_shape, num_label):
        super(CNNClassifier, self).__init__()
        self.norm_layer = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()

        # CNN 레이어 이후 출력 크기 계산
        with torch.no_grad():
            sample_input = torch.zeros(1, *input_shape).unsqueeze(0)
            sample_output = self._forward_features(sample_input)
            flatten_size = sample_output.numel()

        self.fc1 = nn.Linear(flatten_size, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_label)

    def _forward_features(self, x):
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        x = self.norm_layer(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x