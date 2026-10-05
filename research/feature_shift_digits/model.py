"""Classifier architecture of med-air/FedBN nets/models.py::DigitModel.

Three convolutional and three fully connected layers, with shared BN. The
appendix Table 3 accidentally repeats the third convolutional row; the
executable author implementation has three convolutions, reproduced here.
"""
import torch
from torch import nn
from torch.nn import functional as F


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(6272, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (3, 28, 28):
            raise ValueError('DigitCNN requires N x 3 x 28 x 28')
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.fc1(x.reshape(x.shape[0], -1))))
        x = F.relu(self.bn5(self.fc2(x)))
        return self.fc3(x)
