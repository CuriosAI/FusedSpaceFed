"""FedAvg control: same CNN with the first FC layer widened by 17 units."""
from torch import nn
from research.feature_shift_digits.model import DigitCNN


class EnlargedDigitCNN(DigitCNN):
    def __init__(self):
        super().__init__()
        # Common-shape layers retain the same initial draws as DigitCNN.
        self.fc1 = nn.Linear(6272, 2065)
        self.bn4 = nn.BatchNorm1d(2065)
        self.fc2 = nn.Linear(2065, 512)
