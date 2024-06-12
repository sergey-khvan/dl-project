import torch
from torch import nn
from torch.nn import functional as F

class NewNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ComparingModel = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=2200, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(pool_size=2),
            nn.LSTM(input_size=64, hidden_size=64, dropout=0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),     
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.ComparingModel(x)
        return F.sigmoid(x)
