import torch
from torch import nn
from torch.nn import functional as F


class FreqBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=1000, kernel_size=8),
            nn.BatchNorm1d(1000),
            nn.AvgPool1d(293),
            nn.Dropout1d(0.1),
            nn.ReLU(),
            nn.Linear(1, 1),
            nn.Dropout1d(0.1),
        )

    def forward(self, x):
        return self.convblock(x)


class PatternBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=1200, kernel_size=11),
            nn.BatchNorm1d(1200),
            nn.MaxPool1d(290),
            nn.Dropout1d(0.1),
            nn.ReLU(),
            nn.Linear(1, 1),
            nn.Dropout1d(0.1),
        )

    def forward(self, x):
        return self.convblock(x)


class ViraMinerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fq = FreqBranch()
        self.pt = PatternBranch()
        self.fc = nn.Linear(2200, 1)

    def forward(self, x):
        x1 = self.fq(x)
        x2 = self.pt(x)
        x = torch.cat((x1, x2), dim=1)  # dim=1 because 0 is batch idx
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.sigmoid(x)


class FreqNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1000, 1)
        self.freqbranch = FreqBranch()

    def forward(self, x):
        x = self.freqbranch(x)
        x = torch.flatten(x, 1)
        return F.sigmoid(x)


class PatNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1200, 1)
        self.patbranch = PatternBranch()

    def forward(self, x):
        x = self.patbranch(x)
        x = torch.flatten(x, 1)
        return F.sigmoid(x)
