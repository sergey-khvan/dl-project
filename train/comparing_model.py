import torch
from torch import nn
from torch.nn import functional as F

class NewNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 =  nn.Conv1d(in_channels=5, out_channels=64, kernel_size=5)
        self.mxpool =  nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2,  batch_first=True)
        self.dropout = nn.Dropout1d(0.2)
        self.fc1 = nn.Linear(128*148, 64)  
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.mxpool(x)
        x = x.permute(0,2,1)
        x, _ = self.lstm(x)
        x = x.permute(0,2,1)
        x = self.dropout(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.sigmoid(x)
