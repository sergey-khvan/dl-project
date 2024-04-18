import pandas as pd
import torch
from torch.utils.data import Dataset


class DNADataset(Dataset):
    def __init__(self, csv_dir):
        self.dataset = pd.read_csv(csv_dir, header=None)
        self.dataset.drop(0, axis=1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset.iloc[idx]
        code, label = data[0], data[1]
        code = self.encode(code)
        label = torch.Tensor([label])
        return code, label

    def encode(self, code):
        onehot_map = {
            "A": [1, 0, 0, 0, 0],
            "C": [0, 1, 0, 0, 0],
            "G": [0, 0, 1, 0, 0],
            "T": [0, 0, 0, 1, 0],
            "N": [0, 0, 0, 0, 1],
        }
        encoded = []
        for s in code:
            encoded.append(onehot_map[s])
        return torch.Tensor(encoded).T
