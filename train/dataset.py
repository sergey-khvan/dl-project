import pandas as pd
import torch
from torch.utils.data import Dataset
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# THIS PART WAS MADE BY IEGOR
import random

class RandomMutation:
    def __init__(self, mutation_rate=0.01):
        self.mutation_rate = mutation_rate
    
    def __call__(self, sequence):
        mutated_sequence = list(sequence)
        for i in range(len(sequence)):
            if random.random() < self.mutation_rate:
                mutated_sequence[i] = random.choice('ACGT')
        return ''.join(mutated_sequence)

class ReverseComplement:
    def __call__(self, sequence):
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        return ''.join(complement[base] for base in reversed(sequence))

class SequenceShuffle:
    def __init__(self, segment_size=10):
        self.segment_size = segment_size
    
    def __call__(self, sequence):
        segments = [sequence[i:i + self.segment_size] for i in range(0, len(sequence), self.segment_size)]
        random.shuffle(segments)
        return ''.join(segments)
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

class DNADataset(Dataset):
    def __init__(self, csv_dir, transform=None):
        self.dataset = pd.read_csv(csv_dir, header=None)
        self.dataset = self.dataset.drop(0, axis=1)
        # &&&&&&&&&&&&&&&&&&&&&&&&&&&
        # THIS PART WAS MADE BY IEGOR
        self.tranform = transform
        # &&&&&&&&&&&&&&&&&&&&&&&&&&&

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset.iloc[idx]
        code, label = data[1], data[2]
        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        # THIS PART WAS MADE BY IEGOR
        if self.transform:
            code = self.transform(code)
        # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
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
