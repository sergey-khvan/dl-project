import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DNADataset
from model import ViraMinerNet


num_epochs = 10
batch_size = 1
lr = 1e-4

device = "mps" if torch.backends.mps.is_available() else "cpu"

train_dataset = DNADataset("data/fullset_train.csv")
test_dataset = DNADataset("data/fullset_test.csv")
validation_dataset = DNADataset("data/fullset_validation.csv")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


model = ViraMinerNet()
model.to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

for epoch in range(num_epochs):
    run_loss = 0
    model.train()
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for idx, (x, y) in loop:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        loss = loss_fn(preds, y)
        run_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch:[{epoch}/{num_epochs}](Train)")
        loop.set_postfix(train_loss=loss.item())
