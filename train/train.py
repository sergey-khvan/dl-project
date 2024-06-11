import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import yaml
from addict import Dict
from sklearn.metrics import roc_auc_score

from dataset import DNADataset
from model import ViraMinerNet

from torchvisions.transforms import Compose # ADDED BY IEGOR

def get_loader(dataset, batch_size):
    class_weights = [1 / 30, 1]

    sample_weights = [0] * len(dataset)
    for idx, (data, label) in enumerate(dataset):
        class_weight = class_weights[int(label)]
        sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return loader


def main():
    with open("../config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    cfg = Dict(cfg)

    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # PART MADE BY IEGOR
    transform = Compose([
        RandomMutation(mutation_rate=0.05),
        ReverseComplement(),
        SequenceShuffle(segment_size=5)
    ])
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    
    train_dataset = DNADataset(cfg.train_dir, transform) # ADDED "transform" by IEGOR
    validation_dataset = DNADataset(cfg.val_dir)
    test_dataset = DNADataset(cfg.test_dir)

    train_loader = get_loader(train_dataset, batch_size=cfg.batch_size)
    val_loader = DataLoader(validation_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)

    model = ViraMinerNet()
    model.to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.lr)

    if cfg.load_from_checkpoint:
        checkpoint = torch.load(cfg.checkpoint_dir)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for epoch in range(cfg.num_epochs):
        # Training
        train_run_loss = 0
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for idx, (x, y) in loop:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = loss_fn(preds, y)
            train_run_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch:[{epoch}/{cfg.num_epochs}](Train)")
            loop.set_postfix(train_loss=loss.item())
        train_loss = train_run_loss / (idx + 1)
        print("Train loss: ", train_loss)
        # Validation
        val_run_loss = 0
        model.eval()
        all_predictions = []
        all_labels = []
        loop = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
        with torch.no_grad():
            for idx, (x,y) in loop:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
                loss = loss_fn(preds, y)
                val_run_loss += loss

                loop.set_description(f"Epoch:[{epoch}/{cfg.num_epochs}](Val)")
                loop.set_postfix(val_loss=loss.item())
                
        val_loss = val_run_loss / (idx + 1)
        val_roc_score = roc_auc_score(all_labels, all_predictions)
        print("Val loss: ", val_loss)
        print("ROC score: ",val_roc_score)

        if epoch % 20 == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                cfg.checkpoint_dir,
            )


if __name__ == "__main__":
    main()
