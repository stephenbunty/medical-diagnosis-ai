#!/usr/bin/env python3
"""
Train multi-label chest X-ray classifier (NIH ChestX-ray14-style labels).

Expects a CSV `train.csv` with columns: path,label0,label1,... (binary 0/1 per class)
or a folder structure: train/<class_name>/*.png (converted to multi-hot in preprocessing).

This script demonstrates:
- Transfer learning with ResNet50 / optional VGG16
- BCEWithLogitsLoss
- AdamW + cosine schedule
- Early stopping + checkpoint best weights
- TensorBoard logging

Run from project root:
  cd medical-diagnosis-ai
  python ml_models/train_chest_multilabel.py --data-dir datasets/chest_xray_sample

For NIH ChestX-ray14, preprocess with `ml_models/preprocess/nih_csv.py`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.architectures import CHEST_LABELS, MultiLabelChestNet  # noqa: E402
from app.ml.preprocess import chest_transforms_train  # noqa: E402


class ChestCSVDataset(Dataset):
    """CSV columns: path + one column per label in CHEST_LABELS order."""

    def __init__(self, csv_path: Path, root: Path, tfm):
        self.rows: list[tuple[Path, list[float]]] = []
        self.root = root
        self.tfm = tfm
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = self.root / row["path"]
                y = [float(row[name]) for name in CHEST_LABELS]
                self.rows.append((p, y))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        p, y = self.rows[idx]
        img = Image.open(p).convert("RGB")
        return self.tfm(img), torch.tensor(y, dtype=torch.float32)


def make_dummy_dataset(out_dir: Path, n: int = 32) -> Path:
    """Create tiny random dataset + CSV for smoke-testing training."""
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "train.csv"
    rows = []
    for i in range(n):
        arr = torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8).numpy()
        fp = img_dir / f"img_{i}.png"
        Image.fromarray(arr).save(fp)
        labels = torch.randint(0, 2, (len(CHEST_LABELS),)).tolist()
        rows.append({"path": f"images/img_{i}.png", **{CHEST_LABELS[j]: labels[j] for j in range(len(CHEST_LABELS))}})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", *CHEST_LABELS])
        w.writeheader()
        w.writerows(rows)
    return csv_path


def train_one_epoch(model, loader, opt, criterion, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0)
    return total / max(len(loader.dataset), 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total += loss.item() * x.size(0)
    return total / max(len(loader.dataset), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "datasets" / "chest_xray_sample")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone", choices=["resnet50", "vgg16"], default="resnet50")
    ap.add_argument("--patience", type=int, default=3)
    args = ap.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.data_dir / "train.csv"
    if not csv_path.exists():
        print("No train.csv found — generating dummy data for demo training.")
        csv_path = make_dummy_dataset(args.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = chest_transforms_train(224)
    full_ds = ChestCSVDataset(csv_path, args.data_dir, tfm)
    n = len(full_ds)
    n_train = max(int(n * 0.8), 1)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n - n_train])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MultiLabelChestNet(num_labels=len(CHEST_LABELS), backbone=args.backbone).to(device)
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt_dir = ROOT / "ml_models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(ROOT / "ml_models" / "runs" / "chest"))
    best = float("inf")
    bad_epochs = 0

    for epoch in range(args.epochs):
        tr = train_one_epoch(model, train_loader, opt, criterion, device)
        va = eval_epoch(model, val_loader, criterion, device)
        writer.add_scalars("loss", {"train": tr, "val": va}, epoch)
        print(f"Epoch {epoch+1}/{args.epochs} train_loss={tr:.4f} val_loss={va:.4f}")
        if va < best - 1e-4:
            best = va
            bad_epochs = 0
            torch.save(model.state_dict(), ckpt_dir / "chest_resnet50.pt")
            print("  saved best checkpoint")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break
    writer.close()
    print("Done. Weights at", ckpt_dir / "chest_resnet50.pt")


if __name__ == "__main__":
    main()
