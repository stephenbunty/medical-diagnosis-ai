#!/usr/bin/env python3
"""
Train 4-class brain MRI classifier (glioma / meningioma / pituitary / normal).

Expected folder layout (Kaggle Brain Tumor MRI style):
  data_dir/
    train/
      glioma/*.jpg
      meningioma/*.jpg
      pituitary/*.jpg
      normal/*.jpg

Run:
  python ml_models/train_brain_mri.py --data-dir datasets/brain_mri_sample
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.architectures import BRAIN_LABELS, BrainMRINet  # noqa: E402
from app.ml.preprocess import brain_transforms_train  # noqa: E402


def make_dummy_brain(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    train = out / "train"
    for cls in BRAIN_LABELS:
        d = train / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(8):
            arr = torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8).numpy()
            Image.fromarray(arr).save(d / f"{i}.png")
    return train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "datasets" / "brain_mri_sample")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    train_dir = args.data_dir / "train"
    if not train_dir.exists():
        print("Creating dummy brain MRI folder structure…")
        train_dir = make_dummy_brain(args.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = brain_transforms_train(224)
    ds: Dataset = datasets.ImageFolder(str(train_dir), transform=tfm)
    # remap class order to BRAIN_LABELS if names match
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = BrainMRINet(num_classes=len(BRAIN_LABELS)).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt_dir = ROOT / "ml_models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(ROOT / "ml_models" / "runs" / "brain"))

    for epoch in range(args.epochs):
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
        avg = total / max(len(loader.dataset), 1)
        writer.add_scalar("loss/train", avg, epoch)
        print(f"Epoch {epoch+1}/{args.epochs} loss={avg:.4f}")

    torch.save(model.state_dict(), ckpt_dir / "brain_resnet18.pt")
    writer.close()
    print("Saved", ckpt_dir / "brain_resnet18.pt")


if __name__ == "__main__":
    main()
