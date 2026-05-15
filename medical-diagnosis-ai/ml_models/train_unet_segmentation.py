#!/usr/bin/env python3
"""
Train compact U-Net for binary masks (tumor / abnormality).

Provide paired images + masks (PNG, white=foreground):
  data_dir/
    images/*.png
    masks/*.png  (same basename)

If no dataset is present, synthetic random data is generated for a smoke test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.architectures import UNet  # noqa: E402


def _pil_to_tensor01(img: Image.Image, size: int = 256) -> torch.Tensor:
    img = img.resize((size, size)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return (t - 0.45) / 0.25


def _pil_mask01(img: Image.Image, size: int = 256) -> torch.Tensor:
    img = img.resize((size, size)).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class PairDataset(Dataset):
    def __init__(self, root: Path):
        self.root = root
        self.ids = sorted({p.stem for p in (root / "images").glob("*")})

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        sid = self.ids[idx]
        img = Image.open(self.root / "images" / f"{sid}.png")
        mask = Image.open(self.root / "masks" / f"{sid}.png")
        return _pil_to_tensor01(img), _pil_mask01(mask)


def make_synthetic(out: Path, n: int = 16) -> None:
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        base = torch.randint(0, 255, (256, 256, 3), dtype=torch.uint8).numpy()
        Image.fromarray(base).save(out / "images" / f"{i:04d}.png")
        m = (base[:, :, 0] > 200).astype("uint8") * 255
        Image.fromarray(m).save(out / "masks" / f"{i:04d}.png")


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - (2 * inter + eps) / (union + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "datasets" / "unet_sample")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    if not (args.data_dir / "images").exists():
        make_synthetic(args.data_dir)

    ds = PairDataset(args.data_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_ch=3, base=32).to(device)
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    writer = SummaryWriter(log_dir=str(ROOT / "ml_models" / "runs" / "unet"))
    ckpt_dir = ROOT / "ml_models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = bce(logits, y) + dice_loss(logits, y).mean()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        avg = total / max(len(loader.dataset), 1)
        writer.add_scalar("loss/train", avg, epoch)
        print(f"Epoch {epoch+1}/{args.epochs} loss={avg:.4f}")

    torch.save(model.state_dict(), ckpt_dir / "unet_segmentation.pt")
    writer.close()
    print("Saved", ckpt_dir / "unet_segmentation.pt")


if __name__ == "__main__":
    main()
