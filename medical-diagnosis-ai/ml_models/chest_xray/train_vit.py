#!/usr/bin/env python3
"""
Optional Vision Transformer fine-tuning using `timm`.

Install: pip install timm
Run:
  python ml_models/chest_xray/train_vit.py --data-dir datasets/chest_xray_sample

This is a template: wire your CSV/ImageFolder similar to `train_chest_multilabel.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "datasets" / "chest_xray_sample")
    args = ap.parse_args()
    try:
        import timm  # type: ignore
    except ImportError:
        print("Please `pip install timm` to run ViT training.")
        sys.exit(1)

    _ = timm
    _ = args
    print("ViT training template ready — extend with your dataloader and multi-label head.")


if __name__ == "__main__":
    main()
