#!/usr/bin/env python3
"""
Helper to build a multi-label CSV for NIH ChestX-ray14-style training.

NIH provides `Data_Entry_2017.csv` with `Image Index` and pipe-separated `Finding Labels`.
This script maps those labels to the six project classes (binary columns).

Usage (after downloading NIH metadata):
  python ml_models/preprocess/nih_csv.py \\
    --nih-csv /path/to/Data_Entry_2017.csv \\
    --image-root /path/to/images \\
    --out datasets/chest_xray_nih/train.csv

Note: You must comply with the NIH dataset terms of use.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# Map NIH label tokens to our CHEST_LABELS names
NIH_MAP = {
    "Pneumonia": "pneumonia",
    "Effusion": "effusion",
    "Fibrosis": "fibrosis",
    "Cardiomegaly": "cardiomegaly",
    "Atelectasis": "atelectasis",
    # COVID is not in NIH14 — keep column for external COVID datasets
    "COVID-19": "covid19",
}

ORDER = ["pneumonia", "covid19", "effusion", "fibrosis", "cardiomegaly", "atelectasis"]


def parse_labels(cell: str) -> set[str]:
    parts = [p.strip() for p in cell.split("|")]
    return set(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nih-csv", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.nih_csv.open(newline="", encoding="utf-8") as fin, args.out.open("w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        fieldnames = ["path", *ORDER]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            img = row.get("Image Index") or row.get("Image_Index")
            if not img:
                continue
            labels = parse_labels(row.get("Finding Labels", ""))
            vec = {k: 0 for k in ORDER}
            for token, col in NIH_MAP.items():
                if token in labels and col in vec:
                    vec[col] = 1
            rel = str(Path(img).as_posix())
            writer.writerow({"path": rel, **vec})


if __name__ == "__main__":
    main()
