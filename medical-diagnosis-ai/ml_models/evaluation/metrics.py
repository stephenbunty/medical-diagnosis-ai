"""Common metrics for classification and segmentation evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def multilabel_auroc(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Per-label AUROC where each column is a binary task."""
    out: dict[str, float] = {}
    for j in range(y_true.shape[1]):
        try:
            out[f"label_{j}"] = float(roc_auc_score(y_true[:, j], y_score[:, j]))
        except ValueError:
            out[f"label_{j}"] = float("nan")
    return out


def multilabel_map(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score, average="macro"))


def dice_coefficient(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    return float((2 * inter + eps) / (pred.sum() + target.sum() + eps))


def plot_confusion(y_true, y_pred, labels: list[str], out_path):
    """Save a confusion matrix PNG (requires matplotlib)."""
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
