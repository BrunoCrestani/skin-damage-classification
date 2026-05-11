from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .config import CLASSES


def evaluate_predictions(y_true, y_pred, output_dir: Path, prefix: str) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }

    report = classification_report(y_true, y_pred, labels=CLASSES, zero_division=0)
    (output_dir / f"{prefix}_classification_report.txt").write_text(report)
    pd.DataFrame([metrics]).to_csv(output_dir / f"{prefix}_metrics.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    cm_df.to_csv(output_dir / f"{prefix}_confusion_matrix.csv")
    plot_confusion_matrix(cm, output_dir / f"{prefix}_confusion_matrix.png", prefix)
    return metrics


def plot_confusion_matrix(cm: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(CLASSES)),
        yticks=np.arange(len(CLASSES)),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
