"""Evaluation metrics utility for flood susceptibility ablation benchmarking."""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

def evaluate_model(y_true, y_pred, y_prob):
    """
    Calculates comprehensive classification performance metrics.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    }
    return metrics


def save_roc_curves(curves, save_path, title):
    """Save one ROC plot. curves is a list of (label, y_true, y_prob)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, y_true, y_prob in curves:
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_prob)
        score = auc(false_positive_rate, true_positive_rate)
        ax.plot(false_positive_rate, true_positive_rate, linewidth=2, label=f"{label} (AUC = {score:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path
