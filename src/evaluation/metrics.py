"""Classification metrics + bootstrap CIs."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Standard binary classification metrics + threshold-free AUCs."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_pos": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_pos": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_pos": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_proba))
        if len(np.unique(y_true)) > 1
        else float("nan"),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "roc_auc",
    n_iter: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Returns (point_estimate, lower, upper) for the chosen metric via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = compute_metrics(y_true, y_proba, threshold)[metric]

    samples = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        try:
            v = compute_metrics(y_true[idx], y_proba[idx], threshold)[metric]
            if not np.isnan(v):
                samples.append(v)
        except Exception:
            continue
    if not samples:
        return point, float("nan"), float("nan")
    arr = np.array(samples)
    lo = float(np.quantile(arr, alpha / 2))
    hi = float(np.quantile(arr, 1 - alpha / 2))
    return point, lo, hi


def find_best_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, metric: str = "f1_pos", n_grid: int = 201
) -> tuple[float, float]:
    """Sweep thresholds on validation set; return (best_threshold, best_metric_value)."""
    grid = np.linspace(0.01, 0.99, n_grid)
    best_v = -np.inf
    best_t = 0.5
    for t in grid:
        v = compute_metrics(y_true, y_proba, threshold=float(t))[metric]
        if v > best_v:
            best_v = v
            best_t = float(t)
    return best_t, float(best_v)
