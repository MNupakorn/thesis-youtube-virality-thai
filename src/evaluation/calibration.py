"""Probability calibration: Platt scaling and isotonic regression.
Calibrators are FIT on the validation set and APPLIED on the test set.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Logistic regression on raw probabilities (or logits) -> calibrated probability."""

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self.model.fit(probs.reshape(-1, 1), y)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(probs.reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.model.fit(probs, y)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        return self.model.predict(probs)


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """ECE: weighted average of |conf - acc| over equal-width bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if not m.any():
            continue
        conf = float(probs[m].mean())
        acc = float(y[m].mean())
        ece += (m.sum() / n) * abs(conf - acc)
    return float(ece)


def reliability_curve(probs: np.ndarray, y: np.ndarray, n_bins: int = 15):
    """Per-bin (mean_conf, mean_acc, count) for plotting reliability diagrams."""
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if not m.any():
            out.append((float((lo + hi) / 2), float("nan"), 0))
            continue
        out.append((float(probs[m].mean()), float(y[m].mean()), int(m.sum())))
    return out
