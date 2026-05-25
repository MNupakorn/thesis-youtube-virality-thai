"""Statistical tests for comparing classifier predictions on the same test set.

- McNemar's test: pairwise comparison of two classifiers' error patterns.
- Cochran's Q test: extension of McNemar to k>=3 classifiers.
- Paired bootstrap test: difference in any metric, with empirical CI.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar


def mcnemar_pairwise(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    threshold: float = 0.5,
    correction: bool = True,
) -> pd.DataFrame:
    """Run McNemar's test for every pair of classifiers in `preds`.

    `preds[name]` is the (N,) probability vector for class 1; we binarize at `threshold`.
    """
    rows = []
    correct = {
        name: ((p >= threshold).astype(int) == y_true).astype(int) for name, p in preds.items()
    }
    for a, b in combinations(preds.keys(), 2):
        ca = correct[a]
        cb = correct[b]
        n10 = int(((ca == 1) & (cb == 0)).sum())  # a right, b wrong
        n01 = int(((ca == 0) & (cb == 1)).sum())  # a wrong, b right
        table = np.array([[0, n10], [n01, 0]])
        try:
            res = mcnemar(table, exact=False, correction=correction)
            stat = float(res.statistic)
            p = float(res.pvalue)
        except Exception:
            stat, p = float("nan"), float("nan")
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "n_a_right_b_wrong": n10,
                "n_a_wrong_b_right": n01,
                "stat": stat,
                "p_value": p,
            }
        )
    return pd.DataFrame(rows)


def cochrans_q_test(
    y_true: np.ndarray, preds: dict[str, np.ndarray], threshold: float = 0.5
) -> dict[str, float]:
    """Cochran's Q across k>=3 classifiers."""
    correct = np.column_stack(
        [((p >= threshold).astype(int) == y_true).astype(int) for p in preds.values()]
    )
    res = cochrans_q(correct)
    return {
        "stat": float(res.statistic),
        "p_value": float(res.pvalue),
        "df": int(res.df),
        "n_classifiers": correct.shape[1],
    }


def paired_bootstrap_diff(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    metric_fn,
    n_iter: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Empirical CI for (metric(a) - metric(b)) on the test set."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        try:
            v_a = metric_fn(y_true[idx], proba_a[idx])
            v_b = metric_fn(y_true[idx], proba_b[idx])
            diffs.append(v_a - v_b)
        except Exception:
            continue
    arr = np.array(diffs)
    return {
        "diff_mean": float(arr.mean()),
        "diff_lo": float(np.quantile(arr, alpha / 2)),
        "diff_hi": float(np.quantile(arr, 1 - alpha / 2)),
        "p_le_zero": float((arr <= 0).mean()),
        "p_ge_zero": float((arr >= 0).mean()),
    }
