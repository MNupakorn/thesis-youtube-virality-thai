"""SHAP explanations on the LightGBM head over hybrid features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import setup_logger

log = setup_logger("explain.shap")


def explain_with_shap(
    model,
    X_test: np.ndarray,
    feature_names: list[str],
    out_dir: str | Path,
    max_samples: int = 2000,
    seed: int = 42,
) -> dict:
    """Compute SHAP values for a tree model. Saves summary plot + global importance CSV."""
    import shap

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    n = X_test.shape[0]
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        Xs = X_test[idx]
    else:
        Xs = X_test

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(Xs)
    if isinstance(shap_vals, list):
        # binary classifier: take class-1 contributions
        shap_vals = shap_vals[1]

    importance = np.abs(shap_vals).mean(axis=0)
    imp_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    imp_df.to_csv(out_dir / "shap_global_importance.csv", index=False)

    # Save summary plot (lazy to keep matplotlib import light)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals, Xs, feature_names=feature_names, show=False, max_display=25)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        log.warning(f"could not render shap summary: {e}")

    return {"importance": imp_df, "shap_values": shap_vals, "samples": Xs}
