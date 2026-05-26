"""Polished thesis-ready figures: model leaderboard + per-category breakdown.

Outputs:
- reports/figures/leaderboard_with_ci.svg / .png — all models ranked with 95% CIs
- reports/figures/category_breakdown.svg / .png — per-category AUC for top model
- reports/figures/reliability_stacking.svg / .png — calibration of the winner
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.calibration import expected_calibration_error, reliability_curve
from src.utils import setup_logger

log = setup_logger("scripts.thesis_figures")


def model_leaderboard(metrics_csv: Path, out_dir: Path, top_n: int = 12) -> None:
    df = pd.read_csv(metrics_csv).sort_values("roc_auc", ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(df))
    auc = df["roc_auc"].to_numpy()
    err_lo = auc - df["roc_auc_ci_lo"].to_numpy()
    err_hi = df["roc_auc_ci_hi"].to_numpy() - auc
    colors = []
    for name in df["model"]:
        if "stacking" in name:
            colors.append("#dc2626")  # red for ensemble
        elif "transformers" in name:
            colors.append("#2563eb")  # blue for encoders
        elif "hybrid" in name:
            colors.append("#9333ea")  # purple
        else:
            colors.append("#6b7280")  # gray for baselines
    ax.barh(y, auc - 0.5, xerr=[err_lo, err_hi], left=0.5, color=colors,
            edgecolor="black", linewidth=0.4, capsize=3, alpha=0.85)
    for i, v in enumerate(auc):
        ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=8)
    ax.axvline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.6, label="random (0.5)")
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"].tolist(), fontsize=8)
    ax.set_xlabel("Test ROC-AUC (with bootstrap 95% CI)")
    ax.set_xlim(0.50, max(0.78, auc.max() + 0.04))
    ax.set_title("Model leaderboard on the time-aware test set")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "leaderboard_with_ci.svg")
    plt.savefig(out_dir / "leaderboard_with_ci.png", dpi=140)
    plt.close()
    log.info(f"saved leaderboard figure -> {out_dir}")


def category_breakdown(per_cat_csv: Path, by_size_csv: Path, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    df_cat = pd.read_csv(per_cat_csv).sort_values("roc_auc", ascending=True)
    cat_labels = df_cat["category_title"].fillna("(unknown)").tolist()
    ax1.barh(np.arange(len(df_cat)), df_cat["roc_auc"] - 0.5, left=0.5,
             color="#0891b2", edgecolor="black", linewidth=0.4, alpha=0.85)
    for i, (auc, n) in enumerate(zip(df_cat["roc_auc"], df_cat["n"], strict=True)):
        ax1.text(auc + 0.005, i, f"{auc:.3f} (n={n})", va="center", fontsize=8)
    ax1.axvline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.6)
    ax1.set_yticks(np.arange(len(df_cat)))
    ax1.set_yticklabels(cat_labels, fontsize=8)
    ax1.set_xlabel("Test ROC-AUC")
    ax1.set_xlim(0.50, 0.85)
    ax1.set_title("By YouTube category")
    ax1.grid(axis="x", alpha=0.3)

    df_sz = pd.read_csv(by_size_csv)
    if not df_sz.empty:
        labels = df_sz["bucket"].tolist()
        auc = df_sz["roc_auc"].to_numpy()
        err_lo = auc - df_sz["ci_lo"].to_numpy()
        err_hi = df_sz["ci_hi"].to_numpy() - auc
        ax2.bar(np.arange(len(df_sz)), auc - 0.5, bottom=0.5,
                yerr=[err_lo, err_hi], color="#16a34a", edgecolor="black",
                linewidth=0.4, capsize=4, alpha=0.85)
        for i, (a, n) in enumerate(zip(auc, df_sz["n"], strict=True)):
            ax2.text(i, a + 0.012, f"{a:.3f}\n(n={n})", ha="center", fontsize=8)
        ax2.set_xticks(np.arange(len(df_sz)))
        ax2.set_xticklabels(labels, fontsize=8, rotation=10)
        ax2.set_ylim(0.50, max(0.85, auc.max() + 0.05))
        ax2.axhline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.6)
        ax2.set_ylabel("Test ROC-AUC")
        ax2.set_title("Stacking ensemble by channel-size bucket")
        ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Top model robustness across slices", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "robustness_slices.svg")
    plt.savefig(out_dir / "robustness_slices.png", dpi=140)
    plt.close()
    log.info(f"saved robustness figure -> {out_dir}")


def reliability_stacking(out_dir: Path) -> None:
    stk = pd.read_parquet("reports/artifacts/predictions/stacking/stacking_lr.parquet")
    stk_cal = pd.read_parquet("reports/artifacts/predictions/stacking/stacking_lr_calibrated.parquet")
    val = stk[stk["split"] == "val"]
    test = stk[stk["split"] == "test"]
    test_cal = stk_cal[stk_cal["split"] == "test"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8, label="perfect")
    for label, p, color in [
        ("raw stacking", test["y_proba"].to_numpy(), "#dc2626"),
        ("Platt-calibrated", test_cal["y_proba"].to_numpy(), "#2563eb"),
    ]:
        rc = reliability_curve(p, test["y_true"].to_numpy(), n_bins=15)
        mc = np.array([r[0] for r in rc])
        ma = np.array([r[1] for r in rc])
        ece = expected_calibration_error(p, test["y_true"].to_numpy(), n_bins=15)
        ax.plot(mc, ma, "o-", lw=1.5, color=color, label=f"{label} (ECE={ece:.3f})")
    ax.set(xlabel="Mean predicted probability",
           ylabel="Empirical positive rate",
           title="Reliability of the stacking ensemble")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "reliability_stacking.svg")
    plt.savefig(out_dir / "reliability_stacking.png", dpi=140)
    plt.close()
    log.info(f"saved reliability figure -> {out_dir}")


def main() -> None:
    out_dir = Path("reports/figures")
    out_dir.mkdir(exist_ok=True)
    model_leaderboard(Path("reports/tables/all_models_metrics.csv"), out_dir)
    if Path("reports/tables/per_category_breakdown.csv").exists() and Path(
        "reports/tables/stacking_by_channel_size.csv"
    ).exists():
        category_breakdown(
            Path("reports/tables/per_category_breakdown.csv"),
            Path("reports/tables/stacking_by_channel_size.csv"),
            out_dir,
        )
    reliability_stacking(out_dir)


if __name__ == "__main__":
    main()
