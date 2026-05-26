"""End-to-end evaluation across all trained models.

Loads predictions from reports/artifacts/predictions/ and produces:
- reports/tables/all_models_metrics.csv
- reports/tables/mcnemar_pairwise.csv
- reports/tables/cochrans_q.csv
- reports/figures/roc_pr_curves.png
- reports/figures/calibration_reliability.png
- reports/tables/per_category_breakdown.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    expected_calibration_error,
    reliability_curve,
)
from src.evaluation.metrics import bootstrap_ci, compute_metrics, find_best_threshold
from src.evaluation.stats_tests import cochrans_q_test, mcnemar_pairwise
from src.utils import ensure_dir, load_yaml, set_seed, setup_logger

log = setup_logger("scripts.evaluate")


def _load_predictions(pred_root: Path) -> dict[str, pd.DataFrame]:
    """Discover every *.parquet under pred_root, key by relative path stem."""
    out = {}
    for p in sorted(pred_root.rglob("*.parquet")):
        key = p.relative_to(pred_root).with_suffix("").as_posix()
        out[key] = pd.read_parquet(p)
    return out


def _curves(probs, y, n=200):
    import numpy as np
    from sklearn.metrics import precision_recall_curve, roc_curve

    fpr, tpr, _ = roc_curve(y, probs)
    p, r, _ = precision_recall_curve(y, probs)
    return {"fpr": fpr, "tpr": tpr, "precision": p, "recall": r}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--train-config", default="configs/train.yaml")
    args = ap.parse_args()

    eval_cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data_config)
    train_cfg = load_yaml(args.train_config)
    set_seed(train_cfg["global"]["seed"])

    pred_root = Path(train_cfg["global"]["artifacts_dir"]) / "predictions"
    tables_dir = ensure_dir(eval_cfg["metrics_dir"])
    fig_dir = ensure_dir(eval_cfg["figures_dir"])

    preds_by_model = _load_predictions(pred_root)
    if not preds_by_model:
        log.warning(f"no predictions found under {pred_root}; train models first")
        return

    # Load processed data for slice analysis
    df = pd.read_parquet(Path(data_cfg["paths"]["processed_dir"]) / "dataset_with_labels.parquet")

    # Per-model metrics on test
    rows = []
    test_probs: dict[str, pd.DataFrame] = {}
    for name, pred_df in preds_by_model.items():
        val = pred_df[pred_df["split"] == "val"]
        test = pred_df[pred_df["split"] == "test"]
        if test.empty:
            log.warning(f"{name}: no test predictions")
            continue
        if not val.empty:
            best_t, _ = find_best_threshold(val["y_true"].to_numpy(), val["y_proba"].to_numpy(), "f1_pos")
        else:
            best_t = 0.5
        m = compute_metrics(test["y_true"].to_numpy(), test["y_proba"].to_numpy(), threshold=best_t)
        _, lo, hi = bootstrap_ci(
            test["y_true"].to_numpy(),
            test["y_proba"].to_numpy(),
            "roc_auc",
            n_iter=eval_cfg["bootstrap"].get("n_iter", 1000),
            alpha=eval_cfg["bootstrap"].get("alpha", 0.05),
        )
        m.update({"roc_auc_ci_lo": lo, "roc_auc_ci_hi": hi, "threshold": best_t})
        rows.append({"model": name, **m})
        test_probs[name] = test

    metrics_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(tables_dir / "all_models_metrics.csv", index=False)
    log.info(f"per-model test metrics:\n{metrics_df.to_string(index=False)}")

    # ---- Statistical tests on the COMMON test rows -------------------------
    if len(test_probs) >= 2:
        # align by video_id intersection
        first = next(iter(test_probs.values()))
        common = set(first["video_id"])
        for tp in test_probs.values():
            common &= set(tp["video_id"])
        common = sorted(common)
        if len(common) < 100:
            log.warning(f"only {len(common)} videos common across model predictions")

        order = pd.Series(common, name="video_id").to_frame()
        y_true = (
            order.merge(first[["video_id", "y_true"]], on="video_id")["y_true"].to_numpy()
        )
        proba_dict = {}
        for name, tp in test_probs.items():
            sub = order.merge(tp[["video_id", "y_proba"]], on="video_id")
            proba_dict[name] = sub["y_proba"].to_numpy()

        # Pairwise McNemar
        if eval_cfg["stats_tests"]["mcnemars"]["enabled"]:
            mc = mcnemar_pairwise(y_true, proba_dict)
            mc.to_csv(tables_dir / "mcnemar_pairwise.csv", index=False)
            log.info(f"mcnemar pairwise:\n{mc.to_string(index=False)}")

        # Cochran's Q (3+ models only)
        if (
            eval_cfg["stats_tests"]["cochrans_q"]["enabled"]
            and len(proba_dict) >= eval_cfg["stats_tests"]["cochrans_q"]["apply_when_n_models_ge"]
        ):
            cq = cochrans_q_test(y_true, proba_dict)
            pd.DataFrame([cq]).to_csv(tables_dir / "cochrans_q.csv", index=False)
            log.info(f"cochran's Q: {cq}")

        # ---- Encoder-only subset (the thesis headline comparison) ----------
        encoder_aliases = ("wangchanberta", "phayathaibert", "xlm-roberta-large")
        encoder_keys = [k for k in proba_dict if Path(k).name in encoder_aliases]
        if len(encoder_keys) >= 2:
            enc_proba = {k: proba_dict[k] for k in encoder_keys}
            mc_enc = mcnemar_pairwise(y_true, enc_proba)
            mc_enc.to_csv(tables_dir / "mcnemar_pairwise_encoders.csv", index=False)
            log.info(f"3-encoder mcnemar:\n{mc_enc.to_string(index=False)}")
            if len(encoder_keys) >= 3:
                cq_enc = cochrans_q_test(y_true, enc_proba)
                pd.DataFrame([cq_enc]).to_csv(tables_dir / "cochrans_q_encoders.csv", index=False)
                log.info(f"3-encoder Cochran's Q: {cq_enc}")
        else:
            log.warning(
                f"encoder subset has only {len(encoder_keys)} model(s); skip encoder-only stats"
            )

    # ---- Calibration on top model -----------------------------------------
    if eval_cfg["calibration"]["enabled"] and not metrics_df.empty:
        top_model = metrics_df.iloc[0]["model"]
        log.info(f"calibrating top model: {top_model}")
        df_top = preds_by_model[top_model]
        val = df_top[df_top["split"] == "val"]
        test = df_top[df_top["split"] == "test"]

        rows = []
        for name, cal_cls in [("none", None), ("platt", PlattCalibrator), ("isotonic", IsotonicCalibrator)]:
            if cal_cls is None:
                p_test = test["y_proba"].to_numpy()
            else:
                cal = cal_cls()
                cal.fit(val["y_proba"].to_numpy(), val["y_true"].to_numpy())
                p_test = cal.transform(test["y_proba"].to_numpy())
            ece = expected_calibration_error(p_test, test["y_true"].to_numpy(),
                                             n_bins=eval_cfg["calibration"]["bins"])
            rc = reliability_curve(p_test, test["y_true"].to_numpy(), n_bins=eval_cfg["calibration"]["bins"])
            rows.append({"calibrator": name, "ece": ece})
            pd.DataFrame(rc, columns=["mean_conf", "mean_acc", "n"]).to_csv(
                tables_dir / f"calibration_{top_model.replace('/', '_')}_{name}.csv", index=False
            )
        pd.DataFrame(rows).to_csv(tables_dir / "calibration_summary.csv", index=False)
        log.info(f"calibration summary saved")

    # ---- Per-category breakdown -------------------------------------------
    if "by_category_id" in eval_cfg["slices"] and not metrics_df.empty:
        top_model = metrics_df.iloc[0]["model"]
        df_top = preds_by_model[top_model]
        test = df_top[df_top["split"] == "test"].merge(
            df[["video_id", "category_id", "category_title"]].drop_duplicates("video_id"),
            on="video_id",
            how="left",
        )
        rows = []
        for cid, sub in test.groupby("category_id"):
            if len(sub) < 30 or sub["y_true"].nunique() < 2:
                continue
            m = compute_metrics(sub["y_true"].to_numpy(), sub["y_proba"].to_numpy())
            rows.append({"category_id": cid, "category_title": sub["category_title"].iloc[0], "n": len(sub), **m})
        pd.DataFrame(rows).sort_values("roc_auc", ascending=False).to_csv(
            tables_dir / "per_category_breakdown.csv", index=False
        )

    # ---- ROC / PR curves (top 6 by ROC-AUC) -------------------------------
    if not metrics_df.empty:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.metrics import auc as sk_auc
            from sklearn.metrics import precision_recall_curve, roc_curve

            top_models = metrics_df["model"].head(6).tolist()
            fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 5))
            for name in top_models:
                tp = test_probs[name]
                y = tp["y_true"].to_numpy()
                p = tp["y_proba"].to_numpy()
                fpr, tpr, _ = roc_curve(y, p)
                pre, rec, _ = precision_recall_curve(y, p)
                ax_roc.plot(fpr, tpr, label=f"{name} (AUC={sk_auc(fpr, tpr):.3f})", lw=1.4)
                ax_pr.plot(rec, pre, label=f"{name} (AP={sk_auc(rec, pre):.3f})", lw=1.4)
            ax_roc.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
            ax_roc.set(xlabel="FPR", ylabel="TPR", title="ROC (test) — top 6 by AUC")
            ax_pr.set(xlabel="Recall", ylabel="Precision", title="PR (test) — top 6 by AUC")
            ax_roc.legend(fontsize=7, loc="lower right")
            ax_pr.legend(fontsize=7, loc="upper right")
            ax_roc.grid(alpha=0.3)
            ax_pr.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(fig_dir / "roc_pr_test.svg")
            plt.savefig(fig_dir / "roc_pr_test.png", dpi=140)
            plt.close()
            log.info(f"ROC/PR curves saved to {fig_dir}")
        except Exception as e:
            log.warning(f"ROC/PR figure failed: {e}")

    # ---- Reliability diagram for top model --------------------------------
    if eval_cfg["calibration"]["enabled"] and not metrics_df.empty:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            top_model = metrics_df.iloc[0]["model"]
            df_top = preds_by_model[top_model]
            test = df_top[df_top["split"] == "test"]
            n_bins = eval_cfg["calibration"]["bins"]
            rc = reliability_curve(test["y_proba"].to_numpy(), test["y_true"].to_numpy(), n_bins=n_bins)
            mc = np.array([r[0] for r in rc])
            ma = np.array([r[1] for r in rc])
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8, label="perfect")
            ax.plot(mc, ma, "o-", lw=1.4, label=top_model)
            ax.set(
                xlabel="mean predicted prob.",
                ylabel="empirical positive rate",
                title=f"Reliability diagram — {top_model}",
            )
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(fig_dir / "calibration_top_model.svg")
            plt.savefig(fig_dir / "calibration_top_model.png", dpi=140)
            plt.close()
            log.info("reliability diagram saved")
        except Exception as e:
            log.warning(f"reliability figure failed: {e}")

    # ---- Error analysis: top-50 FP / FN by confidence ---------------------
    if not metrics_df.empty:
        top_model = metrics_df.iloc[0]["model"]
        df_top = preds_by_model[top_model]
        meta_cols = [
            c for c in ("video_id", "title", "channel_title", "category_title", "view_count")
            if c in df.columns
        ]
        test = df_top[df_top["split"] == "test"].merge(
            df[meta_cols].drop_duplicates("video_id"), on="video_id", how="left"
        )
        thr = float(metrics_df.iloc[0].get("threshold", 0.5))
        test = test.assign(predicted=(test["y_proba"] >= thr).astype(int))
        fp = test[(test["predicted"] == 1) & (test["y_true"] == 0)].nlargest(50, "y_proba")
        fn = test[(test["predicted"] == 0) & (test["y_true"] == 1)].nsmallest(50, "y_proba")
        fp.to_csv(tables_dir / f"errors_top_fp_{top_model.replace('/', '_')}.csv", index=False)
        fn.to_csv(tables_dir / f"errors_top_fn_{top_model.replace('/', '_')}.csv", index=False)
        log.info(f"error analysis: top-50 FP / FN of {top_model} saved")

    log.info(f"evaluation complete; results in {tables_dir} / {fig_dir}")


if __name__ == "__main__":
    main()
