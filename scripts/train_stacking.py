"""Stacking ensemble: meta-learner over base-model val predictions.

Reads every ``*.parquet`` in ``reports/artifacts/predictions/`` (recursively),
aligns rows by ``video_id`` to a common test set, and fits a small LR
meta-learner on the **val** split, then predicts on **test**.

Output:
- reports/artifacts/predictions/stacking/stacking_lr.parquet
- reports/artifacts/predictions/stacking/stacking_lgbm.parquet (LightGBM head)

The two heads are evaluated alongside everything else by ``scripts/evaluate.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import bootstrap_ci, compute_metrics, find_best_threshold
from src.utils import ensure_dir, load_yaml, set_seed, setup_logger

log = setup_logger("scripts.train_stacking")


def _load_predictions(pred_root: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for p in sorted(pred_root.rglob("*.parquet")):
        # skip our own outputs to avoid recursion when re-running
        if "stacking" in p.parts:
            continue
        key = p.relative_to(pred_root).with_suffix("").as_posix()
        out[key] = pd.read_parquet(p)
    return out


def _build_proba_matrix(
    preds_by_model: dict[str, pd.DataFrame], split: str
) -> tuple[pd.DataFrame, list[str]]:
    """Return wide DataFrame [video_id, label_viral, m1_proba, m2_proba, ...]."""
    keys = sorted(preds_by_model)
    base: pd.DataFrame | None = None
    for k in keys:
        df = preds_by_model[k]
        sub = df[df["split"] == split][["video_id", "y_true", "y_proba"]].rename(
            columns={"y_proba": k}
        )
        if base is None:
            base = sub.rename(columns={"y_true": "label_viral"})
        else:
            base = base.merge(sub.drop(columns=["y_true"]), on="video_id", how="inner")
    return base, keys  # type: ignore[return-value]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-config", default="configs/train.yaml")
    ap.add_argument(
        "--head",
        choices=["lr", "lgbm", "both"],
        default="both",
        help="meta-learner family",
    )
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="model keys to exclude (e.g. transformers/xlm-roberta-large)",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.train_config)
    set_seed(cfg["global"]["seed"])

    root = Path(cfg["global"]["artifacts_dir"])
    pred_root = root / "predictions"
    out_dir = ensure_dir(root / "predictions" / "stacking")

    preds = _load_predictions(pred_root)
    for k in args.exclude:
        preds.pop(k, None)
    log.info(f"stacking over {len(preds)} base models: {list(preds)[:5]} ...")

    val_df, keys_val = _build_proba_matrix(preds, "val")
    test_df, keys_test = _build_proba_matrix(preds, "test")
    if keys_val != keys_test:
        log.warning("val/test base-model sets differ; using intersection")
        common = sorted(set(keys_val) & set(keys_test))
        val_df = val_df[["video_id", "label_viral", *common]]
        test_df = test_df[["video_id", "label_viral", *common]]
        keys = common
    else:
        keys = keys_val

    log.info(f"val n={len(val_df)} | test n={len(test_df)} | {len(keys)} base probas per row")

    x_val = val_df[keys].to_numpy()
    y_val = val_df["label_viral"].to_numpy()
    x_test = test_df[keys].to_numpy()
    y_test = test_df["label_viral"].to_numpy()

    if args.head in ("lr", "both"):
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
        lr.fit(x_val, y_val)
        p_test = lr.predict_proba(x_test)[:, 1]
        p_val_self = lr.predict_proba(x_val)[:, 1]
        rows = pd.concat(
            [
                pd.DataFrame(
                    {
                        "video_id": val_df["video_id"].to_numpy(),
                        "split": "val",
                        "y_true": y_val,
                        "y_proba": p_val_self,
                    }
                ),
                pd.DataFrame(
                    {
                        "video_id": test_df["video_id"].to_numpy(),
                        "split": "test",
                        "y_true": y_test,
                        "y_proba": p_test,
                    }
                ),
            ],
            ignore_index=True,
        )
        rows.to_parquet(out_dir / "stacking_lr.parquet", index=False)
        best_t, _ = find_best_threshold(y_val, p_val_self, "f1_pos")
        m = compute_metrics(y_test, p_test, threshold=best_t)
        _, lo, hi = bootstrap_ci(y_test, p_test, "roc_auc", n_iter=1000)
        log.info(
            f"stacking_lr test: roc_auc={m['roc_auc']:.4f} [{lo:.4f}, {hi:.4f}] "
            f"f1_pos={m['f1_pos']:.4f} threshold={best_t:.3f}"
        )

    if args.head in ("lgbm", "both"):
        import lightgbm as lgb

        gbm = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=30,
            class_weight="balanced",
            random_state=42,
        )
        # tiny model: train on val only is overfit-prone, so use proper CV would be better.
        # For honesty, fit on val and report.
        gbm.fit(x_val, y_val)
        p_test = gbm.predict_proba(x_test)[:, 1]
        p_val_self = gbm.predict_proba(x_val)[:, 1]
        rows = pd.concat(
            [
                pd.DataFrame(
                    {
                        "video_id": val_df["video_id"].to_numpy(),
                        "split": "val",
                        "y_true": y_val,
                        "y_proba": p_val_self,
                    }
                ),
                pd.DataFrame(
                    {
                        "video_id": test_df["video_id"].to_numpy(),
                        "split": "test",
                        "y_true": y_test,
                        "y_proba": p_test,
                    }
                ),
            ],
            ignore_index=True,
        )
        rows.to_parquet(out_dir / "stacking_lgbm.parquet", index=False)
        best_t, _ = find_best_threshold(y_val, p_val_self, "f1_pos")
        m = compute_metrics(y_test, p_test, threshold=best_t)
        _, lo, hi = bootstrap_ci(y_test, p_test, "roc_auc", n_iter=1000)
        log.info(
            f"stacking_lgbm test: roc_auc={m['roc_auc']:.4f} [{lo:.4f}, {hi:.4f}] "
            f"f1_pos={m['f1_pos']:.4f} threshold={best_t:.3f}"
        )

        coef_df = pd.DataFrame(
            {
                "feature": keys,
                "lgbm_importance": gbm.feature_importances_,
            }
        ).sort_values("lgbm_importance", ascending=False)
        coef_df.to_csv(out_dir / "stacking_feature_importance.csv", index=False)
        log.info(f"top-5 base models by LGBM importance:\n{coef_df.head().to_string(index=False)}")


if __name__ == "__main__":
    main()
