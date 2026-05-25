"""Train baseline ML models (LR, LightGBM, XGBoost) on multiple feature sets.

Outputs:
    reports/artifacts/predictions/baselines/{model}_{feature_set}.parquet
        columns: video_id, split, y_true, y_proba
    reports/tables/baselines_metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.evaluation.metrics import bootstrap_ci, compute_metrics, find_best_threshold
from src.features.sentiment import SENTIMENT_FEATURE_COLUMNS
from src.features.structural import STRUCTURED_FEATURE_COLUMNS
from src.features.tfidf import build_tfidf_for_splits
from src.models.baselines import (
    _build_feature_matrix,
    predict_proba_binary,
    save_model,
    train_lightgbm,
    train_logistic_regression,
    train_xgboost,
)
from src.utils import ensure_dir, load_yaml, set_seed, setup_logger

log = setup_logger("scripts.train_baselines")


def _select_features(df: pd.DataFrame, feature_set: str, structured_cols: list[str]):
    X_struct = df[structured_cols].fillna(0.0).to_numpy()
    return X_struct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--features-config", default="configs/features.yaml")
    args = ap.parse_args()

    train_cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data_config)
    feat_cfg = load_yaml(args.features_config)

    set_seed(train_cfg["global"]["seed"])
    mlflow.set_tracking_uri(train_cfg["global"]["mlflow_tracking_uri"])
    mlflow.set_experiment(train_cfg["global"]["experiment_name"])

    proc_path = Path(data_cfg["paths"]["processed_dir"]) / "dataset_with_labels.parquet"
    df = pd.read_parquet(proc_path)
    log.info(f"loaded processed data {df.shape} from {proc_path}")

    structured_cols = [c for c in STRUCTURED_FEATURE_COLUMNS if c in df.columns]
    sent_cols = [c for c in SENTIMENT_FEATURE_COLUMNS if c in df.columns]
    has_sentiment = len(sent_cols) > 0

    out_pred_dir = ensure_dir(Path(train_cfg["global"]["artifacts_dir"]) / "predictions" / "baselines")
    out_model_dir = ensure_dir(Path(train_cfg["global"]["artifacts_dir"]) / "models" / "baselines")
    out_metrics_dir = ensure_dir("reports/tables")

    # TF-IDF (fit on train split only)
    tfidf_cfg = feat_cfg.get("tfidf", {})
    if tfidf_cfg.get("enabled", True):
        _, tfidf_matrices = build_tfidf_for_splits(
            df,
            text_col="title",
            split_col="split",
            fit_on="train",
            max_features=tfidf_cfg.get("max_features", 20000),
            ngram_range=tuple(tfidf_cfg.get("ngram_range", [1, 2])),
            min_df=tfidf_cfg.get("min_df", 5),
            sublinear_tf=tfidf_cfg.get("sublinear_tf", True),
            tokenizer_name=tfidf_cfg.get("tokenizer", "pythainlp"),
        )
    else:
        tfidf_matrices = None

    rows: list[dict] = []

    for fs in train_cfg["baselines"]["feature_sets"]:
        if fs in {"structured_plus_sentiment", "all"} and not has_sentiment:
            log.warning(f"skipping feature_set={fs} (no sentiment columns present)")
            continue

        # Build per-split feature matrices
        X_struct_all = df[structured_cols].fillna(0.0).to_numpy()
        X_sent_all = df[sent_cols].fillna(0.0).to_numpy() if has_sentiment else None

        for split_name in ("train", "val", "test"):
            mask = (df["split"] == split_name).to_numpy()
            if not mask.any():
                continue

        for model_spec in train_cfg["baselines"]["models"]:
            mdl_name = model_spec["name"]
            run_name = f"baseline_{mdl_name}_{fs}"

            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({"model": mdl_name, "feature_set": fs, **model_spec["params"]})

                splits_X = {}
                splits_y = {}
                splits_ids = {}
                for sp in ("train", "val", "test"):
                    m = (df["split"] == sp).to_numpy()
                    if not m.any():
                        continue
                    Xs = _build_feature_matrix(
                        fs,
                        X_struct_all[m],
                        tfidf_matrices[sp] if tfidf_matrices else None,
                        X_sent_all[m] if X_sent_all is not None else None,
                    )
                    splits_X[sp] = Xs
                    splits_y[sp] = df.loc[m, "label_viral"].to_numpy()
                    splits_ids[sp] = df.loc[m, "video_id"].to_numpy()

                # Train
                if mdl_name == "logistic_regression":
                    clf = train_logistic_regression(splits_X["train"], splits_y["train"], dict(model_spec["params"]))
                elif mdl_name == "lightgbm":
                    clf = train_lightgbm(
                        splits_X["train"], splits_y["train"], dict(model_spec["params"]),
                        splits_X.get("val"), splits_y.get("val"),
                    )
                elif mdl_name == "xgboost":
                    clf = train_xgboost(
                        splits_X["train"], splits_y["train"], dict(model_spec["params"]),
                        splits_X.get("val"), splits_y.get("val"),
                    )
                else:
                    raise ValueError(f"unknown model: {mdl_name}")

                # Save model
                save_model(clf, out_model_dir / f"{mdl_name}_{fs}.joblib")

                # Predict on every split
                preds = {sp: predict_proba_binary(clf, splits_X[sp]) for sp in splits_X}

                # Save predictions
                pred_rows = []
                for sp in splits_X:
                    pred_rows.append(
                        pd.DataFrame(
                            {
                                "video_id": splits_ids[sp],
                                "split": sp,
                                "y_true": splits_y[sp],
                                "y_proba": preds[sp],
                            }
                        )
                    )
                pd.concat(pred_rows, ignore_index=True).to_parquet(
                    out_pred_dir / f"{mdl_name}_{fs}.parquet", index=False
                )

                # Tune threshold on val, evaluate on test
                if "val" in preds:
                    best_t, _ = find_best_threshold(splits_y["val"], preds["val"], "f1_pos")
                else:
                    best_t = 0.5

                test_metrics = compute_metrics(splits_y["test"], preds["test"], threshold=best_t)
                pe, lo, hi = bootstrap_ci(splits_y["test"], preds["test"], "roc_auc", n_iter=500)

                test_metrics.update({"roc_auc_ci_lo": lo, "roc_auc_ci_hi": hi, "threshold": best_t})
                mlflow.log_metrics({k: v for k, v in test_metrics.items() if isinstance(v, (int, float))})
                rows.append({"model": mdl_name, "feature_set": fs, **test_metrics})

                log.info(
                    f"{run_name}: roc_auc={test_metrics['roc_auc']:.4f} [{lo:.4f}, {hi:.4f}], "
                    f"f1_pos={test_metrics['f1_pos']:.4f}, threshold={best_t:.3f}"
                )

    res = pd.DataFrame(rows).sort_values(["roc_auc", "f1_pos"], ascending=False)
    res.to_csv(Path(out_metrics_dir) / "baselines_metrics.csv", index=False)
    log.info(f"saved baseline metrics -> reports/tables/baselines_metrics.csv\n{res.to_string(index=False)}")


if __name__ == "__main__":
    main()
