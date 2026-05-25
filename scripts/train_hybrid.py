"""Train the multi-modal hybrid model (proposed method).

Loads:
- Processed labels + structured + sentiment features from data/processed/
- Cached title transformer embeddings from data/interim/title_embeddings.npy

Trains BOTH heads on the fused feature matrix:
- MLP (PyTorch, focal loss)
- LightGBM (great for SHAP)

Outputs predictions + metrics + saved artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.data_processing.splits import class_weight
from src.evaluation.metrics import bootstrap_ci, compute_metrics, find_best_threshold
from src.features.transformer_embed import compute_or_load_embeddings
from src.models.baselines import predict_proba_binary
from src.models.hybrid import (
    assemble_hybrid_features,
    predict_mlp,
    save_hybrid_artifacts,
    train_gbm_head,
    train_mlp_head,
)
from src.utils import ensure_dir, load_yaml, set_seed, setup_logger

log = setup_logger("scripts.train_hybrid")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--features-config", default="configs/features.yaml")
    args = ap.parse_args()

    train_cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data_config)
    feat_cfg = load_yaml(args.features_config)
    h_cfg = train_cfg["hybrid"]

    set_seed(train_cfg["global"]["seed"])
    mlflow.set_tracking_uri(train_cfg["global"]["mlflow_tracking_uri"])
    mlflow.set_experiment(train_cfg["global"]["experiment_name"])

    proc_path = Path(data_cfg["paths"]["processed_dir"]) / "dataset_with_labels.parquet"
    df = pd.read_parquet(proc_path)
    log.info(f"loaded {df.shape} from {proc_path}")

    # Get/compute title embeddings aligned to df row order
    ec = feat_cfg["transformer_embedding"]
    _, embeddings = compute_or_load_embeddings(
        df,
        cache_path=ec["cache_dir"],
        model_name=ec["model_name"],
        pooling=ec.get("pooling", "cls"),
        max_length=ec.get("max_length", 64),
        batch_size=64,
        device="auto",
    )

    # Assemble fused features
    fused = assemble_hybrid_features(
        df,
        title_embeddings=embeddings,
        use_sentiment=True,
        use_structured=True,
        use_embedding=True,
    )

    # Per-split slicing
    masks = {sp: (df["split"] == sp).to_numpy() for sp in ("train", "val", "test")}
    X_split = {sp: fused.X[m] for sp, m in masks.items()}
    y_split = {sp: df.loc[m, "label_viral"].to_numpy() for sp, m in masks.items()}
    id_split = {sp: df.loc[m, "video_id"].to_numpy() for sp, m in masks.items()}

    # ---- MLP head ----------------------------------------------------------
    cw = class_weight(df[df["split"] == "train"], "label_viral")
    pos_alpha = cw.get(1, 0.5) / (cw.get(1, 0.5) + cw.get(0, 1.0))

    out_dir = ensure_dir(Path(train_cfg["global"]["artifacts_dir"]) / "models" / "hybrid")
    pred_dir = ensure_dir(Path(train_cfg["global"]["artifacts_dir"]) / "predictions" / "hybrid")

    with mlflow.start_run(run_name="hybrid_mlp"):
        mlflow.log_params({**{k: str(v) for k, v in h_cfg.items()}, "pos_alpha": pos_alpha})
        mlp_res = train_mlp_head(
            X_train=X_split["train"],
            y_train=y_split["train"],
            X_val=X_split["val"],
            y_val=y_split["val"],
            cfg=h_cfg,
            class_weight_pos=pos_alpha,
        )

        preds = {sp: predict_mlp(mlp_res["model"], X_split[sp]) for sp in X_split}
        rows = [
            pd.DataFrame({"video_id": id_split[sp], "split": sp, "y_true": y_split[sp], "y_proba": preds[sp]})
            for sp in X_split
        ]
        pd.concat(rows, ignore_index=True).to_parquet(pred_dir / "hybrid_mlp.parquet", index=False)

        best_t, _ = find_best_threshold(y_split["val"], preds["val"], "f1_pos")
        m = compute_metrics(y_split["test"], preds["test"], threshold=best_t)
        pe, lo, hi = bootstrap_ci(y_split["test"], preds["test"], "roc_auc", n_iter=500)
        m.update({"roc_auc_ci_lo": lo, "roc_auc_ci_hi": hi, "threshold": best_t, "best_val_auc": mlp_res["best_val_auc"]})
        mlflow.log_metrics({k: v for k, v in m.items() if isinstance(v, (int, float))})
        log.info(f"hybrid_mlp test: {m}")

        save_hybrid_artifacts(
            out_dir,
            mlp_state_dict=mlp_res["model"].net.state_dict(),
            feature_names=fused.feature_names,
        )

    # ---- LightGBM head -----------------------------------------------------
    if h_cfg.get("also_train_gbm_head", True):
        with mlflow.start_run(run_name="hybrid_gbm"):
            gbm = train_gbm_head(
                X_train=X_split["train"],
                y_train=y_split["train"],
                X_val=X_split["val"],
                y_val=y_split["val"],
            )
            preds = {sp: predict_proba_binary(gbm, X_split[sp]) for sp in X_split}
            rows = [
                pd.DataFrame({"video_id": id_split[sp], "split": sp, "y_true": y_split[sp], "y_proba": preds[sp]})
                for sp in X_split
            ]
            pd.concat(rows, ignore_index=True).to_parquet(pred_dir / "hybrid_gbm.parquet", index=False)

            best_t, _ = find_best_threshold(y_split["val"], preds["val"], "f1_pos")
            m = compute_metrics(y_split["test"], preds["test"], threshold=best_t)
            pe, lo, hi = bootstrap_ci(y_split["test"], preds["test"], "roc_auc", n_iter=500)
            m.update({"roc_auc_ci_lo": lo, "roc_auc_ci_hi": hi, "threshold": best_t})
            mlflow.log_metrics({k: v for k, v in m.items() if isinstance(v, (int, float))})
            log.info(f"hybrid_gbm test: {m}")
            save_hybrid_artifacts(out_dir, gbm=gbm, feature_names=fused.feature_names)

    # Persist the fused feature matrix split layout for downstream SHAP
    np.save(out_dir / "X_test.npy", X_split["test"])
    np.save(out_dir / "y_test.npy", y_split["test"])
    log.info(f"hybrid done. artifacts in {out_dir}")


if __name__ == "__main__":
    main()
