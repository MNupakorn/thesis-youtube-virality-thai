"""Fine-tune a Thai transformer model for virality classification.

Usage:
    python scripts/train_transformer.py --model wangchanberta --config configs/train.yaml
    python scripts/train_transformer.py --model typhoon-2.5    --config configs/train.yaml
    python scripts/train_transformer.py --model openthaigpt    --config configs/train.yaml

The first call works on M1/CPU/MPS for WangchanBERTa. The other two need a CUDA GPU
(Colab T4/A100 or Kaggle). See ``notebooks/04_transformer_finetune.ipynb`` for the
Colab-ready version.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd

from src.data_processing.splits import class_weight
from src.evaluation.metrics import bootstrap_ci, compute_metrics, find_best_threshold
from src.models.transformer_finetune import fine_tune
from src.utils import ensure_dir, load_yaml, set_seed, setup_logger

log = setup_logger("scripts.train_transformer")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        required=True,
        choices=["wangchanberta", "phayathaibert", "xlm-roberta-large", "typhoon-2.5", "openthaigpt"],
    )
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--data-config", default="configs/data.yaml")
    args = ap.parse_args()

    train_cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data_config)
    model_cfg = train_cfg["transformers"][args.model]

    set_seed(train_cfg["global"]["seed"])
    mlflow.set_tracking_uri(train_cfg["global"]["mlflow_tracking_uri"])
    mlflow.set_experiment(train_cfg["global"]["experiment_name"])

    proc_path = Path(data_cfg["paths"]["processed_dir"]) / "dataset_with_labels.parquet"
    df = pd.read_parquet(proc_path)
    log.info(f"loaded {df.shape} from {proc_path}")

    # Inverse-frequency weight for the positive class -> alpha for focal loss
    train_df = df[df["split"] == "train"]
    cw = class_weight(train_df, "label_viral")
    pos_alpha = float(cw.get(1, 0.5))
    pos_alpha = pos_alpha / (pos_alpha + cw.get(0, 1.0))   # normalize to [0, 1]

    out_dir = ensure_dir(Path(train_cfg["global"]["artifacts_dir"]) / "models" / args.model)
    pred_dir = ensure_dir(Path(train_cfg["global"]["artifacts_dir"]) / "predictions" / "transformers")

    with mlflow.start_run(run_name=f"transformer_{args.model}"):
        mlflow.log_params({**model_cfg, "model_alias": args.model, "pos_alpha": pos_alpha})

        result = fine_tune(args.model, df, model_cfg, output_dir=out_dir, class_weight_pos=pos_alpha)

        # Save predictions (only for splits that exist)
        rows = []
        for sp, probs in result["predictions"].items():
            sub = df[df["split"] == sp]
            rows.append(
                pd.DataFrame(
                    {
                        "video_id": sub["video_id"].to_numpy(),
                        "split": sp,
                        "y_true": sub["label_viral"].to_numpy(),
                        "y_proba": probs,
                    }
                )
            )
        out_pred = pred_dir / f"{args.model}.parquet"
        pd.concat(rows, ignore_index=True).to_parquet(out_pred, index=False)
        log.info(f"saved predictions -> {out_pred}")

        # Threshold tune on val, evaluate on test
        y_val = df.loc[df["split"] == "val", "label_viral"].to_numpy()
        y_test = df.loc[df["split"] == "test", "label_viral"].to_numpy()
        p_val = result["predictions"].get("val")
        p_test = result["predictions"].get("test")

        if p_val is not None and len(y_val) == len(p_val):
            best_t, _ = find_best_threshold(y_val, p_val, "f1_pos")
        else:
            best_t = 0.5
        test_metrics = compute_metrics(y_test, p_test, threshold=best_t)
        pe, lo, hi = bootstrap_ci(y_test, p_test, "roc_auc", n_iter=500)
        test_metrics.update({"roc_auc_ci_lo": lo, "roc_auc_ci_hi": hi, "threshold": best_t})
        mlflow.log_metrics({k: v for k, v in test_metrics.items() if isinstance(v, (int, float))})
        log.info(
            f"{args.model}: roc_auc={test_metrics['roc_auc']:.4f} [{lo:.4f}, {hi:.4f}], "
            f"f1_pos={test_metrics['f1_pos']:.4f}, threshold={best_t:.3f}"
        )


if __name__ == "__main__":
    main()
