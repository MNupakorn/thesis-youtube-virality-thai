"""Generate model explanations: SHAP (LightGBM head), LIME (transformer titles),
and attention rollout (transformer titles).

Outputs:
- reports/figures/shap_summary.png
- reports/tables/shap_global_importance.csv
- reports/figures/lime/{lime_html, lime_per_example.csv, lime_aggregate_tokens.csv}
- reports/figures/attention/{attention_html, attention_rollout_per_example.parquet}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.explainability.attention_runner import explain_with_attention
from src.explainability.lime_runner import explain_with_lime
from src.explainability.shap_runner import explain_with_shap
from src.utils import ensure_dir, load_yaml, set_seed, setup_logger

log = setup_logger("scripts.explain")


def _run_shap(eval_cfg: dict, train_cfg: dict) -> None:
    hybrid_dir = Path(train_cfg["global"]["artifacts_dir"]) / "models" / "hybrid"
    gbm_path = hybrid_dir / "gbm.joblib"
    feat_path = hybrid_dir / "feature_names.joblib"
    X_test_path = hybrid_dir / "X_test.npy"

    if not (gbm_path.exists() and X_test_path.exists() and feat_path.exists()):
        log.warning(
            f"SHAP skipped: missing artifacts in {hybrid_dir} (run scripts/train_hybrid.py)"
        )
        return

    gbm = joblib.load(gbm_path)
    feature_names = joblib.load(feat_path)
    X_test = np.load(X_test_path)

    fig_dir = ensure_dir(eval_cfg["figures_dir"])
    res = explain_with_shap(
        gbm,
        X_test,
        feature_names,
        out_dir=fig_dir,
        max_samples=eval_cfg["explainability"]["shap"]["max_samples"],
        seed=train_cfg["global"]["seed"],
    )
    res["importance"].to_csv(eval_cfg["metrics_dir"] + "/shap_global_importance.csv", index=False)
    log.info(f"SHAP done -> {fig_dir}")


def _run_lime(eval_cfg: dict, train_cfg: dict, data_cfg: dict, model_alias: str) -> None:
    model_dir = Path(train_cfg["global"]["artifacts_dir"]) / "models" / model_alias
    if not model_dir.exists() or not any(model_dir.iterdir()):
        log.warning(f"LIME skipped: no checkpoint at {model_dir}")
        return

    df = pd.read_parquet(Path(data_cfg["paths"]["processed_dir"]) / "dataset_with_labels.parquet")
    test = df[df["split"] == "test"].reset_index(drop=True)
    titles = test["title"].astype(str).tolist()

    out_dir = ensure_dir(Path(eval_cfg["figures_dir"]) / "lime")
    explain_with_lime(
        model_dir=model_dir,
        titles=titles,
        out_dir=out_dir,
        n_samples_to_explain=eval_cfg["explainability"]["lime"]["n_samples_to_explain"],
        max_length=train_cfg["transformers"][model_alias]["max_length"],
        seed=train_cfg["global"]["seed"],
    )
    log.info(f"LIME done -> {out_dir}")


def _run_attention(eval_cfg: dict, train_cfg: dict, data_cfg: dict, model_alias: str) -> None:
    model_dir = Path(train_cfg["global"]["artifacts_dir"]) / "models" / model_alias
    if not model_dir.exists() or not any(model_dir.iterdir()):
        log.warning(f"attention skipped: no checkpoint at {model_dir}")
        return

    df = pd.read_parquet(Path(data_cfg["paths"]["processed_dir"]) / "dataset_with_labels.parquet")
    test = df[df["split"] == "test"].reset_index(drop=True)
    titles = test["title"].astype(str).tolist()
    labels = test["label_viral"].astype(int).tolist()

    out_dir = ensure_dir(Path(eval_cfg["figures_dir"]) / "attention")
    explain_with_attention(
        model_dir=model_dir,
        titles=titles,
        labels=labels,
        out_dir=out_dir,
        n_samples=eval_cfg["explainability"]["attention"]["n_samples_to_visualize"],
        max_length=train_cfg["transformers"][model_alias]["max_length"],
        seed=train_cfg["global"]["seed"],
    )
    log.info(f"attention rollout done -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--train-config", default="configs/train.yaml")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument(
        "--only", choices=["shap", "lime", "attention"], default=None, help="run only one explainer"
    )
    args = ap.parse_args()

    eval_cfg = load_yaml(args.config)
    train_cfg = load_yaml(args.train_config)
    data_cfg = load_yaml(args.data_config)
    set_seed(train_cfg["global"]["seed"])

    expl = eval_cfg["explainability"]

    if args.only in (None, "shap") and expl["shap"]["enabled"]:
        _run_shap(eval_cfg, train_cfg)

    if args.only in (None, "lime") and expl["lime"]["enabled"]:
        _run_lime(eval_cfg, train_cfg, data_cfg, expl["lime"]["on_model"])

    if args.only in (None, "attention") and expl["attention"]["enabled"]:
        _run_attention(eval_cfg, train_cfg, data_cfg, expl["attention"]["on_model"])


if __name__ == "__main__":
    main()
