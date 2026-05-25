"""Generate SHAP explanations for the LightGBM head trained on hybrid features.

Outputs:
- reports/figures/shap_summary.png
- reports/tables/shap_global_importance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from src.explainability.shap_runner import explain_with_shap
from src.utils import ensure_dir, load_yaml, set_seed, setup_logger

log = setup_logger("scripts.explain")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--train-config", default="configs/train.yaml")
    args = ap.parse_args()

    eval_cfg = load_yaml(args.config)
    train_cfg = load_yaml(args.train_config)
    set_seed(train_cfg["global"]["seed"])

    hybrid_dir = Path(train_cfg["global"]["artifacts_dir"]) / "models" / "hybrid"
    gbm_path = hybrid_dir / "gbm.joblib"
    feat_path = hybrid_dir / "feature_names.joblib"
    X_test_path = hybrid_dir / "X_test.npy"

    if not (gbm_path.exists() and X_test_path.exists() and feat_path.exists()):
        log.error(
            f"missing artifacts in {hybrid_dir}; run scripts/train_hybrid.py first"
        )
        return

    gbm = joblib.load(gbm_path)
    feature_names = joblib.load(feat_path)
    X_test = np.load(X_test_path)

    out_dir = ensure_dir(eval_cfg["figures_dir"])
    res = explain_with_shap(
        gbm,
        X_test,
        feature_names,
        out_dir=out_dir,
        max_samples=eval_cfg["explainability"]["shap"]["max_samples"],
        seed=train_cfg["global"]["seed"],
    )
    # Also drop the importance CSV to tables/
    res["importance"].to_csv(eval_cfg["metrics_dir"] + "/shap_global_importance.csv", index=False)
    log.info("SHAP done")


if __name__ == "__main__":
    main()
