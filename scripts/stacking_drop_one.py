"""Drop-one ablation on the stacking ensemble.

For each base model, refit the meta-LR without that model and report the
test ROC-AUC drop. Larger drops = more important contributions.

Output: reports/tables/stacking_drop_one_ablation.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluation.metrics import bootstrap_ci, compute_metrics
from src.utils import ensure_dir, setup_logger

log = setup_logger("scripts.stacking_drop_one")


def main() -> None:
    pred_root = Path("reports/artifacts/predictions")
    preds: dict[str, pd.DataFrame] = {}
    for p in sorted(pred_root.rglob("*.parquet")):
        if "stacking" in p.parts or "ablation" in p.parts:
            continue
        preds[p.relative_to(pred_root).with_suffix("").as_posix()] = pd.read_parquet(p)

    keys = sorted(preds)

    def matrix(split: str, ks: list[str]) -> pd.DataFrame:
        base = None
        for k in ks:
            sub = preds[k][preds[k]["split"] == split][["video_id", "y_true", "y_proba"]].rename(
                columns={"y_proba": k}
            )
            base = (
                sub.rename(columns={"y_true": "y"})
                if base is None
                else base.merge(sub.drop(columns=["y_true"]), on="video_id", how="inner")
            )
        return base

    val_full = matrix("val", keys)
    test_full = matrix("test", keys)
    yv = val_full["y"].to_numpy()
    yt = test_full["y"].to_numpy()

    # Baseline: full stacking
    lr = LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced", random_state=42)
    lr.fit(val_full[keys].to_numpy(), yv)
    pt = lr.predict_proba(test_full[keys].to_numpy())[:, 1]
    pe_full, lo_full, hi_full = bootstrap_ci(yt, pt, "roc_auc", n_iter=1000)
    log.info(f"FULL: AUC = {pe_full:.4f} [{lo_full:.4f}, {hi_full:.4f}]")

    rows = [
        {
            "removed": "(none)",
            "n_models": len(keys),
            "roc_auc": pe_full,
            "ci_lo": lo_full,
            "ci_hi": hi_full,
            "delta": 0.0,
        }
    ]
    for drop in keys:
        ks = [k for k in keys if k != drop]
        lr2 = LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced", random_state=42)
        lr2.fit(val_full[ks].to_numpy(), yv)
        pt2 = lr2.predict_proba(test_full[ks].to_numpy())[:, 1]
        pe, lo, hi = bootstrap_ci(yt, pt2, "roc_auc", n_iter=400)
        rows.append(
            {
                "removed": drop,
                "n_models": len(ks),
                "roc_auc": pe,
                "ci_lo": lo,
                "ci_hi": hi,
                "delta": pe - pe_full,
            }
        )
        log.info(f"  drop {drop:55s} -> AUC {pe:.4f} (delta={pe - pe_full:+.4f})")

    df = pd.DataFrame(rows).sort_values("delta")
    out = ensure_dir("reports/tables") / "stacking_drop_one_ablation.csv"
    df.to_csv(out, index=False)
    log.info(f"saved {out}")
    log.info(f"\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()
