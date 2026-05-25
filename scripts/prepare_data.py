"""End-to-end data preparation: clean -> labels -> features -> splits -> save.

Outputs a single processed parquet under ``data/processed/`` ready for any model:

    data/processed/dataset_with_labels.parquet
    data/processed/title_embeddings.npy   (optional, if features.transformer_embedding.enabled)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_processing.clean import clean_dataframe
from src.data_processing.labels import compute_labels
from src.data_processing.splits import (
    apply_train_only_imbalance_strategy,
    make_splits,
)
from src.features.sentiment import compute_or_load_sentiment
from src.features.structural import build_structured_features
from src.utils import ensure_dir, load_yaml, set_seed, setup_logger

log = setup_logger("scripts.prepare")


def _read_raw(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--features-config", default="configs/features.yaml")
    ap.add_argument(
        "--skip-sentiment",
        action="store_true",
        help="don't compute sentiment features (useful for fast smoke tests)",
    )
    ap.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="don't compute title transformer embeddings",
    )
    args = ap.parse_args()

    data_cfg = load_yaml(args.data_config)
    feat_cfg = load_yaml(args.features_config)
    set_seed(data_cfg["splits"].get("seed", 42))

    raw_path = Path(data_cfg["paths"]["raw_dir"]) / data_cfg["raw_filename"]
    proc_dir = ensure_dir(data_cfg["paths"]["processed_dir"])

    log.info(f"reading raw data from {raw_path}")
    df = _read_raw(raw_path)
    log.info(f"raw shape: {df.shape}")

    df = clean_dataframe(df, data_cfg.get("clean", {}))
    df = compute_labels(df, data_cfg["labels"])
    df = build_structured_features(df)
    df = make_splits(df, data_cfg["splits"])

    df = apply_train_only_imbalance_strategy(
        df,
        strategy=data_cfg["imbalance"]["strategy"],
        undersample_ratio=data_cfg["imbalance"].get("undersample_ratio", 3.0),
        random_state=data_cfg["imbalance"].get("random_state", 42),
    )

    if not args.skip_sentiment and feat_cfg.get("sentiment"):
        sc = feat_cfg["sentiment"]
        df = compute_or_load_sentiment(
            df,
            cache_path=sc["cache_dir"],
            model_name=sc.get("finetune_repo", sc["model_name"]),
            max_length=sc.get("max_length", 128),
            batch_size=sc.get("batch_size", 64),
            device=sc.get("device", "auto"),
        )

    if not args.skip_embeddings and feat_cfg.get("transformer_embedding", {}).get("enabled"):
        from src.features.transformer_embed import compute_or_load_embeddings

        ec = feat_cfg["transformer_embedding"]
        compute_or_load_embeddings(
            df,
            cache_path=ec["cache_dir"],
            model_name=ec["model_name"],
            pooling=ec.get("pooling", "cls"),
            max_length=ec.get("max_length", 64),
            batch_size=64,
            device="auto",
        )

    out_path = proc_dir / "dataset_with_labels.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"saved processed dataset -> {out_path}  ({df.shape})")

    summary = (
        df.groupby("split")
        .agg(n=("video_id", "count"), pos_rate=("label_viral", "mean"))
        .round(4)
    )
    log.info(f"final split summary:\n{summary}")


if __name__ == "__main__":
    main()
