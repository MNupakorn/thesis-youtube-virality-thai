"""Tests for splits: no channel leakage, time ordering, class balance preserved."""

import numpy as np
import pandas as pd
import pytest

from src.data_processing.splits import (
    apply_train_only_imbalance_strategy,
    make_splits,
)


@pytest.fixture
def toy_split_df():
    rng = np.random.default_rng(0)
    rows = []
    start = pd.Timestamp("2024-01-01", tz="UTC")
    for ch in [f"ch{i}" for i in range(40)]:
        for k in range(30):
            rows.append(
                {
                    "video_id": f"{ch}_{k}",
                    "channel_id": ch,
                    "published_at": start + pd.Timedelta(days=int(rng.integers(0, 700))),
                    "label_viral": int(rng.random() < 0.1),
                }
            )
    return pd.DataFrame(rows)


def test_hybrid_split_test_is_latest(toy_split_df):
    cfg = {"scheme": "hybrid_group_time", "train_pct": 0.7, "val_pct": 0.15, "test_pct": 0.15, "seed": 42}
    out = make_splits(toy_split_df, cfg)
    test_max = out.loc[out["split"] != "test", "published_at"].max()
    test_min = out.loc[out["split"] == "test", "published_at"].min()
    # test set should start at or after the rest's max
    assert test_min >= test_max - pd.Timedelta(seconds=1)


def test_group_only_no_channel_leak(toy_split_df):
    cfg = {"scheme": "group_only", "train_pct": 0.7, "val_pct": 0.15, "test_pct": 0.15, "seed": 42}
    out = make_splits(toy_split_df, cfg)
    train_ch = set(out.loc[out["split"] == "train", "channel_id"])
    val_ch = set(out.loc[out["split"] == "val", "channel_id"])
    test_ch = set(out.loc[out["split"] == "test", "channel_id"])
    assert train_ch.isdisjoint(val_ch)
    assert train_ch.isdisjoint(test_ch)
    assert val_ch.isdisjoint(test_ch)


def test_undersample_only_touches_train(toy_split_df):
    cfg = {"scheme": "hybrid_group_time", "train_pct": 0.7, "val_pct": 0.15, "test_pct": 0.15, "seed": 42}
    out = make_splits(toy_split_df, cfg)
    val_before = (out["split"] == "val").sum()
    test_before = (out["split"] == "test").sum()
    out2 = apply_train_only_imbalance_strategy(out, strategy="undersample", undersample_ratio=2.0)
    assert (out2["split"] == "val").sum() == val_before
    assert (out2["split"] == "test").sum() == test_before
