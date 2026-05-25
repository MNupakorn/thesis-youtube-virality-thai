"""Train / Validation / Test splitting.

Three schemes:
- ``group_only``: GroupShuffleSplit by channel_id to prevent leakage of
  the same channel across splits.
- ``time_only``:  Latest videos by ``published_at`` form the test set.
- ``hybrid_group_time`` (default & recommended):
    1. Reserve the latest ``test_pct`` of videos chronologically as the
       held-out test set (mimics deployment — predict virality of *future*
       videos using past data).
    2. From the remaining (older) data, do a GroupShuffleSplit by
       ``channel_id`` to form train / val. This prevents a channel's
       videos from leaking from train into val and inflating performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.utils import setup_logger

log = setup_logger("data.splits")


def split_hybrid_group_time(
    df: pd.DataFrame,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
    test_pct: float = 0.15,
    seed: int = 42,
    time_col: str = "published_at",
    group_col: str = "channel_id",
) -> pd.DataFrame:
    """Return df with a new 'split' column ∈ {'train','val','test'}."""
    assert abs(train_pct + val_pct + test_pct - 1.0) < 1e-6, "split percentages must sum to 1"

    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    n_test = int(round(n * test_pct))
    test_idx = df.index[-n_test:]
    rest = df.loc[df.index.difference(test_idx)].copy()

    # group split on remainder
    rel_val_pct = val_pct / (train_pct + val_pct)
    gss = GroupShuffleSplit(n_splits=1, test_size=rel_val_pct, random_state=seed)
    train_pos, val_pos = next(gss.split(rest, groups=rest[group_col]))
    train_idx = rest.index[train_pos]
    val_idx = rest.index[val_pos]

    out = df.copy()
    out["split"] = "train"
    out.loc[val_idx, "split"] = "val"
    out.loc[test_idx, "split"] = "test"

    summary = (
        out.groupby("split").agg(
            n=("video_id", "count"),
            n_channels=(group_col, "nunique"),
            pos_rate=("label_viral", "mean"),
        )
    )
    log.info(f"split summary:\n{summary}")
    return out


def split_group_only(
    df: pd.DataFrame,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
    test_pct: float = 0.15,
    seed: int = 42,
    group_col: str = "channel_id",
) -> pd.DataFrame:
    """Pure GroupShuffleSplit on channel_id, no temporal control."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_pct, random_state=seed)
    rest_pos, test_pos = next(gss1.split(df, groups=df[group_col]))
    test_idx = df.index[test_pos]
    rest = df.iloc[rest_pos]

    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=val_pct / (train_pct + val_pct), random_state=seed
    )
    train_pos, val_pos = next(gss2.split(rest, groups=rest[group_col]))
    train_idx = rest.index[train_pos]
    val_idx = rest.index[val_pos]

    out = df.copy()
    out["split"] = "train"
    out.loc[val_idx, "split"] = "val"
    out.loc[test_idx, "split"] = "test"
    return out


def split_time_only(
    df: pd.DataFrame,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
    test_pct: float = 0.15,
    time_col: str = "published_at",
) -> pd.DataFrame:
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    n_train = int(round(n * train_pct))
    n_val = int(round(n * val_pct))
    out = df.copy()
    out["split"] = "test"
    out.iloc[:n_train, out.columns.get_loc("split")] = "train"
    out.iloc[n_train : n_train + n_val, out.columns.get_loc("split")] = "val"
    return out


def make_splits(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Dispatch on configs/data.yaml -> splits.scheme."""
    scheme = cfg.get("scheme", "hybrid_group_time")
    kwargs = dict(
        train_pct=cfg["train_pct"],
        val_pct=cfg["val_pct"],
        test_pct=cfg["test_pct"],
        seed=cfg.get("seed", 42),
    )
    if scheme == "hybrid_group_time":
        return split_hybrid_group_time(df, **kwargs)
    if scheme == "group_only":
        return split_group_only(df, **kwargs)
    if scheme == "time_only":
        kwargs.pop("seed", None)
        return split_time_only(df, **kwargs)
    raise ValueError(f"unknown split scheme: {scheme}")


def apply_train_only_imbalance_strategy(
    df: pd.DataFrame,
    strategy: str = "class_weight_plus_undersample",
    undersample_ratio: float = 3.0,
    random_state: int = 42,
    label_col: str = "label_viral",
) -> pd.DataFrame:
    """Apply imbalance handling ONLY on rows where split=='train'. Val/test untouched."""
    train = df[df["split"] == "train"].copy()
    rest = df[df["split"] != "train"].copy()
    pos = train[train[label_col] == 1]
    neg = train[train[label_col] == 0]

    if strategy in ("undersample", "class_weight_plus_undersample"):
        target_neg = int(min(len(neg), len(pos) * undersample_ratio))
        if target_neg < len(neg):
            neg = neg.sample(target_neg, random_state=random_state)
            log.info(
                f"undersample: neg {len(train) - len(pos)} → {len(neg)} "
                f"(target ratio {undersample_ratio}:1)"
            )

    new_train = pd.concat([pos, neg]).sample(frac=1.0, random_state=random_state)
    out = pd.concat([new_train, rest]).reset_index(drop=True)
    return out


def class_weight(df_train: pd.DataFrame, label_col: str = "label_viral") -> dict[int, float]:
    """Inverse-frequency weights for class_weight= keyword across sklearn / torch."""
    counts = df_train[label_col].value_counts()
    total = counts.sum()
    return {int(c): float(total / (len(counts) * counts[c])) for c in counts.index}


__all__ = [
    "make_splits",
    "apply_train_only_imbalance_strategy",
    "class_weight",
    "split_hybrid_group_time",
    "split_group_only",
    "split_time_only",
]
