"""ViralityIndex computation + binary label.

Definition (matches the thesis with a small upgrade):

    log_views        = log1p(view_count)
    engagement_rate  = (like_count + comment_count) / max(view_count, 1)

For each channel c:
    z_log_views(v)   = (log_views(v) − mean_c(log_views)) / std_c(log_views)
    z_engagement(v)  = (engagement_rate(v) − mean_c(engagement_rate)) / std_c(engagement_rate)

    ViralityIndex(v) = z_log_views(v) + z_engagement(v)
    label_viral(v)   = 1 if ViralityIndex(v) ≥ percentile_90(ViralityIndex_c) else 0

Channels with fewer than `min_videos_per_channel` videos are dropped from
training (their per-channel statistics are unreliable). A secondary global
label can be computed as well for ablation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import setup_logger

log = setup_logger("data.label")


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def compute_engagement_rate(df: pd.DataFrame) -> pd.Series:
    return (df["like_count"].fillna(0) + df["comment_count"].fillna(0)) / df["view_count"].clip(
        lower=1
    )


def compute_virality_index(
    df: pd.DataFrame,
    scope: str = "per_channel",
    min_videos_per_channel: int = 10,
) -> pd.DataFrame:
    """Add columns: log_views, engagement_rate, z_log_views, z_engagement, virality_index."""
    out = df.copy()
    out["log_views"] = np.log1p(out["view_count"].clip(lower=0))
    out["engagement_rate"] = compute_engagement_rate(out)

    if scope == "per_channel":
        # Filter sparse channels
        counts = out.groupby("channel_id")["video_id"].transform("count")
        before = len(out)
        out = out[counts >= min_videos_per_channel].copy()
        log.info(
            f"per_channel zscore: dropped {before - len(out)} rows from channels with <"
            f" {min_videos_per_channel} videos"
        )
        out["z_log_views"] = out.groupby("channel_id")["log_views"].transform(_zscore)
        out["z_engagement"] = out.groupby("channel_id")["engagement_rate"].transform(_zscore)
    elif scope == "global":
        out["z_log_views"] = _zscore(out["log_views"])
        out["z_engagement"] = _zscore(out["engagement_rate"])
    else:
        raise ValueError(f"unknown z_score_scope: {scope}")

    out["virality_index"] = out["z_log_views"] + out["z_engagement"]
    return out.reset_index(drop=True)


def assign_binary_label(
    df: pd.DataFrame,
    percentile: float = 90.0,
    per_channel: bool = True,
    label_col: str = "label_viral",
) -> pd.DataFrame:
    """Mark top-`percentile` of ViralityIndex (per-channel or globally) as viral (1)."""
    out = df.copy()
    q = percentile / 100.0
    if per_channel:
        thresholds = out.groupby("channel_id")["virality_index"].transform(
            lambda s: s.quantile(q)
        )
        out[label_col] = (out["virality_index"] >= thresholds).astype(int)
    else:
        threshold = out["virality_index"].quantile(q)
        out[label_col] = (out["virality_index"] >= threshold).astype(int)
    log.info(
        f"label '{label_col}' positive rate: {out[label_col].mean():.3%} "
        f"({int(out[label_col].sum())}/{len(out)})"
    )
    return out


def compute_labels(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """End-to-end label engineering driven by configs/data.yaml -> labels.*."""
    scope = cfg.get("z_score_scope", "per_channel")
    out = compute_virality_index(df, scope=scope)
    out = assign_binary_label(
        out,
        percentile=cfg.get("threshold_percentile", 90),
        per_channel=(scope == "per_channel"),
        label_col="label_viral",
    )
    if cfg.get("also_compute_global_label", False):
        out = assign_binary_label(
            out,
            percentile=cfg.get("threshold_percentile", 90),
            per_channel=False,
            label_col="label_viral_global",
        )
    return out
