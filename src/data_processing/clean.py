"""Cleaning + canonicalization for the raw YouTube dump."""

from __future__ import annotations

import re
from typing import Any

import isodate
import pandas as pd

from src.utils import setup_logger

log = setup_logger("data.clean")

REQUIRED_COLUMNS = [
    "video_id",
    "title",
    "category_id",
    "published_at",
    "duration",
    "channel_id",
    "subscriber_count",
    "channel_video_count",
    "view_count",
    "like_count",
    "comment_count",
]


def _iso_duration_to_seconds(raw: Any) -> float:
    """ISO 8601 duration (e.g. 'PT3M42S') → seconds. NaN-safe."""
    if not isinstance(raw, str) or not raw.startswith("P"):
        return float("nan")
    try:
        return isodate.parse_duration(raw).total_seconds()
    except Exception:
        return float("nan")


def _coerce_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Float64")


def clean_dataframe(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply the cleaning rules from configs/data.yaml -> clean.* section."""
    n0 = len(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    df = df.copy()

    # Whitespace + obvious garbage
    df["title"] = df["title"].astype("string").str.strip()
    if "description" in df.columns:
        df["description"] = df["description"].astype("string").fillna("")
    if "tags" in df.columns:
        df["tags"] = df["tags"].astype("string").fillna("")

    # Numerical coercion
    for col in [
        "view_count",
        "like_count",
        "comment_count",
        "subscriber_count",
        "channel_video_count",
        "category_id",
    ]:
        df[col] = _coerce_int(df[col])

    # Duration → seconds
    df["duration_sec"] = df["duration"].map(_iso_duration_to_seconds)

    # published_at → tz-aware datetime
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")

    # Drop rows with missing essentials
    before = len(df)
    df = df.dropna(subset=["video_id", "title", "channel_id", "published_at", "view_count"])
    log.info(f"dropped {before - len(df)} rows w/ missing essentials")

    # Title length filter
    min_chars = cfg.get("min_title_length_chars", 1)
    df = df[df["title"].str.len() >= min_chars]

    # De-duplicate video_id (keep the most recently collected if available)
    if cfg.get("drop_duplicate_video_ids", True):
        if "collected_at" in df.columns:
            df = df.sort_values("collected_at").drop_duplicates("video_id", keep="last")
        else:
            df = df.drop_duplicates("video_id", keep="first")

    # Fill engagement NaNs with 0 (channels may disable likes/comments)
    df["like_count"] = df["like_count"].fillna(0)
    df["comment_count"] = df["comment_count"].fillna(0)

    log.info(f"clean: {n0} → {len(df)} rows  ({len(df) / max(n0, 1):.1%} kept)")
    return df.reset_index(drop=True)
