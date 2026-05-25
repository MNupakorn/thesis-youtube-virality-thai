"""YouTube Data API v3 collector (stub).

Replace `collect_videos()` with the user-supplied collection script. The function
must return a pandas.DataFrame with at minimum the columns listed in
``REQUIRED_COLUMNS`` of ``src/data_processing/clean.py``.

Usage from the CLI:
    python scripts/collect_data.py --config configs/data.yaml
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.utils import setup_logger

log = setup_logger("data.collect")


def _client():
    """Build a YouTube Data API v3 client. Requires YOUTUBE_API_KEY env var."""
    from googleapiclient.discovery import build

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "YOUTUBE_API_KEY is not set. Add it to .env or export it before running."
        )
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


def collect_videos(*args, **kwargs) -> pd.DataFrame:
    """Drop-in replacement target. Until the user shares the original collector,
    this raises a clear error explaining what the function should return."""
    raise NotImplementedError(
        "Drop the original YouTube collection script into this module. "
        "It should return a pandas.DataFrame containing the columns: "
        "video_id, title, description, tags, category_id, category_title, "
        "published_at, duration, channel_id, channel_title, subscriber_count, "
        "channel_video_count, view_count, like_count, comment_count "
        "(plus any optional metadata)."
    )


def save_raw(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8")
    log.info(f"saved raw data ({len(df)} rows) -> {path}")
