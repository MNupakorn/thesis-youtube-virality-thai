"""YouTube Data API v3 collector.

Implements the data-collection methodology described in the thesis:
1. Discover the top trending Thai channels via ``videos.list?chart=mostPopular&regionCode=TH``
   across 14 categories (the configured ``CATEGORY_IDS``).
2. For each channel, page ``search.list`` (or ``activities.list``) to enumerate
   uploads in the configured time window.
3. For each upload, batch-fetch ``videos.list?part=snippet,contentDetails,statistics,topicDetails``
   to fill in the metrics we need.

Output: a single pandas.DataFrame with the columns required by
``src/data_processing/clean.REQUIRED_COLUMNS`` plus a few extras.

Quota cost (rough):
- ``videos.list?chart=mostPopular`` = 1 unit per call
- ``search.list``                   = 100 units per call (50 results)
- ``videos.list?id=...``            = 1 unit per call (50 IDs/call)

The default config is conservative: 200 channels × ~50 videos/channel ≈ 10–15 k
``search.list`` calls = 1–1.5 M units. That **exceeds** the free 10 k/day quota,
so plan for either a ~1-month staggered collection or apply for a higher quota.

For thesis reproduction it's typically smarter to:
- Restrict to ~50 channels per category × 14 categories = 700 channels
- Use ``playlistItems`` on each channel's ``uploads`` playlist instead of ``search``
  (only 1 unit per call, 50 items per page) — this is what we implement here.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.utils import setup_logger

log = setup_logger("data.collect")

# 14 standard YouTube category IDs commonly used for Thailand-region trending
CATEGORY_IDS: list[str] = [
    "1",   # Film & Animation
    "2",   # Autos & Vehicles
    "10",  # Music
    "15",  # Pets & Animals
    "17",  # Sports
    "19",  # Travel & Events
    "20",  # Gaming
    "22",  # People & Blogs
    "23",  # Comedy
    "24",  # Entertainment
    "25",  # News & Politics
    "26",  # Howto & Style
    "27",  # Education
    "28",  # Science & Technology
]


def _client():
    from googleapiclient.discovery import build

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "YOUTUBE_API_KEY not set. Add it to .env or export it before running."
        )
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


def discover_thai_channels(
    yt,
    max_per_category: int = 50,
    region: str = "TH",
    category_ids: list[str] | None = None,
) -> dict[str, dict]:
    """Return ``{channel_id: {channel_title, category_id, ...}}`` from trending."""
    category_ids = category_ids or CATEGORY_IDS
    channels: dict[str, dict] = {}
    for cid in tqdm(category_ids, desc="discover"):
        try:
            req = yt.videos().list(
                part="snippet",
                chart="mostPopular",
                regionCode=region,
                videoCategoryId=cid,
                maxResults=min(max_per_category, 50),
            )
            resp = req.execute()
            for item in resp.get("items", []):
                ch_id = item["snippet"]["channelId"]
                if ch_id not in channels:
                    channels[ch_id] = {
                        "channel_id": ch_id,
                        "channel_title": item["snippet"]["channelTitle"],
                        "seed_category_id": cid,
                    }
        except Exception as e:
            log.warning(f"discover failed for category {cid}: {e}")
    log.info(f"discovered {len(channels)} unique channels across {len(category_ids)} categories")
    return channels


def _channel_uploads_playlist(yt, channel_id: str) -> str | None:
    """The ``contentDetails.relatedPlaylists.uploads`` ID for a channel."""
    resp = yt.channels().list(part="contentDetails,snippet,statistics", id=channel_id).execute()
    items = resp.get("items", [])
    if not items:
        return None
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def _channel_meta(yt, channel_ids: list[str]) -> dict[str, dict]:
    """Batch-fetch channel statistics (subscribers, videoCount, viewCount)."""
    out: dict[str, dict] = {}
    for i in range(0, len(channel_ids), 50):
        batch = channel_ids[i : i + 50]
        resp = yt.channels().list(part="snippet,statistics", id=",".join(batch)).execute()
        for item in resp.get("items", []):
            stats = item.get("statistics", {})
            out[item["id"]] = {
                "channel_title": item["snippet"]["title"],
                "subscriber_count": int(stats.get("subscriberCount", 0)) if not stats.get("hiddenSubscriberCount") else None,
                "channel_video_count": int(stats.get("videoCount", 0)),
                "channel_view_count": int(stats.get("viewCount", 0)),
            }
    return out


def list_channel_videos(
    yt,
    channel_id: str,
    published_after: str,
    published_before: str,
    max_videos_per_channel: int = 500,
) -> list[str]:
    """Return video_ids uploaded by ``channel_id`` in the configured window."""
    pl_id = _channel_uploads_playlist(yt, channel_id)
    if not pl_id:
        return []
    video_ids: list[str] = []
    pa = pd.Timestamp(published_after)
    pb = pd.Timestamp(published_before)
    page_token = None
    while True:
        resp = yt.playlistItems().list(
            part="contentDetails", playlistId=pl_id, maxResults=50, pageToken=page_token
        ).execute()
        for item in resp.get("items", []):
            cd = item["contentDetails"]
            published = pd.Timestamp(cd["videoPublishedAt"]) if "videoPublishedAt" in cd else None
            if published is None:
                continue
            if pa <= published.tz_convert("UTC").tz_localize(None) <= pb:
                video_ids.append(cd["videoId"])
        page_token = resp.get("nextPageToken")
        if not page_token or len(video_ids) >= max_videos_per_channel:
            break
    return video_ids


def fetch_video_details(yt, video_ids: list[str]) -> list[dict[str, Any]]:
    """Batch-fetch snippet+statistics+contentDetails+topicDetails for video IDs."""
    rows: list[dict[str, Any]] = []
    for i in tqdm(range(0, len(video_ids), 50), desc="videos.list"):
        batch = video_ids[i : i + 50]
        resp = yt.videos().list(
            part="snippet,statistics,contentDetails,topicDetails", id=",".join(batch)
        ).execute()
        for item in resp.get("items", []):
            sn = item.get("snippet", {})
            st = item.get("statistics", {})
            cd = item.get("contentDetails", {})
            td = item.get("topicDetails", {})
            rows.append(
                {
                    "video_id": item["id"],
                    "title": sn.get("title", ""),
                    "description": sn.get("description", ""),
                    "tags": "|".join(sn.get("tags", []) or []),
                    "category_id": int(sn.get("categoryId", 0)) if sn.get("categoryId") else None,
                    "category_title": None,  # filled later via lookup
                    "published_at": sn.get("publishedAt"),
                    "duration": cd.get("duration"),
                    "channel_id": sn.get("channelId"),
                    "channel_title": sn.get("channelTitle"),
                    "view_count": int(st.get("viewCount", 0)) if st.get("viewCount") else None,
                    "like_count": int(st.get("likeCount", 0)) if st.get("likeCount") else None,
                    "comment_count": int(st.get("commentCount", 0)) if st.get("commentCount") else None,
                    "definition": cd.get("definition"),
                    "caption": cd.get("caption"),
                    "licensed_content": cd.get("licensedContent"),
                    "topic_categories": "|".join(td.get("topicCategories", []) or []),
                    "default_language": sn.get("defaultLanguage"),
                    "default_audio_language": sn.get("defaultAudioLanguage"),
                    "collected_at": pd.Timestamp.utcnow().isoformat(),
                }
            )
    return rows


def _category_titles(yt, region: str = "TH") -> dict[int, str]:
    resp = yt.videoCategories().list(part="snippet", regionCode=region).execute()
    return {int(c["id"]): c["snippet"]["title"] for c in resp.get("items", [])}


def collect_videos(cfg: dict | None = None) -> pd.DataFrame:
    """End-to-end collector. Driven by configs/data.yaml -> collection.* (optional).

    Returns a DataFrame with all the required columns + extras.
    """
    cfg = cfg or {}
    coll_cfg = cfg.get("collection", {})
    region = coll_cfg.get("region", "TH")
    max_per_category = int(coll_cfg.get("max_channels_per_category", 50))
    published_after = coll_cfg.get("published_after", "2024-01-01T00:00:00Z")
    published_before = coll_cfg.get("published_before", "2026-01-01T00:00:00Z")
    max_videos_per_channel = int(coll_cfg.get("max_videos_per_channel", 500))

    yt = _client()

    log.info("Step 1: discovering trending Thai channels...")
    channels = discover_thai_channels(yt, max_per_category=max_per_category, region=region)
    channel_ids = list(channels.keys())

    log.info("Step 2: fetching channel meta...")
    ch_meta = _channel_meta(yt, channel_ids)

    log.info("Step 3: enumerating uploads in window...")
    all_video_ids: list[str] = []
    for ch_id in tqdm(channel_ids, desc="channels"):
        try:
            ids = list_channel_videos(
                yt,
                ch_id,
                published_after=published_after,
                published_before=published_before,
                max_videos_per_channel=max_videos_per_channel,
            )
            all_video_ids.extend(ids)
            time.sleep(0.05)
        except Exception as e:
            log.warning(f"failed to enumerate {ch_id}: {e}")

    log.info(f"total candidate videos: {len(all_video_ids)}")
    log.info("Step 4: fetching video details...")
    rows = fetch_video_details(yt, all_video_ids)
    df = pd.DataFrame(rows)

    log.info("Step 5: enriching channel stats + category titles...")
    cat_map = _category_titles(yt, region=region)
    df["category_title"] = df["category_id"].map(cat_map)
    for col in ("subscriber_count", "channel_video_count", "channel_view_count"):
        df[col] = df["channel_id"].map(lambda c: ch_meta.get(c, {}).get(col))

    log.info(f"collected DataFrame shape: {df.shape}")
    return df


def save_raw(df: pd.DataFrame, path) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8")
    log.info(f"saved raw data ({len(df)} rows) -> {path}")
