"""Tests for ViralityIndex + binary label."""

import numpy as np
import pandas as pd
import pytest

from src.data_processing.labels import (
    assign_binary_label,
    compute_engagement_rate,
    compute_virality_index,
)


@pytest.fixture
def toy_df():
    rng = np.random.default_rng(0)
    rows = []
    for ch in ["A", "B"]:
        for i in range(20):
            rows.append(
                {
                    "video_id": f"{ch}_{i}",
                    "channel_id": ch,
                    "view_count": int(10 ** rng.uniform(2, 6)),
                    "like_count": int(rng.integers(0, 1000)),
                    "comment_count": int(rng.integers(0, 200)),
                }
            )
    return pd.DataFrame(rows)


def test_engagement_rate_is_finite(toy_df):
    er = compute_engagement_rate(toy_df)
    assert (er >= 0).all()
    assert np.isfinite(er).all()


def test_virality_index_per_channel_zero_mean(toy_df):
    vi = compute_virality_index(toy_df, scope="per_channel", min_videos_per_channel=10)
    for ch, sub in vi.groupby("channel_id"):
        assert abs(sub["z_log_views"].mean()) < 1e-6
        assert abs(sub["z_engagement"].mean()) < 1e-6


def test_top_decile_label_is_ten_percent_per_channel(toy_df):
    vi = compute_virality_index(toy_df, scope="per_channel", min_videos_per_channel=10)
    labeled = assign_binary_label(vi, percentile=90, per_channel=True)
    rates = labeled.groupby("channel_id")["label_viral"].mean()
    # Each channel has 20 videos; threshold @ 90th percentile -> exactly 2 are >=, so 10%
    for r in rates:
        assert 0.05 <= r <= 0.20
