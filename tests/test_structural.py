"""Tests for engineered title features."""

import pandas as pd

from src.features.structural import (
    STRUCTURED_FEATURE_COLUMNS,
    add_title_features,
    build_structured_features,
)


def test_title_features_basic():
    df = pd.DataFrame(
        {
            "title": [
                "สิ่งนี้คือสิ่งที่ดีที่สุด!!! 100%",
                "Normal English title",
                "Mixed ภาษา English title 😀😀 #hashtag",
                "",
            ]
        }
    )
    out = add_title_features(df)
    assert out["t_exclaim_n"].iloc[0] == 3
    assert out["t_has_number"].iloc[0] == 1
    assert out["t_curiosity_gap"].iloc[0] >= 1
    assert out["t_hyperbolic"].iloc[0] >= 1
    assert out["t_thai_chars"].iloc[1] == 0
    assert out["t_latin_chars"].iloc[1] > 0
    assert out["t_code_switching"].iloc[2] == 1
    assert out["t_emoji_n"].iloc[2] >= 2
    assert out["t_hashtag_n"].iloc[2] == 1
    assert out["t_char_len"].iloc[3] == 0


def test_build_structured_features_runs_end_to_end():
    df = pd.DataFrame(
        {
            "title": ["ทดสอบไวรัล!", "Test viral!"],
            "published_at": pd.to_datetime(["2025-04-01 20:00:00", "2025-04-02 09:00:00"], utc=True),
            "subscriber_count": [1000, 200_000],
            "channel_video_count": [50, 500],
            "duration_sec": [45, 600],
        }
    )
    out = build_structured_features(df)
    for c in [
        "t_char_len",
        "pub_hour",
        "pub_is_weekend",
        "log_subscriber_count",
        "log_duration_sec",
        "is_short",
    ]:
        assert c in out.columns


def test_structured_feature_columns_exist_after_build():
    df = pd.DataFrame(
        {
            "title": ["ทดสอบ"],
            "published_at": pd.to_datetime(["2025-04-01"], utc=True),
            "subscriber_count": [1],
            "channel_video_count": [1],
            "duration_sec": [10],
            "category_id": [10],
        }
    )
    out = build_structured_features(df)
    missing = [c for c in STRUCTURED_FEATURE_COLUMNS if c not in out.columns]
    assert not missing, f"missing columns: {missing}"
