"""Engineered title features (length, punctuation, emoji, clickbait, code-switching)."""

from __future__ import annotations

import re

import pandas as pd

# Compile once
_THAI_RE = re.compile(r"[฀-๿]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"\d")
_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001f6ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa70-\U0001faff"
    "\U00002600-\U000027bf"
    "\U0001f1e6-\U0001f1ff"
    "]+",
    flags=re.UNICODE,
)
_HASHTAG_RE = re.compile(r"#\S+")
_MENTION_RE = re.compile(r"@\S+")
_BRACKET_RE = re.compile(r"[\[\(\{【〔《].+?[\]\)\}】〕》]")

# Clickbait / curiosity-gap lexicon (Thai + English)
CURIOSITY_GAP_WORDS = [
    # Thai
    "สิ่งนี้",
    "เรื่องนี้",
    "คน​นี้",
    "คนนี้",
    "ที่นี่",
    "แบบนี้",
    "อย่างนี้",
    "ทำไม",
    "ลับ",
    "ความลับ",
    "ห้ามดู",
    "ห้ามพลาด",
    "ต้องดู",
    "เปิดเผย",
    # English
    "this",
    "secret",
    "revealed",
    "exposed",
    "you won",
    "what happens",
    "must watch",
    "don't miss",
    "shocking",
]

HYPERBOLIC_WORDS = [
    # Thai
    "ที่สุด",
    "สุดยอด",
    "เทพ",
    "ดีที่สุด",
    "แรง",
    "ช็อก",
    "ช็อกโลก",
    "ระเบิด",
    "หนัก",
    "ไวรัล",
    "เด็ด",
    "เริ่ด",
    "พัง",
    # English
    "best",
    "ultimate",
    "ever",
    "epic",
    "insane",
    "crazy",
    "biggest",
    "fastest",
    "greatest",
    "shocking",
    "viral",
    "amazing",
    "unbelievable",
]


def _count_thai(s: str) -> int:
    return len(_THAI_RE.findall(s))


def _count_latin(s: str) -> int:
    return len(_LATIN_RE.findall(s))


def _emoji_count(s: str) -> int:
    return sum(len(m) for m in _EMOJI_RE.findall(s))


def _caps_ratio(s: str) -> float:
    latin = _LATIN_RE.findall(s)
    if not latin:
        return 0.0
    return sum(1 for c in latin if c.isupper()) / len(latin)


def _lexicon_score(s: str, words: list[str]) -> int:
    """Number of distinct lexicon entries matched in s (case-insensitive)."""
    s_lo = s.lower()
    return sum(1 for w in words if w in s_lo)


def add_title_features(df: pd.DataFrame, col: str = "title") -> pd.DataFrame:
    """Append engineered title features as new columns."""
    out = df.copy()
    s = out[col].fillna("").astype(str)

    out["t_char_len"] = s.str.len()
    out["t_word_len"] = s.str.split().str.len().fillna(0).astype(int)
    out["t_exclaim_n"] = s.str.count(r"!")
    out["t_question_n"] = s.str.count(r"\?")
    out["t_digit_n"] = s.map(lambda x: len(_DIGIT_RE.findall(x)))
    out["t_emoji_n"] = s.map(_emoji_count)
    out["t_hashtag_n"] = s.map(lambda x: len(_HASHTAG_RE.findall(x)))
    out["t_mention_n"] = s.map(lambda x: len(_MENTION_RE.findall(x)))
    out["t_has_brackets"] = s.map(lambda x: int(bool(_BRACKET_RE.search(x))))
    out["t_has_number"] = (out["t_digit_n"] > 0).astype(int)
    out["t_caps_ratio"] = s.map(_caps_ratio)

    out["t_thai_chars"] = s.map(_count_thai)
    out["t_latin_chars"] = s.map(_count_latin)
    out["t_code_switching"] = ((out["t_thai_chars"] > 0) & (out["t_latin_chars"] > 0)).astype(int)

    out["t_curiosity_gap"] = s.map(lambda x: _lexicon_score(x, CURIOSITY_GAP_WORDS))
    out["t_hyperbolic"] = s.map(lambda x: _lexicon_score(x, HYPERBOLIC_WORDS))

    return out


def add_temporal_features(df: pd.DataFrame, col: str = "published_at") -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[col], utc=True, errors="coerce")
    # Convert to Bangkok time (+07:00) for human-meaningful hour
    bkk = ts.dt.tz_convert("Asia/Bangkok")
    out["pub_hour"] = bkk.dt.hour.fillna(-1).astype(int)
    out["pub_dow"] = bkk.dt.dayofweek.fillna(-1).astype(int)
    out["pub_is_weekend"] = (bkk.dt.dayofweek >= 5).astype("Int64").fillna(0).astype(int)
    out["pub_is_peak"] = bkk.dt.hour.between(19, 22, inclusive="both").fillna(False).astype(int)
    out["pub_month"] = bkk.dt.month.fillna(-1).astype(int)
    return out


def add_channel_features(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np

    out = df.copy()
    for c in ("subscriber_count", "channel_video_count"):
        if c in out.columns:
            out[f"log_{c}"] = np.log1p(out[c].fillna(0).clip(lower=0))
    if "channel_view_count" in out.columns:
        out["log_channel_view_count"] = np.log1p(
            out["channel_view_count"].fillna(0).clip(lower=0)
        )
    return out


def add_video_features(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np

    out = df.copy()
    if "duration_sec" in out.columns:
        out["log_duration_sec"] = np.log1p(out["duration_sec"].fillna(0).clip(lower=0))
        out["is_short"] = (out["duration_sec"] <= 60).astype(int)
    return out


def build_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-call convenience to attach the full structured feature set."""
    out = add_title_features(df)
    out = add_temporal_features(out)
    out = add_channel_features(out)
    out = add_video_features(out)
    return out


STRUCTURED_FEATURE_COLUMNS = [
    # title
    "t_char_len",
    "t_word_len",
    "t_exclaim_n",
    "t_question_n",
    "t_digit_n",
    "t_emoji_n",
    "t_hashtag_n",
    "t_mention_n",
    "t_has_brackets",
    "t_has_number",
    "t_caps_ratio",
    "t_thai_chars",
    "t_latin_chars",
    "t_code_switching",
    "t_curiosity_gap",
    "t_hyperbolic",
    # channel
    "log_subscriber_count",
    "log_channel_video_count",
    # video
    "duration_sec",
    "log_duration_sec",
    "is_short",
    # temporal
    "pub_hour",
    "pub_dow",
    "pub_is_weekend",
    "pub_is_peak",
    "pub_month",
    # category
    "category_id",
]
