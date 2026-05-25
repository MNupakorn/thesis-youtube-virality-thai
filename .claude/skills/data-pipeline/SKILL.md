---
name: data-pipeline
description: >
  Use when working with the YouTube Thai videos dataset: collection via API, cleaning,
  label generation (ViralityIndex), train/val/test splitting (channel-grouped + time-aware),
  feature engineering (structural, sentiment, embeddings, TF-IDF), or any change to
  data/raw -> data/interim -> data/processed flow. Triggers: "prepare data", "regenerate
  labels", "add a feature", "split the data", "fix a leakage", "the dataset changed".
user-invocable: false
---

# Data Pipeline

The dataset is the foundation. Every shortcut here invalidates the thesis.

## 1. Data Flow

```
data/raw/youtube_thai_videos.parquet
       │  (cleaning: drop empties, normalize text, parse dates)
       ▼
data/interim/youtube_clean.parquet
       │  (ViralityIndex per-channel z-scores → top-decile label)
       ▼
data/interim/youtube_labeled.parquet
       │  (channel-grouped train/val + time-aware test)
       ▼
data/interim/youtube_split.parquet
       │  (sentiment features from frozen WangchanBERTa sentiment model)
       ▼  (cached at data/interim/sentiment_cache.parquet, keyed by video_id)
       │
       │  (title CLS embeddings from frozen WangchanBERTa)
       ▼  (cached at data/interim/title_embeddings.npy + .ids.parquet)
       │
       ▼
data/processed/dataset_with_labels.parquet   ← canonical input for ALL training
```

Run with: `make prepare`. See `scripts/prepare_data.py`.

## 2. Label Definition (frozen)

Per channel, compute:
- `z_views = z(log(views + 1))`
- `z_engage = z((likes + comments) / max(views, 1))`
- `ViralityIndex = z_views + z_engage`
- `label_viral = 1 if ViralityIndex >= per-channel 90th percentile else 0`

Resulting positive rate ≈ 10.16% globally.

**Why per-channel:** raw views correlate with channel size. Per-channel z-score asks "is this video viral *for this channel*?" — the question stakeholders actually care about.

Do not change this without an entry in `.claude/memory/decisions.md`.

## 3. Splits (frozen)

```
test  = videos with published_at in the latest 15% of the time range
train ∪ val = the remaining 85%, channel-grouped (no channel in both)
train : val ≈ 70 : 15 of the global total
```

After undersampling (3:1 negative:positive on train only):
- train: 6,964 rows, 25.0% positive
- val: 3,143 rows, 10.4% positive
- test: 3,510 rows, 8.8% positive

**Why time-aware test:** in deployment we predict virality of *future* videos using past data. Random shuffle would leak time-correlated patterns (trends, seasons, channel growth phases).

**Why channel-grouped train/val:** prevents the model from memorizing channel-specific patterns and getting credit for it as language signal.

## 4. Feature Caching

Sentiment and title embeddings are expensive. They are cached keyed by `video_id`:

- `data/interim/sentiment_cache.parquet` — one row per video_id, columns: `sent_pos, sent_neu, sent_neg, sent_q, arousal, valence, polar`
- `data/interim/title_embeddings.npy` — `(N, 768)` float32, ordered by `data/interim/title_embeddings.ids.parquet`

When the canonical input changes, regenerate caches:
```bash
rm data/interim/sentiment_cache.parquet data/interim/title_embeddings.*
make prepare
```

## 5. Feature Sets

Five named feature sets, used across baselines and the hybrid model:

| Name | Components |
|---|---|
| `structured` | 16 hand-engineered title features + channel/temporal numerical metadata |
| `tfidf` | char + word TF-IDF on title (max_features tuned in `configs/features.yaml`) |
| `structured_plus_tfidf` | both |
| `structured_plus_sentiment` | structured + 7-dim sentiment |
| `all` | structured + tfidf + sentiment |

The hybrid model adds a 6th: `cls_emb + sent + struct` (frozen 768-dim CLS + sentiment + structured).

## 6. The 16 Structural Features

`src/features/structural.py` — title-level:

1. char_length, 2. token_length, 3. emoji_count, 4. caps_ratio,
5. q_mark_count, 6. ex_mark_count, 7. has_number, 8. curiosity_th, 9. curiosity_en,
10. hyperbolic_ratio, 11. code_switch_ratio, 12. starts_with_number, 13. has_year,
14. has_emoji_face, 15. has_money_symbol, 16. avg_word_len.

Curiosity lexicons are in `src/features/lexicons/` (Thai + English clickbait words).

## 7. Common Pitfalls

- **pyarrow object-column error** — cast all `object` columns to `str` before parquet save.
- **NaN in channel z-scores** — channels with <10 videos get NaN z. Drop those channels at the cleaning stage; do not impute.
- **Duplicate video_id across raw collection runs** — dedupe by `video_id` keeping the latest `fetched_at`.
- **TF-IDF fit on train+val+test** — only fit on train. Use the fitted vectorizer to transform val/test.

## 8. Data Sanity Checks (always run after `make prepare`)

```python
df = pd.read_parquet("data/processed/dataset_with_labels.parquet")
assert df["split"].value_counts().to_dict() == {"train": 6964, "val": 3143, "test": 3510}
assert df.groupby("split")["label_viral"].mean().round(3).to_dict() == {
    "train": 0.250, "val": 0.104, "test": 0.088
}
# No channel in both train and val:
assert set(df[df.split=="train"]["channel_id"]) & set(df[df.split=="val"]["channel_id"]) == set()
# Test is strictly after train+val:
assert df[df.split=="test"]["published_at"].min() >= df[df.split!="test"]["published_at"].max()
```

If any assertion fails, **stop**. Do not train on a broken split.
