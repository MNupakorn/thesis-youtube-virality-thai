# Architecture Snapshot

**As of:** 2026-05-26
**Canonical dataset:** `data/processed/dataset_with_labels.parquet` (sha256: TBD on next regenerate)

## Module Map

```
src/
├── data_collection/
│   ├── youtube_api.py          # YouTube Data API v3 collector (uses YOUTUBE_API_KEY)
│   └── __init__.py
├── data_processing/
│   ├── clean.py                # text normalization, drop empties, parse dates
│   ├── labels.py               # ViralityIndex (per-channel z), top-decile label
│   └── splits.py               # channel-grouped train/val + time-aware test
├── features/
│   ├── structural.py           # 16 hand-engineered title features
│   ├── lexicons/               # Thai + English clickbait word lists
│   ├── tfidf.py                # char + word TF-IDF (fit on train only)
│   ├── sentiment.py            # frozen WangchanBERTa-sentiment extractor
│   └── transformer_embed.py    # frozen WangchanBERTa CLS embeddings (768-dim)
├── models/
│   ├── baselines.py            # LR / LightGBM / XGBoost across 5 feature sets
│   ├── transformer_finetune.py # full FT for encoders + QLoRA path for decoders (unused)
│   └── hybrid.py               # MLP + LightGBM heads on fused features
├── evaluation/
│   ├── metrics.py              # ROC-AUC, F1, bootstrap CIs
│   ├── calibration.py          # Platt, isotonic, ECE, reliability diagrams
│   └── stats_tests.py          # McNemar pairwise, Cochran's Q
├── explainability/
│   └── shap_runner.py          # SHAP on hybrid LightGBM head
└── utils.py                    # setup_logger, file_sha, set_seed

scripts/
├── collect_data.py
├── prepare_data.py             # data/raw → data/processed (full pipeline)
├── train_baselines.py
├── train_transformer.py        # --model {wangchanberta,phayathaibert,xlm-roberta-large,typhoon-2.5,openthaigpt}
├── train_hybrid.py
├── evaluate.py
├── explain.py
└── cloud/                      # Kaggle Kernels orchestration (under construction)
    ├── build_kaggle_dataset.py
    ├── run_on_kaggle.py
    └── kaggle_kernel_template.py

configs/
├── data.yaml                   # collection + cleaning
├── features.yaml               # TF-IDF + structural feature toggles
├── train.yaml                  # baselines + transformers + hybrid HPs
└── eval.yaml                   # bootstrap n, calibration, stats test config
```

## Data Flow

```
raw → clean → labeled → split → +sentiment → +embeddings → processed
                                                              │
                          ┌───────────────────────────────────┤
                          ▼                                   ▼
                    baselines                       transformer FT
                          │                                   │
                          └──────────┬────────────────────────┘
                                     ▼
                              hybrid (MLP + GBM)
                                     │
                                     ▼
                       reports/predictions/{model}_{split}.parquet
                                     │
                                     ▼
                             scripts/evaluate.py
                                     │
                                     ▼
                       reports/{tables,figures}/
```

## Configs Index

| File | Key sections |
|---|---|
| `configs/data.yaml` | collection (api_key, regions, search queries), cleaning thresholds |
| `configs/features.yaml` | structural toggles, tfidf params, sentiment cache path |
| `configs/train.yaml` | baselines, transformers (3 encoders + 2 decoders), hybrid, hpo |
| `configs/eval.yaml` | bootstrap n, calibration on/off, stats tests on/off |

## Dataset Stats (current)

| | count |
|---|---|
| rows (after cleaning) | 23,431 |
| unique channels | 82 |
| time window | 2025-01-01 → 2026-05-23 (~17 months) |
| view-count range | 0 – 175M |
| global positive rate | 10.16% |

## Split Stats (current)

| split | rows | channels | positive rate |
|---|---|---|---|
| train (after 3:1 undersample) | 6,964 | 60 | 25.0% |
| val | 3,143 | 13 | 10.4% |
| test (latest 15%) | 3,510 | 73 | 8.8% |

## External Resources

- GitHub: https://github.com/MNupakorn/thesis-youtube-virality-thai
- Kaggle username: `mknpk01`
- Kaggle dataset (to be created): `mknpk01/thesis-virality-data`
- MLflow tracking: `file:./mlruns` (committed structure)

## Environment

- Python 3.11 via `uv`
- Local: M1 MacBook Air (CPU + MPS, slow for FT)
- Cloud: Kaggle T4 16GB (primary), Colab T4 (manual fallback)
- Key deps: `pyproject.toml` is the source of truth
