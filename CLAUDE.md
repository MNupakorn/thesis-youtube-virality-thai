# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Identity

**Thai YouTube Virality Prediction (Thesis)** — A comparative study of Thai transformer encoder LMs for binary virality classification on YouTube video titles + metadata + sentiment.

- **Department:** Applied Statistics & Data Analytics, School of Science, KMITL · Academic Year 2025
- **Authors:** Traipoohrin Suebuswansarn (65050330), Teeranon Chailangka (65050417), Phanudech Phuthoson (65050685)
- **Advisor:** Asst. Prof. Dr. Pornpimol Chaiwuttisak
- **Task type:** Binary classification (viral / not viral). This is an **encoder** task — decoder LLMs are out of scope.
- **Primary deliverables:** (1) reproducible pipeline (2) results tables + figures (3) statistical comparison across 3 encoders (4) LaTeX thesis.
- **Repo:** https://github.com/MNupakorn/thesis-youtube-virality-thai

## Models Under Study (encoders only)

| Alias | HF Name | Params | Why |
|---|---|---|---|
| `wangchanberta` | `airesearch/wangchanberta-base-att-spm-uncased` | 105 M | Thai SP, established Thai-NLP baseline |
| `phayathaibert` | `clicknext/phayathaibert` | 278 M | Newer + larger Thai corpus, modern Thai SOTA |
| `xlm-roberta-large` | `FacebookAI/xlm-roberta-large` | 560 M | Multilingual, handles TH+EN code-switching common on YouTube |

A QLoRA path exists in `src/models/transformer_finetune.py` for any future decoder LLM (`typhoon-2.5`, `openthaigpt`) but is **not used** for the thesis comparison.

## Commands

```bash
# Setup
uv venv --python 3.11 && uv pip install -e ".[dev]"

# Pipeline
make prepare           # clean + label + split + structural features + sentiment + embeddings
make train-baselines   # LR / LightGBM / XGBoost across feature sets
make train-wangchan    # local FT (slow on M1 — prefer cloud)
make train-cloud-all   # Kaggle GPU: all 3 encoders (see scripts/cloud/)
make train-hybrid      # MLP + LightGBM heads on fused features
make eval              # ROC-AUC, F1, McNemar, Cochran's Q, calibration
make explain           # SHAP + LIME + attention rollout

# Quality gates
uv run pytest          # unit tests
uv run ruff check .    # lint
uv run black --check . # format check
```

## Directory Structure

```
.
├── configs/                # YAML: data.yaml, features.yaml, train.yaml, eval.yaml
├── data/
│   ├── raw/                # youtube_thai_videos.parquet (collected via API)
│   ├── interim/            # sentiment_cache.parquet, title_embeddings.npy
│   └── processed/          # dataset_with_labels.parquet (canonical input)
├── notebooks/              # 04_transformer_finetune.ipynb (Colab fallback)
├── reports/                # tables/, figures/, predictions/ (auto-generated)
├── scripts/                # CLI entry points (collect, prepare, train_*, evaluate, explain)
│   └── cloud/              # Kaggle Kernels orchestration (build_dataset, run_on_kaggle)
├── src/
│   ├── data_collection/    # YouTube API v3 collector
│   ├── data_processing/    # clean.py, labels.py (ViralityIndex), splits.py
│   ├── features/           # structural.py, sentiment.py, tfidf.py, transformer_embed.py
│   ├── models/             # baselines.py, transformer_finetune.py, hybrid.py
│   ├── evaluation/         # metrics.py, calibration.py, stats_tests.py
│   └── explainability/     # shap_runner.py
├── tests/                  # pytest unit tests
├── mlruns/                 # MLflow tracking (file backend)
└── .claude/                # Claude Code architecture (rules, skills, agents, memory)
```

## Core Rules

1. **This is encoder + classification.** Do not introduce decoder LLMs into the comparison without an explicit decision in `.claude/memory/decisions.md`.
2. **Data leakage is the #1 risk.** Splits are channel-grouped (train/val no shared channel) AND time-aware (test = latest 15% by `published_at`). Never shuffle across this boundary.
3. **Imbalance handling is train-only.** Val/test stay at the natural ~9% positive rate. Undersample/focal/class-weights applied only to the training fold.
4. **Reproducibility is mandatory.** Every script seeds `PYTHONHASHSEED`, NumPy, PyTorch, transformers. Every run is logged to MLflow with config + git SHA + dataset hash.
5. **Configs over flags.** All hyperparameters live in `configs/*.yaml`. Scripts read configs; do not hardcode model names, paths, or HP values in Python.
6. **Predictions are first-class artifacts.** Every model writes `reports/predictions/{model}_{split}.parquet` with `video_id, label, prob` so downstream stats tests can run independently.
7. **Statistical comparison is required.** Pairwise McNemar + Cochran's Q across the 3 encoders on the **same test set**. Bootstrap 95% CIs (n=1000) on every headline metric.
8. **Calibration is reported.** Platt + isotonic fitted on val, evaluated on test, ECE + reliability diagram.
9. **One change at a time.** Never tune model + features + loss in the same run. Each MLflow experiment isolates one variable.
10. **Honest reporting.** If val_auc ≫ test_auc, document the distribution shift in `.claude/memory/decisions.md`. Do not p-hack toward a desired number.

## Coding Standards

- Python 3.11, `uv` for dep management, `pyproject.toml` is the source of truth
- Format: `black` (line 100), `ruff` (E, F, I, B rules)
- Types: type hints on every public function; `from __future__ import annotations`
- Naming: snake_case files/functions, PascalCase classes, UPPER_SNAKE constants
- Logging: `from src.utils import setup_logger` — never `print()`
- DataFrames: pandas; cast object columns to `str` before parquet save
- Tensors: torch; use `device_map="auto"` only when needed, default to explicit `.to(device)`

## Cloud GPU (Kaggle path)

- Primary: **Kaggle Kernels** (free, 30 GPU-h/week, T4 or P100, fully scriptable via `kaggle` CLI)
- Credentials at `~/.kaggle/kaggle.json` (chmod 600). Username: `mknpk01`
- Data shipped as private Kaggle dataset; kernel clones GitHub repo + attaches dataset
- Fallback: Colab notebook `notebooks/04_transformer_finetune.ipynb` (manual upload)
- See `.claude/skills/cloud-gpu-orchestration/SKILL.md`

## Do Not

1. Do not commit `data/raw/`, `data/interim/`, `data/processed/`, `mlruns/`, `reports/predictions/` (gitignored).
2. Do not modify applied migrations or rewrite published model predictions — append a new run.
3. Do not silently change the label definition (`ViralityIndex` top-decile per channel). Any change requires a decision entry.
4. Do not skip the val split. Three-way (train/val/test) is non-negotiable.
5. Do not run hyperparameter search on the test set. HPO is val-only.
6. Do not introduce data augmentation that crosses the time boundary (e.g., synthetic future titles).
7. Do not use `*` imports. Always be explicit.
8. Do not commit secrets (`.env`, `kaggle.json`, HF tokens).

## Environment

`.env.local` (gitignored):
```
YOUTUBE_API_KEY=...
HUGGINGFACE_TOKEN=...      # optional, only if hitting rate limits
KAGGLE_USERNAME=mknpk01    # already in ~/.kaggle/kaggle.json
```

## References

- Research engineering rules: `.claude/rules/research-engineering.md`
- Reproducibility rules: `.claude/rules/reproducibility.md`
- Session continuity: `.claude/rules/session-continuity.md`
- Operating system: `.claude/skills/thesis-operating-system/SKILL.md`
- Memory index: `MEMORY.md` (auto-loaded each session)
