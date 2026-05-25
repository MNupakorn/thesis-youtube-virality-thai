# Thesis Auto-Memory

> Loaded at session start. Keep entries one line each, under ~200 chars.
> Detail lives in `.claude/memory/*.md`. This file is the index.

## Current State
- **Phase:** Pipeline complete (data → baselines → sentiment → hybrid). Encoder FT pending cloud GPU.
- **Stack:** Python 3.11 · uv · pandas · scikit-learn · LightGBM · XGBoost · PyTorch · transformers v5 · MLflow · Kaggle CLI
- **Models in scope:** WangchanBERTa (105M), PhayaThaiBERT (278M), XLM-RoBERTa-large (560M) — encoders only.
- **Dataset:** 23,431 rows, 82 channels, 10.16% positive rate. Split: train 6,964 (undersampled 3:1) / val 3,143 / test 3,510 (latest 15% by time).
- **Last commit:** `3cf31a9 docs(notebook): polish Colab notebook with full upload + train + download flow`
- **Last updated:** 2026-05-26

## Latest Session
- **Goal:** Set up `.claude/` architecture (CLAUDE.md, rules, skills, agents, memory) matching the jinkintech-erp / SKILL.md pattern.
- **Status:** In progress this session.

## Results So Far
- Baselines (real data, channel-grouped + time-aware test):
  - **LightGBM + structured+TF-IDF: ROC-AUC 0.673 [0.643, 0.700]** ← best baseline
  - Sentiment alone: AUC 0.50 (no signal solo, +1.3pt F1 when combined)
- Hybrid model: val AUC 0.70, test AUC 0.56 — time-distribution shift confirmed.
- Encoder FTs: not yet run. Cloud pipeline (Kaggle) under construction.

## Key Files
- Canonical input: `data/processed/dataset_with_labels.parquet`
- Cached features: `data/interim/sentiment_cache.parquet`, `data/interim/title_embeddings.npy`
- Training entry: `scripts/train_transformer.py --model {wangchanberta,phayathaibert,xlm-roberta-large}`
- Eval entry: `scripts/evaluate.py --config configs/eval.yaml`

## Known Issues
- WangchanBERTa local FT on M1 too slow (~4h estimated). Use cloud.
- Hybrid val/test gap is a research finding — report honestly, do not hide.

## Next Priority
1. Finish `scripts/cloud/` (Kaggle Kernels driver) — build dataset uploader + kernel runner + Makefile targets.
2. Trigger first cloud run (WangchanBERTa) on Kaggle T4.
3. Run all 3 encoders, save predictions to `reports/predictions/`.
4. Run `scripts/evaluate.py` with McNemar + Cochran's Q on the 3-encoder set.
5. SHAP/LIME/attention explanations.
6. LaTeX thesis writing.

## Detail Indexes
- [Architecture snapshot](.claude/memory/architecture.md) — directory tree + module map
- [Decisions log](.claude/memory/decisions.md) — why encoders, why this split, etc.
- [Active task](.claude/memory/active.md) — current focused work
- [Changelog](.claude/memory/changelog.md) — per-session diffs
- [Experiments](.claude/memory/experiments.md) — runs + metrics + artifacts
