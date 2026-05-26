# Thesis Auto-Memory

> Loaded at session start. Keep entries one line each, under ~200 chars.
> Detail lives in `.claude/memory/*.md`. This file is the index.

## Current State
- **Phase:** ✅ Pipeline + A+ ensemble upgrades done. Only LaTeX writing remains.
- **Stack:** Python 3.11 · uv · pandas · scikit-learn · LightGBM · XGBoost · PyTorch · transformers v5 · MLflow · Kaggle CLI · HF Jobs · LIME · SHAP
- **Top model:** `stacking/stacking_lr_calibrated` (Platt-calibrated LR over 15 base models) — test ROC-AUC **0.6914 [0.6625, 0.7174]**, ECE 0.016.
- **Base encoders:** WangchanBERTa (0.6402), PhayaThaiBERT (0.6451, best encoder), XLM-R-large (0.5450). Total HF Jobs cost ≈ $0.35.
- **Dataset:** 23,431 rows, 82 channels, 10.16% positive rate. Split: train 6,964 (undersampled 3:1) / val 3,143 / test 3,510 (latest 15% by time).
- **Last commit:** (next push) — stacking ensemble + ablations
- **Last updated:** 2026-05-26 (18:30 ICT)

## Latest Session
- **Goal:** Finish thesis end-to-end (everything except LaTeX).
- **Status:** ✅ DONE. All 3 encoders + baselines + hybrid evaluated; 3-encoder McNemar + Cochran's Q populated; SHAP/LIME/attention generated; HF Jobs path established as the canonical cloud route.

## Headline Result
- **Top:** `stacking_lr_calibrated` ROC-AUC **0.6914 [0.6625, 0.7174]** — Platt-calibrated LR meta-learner over 15 base models. ECE 0.016.
- Best single model: `lightgbm_structured_plus_tfidf` 0.6728. Best encoder: PhayaThaiBERT 0.6451 (PhayaThaiBERT > WangchanBERTa > XLM-R-large; Cochran's Q p ≈ 9e-134).
- Sub-population: stacking AUC 0.7459 on large channels (>1M subs) vs 0.6273 on mid (100k-1M) — viral signal much stronger in big channels.
- Negative finding: multi-field PhayaThaiBERT (title+description+channel) AUC 0.5455 — adding non-title text *hurts*. Thai YouTube descriptions are mostly boilerplate.
- Drop-one ablation identifies `lightgbm_structured_plus_tfidf` (-0.0047) and `phayathaibert` (-0.0041) as most-important contributors to the stack.

## Key Files
- Canonical input: `data/processed/dataset_with_labels.parquet`
- Cached features: `data/interim/sentiment_cache.parquet`, `data/interim/title_embeddings.npy`
- Training: `scripts/train_transformer.py --model {wangchanberta,phayathaibert,xlm-roberta-large}`
- Cloud (Kaggle): `scripts/cloud/run_on_kaggle.py --model X` (queue blocked today)
- Cloud (HF Jobs): `scripts/cloud/run_on_hf_jobs.py --model X` ← used today
- Eval: `scripts/evaluate.py` (auto-discovers all predictions parquets)
- Explain: `scripts/explain.py --only {shap,lime,attention}`
- Predictions: `reports/artifacts/predictions/transformers/{wangchanberta,phayathaibert,xlm-roberta-large}.parquet`
- HF private datasets: `MGodK/thesis-virality-data` (input), `MGodK/thesis-output-{model}-20260526` (per-model outputs)

## Cloud Path: HF Jobs (validated today)
- Cost: t4-small $0.40/h, a10g-small $1.00/h. Whole-thesis run ≈ $0.25.
- Submit one: `make train-hf-wangchan` (or `train-hf-phaya`, `train-hf-xlmr`).
- Submit all detached: `make train-hf-all-detached` then poll each with `--poll-only --job-id <id>`.
- Job state saved at `tmp/hf_jobs/{model}.json`.

## Known Issues (rolled forward)
- `train_hybrid.py` silently skips GBM block after MLP on MPS — workaround in `decisions.md`. Not blocking.
- M1 MPS: PhayaThaiBERT OOMs at 9 GB pool; fp16/bf16 produces NaN. Auto-disabled when no CUDA.
- Kaggle free GPU pool was unusable for >6 h today — abandoned in favor of HF Jobs.

## Next Priority (next session)
1. **LaTeX thesis writing.** All numbers, tables, figures are in `reports/`. Use `thesis-writing` skill.
2. (Optional) Re-run with new HPO (e.g. larger PhayaThaiBERT epochs, learning-rate sweep) if time permits.
3. (Optional) Fix the `train_hybrid.py` MPS bug properly.

## Detail Indexes
- [Architecture snapshot](.claude/memory/architecture.md)
- [Decisions log](.claude/memory/decisions.md)
- [Active task](.claude/memory/active.md)
- [Changelog](.claude/memory/changelog.md)
- [Experiments](.claude/memory/experiments.md)
