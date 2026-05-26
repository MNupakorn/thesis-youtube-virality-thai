# Thesis Auto-Memory

> Loaded at session start. Keep entries one line each, under ~200 chars.
> Detail lives in `.claude/memory/*.md`. This file is the index.

## Current State
- **Phase:** ✅ All 3 encoders trained on HF Jobs (full configs). Eval + Explainability done. Only LaTeX writing remains.
- **Stack:** Python 3.11 · uv · pandas · scikit-learn · LightGBM · XGBoost · PyTorch · transformers v5 · MLflow · Kaggle CLI · HF Jobs · LIME · SHAP
- **Models:** WangchanBERTa (105M, t4-small, 0.6402), **PhayaThaiBERT (278M, t4-small, 0.6451 ← best encoder)**, XLM-R-large (560M, a10g-small, 0.5450). Total HF cost ≈ $0.25.
- **Dataset:** 23,431 rows, 82 channels, 10.16% positive rate. Split: train 6,964 (undersampled 3:1) / val 3,143 / test 3,510 (latest 15% by time).
- **Last commit:** `94e01fb fix(cloud): bash -c not -lc in HF jobs run`
- **Last updated:** 2026-05-26 (17:22 ICT)

## Latest Session
- **Goal:** Finish thesis end-to-end (everything except LaTeX).
- **Status:** ✅ DONE. All 3 encoders + baselines + hybrid evaluated; 3-encoder McNemar + Cochran's Q populated; SHAP/LIME/attention generated; HF Jobs path established as the canonical cloud route.

## Headline Result
- **LightGBM (structured+TFIDF) ROC-AUC 0.6728** beats every encoder.
- **PhayaThaiBERT 0.6451** is the best encoder, > WangchanBERTa 0.6402 > XLM-R-large 0.5450.
- 3-encoder Cochran's Q: stat=612.69, p ≈ 9.0e-134.
- Pairwise McNemar all significant (Phaya > Wang > XLM-R).
- Honest finding: Thai-YouTube virality is dominated by structural/metadata signals, not title text. Discussion-section material.

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
