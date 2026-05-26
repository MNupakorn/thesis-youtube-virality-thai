# Thesis Auto-Memory

> Loaded at session start. Keep entries one line each, under ~200 chars.
> Detail lives in `.claude/memory/*.md`. This file is the index.

## Current State
- **Phase:** Cloud pipeline live + WangchanBERTa local fallback done. PhayaThaiBERT + XLM-R-large pending cloud GPU (M1 OOM).
- **Stack:** Python 3.11 · uv · pandas · scikit-learn · LightGBM · XGBoost · PyTorch · transformers v5 · MLflow · Kaggle CLI · LIME · SHAP
- **Models in scope:** WangchanBERTa (105M, ✅ local 3-epoch), PhayaThaiBERT (278M, pending cloud), XLM-RoBERTa-large (560M, pending cloud).
- **Dataset:** 23,431 rows, 82 channels, 10.16% positive rate. Split: train 6,964 (undersampled 3:1) / val 3,143 / test 3,510 (latest 15% by time).
- **Last commit:** `60fad4a feat(eval): ROC/PR + reliability figures + top-50 FP/FN error analysis`
- **Last updated:** 2026-05-26 (15:06 ICT)

## Latest Session
- **Goal:** Finish thesis pipeline end-to-end (everything except LaTeX).
- **Status:** Pipeline complete, WangchanBERTa fine-tuned locally as fallback. PhayaThaiBERT + XLM-R-large blocked on cloud GPU (Kaggle free pool congested >6 h, M1 too small).

## Results So Far
- Baselines (real data, channel-grouped + time-aware test):
  - **LightGBM + structured+TF-IDF: ROC-AUC 0.673 [0.643, 0.700]** ← best so far
- WangchanBERTa local 3-epoch FT: ROC-AUC 0.575 [0.542, 0.604] — **sub-optimal** (reduced config, max_len 48). Treat as pipeline smoke test.
- Hybrid: val AUC 0.70, test AUC 0.56 — time-distribution shift confirmed.
- 12-classifier Cochran's Q on test: stat 7106, p≈0. Pairwise McNemar 66 pairs all significant.
- SHAP top features: `duration_sec`, `log_subscriber_count`, embedding dims.
- LIME (30 titles), attention rollout (20 titles) generated for WangchanBERTa.

## Key Files
- Canonical input: `data/processed/dataset_with_labels.parquet`
- Cached features: `data/interim/sentiment_cache.parquet`, `data/interim/title_embeddings.npy`
- Training entry: `scripts/train_transformer.py --model {wangchanberta,phayathaibert,xlm-roberta-large}`
- Cloud driver: `scripts/cloud/run_on_kaggle.py --model X` (Kaggle path; HF Jobs path TBD)
- Eval entry: `scripts/evaluate.py --config configs/eval.yaml`
- Explain entry: `scripts/explain.py --only {shap,lime,attention}`
- M1-reduced config: `configs/train_m1.yaml` (NOT for production)

## Known Issues
- Kaggle free GPU pool currently saturated — kernel queued >6 h without starting.
- M1 MPS: PhayaThaiBERT OOMs at 9 GB; fp16/bf16 produces NaN (auto-disabled when no CUDA).
- `train_hybrid.py` silently skips GBM block after MLP — workaround: train GBM standalone (see decisions.md).

## Next Priority (next session, per user request: "ไม่ใช้คอม")
1. Write `scripts/cloud/run_on_hf_jobs.py` using `huggingface-skills:hugging-face-jobs`. Cost ≈ $1 total for 3 encoders on a10g.
2. Run PhayaThaiBERT + XLM-R-large on HF Jobs (full configs/train.yaml — 4-5 epochs, max_len 96).
3. Re-run WangchanBERTa on cloud at full config to replace the M1-reduced run.
4. Re-run `make eval` — `mcnemar_pairwise_encoders.csv` + `cochrans_q_encoders.csv` will populate.
5. Re-run `make explain` if a new top model emerges.
6. LaTeX thesis writing.

## Detail Indexes
- [Architecture snapshot](.claude/memory/architecture.md)
- [Decisions log](.claude/memory/decisions.md)
- [Active task](.claude/memory/active.md)
- [Changelog](.claude/memory/changelog.md)
- [Experiments](.claude/memory/experiments.md)
