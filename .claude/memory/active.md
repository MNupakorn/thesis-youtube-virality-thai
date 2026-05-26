# Active Task

**As of:** 2026-05-26 (end of session)

## Status: WangchanBERTa local fallback DONE; PhayaThaiBERT + XLM-R-large blocked

Kaggle free GPU pool was congested >6 h today (kernel stayed `QUEUED` past the 6 h
driver cap and beyond — never started). Fell back to local M1 MPS for
WangchanBERTa (105M). PhayaThaiBERT (278M) hit MPS OOM at 9 GB pool. XLM-R-large
(560M) not attempted — would OOM. Per user direction, the next session should
move encoder training off this machine entirely (HF Jobs preferred).

## What this session produced (committed unless noted)

### Code
- `scripts/cloud/` Kaggle pipeline (`build_kaggle_dataset.py`,
  `kaggle_kernel_template.py`, `run_on_kaggle.py`) — works correctly, blocked only
  by the free Kaggle GPU shortage.
- Makefile: `kaggle-dataset` + `train-cloud-{wangchan,phaya,xlmr,all}` targets;
  removed out-of-scope `train-typhoon` / `train-openthai`.
- `src/explainability/lime_runner.py` + `attention_runner.py` — LIME for text
  classifier and attention rollout (Abnar & Zuidema 2020).
- `scripts/explain.py` extended with `--only {shap,lime,attention}` switch.
- `scripts/evaluate.py` — fixed Path/str concat bug; added 3-encoder-specific
  McNemar + Cochran's Q subset; ROC/PR curves; reliability diagram; top-50 FP/FN
  error analysis.
- `src/models/transformer_finetune.py` — auto-disable fp16/bf16 when CUDA not
  available (MPS produced NaN logits).

### Data
- Private Kaggle dataset `mknpk01/thesis-virality-data` (4 files, 52 MB) live.

### Models / predictions (M1 local)
- **WangchanBERTa 3-epoch fine-tune** (reduced config, max_len=48, bs=16,
  grad_accum=2, gradient_checkpointing=true). Best checkpoint = checkpoint-218
  (val_auc=0.549). Test ROC-AUC 0.5748 [0.5416, 0.6044]. Predictions parquet
  saved.

### Reports (all on baselines + hybrid + 1 encoder)
- `reports/tables/all_models_metrics.csv` (12 models)
- `reports/tables/mcnemar_pairwise.csv` (66 pairs)
- `reports/tables/cochrans_q.csv` (12 classifiers, p≈0)
- `reports/tables/calibration_*.csv`
- `reports/tables/per_category_breakdown.csv`
- `reports/tables/errors_top_{fp,fn}_*.csv`
- `reports/figures/roc_pr_test.{svg,png}`
- `reports/figures/calibration_top_model.{svg,png}`
- `reports/figures/shap_summary.png` + `shap_global_importance.csv`
- `reports/figures/lime/{lime_html, lime_per_example.csv, lime_aggregate_tokens.csv}`
- `reports/figures/attention/{attention_html, attention_rollout_per_example.parquet}`

## What is still missing (for next session)

1. **PhayaThaiBERT fine-tune** — needs cloud GPU (won't fit M1).
2. **XLM-RoBERTa-large fine-tune** — needs cloud GPU.
3. **Re-run `evaluate.py`** once both encoders' predictions are in — the
   3-encoder-only McNemar + Cochran's Q will populate
   `mcnemar_pairwise_encoders.csv` + `cochrans_q_encoders.csv` (currently skipped
   because we have only 1 encoder).
4. **Calibration** redo on top encoder if it becomes the new top model.
5. **Re-run SHAP + LIME + attention** on the new top model.

## Resume path for next session

User explicitly requested "ไม่ใช้คอม" → move encoder training to cloud.

Recommended order (fastest to set up):
1. Write `scripts/cloud/run_on_hf_jobs.py` parallel to `run_on_kaggle.py`, using
   `huggingface-skills:hugging-face-jobs` skill. Use `hf jobs uv run --flavor
   a10g-small` (or `a10g-large` for XLM-R-large) with the existing
   `scripts/train_transformer.py` entry point. Should cost ~$1 total for all 3
   encoders.
2. As a backup, try Kaggle again — the queue may have cleared. The pipeline is
   already written and the dataset is already pushed.
3. Alternative: Kaggle Notebooks Pro ($10/mo, shorter queue).

Avoid:
- Local M1 for PhayaThaiBERT / XLM-R-large (will OOM).
- Colab — requires interactive OAuth, not headless-friendly.

## Known issues to keep in mind
- `train_hybrid.py` silently skips its GBM block when MLP runs first
  (see `decisions.md`). Workaround: train GBM separately. Artifacts are in place.
- Stale Kaggle kernel `mknpk01/thesis-wangchanberta-20260525` is still queued —
  harmless; it will eventually run or expire.
- `configs/train_m1.yaml` is a local-overrides file with reduced hyperparams.
  It is **not** the production config — production = `configs/train.yaml`.
