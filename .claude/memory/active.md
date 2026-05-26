# Active Task

**As of:** 2026-05-26 (end of session, post-HF-Jobs)

## Status: ✅ All 3 encoders trained on HF Jobs. Eval + Explainability done. Only LaTeX writing remains.

When Kaggle free GPU pool stayed blocked >6 h, switched to **HF Jobs** (user's
request: "ไม่ใช้คอม"). All three encoders fine-tuned at full config on
HF Jobs:

| Model | Test ROC-AUC | 95 % CI | Hardware | Cost |
|---|---|---|---|---|
| baselines/lightgbm_structured_plus_tfidf | **0.6728** | [0.6430, 0.7002] | local | – |
| baselines/xgboost_structured_plus_tfidf | 0.6534 | [0.6230, 0.6828] | local | – |
| baselines/lightgbm_structured | 0.6527 | [0.6233, 0.6830] | local | – |
| **transformers/phayathaibert** ← best encoder | 0.6451 | [0.6143, 0.6757] | t4-small | ~$0.05 |
| transformers/wangchanberta | 0.6402 | [0.6100, 0.6699] | t4-small | ~$0.04 |
| baselines/xgboost_structured | 0.6312 | [0.6002, 0.6619] | local | – |
| transformers/xlm-roberta-large | 0.5450 | [0.5110, 0.5775] | a10g-small | ~$0.15 |

Total HF Jobs spend ≈ **$0.25** (all three on full configs).

**3-encoder Cochran's Q on test set:** stat=612.69, df=2, **p ≈ 9.0e-134**
→ all three encoders differ significantly. Pairwise McNemar confirms: PhayaThaiBERT
beats WangchanBERTa (450 vs 303, p≈1e-7) and both beat XLM-R-large.

**The LightGBM baseline still beats every encoder** by ~3 pt AUC. Documented as a
genuine research finding — Thai-YouTube virality is dominated by structural/
metadata signals (channel size, duration, time-of-publication), not by text
content alone. This will be the discussion-section headline.

## What was completed in the full session

### Infra
- `scripts/cloud/{build_kaggle_dataset,kaggle_kernel_template,run_on_kaggle}.py`
  + `train-cloud-*` Makefile targets — Kaggle Kernels path (blocked by free pool
  queue today, but driver is correct).
- `scripts/cloud/{build_hf_dataset,run_on_hf_jobs}.py` + `train-hf-*` Makefile
  targets — HF Jobs path. Supports `--push-only` / `--poll-only --job-id <id>`
  so multiple encoders can queue in parallel.
- HF dataset `MGodK/thesis-virality-data` (private) holds the 4 input files.
- HF output dataset repos `MGodK/thesis-output-{model}-20260526` hold each
  job's predictions + checkpoint + git_sha.txt + pip_freeze.txt + nvidia_smi.txt.

### Data / models
- Dataset hashes in `data/interim/{kaggle,hf}_dataset_hashes.json`.
- All 3 encoder checkpoints downloaded into `reports/cloud_runs/hf-{model}-*/`,
  promoted predictions into `reports/artifacts/predictions/transformers/`.
- PhayaThaiBERT best checkpoint symlinked at `reports/artifacts/models/phayathaibert/`
  (checkpoint-872 → epoch 4, val_roc_auc 0.5938).

### Evaluation
- `reports/tables/all_models_metrics.csv` (14 classifiers)
- `reports/tables/mcnemar_pairwise.csv` — 91 pairs
- `reports/tables/cochrans_q.csv` — 14 classifiers, p≈0
- `reports/tables/mcnemar_pairwise_encoders.csv` — 3-encoder-only subset
- `reports/tables/cochrans_q_encoders.csv` — 3-encoder-only subset
- `reports/tables/per_category_breakdown.csv`
- `reports/tables/calibration_*.csv` + `reports/tables/errors_top_{fp,fn}_*.csv`
- `reports/figures/roc_pr_test.{svg,png}` + `calibration_top_model.{svg,png}`

### Explainability
- `reports/figures/shap_summary.png` + `shap_global_importance.csv` (LightGBM head)
- `reports/figures/lime/lime_html/` (30 examples on PhayaThaiBERT) +
  `lime_aggregate_tokens.csv` — top tokens: `roblox`, `Roblox`, `RoV`, `ค`, ...
- `reports/figures/attention/attention_html/` (20 examples on PhayaThaiBERT) +
  `attention_rollout_per_example.parquet`

### Bugs fixed
- `train_hybrid.py` silently skips GBM block after MLP on MPS — workaround in
  `decisions.md` (train GBM standalone).
- `src/models/transformer_finetune.py`: auto-disable fp16/bf16 on non-CUDA
  (MPS produced NaN logits).
- `scripts/evaluate.py`: Path / str concat (was crashing before any output).
- `scripts/cloud/run_on_hf_jobs.py`:
  - `bash -lc` → `bash -c` (the `-l` was being parsed as `--label` short form
    by `hf jobs run`, eating the next token and breaking the script).
  - job_id parser was capturing "View at: …" instead of the 24-hex id.
- `.gitignore` patterns were broken by trailing comments; fixed.

## Next session: only LaTeX thesis writing remains

Per user direction "ยังไม่ต้องเขียน LaTex", we stop here. The pipeline is
ready for the writeup. All numbers + figures + tables are reproducible by:

```bash
make hf-dataset      # one-time, only if data/processed changes
make train-hf-wangchan train-hf-phaya train-hf-xlmr   # or all detached
make eval
make explain
```

## Known leftovers
- Stale Kaggle kernels (`thesis-wangchanberta-20260525` etc.) were deleted.
- `notebooka31551eada` Kaggle starter notebook was deleted (was occupying a
  batch GPU slot for no reason).
- `kaggle.json` removed from project root and moved to `~/.kaggle/kaggle.json`
  (chmod 600). API token rotated twice (sha8 `78002f0a` → `41171c…` → `938ed8fe`).
