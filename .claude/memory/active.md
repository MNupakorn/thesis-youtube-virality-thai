# Active Task

**As of:** 2026-05-26

## Current Focus
Cloud GPU pipeline live. WangchanBERTa kernel queued on Kaggle (free T4/P100 pool currently congested). While waiting, completed local baselines + hybrid evaluation and SHAP, and extended `scripts/explain.py` with LIME + attention-rollout runners (ready to run as soon as transformer checkpoints land).

## Done This Session
1. `scripts/cloud/` pipeline written + committed (build_kaggle_dataset, kaggle_kernel_template, run_on_kaggle).
2. Makefile cleaned: removed out-of-scope `train-typhoon` / `train-openthai`; added `train-cloud-{wangchan,phaya,xlmr,all}` + `kaggle-dataset`.
3. Pushed private Kaggle dataset `mknpk01/thesis-virality-data` (4 files, ~52 MB).
4. Pushed WangchanBERTa kernel `mknpk01/thesis-wangchanberta-20260525` at SHA `ba9a1b5`. Driver polling.
5. Regenerated hybrid GBM artifact (`gbm.joblib`, `X_test.npy`, `y_test.npy`) — train_hybrid.py silently skips its GBM block when MLP runs first (see `decisions.md`).
6. Fixed `scripts/evaluate.py` Path-concat bug. Produced `all_models_metrics.csv` (11 models), `mcnemar_pairwise.csv` (55 pairs), `cochrans_q.csv`, calibration tables, `per_category_breakdown.csv`.
7. Wrote `src/explainability/lime_runner.py` + `attention_runner.py`. Extended `scripts/explain.py` with `--only {shap,lime,attention}` switch.
8. SHAP done on hybrid LightGBM head → `reports/figures/shap_summary.png`, `shap_global_importance.csv`. Top features: `duration_sec`, `log_subscriber_count`, embedding dims.

## Next Concrete Step
1. Wait for WangchanBERTa kernel to leave QUEUED. Once `complete`: driver auto-downloads predictions to `reports/predictions/wangchanberta.parquet` + checkpoint to `reports/artifacts/models/wangchanberta/`.
2. Run `make train-cloud-phaya` then `make train-cloud-xlmr` sequentially (Kaggle limits concurrency).
3. Re-run `make eval` (will now include the 3 encoders; McNemar + Cochran's Q will compute over the full set including the 3-encoder subset that the thesis claims).
4. Run `make explain` (SHAP already done; LIME + attention will run once a checkpoint is present).
5. Update `.claude/memory/experiments.md` with per-encoder metrics + SHAs + Kaggle slugs.

## Blocking / Risks
- Kaggle free GPU pool congestion — WangchanBERTa kernel has been QUEUED for ~45 min. Driver caps at 6h. Fallback path: Colab notebook `notebooks/04_transformer_finetune.ipynb` (manual upload).

## Resume Hint
If the session is compacted: re-read `MEMORY.md`, then check `uv run kaggle kernels status mknpk01/thesis-wangchanberta-20260525` and `tail tmp/wangchan_run.log` to see the driver's current state.
