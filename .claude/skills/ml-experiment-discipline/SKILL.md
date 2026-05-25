---
name: ml-experiment-discipline
description: >
  Use when designing, running, or reviewing any training run, hyperparameter search,
  ablation, or evaluation. Covers seeding, MLflow logging, run-naming, artifact saving,
  predictions contract, bootstrap CIs, calibration, and what "done" means for a model.
  Trigger phrases: "let's train", "run an experiment", "tune hyperparameters", "ablate",
  "rerun with different seed", "report metrics", "is this run done?".
user-invocable: false
---

# ML Experiment Discipline

This is how we run experiments in this thesis. The rules exist so future-you (and the committee) can trust the numbers.

## 1. The Predictions Contract

Every model — baseline, transformer, hybrid — writes:

```
reports/predictions/{model}_{split}.parquet
```

Schema:

| column | type | notes |
|---|---|---|
| `video_id` | str | join key with the canonical dataset |
| `label` | int (0/1) | ground truth |
| `prob` | float | P(viral=1) ∈ [0,1] |
| `split` | str | "train" / "val" / "test" |

Evaluation reads these files. No re-training is ever required to recompute metrics or run a stats test.

## 2. The Six Things to Log per Run

1. **Config snapshot.** `mlflow.log_params(flatten(cfg))` — the YAML, flattened.
2. **Git SHA.** `mlflow.set_tag("git_sha", ...)`.
3. **Dataset hash.** sha256 of `data/processed/dataset_with_labels.parquet`, first 12 chars.
4. **Metrics per epoch + final.** `mlflow.log_metric("val_roc_auc", x, step=epoch)`.
5. **Artifact paths.** Predictions parquet, calibration plot, confusion matrix.
6. **Environment.** `pip freeze > env.txt` and log as artifact (only needed for cloud runs).

## 3. Seeding (Cardinal Rule)

Set seeds at the top of every script. See `.claude/rules/reproducibility.md` for the exact block. Without this, comparison across runs is meaningless.

For transformers, also call `transformers.set_seed(SEED)` AFTER importing torch.

## 4. Naming Convention

MLflow run name: `{model}_{feature_set}_{loss}_{date}`

Examples:
- `wangchanberta_title_focal_2026-05-26`
- `lgbm_struct+tfidf_balanced_2026-05-25`
- `hybrid_clsemb+sent+struct_focal_2026-05-26`

## 5. Bootstrap CIs

Headline metrics (ROC-AUC, F1-pos, recall-pos, precision-pos) need 95% CIs by bootstrap n=1000 on the test set.

```python
from src.evaluation.metrics import bootstrap_ci  # implements this
auc, lo, hi = bootstrap_ci(y_test, prob_test, metric="roc_auc", n=1000, seed=SEED)
```

Report as: `0.673 [0.643, 0.700]`.

## 6. Calibration

- Fit Platt (`sklearn.linear_model.LogisticRegression` on logits) and isotonic on **val** predictions.
- Evaluate on **test**. Report ECE (10 bins) before + after each.
- Save reliability diagram to `reports/figures/calibration_{model}.svg`.

## 7. What "Done" Looks Like

A model run is done iff **all** of:

- [ ] Predictions parquet written for train/val/test
- [ ] MLflow run logged with all six items above
- [ ] Bootstrap CIs computed and added to `reports/tables/{model}_metrics.csv`
- [ ] Calibration evaluated (or explicitly skipped with reason)
- [ ] Entry added to `.claude/memory/experiments.md`

Anything missing → not done. Do not move to the next experiment until this one is closed.

## 8. Hyperparameter Search

- **Search space is val-only.** Optuna pruning + TPE on `val_roc_auc`.
- Toggle via `configs/train.yaml → hpo.enabled: true`.
- Each trial logs to a child MLflow run nested under the parent search run.
- Best trial's predictions are saved with suffix `_best`. Other trials' predictions are NOT saved (storage).
- After the search, refit the best config with the held-out test set untouched.

## 9. Ablations

When ablating, change exactly one knob. Common ablations and what they answer:

| Ablation | Question |
|---|---|
| Drop sentiment features | Does sentiment add signal beyond structural? |
| Drop title TF-IDF | Does TF-IDF still help when we have transformer embeddings? |
| Replace focal with CE | Is the imbalance handling necessary? |
| Replace channel-grouped with random split | How much does leakage inflate scores? |
| Replace time-aware with shuffled test | How much of the drop is distribution shift? |

## 10. When a Run Disagrees with Memory

The MLflow run is the truth. The memory file is the index. If they disagree:
1. Re-read the MLflow run.
2. Correct `.claude/memory/experiments.md`.
3. Annotate the disagreement in a one-line note.
