# Research Engineering Rules

Discipline that applies to every change in this repo. These are not aspirational — they are gates.

## Code

- **Read before writing.** Inspect affected files. Match existing patterns instead of inventing new ones.
- **Configs over flags.** Hyperparameters live in `configs/*.yaml`. Scripts read configs; no hardcoded HP in Python.
- **One change per run.** Never combine "new features" + "new model" + "new loss" in one experiment. Isolate variables.
- **No silent defaults.** Every default value in a config is explicit and commented. If a parameter changes the result, it must be visible.
- **Type hints on public functions.** `from __future__ import annotations` at every module top.
- **No `print()`.** Use `setup_logger("module.name")` from `src.utils`.
- **No `*` imports.** Always be explicit.
- **DataFrames in / out, not globals.** Pure functions over hidden state.

## Data

- **Splits are sacred.** Channel-grouped on train/val (no channel appears in both). Time-aware test (latest 15% by `published_at`).
- **No leakage across the time boundary.** No augmentation, no oversampling, no sentiment-recomputation that uses test rows.
- **Train-only imbalance handling.** Val/test stay at natural ~9% positive rate.
- **Cache derived features with hashes.** `data/interim/*.parquet` and `.npy` files include the input data hash in the filename or in a sidecar JSON.
- **Object columns → string before parquet save.** pyarrow rejects mixed types.

## Models

- **Encoders only for the comparison.** The 3 models are fixed: WangchanBERTa, PhayaThaiBERT, XLM-RoBERTa-large.
- **Same Trainer wrapper for all three.** Only batch size, grad accumulation, LR, and gradient_checkpointing differ.
- **Save predictions, not just metrics.** `reports/predictions/{model}_{split}.parquet` with `video_id, label, prob`. Downstream stats tests must be reproducible without re-training.
- **Save tokenizer + model best checkpoint** under `reports/artifacts/{model}/`. Hugging Face format only.
- **`processing_class=tokenizer` for transformers v5.** `tokenizer=` was removed.

## Evaluation

- **Bootstrap 95% CIs (n=1000) on every headline metric.** Point estimates alone are not acceptable.
- **McNemar's test pairwise + Cochran's Q across all three encoders** on the same test set.
- **Calibration on val, evaluated on test.** Report Platt + isotonic, ECE, reliability diagrams.
- **Per-category breakdown.** Gaming / Entertainment / Music breakdowns where category is known.
- **Error analysis.** Save the top-50 false positives and false negatives by confidence.

## Reproducibility

- **Seed everything.** `PYTHONHASHSEED=42`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `transformers.set_seed`.
- **MLflow every run.** Log: config snapshot, git SHA, dataset hash, all metrics, artifact paths.
- **Pin versions.** `pyproject.toml` is the source of truth. `uv.lock` is committed.
- **Document anomalies.** If a run produces an unexpected result, note it in `.claude/memory/experiments.md` and `.claude/memory/decisions.md`.

## Git

- **Conventional commits.** `feat:`, `fix:`, `data:`, `experiment:`, `docs:`, `refactor:`, `test:`.
- **Branch by experiment when results may diverge.** `exp/<short-name>` for anything that could end up in the thesis.
- **Never force-push `main`.** Predictions and decisions are append-only.

## Quality Gates Before "Done"

1. `uv run pytest` passes
2. `uv run ruff check .` clean
3. `uv run black --check .` clean
4. New experiments logged to MLflow
5. `.claude/memory/changelog.md` updated
6. If decision changed (label, split, model set, loss) → `.claude/memory/decisions.md` updated
