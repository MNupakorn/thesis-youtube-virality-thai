---
name: cloud-gpu-orchestration
description: >
  Use when running training on cloud GPU: Kaggle Kernels (primary, free), Colab fallback,
  or HF Jobs. Covers Kaggle CLI auth, dataset upload, kernel push/poll/download, kernel
  template that clones the repo + installs deps + runs training + saves predictions.
  Triggers: "run on cloud", "kaggle", "train on GPU", "the M1 is too slow", "queue another run".
user-invocable: false
---

# Cloud GPU Orchestration

The thesis trains three encoders (105M, 278M, 560M params). The M1 MacBook Air does it slowly. Cloud GPU is the path.

## 1. Primary: Kaggle Kernels (free, automatable)

- **Cost:** $0. 30 GPU-hours/week. T4 16GB or P100 16GB.
- **Why over Colab:** Colab requires interactive login. Kaggle has a CLI with a JSON token, so the whole loop (push → poll → download) is one command.
- **Username:** `mknpk01`. Token at `~/.kaggle/kaggle.json` (chmod 600).

## 2. The Three Pieces

```
scripts/cloud/
├── build_kaggle_dataset.py    # package data/processed + data/interim → private Kaggle dataset
├── run_on_kaggle.py           # push kernel for one model, poll, download outputs to reports/
└── kaggle_kernel_template.py  # the script that runs ON Kaggle (clones repo, trains, saves)
```

Makefile targets:

```
make train-cloud-wangchan   # one model
make train-cloud-phaya
make train-cloud-xlmr
make train-cloud-all        # all three, sequential (Kaggle limits concurrent kernels)
```

## 3. What Runs ON Kaggle

The kernel template does:

```python
# 1. Clone repo at a pinned SHA
!git clone https://github.com/MNupakorn/thesis-youtube-virality-thai.git
%cd thesis-youtube-virality-thai
!git checkout <SHA>

# 2. Install deps via uv (fastest)
!pip install -q -U uv
!uv pip install --system -q -e ".[dev]"

# 3. Copy attached Kaggle dataset → data/
!mkdir -p data/processed data/interim
!cp /kaggle/input/thesis-virality-data/dataset_with_labels.parquet data/processed/
!cp /kaggle/input/thesis-virality-data/sentiment_cache.parquet data/interim/
!cp /kaggle/input/thesis-virality-data/title_embeddings.npy data/interim/
!cp /kaggle/input/thesis-virality-data/title_embeddings.ids.parquet data/interim/

# 4. Train (config is committed)
!python scripts/train_transformer.py --model {MODEL_ALIAS} --config configs/train.yaml

# 5. Save outputs to /kaggle/working (auto-zipped as kernel artifact)
!cp -r reports/predictions /kaggle/working/
!cp -r reports/artifacts /kaggle/working/
!git rev-parse HEAD > /kaggle/working/git_sha.txt
!pip freeze > /kaggle/working/pip_freeze.txt
!nvidia-smi > /kaggle/working/nvidia_smi.txt
```

## 4. The Driver (run_on_kaggle.py)

For one model alias, the local driver:

1. Renders the kernel template with the alias + current git SHA substituted.
2. Writes `kernel-metadata.json` (slug `mknpk01/thesis-{model}-{date}`, GPU enabled, dataset attached).
3. `kaggle kernels push -p <tmpdir>`.
4. Polls `kaggle kernels status <slug>` every 30s. Logs every state change.
5. On `complete`, `kaggle kernels output <slug> -p reports/cloud_runs/{slug}/`.
6. Copies predictions parquet into `reports/predictions/` (the canonical location).
7. Appends a row to `.claude/memory/experiments.md`.

If status is `error`, downloads the log and prints the last 100 lines.

## 5. Dataset Upload (build_kaggle_dataset.py)

```bash
python scripts/cloud/build_kaggle_dataset.py
```

This:
1. Builds `tmp/thesis-virality-data/` containing the 4 needed files (~52MB total).
2. Writes `dataset-metadata.json` with slug `mknpk01/thesis-virality-data`.
3. `kaggle datasets create -p tmp/thesis-virality-data` (first time) or `kaggle datasets version -p ... -m "regenerate"` (subsequent).
4. Records dataset version + sha256 in `.claude/memory/architecture.md`.

Re-run whenever `data/processed/dataset_with_labels.parquet` changes.

## 6. Why Not Colab as Primary

- Colab CLI requires Google OAuth flow — not headless-friendly from this CLI.
- Notebook upload via `files.upload()` is interactive.
- Session limit (~12h) is fine; the inability to script the loop is the blocker.

Colab notebook `notebooks/04_transformer_finetune.ipynb` is kept as a manual fallback if Kaggle is over quota.

## 7. Why Not HF Jobs / Modal / RunPod

- HF Jobs is ~$1-2/h — fine but Kaggle is free.
- Modal needs a credit card to start.
- RunPod is the best paid option if Kaggle quota runs out.

Switch to HF Jobs only when: (a) Kaggle weekly quota exhausted AND (b) deadline pressure justifies $5-20 cost.

## 8. Failure Modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Kernel stuck in `queued` >10 min | Kaggle scheduler busy | Wait. Don't push duplicates. |
| Kernel `error` immediately | Repo clone failed (private?) or syntax error in template | Check log via `kaggle kernels output`. |
| Out-of-memory on T4 | XLM-R-large needs grad checkpointing + batch 8 + grad-accum 4 | Already configured in `configs/train.yaml`. Verify it's not overridden. |
| Output download empty | Kernel did not finish (timeout) | Re-run with fewer epochs or smaller model. |
| Predictions parquet missing | Training crashed late. | Check `nvidia_smi.txt` for OOM, check log tail. |

## 9. After Every Cloud Run

1. Move `reports/cloud_runs/{slug}/predictions/*.parquet` → `reports/predictions/`.
2. Move `reports/cloud_runs/{slug}/artifacts/*` → `reports/artifacts/`.
3. Append row to `.claude/memory/experiments.md` with: timestamp, model, kernel slug, git SHA, val/test ROC-AUC, predictions path.
4. Run `python scripts/evaluate.py --models {finished_models}` to update tables.
