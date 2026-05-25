# Reproducibility Rules

Every result in the thesis must be reproducible from `git checkout <sha> && make prepare && make <target>`.

## Seeds

Set at the very top of every script that touches randomness:

```python
import os, random
import numpy as np
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# torch (if used)
import torch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# transformers
from transformers import set_seed
set_seed(SEED)
```

`PYTHONHASHSEED=42` is also baked into `.claude/settings.json` env so subprocess Python inherits it.

## MLflow

- Backend: `file:./mlruns` (committed-friendly, no DB)
- Experiment names: `thesis-virality-thai` (default)
- Run name convention: `{model}_{feature_set}_{loss}_{date}`
- Log on every run:
  - `mlflow.log_params(cfg_flat)` — full config flattened
  - `mlflow.log_metric` — every metric per epoch + final
  - `mlflow.log_artifact` — predictions parquet, calibration plot, confusion matrix
  - `mlflow.set_tag("git_sha", subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip())`
  - `mlflow.set_tag("dataset_hash", sha256_of_processed_parquet)`

## Dataset Hash

`data/processed/dataset_with_labels.parquet` is the canonical input. Compute its sha256 on load and log it. If two runs disagree, the hash tells you whether the data changed.

```python
import hashlib
def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]
```

## Version Pins

- `pyproject.toml` pins every direct dep with `~=` or `==`
- `uv.lock` is committed
- Notebook env mirrors local: `uv pip install --system -e ".[dev]"` inside the kernel

## Cloud GPU Reproducibility

- Kaggle kernel runs include `git rev-parse HEAD` in the run output
- Kernel pins the same `pyproject.toml` via `uv pip install -e ".[dev]"`
- Outputs include `git_sha.txt`, `pip_freeze.txt`, `nvidia_smi.txt` alongside the predictions parquet

## Result Files

```
reports/
├── tables/                              # CSVs only, no Excel
│   ├── baselines_metrics.csv
│   ├── encoder_metrics.csv
│   ├── mcnemar_pairwise.csv
│   └── cochran_q.csv
├── figures/                             # SVG preferred, PNG fallback
│   ├── roc_test.svg
│   ├── pr_test.svg
│   └── calibration_{model}.svg
├── predictions/                         # parquet, columns: video_id, label, prob, split
│   ├── lgbm_test.parquet
│   ├── wangchanberta_test.parquet
│   ├── phayathaibert_test.parquet
│   └── xlm-roberta-large_test.parquet
└── artifacts/                           # model checkpoints (HF format)
    └── {model}/
```

## What "Done" Means

A model is "done" when:

1. Predictions parquet exists for train/val/test
2. MLflow run is logged
3. Bootstrap CIs computed
4. Row added to `reports/tables/encoder_metrics.csv`
5. Entry added to `.claude/memory/experiments.md`

If any of these are missing, the run is not done — even if metrics look great.
