"""Kernel template that runs ON Kaggle.

This file is *not* executed locally. ``run_on_kaggle.py`` reads it as text,
substitutes ``{{MODEL_ALIAS}}`` and ``{{GIT_SHA}}``, and pushes the rendered
script to Kaggle as the kernel body.

Layout on Kaggle:
- ``/kaggle/input/thesis-virality-data/`` — attached private dataset
- ``/kaggle/working/`` — kernel output, auto-bundled and downloadable

Outputs written to ``/kaggle/working/``:
- ``predictions/{model}.parquet`` (canonical predictions)
- ``artifacts/models/{model}/`` (HF checkpoint)
- ``git_sha.txt``, ``pip_freeze.txt``, ``nvidia_smi.txt``
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_ALIAS = "{{MODEL_ALIAS}}"
GIT_SHA = "{{GIT_SHA}}"
REPO_URL = "https://github.com/MNupakorn/thesis-youtube-virality-thai.git"

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input/thesis-virality-data")
REPO = WORK / "thesis-youtube-virality-thai"


def sh(cmd: str, cwd: Path | None = None) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def main() -> None:
    os.environ["PYTHONHASHSEED"] = "42"

    if REPO.exists():
        shutil.rmtree(REPO)
    sh(f"git clone {REPO_URL} {REPO}")
    sh(f"git checkout {GIT_SHA}", cwd=REPO)

    sh("pip install -q -U uv")
    sh("uv pip install --system -q -e '.[dev]'", cwd=REPO)

    (REPO / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (REPO / "data" / "interim").mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        INPUT / "dataset_with_labels.parquet",
        REPO / "data" / "processed" / "dataset_with_labels.parquet",
    )
    for fn in ("sentiment_cache.parquet", "title_embeddings.npy", "title_embeddings.ids.parquet"):
        shutil.copy2(INPUT / fn, REPO / "data" / "interim" / fn)

    sh(
        f"python scripts/train_transformer.py --model {MODEL_ALIAS} "
        f"--config configs/train.yaml --data-config configs/data.yaml",
        cwd=REPO,
    )

    out_pred = WORK / "predictions"
    out_art = WORK / "artifacts"
    out_pred.mkdir(exist_ok=True)
    out_art.mkdir(exist_ok=True)

    # train_transformer.py writes to: reports/artifacts/predictions/transformers/{model}.parquet
    src_pred = REPO / "reports" / "artifacts" / "predictions" / "transformers"
    if src_pred.exists():
        for p in src_pred.glob("*.parquet"):
            shutil.copy2(p, out_pred / p.name)
    # model checkpoint: reports/artifacts/models/{model}/
    src_models = REPO / "reports" / "artifacts" / "models"
    if src_models.exists():
        shutil.copytree(src_models, out_art / "models", dirs_exist_ok=True)
    src_mlruns = REPO / "mlruns"
    if src_mlruns.exists():
        shutil.copytree(src_mlruns, WORK / "mlruns", dirs_exist_ok=True)

    (WORK / "git_sha.txt").write_text(GIT_SHA + "\n")
    sh(f"pip freeze > {WORK / 'pip_freeze.txt'}")
    try:
        sh(f"nvidia-smi > {WORK / 'nvidia_smi.txt'}")
    except subprocess.CalledProcessError:
        (WORK / "nvidia_smi.txt").write_text("nvidia-smi unavailable\n")

    print(f"DONE. model={MODEL_ALIAS} sha={GIT_SHA}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
