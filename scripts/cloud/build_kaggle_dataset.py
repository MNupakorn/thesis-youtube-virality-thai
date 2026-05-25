"""Package processed + interim data into a private Kaggle dataset.

Slug: ``mknpk01/thesis-virality-data``

First run creates the dataset; subsequent runs version it. The kernel template
attaches this dataset under ``/kaggle/input/thesis-virality-data/``.

Usage::

    python scripts/cloud/build_kaggle_dataset.py
    python scripts/cloud/build_kaggle_dataset.py --message "regenerated after sentiment fix"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from src.utils import project_root, setup_logger

log = setup_logger("scripts.cloud.build_kaggle_dataset")

DATASET_SLUG = "mknpk01/thesis-virality-data"
DATASET_TITLE = "Thesis Virality Data (Thai YouTube)"

FILES = [
    Path("data/processed/dataset_with_labels.parquet"),
    Path("data/interim/sentiment_cache.parquet"),
    Path("data/interim/title_embeddings.npy"),
    Path("data/interim/title_embeddings.ids.parquet"),
]


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def build_staging(staging: Path, root: Path) -> dict[str, str]:
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    hashes: dict[str, str] = {}
    for rel in FILES:
        src = root / rel
        if not src.exists():
            raise FileNotFoundError(f"missing required file: {src}")
        dst = staging / src.name
        shutil.copy2(src, dst)
        hashes[src.name] = file_sha(dst)
        log.info(f"  staged {src.name} ({dst.stat().st_size / 1e6:.1f} MB, sha={hashes[src.name]})")
    return hashes


def write_metadata(staging: Path) -> None:
    meta = {
        "title": DATASET_TITLE,
        "id": DATASET_SLUG,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (staging / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))


def dataset_exists(slug: str) -> bool:
    """Check if the dataset already exists on Kaggle (returns True if any version found)."""
    res = subprocess.run(
        ["kaggle", "datasets", "list", "-m", "--user", slug.split("/")[0]],
        capture_output=True,
        text=True,
        check=False,
    )
    return slug in res.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--message", default="auto regenerate from build_kaggle_dataset.py")
    ap.add_argument(
        "--staging",
        default="tmp/thesis-virality-data",
        help="local staging dir (gitignored)",
    )
    args = ap.parse_args()

    root = project_root()
    staging = root / args.staging

    log.info(f"staging files into {staging}")
    hashes = build_staging(staging, root)
    write_metadata(staging)

    exists = dataset_exists(DATASET_SLUG)
    if exists:
        log.info(f"dataset {DATASET_SLUG} exists -> creating new version")
        cmd = [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(staging),
            "-m",
            args.message,
            "--dir-mode",
            "zip",
        ]
    else:
        log.info(f"dataset {DATASET_SLUG} not found -> creating fresh")
        cmd = [
            "kaggle",
            "datasets",
            "create",
            "-p",
            str(staging),
            "--dir-mode",
            "zip",
        ]

    log.info(f"running: {' '.join(cmd)}")
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise SystemExit(f"kaggle CLI failed with code {res.returncode}")

    sidecar = root / "data" / "interim" / "kaggle_dataset_hashes.json"
    sidecar.write_text(json.dumps({"slug": DATASET_SLUG, "files": hashes}, indent=2))
    log.info(f"wrote {sidecar}")
    log.info("done. Dataset published / versioned.")


if __name__ == "__main__":
    main()
