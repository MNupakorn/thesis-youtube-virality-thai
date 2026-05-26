"""Upload processed + interim data files to a private HF dataset repo.

Mirror of ``build_kaggle_dataset.py`` for the HF Jobs cloud path. The HF Jobs
driver mounts this dataset read-only at ``/data`` inside the job container.

Default repo id: ``mknpk01/thesis-virality-data`` (private).

Usage::

    python -m scripts.cloud.build_hf_dataset
    python -m scripts.cloud.build_hf_dataset --repo-id otheruser/data
    python -m scripts.cloud.build_hf_dataset --commit-message "regenerated"

Requires `hf auth login` first (or HF_TOKEN env var).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.utils import project_root, setup_logger

log = setup_logger("scripts.cloud.build_hf_dataset")

DEFAULT_REPO = "mknpk01/thesis-virality-data"

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--commit-message", default="auto regenerate from build_hf_dataset.py")
    ap.add_argument("--public", action="store_true", help="upload as public (default: private)")
    args = ap.parse_args()

    from huggingface_hub import HfApi, create_repo

    root = project_root()
    api = HfApi()

    log.info(f"creating/ensuring HF dataset repo: {args.repo_id} (private={not args.public})")
    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )

    hashes: dict[str, str] = {}
    for rel in FILES:
        src = root / rel
        if not src.exists():
            raise FileNotFoundError(f"missing required file: {src}")
        h = file_sha(src)
        hashes[src.name] = h
        log.info(f"  uploading {src.name} ({src.stat().st_size / 1e6:.1f} MB, sha={h})")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=src.name,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"{args.commit_message} :: {src.name}",
        )

    sidecar = root / "data" / "interim" / "hf_dataset_hashes.json"
    sidecar.write_text(json.dumps({"repo_id": args.repo_id, "files": hashes}, indent=2))
    log.info(f"wrote {sidecar}")
    log.info(f"done. https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
