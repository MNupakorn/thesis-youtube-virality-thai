"""HF Jobs driver: submit an encoder fine-tune to Hugging Face Jobs and poll.

Usage::

    python -m scripts.cloud.run_on_hf_jobs --model wangchanberta
    python -m scripts.cloud.run_on_hf_jobs --model phayathaibert --flavor t4-small
    python -m scripts.cloud.run_on_hf_jobs --model xlm-roberta-large --flavor a10g-small
    python -m scripts.cloud.run_on_hf_jobs --model wangchanberta --push-only
    python -m scripts.cloud.run_on_hf_jobs --job-id <id> --poll-only

Pre-flight (auto):
1. Working tree is clean.
2. HEAD SHA exists on ``origin/main`` (push it first if not).
3. ``hf auth whoami`` succeeds (token has Write + Manage Jobs scope).

What the job does on HF infrastructure:
1. Clones the repo at the pinned SHA.
2. Installs deps via ``uv pip install --system -e ".[dev]"``.
3. Downloads the input data from ``HF_INPUT_DATASET`` (default
   ``MGodK/thesis-virality-data``) into the canonical paths.
4. Runs ``scripts/train_transformer.py --model X --config configs/train.yaml``.
5. Uploads predictions + checkpoint to ``HF_OUTPUT_REPO`` (a private dataset
   repo named ``mknpk01/thesis-output-{model}-{date}``).
6. Dumps ``git_sha.txt`` + ``pip_freeze.txt`` + ``nvidia_smi.txt`` alongside.

After job completes, the local driver downloads the output repo into
``reports/cloud_runs/hf-{model}-{date}/`` and promotes the predictions parquet
to ``reports/artifacts/predictions/transformers/{model}.parquet``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from src.utils import ensure_dir, project_root, setup_logger

log = setup_logger("scripts.cloud.run_on_hf_jobs")

REPO_URL = "https://github.com/MNupakorn/thesis-youtube-virality-thai.git"
DEFAULT_INPUT_DATASET = "MGodK/thesis-virality-data"
DEFAULT_NAMESPACE = "MGodK"
DEFAULT_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
POLL_SECONDS = 30
POLL_TIMEOUT_MIN = 240  # 4 h hard cap


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, check=False)
    if check and res.returncode != 0:
        log.error(f"command failed: {' '.join(cmd)}")
        log.error(f"stdout: {res.stdout}")
        log.error(f"stderr: {res.stderr}")
        raise SystemExit(res.returncode)
    return res


def preflight(root: Path) -> str:
    status = run(["git", "status", "--porcelain"], cwd=root).stdout.strip()
    if status:
        log.error("working tree is not clean:")
        log.error(status)
        raise SystemExit("commit or stash before submitting (reproducibility rule).")
    sha = run(["git", "rev-parse", "HEAD"], cwd=root).stdout.strip()
    log.info(f"HEAD = {sha}")
    run(["git", "fetch", "origin", "main"], cwd=root)
    contains = run(["git", "branch", "-r", "--contains", sha], cwd=root, check=False).stdout
    if "origin/main" not in contains:
        raise SystemExit(f"SHA {sha} is not on origin/main; push first")
    whoami = run(["uv", "run", "hf", "auth", "whoami"], cwd=root, check=False)
    if whoami.returncode != 0:
        raise SystemExit("hf auth not logged in. Run: uv run hf auth login")
    log.info(f"hf account: {whoami.stdout.strip().splitlines()[0]}")
    return sha


def build_bash_command(
    model: str,
    sha: str,
    input_dataset: str,
    output_repo: str,
    train_config: str = "configs/train.yaml",
) -> str:
    """The bash command executed on HF Jobs. Single-string for `hf jobs run`."""
    return (
        "set -euo pipefail\n"
        "echo '== HF Jobs runner =='\n"
        "echo 'nvidia-smi:'; nvidia-smi || echo 'no GPU'\n"
        "apt-get update -qq && apt-get install -y -qq git curl > /dev/null\n"
        "pip install -q -U uv huggingface_hub\n"
        f"git clone -q {REPO_URL} /work\n"
        "cd /work\n"
        f"git checkout {sha}\n"
        'uv pip install --system -q -e ".[dev]"\n'
        "mkdir -p data/processed data/interim\n"
        "python - <<PY\n"
        "import os\n"
        "from huggingface_hub import snapshot_download\n"
        f"snapshot_download(repo_id={input_dataset!r}, repo_type='dataset', local_dir='/tmp/in')\n"
        "import shutil\n"
        "shutil.copy('/tmp/in/dataset_with_labels.parquet', 'data/processed/dataset_with_labels.parquet')\n"
        "for f in ('sentiment_cache.parquet','title_embeddings.npy','title_embeddings.ids.parquet'):\n"
        "    shutil.copy(f'/tmp/in/{f}', f'data/interim/{f}')\n"
        "print('data staged')\n"
        "PY\n"
        f"python scripts/train_transformer.py --model {model} --config {train_config}\n"
        "mkdir -p /tmp/out\n"
        "cp -r reports/artifacts/predictions /tmp/out/\n"
        "cp -r reports/artifacts/models /tmp/out/\n"
        f"git rev-parse HEAD > /tmp/out/git_sha.txt\n"
        "pip freeze > /tmp/out/pip_freeze.txt\n"
        "nvidia-smi > /tmp/out/nvidia_smi.txt 2>&1 || true\n"
        "python - <<PY\n"
        "from huggingface_hub import HfApi, create_repo\n"
        f"create_repo({output_repo!r}, repo_type='dataset', private=True, exist_ok=True)\n"
        f"HfApi().upload_folder(folder_path='/tmp/out', repo_id={output_repo!r}, repo_type='dataset', "
        "commit_message='HF Jobs encoder fine-tune outputs')\n"
        f"print('uploaded -> {output_repo}')\n"
        "PY\n"
        "echo '== DONE =='\n"
    )


def submit_job(
    model: str,
    sha: str,
    flavor: str,
    image: str,
    timeout: str,
    input_dataset: str,
    output_repo: str,
    train_config: str = "configs/train.yaml",
) -> str:
    """Submit a detached HF Job. Returns the job_id."""
    cmd_str = build_bash_command(model, sha, input_dataset, output_repo, train_config)
    args = [
        "uv",
        "run",
        "hf",
        "jobs",
        "run",
        "--flavor",
        flavor,
        "--timeout",
        timeout,
        "--secrets",
        "HF_TOKEN",
        "--detach",
        "--label",
        f"thesis-encoder-{model}",
        image,
        "bash",
        "-c",
        cmd_str,
    ]
    log.info(f"submitting HF Job: model={model} flavor={flavor} image={image}")
    res = subprocess.run(args, capture_output=True, text=True, check=False)
    sys.stdout.write(res.stdout)
    sys.stderr.write(res.stderr)
    if res.returncode != 0:
        raise SystemExit("hf jobs run failed")
    # Parse job_id from "Job started with ID: <id>" line; fallback: any 24-hex token.
    import re as _re

    output = res.stdout + res.stderr
    m = _re.search(r"Job started with ID:\s*([a-f0-9]{24})", output)
    if not m:
        m = _re.search(r"\b([a-f0-9]{24})\b", output)
    if not m:
        raise SystemExit(f"could not parse job_id from output:\n{output}")
    job_id = m.group(1)
    log.info(f"job_id = {job_id}")
    return job_id


def poll_job(job_id: str) -> str:
    """Poll until the job reaches a terminal state. Returns the final status string."""
    started = time.time()
    last_state: str | None = None
    while True:
        elapsed_min = (time.time() - started) / 60
        if elapsed_min > POLL_TIMEOUT_MIN:
            log.error(f"poll timeout after {POLL_TIMEOUT_MIN} min")
            return "timeout"
        res = subprocess.run(
            ["uv", "run", "hf", "jobs", "inspect", job_id],
            capture_output=True,
            text=True,
            check=False,
        )
        out = res.stdout + res.stderr
        state = "unknown"
        for s in ("COMPLETED", "FAILED", "CANCELED", "RUNNING", "QUEUED", "PENDING"):
            if s in out:
                state = s.lower()
                break
        if state != last_state:
            log.info(f"[{elapsed_min:5.1f} min] state -> {state}")
            last_state = state
        if state in ("completed", "failed", "canceled"):
            return state
        time.sleep(POLL_SECONDS)


def fetch_output(output_repo: str, dest: Path) -> None:
    log.info(f"downloading outputs from {output_repo} -> {dest}")
    ensure_dir(dest)
    subprocess.run(
        [
            "uv",
            "run",
            "hf",
            "download",
            output_repo,
            "--repo-type",
            "dataset",
            "--local-dir",
            str(dest),
        ],
        check=True,
    )


def promote_predictions(out_dir: Path, root: Path, model: str) -> Path | None:
    candidates = list((out_dir / "predictions" / "transformers").glob(f"{model}.parquet"))
    if not candidates:
        candidates = list(out_dir.rglob(f"{model}.parquet"))
    if not candidates:
        log.warning(f"no predictions parquet found under {out_dir}")
        return None
    target = (
        ensure_dir(root / "reports" / "artifacts" / "predictions" / "transformers")
        / f"{model}.parquet"
    )
    shutil.copy2(candidates[0], target)
    log.info(f"promoted {candidates[0]} -> {target}")
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", required=True, choices=["wangchanberta", "phayathaibert", "xlm-roberta-large"]
    )
    ap.add_argument(
        "--flavor", default="t4-small", help="HF Jobs flavor; t4-small ($0.40/h) is default"
    )
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--timeout", default="3h", help="max wallclock e.g. 2h, 90m")
    ap.add_argument("--input-dataset", default=DEFAULT_INPUT_DATASET)
    ap.add_argument(
        "--output-repo", default=None, help="default: {namespace}/thesis-output-{model}-{YYYYMMDD}"
    )
    ap.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    ap.add_argument(
        "--train-config",
        default="configs/train.yaml",
        help="config YAML path inside the cloned repo for train_transformer.py",
    )
    ap.add_argument(
        "--push-only",
        action="store_true",
        help="submit + exit; persist the job_id for later --poll-only",
    )
    ap.add_argument(
        "--poll-only", action="store_true", help="skip submit; poll an existing job + download"
    )
    ap.add_argument("--job-id", default=None, help="required with --poll-only")
    args = ap.parse_args()

    root = project_root()
    state_path = root / "tmp" / "hf_jobs" / f"{args.model}.json"
    ensure_dir(state_path.parent)
    date = datetime.now(UTC).strftime("%Y%m%d")
    output_repo = args.output_repo or f"{args.namespace}/thesis-output-{args.model}-{date}"

    if args.poll_only:
        if not args.job_id and not state_path.exists():
            raise SystemExit("--poll-only requires --job-id or a prior --push-only run")
        if args.job_id:
            job_id = args.job_id
        else:
            saved = json.loads(state_path.read_text())
            job_id = saved["job_id"]
            output_repo = saved.get("output_repo", output_repo)
    else:
        sha = preflight(root)
        job_id = submit_job(
            model=args.model,
            sha=sha,
            flavor=args.flavor,
            image=args.image,
            timeout=args.timeout,
            input_dataset=args.input_dataset,
            output_repo=output_repo,
            train_config=args.train_config,
        )
        state_path.write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "output_repo": output_repo,
                    "sha": sha,
                    "submitted_at": datetime.now(UTC).isoformat(),
                },
                indent=2,
            )
        )
        if args.push_only:
            log.info(f"submitted job_id={job_id}; state saved to {state_path}")
            log.info(f"resume later: --poll-only --job-id {job_id}")
            return

    state = poll_job(job_id)
    cloud_run_dir = ensure_dir(root / "reports" / "cloud_runs" / f"hf-{args.model}-{date}")

    if state == "completed":
        fetch_output(output_repo, cloud_run_dir)
        promote_predictions(cloud_run_dir, root, args.model)
        log.info(f"DONE: job_id={job_id} model={args.model}")
    else:
        log.error(f"job ended with state={state}")
        subprocess.run(["uv", "run", "hf", "jobs", "logs", job_id], check=False)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
