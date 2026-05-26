"""Local driver: push a Kaggle kernel for one encoder, poll, and download results.

Usage::

    python scripts/cloud/run_on_kaggle.py --model wangchanberta
    python scripts/cloud/run_on_kaggle.py --model phayathaibert
    python scripts/cloud/run_on_kaggle.py --model xlm-roberta-large

Pre-flight (auto):
1. Working tree is clean (no uncommitted changes).
2. HEAD SHA exists on ``origin/main`` (push first if not).

What it does:
1. Renders ``kaggle_kernel_template.py`` with model alias + git SHA.
2. Writes ``kernel-metadata.json`` with GPU enabled and the data dataset attached.
3. ``kaggle kernels push`` to slug ``mknpk01/thesis-{model}-{date}``.
4. Polls status every 30s, logs every state change.
5. On ``complete``: downloads outputs, copies predictions parquet to
   ``reports/predictions/`` (canonical location).
6. On ``error``: dumps last 100 lines of the kernel log to stdout.
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

log = setup_logger("scripts.cloud.run_on_kaggle")

KAGGLE_USER = "mknpk01"
DATA_DATASET = "mknpk01/thesis-virality-data"
TEMPLATE = Path("scripts/cloud/kaggle_kernel_template.py")
POLL_SECONDS = 30
POLL_TIMEOUT_MIN = 360  # hard cap 6h


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, check=False)
    if check and res.returncode != 0:
        log.error(f"command failed: {' '.join(cmd)}")
        log.error(f"stdout: {res.stdout}")
        log.error(f"stderr: {res.stderr}")
        raise SystemExit(res.returncode)
    return res


def preflight(root: Path) -> str:
    """Return the HEAD SHA after verifying clean tree + SHA on origin/main."""
    status = run(["git", "status", "--porcelain"], cwd=root).stdout.strip()
    if status:
        log.error("working tree is not clean:")
        log.error(status)
        raise SystemExit("commit or stash before running on Kaggle (reproducibility rule).")

    sha = run(["git", "rev-parse", "HEAD"], cwd=root).stdout.strip()
    log.info(f"HEAD = {sha}")

    run(["git", "fetch", "origin", "main"], cwd=root)
    contains = run(["git", "branch", "-r", "--contains", sha], cwd=root, check=False).stdout
    if "origin/main" not in contains:
        log.error(f"SHA {sha} is not on origin/main. Push it first:")
        log.error("  git push origin main")
        raise SystemExit(1)
    return sha


def render_kernel(template: Path, model: str, sha: str) -> str:
    body = template.read_text()
    return body.replace("{{MODEL_ALIAS}}", model).replace("{{GIT_SHA}}", sha)


def write_kernel_files(staging: Path, model: str, sha: str, slug: str) -> None:
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    body = render_kernel(project_root() / TEMPLATE, model, sha)
    (staging / "kernel.py").write_text(body)

    meta = {
        "id": slug,
        "title": f"thesis-{model}-{datetime.now(UTC).strftime('%Y%m%d')}",
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [DATA_DATASET],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (staging / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))


def push_kernel(staging: Path) -> None:
    log.info(f"pushing kernel from {staging}")
    res = run(["kaggle", "kernels", "push", "-p", str(staging)], check=False)
    sys.stdout.write(res.stdout)
    sys.stderr.write(res.stderr)
    if res.returncode != 0:
        raise SystemExit("kaggle kernels push failed")


def poll_status(slug: str) -> str:
    """Block until status is complete/error/cancel*. Returns final status."""
    started = time.time()
    last_state: str | None = None
    while True:
        elapsed_min = (time.time() - started) / 60
        if elapsed_min > POLL_TIMEOUT_MIN:
            log.error(f"poll timeout after {POLL_TIMEOUT_MIN} min")
            return "timeout"

        res = run(["kaggle", "kernels", "status", slug], check=False)
        out = (res.stdout + res.stderr).lower()
        state = "unknown"
        for s in ("complete", "error", "cancelled", "cancelling", "running", "queued"):
            if s in out:
                state = s
                break
        if state != last_state:
            log.info(f"[{elapsed_min:5.1f} min] status -> {state}")
            last_state = state
        if state in ("complete", "error", "cancelled"):
            return state
        time.sleep(POLL_SECONDS)


def download_outputs(slug: str, dest: Path) -> None:
    ensure_dir(dest)
    log.info(f"downloading outputs to {dest}")
    run(["kaggle", "kernels", "output", slug, "-p", str(dest)])


def dump_log(slug: str, dest: Path) -> None:
    log_path = dest / "kernel.log"
    res = run(["kaggle", "kernels", "output", slug, "-p", str(dest)], check=False)
    log.info(res.stdout)
    if log_path.exists():
        tail = log_path.read_text().splitlines()[-100:]
        log.error("--- kernel log tail ---")
        for line in tail:
            log.error(line)


def promote_predictions(cloud_run_dir: Path, root: Path, model: str) -> Path | None:
    """Copy {model}.parquet from kernel output to reports/predictions/."""
    candidates = (
        list((cloud_run_dir / "predictions").glob("*.parquet"))
        if (cloud_run_dir / "predictions").exists()
        else []
    )
    if not candidates:
        candidates = list(cloud_run_dir.glob(f"**/{model}.parquet"))
    if not candidates:
        log.warning(f"no predictions parquet found under {cloud_run_dir}")
        return None
    target = ensure_dir(root / "reports" / "predictions") / f"{model}.parquet"
    shutil.copy2(candidates[0], target)
    log.info(f"promoted {candidates[0]} -> {target}")
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        required=True,
        choices=["wangchanberta", "phayathaibert", "xlm-roberta-large"],
    )
    ap.add_argument(
        "--staging",
        default=None,
        help="local staging dir (default: tmp/kaggle_kernels/{slug-name})",
    )
    ap.add_argument(
        "--push-only",
        action="store_true",
        help="render + push the kernel and exit. Use to queue multiple models in "
        "parallel; later run with --poll-only (or the bare driver) to wait + collect.",
    )
    ap.add_argument(
        "--poll-only",
        action="store_true",
        help="skip push; assume the kernel is already submitted. Polls + downloads.",
    )
    args = ap.parse_args()

    root = project_root()
    sha = preflight(root)

    date = datetime.now(UTC).strftime("%Y%m%d")
    slug_name = f"thesis-{args.model}-{date}"
    slug = f"{KAGGLE_USER}/{slug_name}"

    staging = Path(args.staging) if args.staging else (root / "tmp" / "kaggle_kernels" / slug_name)

    if not args.poll_only:
        write_kernel_files(staging, args.model, sha, slug)
        push_kernel(staging)
        if args.push_only:
            log.info(f"pushed {slug}; skipping poll (--push-only). Run --poll-only later.")
            return

    state = poll_status(slug)
    cloud_run_dir = ensure_dir(root / "reports" / "cloud_runs" / slug_name)

    if state == "complete":
        download_outputs(slug, cloud_run_dir)
        promote_predictions(cloud_run_dir, root, args.model)
        log.info(f"DONE: {slug} (sha={sha})")
    else:
        log.error(f"kernel finished with state={state}")
        dump_log(slug, cloud_run_dir)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
