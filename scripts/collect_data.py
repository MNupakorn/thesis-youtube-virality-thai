"""Collect raw YouTube videos via Data API v3.

Drop the user-supplied original collector into ``src/data_collection/__init__.py``
(replace ``collect_videos``). This script wires the config + saves the output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.data_collection import collect_videos, save_raw
from src.utils import load_yaml, setup_logger

log = setup_logger("scripts.collect")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()

    load_dotenv()
    cfg = load_yaml(args.config)

    df = collect_videos(cfg)
    out_path = Path(cfg["paths"]["raw_dir"]) / cfg["raw_filename"]
    save_raw(df, out_path)
    log.info(f"collected {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
