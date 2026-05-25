"""Shared utilities: config loading, seeding, logging, device selection."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Seed every source of randomness we touch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except ImportError:
        pass


def get_device(preference: str = "auto") -> str:
    """Resolve device string. preference in {auto, cuda, mps, cpu}."""
    if preference != "auto":
        return preference
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def setup_logger(name: str = "thesis", level: str = "INFO") -> logging.Logger:
    """Build a logger with a rich-friendly format."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def project_root() -> Path:
    """Repository root (assumes this file lives at src/utils.py)."""
    return Path(__file__).resolve().parent.parent


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if missing, return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
