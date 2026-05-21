"""Configuration loading and a couple of small helpers.

We keep this deliberately simple. There's no need for hydra/omegaconf here:
YAML files plus a shallow dict merge cover every override pattern we use.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Resolve the project root once. Every other path in the repo is relative to
# this, which is what keeps the project portable between laptops and Docker.
# With the src-layout (lhfm lives at src/lhfm/) we have to go up an extra
# level: src/lhfm/utils/config.py -> src/lhfm/utils -> src/lhfm -> src -> repo.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`, returning a new dict."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(*names: str) -> dict[str, Any]:
    """Load and merge one or more YAML config files from ``configs/``.

    Later files override earlier ones. Pass file names without the ``.yaml``
    extension, e.g. ``load_config("default", "model")``.

    If no names are provided we just load ``default.yaml`` since that's
    overwhelmingly the common case.
    """
    if not names:
        names = ("default",)

    merged: dict[str, Any] = {}
    for name in names:
        path = CONFIG_DIR / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, cfg)

    # Make every path absolute so downstream code never has to wonder.
    if "paths" in merged:
        for key, val in list(merged["paths"].items()):
            merged["paths"][key] = str(PROJECT_ROOT / val)

    return merged


def resolve_device(preference: str = "auto") -> str:
    """Pick a torch device string respecting the user's preference."""
    pref = (preference or "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        # We trust the caller. If CUDA isn't actually available torch will
        # complain loudly at first .to(device) call, which is what we want.
        return "cuda"
    # auto
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        # torch missing or borked. CPU is fine.
        pass
    return "cpu"


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def project_root() -> Path:
    """Convenience accessor for scripts that want to write next to the repo."""
    return PROJECT_ROOT
