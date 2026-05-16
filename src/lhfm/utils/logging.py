"""Thin wrapper around the stdlib logger so we get a consistent format everywhere."""

from __future__ import annotations

import logging
from typing import Optional


_CONFIGURED = False
_DEFAULT_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def _configure_root(level: int, fmt: str) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers if something else (uvicorn, jupyter) already
    # attached one.
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str, level: Optional[str] = None, fmt: Optional[str] = None) -> logging.Logger:
    """Return a module-level logger, configuring the root logger on first call."""
    lvl = getattr(logging, (level or "INFO").upper(), logging.INFO)
    _configure_root(lvl, fmt or _DEFAULT_FMT)
    return logging.getLogger(name)
