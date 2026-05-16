"""Data-source adapters.

Every external dataset gets one adapter. The adapter's job is to turn
the upstream format (Fitbit CSVs, PhysioNet bundles, your pilot's Google
Sheet) into one long-form DataFrame that matches the schema in
``lhfm.data.validation``. Everything downstream of that is dataset-
agnostic.

Why the abstraction:

- ``--adapter <name>`` is a one-flag switch. Swapping LifeSnaps for
  GLOBEM doesn't touch any model code.
- The validator catches schema slips loudly. If an adapter puts minutes
  into a hours field, training fails immediately with a clear message
  instead of producing a mysteriously bad AUROC.
- Adapter-specific quirks (LifeSnaps has no phone sensing, GLOBEM has
  no HRV, both need weather) live in one place per dataset.

Built-in adapters:
    synthetic - the generator from lhfm.data.synthetic_generator
    lifesnaps - Yfantidou et al. 2022, n=71, Fitbit Sense + EMA
    globem    - Xu et al. 2022, n=497 across 4 institute-years
"""

from __future__ import annotations

from .base import (
    AdapterConfig,
    AdapterError,
    BaseAdapter,
    get_adapter,
    list_adapters,
    preflight_report,
    register_adapter,
)

# Side-effect imports: each module calls register_adapter at module scope.
from . import synthetic, lifesnaps, globem  # noqa: F401

__all__ = [
    "AdapterConfig",
    "AdapterError",
    "BaseAdapter",
    "get_adapter",
    "list_adapters",
    "preflight_report",
    "register_adapter",
]
