"""Trivial adapter wrapper around the existing synthetic generator.

Lets ``scripts/run_pipeline.py --adapter synthetic`` work uniformly with
the real-data adapters, so swapping in real data is a 1-flag change
rather than a different code path.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from lhfm.data.synthetic_generator import GeneratorConfig, SyntheticCohortGenerator

from .base import AdapterConfig, BaseAdapter, register_adapter


log = logging.getLogger(__name__)


class SyntheticAdapter(BaseAdapter):
    NAME = "synthetic"
    REQUIRES_WEATHER_ENRICHMENT = False     # generator emits climate columns

    def __init__(self, config: AdapterConfig,
                 n_participants: int = 250, n_days: int = 90, seed: int = 42):
        # The synthetic generator doesn't need a raw_dir; we mock one.
        if not config.raw_dir.exists():
            config.raw_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(config)
        self.n_participants = n_participants
        self.n_days = n_days
        self.seed = seed

    def load_raw(self) -> pd.DataFrame:
        log.info("[synthetic] generating %d participants × %d days (seed=%d)",
                 self.n_participants, self.n_days, self.seed)
        gen_cfg = GeneratorConfig(
            n_participants=self.n_participants,
            n_days=self.n_days,
            seed=self.seed,
        )
        gen = SyntheticCohortGenerator(gen_cfg)
        df = gen.generate()
        return df


register_adapter(SyntheticAdapter.NAME, SyntheticAdapter)
