"""Base class and registry for dataset adapters.

An adapter turns one external dataset (LifeSnaps CSVs, GLOBEM PhysioNet
bundles, a hand-rolled pilot's Google Sheets export, ...) into one
long-form dataframe matching LHFM's schema. Everything downstream of
that -- features, training, eval, fairness audit -- is dataset-agnostic.

The contract:

1. ``load_raw(self) -> pd.DataFrame`` returns a long-form frame with the
   columns LHFM requires (see ``lhfm.data.validation.REQUIRED_COLUMNS``).
   When the source dataset doesn't have something (e.g. LifeSnaps has no
   smartphone sensing), put NaN and let the missingness flag handle it.
2. The frame must pass ``validate_synthetic_dataframe``.
3. Declare ``REQUIRES_WEATHER_ENRICHMENT = True`` if the source dataset
   has no climate columns. The base class calls ``enrich_with_weather``
   for you; you only need to make sure the rows carry ``latitude`` and
   ``longitude`` (per-row, or via ``config.default_lat``/``default_lon``).
4. Override ``binarize_targets`` only if the source has its own outcome
   label (e.g. GLOBEM ships BDI-II → ``target_depressed`` independent of
   the EMA-based default targets).

Registry: each adapter calls ``register_adapter(name, cls)`` at import.
``scripts/run_pipeline.py --adapter <name>`` looks them up by that name.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


log = logging.getLogger(__name__)


class AdapterError(Exception):
    """An adapter couldn't fulfil the schema contract."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type["BaseAdapter"]] = {}


def register_adapter(name: str, cls: type["BaseAdapter"]) -> None:
    """Register ``cls`` under ``name`` so the CLI flag can find it."""
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        log.debug("re-registering adapter %s (was %s)", name, _REGISTRY[name])
    _REGISTRY[name] = cls


def get_adapter(name: str) -> type["BaseAdapter"]:
    if name not in _REGISTRY:
        raise AdapterError(
            f"unknown adapter {name!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_adapters() -> list[str]:
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AdapterConfig:
    """Common knobs every adapter accepts. Subclass-specific args go on the
    subclass's __init__ (see e.g. SyntheticAdapter)."""

    raw_dir: Path                       # where the source dataset lives
    cache_dir: Path | None = None       # for the weather cache
    enrich_weather: bool = True         # turn off in tests to avoid network
    weather_provider: str = "open-meteo"
    weather_csv: Path | None = None     # required if provider == "csv"
    default_lat: float | None = None    # study-site fallback
    default_lon: float | None = None
    max_participants: int | None = None # smoke-test cap


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------


def _natural_key(s: str) -> list:
    """Sort key that orders 'P9' before 'P10' instead of alphabetically.

    We use this when applying ``max_participants`` so a cap of 10 actually
    picks the first 10 by participant number rather than the first 10
    alphabetically, which is what 'P1, P10, P100, P2, ...' would give you.
    """
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


class BaseAdapter:
    """Subclass this for each new dataset.

    Minimum implementation::

        class MyAdapter(BaseAdapter):
            NAME = "mydataset"

            def load_raw(self) -> pd.DataFrame:
                ...  # return DataFrame matching LHFM schema

        register_adapter(MyAdapter.NAME, MyAdapter)
    """

    NAME: str = ""
    REQUIRES_WEATHER_ENRICHMENT: bool = False
    # Adapter-specific plausibility-range overrides. Merged into the
    # defaults from lhfm.data.validation for the duration of validation.
    # Use this only when the upstream dataset really does use a different
    # scale; the right answer most of the time is to normalise the data,
    # not to widen the validator's tolerance.
    PLAUSIBILITY_OVERRIDES: dict[str, tuple[float, float]] = {}

    def __init__(self, config: AdapterConfig):
        self.config = config
        if not config.raw_dir.exists():
            raise AdapterError(
                f"raw_dir does not exist: {config.raw_dir}. "
                f"Download the dataset first; see docs/adapters/{self.NAME}.md."
            )

    # -- to implement ----------------------------------------------------

    def load_raw(self) -> pd.DataFrame:
        """Return a long-form dataframe matching LHFM's required schema."""
        raise NotImplementedError

    # -- optional overrides ----------------------------------------------

    def binarize_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Default: use ``lhfm.data.preprocessing.binarize_targets``.

        Override when the source has its own outcome label.
        """
        from lhfm.data.preprocessing import binarize_targets
        return binarize_targets(df)

    def participant_subgroups(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hook for adapters that want to attach extra subgroup columns
        (race, SES, region, ...). Default: passthrough."""
        return df

    # -- entry point used by run_pipeline.py -----------------------------

    def build(self) -> pd.DataFrame:
        """Load → enforce schema → enrich weather → subgroups → validate."""
        log.info("[%s] loading from %s", self.NAME, self.config.raw_dir)
        df = self.load_raw()

        if self.config.max_participants is not None:
            keep = sorted(df["participant_id"].unique().tolist(), key=_natural_key)
            keep = keep[: self.config.max_participants]
            df = df[df["participant_id"].isin(keep)].copy()
            log.info("[%s] capped to %d participants", self.NAME, len(keep))

        df = self._enforce_schema(df)

        if self.REQUIRES_WEATHER_ENRICHMENT and self.config.enrich_weather:
            df = self._enrich_with_weather(df)

        df = self.participant_subgroups(df)
        df = self._final_validation(df)

        log.info("[%s] built: %d rows × %d cols, %d participants",
                 self.NAME, len(df), df.shape[1], df["participant_id"].nunique())
        return df

    # -- internals -------------------------------------------------------

    def _enforce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill in any required columns the adapter didn't supply, normalise
        dtypes for ``participant_id`` and ``date``, and reorder columns so
        the required schema is first.
        """
        from lhfm.data.validation import REQUIRED_COLUMNS

        df = df.copy()
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                # Missingness flags default to 0; everything else to NaN.
                if col.startswith("missing_") and col.endswith("_flag"):
                    df[col] = 0
                else:
                    df[col] = np.nan

        # Two foot-guns we hit every time:
        # - participant_id arriving as int when downstream code groups by str
        # - date arriving as ISO string when downstream code wants datetime.date
        df["participant_id"] = df["participant_id"].astype(str).astype(object)
        try:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        except Exception as exc:
            raise AdapterError(f"can't parse date column: {exc}") from exc

        extras = [c for c in df.columns if c not in REQUIRED_COLUMNS]
        return df[REQUIRED_COLUMNS + sorted(extras)]

    def _enrich_with_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        from lhfm.data.weather import enrich_with_weather
        return enrich_with_weather(df, self.config)

    def _final_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        from lhfm.data.validation import (
            PLAUSIBILITY_RANGES,
            validate_synthetic_dataframe,
        )

        # Temporarily merge adapter overrides into the module-level
        # PLAUSIBILITY_RANGES dict. Yes, this is a monkey-patch; the
        # alternative was threading the dict through three layers, and
        # nobody else in the process is running validation concurrently.
        import lhfm.data.validation as v
        saved = v.PLAUSIBILITY_RANGES
        v.PLAUSIBILITY_RANGES = {**saved, **self.PLAUSIBILITY_OVERRIDES}
        try:
            report = validate_synthetic_dataframe(df, strict=False)
        finally:
            v.PLAUSIBILITY_RANGES = saved

        for w in report.warnings:
            log.warning("[%s] %s", self.NAME, w)
        if not report.ok:
            raise AdapterError(
                f"[{self.NAME}] validation failed:\n - "
                + "\n - ".join(report.errors)
            )
        return df


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def preflight_report(df: pd.DataFrame, min_days: int = 14) -> dict[str, Any]:
    """Quick descriptive snapshot of a cohort before training.

    Reports: participant count after the min-length filter, per-modality
    missingness, per-task label balance, demographics. Use this to decide
    in 10 seconds whether the data is in shape to train, rather than
    discovering it 4 minutes into a training run.
    """
    per_pid_days = df.groupby("participant_id")["date"].nunique()
    survivors = per_pid_days[per_pid_days >= min_days]
    report: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_participants": int(df["participant_id"].nunique()),
        "n_participants_after_min_days_filter": int(len(survivors)),
        "n_days_min": int(per_pid_days.min()) if not per_pid_days.empty else 0,
        "n_days_median": int(per_pid_days.median()) if not per_pid_days.empty else 0,
        "n_days_max": int(per_pid_days.max()) if not per_pid_days.empty else 0,
        "date_min": str(df["date"].min()) if "date" in df else None,
        "date_max": str(df["date"].max()) if "date" in df else None,
    }
    if "sex" in df:
        report["sex_distribution"] = df["sex"].value_counts(dropna=False).to_dict() if "sex" in df.columns else {}
    if "age" in df:
        ages = df.drop_duplicates("participant_id")["age"].dropna()
        if len(ages):
            report["age_mean"] = float(ages.mean())
            report["age_std"] = float(ages.std())
    for flag in ("missing_wearable_flag", "missing_phone_flag", "missing_survey_flag"):
        if flag in df:
            report[f"frac_{flag}"] = float(df[flag].mean())

    for col in df.columns:
        if col.startswith("target_"):
            vals = df[col].dropna()
            if len(vals):
                report[f"{col}_n_labelled"] = int(len(vals))
                report[f"{col}_positive_rate"] = float(vals.mean())
    return report
