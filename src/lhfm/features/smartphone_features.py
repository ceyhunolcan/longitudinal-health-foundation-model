"""Smartphone passive-sensing features.

These mirror what GPS- and screen-time-based digital-phenotyping pipelines
typically derive (Torous, Onnela, Saeb and others).

All within-person standardization here is *causal* (expanding mean/std), to
avoid the temporal leakage that crops up when a day-t feature is computed
against the participant's full-timeline distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EXPANDING_MIN_PERIODS = 5


def _causal_zscore(s: pd.Series, min_periods: int = _EXPANDING_MIN_PERIODS) -> pd.Series:
    if s.notna().sum() < min_periods:
        return pd.Series(0.0, index=s.index)
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std(ddof=0)
    z = (s - mu) / sd.replace(0.0, np.nan)
    return z.fillna(0.0)


def compute_smartphone_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add smartphone-derived features.

    New columns:
        screen_time_z, unlock_freq_z, behavioral_regularity
    (mobility_radius_km and location_entropy are passed through from raw.)
    """
    df = df.sort_values(["participant_id", "date"]).copy()
    chunks = []

    for _pid, g in df.groupby("participant_id"):
        g = g.copy()

        # Within-person *causal* z-scores. These give the foundation model a
        # feature that means "is today's screen time higher than this person
        # has tended to run *so far*" -- not "...than they will run over the
        # next month", which is what the non-causal version implied.
        g["screen_time_z"] = _causal_zscore(g["screen_time_minutes"])
        g["unlock_freq_z"] = _causal_zscore(g["phone_unlock_count"])

        # Behavioral regularity: 1 - rolling stdev of the smartphone signals,
        # normalized against a within-person expanding max so the value sits
        # roughly in [0, 1] without peeking at the future.
        sig = g[["screen_time_minutes", "phone_unlock_count", "mobility_radius_km"]]
        rolling_std = sig.rolling(7, min_periods=2).std().mean(axis=1)
        max_std = rolling_std.expanding(min_periods=3).max()
        g["behavioral_regularity"] = 1.0 - (rolling_std / (max_std + 1e-6))
        chunks.append(g)

    return pd.concat(chunks, axis=0).sort_index()
