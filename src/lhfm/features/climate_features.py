"""Climate / environmental exposure features.

Heat-health is an underexplored axis in digital biomarker work; we add a few
candidate features so the foundation model can learn to attend to climate
context the way a clinician asking "how hot was it last night?" would.

Standardizations here are *causal* (expanding stats), avoiding the
temporal-leakage trap where a day-t z-score is computed against the
participant's full timeline mean and std.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EXPANDING_MIN_PERIODS = 5


def compute_climate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add climate-derived features.

    New columns:
        humid_heat_index, nighttime_heat_stress, heat_exposure_3d,
        aqi_burden_7d, climate_stress_score
    """
    df = df.sort_values(["participant_id", "date"]).copy()

    # Humid heat index: combine temperature and humidity. This is a *much*
    # cruder proxy than wet-bulb but it correlates strongly and we don't
    # have dewpoint data.
    df["humid_heat_index"] = df["heat_index"] + 0.05 * (df["humidity"] - 50).clip(lower=0)

    # Night-time heat stress. Without an explicit nightly low we approximate
    # using today's heat_index minus a fixed diurnal offset.
    df["nighttime_heat_stress"] = (df["heat_index"] - 5.0).clip(lower=0)

    chunks = []
    for _pid, g in df.groupby("participant_id"):
        g = g.copy()
        g["heat_exposure_3d"] = g["heat_index"].rolling(3, min_periods=1).mean()
        g["aqi_burden_7d"] = g["aqi"].rolling(7, min_periods=1).mean()

        # Composite "is this a hard climate day for this person" score.
        # Causal within-person standardization to absorb climate-zone
        # differences without leaking the future.
        hi_z = _causal_zscore(g["heat_index"])
        aqi_z = _causal_zscore(g["aqi"])
        g["climate_stress_score"] = 0.6 * hi_z + 0.4 * aqi_z
        chunks.append(g)

    return pd.concat(chunks, axis=0).sort_index()


def _causal_zscore(s: pd.Series, min_periods: int = _EXPANDING_MIN_PERIODS) -> pd.Series:
    """Expanding (past-only) z-score for a single participant's series.

    During the warm-up period we emit 0, which is where the imputation
    layer will hand off to per-participant medians.
    """
    if s.notna().sum() < min_periods:
        return pd.Series(0.0, index=s.index)
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std(ddof=0)
    z = (s - mu) / sd.replace(0.0, np.nan)
    return z.fillna(0.0)
