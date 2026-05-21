"""Derived features from wearable streams (sleep + cardiovascular).

All functions are pure: they take the long-form dataframe in and return it
augmented with new columns. None of them drop or reorder rows.

**Causality of statistics.** Every within-person standardization here uses
*expanding* (or causal-rolling) means and standard deviations. The naive
``(s - s.mean()) / s.std()`` over the full participant timeline is what we
started with and what most digital-health papers ship; it is also a
temporal-leakage bug, because at day 5 of training you're using mean/std
from days 1-90. The expanding form means each day sees only its past.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EXPANDING_MIN_PERIODS = 5


def _causal_zscore(s: pd.Series, min_periods: int = _EXPANDING_MIN_PERIODS) -> pd.Series:
    """Expanding (past-only) z-score for a single participant's series.

    Until ``min_periods`` observations are available we emit 0, which is
    where the imputation layer will hand off to per-participant medians.
    We also guard against the constant-series degenerate case.
    """
    if s.notna().sum() < min_periods:
        return pd.Series(0.0, index=s.index)
    # Expanding mean/std use [0, t] so the value at t depends only on the past
    # and present, never the future.
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std(ddof=0)
    z = (s - mu) / sd.replace(0.0, np.nan)
    # During the warm-up period mu/sd are NaN; backfill those positions with 0.
    return z.fillna(0.0)


def compute_wearable_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer the canonical wearable-derived features.

    Adds the following columns:
        sleep_regularity_index, sleep_duration_7d_mean,
        hrv_dev_from_baseline, rhr_dev_from_baseline,
        stress_burden_7d, recovery_score
    """
    df = df.sort_values(["participant_id", "date"]).copy()
    out_chunks = []

    for _pid, g in df.groupby("participant_id"):
        g = g.copy()

        # Sleep regularity: -|change in midpoint proxy|. Without bed/wake
        # times we approximate using the deviation of sleep_duration from
        # the trailing personal rolling mean. Higher = more regular.
        rolling_sleep = g["sleep_duration"].rolling(7, min_periods=2).mean()
        diff = (g["sleep_duration"] - rolling_sleep).abs()
        g["sleep_regularity_index"] = 1.0 - (diff / (diff.rolling(14, min_periods=3).max() + 1e-6))

        g["sleep_duration_7d_mean"] = rolling_sleep

        # Personal baseline deviations. We use the *participant prior* baseline
        # where available (set at enrollment, not learned from this timeline)
        # because the rolling baseline is undefined in the first few days.
        if "baseline_hrv" in g.columns:
            g["hrv_dev_from_baseline"] = g["hrv_rmssd"] - g["baseline_hrv"]
        else:
            # No enrollment baseline; use expanding mean (causal).
            g["hrv_dev_from_baseline"] = (
                g["hrv_rmssd"] - g["hrv_rmssd"].expanding(min_periods=3).mean()
            )

        # Personal rolling RHR baseline since we don't have a ground-truth
        # baseline RHR in the dataframe. Use a trailing window so this is
        # causal at training time.
        rolling_rhr_base = g["resting_hr"].rolling(14, min_periods=3).mean()
        g["rhr_dev_from_baseline"] = g["resting_hr"] - rolling_rhr_base

        # Stress burden: 7-day mean of *causal* standardized stress.
        # The prior version used the full-cohort timeline for the
        # standardization step which leaked future days into past features.
        z_stress = _causal_zscore(g["stress_score"])
        g["stress_burden_7d"] = z_stress.rolling(7, min_periods=2).mean()

        # Recovery score: composite of (high HRV + adequate sleep + low stress).
        # Each component scaled to roughly [-1, 1] before averaging.
        hrv_term = g["hrv_dev_from_baseline"] / (g["baseline_hrv"].iloc[0] * 0.25 + 1e-6) \
            if "baseline_hrv" in g.columns else pd.Series(0.0, index=g.index)
        sleep_term = (g["sleep_duration"] - g["baseline_sleep_need"]) / 1.5 \
            if "baseline_sleep_need" in g.columns else pd.Series(0.0, index=g.index)
        stress_term = -z_stress

        components = pd.DataFrame({
            "hrv": np.clip(hrv_term, -1.5, 1.5),
            "sleep": np.clip(sleep_term, -1.5, 1.5),
            "stress": np.clip(stress_term, -1.5, 1.5),
        })
        g["recovery_score"] = components.mean(axis=1)

        out_chunks.append(g)

    return pd.concat(out_chunks, axis=0).sort_index()
