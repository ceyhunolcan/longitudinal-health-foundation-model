"""Regression tests pinning the post-audit fixes in place.

If any of these start failing, somebody has reintroduced a bug we
specifically fixed during the senior-PI audit pass. Don't relax the
assertions without reading CHANGELOG.md to understand why they exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lhfm.data.synthetic_generator import generate_synthetic_cohort
from lhfm.data.preprocessing import binarize_targets, build_windows
from lhfm.features import build_full_feature_table
from lhfm.features.baseline_features import (
    AGE_REF_MEAN,
    AGE_REF_STD,
    compute_baseline_features,
)


class TestAgeZSingleParticipant:
    """The API path scores one participant at a time, so age_z must not
    NaN out when the input dataframe contains only one unique age."""

    def test_single_participant_age_z_finite(self):
        df = pd.DataFrame({
            "age": [34] * 14,
            "sex": ["F"] * 14,
            "chronotype": ["intermediate"] * 14,
        })
        out = compute_baseline_features(df)
        assert out["age_z"].notna().all()
        assert np.isfinite(out["age_z"]).all()
        # 34 should be near the reference mean -> near zero.
        assert abs(out["age_z"].iloc[0]) < 0.1

    def test_age_z_uses_fixed_population_stats(self):
        df = pd.DataFrame({"age": [AGE_REF_MEAN], "sex": ["F"], "chronotype": ["morning"]})
        out = compute_baseline_features(df)
        assert out["age_z"].iloc[0] == pytest.approx(0.0)

        df2 = pd.DataFrame({"age": [AGE_REF_MEAN + AGE_REF_STD], "sex": ["M"], "chronotype": ["morning"]})
        out2 = compute_baseline_features(df2)
        assert out2["age_z"].iloc[0] == pytest.approx(1.0)


class TestClimateTargetNaN:
    """target_climate_vulnerable used to default to True when HRV was NaN.
    Now NaN HRV or NaN heat_index should yield NaN target."""

    def test_nan_hrv_yields_nan_target(self):
        df = pd.DataFrame({
            "heat_index": [33.0, 33.0, 33.0],
            "hrv_rmssd": [40.0, np.nan, 60.0],
            "baseline_hrv": [55.0, 55.0, 55.0],
            "survey_mood": [5.0, 5.0, 5.0],
            "survey_stress": [3.0, 3.0, 3.0],
            "sleep_duration": [7.5, 7.5, 7.5],
            "sleep_efficiency": [0.88, 0.88, 0.88],
        })
        out = binarize_targets(df)
        # Row 0: hot day + HRV 15 ms below baseline -> positive.
        assert out["target_climate_vulnerable"].iloc[0] == 1.0
        # Row 1: hot day but HRV missing -> NaN.
        assert np.isnan(out["target_climate_vulnerable"].iloc[1])
        # Row 2: hot day but HRV only 5 below baseline -> not vulnerable.
        assert out["target_climate_vulnerable"].iloc[2] == 0.0

    def test_nan_source_propagates_to_other_targets(self):
        df = pd.DataFrame({
            "heat_index": [22.0, 22.0],
            "hrv_rmssd": [55.0, 55.0],
            "baseline_hrv": [55.0, 55.0],
            "survey_mood": [np.nan, 5.0],
            "survey_stress": [4.0, np.nan],
            "sleep_duration": [7.5, 7.5],
            "sleep_efficiency": [0.88, np.nan],
        })
        out = binarize_targets(df)
        assert np.isnan(out["target_low_mood"].iloc[0])
        assert np.isnan(out["target_high_stress"].iloc[1])
        assert np.isnan(out["target_sleep_disruption"].iloc[1])


class TestBuildWindowsConsecutiveDates:
    """build_windows should skip windows spanning calendar gaps."""

    def test_skips_window_with_date_gap(self):
        # 8 contiguous days, then a 2-day gap, then 8 more.
        dates_a = pd.date_range("2024-01-01", periods=8)
        dates_b = pd.date_range("2024-01-11", periods=8)  # gap of 2 days
        all_dates = list(dates_a) + list(dates_b)
        df = pd.DataFrame({
            "participant_id": ["P01"] * 16,
            "date": [d.date().isoformat() for d in all_dates],
            "x": np.arange(16, dtype=np.float32),
            "target": [0] * 16,
        })

        X_strict, _, _, _ = build_windows(
            df, feature_cols=["x"], target_col="target",
            window_days=7, stride=1, target_mode="next_day",
            require_consecutive_dates=True,
        )
        X_loose, _, _, _ = build_windows(
            df, feature_cols=["x"], target_col="target",
            window_days=7, stride=1, target_mode="next_day",
            require_consecutive_dates=False,
        )
        # Strict mode drops the windows that straddle the gap.
        assert len(X_strict) < len(X_loose)
        assert len(X_strict) > 0


class TestAPIFallbackZeroHandling:
    """The fallback predictor used to replace AQI=0 with the default 50.
    A clean-air day should NOT be treated as missing."""

    def test_coerce_preserves_zero(self):
        # _coerce lives in src.api.main, which requires fastapi to import.
        # Skip when fastapi isn't available -- matches the test_api.py pattern.
        pytest.importorskip("fastapi")
        from lhfm.api.main import _coerce
        assert _coerce(0.0, 50.0) == 0.0
        assert _coerce(0, 50.0) == 0.0
        assert _coerce(None, 50.0) == 50.0
        assert _coerce(float("nan"), 50.0) == 50.0
        assert _coerce(float("inf"), 50.0) == 50.0


class TestHeatIndexHumidityGate:
    """Heat-index regression is invalid below 40% humidity even at high T."""

    def test_dry_hot_day_returns_ambient(self):
        from lhfm.data.synthetic_generator import _heat_index
        t = np.array([35.0, 35.0])
        rh = np.array([20.0, 60.0])   # one dry, one humid
        hi = _heat_index(t, rh)
        # Dry: fall back to ambient.
        assert hi[0] == pytest.approx(35.0)
        # Humid: regression engaged, heat index > ambient.
        assert hi[1] > 35.0
