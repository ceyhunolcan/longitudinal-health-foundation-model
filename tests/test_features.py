"""Tests for the feature-engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_feature_table_shape(engineered, tiny_cohort):
    # Same row count as the raw frame.
    assert len(engineered) == len(tiny_cohort)
    # And meaningfully more columns.
    assert engineered.shape[1] > tiny_cohort.shape[1] + 15


def test_expected_engineered_columns(engineered):
    expected = {
        "sleep_regularity_index", "sleep_duration_7d_mean",
        "hrv_dev_from_baseline", "rhr_dev_from_baseline",
        "stress_burden_7d", "recovery_score",
        "screen_time_z", "unlock_freq_z", "behavioral_regularity",
        "humid_heat_index", "nighttime_heat_stress", "heat_exposure_3d",
        "aqi_burden_7d", "climate_stress_score",
        "any_missing", "consecutive_missing_days",
        "missingness_rate_7d", "modality_dropout_entropy",
        "age_z", "chronotype_score", "sex_male",
    }
    missing = expected - set(engineered.columns)
    assert not missing, f"missing engineered columns: {missing}"


def test_no_nans_in_numeric_features_after_impute(engineered):
    """Imputation should leave no NaN behind in any non-target numeric column."""
    feat = engineered
    numeric = feat.select_dtypes(include=[np.number]).columns
    non_target = [c for c in numeric if not c.startswith("target_")]
    # missing_*_flag are always integer 0/1; they should never be NaN to begin with.
    nan_counts = feat[non_target].isna().sum()
    assert (nan_counts == 0).all(), nan_counts[nan_counts > 0].to_dict()


def test_target_columns_present(engineered):
    for c in ["target_low_mood", "target_high_stress",
              "target_sleep_disruption", "target_climate_vulnerable"]:
        assert c in engineered.columns


def test_within_person_zscore_is_near_zero_mean(engineered):
    """Within-person z-scores should have near-zero mean inside each participant."""
    for pid, g in engineered.groupby("participant_id"):
        # If the column has any variance the within-person z mean should be small.
        z_mean = g["screen_time_z"].mean()
        assert abs(z_mean) < 1e-6 or pd.isna(z_mean)


def test_recovery_score_is_finite(engineered):
    assert np.isfinite(engineered["recovery_score"]).all()


def test_build_windows_shape():
    from lhfm.data.preprocessing import build_windows
    from lhfm.data.synthetic_generator import generate_synthetic_cohort
    from lhfm.features import build_full_feature_table

    df = generate_synthetic_cohort(n_participants=4, n_days=25, seed=1)
    feat = build_full_feature_table(df, impute=True, add_targets=True)
    feature_cols = ["sleep_duration", "hrv_rmssd", "stress_burden_7d", "heat_index"]
    X, y, pids, dates = build_windows(
        feat, feature_cols=feature_cols, target_col="target_low_mood",
        window_days=10, stride=1, target_mode="next_day",
    )
    # 25 days, window 10, next_day target: last_start = 25 - 10 - 1 = 14,
    # so starts run 0..14 inclusive -> 15 windows per participant.
    # 4 participants * 15 starts = 60 windows total.
    assert X.shape == (4 * 15, 10, len(feature_cols))
    assert y.shape == (4 * 15,)
    assert pids.shape == (4 * 15,)
    assert set(pids.tolist()) == set(feat["participant_id"].unique())
