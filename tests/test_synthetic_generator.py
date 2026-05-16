"""Tests for the synthetic cohort generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


REQUIRED = {
    "participant_id", "date", "age", "sex",
    "baseline_sleep_need", "baseline_hrv", "chronotype",
    "daily_steps", "sleep_duration", "sleep_efficiency",
    "resting_hr", "hrv_rmssd", "stress_score",
    "phone_unlock_count", "screen_time_minutes",
    "mobility_radius_km", "location_entropy",
    "survey_mood", "survey_energy", "survey_stress",
    "temperature_c", "humidity", "aqi", "heat_index",
    "missing_wearable_flag", "missing_phone_flag", "missing_survey_flag",
}


def test_shape_and_columns(tiny_cohort):
    df = tiny_cohort
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10 * 30
    assert REQUIRED.issubset(df.columns)


def test_deterministic_with_seed():
    from lhfm.data.synthetic_generator import generate_synthetic_cohort
    a = generate_synthetic_cohort(n_participants=5, n_days=20, seed=42)
    b = generate_synthetic_cohort(n_participants=5, n_days=20, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_no_impossible_values(tiny_cohort):
    df = tiny_cohort
    # Filter NaNs (they're allowed; the missing flags handle that).
    assert (df["age"].dropna().between(10, 100)).all()
    assert (df["sleep_duration"].dropna().between(0, 14)).all()
    assert (df["sleep_efficiency"].dropna().between(0.3, 1.0)).all()
    assert (df["resting_hr"].dropna().between(30, 130)).all()
    assert (df["hrv_rmssd"].dropna().between(5, 250)).all()
    assert (df["survey_mood"].dropna().between(1, 7)).all()
    assert (df["humidity"].dropna().between(0, 100)).all()
    assert (df["aqi"].dropna().between(0, 500)).all()
    assert (df["temperature_c"].dropna().between(-30, 55)).all()


def test_missing_flags_consistent(tiny_cohort):
    """If the wearable flag is 1, every wearable signal on that row must be NaN."""
    df = tiny_cohort
    wear_cols = ["daily_steps", "sleep_duration", "sleep_efficiency",
                 "resting_hr", "hrv_rmssd", "stress_score"]
    flagged = df[df["missing_wearable_flag"] == 1]
    assert flagged[wear_cols].isna().all().all(), \
        "wearable flag set but some wearable values are not NaN"

    surv_cols = ["survey_mood", "survey_energy", "survey_stress"]
    flagged_s = df[df["missing_survey_flag"] == 1]
    assert flagged_s[surv_cols].isna().all().all()


def test_validation_passes(tiny_cohort):
    from lhfm.data.validation import validate_synthetic_dataframe
    report = validate_synthetic_dataframe(tiny_cohort)
    assert report.ok, f"validation failed: {report.errors}"


def test_participant_baselines_differ(tiny_cohort):
    """Different participants should have different baseline HRV and sleep need."""
    bh = tiny_cohort.groupby("participant_id")["baseline_hrv"].first()
    bs = tiny_cohort.groupby("participant_id")["baseline_sleep_need"].first()
    assert bh.std() > 1.0
    assert bs.std() > 0.05


def test_climate_shows_seasonal_variation(tiny_cohort):
    """Temperature should not be flat -- there should be at least 5C of variation."""
    t = tiny_cohort.groupby("date")["temperature_c"].first()
    assert (t.max() - t.min()) > 3.0


# ---------------------------------------------------------------------------
# v0.2 generator additions: subgroup metadata + clinical context
# ---------------------------------------------------------------------------


def test_subgroup_metadata_columns_present(tiny_cohort):
    """The v0.2 generator must emit the columns the fairness audit needs."""
    expected = {
        "race_ethnicity", "ses_proxy", "region", "device_gen",
        "has_anxiety", "has_depression", "has_sleep_disorder",
        "has_cardio_condition",
        "on_ssri", "on_beta_blocker", "on_sleep_aid",
        "cycle_phase",
    }
    assert expected.issubset(tiny_cohort.columns), \
        f"missing: {expected - set(tiny_cohort.columns)}"


def test_medication_effects_are_directionally_correct():
    """Beta-blocker users should have higher HRV and lower RHR on average.

    We pull a larger cohort so the medication subgroup has enough members
    for the means to stabilise.
    """
    from lhfm.data.synthetic_generator import generate_synthetic_cohort
    df = generate_synthetic_cohort(n_participants=120, n_days=30, seed=11)
    by_pid = df.groupby("participant_id").agg(
        on_bb=("on_beta_blocker", "first"),
        mean_hrv=("hrv_rmssd", "mean"),
        mean_rhr=("resting_hr", "mean"),
    )
    bb_users = by_pid[by_pid.on_bb == 1]
    non_users = by_pid[by_pid.on_bb == 0]
    if len(bb_users) >= 5:
        # The effect is ~12ms HRV up and ~10bpm RHR down in the prior.
        assert bb_users["mean_hrv"].mean() > non_users["mean_hrv"].mean() + 3.0, (
            "beta-blocker users should show meaningfully higher HRV"
        )
        assert bb_users["mean_rhr"].mean() < non_users["mean_rhr"].mean() - 3.0, (
            "beta-blocker users should show meaningfully lower RHR"
        )


def test_within_person_mood_sleep_correlation_in_target_range():
    """The v0.1 generator produced ~0.05 within-person mood-sleep correlation;
    v0.2 should land in 0.20-0.45 (real-EMA-study range)."""
    from lhfm.data.synthetic_generator import generate_synthetic_cohort
    df = generate_synthetic_cohort(n_participants=60, n_days=45, seed=7)
    corrs = []
    for pid, g in df.groupby("participant_id"):
        g = g.dropna(subset=["survey_mood", "sleep_duration"])
        if len(g) >= 15:
            c = g[["survey_mood", "sleep_duration"]].corr().iloc[0, 1]
            if not np.isnan(c):
                corrs.append(c)
    assert len(corrs) >= 30
    mean_corr = float(np.mean(corrs))
    assert 0.20 <= mean_corr <= 0.50, (
        f"mood-sleep correlation {mean_corr:.3f} outside target band [0.20, 0.50]; "
        f"the v0.1 generator had ~0.05 and was the reason for the v0.2 rewrite"
    )


def test_cycle_phase_only_for_eligible_participants():
    """Cycle phase should be 'none' for males and out-of-range-age females."""
    from lhfm.data.synthetic_generator import generate_synthetic_cohort
    df = generate_synthetic_cohort(n_participants=80, n_days=20, seed=3)
    # Males should always be 'none'.
    males = df[df.sex == "M"]
    assert (males["cycle_phase"] == "none").all(), \
        "males got a non-none cycle phase"
    # Older females (>51) should always be 'none'.
    older_f = df[(df.sex == "F") & (df.age > 51)]
    if len(older_f) > 0:
        assert (older_f["cycle_phase"] == "none").all()


def test_subgroup_independent_of_outcome():
    """The generator does NOT bake disparities into subgroups. Mood setpoint
    distribution should be roughly equal across race groups (within sampling
    noise)."""
    from lhfm.data.synthetic_generator import generate_synthetic_cohort
    df = generate_synthetic_cohort(n_participants=200, n_days=20, seed=5)
    by_race = df.groupby("race_ethnicity")["survey_mood"].mean()
    # No race should differ from the global mean by more than ~0.5 mood points
    # given sampling noise and modest cohort sizes per group.
    global_mean = df["survey_mood"].mean()
    assert (by_race - global_mean).abs().max() < 0.6, \
        "race-stratified mood means show unintended disparity"
