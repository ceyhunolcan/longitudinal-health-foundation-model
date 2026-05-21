"""Senior-PI regression tests.

These pin the deeper methodological fixes from the second audit pass:

- Train-data-only age standardization (no leak of synthetic prior).
- Causal (expanding) within-person z-scores in features.
- Participant-clustered bootstrap CIs are wider than row-level CIs on
  longitudinal data with high within-person correlation.
- Encoder bounds-checks unseen participant indices.
- Encoder positional encoding rejects oversized inputs cleanly.

Don't relax these without reading the CHANGELOG.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lhfm.features.baseline_features import (
    FALLBACK_AGE_REF_MEAN,
    FALLBACK_AGE_REF_STD,
    compute_baseline_features,
    fit_baseline_reference_stats,
)
from lhfm.features.smartphone_features import compute_smartphone_features
from lhfm.features.wearable_features import compute_wearable_features
from lhfm.utils.metrics import bootstrap_ci


class TestTrainDataAgeStandardization:
    """The audit-1 version of compute_baseline_features used hardcoded
    constants exactly matching the synthetic prior. That's a soft leak of
    population statistics; the fix is to compute them on the training
    split and pass them through.
    """

    def test_fit_baseline_ref_stats_uses_one_row_per_participant(self):
        # 50 days x 3 participants, but only 3 unique ages -> reference mean
        # should be the mean of the 3 ages, not the 150-row mean.
        df = pd.DataFrame({
            "participant_id": ["P01"]*50 + ["P02"]*50 + ["P03"]*50,
            "age": [25]*50 + [40]*50 + [70]*50,
        })
        stats = fit_baseline_reference_stats(df)
        assert stats["age_ref_mean"] == pytest.approx((25 + 40 + 70) / 3)

    def test_fit_handles_tiny_cohort_gracefully(self):
        df = pd.DataFrame({"participant_id": ["P01"], "age": [35]})
        stats = fit_baseline_reference_stats(df)
        # With one participant we can't compute std → fall back to the documented constants.
        assert stats["age_ref_mean"] == FALLBACK_AGE_REF_MEAN
        assert stats["age_ref_std"] == FALLBACK_AGE_REF_STD

    def test_compute_uses_passed_reference_stats(self):
        df = pd.DataFrame({"age": [50.0], "sex": ["F"], "chronotype": ["morning"]})
        out = compute_baseline_features(df, age_ref_mean=40.0, age_ref_std=10.0)
        assert out["age_z"].iloc[0] == pytest.approx(1.0)


class TestCausalFeatureStatistics:
    """Within-person z-scores must not peek at the future."""

    def test_screen_time_z_is_causal(self):
        # A spike on day 15 should not affect the z-score on day 5.
        days = pd.date_range("2024-01-01", periods=30)
        df = pd.DataFrame({
            "participant_id": ["P01"] * 30,
            "date": [d.isoformat() for d in days],
            "screen_time_minutes": [200.0] * 14 + [800.0] + [200.0] * 15,
            "phone_unlock_count": [80.0] * 30,
            "mobility_radius_km": [5.0] * 30,
        })
        out = compute_smartphone_features(df)
        # Re-run with the spike removed; the first 10 days should be identical.
        df2 = df.copy()
        df2.loc[14, "screen_time_minutes"] = 200.0
        out2 = compute_smartphone_features(df2)
        early = out["screen_time_z"].iloc[:10].to_numpy()
        early2 = out2["screen_time_z"].iloc[:10].to_numpy()
        assert np.allclose(early, early2, atol=1e-8), \
            "early-window features changed when a future day was modified -> temporal leak"

    def test_stress_burden_is_causal(self):
        days = pd.date_range("2024-01-01", periods=30)
        base = pd.DataFrame({
            "participant_id": ["P01"] * 30,
            "date": [d.isoformat() for d in days],
            "sleep_duration": [7.5] * 30,
            "sleep_efficiency": [0.88] * 30,
            "hrv_rmssd": [55.0] * 30,
            "resting_hr": [62.0] * 30,
            "baseline_hrv": [55.0] * 30,
            "baseline_sleep_need": [7.5] * 30,
            "stress_score": [30.0] * 30,
            "survey_mood": [5.0] * 30,
            "survey_stress": [3.0] * 30,
        })
        spiked = base.copy()
        spiked.loc[20, "stress_score"] = 95.0   # late spike
        a = compute_wearable_features(base)
        b = compute_wearable_features(spiked)
        # First 10 days' stress_burden_7d should be identical under causal stats.
        a_early = a["stress_burden_7d"].iloc[:10].to_numpy()
        b_early = b["stress_burden_7d"].iloc[:10].to_numpy()
        assert np.allclose(a_early, b_early, equal_nan=True, atol=1e-8), \
            "stress_burden_7d leaked future stress into past windows"


class TestClusteredBootstrap:
    """Participant-clustered CIs should be wider than row-level CIs when
    within-participant data is highly correlated."""

    def test_clustered_ci_is_wider_with_within_person_correlation(self):
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(0)
        n_participants = 20
        windows_per_participant = 30

        groups = np.repeat(np.arange(n_participants), windows_per_participant)
        # Per-person noise creates within-person correlation: all of person i's
        # rows share the same offset. That's exactly the longitudinal setting
        # the cluster bootstrap is designed for.
        person_offset = rng.normal(0, 2.0, size=n_participants)
        # Label: half positive overall, but per-person.
        y_true_per_person = (rng.random(n_participants) > 0.5).astype(int)
        y_true = np.repeat(y_true_per_person, windows_per_participant)
        # Predictions are person-offset around true label with noise.
        y_prob = 1 / (1 + np.exp(
            -(0.5 * (2 * y_true - 1) + person_offset[groups] + 0.1 * rng.standard_normal(len(y_true)))
        ))

        _, lo_row, hi_row = bootstrap_ci(
            y_true, y_prob, roc_auc_score, n_resamples=400, seed=1,
        )
        _, lo_clu, hi_clu = bootstrap_ci(
            y_true, y_prob, roc_auc_score, n_resamples=400, seed=1, groups=groups,
        )
        row_width = hi_row - lo_row
        cluster_width = hi_clu - lo_clu
        # Cluster bootstrap should be meaningfully wider on this kind of data.
        assert cluster_width > row_width * 2.0, (
            f"cluster bootstrap CI width ({cluster_width:.3f}) was not "
            f"sufficiently wider than the naive row-level CI ({row_width:.3f})"
        )

    def test_clustered_rejects_mismatched_lengths(self):
        from sklearn.metrics import roc_auc_score
        y_true = np.array([0, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        wrong_groups = np.array([0, 0, 1])  # wrong length
        with pytest.raises(ValueError, match="groups length"):
            bootstrap_ci(y_true, y_prob, roc_auc_score, groups=wrong_groups)


class TestEncoderBoundsChecks:
    """The encoder must give a clear error in strict mode when given a
    participant index outside its embedding table, and a graceful mean-
    embedding fallback in permissive (inference) mode."""

    def test_unseen_participant_raises_in_strict_mode(self):
        torch = pytest.importorskip("torch")
        from lhfm.models.encoder import MultimodalLongitudinalEncoder
        enc = MultimodalLongitudinalEncoder(
            modality_dims={"wearable": 3}, d_model=16, n_heads=2, n_layers=1,
            max_seq_len=10, n_participants=5,
        )
        # strict by default
        modalities = {"wearable": torch.randn(1, 5, 3)}
        masks = {"wearable": torch.zeros(1, 5)}
        bad_idx = torch.tensor([99], dtype=torch.long)
        with pytest.raises(IndexError, match="participant_idx"):
            enc(modalities, masks=masks, participant_idx=bad_idx)

    def test_unseen_participant_falls_back_to_mean_in_permissive_mode(self):
        torch = pytest.importorskip("torch")
        from lhfm.models.encoder import MultimodalLongitudinalEncoder
        enc = MultimodalLongitudinalEncoder(
            modality_dims={"wearable": 3}, d_model=16, n_heads=2, n_layers=1,
            max_seq_len=10, n_participants=5,
        )
        enc.allow_unknown_participants()
        modalities = {"wearable": torch.randn(2, 5, 3)}
        masks = {"wearable": torch.zeros(2, 5)}
        # Index 0 (known) and 99 (unknown) -- second should be replaced by the mean.
        idx = torch.tensor([0, 99], dtype=torch.long)
        out = enc(modalities, masks=masks, participant_idx=idx)
        assert torch.isfinite(out["representation"]).all()
        # And the strict toggle should round-trip.
        enc.require_known_participants()
        with pytest.raises(IndexError):
            enc(modalities, masks=masks, participant_idx=idx)

    def test_long_sequence_raises_clearly(self):
        torch = pytest.importorskip("torch")
        from lhfm.models.encoder import MultimodalLongitudinalEncoder
        enc = MultimodalLongitudinalEncoder(
            modality_dims={"wearable": 3}, d_model=16, n_heads=2, n_layers=1,
            max_seq_len=8,
        )
        modalities = {"wearable": torch.randn(1, 20, 3)}   # T=20 > max=8
        masks = {"wearable": torch.zeros(1, 20)}
        with pytest.raises(ValueError, match="positional encoding"):
            enc(modalities, masks=masks)


class TestEncoderParameterCount:
    def test_count_parameters_breakdown_sums(self):
        pytest.importorskip("torch")
        from lhfm.models.encoder import MultimodalLongitudinalEncoder
        enc = MultimodalLongitudinalEncoder(
            modality_dims={"wearable": 4, "smartphone": 3},
            d_model=16, n_heads=2, n_layers=1, max_seq_len=10,
        )
        counts = enc.count_parameters(trainable_only=True)
        # Sum of the named parts plus the input norm should equal total.
        parts_sum = sum(v for k, v in counts.items() if k != "total")
        assert counts["total"] == parts_sum
        assert counts["total"] > 0
