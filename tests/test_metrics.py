"""Tests for src/utils/metrics.py.

These deliberately don't depend on torch — they test the pure-numpy /
sklearn helpers so they always run.
"""

from __future__ import annotations

import numpy as np
import pytest

from lhfm.utils.metrics import (
    binary_classification_report,
    bootstrap_ci,
    expected_calibration_error,
    reliability_curve,
)


class TestBinaryClassificationReport:
    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.01, 0.02, 0.99, 0.95, 0.05, 0.90])
        rep = binary_classification_report(y_true, y_prob)
        assert rep["auroc"] == pytest.approx(1.0)
        assert rep["auprc"] == pytest.approx(1.0)
        assert rep["f1"] == pytest.approx(1.0)
        assert rep["brier"] < 0.01
        assert rep["n_pos"] == 3
        assert rep["n_total"] == 6

    def test_random_classifier_auroc_near_half(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=2000)
        y_prob = rng.random(2000)
        rep = binary_classification_report(y_true, y_prob)
        assert 0.45 <= rep["auroc"] <= 0.55

    def test_single_class_returns_nan_for_auroc(self):
        y_true = np.zeros(20, dtype=int)
        y_prob = np.linspace(0, 1, 20)
        rep = binary_classification_report(y_true, y_prob)
        assert np.isnan(rep["auroc"])
        assert np.isnan(rep["auprc"])
        # F1 and brier are still defined.
        assert "f1" in rep and "brier" in rep

    def test_brier_score_present(self):
        rep = binary_classification_report([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
        assert "brier" in rep
        assert rep["brier"] < 0.05


class TestExpectedCalibrationError:
    def test_perfect_calibration_zero_ece(self):
        # If predicted prob == empirical positive rate in every bin, ECE = 0.
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75])
        # bin [0.5, 0.6): 2 samples, 1 positive -> bin_acc=0.5, bin_conf=0.5
        # bin [0.7, 0.8): 4 samples, 3 positive -> bin_acc=0.75, bin_conf=0.75
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        assert ece == pytest.approx(0.0)

    def test_miscalibrated_classifier(self):
        # Always predicts 0.9 but only 30% are positive -> big ECE.
        y_true = np.array([0]*7 + [1]*3)
        y_prob = np.full(10, 0.9)
        ece = expected_calibration_error(y_true, y_prob)
        assert ece > 0.55  # |0.9 - 0.3| = 0.6

    def test_empty_input(self):
        # Should not crash, should return 0 (no bins populated).
        ece = expected_calibration_error(np.array([], dtype=int), np.array([]))
        assert ece == 0.0


class TestReliabilityCurve:
    def test_drops_empty_bins(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.05, 0.95, 0.05, 0.95])
        means, fracs, counts = reliability_curve(y_true, y_prob, n_bins=10)
        # Only 2 bins populated, the rest dropped.
        assert len(means) == 2
        assert len(fracs) == 2
        assert counts.sum() == 4


class TestBootstrapCI:
    def test_perfect_predictor_tight_ci(self):
        from sklearn.metrics import roc_auc_score
        y_true = np.array([0]*30 + [1]*30)
        y_prob = np.concatenate([np.zeros(30) + 0.1, np.zeros(30) + 0.9])
        point, lo, hi = bootstrap_ci(y_true, y_prob, roc_auc_score,
                                     n_resamples=200, seed=42)
        assert point == pytest.approx(1.0)
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(1.0)

    def test_returns_finite_ci_on_noisy_data(self):
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 200)
        # Mildly informative signal.
        y_prob = 0.5 + 0.2 * (2*y_true - 1) + 0.2 * rng.standard_normal(200)
        y_prob = np.clip(y_prob, 0.001, 0.999)
        point, lo, hi = bootstrap_ci(y_true, y_prob, roc_auc_score,
                                     n_resamples=200, seed=1)
        assert np.isfinite(point) and np.isfinite(lo) and np.isfinite(hi)
        assert lo <= point <= hi
        assert hi > lo  # The CI shouldn't collapse to a point.

    def test_single_class_returns_nan_point(self):
        from sklearn.metrics import roc_auc_score
        y_true = np.zeros(20, dtype=int)
        y_prob = np.random.random(20)
        point, lo, hi = bootstrap_ci(y_true, y_prob, roc_auc_score,
                                     n_resamples=50, seed=0)
        assert np.isnan(point)
