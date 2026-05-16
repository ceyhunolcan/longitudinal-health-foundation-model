"""Tests for the fairness audit and climate-regime modules.

Pure numpy/pandas — no torch required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lhfm.utils.climate_regimes import (
    CLIMATE_REGIMES,
    define_climate_regime,
    regime_summary,
    split_train_eval_by_regime,
)
from lhfm.utils.fairness import (
    check_fairness_thresholds,
    fairness_report_to_csv,
    run_fairness_audit,
)


# ---------------------------------------------------------------------------
# Fairness audit
# ---------------------------------------------------------------------------


class TestRunFairnessAudit:
    def _make_balanced(self, n_per_group=200, seed=0):
        """Build predictions that are roughly equal-performance across groups."""
        rng = np.random.default_rng(seed)
        groups = np.repeat(["A", "B"], n_per_group)
        # 30% positive base rate in both groups, same predictor quality.
        y_true = np.concatenate([
            (rng.random(n_per_group) < 0.30).astype(int),
            (rng.random(n_per_group) < 0.30).astype(int),
        ])
        # Predictor strength roughly equal in both halves.
        signal = 1.0 * (2 * y_true - 1) + 0.3 * rng.standard_normal(2 * n_per_group)
        y_prob = 1 / (1 + np.exp(-signal))
        md = pd.DataFrame({"sex": groups, "age": rng.integers(20, 70, size=2 * n_per_group)})
        return y_true, y_prob, md

    def test_audit_returns_one_row_per_axis_level(self):
        yt, yp, md = self._make_balanced(seed=1)
        audit = run_fairness_audit(yt, yp, md, min_subgroup_n=20, bootstrap_resamples=100)
        axes_seen = {r["axis"] for r in audit["per_subgroup"]}
        assert "sex" in axes_seen
        assert "age_band" in axes_seen
        # Two levels in 'sex' → two rows under that axis.
        sex_rows = [r for r in audit["per_subgroup"] if r["axis"] == "sex"]
        assert len(sex_rows) == 2

    def test_balanced_data_has_small_eo_gap(self):
        yt, yp, md = self._make_balanced(seed=1)
        audit = run_fairness_audit(yt, yp, md, min_subgroup_n=20, bootstrap_resamples=200)
        if "sex" in audit["equalised_odds_gap"]:
            gap = audit["equalised_odds_gap"]["sex"]["equalised_odds_violation"]
            # Two random arms of the same DGP shouldn't drift by more than 0.20.
            assert gap < 0.25, f"unexpected gap on balanced data: {gap:.3f}"

    def test_imbalanced_data_flags_violation(self):
        """Construct a deliberately biased predictor and confirm the audit
        catches a large equalised-odds gap."""
        rng = np.random.default_rng(42)
        n = 300
        groups = np.repeat(["A", "B"], n)
        y = (rng.random(2 * n) < 0.3).astype(int)
        # Group A: good predictor. Group B: noise.
        signal = np.where(
            groups == "A",
            1.5 * (2 * y - 1) + 0.2 * rng.standard_normal(2 * n),
            0.1 * rng.standard_normal(2 * n),
        )
        yp = 1 / (1 + np.exp(-signal))
        md = pd.DataFrame({"sex": groups})
        audit = run_fairness_audit(y, yp, md, min_subgroup_n=20, bootstrap_resamples=200)
        ok, violations = check_fairness_thresholds(audit, max_auroc_gap=0.10, max_eo_violation=0.20)
        assert not ok, "expected violation on biased predictor, got none"
        assert any("sex" in v for v in violations)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="out of alignment"):
            run_fairness_audit(
                np.array([0, 1, 0]),
                np.array([0.1, 0.9, 0.2]),
                pd.DataFrame({"sex": ["A", "B"]}),
            )

    def test_small_subgroups_skipped(self):
        rng = np.random.default_rng(0)
        n_big = 200
        n_small = 5
        y = np.concatenate([
            (rng.random(n_big) < 0.3).astype(int),
            (rng.random(n_small) < 0.3).astype(int),
        ])
        p = rng.random(n_big + n_small)
        md = pd.DataFrame({"sex": ["A"] * n_big + ["B"] * n_small})
        audit = run_fairness_audit(y, p, md, min_subgroup_n=30, bootstrap_resamples=50)
        skipped_levels = {lvl for _, lvl, _ in audit["small_subgroups"]}
        assert "B" in skipped_levels

    def test_csv_export(self, tmp_path):
        yt, yp, md = self._make_balanced(seed=1)
        audit = run_fairness_audit(yt, yp, md, min_subgroup_n=20, bootstrap_resamples=100)
        out = tmp_path / "f.csv"
        fairness_report_to_csv(audit, out)
        assert out.exists()
        df = pd.read_csv(out)
        assert {"axis", "level", "auroc", "fpr", "fnr"}.issubset(df.columns)


# ---------------------------------------------------------------------------
# Climate regimes
# ---------------------------------------------------------------------------


class TestClimateRegimes:
    def test_define_regime_masks_are_mutually_exclusive_with_normal(self):
        df = pd.DataFrame({
            "heat_index": [25, 33, 25, 25, 35],
            "temperature_c": [20, 30, 2, 20, 30],
            "aqi": [50, 60, 60, 200, 80],
        })
        heat = define_climate_regime(df, "heat_wave")
        cold = define_climate_regime(df, "cold_snap")
        smoke = define_climate_regime(df, "smoke_episode")
        normal = define_climate_regime(df, "normal")
        # Each row falls into exactly one regime (or is heat_wave overlapping
        # with itself, which is fine). Normal must be disjoint from the others.
        assert not (normal & heat).any()
        assert not (normal & cold).any()
        assert not (normal & smoke).any()

    def test_define_regime_handles_nan(self):
        df = pd.DataFrame({
            "heat_index": [np.nan, 35.0],
            "temperature_c": [np.nan, 30.0],
            "aqi": [np.nan, 60.0],
        })
        # NaN should not be marked as a regime hit.
        for r in ("heat_wave", "cold_snap", "smoke_episode"):
            mask = define_climate_regime(df, r)
            assert not mask.iloc[0]

    def test_regime_summary_returns_expected_keys(self):
        df = pd.DataFrame({
            "participant_id": ["P01"] * 10,
            "heat_index": [25, 33, 25, 25, 35, 25, 25, 25, 25, 25],
            "temperature_c": [20, 30, 2, 20, 30, 20, 20, 20, 20, 20],
            "aqi": [50, 60, 60, 200, 80, 50, 50, 50, 50, 50],
        })
        summ = regime_summary(df)
        assert set(summ.index) == set(CLIMATE_REGIMES)
        assert "n_rows" in summ.columns

    def test_split_returns_disjoint_train_and_eval_when_no_overlap(self):
        df = pd.DataFrame({
            "heat_index": [25, 25, 33, 33],
            "temperature_c": [20, 20, 30, 30],
            "aqi": [50, 50, 60, 60],
        })
        train, evalp = split_train_eval_by_regime(df, train_regimes=("normal",), eval_regime="heat_wave")
        assert len(train) == 2
        assert len(evalp) == 2

    def test_unknown_regime_raises(self):
        df = pd.DataFrame({"heat_index": [25], "temperature_c": [20], "aqi": [50]})
        with pytest.raises(ValueError, match="unknown regime"):
            define_climate_regime(df, "monsoon")
