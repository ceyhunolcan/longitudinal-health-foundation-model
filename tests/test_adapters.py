"""Tests for the data-source adapter machinery.

We test:
- The registry: registering, looking up, errors on unknown names.
- ``BaseAdapter._enforce_schema``: adds missing columns, normalises dtypes,
  reorders to required-cols-first.
- ``preflight_report``: returns the keys downstream tools depend on.
- ``LifeSnapsAdapter`` against synthetic LifeSnaps-shaped CSVs (units,
  EMA rescaling, demographics merge, missingness flags).
- ``GlobemAdapter`` against synthetic GLOBEM-shaped CSVs (multi-
  institute join, EMA forward-fill).
- ``weather._pm25_to_aqi``: a few breakpoint checks against the
  US-EPA reference values.

The two real-data adapters are tested on synthetic data shaped like the
real datasets. We don't hit Open-Meteo here -- the weather enricher is
unit-tested separately with mocks; the adapters always run with
``enrich_weather=False`` in tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lhfm.data.adapters import (
    AdapterConfig,
    AdapterError,
    BaseAdapter,
    get_adapter,
    list_adapters,
    preflight_report,
    register_adapter,
)
from lhfm.data.weather import _pm25_to_aqi


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_builtin_adapters_registered(self):
        names = list_adapters()
        for n in ("synthetic", "lifesnaps", "globem"):
            assert n in names, f"adapter {n} not auto-registered"

    def test_unknown_name_raises(self):
        with pytest.raises(AdapterError, match="unknown adapter"):
            get_adapter("not-a-real-adapter")

    def test_can_register_custom(self):
        class Dummy(BaseAdapter):
            NAME = "_test_dummy"
            def load_raw(self):
                return pd.DataFrame()

        register_adapter(Dummy.NAME, Dummy)
        assert get_adapter("_test_dummy") is Dummy


# ---------------------------------------------------------------------------
# BaseAdapter schema enforcement
# ---------------------------------------------------------------------------


class _BareAdapter(BaseAdapter):
    NAME = "_bare"
    REQUIRES_WEATHER_ENRICHMENT = False

    def load_raw(self):
        # Deliberately sparse so we can test the schema enforcer.
        return pd.DataFrame({
            "participant_id": [1, 1, 2, 2],     # int, must become str
            "date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
            "age": [25, 25, 30, 30],
            "sex": ["F", "F", "M", "M"],
            "sleep_duration": [7.0, 7.5, 8.0, 6.5],
            "hrv_rmssd": [55.0, 50.0, 60.0, 45.0],
            "survey_mood": [5, 4, 6, 5],
        })


class TestSchemaEnforcement:
    def test_missing_cols_become_nan(self, tmp_path):
        cfg = AdapterConfig(raw_dir=tmp_path, enrich_weather=False)
        df = _BareAdapter(cfg)._enforce_schema(_BareAdapter(cfg).load_raw())
        assert "screen_time_minutes" in df.columns
        assert df["screen_time_minutes"].isna().all()
        assert "heat_index" in df.columns
        assert df["heat_index"].isna().all()

    def test_missing_flags_default_to_zero(self, tmp_path):
        cfg = AdapterConfig(raw_dir=tmp_path, enrich_weather=False)
        df = _BareAdapter(cfg)._enforce_schema(_BareAdapter(cfg).load_raw())
        # Flag columns are added with 0 rather than NaN so downstream
        # logical ops don't surprise.
        assert (df["missing_wearable_flag"] == 0).all()
        assert (df["missing_phone_flag"] == 0).all()

    def test_participant_id_coerced_to_str(self, tmp_path):
        cfg = AdapterConfig(raw_dir=tmp_path, enrich_weather=False)
        df = _BareAdapter(cfg)._enforce_schema(_BareAdapter(cfg).load_raw())
        assert df["participant_id"].dtype == object
        assert all(isinstance(p, str) for p in df["participant_id"])

    def test_required_cols_appear_first(self, tmp_path):
        cfg = AdapterConfig(raw_dir=tmp_path, enrich_weather=False)
        raw = _BareAdapter(cfg).load_raw()
        raw["extra_metadata"] = "x"   # an adapter-specific extra column
        df = _BareAdapter(cfg)._enforce_schema(raw)
        # required cols should all come before any extras
        from lhfm.data.validation import REQUIRED_COLUMNS
        rc_positions = [df.columns.get_loc(c) for c in REQUIRED_COLUMNS]
        extras_positions = [df.columns.get_loc(c) for c in df.columns if c not in REQUIRED_COLUMNS]
        assert max(rc_positions) < min(extras_positions)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


class TestPreflight:
    def test_report_keys_for_full_cohort(self):
        days = pd.date_range("2024-01-01", periods=20)
        rows = []
        for pid in ("P01", "P02", "P03"):
            for d in days:
                rows.append({
                    "participant_id": pid,
                    "date": d.date(),
                    "age": 25 if pid == "P01" else 30,
                    "sex": "F" if pid == "P01" else "M",
                    "missing_wearable_flag": 0,
                    "missing_phone_flag": 0,
                    "missing_survey_flag": 1,
                    "target_low_mood": (d.day % 4 == 0),
                })
        df = pd.DataFrame(rows)
        rep = preflight_report(df, min_days=14)
        assert rep["n_participants"] == 3
        assert rep["n_participants_after_min_days_filter"] == 3
        assert rep["n_days_min"] == 20
        assert rep["sex_distribution"] == {"M": 40, "F": 20}
        assert "target_low_mood_positive_rate" in rep
        assert rep["frac_missing_phone_flag"] == 0.0
        assert rep["frac_missing_survey_flag"] == 1.0

    def test_min_days_filter_excludes_short_timelines(self):
        rows = []
        for pid, days_count in [("P_short", 5), ("P_long", 30)]:
            for d in pd.date_range("2024-01-01", periods=days_count):
                rows.append({
                    "participant_id": pid,
                    "date": d.date(),
                    "missing_wearable_flag": 0,
                    "missing_phone_flag": 0,
                    "missing_survey_flag": 0,
                })
        df = pd.DataFrame(rows)
        rep = preflight_report(df, min_days=14)
        assert rep["n_participants"] == 2
        assert rep["n_participants_after_min_days_filter"] == 1


# ---------------------------------------------------------------------------
# LifeSnaps adapter on synthetic LifeSnaps-shaped data
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_lifesnaps_dir(tmp_path):
    """Lay out CSVs in LifeSnaps's documented shape."""
    rng = np.random.default_rng(7)
    rows = []
    for u in range(5):
        for d in range(20):
            rows.append({
                "id": f"P{u:03d}",
                "date": f"2022-06-{d+1:02d}",
                "rmssd": float(rng.uniform(20, 80)),
                "resting_hr": float(rng.uniform(55, 80)),
                "steps": float(rng.uniform(2000, 12000)),
                "sleep_duration": float(rng.uniform(300, 540)),     # minutes
                "sleep_efficiency": float(rng.uniform(75, 95)),     # percent
                "stress_score": float(rng.uniform(30, 80)),
                "mood": int(rng.choice([1, 2, 3, 4, 5])),
                "alert": int(rng.choice([1, 2, 3, 4, 5])),
                "stressed_lf": int(rng.choice([1, 2, 3, 4, 5])),
            })
    pd.DataFrame(rows).to_csv(tmp_path / "daily_fitbit_sema_df_unprocessed.csv", index=False)

    sv = pd.DataFrame({
        "id": [f"P{u:03d}" for u in range(5)],
        "age": [22, 35, 41, 28, 60],
        "gender": ["M", "F", "F", "M", "F"],
        "country": ["US", "DE", "GR", "US", "UK"],
        "timezone": ["America/New_York", "Europe/Berlin", "Europe/Athens",
                     "America/Los_Angeles", "Europe/London"],
    })
    sv.to_csv(tmp_path / "surveys.csv", index=False)
    return tmp_path


class TestLifeSnapsAdapter:
    def test_unit_conversion_sleep_duration_min_to_h(self, fake_lifesnaps_dir):
        cfg = AdapterConfig(raw_dir=fake_lifesnaps_dir, enrich_weather=False)
        df = get_adapter("lifesnaps")(cfg).build()
        # All sleep durations should be in 5-9 h after conversion.
        sd = df["sleep_duration"].dropna()
        assert (sd >= 5.0).all() and (sd <= 9.0).all()

    def test_unit_conversion_sleep_efficiency_pct_to_decimal(self, fake_lifesnaps_dir):
        cfg = AdapterConfig(raw_dir=fake_lifesnaps_dir, enrich_weather=False)
        df = get_adapter("lifesnaps")(cfg).build()
        se = df["sleep_efficiency"].dropna()
        assert (se >= 0.55).all() and (se <= 1.0).all()

    def test_ema_rescaled_to_1_7(self, fake_lifesnaps_dir):
        cfg = AdapterConfig(raw_dir=fake_lifesnaps_dir, enrich_weather=False)
        df = get_adapter("lifesnaps")(cfg).build()
        for col in ("survey_mood", "survey_energy", "survey_stress"):
            vals = df[col].dropna()
            assert vals.min() >= 1.0
            assert vals.max() <= 7.0

    def test_phone_modality_always_missing(self, fake_lifesnaps_dir):
        """LifeSnaps has no smartphone passive sensing → phone flag = 1."""
        cfg = AdapterConfig(raw_dir=fake_lifesnaps_dir, enrich_weather=False)
        df = get_adapter("lifesnaps")(cfg).build()
        assert (df["missing_phone_flag"] == 1).all()
        for c in ("screen_time_minutes", "mobility_radius_km", "location_entropy"):
            assert df[c].isna().all()

    def test_demographics_merged(self, fake_lifesnaps_dir):
        cfg = AdapterConfig(raw_dir=fake_lifesnaps_dir, enrich_weather=False)
        df = get_adapter("lifesnaps")(cfg).build()
        # Sex values come from the surveys.csv we wrote.
        # First-letter upper-case + 'O' → 'X' mapping applied.
        sexes = df.drop_duplicates("participant_id")["sex"].tolist()
        assert set(sexes) <= {"M", "F", "X"}

    def test_missing_raw_dir_raises_helpfully(self, tmp_path):
        cfg = AdapterConfig(raw_dir=tmp_path / "nope", enrich_weather=False)
        with pytest.raises(AdapterError, match="raw_dir does not exist"):
            get_adapter("lifesnaps")(cfg)


# ---------------------------------------------------------------------------
# GLOBEM adapter on synthetic GLOBEM-shaped data
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_globem_dir(tmp_path):
    """Two institute-year folders with a minimal feature + survey set."""
    rng = np.random.default_rng(11)
    base = tmp_path
    for inst in ("INS-W_1", "INS-W_2"):
        feat = base / inst / "FeatureData"
        surv = base / inst / "SurveyData"
        feat.mkdir(parents=True, exist_ok=True)
        surv.mkdir(parents=True, exist_ok=True)

        pids = [f"{inst}_P{u:03d}" for u in range(4)]
        dates = pd.date_range("2020-09-01", periods=20).strftime("%Y-%m-%d").tolist()
        n = len(pids) * len(dates)
        rows = [{"pid": p, "date": d} for p in pids for d in dates]

        steps = pd.DataFrame(rows)
        steps["f_steps:fitbit_steps_summary_summed:allday"] = rng.uniform(2000, 10000, size=n)
        steps.to_csv(feat / "steps.csv", index=False)

        sleep = pd.DataFrame(rows)
        sleep["f_slp:fitbit_sleep_summary_minutesasleep:allday"] = rng.uniform(300, 540, size=n)
        sleep["f_slp:fitbit_sleep_summary_efficiency:allday"] = rng.uniform(75, 95, size=n)
        sleep["f_slp:fitbit_sleep_summary_restingheartrate:allday"] = rng.uniform(55, 80, size=n)
        sleep.to_csv(feat / "sleep.csv", index=False)

        screen = pd.DataFrame(rows)
        # In raw GLOBEM this is in seconds. Use values clearly > 5000 so
        # the adapter's heuristic kicks in.
        screen["f_screen:phone_screen_rapids_sumdurationunlock:allday"] = rng.uniform(7200, 36000, size=n)
        screen["f_screen:phone_screen_rapids_countepisodeunlock:allday"] = rng.uniform(40, 200, size=n)
        screen.to_csv(feat / "screen.csv", index=False)

        loc = pd.DataFrame(rows)
        loc["f_loc:phone_locations_doryab_locationentropy:allday"] = rng.uniform(0.3, 2.0, size=n)
        loc["f_loc:phone_locations_doryab_radiusgyration:allday"] = rng.uniform(1.0, 20.0, size=n)
        loc.to_csv(feat / "location.csv", index=False)

        demo = pd.DataFrame({
            "pid": pids,
            "age": [20, 22, 19, 21],
            "gender": ["F", "M", "F", "M"],
        })
        demo.to_csv(surv / "pre_survey.csv", index=False)

        dep = pd.DataFrame({
            "pid": pids,
            "BDI_total": [5, 18, 7, 22],   # binarises to [0, 1, 0, 1] at threshold 10
        })
        dep.to_csv(surv / "dep_endterm.csv", index=False)

    return base


class TestGlobemAdapter:
    def test_loads_both_institutes(self, fake_globem_dir):
        cfg = AdapterConfig(raw_dir=fake_globem_dir, enrich_weather=False)
        df = get_adapter("globem")(cfg).build()
        # 2 institutes × 4 participants = 8 participants, × 20 days = 160 rows
        assert df["participant_id"].nunique() == 8
        assert len(df) == 160
        assert df["institute_year"].nunique() == 2

    def test_hrv_is_explicitly_nan(self, fake_globem_dir):
        cfg = AdapterConfig(raw_dir=fake_globem_dir, enrich_weather=False)
        df = get_adapter("globem")(cfg).build()
        assert df["hrv_rmssd"].isna().all(), "GLOBEM Fitbit Charge has no RMSSD"

    def test_screen_time_seconds_converted_to_minutes(self, fake_globem_dir):
        cfg = AdapterConfig(raw_dir=fake_globem_dir, enrich_weather=False)
        df = get_adapter("globem")(cfg).build()
        # Raw values were 7200-36000 seconds = 120-600 minutes.
        vals = df["screen_time_minutes"].dropna()
        assert (vals >= 100).all()
        assert (vals <= 700).all()

    def test_institute_coords_set(self, fake_globem_dir):
        cfg = AdapterConfig(raw_dir=fake_globem_dir, enrich_weather=False)
        df = get_adapter("globem")(cfg).build()
        # INS-W → Seattle (47.66, -122.31)
        assert (df["latitude"].round(2) == 47.66).all()
        assert (df["longitude"].round(2) == -122.31).all()


# ---------------------------------------------------------------------------
# Weather helper: PM2.5 → AQI
# ---------------------------------------------------------------------------


class TestPM25ToAQI:
    def test_known_breakpoints(self):
        # US-EPA reference values.
        assert _pm25_to_aqi(0.0) == pytest.approx(0.0)
        assert _pm25_to_aqi(12.0) == pytest.approx(50.0, abs=0.5)
        # 35.4 → 100 (top of "Moderate" band)
        assert _pm25_to_aqi(35.4) == pytest.approx(100.0, abs=0.5)
        # 150.4 → 200 (top of "Unhealthy" band)
        assert _pm25_to_aqi(150.4) == pytest.approx(200.0, abs=0.5)

    def test_handles_invalid_input(self):
        import math
        assert math.isnan(_pm25_to_aqi(-1.0))
        assert math.isnan(_pm25_to_aqi(float("nan")))
        assert math.isnan(_pm25_to_aqi(None))

    def test_caps_at_500(self):
        assert _pm25_to_aqi(10_000.0) == 500.0
