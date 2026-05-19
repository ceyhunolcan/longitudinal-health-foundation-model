"""LifeSnaps adapter (Yfantidou et al., Sci. Data 2022).

n=71, 4 months, Fitbit Sense + SEMA app EMA. The lowest-friction real
dataset for LHFM: public on Kaggle (no DUA), or Zenodo (short request form).

Download:
    Kaggle:  kaggle.com/datasets/skywescar/lifesnaps-fitbit-dataset
    Zenodo:  zenodo.org/records/6832186

Expected raw_dir layout (Kaggle):
    daily_fitbit_sema_df_unprocessed.csv    one row per (participant, day)
    hourly_fitbit_sema_df_unprocessed.csv   finer-grained, currently unused
    sema_data.csv                           EMA mood/stress/wellbeing items
    surveys.csv                             STAI, BFI, PSS, demographics
    ...

What LifeSnaps gives us:
    Wearable     full Fitbit Sense - RMSSD, RHR, sleep, steps, stress score
    EMA          mood, stress, alertness on a 1-5 Likert (we rescale to 1-7)

What it doesn't give us (handled by setting the corresponding flag):
    Smartphone   Fitbit doesn't expose phone sensing. missing_phone_flag = 1.
    Climate      Open-Meteo enrichment based on participant timezone -
                 LifeSnaps releases tz, not coordinates. See
                 LIFESNAPS_TIMEZONE_COORDS for the city centroids we use.
    Chronotype   no MEQ in LifeSnaps. Defaults to 'intermediate'.

Outcomes. LifeSnaps ships STAI baselines and PHQ-style items in the
post-study survey, but those are participant-level, not day-level. The
adapter keeps LHFM's default day-level binarisation (thresholds on EMA
mood/stress/sleep efficiency). For day-level anxiety prediction,
subclass and override ``binarize_targets``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import AdapterError, BaseAdapter, register_adapter


log = logging.getLogger(__name__)


# Approximate centroid for each timezone LifeSnaps releases. Used only
# when latitude/longitude aren't otherwise available. The cohort spans
# Europe and North America according to the data card.
LIFESNAPS_TIMEZONE_COORDS = {
    "Europe/Athens":    (37.98, 23.73),
    "Europe/London":    (51.51, -0.13),
    "Europe/Berlin":    (52.52, 13.40),
    "Europe/Madrid":    (40.42, -3.70),
    "Europe/Stockholm": (59.33, 18.07),
    "America/New_York": (40.71, -74.00),
    "America/Chicago":  (41.88, -87.63),
    "America/Los_Angeles": (34.05, -118.24),
    "Asia/Kolkata":     (28.61, 77.21),
}


class LifeSnapsAdapter(BaseAdapter):
    NAME = "lifesnaps"
    REQUIRES_WEATHER_ENRICHMENT = True

    # LifeSnaps reports stress on Fitbit's 0-100 scale, which is fine,
    # but sleep efficiency is sometimes 0-100 in the raw CSV. We rescale
    # in load_raw; this is the safety belt.
    PLAUSIBILITY_OVERRIDES: dict[str, tuple[float, float]] = {
        # Fitbit RMSSD can be quite low in this device-validation cohort;
        # widen slightly to avoid noisy warnings during preflight.
        "hrv_rmssd": (3, 250),
    }

    # ------------------------------------------------------------------

    def load_raw(self) -> pd.DataFrame:
        raw = self.config.raw_dir
        daily_path = raw / "daily_fitbit_sema_df_unprocessed.csv"
        sema_path = raw / "sema_data.csv"
        survey_path = raw / "surveys.csv"

        if not daily_path.exists():
            raise AdapterError(
                f"Expected daily_fitbit_sema_df_unprocessed.csv in {raw}. "
                f"Download from Kaggle: kaggle datasets download "
                f"skywescar/lifesnaps-fitbit-dataset, unzip into {raw}."
            )

        log.info("[lifesnaps] reading %s", daily_path.name)
        daily = pd.read_csv(daily_path)

        # Map LifeSnaps columns -> LHFM schema. Most names need rewriting.
        # Reference: Yfantidou et al. 2022 supplementary table.
        col_map = {
            "id": "participant_id",
            "date": "date",
            # Fitbit physiology
            "rmssd": "hrv_rmssd",
            "resting_hr": "resting_hr",
            "steps": "daily_steps",
            "sleep_duration": "sleep_duration",
            "sleep_efficiency": "sleep_efficiency",
            "stress_score": "stress_score",
            # Demographics live in the daily CSV (RAIS/Zenodo layout)
            "age": "age",
            "gender": "sex",
        }
        present = {k: v for k, v in col_map.items() if k in daily.columns}
        if not present:
            raise AdapterError(
                f"None of the expected LifeSnaps columns found in {daily_path}. "
                f"Sample columns: {sorted(daily.columns)[:15]} ... "
                f"Did Kaggle change the column names?"
            )
        df = daily.rename(columns=present)[list(present.values())].copy()

        # LifeSnaps stores SEMA EMA as one-hot per emotion (HAPPY, SAD, TIRED,
        # ALERT, "TENSE/ANXIOUS", "RESTED/RELAXED", NEUTRAL). Collapse to a
        # single 1-7 valence/energy/stress score, NaN if no emotion reported.
        def _sema_col(name):
            return pd.to_numeric(daily.get(name, 0), errors="coerce").fillna(0)

        happy  = _sema_col("HAPPY");          sad    = _sema_col("SAD")
        tired  = _sema_col("TIRED");          alert  = _sema_col("ALERT")
        tense  = _sema_col("TENSE/ANXIOUS");  rested = _sema_col("RESTED/RELAXED")
        neutral = _sema_col("NEUTRAL")

        any_emotion = (happy + sad + tired + alert + tense + rested + neutral) > 0
        df["survey_mood"]   = (4 + 1.5 * (happy - sad)).where(any_emotion, np.nan).clip(1, 7).values
        df["survey_energy"] = (4 + 1.5 * (alert - tired)).where(any_emotion, np.nan).clip(1, 7).values
        df["survey_stress"] = (4 + 2.0 * tense - 1.0 * rested).where(any_emotion, np.nan).clip(1, 7).values

        # Normalise units --------------------------------------------------
        # Sleep duration in LifeSnaps is in minutes; LHFM wants hours.
        if "sleep_duration" in df:
            sd = pd.to_numeric(df["sleep_duration"], errors="coerce")
            # Heuristic: if median > 30 it's clearly minutes; otherwise it's
            # already in hours (some preprocessed versions on Kaggle differ).
            if sd.dropna().median() > 30:
                df["sleep_duration"] = sd / 60.0
            else:
                df["sleep_duration"] = sd

        # Sleep efficiency: LifeSnaps gives 0-100 percent; LHFM wants 0-1.
        if "sleep_efficiency" in df:
            se = pd.to_numeric(df["sleep_efficiency"], errors="coerce")
            if se.dropna().median() > 1.5:    # clearly on a 0-100 scale
                df["sleep_efficiency"] = se / 100.0


        # ("Very sad" .. "Very happy"). LHFM downstream code assumes 1-7.
        # Linear rescale 1..5 -> 1..7 so the binarisation thresholds in
        # binarize_targets still hit the right populations.
        for col in ("survey_mood", "survey_energy", "survey_stress"):
            if col in df:
                s = pd.to_numeric(df[col], errors="coerce")
                if s.dropna().max() <= 5.5:
                    df[col] = 1 + (s - 1) * (7 - 1) / (5 - 1)
                else:
                    df[col] = s

        # Coerce demographics to numeric. LifeSnaps stores age as a range
        # string like "<30", "30-40", "40-45"; we take the lower-bound integer.
        if "age" in df.columns:
            ages = df["age"].astype(str).str.extract(r"(\d+)", expand=False)
            df["age"] = pd.to_numeric(ages, errors="coerce")
        if "bmi" in df.columns:
            df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

        # Demographics are sourced from the daily CSV directly (above).
        # Normalise sex codes (LifeSnaps uses "MALE"/"FEMALE"/"NB"/"NA"/...).
        if "sex" in df.columns:
            df["sex"] = df["sex"].astype(str).str.upper().str[0].replace({"O": "X", "N": "X"})

        # Lat/lon from timezone (best we can do; LifeSnaps doesn't release
        # coordinates). Two-step: try the participant's timezone, else
        # fall back to the adapter's default site.
        if "timezone" in df:
            coords = df["timezone"].map(LIFESNAPS_TIMEZONE_COORDS)
            df["latitude"] = coords.apply(lambda c: c[0] if isinstance(c, tuple) else np.nan)
            df["longitude"] = coords.apply(lambda c: c[1] if isinstance(c, tuple) else np.nan)

        # Per-participant baselines: median of first 14 days (or fewer if
        # the timeline is short). This is what LHFM expects for the
        # encoder's static features.
        df = df.sort_values(["participant_id", "date"])
        baselines = (
            df.groupby("participant_id")
              .head(14)
              .groupby("participant_id")[["hrv_rmssd", "sleep_duration"]]
              .median()
              .rename(columns={"hrv_rmssd": "baseline_hrv",
                                "sleep_duration": "baseline_sleep_need"})
        )
        df = df.merge(baselines, on="participant_id", how="left")

        # Chronotype: LifeSnaps doesn't ship a direct chronotype assessment.
        # We mark it "intermediate" -- the most common category in any cohort,
        # documented as a known limitation in the adapter README.
        df["chronotype"] = "intermediate"

        # Smartphone columns are NOT in LifeSnaps: leave NaN and flag.
        for c in ("phone_unlock_count", "screen_time_minutes",
                  "mobility_radius_km", "location_entropy"):
            df[c] = np.nan
        df["missing_phone_flag"] = 1

        # Wearable & survey flags from row-level missingness.
        wear_cols = [c for c in
                     ("daily_steps", "sleep_duration", "sleep_efficiency",
                      "resting_hr", "hrv_rmssd", "stress_score")
                     if c in df.columns]
        df["missing_wearable_flag"] = (
            df[wear_cols].isna().any(axis=1).astype(int) if wear_cols else 1
        )
        surv_cols = [c for c in
                     ("survey_mood", "survey_energy", "survey_stress")
                     if c in df.columns]
        df["missing_survey_flag"] = (
            df[surv_cols].isna().any(axis=1).astype(int) if surv_cols else 1
        )

        log.info("[lifesnaps] built: %d rows, %d participants",
                 len(df), df["participant_id"].nunique())
        return df


register_adapter(LifeSnapsAdapter.NAME, LifeSnapsAdapter)
