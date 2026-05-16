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
            "sleep_duration": "sleep_duration",        # we normalise units below
            "sleep_efficiency": "sleep_efficiency",    # ditto
            "stress_score": "stress_score",
            # EMA fields (folded into daily here; SEMA file has the
            # higher-resolution version)
            "mood": "survey_mood",
            "alert": "survey_energy",
            "stressed_lf": "survey_stress",
        }
        present = {k: v for k, v in col_map.items() if k in daily.columns}
        if not present:
            raise AdapterError(
                f"None of the expected LifeSnaps columns found in {daily_path}. "
                f"Sample columns: {sorted(daily.columns)[:15]} ... "
                f"Did Kaggle change the column names?"
            )
        df = daily.rename(columns=present)[list(present.values())]

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

        # EMA scales: LifeSnaps SEMA app uses a 1-5 Likert for mood
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

        # Demographics from surveys.csv -----------------------------------
        if survey_path.exists():
            log.info("[lifesnaps] joining demographics from %s", survey_path.name)
            sv = pd.read_csv(survey_path)
            keep = [c for c in ("id", "age", "gender", "country", "timezone") if c in sv.columns]
            sv = sv[keep].drop_duplicates("id")
            sv = sv.rename(columns={"id": "participant_id", "gender": "sex"})
            if "sex" in sv:
                sv["sex"] = sv["sex"].astype(str).str.upper().str[0].replace({"O": "X"})
            df = df.merge(sv, on="participant_id", how="left")
        else:
            log.warning("[lifesnaps] %s missing; demographics will be NaN", survey_path.name)

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
