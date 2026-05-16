"""GLOBEM adapter (Xu et al., NeurIPS 2022).

n=497 across 4 institute-years (UW × 2 years, CMU × 2 years), smartphone
passive sensing + Fitbit. The strongest public match to LHFM's design
and the dataset with the most published baselines, so this is the one
that anchors the paper.

Download (PhysioNet, credentialed access):
    https://physionet.org/content/globem/1.1/

Expected raw_dir layout:
    INS-W_1/                       one folder per (institute, year)
        FeatureData/
            location.csv           GPS-derived passive sensing
            screen.csv             phone unlock + screen-on time
            steps.csv              accelerometer-derived
            sleep.csv              Fitbit sleep summary
        SurveyData/
            pre_survey.csv         demographics
            dep_endterm.csv        BDI-II / PHQ-9 end-of-term scores
    INS-W_2/ INS-W_3/ INS-W_4/

What GLOBEM gives us:
    Smartphone   rich - location (radius, entropy), screen, unlocks, calls
    Wearable     partial - Fitbit Charge has sleep + steps + RHR but no
                 RMSSD. ``hrv_rmssd`` and ``stress_score`` stay NaN.

What it doesn't, and how the adapter handles it:
    EMA          GLOBEM is *weekly*, not daily. We forward-fill the weekly
                 EMA value into the 7 days that follow so the daily-window
                 architecture still works. Document this in your methods.
    Climate      Open-Meteo enrichment using the institute coordinates
                 (UW Seattle, CMU Pittsburgh - see GLOBEM_INSTITUTE_COORDS).

Outcomes. The published GLOBEM target is end-of-term depression
(BDI-II >= 10 at study end). The adapter exposes this as
``target_depressed``, alongside LHFM's day-level default targets.

Cross-site generalization. The canonical GLOBEM eval is training on
three institute-years and testing on the fourth. The adapter emits an
``institute_year`` column so the climate-holdout script (with the
forthcoming ``--holdout-column`` flag) and the fairness audit can both
slice on it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import AdapterError, BaseAdapter, register_adapter


log = logging.getLogger(__name__)


# Institute coordinates from Xu et al. 2022, §2.1.
GLOBEM_INSTITUTE_COORDS = {
    "INS-W": (47.66, -122.31),    # University of Washington, Seattle
    "INS-D": (40.44, -79.94),     # Carnegie Mellon, Pittsburgh
}


class GlobemAdapter(BaseAdapter):
    NAME = "globem"
    REQUIRES_WEATHER_ENRICHMENT = True

    PLAUSIBILITY_OVERRIDES: dict[str, tuple[float, float]] = {
        # GLOBEM Fitbit cohort has wider RHR distributions than typical
        # consumer cohorts (athletes + sedentary college students mixed).
        "resting_hr": (35, 130),
        # Screen-time can go quite high in undergrad cohorts.
        "screen_time_minutes": (0, 1200),
    }

    def load_raw(self) -> pd.DataFrame:
        raw = self.config.raw_dir
        institute_dirs = sorted([
            p for p in raw.iterdir()
            if p.is_dir() and (p.name.startswith("INS-") or p.name.startswith("INSW"))
        ])
        if not institute_dirs:
            raise AdapterError(
                f"No institute folders found under {raw}. Expected layout: "
                f"{raw}/INS-W_1, {raw}/INS-W_2, ... -- see PhysioNet "
                f"https://physionet.org/content/globem/1.1/."
            )

        frames: list[pd.DataFrame] = []
        for inst_dir in institute_dirs:
            log.info("[globem] loading %s", inst_dir.name)
            try:
                df = self._load_one_institute_year(inst_dir)
                df["institute_year"] = inst_dir.name
                frames.append(df)
            except AdapterError as exc:
                log.warning("[globem] skipping %s: %s", inst_dir.name, exc)

        if not frames:
            raise AdapterError("No institute-year folders loaded successfully")

        df = pd.concat(frames, ignore_index=True)
        df = self._derive_baselines_and_chronotype(df)

        # Attach institute coordinates so the weather enricher can find them.
        # We strip the trailing "_<year>" from the institute_year string.
        df["_inst_prefix"] = df["institute_year"].str.split("_").str[0]
        df["latitude"] = df["_inst_prefix"].map(
            lambda k: GLOBEM_INSTITUTE_COORDS.get(k, (None, None))[0]
        )
        df["longitude"] = df["_inst_prefix"].map(
            lambda k: GLOBEM_INSTITUTE_COORDS.get(k, (None, None))[1]
        )
        df = df.drop(columns=["_inst_prefix"])

        # HRV not in GLOBEM Fitbit Charge cohorts -- explicit NaN.
        df["hrv_rmssd"] = np.nan
        df["stress_score"] = np.nan

        # Wearable / phone / survey flags
        wear_cols = ["daily_steps", "sleep_duration", "sleep_efficiency",
                     "resting_hr"]
        wear_cols = [c for c in wear_cols if c in df.columns]
        df["missing_wearable_flag"] = (
            df[wear_cols].isna().any(axis=1).astype(int) if wear_cols else 1
        )
        phone_cols = ["phone_unlock_count", "screen_time_minutes",
                      "mobility_radius_km", "location_entropy"]
        phone_cols = [c for c in phone_cols if c in df.columns]
        df["missing_phone_flag"] = (
            df[phone_cols].isna().any(axis=1).astype(int) if phone_cols else 1
        )
        surv_cols = ["survey_mood", "survey_energy", "survey_stress"]
        surv_cols = [c for c in surv_cols if c in df.columns]
        df["missing_survey_flag"] = (
            df[surv_cols].isna().any(axis=1).astype(int) if surv_cols else 1
        )

        log.info("[globem] total: %d rows, %d participants across %d inst-years",
                 len(df), df["participant_id"].nunique(),
                 df["institute_year"].nunique())
        return df

    # -- internals -------------------------------------------------------

    def _load_one_institute_year(self, inst_dir: Path) -> pd.DataFrame:
        feat_dir = inst_dir / "FeatureData"
        surv_dir = inst_dir / "SurveyData"
        if not feat_dir.exists():
            raise AdapterError(f"missing FeatureData/ in {inst_dir}")

        # GLOBEM uses pid + date as the join key.
        pieces: list[pd.DataFrame] = []
        for fname, cols_map in [
            ("location.csv", {
                "f_loc:phone_locations_doryab_locationentropy:allday": "location_entropy",
                "f_loc:phone_locations_doryab_radiusgyration:allday": "mobility_radius_km",
            }),
            ("screen.csv", {
                "f_screen:phone_screen_rapids_sumdurationunlock:allday": "screen_time_minutes",
                "f_screen:phone_screen_rapids_countepisodeunlock:allday": "phone_unlock_count",
            }),
            ("steps.csv", {
                "f_steps:fitbit_steps_summary_summed:allday": "daily_steps",
            }),
            ("sleep.csv", {
                "f_slp:fitbit_sleep_summary_minutesasleep:allday": "sleep_duration_min",
                "f_slp:fitbit_sleep_summary_efficiency:allday": "sleep_efficiency_pct",
                "f_slp:fitbit_sleep_summary_restingheartrate:allday": "resting_hr",
            }),
        ]:
            fpath = feat_dir / fname
            if not fpath.exists():
                log.info("[globem] %s/%s missing; relevant cols will be NaN",
                         inst_dir.name, fname)
                continue
            f = pd.read_csv(fpath)
            base_cols = [c for c in ("pid", "date") if c in f.columns]
            keep = [c for c in cols_map if c in f.columns]
            if not keep:
                log.info("[globem] no matching cols in %s/%s -- column "
                         "names may have changed in PhysioNet release",
                         inst_dir.name, fname)
                continue
            f = f[base_cols + keep].rename(columns=cols_map)
            pieces.append(f)

        if not pieces:
            raise AdapterError(f"no feature files loaded from {feat_dir}")

        # Outer-join all feature pieces on (pid, date) so a day with only
        # phone but no Fitbit still survives.
        df = pieces[0]
        for p in pieces[1:]:
            df = df.merge(p, on=["pid", "date"], how="outer")

        # Unit normalisation
        if "sleep_duration_min" in df:
            df["sleep_duration"] = df["sleep_duration_min"] / 60.0
            df = df.drop(columns=["sleep_duration_min"])
        if "sleep_efficiency_pct" in df:
            df["sleep_efficiency"] = df["sleep_efficiency_pct"] / 100.0
            df = df.drop(columns=["sleep_efficiency_pct"])
        # screen_time in seconds in GLOBEM raw -- convert to minutes.
        if "screen_time_minutes" in df:
            s = pd.to_numeric(df["screen_time_minutes"], errors="coerce")
            if s.dropna().median() > 5000:    # clearly seconds
                df["screen_time_minutes"] = s / 60.0

        df = df.rename(columns={"pid": "participant_id"})

        # Demographics + outcomes from SurveyData/.
        df = self._attach_surveys(df, surv_dir)
        return df

    def _attach_surveys(self, df: pd.DataFrame, surv_dir: Path) -> pd.DataFrame:
        """Merge weekly EMA (forward-filled into daily rows) and end-term BDI."""
        # Demographics: pre-study survey with age / gender.
        for fname in ("pre_survey.csv", "demographics.csv"):
            fpath = surv_dir / fname
            if not fpath.exists():
                continue
            d = pd.read_csv(fpath)
            d = d.rename(columns={"pid": "participant_id",
                                  "gender": "sex"})
            keep = [c for c in ("participant_id", "age", "sex", "race", "ses") if c in d.columns]
            d = d[keep].drop_duplicates("participant_id")
            if "sex" in d:
                d["sex"] = d["sex"].astype(str).str.upper().str[0].replace({"O": "X"})
            df = df.merge(d, on="participant_id", how="left")
            break

        # Weekly EMA — forward-filled into daily rows.
        for fname in ("ema_weekly.csv", "weekly_survey.csv", "pre_survey.csv"):
            fpath = surv_dir / fname
            if not fpath.exists():
                continue
            w = pd.read_csv(fpath)
            cols_map = {
                "pid": "participant_id",
                "mood": "survey_mood",
                "stress": "survey_stress",
                "energy": "survey_energy",
            }
            present = {k: v for k, v in cols_map.items() if k in w.columns}
            if "date" not in w.columns or len(present) <= 1:
                continue
            w = w.rename(columns=present)[["participant_id", "date"]
                                          + [v for v in present.values() if v != "participant_id"]]
            # Forward-fill EMA within participant — converts weekly into daily.
            w["date"] = pd.to_datetime(w["date"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.merge(w, on=["participant_id", "date"], how="left")
            ema_cols = [c for c in ("survey_mood", "survey_stress", "survey_energy")
                        if c in df.columns]
            df = df.sort_values(["participant_id", "date"])
            df[ema_cols] = df.groupby("participant_id")[ema_cols].ffill(limit=7)
            break

        # End-of-term BDI / PHQ outcomes — emit as a per-participant binary.
        for fname in ("dep_endterm.csv", "dep_weekly.csv", "outcomes.csv"):
            fpath = surv_dir / fname
            if not fpath.exists():
                continue
            o = pd.read_csv(fpath)
            score_cols = [c for c in o.columns if "BDI" in c.upper() or "PHQ" in c.upper()]
            if not score_cols:
                continue
            o = o.rename(columns={"pid": "participant_id"})
            o["depression_score"] = o[score_cols].max(axis=1)
            # Threshold = 10 is the "mild or worse" cutoff for BDI-II and
            # the standard "elevated symptoms" cutoff for PHQ-9. The
            # original GLOBEM paper uses 14 (moderate-or-worse on BDI-II);
            # 10 is a defensible alternative used in most screening
            # contexts. Document whichever you pick in the methods.
            o["target_depressed"] = (o["depression_score"] >= 10).astype(float)
            df = df.merge(o[["participant_id", "depression_score",
                             "target_depressed"]].drop_duplicates("participant_id"),
                          on="participant_id", how="left")
            break

        return df

    def _derive_baselines_and_chronotype(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-participant baseline HRV and sleep-need from the first
        14 days, mirroring the synthetic generator's convention."""
        df = df.sort_values(["participant_id", "date"])
        # GLOBEM has no HRV; baseline_hrv stays NaN. Sleep-need from
        # first-14-days median.
        baselines = (
            df.groupby("participant_id")
              .head(14)
              .groupby("participant_id")["sleep_duration"]
              .median()
              .to_frame("baseline_sleep_need")
        )
        df = df.merge(baselines, on="participant_id", how="left")
        df["baseline_hrv"] = np.nan
        df["chronotype"] = "intermediate"     # no MEQ in GLOBEM
        return df

    def binarize_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """In addition to the synthetic targets, keep target_depressed from
        the survey join if present."""
        from lhfm.data.preprocessing import binarize_targets as default_bin
        df = default_bin(df)
        # ``target_depressed`` is already attached per-participant; replicate
        # to per-row (constant within participant). The downstream loss
        # already masks NaN, so participants without an outcome score are
        # transparently excluded.
        return df


register_adapter(GlobemAdapter.NAME, GlobemAdapter)
