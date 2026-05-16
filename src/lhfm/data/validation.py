"""Sanity checks for the synthetic (and eventually real) longitudinal dataframe.

We validate aggressively because every downstream module silently assumes
things like "sleep duration is in hours" or "HRV is in milliseconds". When
those assumptions break it manifests as a model that mysteriously won't
converge. Better to fail loudly here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "participant_id", "date", "age", "sex",
    "baseline_sleep_need", "baseline_hrv", "chronotype",
    "daily_steps", "sleep_duration", "sleep_efficiency",
    "resting_hr", "hrv_rmssd", "stress_score",
    "phone_unlock_count", "screen_time_minutes",
    "mobility_radius_km", "location_entropy",
    "survey_mood", "survey_energy", "survey_stress",
    "temperature_c", "humidity", "aqi", "heat_index",
    "missing_wearable_flag", "missing_phone_flag", "missing_survey_flag",
]


# Plausibility ranges. NaN values are allowed; only non-NaN values are checked.
PLAUSIBILITY_RANGES: dict[str, tuple[float, float]] = {
    "age": (10, 100),
    "sleep_duration": (0, 14),
    "sleep_efficiency": (0.3, 1.0),
    "resting_hr": (30, 130),
    "hrv_rmssd": (5, 250),
    "stress_score": (0, 100),
    "phone_unlock_count": (0, 600),
    "screen_time_minutes": (0, 1000),
    "mobility_radius_km": (0, 500),
    "location_entropy": (0, 5),
    "survey_mood": (1, 7),
    "survey_energy": (1, 7),
    "survey_stress": (1, 7),
    "temperature_c": (-30, 55),
    "humidity": (0, 100),
    "aqi": (0, 500),
    "heat_index": (-30, 65),
    "daily_steps": (0, 60000),
}


@dataclass
class ValidationReport:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValueError("Dataset failed validation:\n - " + "\n - ".join(self.errors))


def validate_synthetic_dataframe(df: pd.DataFrame, strict: bool = False) -> ValidationReport:
    """Run a battery of checks against the long-form dataframe.

    Set ``strict=True`` to also surface things that are merely odd (e.g.
    fewer than 14 days per participant) as errors instead of warnings.
    """
    report = ValidationReport()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report.ok = False
        report.errors.append(f"missing required columns: {missing}")
        return report  # nothing else makes sense without the columns

    # Type / structural checks ------------------------------------------------
    if df["participant_id"].isna().any():
        report.ok = False
        report.errors.append("participant_id contains NaN")

    try:
        _ = pd.to_datetime(df["date"])
    except Exception as exc:
        report.ok = False
        report.errors.append(f"date column unparseable: {exc}")

    # Plausibility ranges -----------------------------------------------------
    for col, (lo, hi) in PLAUSIBILITY_RANGES.items():
        s = df[col].dropna()
        if s.empty:
            report.warnings.append(f"{col} is entirely missing")
            continue
        bad = ((s < lo) | (s > hi)).sum()
        if bad > 0:
            msg = f"{col}: {bad} values outside plausible range [{lo}, {hi}]"
            if strict:
                report.ok = False
                report.errors.append(msg)
            else:
                report.warnings.append(msg)

    # Per-participant timeline checks ----------------------------------------
    per_pid = df.groupby("participant_id")["date"].nunique()
    if per_pid.min() < 14:
        msg = f"some participants have fewer than 14 days of data (min={per_pid.min()})"
        if strict:
            report.ok = False; report.errors.append(msg)
        else:
            report.warnings.append(msg)

    # Quick descriptive summary for logs / manifests
    report.summary = {
        "n_rows": int(len(df)),
        "n_participants": int(df["participant_id"].nunique()),
        "n_days_min": int(per_pid.min()) if not per_pid.empty else 0,
        "n_days_max": int(per_pid.max()) if not per_pid.empty else 0,
        "frac_missing_wearable": float(df["missing_wearable_flag"].mean()),
        "frac_missing_phone": float(df["missing_phone_flag"].mean()),
        "frac_missing_survey": float(df["missing_survey_flag"].mean()),
    }
    return report
