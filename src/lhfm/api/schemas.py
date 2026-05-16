"""Pydantic request/response schemas for the FastAPI service.

The API accepts a 14-day window of per-day records and returns a dict of
binary risk probabilities plus a short, human-readable explanation. We
intentionally do NOT return any clinical recommendation -- this is a
research prototype.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class DailyRecord(BaseModel):
    """One day of multimodal observations for a single participant.

    Any numeric field may be ``None`` to indicate a missing measurement; the
    server will impute as required. The ``missing_*_flag`` fields are
    *optional*: if you don't pass them, they'll be inferred from which
    numeric fields are None.
    """

    date: date

    # Wearable
    daily_steps: Optional[float] = None
    sleep_duration: Optional[float] = Field(None, description="hours")
    sleep_efficiency: Optional[float] = None
    resting_hr: Optional[float] = None
    hrv_rmssd: Optional[float] = Field(None, description="ms")
    stress_score: Optional[float] = None

    # Smartphone
    phone_unlock_count: Optional[float] = None
    screen_time_minutes: Optional[float] = None
    mobility_radius_km: Optional[float] = None
    location_entropy: Optional[float] = None

    # EMA / self-report
    survey_mood: Optional[float] = Field(None, ge=1.0, le=7.0)
    survey_energy: Optional[float] = Field(None, ge=1.0, le=7.0)
    survey_stress: Optional[float] = Field(None, ge=1.0, le=7.0)

    # Environmental
    temperature_c: Optional[float] = None
    humidity: Optional[float] = Field(None, ge=0.0, le=100.0)
    aqi: Optional[float] = Field(None, ge=0.0, le=500.0)
    heat_index: Optional[float] = None

    # Optional explicit missingness flags (0/1)
    missing_wearable_flag: Optional[int] = None
    missing_phone_flag: Optional[int] = None
    missing_survey_flag: Optional[int] = None


class ParticipantProfile(BaseModel):
    """Slowly-varying personal context used to anchor the baseline embedding.

    ``sex`` accepts F, M, or X (unknown / non-binary / prefer not to say).
    The model treats only ``sex_male`` (0/1) as a feature; anything that's
    not "M" maps to 0. If you need a richer encoding, replace the
    ``compute_baseline_features`` function.
    """

    participant_id: str
    age: int = Field(..., ge=10, le=110)
    sex: str = Field(..., pattern="^[FfMmXx]$")
    chronotype: str = Field(..., pattern="^(morning|intermediate|evening)$")
    baseline_sleep_need: float = Field(..., gt=3.0, lt=12.0)
    baseline_hrv: float = Field(..., gt=5.0, lt=250.0)

    @field_validator("sex")
    @classmethod
    def _normalize_sex(cls, v: str) -> str:
        return v.upper()


class PredictRequest(BaseModel):
    """Payload for the /predict endpoint."""

    profile: ParticipantProfile
    window: list[DailyRecord] = Field(..., min_length=7, max_length=30)

    @field_validator("window")
    @classmethod
    def _check_dates_unique_and_ordered(cls, v):
        dates = [r.date for r in v]
        if len(set(dates)) != len(dates):
            raise ValueError("window contains duplicate dates")
        if dates != sorted(dates):
            raise ValueError("window dates must be in ascending order")
        return v


class RiskScore(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0)
    label: str        # "elevated" / "moderate" / "low"


class PredictResponse(BaseModel):
    participant_id: str
    window_end_date: date
    low_mood_risk: RiskScore
    stress_risk: RiskScore
    sleep_disruption_risk: RiskScore
    climate_vulnerability_risk: RiskScore
    explanation: list[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_loaded: bool
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    encoder_d_model: Optional[int] = None
