"""FastAPI service that wraps the trained foundation model.

The service is designed to degrade gracefully:

- If no trained checkpoint is present we still respond to /health and /
  with informative messages.
- /predict in "no model" mode falls back to a simple rule-based stand-in so
  the dashboard and integration tests have something to talk to. The
  response payload is identical so clients don't need a code path for
  "untrained".

This is *not* a production-grade inference server. There's no auth, no
rate limiting, no batching. Don't deploy it externally.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from lhfm.api.schemas import (
    DailyRecord,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    RiskScore,
)
from lhfm.features import build_full_feature_table
from lhfm.utils.logging import get_logger


log = get_logger(__name__)

DISCLAIMER = (
    "Research prototype. Not a medical device. Outputs are not diagnostic "
    "and must not be used to make clinical decisions."
)

# Default window length the model expects. This is overridden by the value
# saved alongside the checkpoint (see train_model.py).
DEFAULT_WINDOW_DAYS = 14


# ---------------------------------------------------------------------------
# Model loading. We resolve once at startup so /predict stays fast.
# ---------------------------------------------------------------------------


class _ModelHolder:
    """Holds the model and meta. Falls back gracefully when torch is missing."""

    def __init__(self):
        self.model = None
        self.feature_columns: list[str] = []
        self.modality_slices: dict[str, tuple[int, int]] = {}
        self.task_names: list[str] = []
        self.d_model: Optional[int] = None
        self.window_days: int = DEFAULT_WINDOW_DAYS

    def try_load(self, checkpoint_path: Path | str) -> bool:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            log.info("no checkpoint at %s; running in fallback mode", ckpt_path)
            return False

        try:
            from lhfm.checkpoints import load_downstream
        except ImportError as exc:
            # The loader lazy-imports torch, so this path covers "torch
            # not installed" cleanly.
            log.warning("torch not importable -- fallback mode (%s)", exc)
            return False

        try:
            loaded = load_downstream(
                ckpt_path,
                map_location="cpu",
                allow_unknown_participants=True,
            )
        except FileNotFoundError as exc:
            log.warning("checkpoint sidecar missing -- fallback mode: %s", exc)
            return False
        except Exception as exc:
            log.exception("failed to load checkpoint: %s", exc)
            return False

        meta = loaded.meta
        self.model = loaded.model
        self.feature_columns = meta.get("feature_columns", [])
        self.modality_slices = loaded.modality_slices
        self.task_names = loaded.task_names
        self.d_model = int(meta.get("d_model", 128))
        self.window_days = loaded.window_days
        log.info(
            "loaded model from %s (d_model=%d, tasks=%s, window=%d)",
            ckpt_path, d_model, task_names, window_days,
        )
        return True


MODEL = _ModelHolder()

# Path is configurable via env var so docker-compose can mount a different
# checkpoint without rebuilding.
DEFAULT_CHECKPOINT = os.environ.get(
    "LHFM_CHECKPOINT",
    str(Path(__file__).resolve().parents[3] / "checkpoints" / "downstream.pt"),
)


# ---------------------------------------------------------------------------
# App. We use the modern lifespan context manager instead of the deprecated
# @app.on_event("startup") decorator.
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    MODEL.try_load(DEFAULT_CHECKPOINT)
    yield
    # No teardown needed: the model just lives until process exit.


app = FastAPI(
    title="longitudinal-health-foundation-model",
    description=(
        "Research prototype API for personalized behavioral health risk "
        "prediction from multimodal passive-sensing data. Not a medical device."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    """Project info banner."""
    return {
        "project": "longitudinal-health-foundation-model",
        "version": "0.1.0",
        "description": (
            "Self-supervised multimodal modeling of wearable, smartphone, and "
            "environmental signals for personalized behavioral health risk prediction."
        ),
        "model_loaded": MODEL.model is not None,
        "endpoints": ["/", "/health", "/predict", "/docs"],
        "disclaimer": DISCLAIMER,
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=MODEL.model is not None,
        encoder_d_model=MODEL.d_model,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Score a participant's recent window for the four risk tasks.

    When a trained model is loaded, the ``explanation`` field is generated
    by integrated gradients over the encoder (faithful, gradient-based).
    Without a loaded model we fall back to the rule-based panel so the
    endpoint stays usable for demos.
    """
    try:
        df = _request_to_dataframe(req)
        feat = build_full_feature_table(df, impute=True, add_targets=False)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"feature engineering failed: {exc}") from exc

    last = feat.iloc[-1]

    if MODEL.model is not None:
        probs, confidence, explanation = _predict_with_model(feat)
    else:
        probs, confidence = _fallback_predict(last)
        explanation = _build_explanation(last, probs)

    return PredictResponse(
        participant_id=req.profile.participant_id,
        window_end_date=req.window[-1].date,
        low_mood_risk=_risk(probs.get("low_mood", 0.0)),
        stress_risk=_risk(probs.get("high_stress", 0.0)),
        sleep_disruption_risk=_risk(probs.get("sleep_disruption", 0.0)),
        climate_vulnerability_risk=_risk(probs.get("climate_vulnerable", 0.0)),
        explanation=explanation,
        confidence=float(confidence),
        model_loaded=MODEL.model is not None,
        disclaimer=DISCLAIMER,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _request_to_dataframe(req: PredictRequest) -> pd.DataFrame:
    """Flatten the request into the long-form dataframe build_full_feature_table expects."""
    rows = []
    for rec in req.window:
        rec_dict = rec.model_dump()
        # Infer the missing flags when the client didn't supply them.
        wear_cols = ["daily_steps", "sleep_duration", "sleep_efficiency",
                     "resting_hr", "hrv_rmssd", "stress_score"]
        phone_cols = ["phone_unlock_count", "screen_time_minutes",
                      "mobility_radius_km", "location_entropy"]
        surv_cols = ["survey_mood", "survey_energy", "survey_stress"]
        if rec_dict.get("missing_wearable_flag") is None:
            rec_dict["missing_wearable_flag"] = int(all(rec_dict.get(c) is None for c in wear_cols))
        if rec_dict.get("missing_phone_flag") is None:
            rec_dict["missing_phone_flag"] = int(all(rec_dict.get(c) is None for c in phone_cols))
        if rec_dict.get("missing_survey_flag") is None:
            rec_dict["missing_survey_flag"] = int(all(rec_dict.get(c) is None for c in surv_cols))

        rec_dict["participant_id"] = req.profile.participant_id
        rec_dict["age"] = req.profile.age
        rec_dict["sex"] = req.profile.sex.upper()
        rec_dict["chronotype"] = req.profile.chronotype
        rec_dict["baseline_sleep_need"] = req.profile.baseline_sleep_need
        rec_dict["baseline_hrv"] = req.profile.baseline_hrv
        rec_dict["date"] = rec.date.isoformat()
        rows.append(rec_dict)

    df = pd.DataFrame(rows)
    # Heat index might be missing in the request; we approximate it if so.
    if df["heat_index"].isna().all():
        df["heat_index"] = df["temperature_c"].fillna(20.0)
    return df


def _predict_with_model(feat: pd.DataFrame):
    """Run the model on the most recent window and compute IG explanation.

    Returns ``(probs, confidence, explanation)``. The explanation is a list
    of human-readable strings produced by integrated-gradients attribution
    against the highest-probability task (so reviewers see *why* the model
    is most confidently flagging this person, rather than averaged noise).
    """
    import torch
    from lhfm.interpretability import attribute, humanize_attribution

    cols = MODEL.feature_columns
    wd = MODEL.window_days
    window = feat.tail(wd)
    if len(window) < max(7, wd // 2):
        raise HTTPException(
            status_code=422,
            detail=f"need at least {max(7, wd // 2)} feature rows (got {len(window)})",
        )
    if len(window) < wd:
        pad = wd - len(window)
        pad_rows = pd.DataFrame(
            np.zeros((pad, len(cols)), dtype=np.float32),
            columns=cols,
        )
        window = pd.concat([pad_rows, window[cols]], axis=0, ignore_index=True)
        pad_mask = np.concatenate([np.ones(pad), np.zeros(wd - pad)]).astype(bool)
    else:
        pad_mask = np.zeros(wd, dtype=bool)
    X = window[cols].to_numpy(dtype=np.float32)
    X = np.expand_dims(X, axis=0)  # (1, T, F)

    modalities = {
        name: torch.from_numpy(X[:, :, a:b]).float()
        for name, (a, b) in MODEL.modality_slices.items()
    }
    masks = {
        name: torch.zeros(X.shape[0], X.shape[1], dtype=torch.float32)
        for name in MODEL.modality_slices
    }
    pad_mask_t = torch.from_numpy(pad_mask).unsqueeze(0)  # (1, T)

    with torch.no_grad():
        logits = MODEL.model(
            modalities, masks=masks, participant_idx=None, pad_mask=pad_mask_t,
        )
        probs = {k: float(torch.sigmoid(v).cpu().numpy().reshape(-1)[0])
                 for k, v in logits.items()}

    confidence = float(np.mean([abs(p - 0.5) * 2 for p in probs.values()]))

    # Integrated gradients on the highest-probability task. We compute it
    # outside the no_grad block so autograd sees the path.
    try:
        focus_task = max(probs, key=probs.get)
        # Re-wrap inputs as leaf tensors (the ones above are read-only inside no_grad).
        ig_modalities = {
            name: torch.from_numpy(X[:, :, a:b]).float()
            for name, (a, b) in MODEL.modality_slices.items()
        }
        result = attribute(
            MODEL.model, ig_modalities, task=focus_task,
            masks=masks, pad_mask=pad_mask_t,
            n_steps=32, device="cpu",
        )
        explanation = [f"Strongest signal: {focus_task} (probability {probs[focus_task]:.2f})."]
        explanation.extend(humanize_attribution(
            result,
            feature_columns=cols,
            modality_slices=MODEL.modality_slices,
            top_k=4,
        ))
    except Exception as exc:
        log.exception("IG attribution failed, falling back to rule-based: %s", exc)
        explanation = _build_explanation(feat.iloc[-1], probs)

    return probs, confidence, explanation


def _coerce(value, default: float) -> float:
    """Convert a possibly-None / possibly-NaN value to a clean float.

    The naive ``x or default`` short-circuit replaces *legitimate zeros* with
    the default, which is wrong for things like AQI (0 is the cleanest air
    you can have). We treat only None and NaN as "missing".
    """
    if value is None:
        return float(default)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return v


def _fallback_predict(last_row: pd.Series):
    """Simple deterministic rule-based stand-in used when no model is loaded.

    This makes the API and dashboard demo'able out of the box without
    training. It is intentionally crude.
    """
    def _sig(z):
        return 1.0 / (1.0 + np.exp(-z))

    sleep = _coerce(last_row.get("sleep_duration"), 7.5)
    stress = _coerce(last_row.get("survey_stress"), 3.0)
    aqi = _coerce(last_row.get("aqi"), 50.0)
    eff = _coerce(last_row.get("sleep_efficiency"), 0.88)
    hi = _coerce(last_row.get("heat_index"), 22.0)
    hrv_dev = _coerce(last_row.get("hrv_dev_from_baseline"), 0.0)

    # Low mood: poor sleep + high stress + bad AQI -> risk up.
    z_mood = -2.0 + 0.5 * max(0, 7.5 - sleep) + 0.4 * max(0, stress - 3.0) \
        + 0.02 * max(0, aqi - 80.0)
    low_mood = float(_sig(z_mood))

    z_stress = -1.5 + 0.6 * max(0, stress - 3.5) + 0.4 * max(0, 7.5 - sleep)
    high_stress = float(_sig(z_stress))

    sleep_disrupt = float(_sig(-30.0 * (eff - 0.80)))

    z_climate = -2.0 + 0.5 * max(0, hi - 32.0) + 0.1 * max(0, -hrv_dev)
    climate_vuln = float(_sig(z_climate))

    probs = {
        "low_mood": low_mood, "high_stress": high_stress,
        "sleep_disruption": sleep_disrupt, "climate_vulnerable": climate_vuln,
    }
    # We hand back a fixed low confidence because this is not a learned model.
    return probs, 0.25


def _risk(prob: float) -> RiskScore:
    if prob >= 0.66:
        label = "elevated"
    elif prob >= 0.33:
        label = "moderate"
    else:
        label = "low"
    return RiskScore(probability=float(prob), label=label)


def _build_explanation(last_row: pd.Series, probs: dict) -> list[str]:
    """A small set of human-readable bullet points keyed off the latest day."""
    msgs: list[str] = []

    sleep = _coerce(last_row.get("sleep_duration"), float("nan"))
    eff = _coerce(last_row.get("sleep_efficiency"), float("nan"))
    hrv_dev = _coerce(last_row.get("hrv_dev_from_baseline"), 0.0)
    aqi = _coerce(last_row.get("aqi"), float("nan"))
    hi = _coerce(last_row.get("heat_index"), float("nan"))

    if not np.isnan(sleep) and sleep < 6.5:
        msgs.append(f"Recent sleep duration ({sleep:.1f} h) is below the typical 7-8 h range.")
    if not np.isnan(eff) and eff < 0.80:
        msgs.append(f"Sleep efficiency ({eff*100:.0f}%) is below 80%, suggesting fragmented sleep.")
    if hrv_dev < -8:
        msgs.append(f"HRV is {abs(hrv_dev):.0f} ms below the personal baseline -- elevated physiological strain.")
    if not np.isnan(aqi) and aqi > 100:
        msgs.append(f"Air quality is poor (AQI {aqi:.0f}).")
    if not np.isnan(hi) and hi > 32:
        msgs.append(f"Apparent temperature is elevated ({hi:.1f}°C).")

    if probs.get("low_mood", 0) >= 0.66:
        msgs.append("Low-mood risk is in the elevated range; this is a research signal, not a diagnosis.")
    if probs.get("climate_vulnerable", 0) >= 0.66:
        msgs.append("Climate-vulnerability risk is elevated; hydration and indoor cooling are sensible general precautions.")

    if not msgs:
        msgs.append("No notable risk drivers detected in the recent window.")
    return msgs
