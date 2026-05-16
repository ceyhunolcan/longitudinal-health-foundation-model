"""API tests. Skipped automatically if FastAPI / starlette TestClient unavailable."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

fastapi = pytest.importorskip("fastapi")
# httpx is needed by starlette's TestClient in recent versions.
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from lhfm.api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    payload = r.json()
    assert payload["project"] == "longitudinal-health-foundation-model"
    assert "disclaimer" in payload


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "ok"
    assert "model_loaded" in payload


def _make_window(n_days: int = 14) -> list[dict]:
    today = date(2024, 7, 1)
    return [
        {
            "date": (today + timedelta(days=i)).isoformat(),
            "daily_steps": 7500 + (i % 4) * 200,
            "sleep_duration": 7.2 - (i % 3) * 0.4,
            "sleep_efficiency": 0.87,
            "resting_hr": 62.0,
            "hrv_rmssd": 50.0,
            "stress_score": 30.0,
            "phone_unlock_count": 80.0,
            "screen_time_minutes": 210.0,
            "mobility_radius_km": 6.0,
            "location_entropy": 1.1,
            "survey_mood": 5.0 if i % 5 else 3.0,
            "survey_energy": 4.5,
            "survey_stress": 3.0 if i % 5 else 5.5,
            "temperature_c": 24.0 + (i % 7),
            "humidity": 55.0,
            "aqi": 60.0,
            "heat_index": 25.0,
        }
        for i in range(n_days)
    ]


def test_predict_returns_all_risk_fields(client):
    payload = {
        "profile": {
            "participant_id": "TEST_001", "age": 35, "sex": "F",
            "chronotype": "intermediate",
            "baseline_sleep_need": 7.8, "baseline_hrv": 55.0,
        },
        "window": _make_window(14),
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    for field in [
        "low_mood_risk", "stress_risk",
        "sleep_disruption_risk", "climate_vulnerability_risk",
    ]:
        assert field in body
        assert 0.0 <= body[field]["probability"] <= 1.0
        assert body[field]["label"] in {"low", "moderate", "elevated"}
    assert isinstance(body["explanation"], list) and body["explanation"]


def test_predict_rejects_too_short_window(client):
    payload = {
        "profile": {
            "participant_id": "TEST_002", "age": 30, "sex": "M",
            "chronotype": "morning",
            "baseline_sleep_need": 7.5, "baseline_hrv": 60.0,
        },
        "window": _make_window(3),
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_rejects_duplicate_dates(client):
    window = _make_window(7)
    window[3]["date"] = window[2]["date"]
    payload = {
        "profile": {
            "participant_id": "TEST_003", "age": 28, "sex": "F",
            "chronotype": "evening",
            "baseline_sleep_need": 8.0, "baseline_hrv": 70.0,
        },
        "window": window,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
