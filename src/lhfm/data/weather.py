"""Weather and air-quality enrichment for cohorts that ship without climate.

LifeSnaps and GLOBEM both collect rich behavioural data but no temperature,
humidity, or AQI. LHFM's climate-health framing needs those columns, so
we fetch them from Open-Meteo.

Three backends, in order of how often you'll actually use them:

- ``open-meteo`` (default): free historical archive going back to 1940
  for temperature/humidity, plus a separate air-quality endpoint.
  Heads up: the air-quality endpoint reliably serves only the last ~90
  days. For older cohorts (LifeSnaps 2021-2022, GLOBEM 2018-2021), the
  ``aqi`` column will mostly come back NaN. The pipeline handles that
  gracefully -- missing AQI just shows up in ``missingness_rate_7d`` --
  but be aware your "climate-vulnerable" target will degrade.
- ``csv``: read a precomputed file. Use this when the dataset coordinator
  hands you a climate table (some IRB cohorts ship one) or when you've
  downloaded archive AQI from a paid provider like OpenWeather.
- ``noaa``: stub for now. Will fill in if anyone actually needs US-only.

Per-(lat, lon, date) lookup. When the cohort has no per-row coordinates,
the adapter supplies a single study-site lat/lon via ``AdapterConfig``.

Caching: every fetched site goes to ``data/cache/weather_cache.json``,
keyed by (rounded lat, rounded lon). The cached payload covers a full
date range, and we slice into it on subsequent calls -- so widening the
date range only fetches the new tail, not the whole window again.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd


log = logging.getLogger(__name__)


# Open-Meteo endpoints. Free; attribution required for published work.
# https://open-meteo.com/en/terms
_OPEN_METEO_HISTORICAL = "https://archive-api.open-meteo.com/v1/archive"
_OPEN_METEO_AQ = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Open-Meteo's air-quality archive is officially "past_days <= 92".
# Anything older comes back empty or unreliable. We use this constant to
# decide whether even to attempt an AQ request rather than wasting a
# round-trip.
_AQ_RELIABLE_DAYS = 90


def enrich_with_weather(df: pd.DataFrame, config) -> pd.DataFrame:
    """Merge temperature, humidity, AQI, and heat index into a long-form df.

    Required columns: ``participant_id``, ``date``. Optional
    ``latitude``/``longitude`` per row; if absent, falls back to
    ``config.default_lat``/``default_lon``.

    Strategy:
      1. ``csv`` provider: skip the API entirely and load the precomputed file.
      2. ``open-meteo``: dedupe rows to unique (lat, lon, date) triples,
         batch by site, cache responses on disk, merge back.
    """
    df = df.copy()
    climate_cols = ["temperature_c", "humidity", "aqi", "heat_index"]

    # If climate is already populated (e.g. the synthetic generator does
    # this for us), nothing to do.
    if all(c in df.columns and df[c].notna().any() for c in climate_cols):
        log.info("[weather] climate already populated; skipping enrichment")
        return df

    # Make sure those columns at least exist so downstream code doesn't crash.
    for c in climate_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Resolve where each row sits in space.
    if "latitude" in df.columns and "longitude" in df.columns and df["latitude"].notna().any():
        lat_col, lon_col = "latitude", "longitude"
    else:
        if config.default_lat is None or config.default_lon is None:
            log.warning(
                "[weather] no latitude/longitude columns and no "
                "default_lat/default_lon in adapter config; climate "
                "columns stay NaN. Set them on the AdapterConfig."
            )
            return df
        df["latitude"] = config.default_lat
        df["longitude"] = config.default_lon
        lat_col, lon_col = "latitude", "longitude"

    provider = (config.weather_provider or "open-meteo").lower()
    if provider == "csv":
        return _enrich_from_csv(df, config)
    if provider == "noaa":
        raise NotImplementedError(
            "NOAA backend not implemented yet; use 'open-meteo' or 'csv'."
        )
    if provider != "open-meteo":
        raise ValueError(f"unknown weather_provider {provider!r}")

    cache_dir = config.cache_dir or Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "weather_cache.json"
    cache = _load_cache(cache_path)

    # Round to 2 decimals (~1 km). Cities collapse to one lookup.
    df["_lat_r"] = df[lat_col].round(2)
    df["_lon_r"] = df[lon_col].round(2)
    df["date"] = pd.to_datetime(df["date"])

    sites = (
        df.groupby(["_lat_r", "_lon_r"])
          .agg(date_min=("date", "min"), date_max=("date", "max"))
          .reset_index()
    )
    log.info("[weather] need climate for %d distinct sites", len(sites))

    fetched: list[pd.DataFrame] = []
    for _, site in sites.iterrows():
        lat = float(site["_lat_r"])
        lon = float(site["_lon_r"])
        d0 = site["date_min"].date()
        d1 = site["date_max"].date()
        site_df = _fetch_one_site(lat, lon, d0, d1, cache)
        if site_df is not None:
            site_df["_lat_r"] = lat
            site_df["_lon_r"] = lon
            fetched.append(site_df)

    _save_cache(cache_path, cache)

    if not fetched:
        log.warning("[weather] no weather rows fetched; climate stays NaN")
        return df.drop(columns=["_lat_r", "_lon_r"], errors="ignore")

    weather = pd.concat(fetched, ignore_index=True)
    weather["date"] = pd.to_datetime(weather["date"])

    # Left merge: keep every original row, attach weather where we have it.
    merged = df.merge(
        weather,
        on=["_lat_r", "_lon_r", "date"],
        how="left",
        suffixes=("", "_w"),
    )
    for c in climate_cols:
        if c + "_w" in merged.columns:
            merged[c] = merged[c].fillna(merged[c + "_w"])
            merged = merged.drop(columns=[c + "_w"])
    merged = merged.drop(columns=["_lat_r", "_lon_r"], errors="ignore")
    merged["date"] = pd.to_datetime(merged["date"]).dt.date
    return merged


# ---------------------------------------------------------------------------
# Single-site fetch
# ---------------------------------------------------------------------------


def _fetch_one_site(lat: float, lon: float, d0, d1, cache: dict) -> pd.DataFrame | None:
    """Fetch (or pull from cache) the daily weather for one site + date range.

    Cache shape: ``{site_key: {"date": [...], "temperature_c": [...], ...}}``
    keyed by site (lat, lon), so date ranges are unioned rather than
    treated as separate cache entries.
    """
    try:
        import requests  # local import: keep the rest of LHFM requests-free
    except ImportError:
        log.error("[weather] `requests` not installed; pip install requests")
        return None

    site_key = f"{lat:.2f},{lon:.2f}"
    cached = cache.get(site_key)
    have_dates = set(cached.get("date", [])) if cached else set()
    requested_dates = pd.date_range(d0, d1).strftime("%Y-%m-%d").tolist()
    missing_dates = [d for d in requested_dates if d not in have_dates]

    if not missing_dates and cached is not None:
        return pd.DataFrame(cached)

    # We only fetch the missing tail. In practice for first-time runs that
    # equals the whole range; on re-runs after extending the cohort by a
    # week, we only fetch the new week.
    fetch_d0 = pd.to_datetime(min(missing_dates)).date()
    fetch_d1 = pd.to_datetime(max(missing_dates)).date()
    log.info("[weather] open-meteo fetch %s [%s..%s] (%d days)",
             site_key, fetch_d0, fetch_d1, len(missing_dates))

    # Daily weather (historical archive, ERA5; works back to 1940).
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": str(fetch_d0), "end_date": str(fetch_d1),
        "daily": ",".join([
            "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
            "relative_humidity_2m_mean", "apparent_temperature_max",
        ]),
        "timezone": "UTC",
    }
    try:
        r = requests.get(_OPEN_METEO_HISTORICAL, params=params, timeout=30)
        r.raise_for_status()
    except Exception as exc:
        log.warning("[weather] open-meteo historical failed: %s", exc)
        return None
    data = r.json().get("daily", {})
    if not data or "time" not in data:
        log.warning("[weather] empty response for %s", site_key)
        return None

    fetched = pd.DataFrame({
        "date": data["time"],
        "temperature_c": data.get("temperature_2m_mean") or [np.nan] * len(data["time"]),
        "humidity": data.get("relative_humidity_2m_mean") or [np.nan] * len(data["time"]),
        # apparent_temperature_max is the closest free thing to a heat
        # index. It tracks Steadman's heat index well above ~27°C and
        # tracks ambient temperature below that, which is exactly what
        # we want.
        "heat_index": (
            data.get("apparent_temperature_max")
            or data.get("temperature_2m_mean")
            or [np.nan] * len(data["time"])
        ),
        "aqi": [np.nan] * len(data["time"]),    # filled in below if we can
    })

    # AQI: only attempt if the fetch window is within the AQ archive's
    # reliable past-90-day window. We compute the cutoff against today's
    # UTC date rather than the cohort's max date so we don't burn API
    # calls knowing they'll fail.
    today = pd.Timestamp.utcnow().normalize().date()
    days_back = (today - fetch_d1).days
    if days_back <= _AQ_RELIABLE_DAYS:
        aq = _fetch_aq(requests, lat, lon, fetch_d0, fetch_d1)
        if aq is not None:
            fetched = fetched.merge(aq, on="date", how="left", suffixes=("", "_aq"))
            if "aqi_aq" in fetched.columns:
                fetched["aqi"] = fetched["aqi"].fillna(fetched["aqi_aq"])
                fetched = fetched.drop(columns=["aqi_aq"])
    else:
        log.info(
            "[weather] %s is %d days old, beyond AQ archive (~%d days); "
            "AQI will be NaN for this range",
            site_key, days_back, _AQ_RELIABLE_DAYS,
        )

    # Union with any previously cached data for this site.
    if cached is not None:
        prev = pd.DataFrame(cached)
        combined = pd.concat([prev, fetched], ignore_index=True)
        combined = combined.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    else:
        combined = fetched.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

    cache[site_key] = combined.to_dict("list")
    time.sleep(0.2)  # be polite to the free API
    return combined


def _fetch_aq(requests, lat: float, lon: float, d0, d1) -> pd.DataFrame | None:
    """Hourly PM2.5 → daily-mean PM2.5 → US-EPA AQI."""
    try:
        ap = requests.get(_OPEN_METEO_AQ, params={
            "latitude": lat, "longitude": lon,
            "start_date": str(d0), "end_date": str(d1),
            "hourly": "pm2_5",
            "timezone": "UTC",
        }, timeout=30)
        ap.raise_for_status()
    except Exception as exc:
        log.info("[weather] AQ fetch failed (%s); AQI stays NaN", exc)
        return None
    hourly = ap.json().get("hourly", {})
    if not hourly or "time" not in hourly or not hourly.get("pm2_5"):
        return None
    aq = pd.DataFrame(hourly)
    aq["date"] = pd.to_datetime(aq["time"]).dt.strftime("%Y-%m-%d")
    daily = aq.groupby("date", as_index=False)["pm2_5"].mean()
    daily["aqi"] = daily["pm2_5"].apply(_pm25_to_aqi)
    return daily[["date", "aqi"]]


# ---------------------------------------------------------------------------
# CSV backend
# ---------------------------------------------------------------------------


def _enrich_from_csv(df: pd.DataFrame, config) -> pd.DataFrame:
    """Merge from a precomputed weather CSV.

    Required columns: ``date``, ``temperature_c``, ``humidity``, ``aqi``,
    ``heat_index``. If the CSV also has ``latitude``/``longitude``, we
    use them for multi-site cohorts.
    """
    if config.weather_csv is None:
        raise ValueError("weather_provider='csv' but no weather_csv set on AdapterConfig")
    wx = pd.read_csv(config.weather_csv)
    wx["date"] = pd.to_datetime(wx["date"]).dt.date
    df["date"] = pd.to_datetime(df["date"]).dt.date

    merge_on = ["date"]
    if {"latitude", "longitude"}.issubset(df.columns) and {"latitude", "longitude"}.issubset(wx.columns):
        merge_on = ["date", "latitude", "longitude"]
    return df.merge(wx, on=merge_on, how="left", suffixes=("", "_w"))


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def _load_cache(path: Path) -> dict:
    """JSON sidecar. Small, diffable in PRs, fine for our cohort sizes."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            log.warning("[weather] cache load failed (%s); starting fresh", exc)
    return {}


def _save_cache(path: Path, cache: dict) -> None:
    try:
        path.write_text(json.dumps(cache, default=str))
    except Exception as exc:
        log.warning("[weather] cache save failed: %s", exc)


# ---------------------------------------------------------------------------
# PM2.5 -> US-EPA AQI
# ---------------------------------------------------------------------------


def _pm25_to_aqi(pm25: float) -> float:
    """Convert a PM2.5 reading (µg/m³) to the US-EPA AQI scale.

    Standard EPA breakpoints. Returns NaN for negative or non-finite input.
    Anything above 500.4 µg/m³ saturates at AQI 500.
    """
    if pm25 is None or not np.isfinite(pm25) or pm25 < 0:
        return float("nan")
    # (C_lo, C_hi, I_lo, I_hi)
    bps = [
        (0.0,   12.0,    0,  50),
        (12.1,  35.4,   51, 100),
        (35.5,  55.4,  101, 150),
        (55.5, 150.4,  151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for lo, hi, ilo, ihi in bps:
        if lo <= pm25 <= hi:
            return float((ihi - ilo) / (hi - lo) * (pm25 - lo) + ilo)
    return 500.0
