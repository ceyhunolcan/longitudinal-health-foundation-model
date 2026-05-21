"""Climate-regime generalization helpers.

The model claims to be climate-aware. The straightforward way to test
that claim is to hold out heat-wave (or cold-snap, or smoke-episode)
days at training time and check whether discrimination survives at
test time on the held-out regime. If it collapses, the climate-health
framing is rhetorical; if it holds up, the model has learned something
about climate stress that generalises.

This module is just the plumbing -- regime masks, regime summaries, and
a row-level split helper. The actual train-then-test loop lives in
``scripts/run_climate_holdout.py``.

API:
    CLIMATE_REGIMES                 - tuple of regime names
    define_climate_regime(df, name) - bool mask per row
    regime_summary(df)              - per-regime counts + label balance
    split_train_eval_by_regime(...) - row-level split for the holdout
"""

from __future__ import annotations

import pandas as pd

CLIMATE_REGIMES = ("normal", "heat_wave", "cold_snap", "smoke_episode")


def define_climate_regime(df: pd.DataFrame, regime: str) -> pd.Series:
    """Return a boolean mask over ``df`` rows for the named climate regime.

    Definitions (intentionally simple; should be refined for real data):

    - ``heat_wave``     : heat_index >= 32°C
    - ``cold_snap``     : temperature_c <= 5°C
    - ``smoke_episode`` : aqi >= 150
    - ``normal``        : none of the above
    """
    if regime not in CLIMATE_REGIMES:
        raise ValueError(f"unknown regime {regime!r}; expected one of {CLIMATE_REGIMES}")
    hi = df["heat_index"].astype(float)
    tc = df["temperature_c"].astype(float)
    aqi = df["aqi"].astype(float)
    heat = (hi >= 32.0).fillna(False)
    cold = (tc <= 5.0).fillna(False)
    smoke = (aqi >= 150.0).fillna(False)
    if regime == "heat_wave":
        return heat
    if regime == "cold_snap":
        return cold
    if regime == "smoke_episode":
        return smoke
    # normal
    return ~(heat | cold | smoke)


def regime_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Quick table of per-regime row counts and target prevalences.

    Useful for spotting the case where holding out heat-wave days for
    evaluation leaves you with too few test rows to draw any conclusion.
    """
    out = {}
    for regime in CLIMATE_REGIMES:
        mask = define_climate_regime(df, regime)
        sub = df.loc[mask]
        out[regime] = {
            "n_rows": int(mask.sum()),
            "n_participants": int(sub["participant_id"].nunique()) if "participant_id" in df else None,
        }
        for tcol in ("target_low_mood", "target_high_stress",
                     "target_sleep_disruption", "target_climate_vulnerable"):
            if tcol in df.columns:
                vals = sub[tcol].dropna()
                out[regime][f"{tcol}_prevalence"] = float(vals.mean()) if len(vals) else float("nan")
                out[regime][f"{tcol}_n_labelled"] = len(vals)
    return pd.DataFrame(out).T


def split_train_eval_by_regime(
    df: pd.DataFrame,
    train_regimes: tuple[str, ...] = ("normal",),
    eval_regime: str = "heat_wave",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows so the model sees only ``train_regimes`` during training
    and is evaluated on ``eval_regime`` (and ``normal`` for reference).

    This is *row-level* splitting — useful for evaluating already-trained
    features, but if you want to actually retrain on it, you'll need to
    re-window so each window stays within a single regime. Most often
    that's overkill for a synthetic prototype, so callers stick with
    row-level splits and the CLI's `--regime-holdout` flag in
    ``scripts/run_climate_holdout.py``.
    """
    train_mask = pd.Series(False, index=df.index)
    for r in train_regimes:
        train_mask = train_mask | define_climate_regime(df, r)
    eval_mask = define_climate_regime(df, eval_regime)
    return df.loc[train_mask].copy(), df.loc[eval_mask].copy()
