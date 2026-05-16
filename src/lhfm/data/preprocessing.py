"""Preprocessing: imputation, windowing, and participant-level splits.

The downstream model consumes fixed-length windows of consecutive days.
Splitting must be done by *participant*, never by row, otherwise the same
person leaks across train/val/test which would make every reported metric
optimistic.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------


def forward_fill_within_participant(
    df: pd.DataFrame,
    cols: Iterable[str],
    max_gap: int = 3,
) -> pd.DataFrame:
    """Forward-fill numeric columns inside each participant timeline.

    We cap the fill at ``max_gap`` consecutive days so very long stretches of
    missingness aren't silently papered over. Anything beyond that stays NaN
    and is later imputed with the participant's median (and finally the
    cohort median, if even that is missing).
    """
    df = df.sort_values(["participant_id", "date"]).copy()
    cols = list(cols)
    # We assign the filled values back rather than going through a groupby
    # apply because newer pandas versions handle that path inconsistently
    # (sometimes dropping the groupby column).
    df[cols] = (
        df.groupby("participant_id", sort=False)[cols]
        .transform(lambda s: s.ffill(limit=max_gap))
    )
    return df


def impute_remaining_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Per-participant median imputation, with cohort median as a backstop."""
    df = df.copy()
    cols = list(cols)
    cohort_medians = df[cols].median(numeric_only=True)

    # Per-person median fill via transform (avoids the groupby/apply gotcha
    # that strips the participant_id column on some pandas versions).
    def _fill_with_person_median(s: pd.Series) -> pd.Series:
        med = s.median()
        if pd.isna(med):
            med = cohort_medians.get(s.name, 0.0)
        return s.fillna(med)

    df[cols] = (
        df.groupby("participant_id", sort=False)[cols]
        .transform(_fill_with_person_median)
    )
    # Last-resort backstop in case some column is fully empty.
    df[cols] = df[cols].fillna(cohort_medians).fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def build_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    window_days: int = 14,
    stride: int = 1,
    target_mode: str = "next_day",
    require_consecutive_dates: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Produce sliding windows for sequence modelling.

    Parameters
    ----------
    df : long-form dataframe with at least ``participant_id`` and ``date``.
    feature_cols : list of numeric columns to feed the encoder.
    target_col : binary 0/1 column to predict.
    window_days : sequence length.
    stride : step between consecutive windows.
    target_mode : ``"next_day"`` predicts the value of ``target_col`` on the
        day after the window; ``"same_day"`` uses the final day of the window.
    require_consecutive_dates : if True (default), windows that span a gap
        in the date sequence (e.g. participant skipped a calendar day with
        no row at all) are dropped. Set to False if you trust the row-order
        to encode contiguous days regardless of explicit dates.

    Returns
    -------
    X : (N, window_days, n_features)
    y : (N,)
    pids : (N,) array of participant_ids, one per window
    end_dates : (N,) array of window-end dates (ISO strings)
    """
    df = df.sort_values(["participant_id", "date"]).reset_index(drop=True)
    X_chunks, y_chunks, pid_chunks, date_chunks = [], [], [], []

    for pid, group in df.groupby("participant_id"):
        feats = group[feature_cols].to_numpy(dtype=np.float32)
        tgts = group[target_col].to_numpy(dtype=np.float32)
        dates = pd.to_datetime(group["date"]).to_numpy()

        last_start = len(group) - window_days - (1 if target_mode == "next_day" else 0)
        if last_start < 0:
            continue

        for s in range(0, last_start + 1, stride):
            e = s + window_days

            if require_consecutive_dates:
                window_dates = dates[s:e + (1 if target_mode == "next_day" else 0)]
                diffs = np.diff(window_dates).astype("timedelta64[D]").astype(int)
                if (diffs != 1).any():
                    # Skip windows that span a date gap.
                    continue

            X_chunks.append(feats[s:e])
            if target_mode == "next_day":
                y_chunks.append(tgts[e])      # one day past the window
                date_chunks.append(dates[e - 1])
            else:
                y_chunks.append(tgts[e - 1])
                date_chunks.append(dates[e - 1])
            pid_chunks.append(pid)

    if not X_chunks:
        return (np.empty((0, window_days, len(feature_cols)), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=object),
                np.empty((0,), dtype=object))

    X = np.stack(X_chunks, axis=0)
    y = np.array(y_chunks, dtype=np.float32)
    pids = np.array(pid_chunks, dtype=object)
    end_dates = np.array(date_chunks, dtype=object)
    return X, y, pids, end_dates


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def train_val_test_split_by_participant(
    df: pd.DataFrame,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split the cohort by *participant id*.

    Returns a dict with keys ``train``, ``val``, ``test``. The exact fractions
    are approximate when participant counts don't divide evenly.
    """
    rng = np.random.default_rng(seed)
    pids = np.array(sorted(df["participant_id"].unique()))
    rng.shuffle(pids)

    n = len(pids)
    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * val_fraction)))

    test_ids = set(pids[:n_test])
    val_ids = set(pids[n_test:n_test + n_val])
    train_ids = set(pids[n_test + n_val:])

    return {
        "train": df[df["participant_id"].isin(train_ids)].copy(),
        "val": df[df["participant_id"].isin(val_ids)].copy(),
        "test": df[df["participant_id"].isin(test_ids)].copy(),
    }


def binarize_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add 0/1 target columns used by the downstream classification heads.

    Thresholds are deliberately written here in code rather than in the YAML
    config so the relationship between raw column and binary label is obvious
    when reading the code.
    """
    df = df.copy()

    # Low mood: EMA scale 1-7, treat <= 3 as a "bad day".
    df["target_low_mood"] = (df["survey_mood"] <= 3).astype(int)

    # High stress: EMA scale 1-7, treat >= 5 as a high-stress day.
    df["target_high_stress"] = (df["survey_stress"] >= 5).astype(int)

    # Sleep disruption: efficiency below 0.80 OR duration <= 5h.
    df["target_sleep_disruption"] = (
        (df["sleep_efficiency"] < 0.80) | (df["sleep_duration"] <= 5.0)
    ).astype(int)

    # Climate-vulnerable day: heat index > 32C AND HRV at least 10 ms below
    # the personal baseline (i.e. the person is showing physiological strain
    # on a hot day). When HRV is missing we can't make this call, so we leave
    # the label as NaN -- the training loop masks NaN targets out of the loss.
    hot_day = df["heat_index"] > 32.0
    hrv_drop = df["baseline_hrv"] - df["hrv_rmssd"]
    strained = hrv_drop >= 10.0

    df["target_climate_vulnerable"] = (hot_day & strained).astype(float)
    df.loc[df["hrv_rmssd"].isna(), "target_climate_vulnerable"] = np.nan
    df.loc[df["heat_index"].isna(), "target_climate_vulnerable"] = np.nan

    # NaN in any source column should not silently become 0.
    for tcol, source in [
        ("target_low_mood", "survey_mood"),
        ("target_high_stress", "survey_stress"),
        ("target_sleep_disruption", "sleep_efficiency"),
    ]:
        # Cast to float first so we can store NaN; integer columns can't.
        df[tcol] = df[tcol].astype(float)
        df.loc[df[source].isna(), tcol] = np.nan

    return df
