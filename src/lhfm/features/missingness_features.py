"""Features that describe *the pattern of missingness itself*.

Informative missingness is the rule, not the exception, in passive-sensing
studies. People skip surveys when they're stressed; they leave the watch on
the charger when they're depressed; phones go uncharged during travel. So
the *shape* of the missing-data process is a feature in its own right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add missingness-pattern features.

    New columns:
        any_missing, consecutive_missing_days, missingness_rate_7d,
        modality_dropout_entropy
    """
    df = df.sort_values(["participant_id", "date"]).copy()
    df["any_missing"] = (
        df["missing_wearable_flag"]
        | df["missing_phone_flag"]
        | df["missing_survey_flag"]
    ).astype(int)

    chunks = []
    for _pid, g in df.groupby("participant_id"):
        g = g.copy()

        # Consecutive run length of any-missing days.
        run = 0
        runs = []
        for v in g["any_missing"].values:
            run = run + 1 if v else 0
            runs.append(run)
        g["consecutive_missing_days"] = runs

        g["missingness_rate_7d"] = g["any_missing"].rolling(7, min_periods=1).mean()

        # Per-row entropy across the three modality flags. With three binary
        # flags the max entropy is log(3) so we normalize by it.
        roll = (
            g[["missing_wearable_flag", "missing_phone_flag", "missing_survey_flag"]]
            .rolling(7, min_periods=1)
            .mean()
            .values
        )
        # Normalize each row to a probability distribution. If everything is
        # zero (no missingness in the window) we set entropy to 0.
        row_sum = roll.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            p = np.where(row_sum > 0, roll / row_sum, 0.0)
            ent = -np.where(p > 0, p * np.log(p + 1e-12), 0.0).sum(axis=1)
        g["modality_dropout_entropy"] = ent / np.log(3.0)

        chunks.append(g)

    return pd.concat(chunks, axis=0).sort_index()
