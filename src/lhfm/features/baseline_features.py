"""Personal-baseline features and the master feature-table builder.

The "baseline" features encode *who the participant is* in a few static or
slowly-varying numbers: age, chronotype, prior HRV, etc. The model also
learns a participant embedding, but explicit baseline features give a useful
inductive bias and let us interpret risk in personalized terms.

Age standardization is *parameterized*, not hardcoded. The right value for
``AGE_REF_MEAN`` and ``AGE_REF_STD`` is the *training-set* mean and std,
which the training pipeline computes once and persists to the checkpoint
meta. The fallback constants below match the synthetic generator's age
prior and exist only so that downstream code (notebooks, smoke tests,
single-participant inference before the model has been fit) doesn't crash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .climate_features import compute_climate_features
from .missingness_features import compute_missingness_features
from .smartphone_features import compute_smartphone_features
from .wearable_features import compute_wearable_features

_CHRONOTYPE_MAP = {"morning": -1.0, "intermediate": 0.0, "evening": 1.0}

# Fallback population reference stats. These match the synthetic generator's
# age prior (~ N(34, 11)) and are used **only** when the caller hasn't passed
# explicit reference statistics. In any real evaluation those should come
# from the *training* split, never from the dataframe being scored, and
# never from the full cohort.
FALLBACK_AGE_REF_MEAN = 34.0
FALLBACK_AGE_REF_STD = 11.0


def fit_baseline_reference_stats(train_df: pd.DataFrame) -> dict[str, float]:
    """Compute population-reference statistics on the *training* split.

    Returns a dict suitable for persisting alongside model weights and
    loading back at inference time. We take one row per participant so a
    participant with many days doesn't dominate the cohort mean.
    """
    per_participant = (
        train_df[["participant_id", "age"]]
        .dropna()
        .drop_duplicates(subset="participant_id")
    )
    if len(per_participant) < 2:
        return {"age_ref_mean": FALLBACK_AGE_REF_MEAN,
                "age_ref_std": FALLBACK_AGE_REF_STD}
    age = per_participant["age"].astype(float)
    mu = float(age.mean())
    sd = float(age.std(ddof=1))
    if sd <= 1e-6:
        sd = FALLBACK_AGE_REF_STD
    return {"age_ref_mean": mu, "age_ref_std": sd}


def compute_baseline_features(
    df: pd.DataFrame,
    age_ref_mean: float | None = None,
    age_ref_std: float | None = None,
) -> pd.DataFrame:
    """Add baseline / personal-context features.

    ``age_z`` is computed against either explicit reference statistics (the
    correct path when fitting / evaluating with a trained model) or the
    fallback constants documented at module level. The single-participant
    API path uses the fallback because the trained reference stats are
    loaded separately by the API and applied directly to the request.

    New columns:
        age_z, chronotype_score, sex_male
    """
    df = df.copy()
    mu = FALLBACK_AGE_REF_MEAN if age_ref_mean is None else float(age_ref_mean)
    sd = FALLBACK_AGE_REF_STD if age_ref_std is None else float(age_ref_std)
    if sd <= 1e-6:
        sd = FALLBACK_AGE_REF_STD
    age = df["age"].astype(float)
    df["age_z"] = (age - mu) / sd
    df["chronotype_score"] = df["chronotype"].map(_CHRONOTYPE_MAP).fillna(0.0)
    df["sex_male"] = (df["sex"].astype(str).str.upper() == "M").astype(int)
    return df


def build_full_feature_table(
    df: pd.DataFrame,
    impute: bool = True,
    add_targets: bool = True,
    age_ref_mean: float | None = None,
    age_ref_std: float | None = None,
) -> pd.DataFrame:
    """One-call pipeline: raw long-form -> fully engineered feature table.

    Parameters
    ----------
    df : raw long-form dataframe (output of the synthetic generator).
    impute : if True, run forward-fill and per-participant median imputation
             on the engineered numeric columns. This is what the downstream
             model expects, but you may want it off for descriptive plots.
    add_targets : if True, add the binary target columns used by the
                  downstream classification heads.
    age_ref_mean, age_ref_std : optional population reference stats for
        age standardization. Pass the values returned by
        :func:`fit_baseline_reference_stats` on the training split. When
        omitted we fall back to the synthetic-prior constants.

    Returns
    -------
    pd.DataFrame with all original columns plus the engineered features
    (and optionally targets), sorted by (participant_id, date).
    """
    out = compute_wearable_features(df)
    out = compute_smartphone_features(out)
    out = compute_climate_features(out)
    out = compute_missingness_features(out)
    out = compute_baseline_features(out, age_ref_mean=age_ref_mean, age_ref_std=age_ref_std)

    if add_targets:
        from lhfm.data.preprocessing import binarize_targets
        out = binarize_targets(out)

    if impute:
        from lhfm.data.preprocessing import (
            forward_fill_within_participant,
            impute_remaining_numeric,
        )
        numeric = out.select_dtypes(include=[np.number]).columns.tolist()
        target_cols = [c for c in out.columns if c.startswith("target_")]
        fill_cols = [c for c in numeric if c not in target_cols
                     and not c.startswith("missing_")]
        out = forward_fill_within_participant(out, cols=fill_cols, max_gap=3)
        out = impute_remaining_numeric(out, cols=fill_cols)

    return out.sort_values(["participant_id", "date"]).reset_index(drop=True)


# Kept for backward compatibility; explicitly documented as fallback constants.
AGE_REF_MEAN = FALLBACK_AGE_REF_MEAN
AGE_REF_STD = FALLBACK_AGE_REF_STD
