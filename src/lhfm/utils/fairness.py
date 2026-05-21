"""Subgroup-stratified fairness audit for downstream tasks.

Slices the held-out test set by participant-level attributes (sex, age
band, race/ethnicity, SES proxy, geographic region, device generation,
depression status, anxiety status) and reports per-subgroup AUROC with
participant-clustered bootstrap CIs, AUPRC, ECE, Brier, FPR, FNR. The
per-axis equalised-odds gap is max-FPR-gap + max-FNR-gap.

A note on running this against the synthetic generator: the generator
draws subgroups independently of outcomes, so a well-trained model
should pass with low gaps. The audit on synthetic data is therefore a
pipeline check, not a finding. Real cohorts are where this gets
interesting.

API:
    run_fairness_audit(y_true, y_prob, metadata, threshold=0.5, ...) -> dict
    fairness_report_to_csv(audit, path) -> None
    check_fairness_thresholds(audit, max_auroc_gap, max_eo_violation)
        -> (ok, violations)

The third one is what ``--fail-on-violation`` calls under the hood, so
it's also a CI gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import binary_classification_report, bootstrap_ci, expected_calibration_error

# Subgroups we report on by default. Each entry maps an *output axis name*
# to either:
#   - a string: the source column name in metadata (used verbatim as labels)
#   - a tuple (source_col, transform): source column + a callable that takes
#     the source series and returns a categorical series of labels
DEFAULT_SUBGROUPS: dict[str, Any] = {
    "sex": "sex",
    "race_ethnicity": "race_ethnicity",
    "ses_proxy": "ses_proxy",
    "region": "region",
    "device_gen": "device_gen",
    "has_depression": "has_depression",
    "has_anxiety": "has_anxiety",
    # Age binned into clinically meaningful bands.
    "age_band": ("age", lambda s: pd.cut(
        s.astype(float),
        bins=[0, 25, 35, 50, 65, 200],
        labels=["18-24", "25-34", "35-49", "50-64", "65+"],
        include_lowest=True,
    )),
}


def run_fairness_audit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metadata: pd.DataFrame,
    threshold: float = 0.5,
    min_subgroup_n: int = 20,
    bootstrap_resamples: int = 500,
    seed: int = 0,
    subgroups: dict[str, Any] | None = None,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute per-subgroup metrics and equalised-odds gaps.

    Parameters
    ----------
    y_true, y_prob : aligned (N,) prediction arrays.
    metadata : (N, M) dataframe of subgroup attributes per prediction row.
        Columns missing from ``DEFAULT_SUBGROUPS`` (or ``subgroups``) are
        silently ignored.
    threshold : decision threshold for FPR / FNR / equalised-odds.
    min_subgroup_n : subgroups smaller than this are dropped from the
        report (and noted in ``small_subgroups``). Keeps tables from being
        dominated by single-row strata that don't carry statistical signal.
    bootstrap_resamples : per-subgroup bootstrap for AUROC CIs. Uses cluster
        bootstrap if ``groups`` is supplied (participant ids).
    subgroups : optional override for ``DEFAULT_SUBGROUPS``.
    groups : optional (N,) array of participant ids for clustered bootstrap.

    Returns
    -------
    dict with keys:
        - ``per_subgroup``: list of dicts, one per (axis, level)
        - ``equalised_odds_gap``: dict from axis -> max-FPR-gap + max-FNR-gap
        - ``small_subgroups``: list of (axis, level, n) we skipped
    """
    if subgroups is None:
        subgroups = DEFAULT_SUBGROUPS

    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    if len(y_true) != len(metadata):
        raise ValueError(
            f"metadata rows ({len(metadata)}) and predictions ({len(y_true)}) "
            f"are out of alignment"
        )
    if groups is not None:
        groups = np.asarray(groups).ravel()
        if len(groups) != len(y_true):
            raise ValueError("groups length must match predictions")

    # Drop NaN labels up front so every subgroup metric is computed on
    # actually-observed labels.
    valid = ~np.isnan(y_true.astype(float))
    y_true, y_prob = y_true[valid], y_prob[valid]
    metadata = metadata.iloc[valid].reset_index(drop=True)
    groups_v = groups[valid] if groups is not None else None

    from sklearn.metrics import roc_auc_score

    per_subgroup: list[dict[str, Any]] = []
    small: list[tuple[str, str, int]] = []
    fprs_by_axis: dict[str, dict[str, float]] = {}
    fnrs_by_axis: dict[str, dict[str, float]] = {}

    for axis_name, spec in subgroups.items():
        # Resolve (source column, transform) from the registry entry.
        if isinstance(spec, tuple) and len(spec) == 2:
            source_col, transform = spec
        else:
            source_col, transform = spec, None
        if source_col not in metadata.columns:
            continue
        series = metadata[source_col]
        if transform is not None:
            series = transform(series)
        for level, mask_series in _iter_levels(series):
            mask = mask_series.to_numpy()
            n = int(mask.sum())
            if n < min_subgroup_n:
                small.append((axis_name, str(level), n))
                continue
            yt = y_true[mask]
            yp = y_prob[mask]
            yhat = (yp >= threshold).astype(int)
            n_pos = int(yt.sum())
            n_neg = int(n - n_pos)
            # AUROC defined only when both classes present in this subgroup.
            if n_pos > 0 and n_neg > 0 and len(np.unique(yt)) >= 2:
                rep = binary_classification_report(yt, yp, threshold=threshold)
                rep["ece"] = expected_calibration_error(yt, yp)
                _auroc_pt, auroc_lo, auroc_hi = bootstrap_ci(
                    yt, yp, roc_auc_score,
                    n_resamples=bootstrap_resamples, seed=seed,
                    groups=groups_v[mask] if groups_v is not None else None,
                )
                rep["auroc_ci"] = (auroc_lo, auroc_hi)
                # FPR = P(yhat=1 | y=0); FNR = P(yhat=0 | y=1)
                fpr = float((yhat[yt == 0] == 1).mean()) if n_neg > 0 else float("nan")
                fnr = float((yhat[yt == 1] == 0).mean()) if n_pos > 0 else float("nan")
            else:
                rep = {
                    "auroc": float("nan"), "auprc": float("nan"),
                    "auroc_ci": (float("nan"), float("nan")),
                    "f1": float("nan"), "brier": float("nan"), "ece": float("nan"),
                    "n_pos": n_pos, "n_total": n,
                }
                fpr = float("nan")
                fnr = float("nan")

            rep["fpr"] = fpr
            rep["fnr"] = fnr
            per_subgroup.append({
                "axis": axis_name, "level": str(level), "n": n,
                "n_pos": n_pos, "n_neg": n_neg,
                **rep,
            })

            fprs_by_axis.setdefault(axis_name, {})[str(level)] = fpr
            fnrs_by_axis.setdefault(axis_name, {})[str(level)] = fnr

    # Equalised-odds gap per axis: max FPR difference + max FNR difference.
    eo_gap: dict[str, dict[str, float]] = {}
    for col in fprs_by_axis:
        fprs = [v for v in fprs_by_axis[col].values() if not np.isnan(v)]
        fnrs = [v for v in fnrs_by_axis[col].values() if not np.isnan(v)]
        if len(fprs) >= 2 and len(fnrs) >= 2:
            eo_gap[col] = {
                "max_fpr_gap": float(max(fprs) - min(fprs)),
                "max_fnr_gap": float(max(fnrs) - min(fnrs)),
                "equalised_odds_violation": float(
                    (max(fprs) - min(fprs)) + (max(fnrs) - min(fnrs))
                ),
            }

    return {
        "per_subgroup": per_subgroup,
        "equalised_odds_gap": eo_gap,
        "small_subgroups": small,
        "threshold": threshold,
        "min_subgroup_n": min_subgroup_n,
    }


def _iter_levels(series: pd.Series):
    """Yield (level, boolean-mask) for each category, skipping NaN."""
    s = series.astype("object").where(~pd.isna(series), other=None)
    for level in pd.unique(s.dropna()):
        yield level, (s == level)


def fairness_report_to_csv(audit: dict[str, Any], path: Path | str) -> None:
    """Write the per-subgroup table + equalised-odds gaps to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if audit["per_subgroup"]:
        rows = []
        for r in audit["per_subgroup"]:
            auroc_ci = r.get("auroc_ci", (float("nan"), float("nan")))
            rows.append({
                "axis": r["axis"], "level": r["level"], "n": r["n"],
                "n_pos": r["n_pos"], "n_neg": r["n_neg"],
                "auroc": r["auroc"],
                "auroc_ci_low": auroc_ci[0], "auroc_ci_high": auroc_ci[1],
                "auprc": r["auprc"], "f1": r["f1"], "brier": r["brier"],
                "ece": r["ece"], "fpr": r["fpr"], "fnr": r["fnr"],
            })
        pd.DataFrame(rows).to_csv(path, index=False)
        # Equalised-odds summary as a sibling CSV.
        if audit["equalised_odds_gap"]:
            eo_rows = [
                {"axis": axis, **gaps}
                for axis, gaps in audit["equalised_odds_gap"].items()
            ]
            pd.DataFrame(eo_rows).to_csv(
                path.with_name(path.stem + "_equalised_odds.csv"), index=False,
            )


def check_fairness_thresholds(
    audit: dict[str, Any],
    max_auroc_gap: float = 0.10,
    max_eo_violation: float = 0.20,
) -> tuple[bool, list[str]]:
    """Return ``(ok, violation_messages)`` for CI gating.

    Raises no exceptions; the caller decides whether to fail CI on
    violations. Defaults are conservative — 10 percentage points of AUROC
    spread or 20 percentage points of FPR+FNR drift is the kind of gap a
    real reviewer would flag.
    """
    violations: list[str] = []

    # AUROC spread per axis.
    by_axis: dict[str, list[float]] = {}
    for r in audit["per_subgroup"]:
        if not np.isnan(r["auroc"]):
            by_axis.setdefault(r["axis"], []).append(r["auroc"])
    for axis, vals in by_axis.items():
        if len(vals) < 2:
            continue
        spread = max(vals) - min(vals)
        if spread > max_auroc_gap:
            violations.append(
                f"AUROC spread on {axis} = {spread:.3f} exceeds threshold {max_auroc_gap:.3f} "
                f"(levels: {sorted(vals)})"
            )

    # Equalised-odds violation per axis.
    for axis, gaps in audit["equalised_odds_gap"].items():
        v = gaps["equalised_odds_violation"]
        if v > max_eo_violation:
            violations.append(
                f"equalised-odds violation on {axis} = {v:.3f} (FPR gap {gaps['max_fpr_gap']:.3f} + "
                f"FNR gap {gaps['max_fnr_gap']:.3f}) exceeds threshold {max_eo_violation:.3f}"
            )

    return len(violations) == 0, violations
