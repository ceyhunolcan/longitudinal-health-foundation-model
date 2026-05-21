"""Evaluation metrics for binary risk classification.

We keep this thin on purpose. sklearn already does the heavy lifting; we just
wrap things so the rest of the code can call one function and get a tidy dict.

The bootstrap_ci helper exists because a single AUROC value on a small test
set isn't very informative without a confidence interval. Reviewers will ask.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def binary_classification_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute AUROC / AUPRC / F1 / Brier / confusion matrix at a chosen threshold.

    Returns a plain dict so it serializes to JSON cleanly. We guard against
    degenerate cases (only one class in y_true) which can happen on tiny
    validation splits.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    base = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)) if len(y_true) else float("nan"),
        "threshold": threshold,
        "n_pos": int(y_true.sum()),
        "n_total": len(y_true),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if len(np.unique(y_true)) < 2:
        # AUROC / AUPRC undefined; report only what we can.
        base["auroc"] = float("nan")
        base["auprc"] = float("nan")
        return base

    base["auroc"] = float(roc_auc_score(y_true, y_prob))
    base["auprc"] = float(average_precision_score(y_true, y_prob))
    return base


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute ECE with equal-width probability bins.

    A standard reliability-diagram style summary. Not perfect (adaptive bins
    are arguably better for very imbalanced labels) but adequate for the
    research-prototype scale we care about here.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in itertools.pairwise(bins):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1.0 else (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        bin_conf = float(y_prob[mask].mean())
        bin_acc = float(y_true[mask].mean())
        weight = mask.sum() / n
        ece += weight * abs(bin_conf - bin_acc)
    return float(ece)


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean predicted prob, fraction positive, count) per bin.

    Returned arrays are aligned and have length <= n_bins (empty bins dropped).
    Useful for plotting calibration curves.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    pred_means, frac_pos, counts = [], [], []
    for lo, hi in itertools.pairwise(bins):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1.0 else (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        pred_means.append(float(y_prob[mask].mean()))
        frac_pos.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))
    return np.array(pred_means), np.array(frac_pos), np.array(counts)


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval for a metric.

    Returns ``(point_estimate, ci_low, ci_high)``.

    For longitudinal / repeated-measures data the right resampling unit is
    the *participant*, not the row. Multiple windows from the same person
    are not iid. Pass ``groups`` (a length-N array of participant ids or
    integer cluster labels) to do a **cluster bootstrap**: resample whole
    participants with replacement, concatenate their rows, then compute the
    metric. The naive row-level bootstrap (``groups=None``) overstates
    confidence by an order of magnitude on typical longitudinal cohorts;
    reach for it only when you genuinely have iid samples.

    Iterations that yield a single-class bootstrap sample are skipped (so
    they don't contribute NaN). For AUROC, pass ``sklearn.metrics.roc_auc_score``
    as ``metric_fn``; for AUPRC, ``average_precision_score``.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    n = len(y_true)
    rng = np.random.default_rng(seed)

    point = (
        float(metric_fn(y_true, y_prob))
        if len(np.unique(y_true)) >= 2 else float("nan")
    )

    if groups is not None:
        groups = np.asarray(groups).ravel()
        if len(groups) != n:
            raise ValueError(
                f"groups length {len(groups)} doesn't match samples ({n})"
            )
        unique_groups = np.array(sorted(set(groups.tolist())))
        # Pre-compute the row indices for each group so resampling is O(1).
        group_to_rows = {g: np.where(groups == g)[0] for g in unique_groups}
        n_groups = len(unique_groups)

        vals = []
        for _ in range(n_resamples):
            sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
            idx = np.concatenate([group_to_rows[g] for g in sampled_groups])
            yt = y_true[idx]
            yp = y_prob[idx]
            if len(np.unique(yt)) < 2:
                continue
            try:
                vals.append(float(metric_fn(yt, yp)))
            except ValueError:
                continue
    else:
        vals = []
        for _ in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            yt = y_true[idx]
            yp = y_prob[idx]
            if len(np.unique(yt)) < 2:
                continue
            try:
                vals.append(float(metric_fn(yt, yp)))
            except ValueError:
                continue

    if not vals:
        return point, float("nan"), float("nan")
    lo = float(np.quantile(vals, alpha / 2))
    hi = float(np.quantile(vals, 1 - alpha / 2))
    return point, lo, hi
