"""Evaluation: AUROC/AUPRC/F1/calibration + classical baselines.

Two responsibilities:

1. ``evaluate_downstream`` -- given a trained ``DownstreamRiskModel`` and a
   test dataset, compute metrics and dump tidy tables.
2. ``baseline_comparison`` -- train logistic regression, random forest, and
   (if available) XGBoost on a flattened feature matrix to give a context
   against which the foundation model can be judged.

Both functions return plain dicts so callers can json-dump them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from lhfm.models.downstream import DownstreamRiskModel
from lhfm.utils.logging import get_logger
from lhfm.utils.metrics import (
    binary_classification_report,
    expected_calibration_error,
)
from .dataset import LongitudinalWindowDataset, collate_windows


log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Deep model evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_downstream(
    model: DownstreamRiskModel,
    test_dataset: LongitudinalWindowDataset,
    task_names: Iterable[str],
    device: str = "cpu",
    batch_size: int = 64,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 0,
    cluster_bootstrap: bool = True,
) -> dict[str, dict]:
    """Run the model over a held-out dataset and return per-task metrics
    with bootstrap 95% CIs for AUROC and AUPRC.

    When ``cluster_bootstrap=True`` (default) and the dataset carries
    ``participant_idx``, the bootstrap resamples *participants* with
    replacement rather than individual windows. This is the methodologically
    correct unit for longitudinal data; the row-level alternative gives
    artificially tight CIs because multiple windows from the same person are
    not independent.

    Returns
    -------
    {task_name: {auroc, auroc_ci, auprc, auprc_ci, f1, ece, brier,
                 confusion_matrix, n_pos, n_total, y_true, y_prob,
                 bootstrap_unit}}
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    from lhfm.utils.metrics import bootstrap_ci

    model = model.to(device).eval()
    task_names = list(task_names)

    loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_windows,
    )

    all_preds = {n: [] for n in task_names}
    all_targets = {n: [] for n in task_names}
    all_pids: list[int] = []

    has_pid = test_dataset.participant_idx is not None

    for batch in loader:
        modalities = {k: v.to(device) for k, v in batch["modalities"].items()}
        masks = {k: v.to(device) for k, v in batch["masks"].items()}
        participant_idx = (
            batch["participant_idx"].to(device)
            if "participant_idx" in batch else None
        )
        y = batch["y"].cpu().numpy()
        if has_pid:
            all_pids.extend(batch["participant_idx"].cpu().numpy().tolist())

        logits = model(modalities, masks=masks, participant_idx=participant_idx)
        for i, name in enumerate(task_names):
            prob = torch.sigmoid(logits[name]).cpu().numpy()
            all_preds[name].append(prob)
            all_targets[name].append(y[:, i])

    pid_array = np.array(all_pids, dtype=np.int64) if has_pid else None
    use_clusters = cluster_bootstrap and has_pid
    bootstrap_unit = "participant" if use_clusters else "window"

    results: dict[str, dict] = {}
    for name in task_names:
        p = np.concatenate(all_preds[name]) if all_preds[name] else np.empty(0)
        t = np.concatenate(all_targets[name]) if all_targets[name] else np.empty(0)
        valid = ~np.isnan(t)
        if valid.sum() < 2 or len(np.unique(t[valid])) < 2:
            results[name] = {
                "auroc": float("nan"), "auprc": float("nan"),
                "auroc_ci": (float("nan"), float("nan")),
                "auprc_ci": (float("nan"), float("nan")),
                "f1": float("nan"), "ece": float("nan"), "brier": float("nan"),
                "confusion_matrix": [[0, 0], [0, 0]],
                "n_pos": int(np.nansum(t)),
                "n_total": int(valid.sum()),
                "bootstrap_unit": bootstrap_unit,
            }
            continue

        rep = binary_classification_report(t[valid], p[valid])
        rep["ece"] = expected_calibration_error(t[valid], p[valid])
        groups = pid_array[valid] if use_clusters else None
        _, lo, hi = bootstrap_ci(
            t[valid], p[valid], roc_auc_score,
            n_resamples=bootstrap_resamples, seed=bootstrap_seed,
            groups=groups,
        )
        rep["auroc_ci"] = (lo, hi)
        _, lo, hi = bootstrap_ci(
            t[valid], p[valid], average_precision_score,
            n_resamples=bootstrap_resamples, seed=bootstrap_seed + 1,
            groups=groups,
        )
        rep["auprc_ci"] = (lo, hi)
        rep["bootstrap_unit"] = bootstrap_unit
        rep["y_true"] = t[valid].astype(int).tolist()
        rep["y_prob"] = p[valid].astype(float).tolist()
        results[name] = rep

    return results


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------


def _flatten_windows(X: np.ndarray) -> np.ndarray:
    """Collapse the time axis by taking simple per-window summary statistics.

    This is the canonical "treat the window as a fixed feature vector"
    baseline you see in most digital-health papers. We use mean, std, last
    and slope across time, which is enough to beat a naive baseline by a
    comfortable margin without doing anything sequence-aware.
    """
    B, T, F = X.shape
    means = X.mean(axis=1)
    stds = X.std(axis=1)
    last = X[:, -1, :]
    # Simple slope per feature via least squares against time.
    t = np.arange(T, dtype=np.float32)
    t_mean = t.mean()
    t_centered = t - t_mean
    denom = (t_centered ** 2).sum() + 1e-6
    # (B, T, F) -> per-feature slope via einsum-style reduction.
    slopes = ((X - means[:, None, :]) * t_centered[None, :, None]).sum(axis=1) / denom
    return np.concatenate([means, stds, last, slopes], axis=1)


def baseline_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_name: str,
    include_xgboost: bool = True,
    random_state: int = 42,
) -> dict[str, dict]:
    """Train classical baselines on flattened windows and return their metrics.

    ``y_train`` / ``y_test`` are 1D binary label arrays. NaN labels are
    filtered out before training.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    Xtr = _flatten_windows(X_train)
    Xte = _flatten_windows(X_test)

    # Drop rows where label is NaN.
    tr_mask = ~np.isnan(y_train)
    te_mask = ~np.isnan(y_test)
    Xtr, ytr = Xtr[tr_mask], y_train[tr_mask].astype(int)
    Xte, yte = Xte[te_mask], y_test[te_mask].astype(int)

    out: dict[str, dict] = {}

    if len(np.unique(ytr)) < 2:
        log.warning("baseline_comparison: only one class present for task %s, skipping", task_name)
        return out

    # Logistic regression -- balanced class weights since labels are
    # frequently imbalanced.
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
    ])
    lr.fit(Xtr, ytr)
    out["logreg"] = _baseline_metrics(yte, lr.predict_proba(Xte)[:, 1])

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1,
        class_weight="balanced", random_state=random_state,
    )
    rf.fit(Xtr, ytr)
    out["random_forest"] = _baseline_metrics(yte, rf.predict_proba(Xte)[:, 1])

    if include_xgboost:
        try:
            import xgboost as xgb
            pos = max(1, ytr.sum())
            neg = max(1, len(ytr) - ytr.sum())
            xgb_clf = xgb.XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                eval_metric="logloss", tree_method="hist",
                scale_pos_weight=float(neg) / float(pos),
                random_state=random_state,
            )
            xgb_clf.fit(Xtr, ytr, verbose=False)
            out["xgboost"] = _baseline_metrics(yte, xgb_clf.predict_proba(Xte)[:, 1])
        except ImportError:
            log.info("xgboost not installed -- skipping that baseline (this is fine)")

    return out


def _baseline_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    rep = binary_classification_report(y_true, y_prob)
    rep["ece"] = expected_calibration_error(y_true, y_prob)
    return rep


# ---------------------------------------------------------------------------
# Representation visualization
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_representations(
    model: DownstreamRiskModel,
    dataset: LongitudinalWindowDataset,
    device: str = "cpu",
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (N, d_model) embeddings and (N,) participant indices.

    Used to draw PCA/t-SNE plots in the notebooks.
    """
    model = model.to(device).eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_windows,
    )
    reps, pids = [], []
    for batch in loader:
        modalities = {k: v.to(device) for k, v in batch["modalities"].items()}
        masks = {k: v.to(device) for k, v in batch["masks"].items()}
        participant_idx = (
            batch["participant_idx"].to(device)
            if "participant_idx" in batch else None
        )
        enc_out = model.encoder(modalities, masks=masks, participant_idx=participant_idx)
        reps.append(enc_out["representation"].cpu().numpy())
        if participant_idx is not None:
            pids.append(participant_idx.cpu().numpy())
        else:
            pids.append(np.zeros(modalities[next(iter(modalities))].shape[0], dtype=np.int64))
    return np.concatenate(reps, axis=0), np.concatenate(pids, axis=0)


def save_results_table(results: dict[str, dict], path: Path | str) -> None:
    """Dump a tidy CSV summarising per-task metrics with bootstrap CIs.

    The ``bootstrap_unit`` column distinguishes participant-clustered CIs
    (the methodologically correct unit for longitudinal data) from
    row-level ones (overconfident; included only for back-compat).
    """
    import pandas as pd
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for task, m in results.items():
        auroc_ci = m.get("auroc_ci", (float("nan"), float("nan")))
        auprc_ci = m.get("auprc_ci", (float("nan"), float("nan")))
        rows.append({
            "task": task,
            "auroc": m.get("auroc"),
            "auroc_ci_low": auroc_ci[0],
            "auroc_ci_high": auroc_ci[1],
            "auprc": m.get("auprc"),
            "auprc_ci_low": auprc_ci[0],
            "auprc_ci_high": auprc_ci[1],
            "f1": m.get("f1"),
            "ece": m.get("ece"),
            "brier": m.get("brier"),
            "n_pos": m.get("n_pos"),
            "n_total": m.get("n_total"),
            "bootstrap_unit": m.get("bootstrap_unit", "unknown"),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
