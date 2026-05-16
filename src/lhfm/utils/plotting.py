"""Plotting helpers built on matplotlib (no seaborn, per project policy).

Everything here returns a matplotlib Figure so callers can decide whether to
save, display, or embed in Streamlit.
"""

from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import reliability_curve


def plot_participant_trends(
    df: pd.DataFrame,
    columns: Iterable[str],
    title: Optional[str] = None,
) -> plt.Figure:
    """Stack of small line plots, one per column, sharing the x-axis.

    Expects a date-indexed (or 'date'-columned) frame for a single participant.
    """
    cols = list(columns)
    if "date" in df.columns:
        x = pd.to_datetime(df["date"])
    else:
        x = df.index

    fig, axes = plt.subplots(len(cols), 1, figsize=(9, 1.6 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        if col not in df.columns:
            ax.text(0.5, 0.5, f"{col} not available", transform=ax.transAxes,
                    ha="center", va="center", color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        ax.plot(x, df[col].values, linewidth=1.2)
        ax.set_ylabel(col, fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("date")
    if title:
        fig.suptitle(title, y=1.0, fontsize=11)
    fig.tight_layout()
    return fig


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> plt.Figure:
    """Reliability diagram plus the y=x reference line."""
    mean_pred, frac_pos, _ = reliability_curve(y_true, y_prob, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    if len(mean_pred) > 0:
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=1.5, label="model")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("fraction of positives")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("calibration")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_confusion(cm: np.ndarray, labels: tuple[str, str] = ("neg", "pos")) -> plt.Figure:
    """Tiny annotated confusion matrix."""
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_embedding_2d(coords: np.ndarray, labels: np.ndarray, title: str = "embedding") -> plt.Figure:
    """Scatter of N x 2 coordinates colored by an integer label vector."""
    fig, ax = plt.subplots(figsize=(5, 5))
    uniq = np.unique(labels)
    for u in uniq:
        m = labels == u
        ax.scatter(coords[m, 0], coords[m, 1], s=10, alpha=0.7, label=str(u))
    ax.set_title(title)
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    ax.grid(True, alpha=0.3)
    if len(uniq) <= 10:
        ax.legend(fontsize=7, markerscale=1.5)
    fig.tight_layout()
    return fig


def plot_missingness_heatmap(df: pd.DataFrame, participant_id: str) -> plt.Figure:
    """Quick visual of which modalities went missing day-by-day for one participant."""
    sub = df[df["participant_id"] == participant_id].sort_values("date")
    cols = ["missing_wearable_flag", "missing_phone_flag", "missing_survey_flag"]
    cols = [c for c in cols if c in sub.columns]
    if not cols:
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.text(0.5, 0.5, "no missingness flags present", ha="center", va="center")
        ax.axis("off")
        return fig

    mat = sub[cols].values.T  # rows = modality, cols = day
    fig, ax = plt.subplots(figsize=(9, 1 + 0.4 * len(cols)))
    ax.imshow(mat, aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels([c.replace("missing_", "").replace("_flag", "") for c in cols])
    ax.set_xlabel("day index")
    ax.set_title(f"missingness pattern: {participant_id}")
    fig.tight_layout()
    return fig
