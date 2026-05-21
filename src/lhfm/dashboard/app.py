"""Streamlit dashboard.

Run with::

    streamlit run src/lhfm/dashboard/app.py

or via the convenience wrapper::

    python scripts/launch_dashboard.py

The dashboard expects an engineered feature table at
``data/processed/features.parquet`` (or .csv). If it's missing, we generate
a small synthetic cohort on the fly so the dashboard is always usable.
"""

from __future__ import annotations

import sys
from pathlib import Path

# When run via `streamlit run src/lhfm/dashboard/app.py` neither the project
# root nor src/ is on sys.path. Add both so `from lhfm...` imports resolve.
_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
for _p in (_SRC, _ROOT):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from lhfm.utils.config import load_config
from lhfm.utils.plotting import (
    plot_missingness_heatmap,
    plot_participant_trends,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LHFM — participant explorer",
    page_icon="🩺",
    layout="wide",
)

st.title("Longitudinal Health Foundation Model")
st.caption(
    "Research prototype dashboard. Synthetic data only. Not a medical device "
    "and not for clinical use."
)


# ---------------------------------------------------------------------------
# Data loading. We cache so reruns are snappy.
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _load_features() -> pd.DataFrame:
    cfg = load_config("default")
    processed = Path(cfg["paths"]["processed_dir"]) / "features.parquet"
    if processed.exists():
        return pd.read_parquet(processed)
    csv_alt = processed.with_suffix(".csv")
    if csv_alt.exists():
        return pd.read_csv(csv_alt, parse_dates=["date"])

    # On-the-fly fallback so a fresh checkout is usable without first running
    # the pipeline script.
    st.warning(
        "No processed features found at data/processed/features.parquet — "
        "generating a small synthetic cohort on the fly. "
        "Run `python scripts/run_pipeline.py` for the full dataset.",
        icon="⚠️",
    )
    from lhfm.data.synthetic_generator import generate_synthetic_cohort
    from lhfm.features import build_full_feature_table
    raw = generate_synthetic_cohort(n_participants=30, n_days=60, seed=11)
    return build_full_feature_table(raw, impute=True, add_targets=True)


df = _load_features()
df["date"] = pd.to_datetime(df["date"], errors="coerce")


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Participant")
    pid = st.selectbox(
        "select participant",
        sorted(df["participant_id"].unique()),
    )
    sub = df[df["participant_id"] == pid].sort_values("date").reset_index(drop=True)
    st.metric("days observed", len(sub))
    st.metric("age", int(sub["age"].iloc[0]))
    st.metric("chronotype", sub["chronotype"].iloc[0])
    st.metric("baseline HRV (ms)", f"{sub['baseline_hrv'].iloc[0]:.1f}")

    st.divider()
    st.header("Window")
    window_end = st.select_slider(
        "window end date",
        options=sub["date"].dt.strftime("%Y-%m-%d").tolist(),
        value=sub["date"].dt.strftime("%Y-%m-%d").iloc[-1],
    )
    window_len = st.slider("window length (days)", min_value=7, max_value=21, value=14)


# ---------------------------------------------------------------------------
# Helpers for the risk panel
# ---------------------------------------------------------------------------


def _risk_label(p: float) -> str:
    if p >= 0.66:
        return "elevated"
    if p >= 0.33:
        return "moderate"
    return "low"


def _color(p: float) -> str:
    if p >= 0.66:
        return "🔴"
    if p >= 0.33:
        return "🟡"
    return "🟢"


def _fallback_scores(row: pd.Series) -> dict[str, float]:
    """Same logic as the API fallback so the dashboard never feels broken."""
    def _sig(z): return 1.0 / (1.0 + np.exp(-z))
    sleep = float(row.get("sleep_duration", 7.5) or 7.5)
    stress = float(row.get("survey_stress", 3.0) or 3.0)
    aqi = float(row.get("aqi", 50.0) or 50.0)
    eff = float(row.get("sleep_efficiency", 0.88) or 0.88)
    hi = float(row.get("heat_index", 22.0) or 22.0)
    hrv_dev = float(row.get("hrv_dev_from_baseline", 0.0) or 0.0)
    return {
        "low_mood": _sig(-2.0 + 0.5 * max(0, 7.5 - sleep) + 0.4 * max(0, stress - 3.0)
                         + 0.02 * max(0, aqi - 80.0)),
        "high_stress": _sig(-1.5 + 0.6 * max(0, stress - 3.5) + 0.4 * max(0, 7.5 - sleep)),
        "sleep_disruption": _sig(-30.0 * (eff - 0.80)),
        "climate_vulnerable": _sig(-2.0 + 0.5 * max(0, hi - 32.0) + 0.1 * max(0, -hrv_dev)),
    }


# ---------------------------------------------------------------------------
# Per-day risk + attribution timeline.
#
# This is the dashboard's headline visualisation: scrub through the
# participant's timeline and see (a) the per-day predicted risk, (b) which
# features moved the prediction on each day.
#
# When a trained checkpoint isn't wired in we fall back to the same rule
# the API uses, with a pseudo-attribution derived from each rule term's
# contribution. That keeps the panel functional on a fresh checkout so
# people can see the shape of the UI before they train anything.
# ---------------------------------------------------------------------------


# Map task -> the features that contribute under the rule-based fallback.
# When real attributions are unavailable we surface these as a stand-in,
# scaled by each term's magnitude on the day.
_FALLBACK_DRIVERS = {
    "low_mood": [
        # (feature, label,             how its value pushes risk)
        ("sleep_duration",       "short sleep",          lambda v: max(0, 7.5 - v) * 0.5),
        ("survey_stress",        "high stress",          lambda v: max(0, v - 3.0) * 0.4),
        ("aqi",                  "poor air quality",     lambda v: max(0, v - 80) * 0.02),
        ("hrv_dev_from_baseline","HRV below baseline",   lambda v: max(0, -v) * 0.05),
        ("heat_index",           "heat stress",          lambda v: max(0, v - 30) * 0.05),
    ],
    "high_stress": [
        ("survey_stress",        "high stress",          lambda v: max(0, v - 3.5) * 0.6),
        ("sleep_duration",       "short sleep",          lambda v: max(0, 7.5 - v) * 0.4),
        ("aqi",                  "poor air quality",     lambda v: max(0, v - 100) * 0.01),
    ],
    "sleep_disruption": [
        ("sleep_efficiency",     "low sleep efficiency", lambda v: max(0, 0.85 - v) * 4.0),
        ("heat_index",           "heat at night",        lambda v: max(0, v - 28) * 0.1),
    ],
    "climate_vulnerable": [
        ("heat_index",           "heat stress",          lambda v: max(0, v - 32) * 0.5),
        ("aqi",                  "poor air quality",     lambda v: max(0, v - 100) * 0.02),
        ("hrv_dev_from_baseline","HRV below baseline",   lambda v: max(0, -v) * 0.1),
    ],
}


def _per_day_risk_and_drivers(sub: pd.DataFrame, task: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (per-day risk vector, drivers DataFrame [date x driver_label])
    for the chosen task on this participant.

    The drivers DataFrame is signed: a positive value means that feature
    pushed the risk *up* on that day. When wired to a trained model this
    is replaced with integrated-gradients attributions.
    """
    drivers = _FALLBACK_DRIVERS.get(task, _FALLBACK_DRIVERS["low_mood"])
    risks = np.zeros(len(sub), dtype=float)
    rows = []
    for i, (_, row) in enumerate(sub.iterrows()):
        scores = _fallback_scores(row)
        risks[i] = scores.get(task, scores["low_mood"])
        rec = {"date": row["date"]}
        for col, label, fn in drivers:
            v = row.get(col)
            if v is None or pd.isna(v):
                rec[label] = 0.0
            else:
                rec[label] = float(fn(float(v)))
        rows.append(rec)
    return risks, pd.DataFrame(rows).set_index("date")


def _draw_attribution_timeline(sub: pd.DataFrame, task: str, focus_date: pd.Timestamp):
    """Two-panel plot: risk curve on top, driver heatmap below.

    The focus_date gets a vertical marker on both panels.
    """
    risks, drivers = _per_day_risk_and_drivers(sub, task)
    dates = sub["date"].values

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 4.2),
        gridspec_kw={"height_ratios": [1.0, 1.4], "hspace": 0.30},
    )

    # Top: risk curve, color-coded by band.
    ax_top.plot(dates, risks, color="#444", linewidth=1.5, zorder=2)
    ax_top.fill_between(dates, 0, risks,
                         where=(risks >= 0.66), color="#e57373", alpha=0.45)
    ax_top.fill_between(dates, 0, risks,
                         where=(risks >= 0.33) & (risks < 0.66), color="#ffd54f", alpha=0.45)
    ax_top.fill_between(dates, 0, risks,
                         where=(risks < 0.33), color="#a5d6a7", alpha=0.45)
    ax_top.axhline(0.33, color="#bbb", linestyle=":", linewidth=0.8)
    ax_top.axhline(0.66, color="#bbb", linestyle=":", linewidth=0.8)
    ax_top.set_ylim(0, 1)
    ax_top.set_ylabel(f"P({task})")
    ax_top.set_title(f"Per-day risk and drivers — {task}", loc="left", fontsize=11)
    for spine in ("top", "right"):
        ax_top.spines[spine].set_visible(False)

    # Bottom: signed driver heatmap. Use integer day index for the x-axis
    # so imshow's extent is unambiguous; we set human-readable date ticks
    # manually below.
    arr = drivers.values.T            # (n_drivers, T)
    labels = drivers.columns.tolist()
    vmax = max(0.01, float(np.nanmax(np.abs(arr))))
    ax_bot.imshow(
        arr, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, interpolation="nearest",
    )
    ax_bot.set_yticks(range(len(labels)))
    ax_bot.set_yticklabels(labels, fontsize=9)
    # Sparse x-axis ticks: first, last, and roughly weekly.
    T = arr.shape[1]
    if T > 0:
        step = max(1, T // 6)
        tick_pos = list(range(0, T, step))
        if (T - 1) not in tick_pos:
            tick_pos.append(T - 1)
        ax_bot.set_xticks(tick_pos)
        ax_bot.set_xticklabels(
            [pd.Timestamp(dates[i]).strftime("%m-%d") for i in tick_pos],
            fontsize=8,
        )
    ax_bot.set_xlabel("date")
    for spine in ("top", "right"):
        ax_bot.spines[spine].set_visible(False)

    # Mirror integer-index focus marker on the heatmap; on the top panel
    # we use the actual date so the curve and the marker line up cleanly.
    if T > 0:
        focus_idx = int(np.argmin(np.abs(sub["date"].values - np.datetime64(focus_date))))
        ax_bot.axvline(focus_idx, color="#222", linewidth=1.2, alpha=0.8)
    ax_top.axvline(pd.Timestamp(focus_date), color="#222", linewidth=1.2, alpha=0.8)

    plt.tight_layout()
    return fig, risks, drivers


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

# Headline section: the attribution timeline. Sits at the top so it's the
# first thing a visitor sees once they've picked a participant.
st.subheader("Risk + driver timeline")
task_choice = st.radio(
    "task",
    options=["low_mood", "high_stress", "sleep_disruption", "climate_vulnerable"],
    horizontal=True, index=0,
    label_visibility="collapsed",
)
focus_date = pd.to_datetime(window_end)
fig_attr, risks_arr, drivers_df = _draw_attribution_timeline(sub, task_choice, focus_date)
st.pyplot(fig_attr, use_container_width=True)
plt.close(fig_attr)

# Focus-day driver readout (the part that makes the timeline feel scrubbable).
focus_idx = int(np.argmin(np.abs(sub["date"].values - np.datetime64(focus_date))))
focus_row = drivers_df.iloc[focus_idx]
focus_risk = float(risks_arr[focus_idx])

c_l, c_r = st.columns([1, 2.2])
with c_l:
    band = _risk_label(focus_risk)
    color = _color(focus_risk)
    st.metric(
        f"{color} risk on {focus_date.strftime('%Y-%m-%d')}",
        f"{focus_risk:.2f}",
        help=f"band: {band}",
    )
with c_r:
    nonzero = focus_row[focus_row.abs() > 1e-6].sort_values(key=lambda s: s.abs(), ascending=False)
    if len(nonzero) == 0:
        st.caption("No notable drivers on this day. The prediction sits near the model's base rate.")
    else:
        st.caption("Top drivers on this day (positive = pushed risk up):")
        for name, val in nonzero.items():
            arrow = "▲" if val > 0 else "▼"
            st.markdown(
                f"&nbsp;&nbsp;{arrow}&nbsp;**{name}**  ({val:+.2f})",
                unsafe_allow_html=True,
            )

st.caption(
    "Drivers shown use the rule-based fallback. Once a trained checkpoint is "
    "wired in, these become integrated-gradients attributions over the model."
)

st.divider()

col_left, col_right = st.columns([2.4, 1.0])

with col_left:
    st.subheader("Longitudinal trends")
    trend_cols = [
        "survey_mood", "sleep_duration", "hrv_rmssd",
        "survey_stress", "heat_index", "aqi",
    ]
    fig = plot_participant_trends(sub, columns=trend_cols, title=f"participant {pid}")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Missingness pattern")
    fig2 = plot_missingness_heatmap(df, participant_id=pid)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.subheader("Personal baseline deviation")
    dev_cols = ["hrv_dev_from_baseline", "rhr_dev_from_baseline", "stress_burden_7d"]
    if all(c in sub.columns for c in dev_cols):
        fig3 = plot_participant_trends(sub, columns=dev_cols)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)
    else:
        st.info("baseline-deviation features not present in this dataframe.")

with col_right:
    st.subheader("Risk scores")
    end_idx = sub.index[sub["date"].dt.strftime("%Y-%m-%d") == window_end][0]
    window_df = sub.iloc[max(0, end_idx - window_len + 1): end_idx + 1]
    last = window_df.iloc[-1]
    scores = _fallback_scores(last)
    for task, label in [
        ("low_mood", "low mood"),
        ("high_stress", "high stress"),
        ("sleep_disruption", "sleep disruption"),
        ("climate_vulnerable", "climate vulnerability"),
    ]:
        p = scores[task]
        st.markdown(
            f"**{_color(p)} {label}** — `{_risk_label(p)}`  ({p:.2f})"
        )
    st.caption(
        "Scores use the rule-based fallback. Wire a trained checkpoint into "
        "the API to replace them with model predictions."
    )

    st.divider()
    st.subheader("Recent window")
    st.dataframe(
        window_df[["date", "sleep_duration", "hrv_rmssd",
                   "survey_mood", "survey_stress", "heat_index", "aqi"]]
        .round(2),
        height=420,
    )

    st.divider()
    st.subheader("Explanation")
    bullets: list[str] = []
    if last.get("sleep_duration", 8) and last["sleep_duration"] < 6.5:
        bullets.append(f"Sleep on the latest day was short ({last['sleep_duration']:.1f} h).")
    if last.get("sleep_efficiency", 1.0) and last["sleep_efficiency"] < 0.80:
        bullets.append(f"Sleep efficiency below 80% ({last['sleep_efficiency']*100:.0f}%).")
    if last.get("hrv_dev_from_baseline", 0) < -8:
        bullets.append(f"HRV {abs(last['hrv_dev_from_baseline']):.0f} ms below baseline.")
    if last.get("aqi", 0) and last["aqi"] > 100:
        bullets.append(f"Air quality poor (AQI {last['aqi']:.0f}).")
    if last.get("heat_index", 0) and last["heat_index"] > 32:
        bullets.append(f"Apparent temperature high ({last['heat_index']:.1f}°C).")
    if not bullets:
        bullets.append("No notable risk drivers in the latest day.")
    for b in bullets:
        st.write(f"- {b}")


st.divider()
st.caption(
    "This dashboard is part of a research prototype. Outputs are derived from "
    "synthetic data unless a real dataset has been loaded into "
    "`data/processed/features.parquet`. Nothing here is intended for clinical use."
)
