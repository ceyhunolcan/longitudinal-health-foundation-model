"""Pretraining-scale ablation: how does downstream AUROC grow with cohort size?

This is the figure reviewers ask about first when the word "foundation
model" appears in an abstract. The script trains the SSL encoder at
multiple cohort sizes, fine-tunes the downstream heads at each size, and
emits:

    results/tables/scale_ablation.csv    raw numbers
    results/figures/scale_ablation.png   the canonical figure

Each row is one (cohort_size, task) point with 95% participant-clustered
bootstrap CIs already computed. The CSV is enough for an external plotter
(matplotlib in a notebook, ggplot in R, etc.) if our default plot isn't
to your taste.

Usage::

    python scripts/run_scale_ablation.py                       # default sizes
    python scripts/run_scale_ablation.py --sizes 25 50 100 200 # custom
    python scripts/run_scale_ablation.py --seeds 3             # average over 3 seeds
    python scripts/run_scale_ablation.py --downstream-epochs 5 --ssl-epochs 5  # faster

The script generates fresh synthetic data at each cohort size (since this
is the natural way to vary cohort size in the prototype). With a real-data
adapter, you'd instead subsample from a fixed pool; the script's --feature-
file argument supports that path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import functools
import operator

from lhfm.data.preprocessing import (
    build_windows,
    train_val_test_split_by_participant,
)
from lhfm.data.synthetic_generator import generate_synthetic_cohort
from lhfm.features import build_full_feature_table
from lhfm.features.baseline_features import (
    compute_baseline_features as _cbf,
)
from lhfm.features.baseline_features import (
    fit_baseline_reference_stats,
)
from lhfm.training.dataset import LongitudinalWindowDataset
from lhfm.training.evaluate import evaluate_downstream
from lhfm.utils.config import load_config, resolve_device, set_global_seed
from lhfm.utils.logging import get_logger

log = get_logger("scale_ablation")


FEATURE_GROUPS = {
    "wearable": [
        "sleep_duration", "sleep_efficiency", "sleep_regularity_index",
        "hrv_dev_from_baseline", "rhr_dev_from_baseline", "stress_burden_7d",
    ],
    "smartphone": [
        "screen_time_z", "unlock_freq_z", "mobility_radius_km",
        "location_entropy", "behavioral_regularity",
    ],
    "climate": [
        "temperature_c", "heat_index", "aqi",
        "humid_heat_index", "nighttime_heat_stress",
    ],
    "baseline": [
        "age_z", "chronotype_score", "baseline_hrv", "baseline_sleep_need",
    ],
}
TASK_TO_COLUMN = {
    "low_mood": "target_low_mood",
    "high_stress": "target_high_stress",
    "sleep_disruption": "target_sleep_disruption",
    "climate_vulnerable": "target_climate_vulnerable",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--sizes", type=int, nargs="+",
        default=[20, 50, 100, 200, 400],
        help="cohort sizes (number of participants) to sweep over",
    )
    p.add_argument(
        "--days", type=int, default=60,
        help="days per participant in the synthetic cohort",
    )
    p.add_argument("--seeds", type=int, default=2,
                   help="how many seeds per cohort size to average")
    p.add_argument("--ssl-epochs", type=int, default=5)
    p.add_argument("--downstream-epochs", type=int, default=10)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-pretrain", action="store_true",
                   help="skip SSL pretraining at each size (probes whether SSL helps at all)")
    p.add_argument("--out-csv", type=str, default=None)
    p.add_argument("--out-figure", type=str, default=None)
    return p.parse_args()


def _make_cohort_dataset(n_participants: int, n_days: int, seed: int, cfg: dict):
    """Build train/val/test datasets at a given cohort size."""
    raw = generate_synthetic_cohort(
        n_participants=n_participants, n_days=n_days, seed=seed,
    )
    feat = build_full_feature_table(raw, impute=True, add_targets=True)
    window_days = cfg["training"]["window_days"]
    pid_lengths = feat.groupby("participant_id")["date"].count()
    feat = feat[feat["participant_id"].isin(
        pid_lengths[pid_lengths >= window_days + 1].index
    )]
    splits = train_val_test_split_by_participant(
        feat, val_fraction=cfg["training"]["val_fraction"],
        test_fraction=cfg["training"]["test_fraction"],
        seed=seed,
    )
    # Fit and reapply train-only age stats.
    ref = fit_baseline_reference_stats(splits["train"])
    for k in splits:
        splits[k] = _cbf(splits[k], age_ref_mean=ref["age_ref_mean"],
                         age_ref_std=ref["age_ref_std"])

    feature_cols = functools.reduce(operator.iadd, FEATURE_GROUPS.values(), [])
    cursor, slices = 0, {}
    for mod, cols in FEATURE_GROUPS.items():
        slices[mod] = (cursor, cursor + len(cols))
        cursor += len(cols)
    modality_dims = {m: len(cols) for m, cols in FEATURE_GROUPS.items()}
    task_names = cfg["downstream"]["tasks"]

    def build(split_df, mode="next_day"):
        X, _, pids, end_dates = build_windows(
            split_df, feature_cols=feature_cols,
            target_col=TASK_TO_COLUMN[task_names[0]],
            window_days=window_days, stride=1, target_mode=mode,
        )
        if X.shape[0] == 0:
            return X, np.zeros((0, len(task_names)), dtype=np.float32), pids
        target_dates = pd.to_datetime(end_dates) + pd.Timedelta(days=1)
        target_cols = [TASK_TO_COLUMN[t] for t in task_names]
        long = split_df[["participant_id", "date", *target_cols]].copy()
        long["date"] = pd.to_datetime(long["date"])
        long = long.set_index(["participant_id", "date"])
        Y = np.full((X.shape[0], len(task_names)), np.nan, dtype=np.float32)
        for j, key in enumerate(zip(pids.tolist(), target_dates, strict=False)):
            try:
                row = long.loc[key]
                for i, col in enumerate(target_cols):
                    val = row[col]
                    Y[j, i] = float(val) if not pd.isna(val) else np.nan
            except KeyError:
                pass
        return X, Y, pids

    Xtr, Ytr, ptr = build(splits["train"])
    Xva, Yva, pva = build(splits["val"])
    Xte, Yte, pte = build(splits["test"])

    pid_to_idx = {p: i for i, p in enumerate(sorted(feat["participant_id"].unique()))}
    n_pids = len(pid_to_idx)
    idx_tr = np.array([pid_to_idx[p] for p in ptr], dtype=np.int64)
    idx_va = np.array([pid_to_idx[p] for p in pva], dtype=np.int64)
    idx_te = np.array([pid_to_idx[p] for p in pte], dtype=np.int64)

    train_ds = LongitudinalWindowDataset(Xtr, Ytr, slices, participant_idx=idx_tr)
    val_ds = LongitudinalWindowDataset(Xva, Yva, slices, participant_idx=idx_va)
    test_ds = LongitudinalWindowDataset(Xte, Yte, slices, participant_idx=idx_te)
    return train_ds, val_ds, test_ds, modality_dims, task_names, n_pids


def _run_one(size: int, seed: int, args, cfg, device):
    """Train + evaluate at one (size, seed) point. Returns a list of rows."""
    set_global_seed(seed)
    train_ds, val_ds, test_ds, modality_dims, task_names, n_pids = _make_cohort_dataset(
        size, args.days, seed, cfg,
    )

    from lhfm.models.encoder import MultimodalLongitudinalEncoder
    from lhfm.models.self_supervised import SSLLossWeights
    from lhfm.training.train_downstream import train_downstream
    from lhfm.training.train_ssl import pretrain_ssl

    encoder_cfg = cfg["encoder"]
    encoder = MultimodalLongitudinalEncoder(
        modality_dims=modality_dims,
        d_model=encoder_cfg["d_model"],
        n_heads=encoder_cfg["n_heads"],
        n_layers=encoder_cfg["n_layers"],
        ff_dim=encoder_cfg["ff_dim"],
        dropout=encoder_cfg["dropout"],
        max_seq_len=encoder_cfg["max_seq_len"],
        n_participants=n_pids if encoder_cfg["use_participant_embedding"] else 0,
        participant_embedding_dim=encoder_cfg["participant_embedding_dim"],
    )

    if not args.no_pretrain:
        ssl_cfg = cfg["ssl"]
        pretrain_ssl(
            encoder,
            train_dataset=train_ds, val_dataset=val_ds,
            epochs=args.ssl_epochs, batch_size=cfg["training"]["batch_size"],
            lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"],
            mask_ratio=ssl_cfg["mask_ratio"],
            weights=SSLLossWeights(
                recon=ssl_cfg["recon_weight"],
                next_day=ssl_cfg["nextday_weight"],
                contrastive=ssl_cfg["contrastive_weight"],
                temperature=ssl_cfg["contrastive_temperature"],
            ),
            device=device, num_workers=cfg["training"]["num_workers"],
            checkpoint_path=None,
            early_stopping_patience=cfg["training"]["early_stopping_patience"],
        )

    state = train_downstream(
        encoder=encoder, task_names=task_names,
        train_dataset=train_ds, val_dataset=val_ds,
        epochs=args.downstream_epochs, batch_size=cfg["training"]["batch_size"],
        lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"],
        device=device, num_workers=cfg["training"]["num_workers"],
        freeze_encoder=False, checkpoint_path=None,
        early_stopping_patience=cfg["training"]["early_stopping_patience"],
    )

    results = evaluate_downstream(
        state.model, test_ds, task_names=task_names, device=device,
        batch_size=cfg["training"]["batch_size"],
        bootstrap_resamples=500, cluster_bootstrap=True,
    )

    rows = []
    for task, m in results.items():
        auroc_ci = m.get("auroc_ci", (np.nan, np.nan))
        auprc_ci = m.get("auprc_ci", (np.nan, np.nan))
        rows.append({
            "n_participants": size,
            "seed": seed,
            "task": task,
            "auroc": m.get("auroc"),
            "auroc_ci_low": auroc_ci[0],
            "auroc_ci_high": auroc_ci[1],
            "auprc": m.get("auprc"),
            "auprc_ci_low": auprc_ci[0],
            "auprc_ci_high": auprc_ci[1],
            "n_test_pos": m.get("n_pos"),
            "n_test_total": m.get("n_total"),
            "ssl_pretrained": not args.no_pretrain,
        })
    return rows


def main() -> int:
    args = parse_args()
    cfg = load_config("default", "model", "features")
    device = resolve_device(args.device or cfg["training"]["device"])
    log.info("scale ablation: sizes=%s seeds=%s device=%s",
             args.sizes, args.seeds, device)

    all_rows: list[dict] = []
    for size in args.sizes:
        for seed_idx in range(args.seeds):
            seed = 1000 * size + seed_idx
            log.info("---- size=%d seed=%d ----", size, seed)
            try:
                rows = _run_one(size, seed, args, cfg, device)
                all_rows.extend(rows)
            except Exception as exc:
                log.exception("size=%d seed=%d failed: %s", size, seed, exc)

    out_csv = Path(args.out_csv) if args.out_csv else (
        Path(cfg["paths"]["results_dir"]) / "tables" / "scale_ablation.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    log.info("wrote %s", out_csv)

    # Render the headline figure.
    out_fig = Path(args.out_figure) if args.out_figure else (
        Path(cfg["paths"]["results_dir"]) / "figures" / "scale_ablation.png"
    )
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    try:
        _render_figure(df, out_fig)
        log.info("wrote %s", out_fig)
    except Exception as exc:
        log.warning("figure rendering failed: %s", exc)

    return 0


def _render_figure(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        return

    tasks = sorted(df["task"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for task in tasks:
        sub = df[df["task"] == task].dropna(subset=["auroc"])
        if sub.empty:
            continue
        agg = sub.groupby("n_participants").agg(
            auroc_mean=("auroc", "mean"),
            auroc_lo=("auroc_ci_low", "mean"),
            auroc_hi=("auroc_ci_high", "mean"),
        ).reset_index()
        ax.plot(agg["n_participants"], agg["auroc_mean"], marker="o", label=task)
        ax.fill_between(
            agg["n_participants"], agg["auroc_lo"], agg["auroc_hi"], alpha=0.15,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Pretraining cohort size (participants)")
    ax.set_ylabel("Test AUROC")
    ax.set_title("Downstream AUROC vs. pretraining cohort size")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    fig.text(
        0.01, -0.02,
        "Bands: mean of participant-clustered 95% bootstrap CIs across seeds. "
        "Synthetic cohort. Not a clinical claim.",
        fontsize=7, color="gray",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
