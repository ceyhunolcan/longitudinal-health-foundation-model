"""60-second demo: synthetic cohort → features → tiny model → AUROC.

Single command, single screen of output, ends with bootstrap-CI AUROC for
the four downstream tasks. Designed to fit in a tweet/screenshot.

Run::

    make demo

or directly::

    python scripts/demo.py

Tunables (all optional; defaults are chosen so the full run fits in ~60s
on a 2020-era laptop CPU):

    --participants  60         small enough for fast SSL, big enough for real CIs
    --days          60         long enough that the 14-day window has headroom
    --ssl-epochs    3          tiny SSL pretrain
    --epochs        5          tiny downstream
    --seed          7
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _hr(title: str = "") -> None:
    """One-line section separator."""
    print()
    if title:
        print(f"── {title} " + "─" * max(0, 70 - len(title) - 4))
    else:
        print("─" * 72)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--participants", type=int, default=60)
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--ssl-epochs", type=int, default=3)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-ssl", action="store_true",
                   help="skip SSL pretrain, train downstream only (faster)")
    args = p.parse_args()

    print("LHFM 60-second demo")
    print(f"  config: {args.participants} participants x {args.days} days, "
          f"seed={args.seed}")
    print(f"          ssl_epochs={args.ssl_epochs}, downstream_epochs={args.epochs}")
    t0 = time.time()

    # ---- generate ------------------------------------------------------
    _hr("1/4  generate synthetic cohort")
    from lhfm.data.synthetic_generator import generate_synthetic_cohort
    raw = generate_synthetic_cohort(
        n_participants=args.participants,
        n_days=args.days,
        seed=args.seed,
    )
    print(f"  {len(raw)} rows, {raw['participant_id'].nunique()} participants")
    print("  modalities present: wearable + smartphone + EMA + climate")
    print(f"  elapsed: {time.time() - t0:.1f}s")

    # ---- features ------------------------------------------------------
    _hr("2/4  engineer features")
    from lhfm.features import build_full_feature_table
    feat = build_full_feature_table(raw, impute=True, add_targets=True)
    target_cols = [c for c in feat.columns if c.startswith("target_")]
    print(f"  feature table: {feat.shape[0]} rows x {feat.shape[1]} columns")
    print("  task          positive_rate  n_labelled")
    for t in target_cols:
        vals = feat[t].dropna()
        print(f"    {t:30s} {vals.mean():>6.3f}    {len(vals):>5d}")
    print(f"  elapsed: {time.time() - t0:.1f}s")

    # ---- train ---------------------------------------------------------
    _hr("3/4  train tiny SSL + downstream (CPU)")
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  torch not installed; skipping training step.")
        print("  pip install torch (or `make install-cpu-torch`) and retry.")
        return 0

    import numpy as np

    from lhfm.data.preprocessing import (
        build_windows,
        train_val_test_split_by_participant,
    )
    from lhfm.features.baseline_features import (
        compute_baseline_features,
        fit_baseline_reference_stats,
    )
    from lhfm.models.encoder import MultimodalLongitudinalEncoder
    from lhfm.training.dataset import LongitudinalWindowDataset
    from lhfm.training.train_downstream import train_downstream
    from lhfm.training.train_ssl import pretrain_ssl
    from lhfm.utils.config import resolve_device, set_global_seed

    set_global_seed(args.seed)
    device = resolve_device("cpu")

    # Re-standardize age against the train split only (no leakage).
    splits = train_val_test_split_by_participant(
        feat, val_fraction=0.15, test_fraction=0.20, seed=args.seed,
    )
    ref_stats = fit_baseline_reference_stats(splits["train"])
    for k in ("train", "val", "test"):
        splits[k] = compute_baseline_features(
            splits[k],
            age_ref_mean=ref_stats["age_ref_mean"],
            age_ref_std=ref_stats["age_ref_std"],
        )

    # Feature columns: everything numeric except IDs, dates, targets, and metadata.
    drop = {"participant_id", "date", "sex", "chronotype", "race_ethnicity",
            "ses_proxy", "region", "device_gen", "cycle_phase"} | set(target_cols)
    feature_cols = [c for c in feat.columns if c not in drop and feat[c].dtype.kind in "fi"]

    # Modality slices: keep it simple, single "all" modality for this tiny demo.
    modality_slices = {"all": (0, len(feature_cols))}
    modality_dims = {"all": len(feature_cols)}

    def _build(split_df, primary_target):
        X, _, pids, end_dates = build_windows(
            split_df, feature_cols=feature_cols,
            target_col=primary_target,
            window_days=14, stride=1, target_mode="next_day",
        )
        if X.shape[0] == 0:
            return None
        target_dates = (
            __import__("pandas").to_datetime(end_dates)
            + __import__("pandas").Timedelta(days=1)
        )
        long = split_df[["participant_id", "date", *target_cols]].copy()
        long["date"] = __import__("pandas").to_datetime(long["date"])
        long = long.set_index(["participant_id", "date"])
        Y = np.full((X.shape[0], len(target_cols)), np.nan, dtype=np.float32)
        for j, key in enumerate(zip(pids.tolist(), target_dates, strict=False)):
            try:
                row = long.loc[key]
                for i, col in enumerate(target_cols):
                    v = row[col]
                    Y[j, i] = float(v) if not __import__("pandas").isna(v) else np.nan
            except KeyError:
                pass
        return X, Y, pids

    train_data = _build(splits["train"], target_cols[0])
    val_data = _build(splits["val"], target_cols[0])
    test_data = _build(splits["test"], target_cols[0])
    if train_data is None or test_data is None:
        print("  not enough data to train; try --participants 80 --days 75")
        return 1

    Xtr, Ytr, pids_tr = train_data
    Xva, Yva, pids_va = val_data if val_data else (Xtr[:1], Ytr[:1], pids_tr[:1])
    Xte, Yte, pids_te = test_data
    print(f"  windows: train={Xtr.shape[0]}, val={Xva.shape[0]}, test={Xte.shape[0]}")

    all_pids = sorted(feat["participant_id"].unique())
    pid_to_idx = {p: i for i, p in enumerate(all_pids)}
    idx_tr = np.array([pid_to_idx[p] for p in pids_tr], dtype=np.int64)
    idx_va = np.array([pid_to_idx[p] for p in pids_va], dtype=np.int64)
    idx_te = np.array([pid_to_idx[p] for p in pids_te], dtype=np.int64)

    train_ds = LongitudinalWindowDataset(Xtr, Ytr, modality_slices, participant_idx=idx_tr)
    val_ds = LongitudinalWindowDataset(Xva, Yva, modality_slices, participant_idx=idx_va)
    test_ds = LongitudinalWindowDataset(Xte, Yte, modality_slices, participant_idx=idx_te)

    encoder = MultimodalLongitudinalEncoder(
        modality_dims=modality_dims,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_seq_len=14,
        n_participants=len(all_pids),
    )

    if not args.no_ssl:
        print(f"  SSL pretrain ({args.ssl_epochs} epochs)...", flush=True)
        pretrain_ssl(
            encoder=encoder,
            reconstruction_target_modality="all",
            train_dataset=train_ds, val_dataset=val_ds,
            epochs=args.ssl_epochs, batch_size=32, lr=3e-4,
            weight_decay=0.0, device=device,
            checkpoint_path=None,
        )
    else:
        print("  (skipping SSL)")

    print(f"  downstream ({args.epochs} epochs)...", flush=True)
    state = train_downstream(
        encoder=encoder, task_names=[c.replace("target_", "") for c in target_cols],
        train_dataset=train_ds, val_dataset=val_ds,
        epochs=args.epochs, batch_size=32, lr=5e-4,
        weight_decay=1e-5, device=device,
        freeze_encoder=False, checkpoint_path=None,
        early_stopping_patience=10,
    )
    print(f"  elapsed: {time.time() - t0:.1f}s")

    # ---- eval ----------------------------------------------------------
    _hr("4/4  evaluate (participant-clustered bootstrap CI)")
    from lhfm.training.evaluate import evaluate_downstream
    results = evaluate_downstream(
        state.model,
        test_ds,
        task_names=[c.replace("target_", "") for c in target_cols],
        device=device,
        batch_size=32,
        bootstrap_resamples=300,
    )
    print("  task              AUROC      95% CI            AUPRC   ECE     n_test")
    for task, r in results.items():
        ci = r.get("auroc_ci", (float("nan"), float("nan")))
        print(f"    {task:18s} {r['auroc']:.3f}    "
              f"[{ci[0]:.3f}, {ci[1]:.3f}]    "
              f"{r['auprc']:.3f}   {r['ece']:.3f}   {r['n_total']}")

    elapsed = time.time() - t0
    _hr()
    print(f"  total elapsed: {elapsed:.1f}s")
    print()
    print("  Next steps:")
    print("    streamlit run src/lhfm/dashboard/app.py     # explore predictions visually")
    print("    make fairness-audit                         # subgroup-stratified AUROC")
    print("    make climate-holdout                        # heat-wave generalization")
    print()
    print("  For real data, see docs/adapters/lifesnaps.md and docs/adapters/globem.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
