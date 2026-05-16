"""Subgroup-stratified fairness audit on the held-out test split.

Usage::

    python scripts/run_fairness_audit.py
    python scripts/run_fairness_audit.py --checkpoint checkpoints/downstream-ema-blind.pt
    python scripts/run_fairness_audit.py --max-auroc-gap 0.05 --fail-on-violation

Outputs:
    results/tables/fairness_<task>.csv             per-subgroup rows
    results/tables/fairness_<task>_equalised_odds.csv
    results/tables/fairness_summary.json           CI-friendly summary

The synthetic generator does not bake in subgroup disparities, so on
synthetic data this script should report **no** material violations. The
audit exists primarily for the moment a real cohort is plugged in --
running it now guarantees the pipeline works.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lhfm.data.preprocessing import build_windows, train_val_test_split_by_participant  # noqa: E402
from lhfm.features.baseline_features import compute_baseline_features as _cbf  # noqa: E402
from lhfm.training.dataset import LongitudinalWindowDataset  # noqa: E402
from lhfm.training.evaluate import evaluate_downstream  # noqa: E402
from lhfm.utils.config import load_config, resolve_device, set_global_seed  # noqa: E402
from lhfm.utils.fairness import (  # noqa: E402
    check_fairness_thresholds,
    fairness_report_to_csv,
    run_fairness_audit,
)
from lhfm.utils.logging import get_logger  # noqa: E402


log = get_logger("fairness_audit")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--features", type=str, default=None)
    p.add_argument("--split", choices=["test", "val", "train"], default="test")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-subgroup-n", type=int, default=30)
    p.add_argument("--bootstrap-resamples", type=int, default=500)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--max-auroc-gap", type=float, default=0.10,
                   help="fail when any axis has AUROC spread > this")
    p.add_argument("--max-eo-violation", type=float, default=0.20,
                   help="fail when any axis has FPR+FNR drift > this")
    p.add_argument("--fail-on-violation", action="store_true",
                   help="exit non-zero if thresholds are violated (for CI gating)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config("default", "model", "features")
    set_global_seed(cfg["project"]["seed"])
    device = resolve_device(args.device or cfg["training"]["device"])

    ckpt_path = Path(args.checkpoint) if args.checkpoint else (
        Path(cfg["paths"]["checkpoint_dir"]) / "downstream.pt"
    )
    meta_path = ckpt_path.with_suffix(".meta.json")
    if not ckpt_path.exists() or not meta_path.exists():
        log.error("checkpoint or meta missing: %s, %s", ckpt_path, meta_path)
        return 2
    meta = json.loads(meta_path.read_text())
    log.info("auditing %s (run_tag=%s)", ckpt_path.name, meta.get("run_tag", "?"))

    # Rebuild model.
    import torch
    from lhfm.models.encoder import MultimodalLongitudinalEncoder
    from lhfm.models.downstream import DownstreamRiskModel
    encoder = MultimodalLongitudinalEncoder(
        modality_dims={k: int(v) for k, v in meta["modality_dims"].items()},
        d_model=int(meta["d_model"]),
        n_heads=int(meta["n_heads"]),
        n_layers=int(meta["n_layers"]),
        max_seq_len=int(meta["max_seq_len"]),
        n_participants=int(meta.get("n_participants", 0)),
    )
    model = DownstreamRiskModel(encoder=encoder, task_names=meta["task_names"])
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    # Load features.
    if args.features:
        feat_path = Path(args.features)
    else:
        feat_path = Path(cfg["paths"]["processed_dir"]) / "features.parquet"
        if not feat_path.exists():
            feat_path = feat_path.with_suffix(".csv")
    if not feat_path.exists():
        log.error("no features at %s", feat_path)
        return 2
    df = (
        pd.read_parquet(feat_path) if feat_path.suffix == ".parquet"
        else pd.read_csv(feat_path, parse_dates=["date"])
    )
    df["date"] = pd.to_datetime(df["date"])

    # The fairness audit needs subgroup metadata. If it's missing (e.g.
    # someone loaded a v1 features file), regenerate from the raw cohort.
    needed = {"race_ethnicity", "ses_proxy", "region", "device_gen",
              "has_anxiety", "has_depression"}
    if not needed.issubset(df.columns):
        log.warning(
            "features file lacks subgroup metadata (%s); the audit will still "
            "run but won't slice on those axes. Re-run scripts/run_pipeline.py "
            "to regenerate with the v2 generator.",
            sorted(needed - set(df.columns)),
        )

    # Re-apply age standardization with the training-time reference stats.
    df = _cbf(
        df,
        age_ref_mean=meta.get("age_ref_mean"),
        age_ref_std=meta.get("age_ref_std"),
    )
    window_days = int(meta.get("window_days", cfg["training"]["window_days"]))
    pid_lengths = df.groupby("participant_id")["date"].count()
    df = df[df["participant_id"].isin(pid_lengths[pid_lengths >= window_days + 1].index)]

    splits = train_val_test_split_by_participant(
        df, val_fraction=cfg["training"]["val_fraction"],
        test_fraction=cfg["training"]["test_fraction"],
        seed=cfg["project"]["seed"],
    )
    sub = splits[args.split]
    log.info("auditing on split=%s (%d rows, %d participants)",
             args.split, len(sub), sub["participant_id"].nunique())

    TASK_TO_COLUMN = {
        "low_mood": "target_low_mood",
        "high_stress": "target_high_stress",
        "sleep_disruption": "target_sleep_disruption",
        "climate_vulnerable": "target_climate_vulnerable",
    }
    feature_cols = list(meta["feature_columns"])
    modality_slices = {k: tuple(v) for k, v in meta["modality_slices"].items()}
    task_names = meta["task_names"]

    X, _, pids, end_dates = build_windows(
        sub, feature_cols=feature_cols,
        target_col=TASK_TO_COLUMN[task_names[0]],
        window_days=window_days, stride=1, target_mode="next_day",
    )
    if X.shape[0] == 0:
        log.error("no windows in split=%s; can't audit", args.split)
        return 2

    # Join labels.
    target_dates = pd.to_datetime(end_dates) + pd.Timedelta(days=1)
    target_cols = [TASK_TO_COLUMN[t] for t in task_names]
    long = sub[["participant_id", "date", *target_cols]].copy()
    long["date"] = pd.to_datetime(long["date"])
    long = long.set_index(["participant_id", "date"])
    Y = np.full((X.shape[0], len(task_names)), np.nan, dtype=np.float32)
    for j, key in enumerate(zip(pids.tolist(), target_dates)):
        try:
            row = long.loc[key]
            for i, col in enumerate(target_cols):
                v = row[col]
                Y[j, i] = float(v) if not pd.isna(v) else np.nan
        except KeyError:
            pass

    # One row of metadata per window. We attach the *window's last day's*
    # participant-level attributes — they're slowly-varying so this is fine.
    md_cols = [c for c in (
        "sex", "race_ethnicity", "ses_proxy", "region", "device_gen",
        "has_anxiety", "has_depression", "age",
    ) if c in sub.columns]
    long_md = sub[["participant_id", "date", *md_cols]].copy()
    long_md["date"] = pd.to_datetime(long_md["date"])
    long_md = long_md.set_index(["participant_id", "date"])
    rows_meta = []
    end_dates_pd = pd.to_datetime(end_dates)
    for pid_i, edate in zip(pids.tolist(), end_dates_pd):
        try:
            rows_meta.append(long_md.loc[(pid_i, edate)].to_dict())
        except KeyError:
            rows_meta.append({c: None for c in md_cols})
    metadata = pd.DataFrame(rows_meta)

    pid_to_idx = {p: i for i, p in enumerate(sorted(df["participant_id"].unique()))}
    idx = np.array([pid_to_idx[p] for p in pids], dtype=np.int64)
    ds = LongitudinalWindowDataset(X, Y, modality_slices, participant_idx=idx)
    # Get predictions per-task in one pass.
    results = evaluate_downstream(
        model, ds, task_names=task_names, device=device,
        batch_size=cfg["training"]["batch_size"],
        bootstrap_resamples=200,   # smaller; per-task audit does its own bootstrap
    )

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(cfg["paths"]["results_dir"]) / "tables"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"audited_checkpoint": str(ckpt_path),
                     "split": args.split, "threshold": args.threshold,
                     "tasks": {}}

    any_violation = False
    for task in task_names:
        r = results[task]
        # The raw arrays are only populated when both classes are present.
        if "y_true" not in r:
            log.info("task %s has only one class on this split; skipping fairness audit", task)
            continue
        yt = np.array(r["y_true"])
        yp = np.array(r["y_prob"])
        # Subset metadata + groups to valid rows (same trick evaluate_downstream uses).
        # Build the boolean valid mask the same way: we masked y on entry.
        # The per-row metadata + groups still match in length.
        # But: evaluate_downstream subsamples to "valid" inside. Reconstruct here.
        valid = ~np.isnan(Y[:, task_names.index(task)])
        md = metadata.iloc[valid].reset_index(drop=True)
        groups_v = idx[valid]
        audit = run_fairness_audit(
            yt, yp, metadata=md,
            threshold=args.threshold,
            min_subgroup_n=args.min_subgroup_n,
            bootstrap_resamples=args.bootstrap_resamples,
            groups=groups_v,
        )
        fairness_report_to_csv(audit, out_dir / f"fairness_{task}.csv")
        log.info("wrote %s (%d subgroup rows)",
                 out_dir / f"fairness_{task}.csv", len(audit["per_subgroup"]))

        ok, violations = check_fairness_thresholds(
            audit,
            max_auroc_gap=args.max_auroc_gap,
            max_eo_violation=args.max_eo_violation,
        )
        if not ok:
            any_violation = True
            for v in violations:
                log.warning("[%s] %s", task, v)

        summary["tasks"][task] = {
            "n_subgroups_audited": len(audit["per_subgroup"]),
            "n_subgroups_skipped_small": len(audit["small_subgroups"]),
            "ok": ok,
            "violations": violations,
            "equalised_odds_gap": audit["equalised_odds_gap"],
        }

    (out_dir / "fairness_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", out_dir / "fairness_summary.json")

    if args.fail_on_violation and any_violation:
        log.error("fairness violations exceeded thresholds; failing")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
