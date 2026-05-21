"""Evaluate a saved checkpoint on a held-out dataset.

Usage::

    python scripts/evaluate_model.py                       # default checkpoints/ + processed/
    python scripts/evaluate_model.py --checkpoint path/to/downstream.pt
    python scripts/evaluate_model.py --features path/to/features.csv
    python scripts/evaluate_model.py --bootstrap-resamples 2000

Why this exists separately from train_model.py:
    - Re-running evaluation with different bootstrap settings doesn't need a
      training pass.
    - Lets you swap in a different (real-data) feature CSV without
      monkeypatching the training script.
    - Gives senior reviewers a clean way to verify "the metrics in
      results/tables/metrics_test.csv are actually what this checkpoint
      produces on the held-out cohort right now".
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
sys.path.insert(0, str(ROOT))

from lhfm.data.preprocessing import (
    build_windows,
    train_val_test_split_by_participant,
)
from lhfm.features.baseline_features import (
    compute_baseline_features as _cbf,
)
from lhfm.training.dataset import LongitudinalWindowDataset
from lhfm.training.evaluate import evaluate_downstream, save_results_table
from lhfm.utils.config import load_config, resolve_device, set_global_seed
from lhfm.utils.logging import get_logger

log = get_logger("evaluate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=str, default=None,
                   help="path to downstream.pt; defaults to checkpoints/downstream.pt")
    p.add_argument("--features", type=str, default=None,
                   help="path to engineered features (csv/parquet); defaults to "
                        "data/processed/features.{parquet,csv}")
    p.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--bootstrap-resamples", type=int, default=1000)
    p.add_argument("--no-cluster-bootstrap", action="store_true",
                   help="resample windows instead of participants (typically wrong "
                        "for longitudinal data; use only to reproduce old numbers)")
    p.add_argument("--out-dir", type=str, default=None,
                   help="where to write metrics_test.csv / .json (default: results/tables/)")
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
    log.info(
        "evaluating checkpoint %s (run_tag=%s, git_sha=%s, config_hash=%s)",
        ckpt_path.name, meta.get("run_tag", "?"),
        meta.get("git_sha", "?"), meta.get("config_hash", "?"),
    )

    # Reconstruct the model.
    import torch

    from lhfm.models.downstream import DownstreamRiskModel
    from lhfm.models.encoder import MultimodalLongitudinalEncoder
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
    log.info("loading features from %s", feat_path)
    df = (
        pd.read_parquet(feat_path) if feat_path.suffix == ".parquet"
        else pd.read_csv(feat_path, parse_dates=["date"])
    )
    df["date"] = pd.to_datetime(df["date"])

    # Re-apply age standardization with the *training-time* reference stats.
    df = _cbf(
        df,
        age_ref_mean=meta.get("age_ref_mean"),
        age_ref_std=meta.get("age_ref_std"),
    )

    window_days = int(meta.get("window_days", cfg["training"]["window_days"]))
    pid_lengths = df.groupby("participant_id")["date"].count()
    df = df[df["participant_id"].isin(pid_lengths[pid_lengths >= window_days + 1].index)]

    splits = train_val_test_split_by_participant(
        df,
        val_fraction=cfg["training"]["val_fraction"],
        test_fraction=cfg["training"]["test_fraction"],
        seed=cfg["project"]["seed"],
    )
    if args.split == "all":
        sub = df
    else:
        sub = splits[args.split]
    log.info("evaluating split=%s: %d rows, %d participants",
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
    target_dates = pd.to_datetime(end_dates) + pd.Timedelta(days=1)
    target_cols = [TASK_TO_COLUMN[t] for t in task_names]
    long = sub[["participant_id", "date", *target_cols]].copy()
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

    pid_to_idx = {p: i for i, p in enumerate(sorted(df["participant_id"].unique()))}
    idx = np.array([pid_to_idx[p] for p in pids], dtype=np.int64)
    ds = LongitudinalWindowDataset(X, Y, modality_slices, participant_idx=idx)

    results = evaluate_downstream(
        model, ds, task_names=task_names, device=device,
        batch_size=cfg["training"]["batch_size"],
        bootstrap_resamples=args.bootstrap_resamples,
        cluster_bootstrap=not args.no_cluster_bootstrap,
    )

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(cfg["paths"]["results_dir"]) / "tables"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    save_results_table(results, out_dir / f"metrics_{args.split}.csv")
    json_results = {
        t: {k: v for k, v in r.items() if k not in ("y_true", "y_prob")}
        for t, r in results.items()
    }
    (out_dir / f"metrics_{args.split}.json").write_text(json.dumps(json_results, indent=2))
    log.info("wrote %s", out_dir / f"metrics_{args.split}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
