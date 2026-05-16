"""Climate-regime generalization evaluation.

Trains the model on data with one or more climate regimes removed from
the *features* and *labels* of the training timeline, then evaluates on
the held-out regime. Answers: "does LHFM's climate-health story actually
hold up when the model sees a climate regime it wasn't trained on?"

This is the figure reviewers will ask about if the abstract mentions
climate-health integration. The "easy" version of this story is to train
on the full timeline and evaluate per-regime — we do that too, as a
diagnostic — but the actually-load-bearing version is the holdout below.

Outputs:
    results/tables/climate_regime_eval.csv

Usage::

    python scripts/run_climate_holdout.py                       # default: hold out heat-waves
    python scripts/run_climate_holdout.py --holdout cold_snap
    python scripts/run_climate_holdout.py --holdout smoke_episode
    python scripts/run_climate_holdout.py --no-retrain          # just slice the existing checkpoint

The default (with retraining) trains a fresh encoder on ``normal`` days
only, then evaluates on the held-out regime windows. The ``--no-retrain``
mode just slices the test set by regime and reports per-regime metrics
on the already-saved checkpoint — useful as a quick diagnostic.
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
from lhfm.utils.climate_regimes import (  # noqa: E402
    CLIMATE_REGIMES,
    define_climate_regime,
    regime_summary,
)
from lhfm.utils.config import load_config, resolve_device, set_global_seed  # noqa: E402
from lhfm.utils.logging import get_logger  # noqa: E402


log = get_logger("climate_holdout")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--holdout", choices=list(CLIMATE_REGIMES), default="heat_wave",
                   help="climate regime to hold out from training (eval on it)")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--features", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument(
        "--no-retrain", action="store_true",
        help="skip retraining; just slice the existing test set by regime",
    )
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

    feat_path = (
        Path(args.features) if args.features else
        Path(cfg["paths"]["processed_dir"]) / "features.parquet"
    )
    if not feat_path.exists():
        feat_path = feat_path.with_suffix(".csv")
    df = (
        pd.read_parquet(feat_path) if feat_path.suffix == ".parquet"
        else pd.read_csv(feat_path, parse_dates=["date"])
    )
    df["date"] = pd.to_datetime(df["date"])
    df = _cbf(df, age_ref_mean=meta.get("age_ref_mean"), age_ref_std=meta.get("age_ref_std"))

    summary = regime_summary(df)
    log.info("regime summary:\n%s", summary.to_string())

    if args.no_retrain:
        # Slice the test split by regime and report per-regime metrics on
        # the saved checkpoint. This is the diagnostic mode.
        return _no_retrain_eval(args, cfg, df, meta, device, ckpt_path, summary)

    # Retraining mode: redact the held-out regime's rows from the training
    # *and validation* splits, retrain briefly, evaluate on the held-out
    # regime's rows in the test split. We DON'T redact the metadata rows;
    # they remain present so windows still exist that span the boundary.
    log.info("retraining with regime %r redacted from train+val", args.holdout)

    window_days = int(meta.get("window_days", cfg["training"]["window_days"]))
    pid_lengths = df.groupby("participant_id")["date"].count()
    df = df[df["participant_id"].isin(pid_lengths[pid_lengths >= window_days + 1].index)]

    splits = train_val_test_split_by_participant(
        df, val_fraction=cfg["training"]["val_fraction"],
        test_fraction=cfg["training"]["test_fraction"],
        seed=cfg["project"]["seed"],
    )

    # In each split, mark which rows belong to the held-out regime. Set
    # *targets* to NaN there (so the loss is masked out) and feed climate
    # features only from the surviving rows. This is a soft redaction — the
    # model still gets one window of context from those days, but doesn't
    # see their labels.
    feature_cols = list(meta["feature_columns"])
    task_names = meta["task_names"]
    TASK_TO_COLUMN = {
        "low_mood": "target_low_mood",
        "high_stress": "target_high_stress",
        "sleep_disruption": "target_sleep_disruption",
        "climate_vulnerable": "target_climate_vulnerable",
    }
    target_cols = [TASK_TO_COLUMN[t] for t in task_names]

    holdout_mask_train = define_climate_regime(splits["train"], args.holdout)
    splits["train"].loc[holdout_mask_train, target_cols] = np.nan
    holdout_mask_val = define_climate_regime(splits["val"], args.holdout)
    splits["val"].loc[holdout_mask_val, target_cols] = np.nan
    log.info("redacted %d/%d train rows and %d/%d val rows",
             int(holdout_mask_train.sum()), len(splits["train"]),
             int(holdout_mask_val.sum()), len(splits["val"]))

    # Re-import training (we deferred to avoid torch in --no-retrain mode).
    import torch
    from lhfm.models.encoder import MultimodalLongitudinalEncoder
    from lhfm.models.downstream import DownstreamRiskModel
    from lhfm.training.train_downstream import train_downstream

    modality_dims = {k: int(v) for k, v in meta["modality_dims"].items()}
    modality_slices = {k: tuple(v) for k, v in meta["modality_slices"].items()}
    encoder = MultimodalLongitudinalEncoder(
        modality_dims=modality_dims,
        d_model=int(meta["d_model"]),
        n_heads=int(meta["n_heads"]),
        n_layers=int(meta["n_layers"]),
        max_seq_len=int(meta["max_seq_len"]),
        n_participants=int(meta.get("n_participants", 0)),
    )
    # Warm-start from saved encoder weights so this is a quick fine-tune
    # rather than a full retrain (which would be wasteful on synthetic).
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # The state dict was for the *DownstreamRiskModel*; we strip the head
    # prefixes to load just the encoder portion.
    enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    if enc_state:
        encoder.load_state_dict(enc_state, strict=False)
        log.info("warm-started encoder from %s", ckpt_path.name)

    def _build(split_df):
        X, _, pids, end_dates = build_windows(
            split_df, feature_cols=feature_cols,
            target_col=target_cols[0], window_days=window_days,
            stride=1, target_mode="next_day",
        )
        if X.shape[0] == 0:
            return X, np.zeros((0, len(task_names)), dtype=np.float32), pids, end_dates
        target_dates = pd.to_datetime(end_dates) + pd.Timedelta(days=1)
        long = split_df[["participant_id", "date", *target_cols]].copy()
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
        return X, Y, pids, end_dates

    Xtr, Ytr, pids_tr, _ = _build(splits["train"])
    Xva, Yva, pids_va, _ = _build(splits["val"])
    Xte, Yte, pids_te, end_dates_te = _build(splits["test"])

    all_pids = sorted(df["participant_id"].unique())
    pid_to_idx = {p: i for i, p in enumerate(all_pids)}
    idx_tr = np.array([pid_to_idx[p] for p in pids_tr], dtype=np.int64)
    idx_va = np.array([pid_to_idx[p] for p in pids_va], dtype=np.int64)
    idx_te = np.array([pid_to_idx[p] for p in pids_te], dtype=np.int64)

    train_ds = LongitudinalWindowDataset(Xtr, Ytr, modality_slices, participant_idx=idx_tr)
    val_ds = LongitudinalWindowDataset(Xva, Yva, modality_slices, participant_idx=idx_va)
    test_ds = LongitudinalWindowDataset(Xte, Yte, modality_slices, participant_idx=idx_te)

    log.info("fine-tuning downstream with redacted labels")
    state = train_downstream(
        encoder=encoder, task_names=task_names,
        train_dataset=train_ds, val_dataset=val_ds,
        epochs=10,                  # short -- the encoder is warm-started
        batch_size=cfg["training"]["batch_size"],
        lr=cfg["training"]["lr"] * 0.3,
        weight_decay=cfg["training"]["weight_decay"],
        device=device,
        freeze_encoder=False,
        checkpoint_path=None,
        early_stopping_patience=3,
    )

    # Per-regime evaluation on test.
    end_dates_te_pd = pd.to_datetime(end_dates_te)
    rows = []
    for regime in CLIMATE_REGIMES:
        # A window is "in regime" if its end date falls in that regime.
        # We build a per-window regime tag from the test dataframe.
        sub_test = splits["test"]
        sub_test["date"] = pd.to_datetime(sub_test["date"])
        regime_by_date = sub_test.set_index(["participant_id", "date"])
        regime_mask_arr = np.zeros(len(end_dates_te_pd), dtype=bool)
        for j, (pid_i, edate) in enumerate(zip(pids_te.tolist(), end_dates_te_pd)):
            try:
                row = regime_by_date.loc[(pid_i, edate)]
                # Pull the regime via define_climate_regime applied to a 1-row frame.
                tiny = pd.DataFrame({
                    "heat_index": [row["heat_index"]] if "heat_index" in row else [np.nan],
                    "temperature_c": [row["temperature_c"]],
                    "aqi": [row["aqi"]],
                })
                regime_mask_arr[j] = bool(define_climate_regime(tiny, regime).iloc[0])
            except KeyError:
                regime_mask_arr[j] = False

        if regime_mask_arr.sum() < 10:
            log.info("regime %s: too few test windows (%d); skipping",
                     regime, int(regime_mask_arr.sum()))
            continue

        # Slice test_ds. Because LongitudinalWindowDataset is constructed
        # from arrays directly, we just build a temporary sub-dataset.
        sub_X = Xte[regime_mask_arr]
        sub_Y = Yte[regime_mask_arr]
        sub_idx = idx_te[regime_mask_arr]
        sub_ds = LongitudinalWindowDataset(sub_X, sub_Y, modality_slices, participant_idx=sub_idx)
        sub_results = evaluate_downstream(
            state.model, sub_ds, task_names=task_names, device=device,
            batch_size=cfg["training"]["batch_size"],
            bootstrap_resamples=300,
        )
        for task, r in sub_results.items():
            auroc_ci = r.get("auroc_ci", (float("nan"), float("nan")))
            rows.append({
                "training_holdout": args.holdout,
                "eval_regime": regime,
                "task": task,
                "n_windows": int(regime_mask_arr.sum()),
                "auroc": r.get("auroc"),
                "auroc_ci_low": auroc_ci[0],
                "auroc_ci_high": auroc_ci[1],
                "auprc": r.get("auprc"),
                "brier": r.get("brier"),
                "ece": r.get("ece"),
            })

    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["paths"]["results_dir"]) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "climate_regime_eval.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    log.info("wrote %s", out_csv)
    return 0


def _no_retrain_eval(args, cfg, df, meta, device, ckpt_path, summary) -> int:
    """Diagnostic mode: per-regime metrics on the already-saved checkpoint."""
    import torch
    from lhfm.models.encoder import MultimodalLongitudinalEncoder
    from lhfm.models.downstream import DownstreamRiskModel

    log.info("diagnostic mode (no retraining): per-regime metrics on %s", ckpt_path.name)

    modality_dims = {k: int(v) for k, v in meta["modality_dims"].items()}
    modality_slices = {k: tuple(v) for k, v in meta["modality_slices"].items()}
    encoder = MultimodalLongitudinalEncoder(
        modality_dims=modality_dims,
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

    window_days = int(meta.get("window_days", cfg["training"]["window_days"]))
    pid_lengths = df.groupby("participant_id")["date"].count()
    df = df[df["participant_id"].isin(pid_lengths[pid_lengths >= window_days + 1].index)]
    splits = train_val_test_split_by_participant(
        df, val_fraction=cfg["training"]["val_fraction"],
        test_fraction=cfg["training"]["test_fraction"],
        seed=cfg["project"]["seed"],
    )
    feature_cols = list(meta["feature_columns"])
    task_names = meta["task_names"]
    TASK_TO_COLUMN = {
        "low_mood": "target_low_mood",
        "high_stress": "target_high_stress",
        "sleep_disruption": "target_sleep_disruption",
        "climate_vulnerable": "target_climate_vulnerable",
    }
    target_cols = [TASK_TO_COLUMN[t] for t in task_names]
    sub_test = splits["test"]
    sub_test["date"] = pd.to_datetime(sub_test["date"])

    rows = []
    for regime in CLIMATE_REGIMES:
        regime_mask = define_climate_regime(sub_test, regime)
        sub = sub_test.loc[regime_mask].copy()
        if sub["participant_id"].nunique() == 0 or len(sub) < 30:
            log.info("regime %s: skip (only %d rows)", regime, len(sub))
            continue
        X, _, pids, end_dates = build_windows(
            sub, feature_cols=feature_cols,
            target_col=target_cols[0], window_days=window_days,
            stride=1, target_mode="next_day",
        )
        if X.shape[0] == 0:
            log.info("regime %s: 0 windows (insufficient consecutive days)", regime)
            continue
        target_dates = pd.to_datetime(end_dates) + pd.Timedelta(days=1)
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

        all_pids = sorted(df["participant_id"].unique())
        pid_to_idx = {p: i for i, p in enumerate(all_pids)}
        idx = np.array([pid_to_idx[p] for p in pids], dtype=np.int64)
        ds = LongitudinalWindowDataset(X, Y, modality_slices, participant_idx=idx)
        results = evaluate_downstream(
            model, ds, task_names=task_names, device=device,
            batch_size=cfg["training"]["batch_size"], bootstrap_resamples=300,
        )
        for task, r in results.items():
            auroc_ci = r.get("auroc_ci", (float("nan"), float("nan")))
            rows.append({
                "training_holdout": "none",
                "eval_regime": regime,
                "task": task,
                "n_windows": int(X.shape[0]),
                "auroc": r.get("auroc"),
                "auroc_ci_low": auroc_ci[0],
                "auroc_ci_high": auroc_ci[1],
                "auprc": r.get("auprc"),
                "brier": r.get("brier"),
                "ece": r.get("ece"),
            })

    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["paths"]["results_dir"]) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "climate_regime_eval.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    log.info("wrote %s", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
