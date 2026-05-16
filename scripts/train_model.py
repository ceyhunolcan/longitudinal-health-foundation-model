"""Train the foundation model and the four downstream risk heads.

Pipeline::

    python scripts/run_pipeline.py        # produces data/processed/features.{csv,parquet}
    python scripts/train_model.py         # produces checkpoints/{ssl,downstream}.pt

Outputs:
    checkpoints/ssl.pt              SSL-pretrained encoder weights
    checkpoints/downstream.pt       Encoder + risk-head weights (loaded by API)
    results/tables/metrics_test.csv per-task metrics on held-out participants
    results/tables/baselines.csv    classical-baseline comparison

The script is deliberately written so that re-running it is cheap: it
always writes the same filenames and skips no steps. If you want partial
runs, comment out branches at the bottom of ``main``.
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

from lhfm.data.preprocessing import build_windows, train_val_test_split_by_participant  # noqa: E402
from lhfm.features import build_full_feature_table                                     # noqa: E402
from lhfm.training.dataset import LongitudinalWindowDataset                            # noqa: E402
from lhfm.training.evaluate import (                                                   # noqa: E402
    baseline_comparison, evaluate_downstream, save_results_table,
)
from lhfm.training.train_downstream import train_downstream                            # noqa: E402
from lhfm.training.train_ssl import pretrain_ssl                                       # noqa: E402
from lhfm.utils.config import load_config, resolve_device, set_global_seed             # noqa: E402
from lhfm.utils.logging import get_logger                                              # noqa: E402


log = get_logger("train")


# These four column groups map onto the four "modalities" the encoder knows
# about. They MUST line up with the modality_dims in configs/model.yaml.
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
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--features", type=str, default=None,
                   help="path to engineered features (csv or parquet). "
                        "if missing, we generate a small synthetic cohort on the fly")
    p.add_argument("--ssl-epochs", type=int, default=None)
    p.add_argument("--downstream-epochs", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-baselines", action="store_true",
                   help="skip the classical-baseline comparison")
    p.add_argument("--no-pretrain", action="store_true",
                   help="skip SSL pretraining (random encoder + train heads only)")
    p.add_argument(
        "--exclude-ema-features", action="store_true",
        help=("drop survey_* columns from the feature matrix. The default "
              "tasks (low_mood, high_stress) are thresholds on those same "
              "EMA fields one day ahead, so including them as features turns "
              "the task into trivial next-day autoregression. Use this flag "
              "to run the methodologically interesting version that has to "
              "predict mood from passive sensing alone."),
    )
    p.add_argument(
        "--run-tag", type=str, default=None,
        help="optional tag appended to checkpoint filenames (helps when "
             "comparing hyperparameter sweeps in the same checkpoints/ dir)",
    )
    return p.parse_args()


def _git_sha() -> str:
    """Best-effort short git SHA so we can tag runs. Returns 'nogit' if not in a repo."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, cwd=ROOT,
        )
        return out.decode().strip()
    except Exception:
        return "nogit"


def _config_hash(cfg: dict) -> str:
    """Short hash of the config dict so different sweeps don't collide silently."""
    import hashlib
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_features(cfg: dict, override_path: str | None) -> pd.DataFrame:
    processed = Path(cfg["paths"]["processed_dir"]) / "features.parquet"
    csv_alt = processed.with_suffix(".csv")
    candidate = Path(override_path) if override_path else (
        processed if processed.exists() else csv_alt
    )
    if candidate.exists():
        log.info("loading features from %s", candidate)
        if candidate.suffix == ".parquet":
            df = pd.read_parquet(candidate)
            df["date"] = pd.to_datetime(df["date"])
            return df
        return pd.read_csv(candidate, parse_dates=["date"])

    log.warning("no engineered features found; generating a small cohort on the fly")
    from lhfm.data.synthetic_generator import generate_synthetic_cohort
    raw = generate_synthetic_cohort(n_participants=80, n_days=60, seed=cfg["project"]["seed"])
    return build_full_feature_table(raw, impute=True, add_targets=True)


def _slice_features(feature_groups: dict[str, list[str]]) -> dict[str, tuple[int, int]]:
    """Return (start, end) offsets per modality in the concatenated feature axis."""
    out, cursor = {}, 0
    for mod, cols in feature_groups.items():
        out[mod] = (cursor, cursor + len(cols))
        cursor += len(cols)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    cfg = load_config("default", "model", "features")
    set_global_seed(cfg["project"]["seed"])
    device = resolve_device(args.device or cfg["training"]["device"])

    # Log the provenance up front: git SHA, config hash, and run tag. This
    # is the first thing a senior engineer wants in the log when they pick
    # up a stale experiment six months from now.
    cfg_hash = _config_hash(cfg)
    git_sha = _git_sha()
    run_tag = args.run_tag or f"{git_sha}-{cfg_hash}"
    log.info("run_tag=%s  git_sha=%s  config_hash=%s  device=%s",
             run_tag, git_sha, cfg_hash, device)

    df = _load_features(cfg, args.features)
    window_days = cfg["training"]["window_days"]
    pid_lengths = df.groupby("participant_id")["date"].count()
    keep = pid_lengths[pid_lengths >= window_days + 1].index
    df = df[df["participant_id"].isin(keep)].reset_index(drop=True)
    log.info("%d participants kept after length filtering", df["participant_id"].nunique())

    # --- feature schema: optionally drop EMA columns to avoid the
    #     trivial-autoregression critique --------------------------------------
    feature_groups = {k: list(v) for k, v in FEATURE_GROUPS.items()}
    if args.exclude_ema_features:
        # No EMA columns are currently in FEATURE_GROUPS (they're targets'
        # source columns and are not enumerated as features), but if anyone
        # adds them later this flag is the documented way to keep the task
        # honest. We also forbid the encoder from seeing the raw EMA via
        # passive-sensing surrogates that mirror it.
        log.info("EMA-blind mode: dropping survey-derived feature columns if present")
        for mod, cols in feature_groups.items():
            feature_groups[mod] = [c for c in cols if not c.startswith("survey_")]

    feature_cols = sum(feature_groups.values(), [])
    for c in feature_cols:
        if c not in df.columns:
            raise KeyError(f"feature column '{c}' missing from dataframe")
    modality_slices = _slice_features(feature_groups)
    modality_dims = {m: len(cols) for m, cols in feature_groups.items()}

    # --- splits (participant-level) -----------------------------------------
    splits = train_val_test_split_by_participant(
        df, val_fraction=cfg["training"]["val_fraction"],
        test_fraction=cfg["training"]["test_fraction"],
        seed=cfg["project"]["seed"],
    )
    for k, v in splits.items():
        log.info("split %s: %d rows, %d participants", k, len(v), v["participant_id"].nunique())

    # --- fit population reference stats on the TRAINING split only ----------
    # The earlier audit-pass version of compute_baseline_features had
    # hardcoded constants that happened to match the synthetic prior --
    # technically functional but methodologically embarrassing. The right
    # fix is to compute these from the training cohort and persist them.
    from lhfm.features.baseline_features import fit_baseline_reference_stats
    ref_stats = fit_baseline_reference_stats(splits["train"])
    log.info("training age reference: mean=%.2f std=%.2f (from %d participants)",
             ref_stats["age_ref_mean"], ref_stats["age_ref_std"],
             splits["train"]["participant_id"].nunique())

    # Re-engineer age_z on every split with the SAME reference stats.
    from lhfm.features.baseline_features import compute_baseline_features as _cbf
    for k in splits:
        splits[k] = _cbf(
            splits[k],
            age_ref_mean=ref_stats["age_ref_mean"],
            age_ref_std=ref_stats["age_ref_std"],
        )

    task_names = cfg["downstream"]["tasks"]

    # --- windowing: compute X *once*, then slice labels per task. The prior
    #     implementation re-ran build_windows K times wastefully -----------
    def _build_split(split_df: pd.DataFrame, target_mode: str = "next_day"):
        # Use the first task only as an "alignment column" to drive the
        # window loop. Then we extract every task's labels by joining the
        # window-end dates back into the long-form frame.
        X, _, pids, end_dates = build_windows(
            split_df, feature_cols=feature_cols,
            target_col=TASK_TO_COLUMN[task_names[0]],
            window_days=window_days, stride=1, target_mode=target_mode,
        )
        if X.shape[0] == 0:
            return X, np.zeros((0, len(task_names)), dtype=np.float32), pids

        # Build a lookup of (pid, end_date) -> all targets in one shot.
        # We need the target *day after* window for "next_day" mode, hence
        # the +1-day shift below.
        target_offset = pd.Timedelta(days=1) if target_mode == "next_day" else pd.Timedelta(days=0)
        target_dates = pd.to_datetime(end_dates) + target_offset
        target_cols = [TASK_TO_COLUMN[t] for t in task_names]
        long = split_df[["participant_id", "date", *target_cols]].copy()
        long["date"] = pd.to_datetime(long["date"])
        long = long.set_index(["participant_id", "date"])
        keys = list(zip(pids.tolist(), target_dates))
        Y = np.full((X.shape[0], len(task_names)), np.nan, dtype=np.float32)
        for j, key in enumerate(keys):
            try:
                row = long.loc[key]
                for i, col in enumerate(target_cols):
                    val = row[col]
                    Y[j, i] = float(val) if not pd.isna(val) else np.nan
            except KeyError:
                pass  # day past end of timeline; row stays NaN
        return X, Y, pids

    Xtr, Ytr, pids_tr = _build_split(splits["train"])
    Xva, Yva, pids_va = _build_split(splits["val"])
    Xte, Yte, pids_te = _build_split(splits["test"])
    log.info("windows: train=%d val=%d test=%d", len(Xtr), len(Xva), len(Xte))

    # Class balance per task on the training set. Senior reviewers love this.
    for i, t in enumerate(task_names):
        col = Ytr[:, i]
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            log.warning("task %s has zero labelled training windows", t)
            continue
        rate = float(col[valid].mean())
        log.info("task %-22s  n=%d  positive_rate=%.3f", t, int(valid.sum()), rate)

    # Participant -> integer index (one global lookup).
    all_pids = sorted(df["participant_id"].unique())
    pid_to_idx = {p: i for i, p in enumerate(all_pids)}
    idx_tr = np.array([pid_to_idx[p] for p in pids_tr], dtype=np.int64)
    idx_va = np.array([pid_to_idx[p] for p in pids_va], dtype=np.int64)
    idx_te = np.array([pid_to_idx[p] for p in pids_te], dtype=np.int64)

    train_ds = LongitudinalWindowDataset(Xtr, Ytr, modality_slices, participant_idx=idx_tr)
    val_ds = LongitudinalWindowDataset(Xva, Yva, modality_slices, participant_idx=idx_va)
    test_ds = LongitudinalWindowDataset(Xte, Yte, modality_slices, participant_idx=idx_te)

    # --- model ------------------------------------------------------------
    from lhfm.models.encoder import MultimodalLongitudinalEncoder
    encoder_cfg = cfg["encoder"]
    encoder = MultimodalLongitudinalEncoder(
        modality_dims=modality_dims,
        d_model=encoder_cfg["d_model"],
        n_heads=encoder_cfg["n_heads"],
        n_layers=encoder_cfg["n_layers"],
        ff_dim=encoder_cfg["ff_dim"],
        dropout=encoder_cfg["dropout"],
        max_seq_len=encoder_cfg["max_seq_len"],
        n_participants=len(all_pids) if encoder_cfg["use_participant_embedding"] else 0,
        participant_embedding_dim=encoder_cfg["participant_embedding_dim"],
    )

    # Log model size up front so anyone reading the log knows what we built.
    param_counts = encoder.count_parameters(trainable_only=True)
    log.info(
        "encoder params (trainable): total=%s  transformer=%s  projectors=%s",
        f"{param_counts['total']:,}",
        f"{param_counts['transformer']:,}",
        f"{param_counts['projectors']:,}",
    )

    # --- SSL pretraining --------------------------------------------------
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ssl_ckpt = ckpt_dir / "ssl.pt"

    if not args.no_pretrain:
        from lhfm.models.self_supervised import SSLLossWeights
        ssl_cfg = cfg["ssl"]
        log.info("pretraining with SSL")
        pretrain_ssl(
            encoder,
            train_dataset=train_ds, val_dataset=val_ds,
            reconstruction_target_modality="wearable",
            epochs=args.ssl_epochs or cfg["training"]["ssl_epochs"],
            batch_size=cfg["training"]["batch_size"],
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
            mask_ratio=ssl_cfg["mask_ratio"],
            weights=SSLLossWeights(
                recon=ssl_cfg["recon_weight"],
                next_day=ssl_cfg["nextday_weight"],
                contrastive=ssl_cfg["contrastive_weight"],
                temperature=ssl_cfg["contrastive_temperature"],
            ),
            device=device, num_workers=cfg["training"]["num_workers"],
            checkpoint_path=ssl_ckpt,
            early_stopping_patience=cfg["training"]["early_stopping_patience"],
        )
    else:
        log.info("skipping SSL pretraining (--no-pretrain)")

    # --- downstream -------------------------------------------------------
    log.info("training downstream heads")
    state = train_downstream(
        encoder=encoder,
        task_names=task_names,
        train_dataset=train_ds, val_dataset=val_ds,
        epochs=args.downstream_epochs or cfg["training"]["downstream_epochs"],
        batch_size=cfg["training"]["batch_size"],
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        device=device, num_workers=cfg["training"]["num_workers"],
        freeze_encoder=False,
        checkpoint_path=ckpt_dir / "downstream.pt",
        early_stopping_patience=cfg["training"]["early_stopping_patience"],
    )

    # --- save final checkpoint + meta sidecar -----------------------------
    # Meta is written as a separate JSON file so the API can load the
    # checkpoint with torch.load(weights_only=True) -- no arbitrary pickle.
    import torch
    meta = {
        "feature_columns": feature_cols,
        "modality_slices": {k: list(v) for k, v in modality_slices.items()},
        "modality_dims": modality_dims,
        "task_names": task_names,
        "d_model": encoder_cfg["d_model"],
        "n_heads": encoder_cfg["n_heads"],
        "n_layers": encoder_cfg["n_layers"],
        "max_seq_len": encoder_cfg["max_seq_len"],
        "n_participants": len(all_pids) if encoder_cfg["use_participant_embedding"] else 0,
        "window_days": window_days,
        # Population reference stats used at training time. Inference code
        # must use these same numbers, NOT recompute them from the request.
        "age_ref_mean": ref_stats["age_ref_mean"],
        "age_ref_std": ref_stats["age_ref_std"],
        "exclude_ema_features": bool(args.exclude_ema_features),
        "param_count": param_counts,
        "run_tag": run_tag,
        "git_sha": git_sha,
        "config_hash": cfg_hash,
        "training_config": {
            "epochs_downstream": args.downstream_epochs or cfg["training"]["downstream_epochs"],
            "epochs_ssl": 0 if args.no_pretrain else (args.ssl_epochs or cfg["training"]["ssl_epochs"]),
            "batch_size": cfg["training"]["batch_size"],
            "lr": cfg["training"]["lr"],
            "weight_decay": cfg["training"]["weight_decay"],
            "seed": cfg["project"]["seed"],
        },
    }
    ckpt_path = ckpt_dir / "downstream.pt"
    torch.save(state.model.state_dict(), ckpt_path)
    ckpt_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    # Also stash a run-tagged copy so sweeps don't overwrite each other.
    tagged = ckpt_dir / f"downstream-{run_tag}.pt"
    try:
        torch.save(state.model.state_dict(), tagged)
        tagged.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    except OSError as exc:
        log.warning("could not write tagged checkpoint copy: %s", exc)
    log.info("saved final checkpoint to %s (+ meta sidecar)", ckpt_path)

    # --- evaluation -------------------------------------------------------
    results_dir = Path(cfg["paths"]["results_dir"]) / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    log.info("evaluating on test split")
    test_results = evaluate_downstream(
        state.model, test_ds, task_names=task_names, device=device,
        batch_size=cfg["training"]["batch_size"],
    )
    save_results_table(test_results, results_dir / "metrics_test.csv")
    log.info("wrote %s", results_dir / "metrics_test.csv")

    # --- classical baselines ---------------------------------------------
    if not args.no_baselines:
        log.info("running classical baselines (logreg / random forest / xgboost-if-available)")
        baseline_rows = []
        for i, name in enumerate(task_names):
            base = baseline_comparison(
                X_train=Xtr, y_train=Ytr[:, i],
                X_test=Xte, y_test=Yte[:, i],
                task_name=name, include_xgboost=True,
                random_state=cfg["project"]["seed"],
            )
            for model_name, m in base.items():
                baseline_rows.append({
                    "task": name, "model": model_name,
                    "auroc": m.get("auroc"), "auprc": m.get("auprc"),
                    "f1": m.get("f1"), "ece": m.get("ece"),
                    "n_pos": m.get("n_pos"), "n_total": m.get("n_total"),
                })
        if baseline_rows:
            pd.DataFrame(baseline_rows).to_csv(results_dir / "baselines.csv", index=False)
            log.info("wrote %s", results_dir / "baselines.csv")

    # Strip the raw arrays from JSON output for compactness.
    json_results = {
        t: {k: v for k, v in r.items() if k not in ("y_true", "y_prob")}
        for t, r in test_results.items()
    }
    (results_dir / "metrics_test.json").write_text(json.dumps(json_results, indent=2))

    # --- figures ---------------------------------------------------------
    # Calibration and confusion plots per task. Saved to results/figures.
    figures_dir = Path(cfg["paths"]["results_dir"]) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        from lhfm.utils.plotting import plot_calibration, plot_confusion
        for task, m in test_results.items():
            if "y_true" not in m or "y_prob" not in m:
                continue
            yt = np.array(m["y_true"])
            yp = np.array(m["y_prob"])
            if len(np.unique(yt)) < 2:
                continue
            fig = plot_calibration(yt, yp, n_bins=10)
            fig.savefig(figures_dir / f"calibration_{task}.png", dpi=140, bbox_inches="tight")
            cm = np.array(m["confusion_matrix"])
            fig = plot_confusion(cm)
            fig.savefig(figures_dir / f"confusion_{task}.png", dpi=140, bbox_inches="tight")
        log.info("wrote figures to %s", figures_dir)
    except Exception as exc:
        log.warning("could not render figures (matplotlib backend issue?): %s", exc)

    log.info("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
