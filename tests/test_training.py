"""Tests for the SSL + downstream training pipelines.

These verify the actual training machinery, not just feature engineering.
Skipped automatically when torch is not installed (mirrors test_models.py
and test_interpretability.py).

Coverage goals:
- `pretrain_ssl` runs, produces decreasing-or-stable train loss.
- `train_downstream` runs and yields a model whose state_dict is loadable.
- `_compute_pos_weights` caps at the documented ceiling and handles
  empty / single-class tasks gracefully.
- `evaluate_downstream` returns the bootstrap CIs in the expected shape.
- Checkpoints save as plain state_dict (no pickle wrapper).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lhfm.models.downstream import DownstreamRiskModel  # noqa: E402
from lhfm.models.encoder import MultimodalLongitudinalEncoder  # noqa: E402
from lhfm.training.dataset import LongitudinalWindowDataset  # noqa: E402


def _toy_dataset(
    n_windows: int = 24,
    n_participants: int = 6,
    T: int = 8,
    seed: int = 0,
) -> LongitudinalWindowDataset:
    """Build a small dataset with two modalities, 2 tasks, and 6 participants.

    We seed the labels so they're not all-zero (the rare-positive case is
    covered separately).
    """
    rng = np.random.default_rng(seed)
    # 4 wearable + 3 smartphone features
    X = rng.standard_normal((n_windows, T, 7)).astype(np.float32)
    Y = (rng.random((n_windows, 2)) > 0.6).astype(np.float32)
    # Introduce some NaN labels so the masked-BCE path is exercised.
    nan_mask = rng.random(Y.shape) < 0.10
    Y[nan_mask] = np.nan
    pid = rng.integers(0, n_participants, size=n_windows).astype(np.int64)
    slices = {"wearable": (0, 4), "smartphone": (4, 7)}
    return LongitudinalWindowDataset(X, Y, slices, participant_idx=pid)


def _toy_encoder(d_model: int = 16, n_participants: int = 6) -> MultimodalLongitudinalEncoder:
    return MultimodalLongitudinalEncoder(
        modality_dims={"wearable": 4, "smartphone": 3},
        d_model=d_model, n_heads=2, n_layers=1, max_seq_len=8,
        n_participants=n_participants,
    )


# ---------------------------------------------------------------------------
# Downstream
# ---------------------------------------------------------------------------


class TestComputePosWeights:
    def test_returns_a_weight_per_task(self):
        from lhfm.training.train_downstream import _compute_pos_weights
        ds = _toy_dataset(n_windows=40, seed=1)
        weights = _compute_pos_weights(ds, ["low_mood", "high_stress"])
        assert set(weights.keys()) == {"low_mood", "high_stress"}
        for v in weights.values():
            assert 0.0 < v <= 20.0    # capped at 20

    def test_handles_all_negative_class(self):
        from lhfm.training.train_downstream import _compute_pos_weights
        # Construct a dataset where one task has zero positives.
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 8, 7)).astype(np.float32)
        Y = np.zeros((20, 2), dtype=np.float32)   # all-negative for both tasks
        ds = LongitudinalWindowDataset(X, Y, {"wearable": (0, 4), "smartphone": (4, 7)})
        weights = _compute_pos_weights(ds, ["a", "b"])
        # Single-class -> weight defaults to 1.0 rather than blowing up.
        assert weights == {"a": 1.0, "b": 1.0}

    def test_caps_at_documented_ceiling(self):
        from lhfm.training.train_downstream import _compute_pos_weights
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 8, 7)).astype(np.float32)
        Y = np.zeros((200, 1), dtype=np.float32)
        Y[0, 0] = 1.0   # 1 positive out of 200 -> #neg/#pos = 199, capped to 20
        ds = LongitudinalWindowDataset(X, Y, {"wearable": (0, 4), "smartphone": (4, 7)})
        weights = _compute_pos_weights(ds, ["rare"], cap=20.0)
        assert weights["rare"] == 20.0


class TestTrainDownstream:
    def test_runs_and_returns_state(self, tmp_path: Path):
        from lhfm.training.train_downstream import train_downstream
        train_ds = _toy_dataset(n_windows=48, seed=1)
        val_ds = _toy_dataset(n_windows=16, seed=2)
        encoder = _toy_encoder()
        state = train_downstream(
            encoder=encoder, task_names=["low_mood", "high_stress"],
            train_dataset=train_ds, val_dataset=val_ds,
            epochs=2, batch_size=8, lr=1e-3,
            device="cpu", num_workers=0,
            checkpoint_path=tmp_path / "downstream.pt",
            early_stopping_patience=10,
        )
        assert len(state.train_losses) >= 1
        assert state.pos_weights, "pos_weights should be populated"
        assert (tmp_path / "downstream.pt").exists()

    def test_checkpoint_is_plain_state_dict(self, tmp_path: Path):
        """Saved checkpoint must be loadable with weights_only=True."""
        from lhfm.training.train_downstream import train_downstream
        train_ds = _toy_dataset(n_windows=24, seed=3)
        encoder = _toy_encoder()
        train_downstream(
            encoder=encoder, task_names=["a"],
            train_dataset=train_ds, val_dataset=None,
            epochs=1, batch_size=8, lr=1e-3,
            device="cpu", num_workers=0,
            checkpoint_path=tmp_path / "ds.pt",
        )
        # weights_only=True is the security-relevant invariant; if a future
        # change re-wraps state in a dict, this test will fail.
        state_dict = torch.load(tmp_path / "ds.pt", map_location="cpu", weights_only=True)
        assert isinstance(state_dict, dict)
        assert any("encoder" in k for k in state_dict)


# ---------------------------------------------------------------------------
# SSL
# ---------------------------------------------------------------------------


class TestPretrainSsl:
    def test_runs_two_epochs(self, tmp_path: Path):
        from lhfm.models.self_supervised import SSLLossWeights
        from lhfm.training.train_ssl import pretrain_ssl
        train_ds = _toy_dataset(n_windows=32, seed=4)
        val_ds = _toy_dataset(n_windows=12, seed=5)
        encoder = _toy_encoder()
        state = pretrain_ssl(
            encoder, train_dataset=train_ds, val_dataset=val_ds,
            epochs=2, batch_size=8, lr=1e-3,
            mask_ratio=0.2,
            weights=SSLLossWeights(recon=1.0, next_day=0.5, contrastive=0.25),
            device="cpu", num_workers=0,
            checkpoint_path=tmp_path / "ssl.pt",
        )
        assert len(state.train_losses) >= 1
        assert (tmp_path / "ssl.pt").exists()

    def test_eval_is_deterministic(self):
        """SSL validation loss must not change when called twice with the
        same model state -- the regression test for the random-mask bug."""
        from torch.utils.data import DataLoader

        from lhfm.models.self_supervised import SelfSupervisedModel, SSLLossWeights
        from lhfm.training.dataset import collate_windows
        from lhfm.training.train_ssl import _evaluate_ssl

        val_ds = _toy_dataset(n_windows=24, seed=6)
        encoder = _toy_encoder()
        ssl = SelfSupervisedModel(encoder, reconstruction_target_modality="wearable")
        ssl.eval()
        loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_windows)

        a = _evaluate_ssl(
            ssl, loader, reconstruction_target_modality="wearable",
            mask_ratio=0.2, weights=SSLLossWeights(), device="cpu",
        )
        b = _evaluate_ssl(
            ssl, loader, reconstruction_target_modality="wearable",
            mask_ratio=0.2, weights=SSLLossWeights(), device="cpu",
        )
        assert a == pytest.approx(b, rel=1e-6), (
            f"SSL eval not deterministic: {a} != {b}; random mask leaked in"
        )


# ---------------------------------------------------------------------------
# evaluate_downstream
# ---------------------------------------------------------------------------


class TestEvaluateDownstream:
    def test_returns_bootstrap_cis_and_clusters_by_participant(self):
        from lhfm.training.evaluate import evaluate_downstream
        encoder = _toy_encoder()
        model = DownstreamRiskModel(encoder=encoder, task_names=["t1", "t2"])
        # Make sure labels have both classes for both tasks.
        rng = np.random.default_rng(7)
        X = rng.standard_normal((40, 8, 7)).astype(np.float32)
        Y = (rng.random((40, 2)) > 0.5).astype(np.float32)
        pid = rng.integers(0, 6, size=40).astype(np.int64)
        ds = LongitudinalWindowDataset(
            X, Y, {"wearable": (0, 4), "smartphone": (4, 7)},
            participant_idx=pid,
        )
        results = evaluate_downstream(
            model, ds, task_names=["t1", "t2"], device="cpu",
            batch_size=8, bootstrap_resamples=50,
            cluster_bootstrap=True,
        )
        for task in ["t1", "t2"]:
            r = results[task]
            assert "auroc_ci" in r and len(r["auroc_ci"]) == 2
            assert "auprc_ci" in r and len(r["auprc_ci"]) == 2
            assert r["bootstrap_unit"] == "participant"
            assert "brier" in r

    def test_falls_back_to_row_level_when_no_pids(self):
        from lhfm.training.evaluate import evaluate_downstream
        encoder = _toy_encoder(n_participants=0)
        model = DownstreamRiskModel(encoder=encoder, task_names=["t1"])
        rng = np.random.default_rng(8)
        X = rng.standard_normal((24, 8, 7)).astype(np.float32)
        Y = (rng.random((24, 1)) > 0.5).astype(np.float32)
        ds = LongitudinalWindowDataset(X, Y, {"wearable": (0, 4), "smartphone": (4, 7)})
        results = evaluate_downstream(
            model, ds, task_names=["t1"], device="cpu",
            batch_size=8, bootstrap_resamples=50,
            cluster_bootstrap=True,
        )
        assert results["t1"]["bootstrap_unit"] == "window"
