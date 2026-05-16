"""Model tests. Skipped automatically if torch is not installed."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_encoder_forward_pass_shapes():
    from lhfm.models.encoder import MultimodalLongitudinalEncoder
    modality_dims = {"wearable": 6, "smartphone": 5, "climate": 5, "baseline": 4}
    encoder = MultimodalLongitudinalEncoder(
        modality_dims=modality_dims,
        d_model=32, n_heads=4, n_layers=2, max_seq_len=14,
        n_participants=20, participant_embedding_dim=8,
    )
    B, T = 3, 14
    modalities = {k: torch.randn(B, T, v) for k, v in modality_dims.items()}
    masks = {k: torch.zeros(B, T) for k in modality_dims}
    participant_idx = torch.tensor([0, 1, 2], dtype=torch.long)

    out = encoder(modalities, masks=masks, participant_idx=participant_idx)
    assert out["representation"].shape == (B, 32)
    assert out["per_step"].shape == (B, T, 32)


def test_encoder_rejects_wrong_feature_dim():
    from lhfm.models.encoder import MultimodalLongitudinalEncoder
    enc = MultimodalLongitudinalEncoder(
        modality_dims={"wearable": 6}, d_model=16, n_heads=2, n_layers=1, max_seq_len=10,
    )
    bad = {"wearable": torch.randn(2, 10, 7)}  # wrong width
    masks = {"wearable": torch.zeros(2, 10)}
    with pytest.raises(ValueError):
        enc(bad, masks=masks)


def test_ssl_model_returns_all_heads():
    from lhfm.models.encoder import MultimodalLongitudinalEncoder
    from lhfm.models.self_supervised import SelfSupervisedModel
    encoder = MultimodalLongitudinalEncoder(
        modality_dims={"wearable": 4, "smartphone": 3},
        d_model=16, n_heads=2, n_layers=1, max_seq_len=10,
    )
    ssl = SelfSupervisedModel(encoder, reconstruction_target_modality="wearable")
    modalities = {"wearable": torch.randn(2, 10, 4), "smartphone": torch.randn(2, 10, 3)}
    masks = {"wearable": torch.zeros(2, 10), "smartphone": torch.zeros(2, 10)}
    out = ssl(modalities, masks=masks)
    assert "recon" in out and out["recon"].shape == (2, 10, 4)
    assert "next_day" in out and out["next_day"].shape == (2, 4)
    assert "proj" in out and out["proj"].shape[0] == 2


def test_ssl_loss_decreases_when_recon_is_perfect():
    """Sanity: zero residuals should give zero recon loss."""
    from lhfm.models.self_supervised import ssl_loss, SSLLossWeights
    recon = torch.randn(2, 5, 4)
    mask = torch.ones(2, 5)
    out_a = {
        "recon": recon, "next_day": torch.zeros(2, 4),
        "proj": torch.nn.functional.normalize(torch.randn(2, 8), dim=-1),
    }
    targets = {
        "recon_target": recon,                  # identical -> zero error
        "recon_mask": mask,
        "next_day_target": torch.zeros(2, 4),
    }
    losses = ssl_loss(out_a, targets, outputs_b=None, weights=SSLLossWeights(contrastive=0.0))
    assert losses["recon"].item() == pytest.approx(0.0, abs=1e-6)
    assert losses["next_day"].item() == pytest.approx(0.0, abs=1e-6)


def test_downstream_model_predict_proba_range():
    from lhfm.models.encoder import MultimodalLongitudinalEncoder
    from lhfm.models.downstream import DownstreamRiskModel
    encoder = MultimodalLongitudinalEncoder(
        modality_dims={"wearable": 4, "smartphone": 3},
        d_model=16, n_heads=2, n_layers=1, max_seq_len=10,
    )
    model = DownstreamRiskModel(encoder=encoder, task_names=["low_mood", "high_stress"])
    modalities = {"wearable": torch.randn(2, 10, 4), "smartphone": torch.randn(2, 10, 3)}
    masks = {"wearable": torch.zeros(2, 10), "smartphone": torch.zeros(2, 10)}
    probs = model.predict_proba(modalities, masks=masks)
    for v in probs.values():
        assert v.shape == (2,)
        assert (v >= 0).all() and (v <= 1).all()
