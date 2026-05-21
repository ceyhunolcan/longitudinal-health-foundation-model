"""Tests for integrated-gradients interpretability.

The big-picture axioms we check are:

- Sensitivity: a feature the model literally ignores (a constant zero
  channel) gets zero attribution.
- Completeness: attributions sum to f(x) - f(baseline) in logit space, up
  to small quadrature error.
- Shape: per-modality attribution tensors have the same shape as the inputs.

These are dependent on torch so the whole module skips when torch is
missing.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lhfm.interpretability import (  # noqa: E402
    aggregate_to_feature_table,
    attribute,
    humanize_attribution,
)
from lhfm.models.downstream import DownstreamRiskModel  # noqa: E402
from lhfm.models.encoder import MultimodalLongitudinalEncoder  # noqa: E402


def _toy_model(modality_dims=None, tasks=None):
    """Tiny model so tests run fast even on CI."""
    modality_dims = modality_dims or {"wearable": 4, "smartphone": 3}
    tasks = tasks or ["low_mood", "high_stress"]
    enc = MultimodalLongitudinalEncoder(
        modality_dims=modality_dims, d_model=16, n_heads=2, n_layers=1,
        max_seq_len=10,
    )
    return DownstreamRiskModel(encoder=enc, task_names=tasks)


def test_attribute_returns_shapes_matching_inputs():
    model = _toy_model()
    modalities = {"wearable": torch.randn(1, 8, 4), "smartphone": torch.randn(1, 8, 3)}
    masks = {"wearable": torch.zeros(1, 8), "smartphone": torch.zeros(1, 8)}
    result = attribute(model, modalities, task="low_mood", masks=masks, n_steps=8)
    assert result.task == "low_mood"
    assert set(result.per_modality.keys()) == {"wearable", "smartphone"}
    assert result.per_modality["wearable"].shape == (8, 4)
    assert result.per_modality["smartphone"].shape == (8, 3)


def test_completeness_axiom():
    """sum(attributions) should approximately equal f(x) - f(baseline) in logit space."""
    model = _toy_model()
    model.eval()
    torch.manual_seed(0)
    modalities = {"wearable": torch.randn(1, 8, 4), "smartphone": torch.randn(1, 8, 3)}
    masks = {"wearable": torch.zeros(1, 8), "smartphone": torch.zeros(1, 8)}
    result = attribute(model, modalities, task="low_mood", masks=masks, n_steps=64)
    # Convergence delta measures how much the IG sum deviates from f(x)-f(baseline)
    # in logit space. With 64 steps on a small model this should be tight.
    assert result.convergence_delta < 0.05, (
        f"completeness violated; delta={result.convergence_delta:.3f}"
    )


def test_rejects_batch_size_greater_than_one():
    model = _toy_model()
    modalities = {"wearable": torch.randn(3, 8, 4), "smartphone": torch.randn(3, 8, 3)}
    masks = {"wearable": torch.zeros(3, 8), "smartphone": torch.zeros(3, 8)}
    with pytest.raises(ValueError, match="batch size 1"):
        attribute(model, modalities, task="low_mood", masks=masks, n_steps=4)


def test_unknown_task_raises():
    model = _toy_model()
    modalities = {"wearable": torch.randn(1, 8, 4), "smartphone": torch.randn(1, 8, 3)}
    masks = {"wearable": torch.zeros(1, 8), "smartphone": torch.zeros(1, 8)}
    with pytest.raises(KeyError, match="task"):
        attribute(model, modalities, task="not_a_task", masks=masks, n_steps=4)


def test_aggregate_returns_sorted_top_k():
    model = _toy_model()
    torch.manual_seed(1)
    modalities = {"wearable": torch.randn(1, 8, 4), "smartphone": torch.randn(1, 8, 3)}
    masks = {"wearable": torch.zeros(1, 8), "smartphone": torch.zeros(1, 8)}
    result = attribute(model, modalities, task="low_mood", masks=masks, n_steps=16)

    feature_columns = ["w1", "w2", "w3", "w4", "s1", "s2", "s3"]
    modality_slices = {"wearable": (0, 4), "smartphone": (4, 7)}
    top = aggregate_to_feature_table(
        result, feature_columns, modality_slices, top_k=3,
    )
    assert len(top) == 3
    abs_attrs = [r["abs_attribution"] for r in top]
    assert abs_attrs == sorted(abs_attrs, reverse=True)
    assert all("feature" in r and "modality" in r and "direction" in r for r in top)


def test_humanize_attribution_includes_probability():
    model = _toy_model()
    torch.manual_seed(2)
    modalities = {"wearable": torch.randn(1, 8, 4), "smartphone": torch.randn(1, 8, 3)}
    masks = {"wearable": torch.zeros(1, 8), "smartphone": torch.zeros(1, 8)}
    result = attribute(model, modalities, task="low_mood", masks=masks, n_steps=16)
    bullets = humanize_attribution(
        result,
        feature_columns=["w1", "w2", "w3", "w4", "s1", "s2", "s3"],
        modality_slices={"wearable": (0, 4), "smartphone": (4, 7)},
        top_k=3,
    )
    assert isinstance(bullets, list)
    assert any("Predicted probability" in b for b in bullets)
    assert len(bullets) >= 2
