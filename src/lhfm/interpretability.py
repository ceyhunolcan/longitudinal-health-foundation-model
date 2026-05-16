"""Integrated gradients for the foundation model.

Replaces the rule-based "explanation" panel from earlier revisions with
integrated gradients (Sundararajan, Taly & Yan, ICML 2017). Output is a
per-feature attribution score: positive pushes the prediction toward
"risk elevated", negative toward "risk low".

Why IG and not SHAP, LIME, or attention rollout. IG is axiomatic --
completeness (attributions sum to f(x) - f(baseline)) and implementation
invariance (functionally identical networks give the same attributions).
SHAP shares completeness but is much slower on sequence models. Attention
rollout is fast but isn't faithful in the sense that matters when we
want to tell a clinician "this prediction came from these features."

Cost is one forward pass plus n_steps backward passes (default 32). On
CPU at our model size that fits inside the API's per-request budget.

The baseline is all-zeros. That's defensible here because every upstream
feature is already mean-centered (z-scores, deviations from personal
baseline, etc.), so all-zeros is a neutral day. A cohort-mean baseline
is a defensible alternative -- pass it explicitly via the ``baseline``
arg if you want it.

Main entry point:
    attribute(model, modalities, task, baseline=None, n_steps=32) -> dict

Returns a dict keyed by modality name, each value a (T, F) attribution
tensor. ``aggregate_to_feature_table`` flattens that into a tidy
dataframe sorted by absolute attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from lhfm.models.downstream import DownstreamRiskModel


@dataclass
class AttributionResult:
    """Per-feature attribution for a single (window, task) pair.

    Attributes
    ----------
    task : the task name attributions were computed against.
    per_modality : dict from modality name to (T, F_m) numpy array of IG
        scores. Positive => evidence for the positive class (e.g. low mood).
    convergence_delta : |f(x) - f(baseline) - sum(attributions)|. IG is
        exact in the limit of infinite steps; this number tells you how
        close to exact you got. Anything under 0.05 of the prediction
        magnitude is fine; bigger means crank up ``n_steps``.
    pred_prob : the model's predicted probability for this input.
    baseline_prob : the model's predicted probability for the baseline.
    """

    task: str
    per_modality: dict[str, np.ndarray]
    convergence_delta: float
    pred_prob: float
    baseline_prob: float


def attribute(
    model: DownstreamRiskModel,
    modalities: dict[str, torch.Tensor],
    task: str,
    masks: dict[str, torch.Tensor] | None = None,
    participant_idx: torch.Tensor | None = None,
    pad_mask: torch.Tensor | None = None,
    baseline: Optional[dict[str, torch.Tensor]] = None,
    n_steps: int = 32,
    device: str = "cpu",
) -> AttributionResult:
    """Run integrated gradients on a single example.

    Parameters
    ----------
    model : a trained DownstreamRiskModel. Must be in eval mode; we don't
        modify its mode but you should set it yourself if you care about
        dropout determinism.
    modalities : dict from modality name to (1, T, F_m) tensor. Batch size
        must be 1 -- IG runs on individual examples by construction.
    task : the head name to attribute against (e.g. ``"low_mood"``).
    baseline : optional dict in the same shape as ``modalities``. We default
        to all-zeros, which is a defensible neutral baseline because every
        upstream z-score and deviation feature is centered at zero. Pass a
        cohort-mean tensor instead if you want a "compared to a typical
        person" attribution.
    n_steps : number of Riemann steps along the straight-line path. 32 is
        a reasonable speed/accuracy trade-off on CPU. Increase to 64 if
        ``convergence_delta`` is unsatisfactory.

    Returns
    -------
    AttributionResult
    """
    # We require batch size 1 -- attributing a whole batch in one pass mixes
    # different examples' gradient paths and produces meaningless per-row
    # scores. Loop over the batch outside this function.
    for name, x in modalities.items():
        if x.shape[0] != 1:
            raise ValueError(
                f"attribute() expects batch size 1 (got {x.shape[0]} for modality '{name}'). "
                f"Loop over the batch in the caller."
            )

    if task not in model.task_names:
        raise KeyError(f"task '{task}' not in model.task_names={model.task_names}")

    # Build the all-zeros baseline if the user didn't supply one.
    if baseline is None:
        baseline = {k: torch.zeros_like(v) for k, v in modalities.items()}
    else:
        # Same keys + shapes as the input.
        for k, v in modalities.items():
            if k not in baseline or baseline[k].shape != v.shape:
                raise ValueError(f"baseline mismatch on modality '{k}'")

    model = model.to(device).eval()

    # Move everything to device.
    modalities = {k: v.to(device) for k, v in modalities.items()}
    baseline = {k: v.to(device) for k, v in baseline.items()}
    if masks is not None:
        masks = {k: v.to(device) for k, v in masks.items()}
    if participant_idx is not None:
        participant_idx = participant_idx.to(device)
    if pad_mask is not None:
        pad_mask = pad_mask.to(device)

    # Endpoint predictions for the completeness check.
    with torch.no_grad():
        f_x = _scalar_logit(
            model, modalities, masks, participant_idx, pad_mask, task,
        )
        f_b = _scalar_logit(
            model, baseline, masks, participant_idx, pad_mask, task,
        )
    pred_prob = float(torch.sigmoid(f_x).item())
    baseline_prob = float(torch.sigmoid(f_b).item())

    # Average gradients along the straight-line path baseline -> input.
    # We accumulate into the same shape as each modality so we can return
    # per-feature attributions per modality.
    accum: dict[str, torch.Tensor] = {
        k: torch.zeros_like(v) for k, v in modalities.items()
    }

    # We use trapezoidal-ish midpoint quadrature: alphas at the midpoints
    # of n_steps equal-width intervals. This is the standard IG choice.
    alphas = torch.linspace(0.0, 1.0, steps=n_steps + 1, device=device)
    mid_alphas = 0.5 * (alphas[:-1] + alphas[1:])    # (n_steps,)
    step_weight = 1.0 / n_steps

    for a in mid_alphas:
        interp = {
            k: (baseline[k] + a * (modalities[k] - baseline[k])).detach().clone().requires_grad_(True)
            for k in modalities
        }

        logit = _scalar_logit(model, interp, masks, participant_idx, pad_mask, task)
        # Sum so .backward gives one gradient per modality.
        model.zero_grad(set_to_none=True)
        logit.backward()

        for k in modalities:
            if interp[k].grad is None:
                continue   # shouldn't happen but defend against frozen subgraphs
            accum[k] = accum[k] + interp[k].grad.detach() * step_weight

    # Final attribution = (x - baseline) elementwise times averaged gradient.
    per_modality = {}
    sum_attrib = 0.0
    for k in modalities:
        attr = ((modalities[k] - baseline[k]) * accum[k]).detach()
        per_modality[k] = attr.squeeze(0).cpu().numpy()  # drop batch dim
        sum_attrib += float(attr.sum().item())

    # Convergence: completeness axiom says sum(attributions) ≈ f(x) - f(baseline)
    # (working in logit space; we report sigmoid-space numbers for the user
    # but compare in logit space because that's what IG attributes).
    convergence_delta = float(abs((f_x - f_b).item() - sum_attrib))

    return AttributionResult(
        task=task,
        per_modality=per_modality,
        convergence_delta=convergence_delta,
        pred_prob=pred_prob,
        baseline_prob=baseline_prob,
    )


def _scalar_logit(
    model: DownstreamRiskModel,
    modalities: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor] | None,
    participant_idx: torch.Tensor | None,
    pad_mask: torch.Tensor | None,
    task: str,
) -> torch.Tensor:
    """Forward pass returning the chosen task's scalar logit."""
    logits = model(
        modalities, masks=masks,
        participant_idx=participant_idx, pad_mask=pad_mask,
    )
    out = logits[task]
    # Shape can be (1,) or () depending on the head -- squeeze to scalar.
    return out.reshape(-1)[0]


def aggregate_to_feature_table(
    result: AttributionResult,
    feature_columns: list[str],
    modality_slices: dict[str, tuple[int, int]],
    aggregation: str = "mean",
    top_k: int = 5,
) -> list[dict]:
    """Turn a (T, F_m) attribution into a flat, tidy "top features" list.

    Parameters
    ----------
    result : output of :func:`attribute`.
    feature_columns : the full concatenated list of feature column names.
    modality_slices : the offsets that map each modality back into
        ``feature_columns`` (same dict the encoder uses).
    aggregation : how to collapse the time axis. ``"mean"`` averages per
        feature; ``"sum"`` totals (useful if you care about which feature
        contributed most overall); ``"last"`` keeps only the final day's
        attribution.
    top_k : how many features to return (sorted by absolute attribution).

    Returns
    -------
    A list of ``{"feature", "modality", "attribution", "abs_attribution",
    "direction"}`` dicts, sorted by abs_attribution descending.
    """
    if aggregation not in {"mean", "sum", "last"}:
        raise ValueError(f"unknown aggregation '{aggregation}'")

    rows = []
    for modality, (a, b) in modality_slices.items():
        attr = result.per_modality.get(modality)
        if attr is None:
            continue   # modality not in this result
        if aggregation == "mean":
            collapsed = attr.mean(axis=0)
        elif aggregation == "sum":
            collapsed = attr.sum(axis=0)
        else:
            collapsed = attr[-1, :]
        cols_for_modality = feature_columns[a:b]
        for col, score in zip(cols_for_modality, collapsed, strict=False):
            rows.append({
                "feature": col,
                "modality": modality,
                "attribution": float(score),
                "abs_attribution": float(abs(score)),
                "direction": "increases risk" if score > 0 else "decreases risk",
            })

    rows.sort(key=lambda r: r["abs_attribution"], reverse=True)
    return rows[:top_k]


def humanize_attribution(
    result: AttributionResult,
    feature_columns: list[str],
    modality_slices: dict[str, tuple[int, int]],
    top_k: int = 4,
) -> list[str]:
    """Produce short natural-language bullets from an AttributionResult.

    These are the strings the API returns as the ``explanation`` field
    when a trained model is loaded. They replace the rule-based panel.
    """
    top = aggregate_to_feature_table(
        result, feature_columns, modality_slices, aggregation="mean", top_k=top_k,
    )
    delta = result.pred_prob - result.baseline_prob
    msgs = [
        f"Predicted probability: {result.pred_prob:.2f}  "
        f"(neutral baseline: {result.baseline_prob:.2f}, "
        f"shift: {delta:+.2f}).",
    ]
    for row in top:
        verb = "raised" if row["attribution"] > 0 else "lowered"
        msgs.append(
            f"{row['feature']} ({row['modality']}) {verb} the predicted risk "
            f"(attribution = {row['attribution']:+.3f})."
        )
    if result.convergence_delta > 0.1:
        msgs.append(
            f"Note: attribution convergence is loose "
            f"(delta = {result.convergence_delta:.2f}); increase n_steps for tighter scores."
        )
    return msgs
