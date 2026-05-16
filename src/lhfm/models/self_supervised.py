"""Self-supervised pre-training for the longitudinal encoder.

Three objectives, in order of importance:

1. **Masked feature reconstruction** -- randomly zero out a fraction of the
   per-step *wearable* features and ask the model to recover them. The
   encoder gets to see all other modalities and timesteps, so reconstruction
   forces it to leverage temporal and cross-modal context.
2. **Next-day state prediction** -- given the first T-1 days, predict the
   last day's wearable feature vector. A classic autoregressive-style
   pretext that gives the model a sense of trajectory.
3. **Contrastive participant identification** -- two non-overlapping windows
   from the *same* participant should embed closer than windows from
   different participants (a SimCLR-style InfoNCE).

The total loss is a weighted sum. Weights live in ``configs/model.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import MultimodalLongitudinalEncoder


@dataclass
class SSLLossWeights:
    recon: float = 1.0
    next_day: float = 0.5
    contrastive: float = 0.25
    temperature: float = 0.1


class SelfSupervisedModel(nn.Module):
    """Wraps the encoder with the heads needed for SSL pretraining."""

    def __init__(
        self,
        encoder: MultimodalLongitudinalEncoder,
        reconstruction_target_modality: str = "wearable",
    ):
        super().__init__()
        self.encoder = encoder
        self.recon_target = reconstruction_target_modality

        if reconstruction_target_modality not in encoder.modality_dims:
            raise ValueError(
                f"reconstruction target '{reconstruction_target_modality}' not in "
                f"encoder modalities {list(encoder.modality_dims.keys())}"
            )
        out_dim = encoder.modality_dims[reconstruction_target_modality]

        # Per-step decoder for masked reconstruction.
        self.recon_head = nn.Sequential(
            nn.Linear(encoder.d_model, encoder.d_model),
            nn.GELU(),
            nn.Linear(encoder.d_model, out_dim),
        )

        # Next-day predictor: takes pooled representation and predicts the
        # held-out final day's feature vector.
        self.nextday_head = nn.Sequential(
            nn.Linear(encoder.d_model, encoder.d_model),
            nn.GELU(),
            nn.Linear(encoder.d_model, out_dim),
        )

        # Contrastive projection head (SimCLR style).
        self.proj_head = nn.Sequential(
            nn.Linear(encoder.d_model, encoder.d_model),
            nn.GELU(),
            nn.Linear(encoder.d_model, encoder.d_model // 2),
        )

    def forward(
        self,
        modalities: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor] | None = None,
        participant_idx: torch.Tensor | None = None,
    ):
        out = self.encoder(modalities, masks=masks, participant_idx=participant_idx)
        recon = self.recon_head(out["per_step"])          # (B, T, F_target)
        next_day = self.nextday_head(out["representation"])  # (B, F_target)
        z = F.normalize(self.proj_head(out["representation"]), dim=-1)  # (B, D')
        return {
            "representation": out["representation"],
            "per_step": out["per_step"],
            "recon": recon,
            "next_day": next_day,
            "proj": z,
        }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _info_nce(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.1,
    participant_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Symmetric InfoNCE over (B, D) pairs.

    Both inputs are assumed already L2-normalised. ``participant_idx``, if
    supplied, lets us mask out *other* same-participant windows from the
    negative pool: if two rows of the batch come from the same person and
    are not the paired view, treating them as negatives is actively wrong.
    """
    logits = z_a @ z_b.t() / temperature
    targets = torch.arange(z_a.size(0), device=z_a.device)

    if participant_idx is not None:
        # Build a (B, B) bool mask of "same participant, different row".
        same = participant_idx.unsqueeze(0) == participant_idx.unsqueeze(1)
        eye = torch.eye(z_a.size(0), dtype=torch.bool, device=z_a.device)
        bad_neg = same & ~eye   # same person, not the paired diagonal
        # Mask those out by setting logits to -inf so softmax skips them.
        logits = logits.masked_fill(bad_neg, float("-inf"))

    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


def ssl_loss(
    outputs_a: dict[str, torch.Tensor],
    targets_a: dict[str, torch.Tensor],
    outputs_b: dict[str, torch.Tensor] | None = None,
    weights: SSLLossWeights | None = None,
    participant_idx: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the total SSL loss plus its individual components.

    Parameters
    ----------
    outputs_a : output of :class:`SelfSupervisedModel` on the first view.
    targets_a : dict with keys 'recon_target' (B, T, F), 'recon_mask' (B, T)
        and 'next_day_target' (B, F). The recon_mask is 1.0 where the
        feature was masked (and is therefore part of the loss).
    outputs_b : optional output on a second view of the same participants,
        used for the contrastive objective. Pass None to skip the
        contrastive term.
    weights : SSLLossWeights or None for defaults.
    participant_idx : optional (B,) long tensor used to mask same-participant
        non-paired entries out of the contrastive negative pool.
    """
    if weights is None:
        weights = SSLLossWeights()

    recon = outputs_a["recon"]
    target = targets_a["recon_target"]
    mask = targets_a["recon_mask"]  # (B, T) -- 1 where masked

    # MSE only on masked positions. Add a tiny eps to avoid div-by-zero when
    # a freak batch contains no masked timesteps.
    diff_sq = (recon - target).pow(2).mean(dim=-1)  # (B, T)
    masked_diff = diff_sq * mask
    recon_loss = masked_diff.sum() / (mask.sum() + 1e-6)

    nextday_loss = F.mse_loss(outputs_a["next_day"], targets_a["next_day_target"])

    if outputs_b is not None:
        contrastive_loss = _info_nce(
            outputs_a["proj"], outputs_b["proj"],
            temperature=weights.temperature,
            participant_idx=participant_idx,
        )
    else:
        contrastive_loss = torch.tensor(0.0, device=recon.device)

    total = (
        weights.recon * recon_loss
        + weights.next_day * nextday_loss
        + weights.contrastive * contrastive_loss
    )
    return {
        "total": total,
        "recon": recon_loss.detach(),
        "next_day": nextday_loss.detach(),
        "contrastive": contrastive_loss.detach(),
    }


# ---------------------------------------------------------------------------
# Utilities for building a masked view of a batch
# ---------------------------------------------------------------------------


def make_masked_view(
    x: torch.Tensor,
    mask_ratio: float = 0.15,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero out a random subset of timesteps for the target modality.

    Returns the masked tensor (same shape as input) and a (B, T) mask with
    1.0 at masked positions.
    """
    if not 0.0 <= mask_ratio < 1.0:
        raise ValueError("mask_ratio must be in [0, 1)")
    B, T, _ = x.shape
    if generator is None:
        rand = torch.rand(B, T, device=x.device)
    else:
        rand = torch.rand(B, T, device=x.device, generator=generator)
    mask = (rand < mask_ratio).to(x.dtype)
    x_masked = x * (1.0 - mask).unsqueeze(-1)
    return x_masked, mask
