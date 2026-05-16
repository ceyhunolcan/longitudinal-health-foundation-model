"""Downstream classification heads on top of the pretrained encoder.

We train one small MLP per binary risk task. They share the encoder, which
can be either frozen ("linear-probe-style") or finetuned end-to-end.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from .encoder import MultimodalLongitudinalEncoder


class _RiskHead(nn.Module):
    """Two-layer MLP returning a single logit."""

    def __init__(self, d_in: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DownstreamRiskModel(nn.Module):
    """Encoder + one classification head per task.

    Parameters
    ----------
    encoder : the pretrained foundation model.
    task_names : list of task identifiers. One head per name.
    hidden_dim, dropout : head hyperparameters.
    freeze_encoder : if True, encoder parameters are frozen and only the
        risk heads are trained ("linear probe" style).
    """

    def __init__(
        self,
        encoder: MultimodalLongitudinalEncoder,
        task_names: Iterable[str],
        hidden_dim: int = 64,
        dropout: float = 0.2,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.task_names = list(task_names)
        if not self.task_names:
            raise ValueError("task_names must contain at least one task")

        self.heads = nn.ModuleDict({
            name: _RiskHead(encoder.d_model, hidden=hidden_dim, dropout=dropout)
            for name in self.task_names
        })

        self.freeze_encoder = bool(freeze_encoder)
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(
        self,
        modalities: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor] | None = None,
        participant_idx: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # If the encoder is frozen we still need gradients for the heads, so
        # we just skip the grad on encoder via no_grad context.
        if self.freeze_encoder:
            with torch.no_grad():
                enc = self.encoder(
                    modalities, masks=masks, participant_idx=participant_idx,
                    pad_mask=pad_mask,
                )
            rep = enc["representation"].detach()
        else:
            enc = self.encoder(
                modalities, masks=masks, participant_idx=participant_idx,
                pad_mask=pad_mask,
            )
            rep = enc["representation"]

        return {name: head(rep) for name, head in self.heads.items()}

    @torch.no_grad()
    def predict_proba(
        self,
        modalities: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor] | None = None,
        participant_idx: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Convenience wrapper: returns sigmoid probabilities."""
        self.eval()
        logits = self.forward(
            modalities, masks=masks, participant_idx=participant_idx,
            pad_mask=pad_mask,
        )
        return {k: torch.sigmoid(v) for k, v in logits.items()}
