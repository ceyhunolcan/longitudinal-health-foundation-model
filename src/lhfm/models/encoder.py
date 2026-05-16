"""MultimodalLongitudinalEncoder: the heart of the foundation model.

Inputs per timestep
-------------------
- wearable features        (continuous)
- smartphone features      (continuous)
- climate features         (continuous)
- baseline features        (continuous, slowly varying)
- missingness mask         (binary, per-modality)

Outputs
-------
- ``representation`` : (B, d_model) -- pooled trajectory embedding
- ``per_step``       : (B, T, d_model) -- the contextualized per-day states,
                       useful for masked reconstruction and next-day heads.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .transformer import TemporalTransformer


class _ModalityProjector(nn.Module):
    """Tiny MLP that projects a single modality to the shared d_model space.

    A linear layer plus GELU plus dropout was enough in our experiments.
    Bias is on; layernorm afterwards keeps gradients stable when several
    modalities are summed.
    """

    def __init__(self, in_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


class MultimodalLongitudinalEncoder(nn.Module):
    """Modality-specific projections + missingness embedding + transformer.

    Parameters
    ----------
    modality_dims : mapping from modality name to the number of input
        features for that modality. The forward pass expects a dict with the
        same keys.
    d_model, n_heads, n_layers, ff_dim, dropout, max_seq_len : transformer
        hyperparameters (see :class:`TemporalTransformer`).
    n_participants : if > 0, an embedding table of this size is added,
        keyed by an integer participant index. Pass 0 to disable.
    participant_embedding_dim : projection size when participant embeddings
        are enabled; they are linearly mapped up to d_model before being
        added to each timestep.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 90,
        n_participants: int = 0,
        participant_embedding_dim: int = 16,
    ):
        super().__init__()
        if not modality_dims:
            raise ValueError("modality_dims must contain at least one entry")

        self.modality_dims = dict(modality_dims)
        self.d_model = d_model

        # Per-modality input projectors.
        self.projectors = nn.ModuleDict({
            name: _ModalityProjector(dim, d_model, dropout=dropout)
            for name, dim in self.modality_dims.items()
        })

        # Missingness mask embedding: a learnable vector per modality, scaled
        # by the per-step proportion of that modality's features which were
        # imputed (i.e. the mask). It's an alternative to the more common
        # "concatenate mask to inputs" trick and tends to be a bit more
        # parameter-efficient.
        self.mask_embeddings = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(d_model))
            for name in self.modality_dims
        })

        # Optional participant embedding. We keep the embedding table small
        # and then project up to d_model so we don't blow up parameters on
        # large cohorts.
        self.n_participants = int(n_participants)
        if self.n_participants > 0:
            self.participant_embedding = nn.Embedding(self.n_participants, participant_embedding_dim)
            self.participant_proj = nn.Linear(participant_embedding_dim, d_model)
        else:
            self.participant_embedding = None
            self.participant_proj = None

        # Behaviour for participant_idx outside the embedding table: strict
        # at training time (catches index bugs), permissive at inference
        # (unseen participants fall back to the mean embedding).
        self.strict_unknown_participants: bool = True

        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.transformer = TemporalTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # Lightweight attention-pool over time, used to get a single vector
        # per participant trajectory. Mean-pool also works fine; we use a
        # learned-query attention pool because it usually helps a bit.
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )

    def allow_unknown_participants(self) -> None:
        """Route unknown ``participant_idx`` to the mean embedding instead
        of raising. Intended for inference / API code paths."""
        self.strict_unknown_participants = False

    def require_known_participants(self) -> None:
        """Re-enable strict bounds checking on ``participant_idx``."""
        self.strict_unknown_participants = True

    # ------------------------------------------------------------------ inspection

    def count_parameters(self, trainable_only: bool = True) -> dict[str, int]:
        """Return a small parameter-count breakdown for logging.

        ``trainable_only`` controls whether frozen params count.
        """
        def _n(module: nn.Module) -> int:
            return sum(
                p.numel() for p in module.parameters()
                if (not trainable_only) or p.requires_grad
            )

        out = {
            "projectors": sum(_n(m) for m in self.projectors.values()),
            "mask_embeddings": sum(p.numel() for p in self.mask_embeddings.values()),
            "transformer": _n(self.transformer),
            "pool_attn": _n(self.pool_attn) + self.pool_query.numel(),
            "input_norm": _n(self.input_norm),
        }
        if self.participant_embedding is not None:
            out["participant_embedding"] = (
                _n(self.participant_embedding) + _n(self.participant_proj)
            )
        out["total"] = sum(out.values())
        return out

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        modalities: dict[str, torch.Tensor],
        masks: Optional[dict[str, torch.Tensor]] = None,
        participant_idx: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        modalities : dict from modality name to tensor of shape (B, T, F_m).
        masks : optional dict from modality name to (B, T) tensor with values
            in [0, 1] indicating *fraction of features imputed* for that
            modality at that step. Missing modalities default to all-zeros.
        participant_idx : optional (B,) long tensor of participant indices.
            Ignored if the encoder was constructed without participant
            embeddings.
        pad_mask : optional (B, T) bool tensor with True at padded positions.

        Returns
        -------
        dict with keys:
            - 'representation' : (B, d_model)
            - 'per_step'       : (B, T, d_model)
        """
        # Sanity check: every modality the encoder was built for needs an input.
        missing = [k for k in self.modality_dims if k not in modalities]
        if missing:
            raise KeyError(f"missing modality inputs: {missing}")

        # Project each modality and accumulate. Sum > concat in our setting
        # because feature counts per modality are wildly different and a
        # concat blows up d_model.
        B, T, _ = modalities[next(iter(modalities))].shape
        device = modalities[next(iter(modalities))].device
        fused = torch.zeros(B, T, self.d_model, device=device)

        for name, dim in self.modality_dims.items():
            x = modalities[name]
            if x.shape[-1] != dim:
                raise ValueError(
                    f"modality '{name}' expected {dim} features, got {x.shape[-1]}"
                )
            proj = self.projectors[name](x)

            if masks is not None and name in masks:
                m = masks[name].to(proj.dtype).unsqueeze(-1)  # (B, T, 1)
                proj = proj + m * self.mask_embeddings[name].view(1, 1, -1)

            fused = fused + proj

        # Add participant embedding (broadcast over T) if enabled.
        if self.participant_embedding is not None and participant_idx is not None:
            if participant_idx.numel() > 0:
                max_idx = int(participant_idx.max().item())
                if max_idx >= self.n_participants:
                    # Two regimes here:
                    #   - strict_unknown_participants=True (default at training):
                    #     raise so we never silently train on a corrupted index.
                    #   - strict_unknown_participants=False (API/inference):
                    #     route unknown indices to the mean of the learned
                    #     embedding table. This is the right default at serve
                    #     time -- the model still gets a sensible personal
                    #     prior, just one shrunk toward the cohort.
                    if self.strict_unknown_participants:
                        raise IndexError(
                            f"participant_idx contains {max_idx} but the encoder was "
                            f"built with n_participants={self.n_participants}. "
                            f"Pass strict_unknown_participants=False (or call "
                            f"encoder.allow_unknown_participants()) to fall back to "
                            f"the mean embedding instead."
                        )
                    # Clamp out-of-range indices into a safe sentinel slot
                    # and overwrite their embedding with the table mean.
                    in_range = participant_idx < self.n_participants
                    safe_idx = participant_idx.clamp(max=self.n_participants - 1)
                    pe = self.participant_embedding(safe_idx)
                    mean_embedding = self.participant_embedding.weight.mean(dim=0, keepdim=True)
                    # For each row that was OOR, replace with the mean.
                    pe = torch.where(
                        in_range.unsqueeze(-1), pe,
                        mean_embedding.expand_as(pe),
                    )
                else:
                    pe = self.participant_embedding(participant_idx)
            else:
                pe = self.participant_embedding(participant_idx)
            pe = self.participant_proj(pe).unsqueeze(1)  # (B, 1, d_model)
            fused = fused + pe

        fused = self.input_dropout(self.input_norm(fused))

        per_step = self.transformer(fused, key_padding_mask=pad_mask)

        # Attention-pool to a single vector. Repeat the query across batch.
        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(query, per_step, per_step, key_padding_mask=pad_mask)
        representation = pooled.squeeze(1)  # (B, d_model)

        return {"representation": representation, "per_step": per_step}
