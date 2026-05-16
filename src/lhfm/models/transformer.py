"""Temporal transformer encoder used inside the longitudinal model.

Nothing here is exotic. We use PyTorch's TransformerEncoder under the hood
and add a sinusoidal positional encoding. The interesting part lives in
encoder.py where multiple modalities are fused before this transformer
processes the time dimension.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Stored as a buffer so it moves with the module across devices without
    being treated as a trainable parameter.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to ``x`` of shape (B, T, d_model).

        Raises a clear error if ``T`` exceeds ``max_len``; the alternative is
        a confusing shape-broadcast failure deep inside the transformer.
        """
        T = x.size(1)
        max_len = self.pe.size(1)
        if T > max_len:
            raise ValueError(
                f"sequence length {T} exceeds positional encoding max_len {max_len}; "
                f"reconstruct the encoder with max_seq_len >= {T}"
            )
        return x + self.pe[:, :T]


class TemporalTransformer(nn.Module):
    """Stack of transformer encoder layers operating on (B, T, d_model)."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 90,
    ):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.d_model = d_model

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply positional encoding and run the encoder stack.

        ``x`` is (B, T, d_model). ``key_padding_mask`` if provided should be
        a bool tensor of shape (B, T) with True where tokens should be ignored.
        """
        x = self.pos_enc(x)
        return self.encoder(x, src_key_padding_mask=key_padding_mask)
