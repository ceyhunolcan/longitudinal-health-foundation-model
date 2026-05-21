"""PyTorch Dataset wrappers around the windowed numpy arrays.

We keep this minimal. The heavy lifting lives in
``src.data.preprocessing.build_windows`` which produces numpy arrays; here
we just wrap them with the modality-split that the encoder expects.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class LongitudinalWindowDataset(Dataset):
    """Dataset of fixed-length feature windows split by modality.

    Parameters
    ----------
    X : (N, T, F_total) float32 array. Features are expected to be in the
        order returned by :func:`build_windows` for the concatenated
        feature column list.
    y : (N,) or (N, K) float32 array of binary labels. May contain NaN where
        the label is unobserved; the training loop masks these out.
    participant_idx : (N,) int64 array indexing into the encoder's
        participant embedding. Pass None to disable.
    modality_slices : dict from modality name to (start, end) feature-axis
        indices in ``X``.
    masks : optional (N, T, F_total) array of imputation flags (1 where the
        original feature was missing). Per-modality means are computed on
        the fly inside ``__getitem__``.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        modality_slices: dict[str, tuple[int, int]],
        participant_idx: np.ndarray | None = None,
        masks: np.ndarray | None = None,
    ):
        self.X = X.astype(np.float32)
        # Force at least 2D so single-task case still indexes correctly.
        self.y = y.astype(np.float32)
        if self.y.ndim == 1:
            self.y = self.y[:, None]
        self.modality_slices = dict(modality_slices)
        self.participant_idx = (
            participant_idx.astype(np.int64) if participant_idx is not None else None
        )
        self.masks = masks.astype(np.float32) if masks is not None else None

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        x = self.X[i]
        modalities = {}
        per_modality_mask = {}
        for name, (a, b) in self.modality_slices.items():
            modalities[name] = torch.from_numpy(x[:, a:b])
            if self.masks is not None:
                # Mean of the binary "this feature was imputed" flags inside
                # the modality slice, per timestep. So mask value in [0, 1].
                per_modality_mask[name] = torch.from_numpy(
                    self.masks[i, :, a:b].mean(axis=-1)
                )
            else:
                per_modality_mask[name] = torch.zeros(x.shape[0], dtype=torch.float32)

        sample = {
            "modalities": modalities,
            "masks": per_modality_mask,
            "y": torch.from_numpy(self.y[i]),
        }
        if self.participant_idx is not None:
            sample["participant_idx"] = torch.tensor(self.participant_idx[i], dtype=torch.long)
        return sample


def collate_windows(batch: Sequence[dict]) -> dict:
    """Stack a list of samples into a batched dict."""
    out: dict = {}
    out["modalities"] = {
        k: torch.stack([s["modalities"][k] for s in batch], dim=0)
        for k in batch[0]["modalities"]
    }
    out["masks"] = {
        k: torch.stack([s["masks"][k] for s in batch], dim=0)
        for k in batch[0]["masks"]
    }
    out["y"] = torch.stack([s["y"] for s in batch], dim=0)
    if "participant_idx" in batch[0]:
        out["participant_idx"] = torch.stack(
            [s["participant_idx"] for s in batch], dim=0
        )
    return out
