"""Self-supervised pretraining loop.

We keep this single-file and dependency-light on purpose. A research repo
needs to be easy to read top-to-bottom; nothing here imports from
PyTorch Lightning or anything similar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lhfm.models.encoder import MultimodalLongitudinalEncoder
from lhfm.models.self_supervised import (
    SelfSupervisedModel,
    SSLLossWeights,
    make_masked_view,
    ssl_loss,
)
from lhfm.utils.logging import get_logger

from .dataset import LongitudinalWindowDataset, collate_windows

log = get_logger(__name__)


@dataclass
class SSLTrainState:
    encoder: MultimodalLongitudinalEncoder
    ssl_model: SelfSupervisedModel
    train_losses: list[float]
    val_losses: list[float]


def pretrain_ssl(
    encoder: MultimodalLongitudinalEncoder,
    train_dataset: LongitudinalWindowDataset,
    val_dataset: LongitudinalWindowDataset | None = None,
    reconstruction_target_modality: str = "wearable",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 5e-4,
    weight_decay: float = 1e-5,
    mask_ratio: float = 0.20,
    weights: SSLLossWeights | None = None,
    device: str = "cpu",
    num_workers: int = 0,
    checkpoint_path: Path | str | None = None,
    early_stopping_patience: int = 5,
) -> SSLTrainState:
    """Run masked-reconstruction + next-day + contrastive pretraining.

    The contrastive branch consumes two non-overlapping halves of each
    window. If the window is shorter than 4 we silently disable that term.
    """
    weights = weights or SSLLossWeights()
    ssl_model = SelfSupervisedModel(encoder, reconstruction_target_modality=reconstruction_target_modality).to(device)

    optim = torch.optim.AdamW(
        ssl_model.parameters(), lr=lr, weight_decay=weight_decay
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_windows, drop_last=True,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, collate_fn=collate_windows)
        if val_dataset is not None else None
    )

    train_losses, val_losses = [], []
    best_val = float("inf")
    patience = 0

    for epoch in range(epochs):
        ssl_model.train()
        epoch_losses = []
        for batch in train_loader:
            modalities = {k: v.to(device) for k, v in batch["modalities"].items()}
            masks = {k: v.to(device) for k, v in batch["masks"].items()}
            participant_idx = (
                batch["participant_idx"].to(device)
                if "participant_idx" in batch else None
            )

            target_full = modalities[reconstruction_target_modality]  # (B, T, F)
            _B, _T, _ = target_full.shape

            # Build the masked and "second view" inputs.
            target_masked, recon_mask = make_masked_view(target_full, mask_ratio=mask_ratio)
            modalities_a = dict(modalities)
            modalities_a[reconstruction_target_modality] = target_masked

            # Next-day target = the actual last day's target features.
            next_day_target = target_full[:, -1, :]

            # Second view for contrastive: same modalities but with a
            # different random mask. This is a deliberately simple form of
            # augmentation; better augmentations are an obvious extension.
            target_masked_b, _ = make_masked_view(target_full, mask_ratio=mask_ratio)
            modalities_b = dict(modalities)
            modalities_b[reconstruction_target_modality] = target_masked_b

            out_a = ssl_model(modalities_a, masks=masks, participant_idx=participant_idx)
            out_b = ssl_model(modalities_b, masks=masks, participant_idx=participant_idx)

            losses = ssl_loss(
                outputs_a=out_a,
                targets_a={
                    "recon_target": target_full,
                    "recon_mask": recon_mask,
                    "next_day_target": next_day_target,
                },
                outputs_b=out_b,
                weights=weights,
                participant_idx=participant_idx,
            )

            optim.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(ssl_model.parameters(), max_norm=1.0)
            optim.step()

            epoch_losses.append(float(losses["total"].item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        train_losses.append(train_loss)

        val_loss = float("nan")
        if val_loader is not None:
            val_loss = _evaluate_ssl(
                ssl_model, val_loader,
                reconstruction_target_modality=reconstruction_target_modality,
                mask_ratio=mask_ratio, weights=weights, device=device,
            )
            val_losses.append(val_loss)

        log.info(
            "ssl epoch %d/%d train=%.4f val=%.4f",
            epoch + 1, epochs, train_loss, val_loss,
        )

        # Save best by val loss (or by train loss if no val provided).
        track = val_loss if val_loader is not None else train_loss
        if track < best_val - 1e-6:
            best_val = track
            patience = 0
            if checkpoint_path is not None:
                _save_checkpoint(ssl_model, checkpoint_path)
        else:
            patience += 1
            if patience >= early_stopping_patience:
                log.info("early stopping at epoch %d", epoch + 1)
                break

    return SSLTrainState(
        encoder=ssl_model.encoder,
        ssl_model=ssl_model,
        train_losses=train_losses,
        val_losses=val_losses,
    )


@torch.no_grad()
def _evaluate_ssl(model, loader, reconstruction_target_modality, mask_ratio, weights, device):
    """Validation pass. We use a *fixed* mask generator so the validation
    loss is comparable across epochs; otherwise early stopping is noisy
    because each call sees a different random mask.
    """
    model.eval()
    losses = []
    # Deterministic per-call generator. We can't reuse a global one because
    # PyTorch generators don't survive being moved across .to(device) calls
    # reliably; instead we seed it the same way every epoch.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(20240514)

    for batch in loader:
        modalities = {k: v.to(device) for k, v in batch["modalities"].items()}
        masks = {k: v.to(device) for k, v in batch["masks"].items()}
        participant_idx = (
            batch["participant_idx"].to(device)
            if "participant_idx" in batch else None
        )

        target_full = modalities[reconstruction_target_modality]
        # Build the mask on CPU with the seeded generator, then move it over.
        rand = torch.rand(target_full.shape[0], target_full.shape[1], generator=gen)
        recon_mask = (rand < mask_ratio).to(target_full.dtype).to(device)
        target_masked = target_full * (1.0 - recon_mask).unsqueeze(-1)
        modalities_a = dict(modalities)
        modalities_a[reconstruction_target_modality] = target_masked
        next_day_target = target_full[:, -1, :]

        out_a = model(modalities_a, masks=masks, participant_idx=participant_idx)
        loss = ssl_loss(
            outputs_a=out_a,
            targets_a={
                "recon_target": target_full,
                "recon_mask": recon_mask,
                "next_day_target": next_day_target,
            },
            outputs_b=None,                  # skip contrastive at val time
            weights=weights,
        )
        losses.append(float(loss["total"].item()))
    return float(np.mean(losses)) if losses else float("nan")


def _save_checkpoint(model, path):
    """Save only the state_dict (no pickle wrapper) for safe loading."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
