"""Train the downstream multi-task risk heads on top of the SSL encoder.

The same encoder is shared across all tasks. Each task has its own binary
cross-entropy head. Targets are allowed to contain NaN for "label missing":
those positions are masked out of the loss.

Class imbalance is handled with per-task ``pos_weight`` values computed
once from the training labels and reused throughout training. Without this,
the rare-positive tasks (climate_vulnerable in particular) collapse to
predicting ``0`` and AUROC degenerates.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lhfm.models.downstream import DownstreamRiskModel
from lhfm.models.encoder import MultimodalLongitudinalEncoder
from lhfm.utils.logging import get_logger
from lhfm.utils.metrics import binary_classification_report

from .dataset import LongitudinalWindowDataset, collate_windows

log = get_logger(__name__)


@dataclass
class DownstreamTrainState:
    model: DownstreamRiskModel
    train_losses: list[float]
    val_losses: list[float]
    val_metrics: list[dict]
    pos_weights: dict[str, float]


def _masked_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """BCE-with-logits that ignores NaN targets.

    Returns a scalar; if no valid targets exist in the batch we return 0
    (and the caller will see no gradient flow which is exactly right).
    """
    valid = ~torch.isnan(targets)
    if not valid.any():
        return torch.zeros((), device=logits.device, requires_grad=True)
    return F.binary_cross_entropy_with_logits(
        logits[valid], targets[valid].float(),
        pos_weight=pos_weight,
    )


def _compute_pos_weights(
    dataset: LongitudinalWindowDataset,
    task_names: list[str],
    cap: float = 20.0,
) -> dict[str, float]:
    """Compute per-task pos_weight = #neg / #pos on the training set.

    Capped at ``cap`` so that tasks with very few positives (e.g. a handful
    of climate-vulnerable days) don't destabilize training.
    """
    y = dataset.y  # (N, K)
    out: dict[str, float] = {}
    for i, name in enumerate(task_names):
        col = y[:, i]
        valid = ~np.isnan(col)
        n_pos = float(np.nansum(col[valid]))
        n_neg = float(valid.sum() - n_pos)
        if n_pos < 1 or n_neg < 1:
            out[name] = 1.0
            continue
        out[name] = float(min(cap, n_neg / n_pos))
    return out


def train_downstream(
    encoder: MultimodalLongitudinalEncoder,
    task_names: Iterable[str],
    train_dataset: LongitudinalWindowDataset,
    val_dataset: LongitudinalWindowDataset | None = None,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 5e-4,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    num_workers: int = 0,
    freeze_encoder: bool = False,
    checkpoint_path: Path | str | None = None,
    early_stopping_patience: int = 5,
) -> DownstreamTrainState:
    """Train ``DownstreamRiskModel`` and return the trained instance."""
    task_names = list(task_names)
    model = DownstreamRiskModel(
        encoder=encoder, task_names=task_names, freeze_encoder=freeze_encoder
    ).to(device)
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )

    # Per-task pos_weights: #neg / #pos on the training set, capped at 20
    # so that very rare tasks don't blow up. Sent to the model device once.
    pos_weights = _compute_pos_weights(train_dataset, task_names)
    log.info("class imbalance pos_weights: %s",
             {k: round(v, 3) for k, v in pos_weights.items()})
    pos_weight_tensors = {
        name: torch.tensor(pw, dtype=torch.float32, device=device)
        for name, pw in pos_weights.items()
    }

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_windows, drop_last=False,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, collate_fn=collate_windows)
        if val_dataset is not None else None
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_metrics: list[dict] = []

    # Track val-AUROC averaged over evaluable tasks. NaN aurocs (tasks with
    # all-zero validation labels) are dropped; if no task is evaluable we
    # fall back to the primary task alone. This makes the saved encoder
    # the one that best serves *the multi-task average* rather than any
    # single head, which was the behaviour before this patch and tended to
    # halt training while non-primary heads were still improving (visible
    # on the synthetic_paper run, where climate_vulnerable val AUROC
    # climbed from 0.67 to 0.94 across epochs 1-6 while low_mood plateaued).
    primary_task = task_names[0] if task_names else None
    best_score = float("-inf")
    patience = 0

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for batch in train_loader:
            modalities = {k: v.to(device) for k, v in batch["modalities"].items()}
            masks = {k: v.to(device) for k, v in batch["masks"].items()}
            participant_idx = (
                batch["participant_idx"].to(device)
                if "participant_idx" in batch else None
            )
            y = batch["y"].to(device)  # (B, K)

            logits = model(modalities, masks=masks, participant_idx=participant_idx)
            total = torch.zeros((), device=device)
            for i, name in enumerate(task_names):
                total = total + _masked_bce(
                    logits[name], y[:, i],
                    pos_weight=pos_weight_tensors[name],
                )
            total = total / max(1, len(task_names))

            optim.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0,
            )
            optim.step()
            batch_losses.append(float(total.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        train_losses.append(train_loss)

        val_loss = float("nan")
        metric_summary = {}
        if val_loader is not None:
            val_loss, metric_summary = _validate_downstream(
                model, val_loader, task_names, device
            )
            val_losses.append(val_loss)
            val_metrics.append(metric_summary)

        log.info(
            "downstream epoch %d/%d train=%.4f val=%.4f  %s",
            epoch + 1, epochs, train_loss, val_loss,
            " ".join(f"{k}_auroc={v.get('auroc', float('nan')):.3f}"
                     for k, v in metric_summary.items()),
        )

        # Score = mean val AUROC across tasks with at least one positive in
        # the validation fold (i.e. AUROC is well-defined). NaN-safe.
        aurocs = []
        for t in task_names:
            a = metric_summary.get(t, {}).get("auroc", float("nan"))
            if a == a:  # not NaN
                aurocs.append(a)
        if aurocs:
            track = sum(aurocs) / len(aurocs)
        else:
            # Fallback: no task evaluable on val. Use primary if it has any
            # value at all, else -inf.
            primary_auroc = metric_summary.get(primary_task, {}).get("auroc", float("nan")) if primary_task else float("nan")
            track = primary_auroc if primary_auroc == primary_auroc else float("-inf")
        if track > best_score + 1e-6:
            best_score = track
            patience = 0
            if checkpoint_path is not None:
                _save_checkpoint(model, checkpoint_path)
        else:
            patience += 1
            if patience >= early_stopping_patience:
                log.info("early stopping at epoch %d (best mean-over-tasks val AUROC=%.3f)",
                         epoch + 1, best_score)
                break

    return DownstreamTrainState(
        model=model, train_losses=train_losses,
        val_losses=val_losses, val_metrics=val_metrics,
        pos_weights=pos_weights,
    )


@torch.no_grad()
def _validate_downstream(model, loader, task_names, device):
    """Validation pass. We compute unweighted BCE here so the value is
    comparable to the per-batch training average regardless of how the
    pos_weights change between runs.
    """
    model.eval()
    losses = []
    preds = {n: [] for n in task_names}
    targs = {n: [] for n in task_names}

    for batch in loader:
        modalities = {k: v.to(device) for k, v in batch["modalities"].items()}
        masks = {k: v.to(device) for k, v in batch["masks"].items()}
        participant_idx = (
            batch["participant_idx"].to(device)
            if "participant_idx" in batch else None
        )
        y = batch["y"].to(device)

        logits = model(modalities, masks=masks, participant_idx=participant_idx)
        batch_loss = torch.zeros((), device=device)
        for i, name in enumerate(task_names):
            batch_loss = batch_loss + _masked_bce(logits[name], y[:, i])
            prob = torch.sigmoid(logits[name]).cpu().numpy()
            tgt = y[:, i].cpu().numpy()
            preds[name].append(prob)
            targs[name].append(tgt)
        losses.append(float((batch_loss / max(1, len(task_names))).item()))

    summary = {}
    for n in task_names:
        p = np.concatenate(preds[n]) if preds[n] else np.empty(0)
        t = np.concatenate(targs[n]) if targs[n] else np.empty(0)
        valid = ~np.isnan(t)
        if valid.sum() < 2 or len(np.unique(t[valid])) < 2:
            summary[n] = {"auroc": float("nan"), "auprc": float("nan"), "f1": float("nan")}
            continue
        summary[n] = binary_classification_report(t[valid], p[valid])
    return float(np.mean(losses)) if losses else float("nan"), summary


def _save_checkpoint(model, path):
    """Save only the state_dict so the API can torch.load(weights_only=True)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
