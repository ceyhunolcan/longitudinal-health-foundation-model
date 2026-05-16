"""Single source of truth for loading a saved LHFM checkpoint.

Three callers want to rebuild a trained model from disk: the FastAPI
inference service (``lhfm.api.main``), the Streamlit dashboard
(``lhfm.dashboard.app``), and the evaluation/audit scripts
(``scripts/run_fairness_audit.py``, ``scripts/run_climate_holdout.py``,
``scripts/evaluate_model.py``). Each of them used to inline ~30 lines
of "load .pt, load .meta.json sidecar, build encoder, build
downstream, load state dict, switch to eval mode". The copies drifted
across audits. This module owns the canonical loading path.

What gets saved with a checkpoint:

    checkpoints/<run_tag>/
        downstream.pt           torch state_dict (no pickle wrapper)
        downstream.meta.json    architecture + training metadata
        age_reference.json      (optional) {age_ref_mean, age_ref_std}
                                for downstream code that re-standardizes age
        feature_schema.json     (optional) ordered feature columns and
                                modality slices, mirroring what training saw

The .meta.json is the load-bearing one: it tells us how big to make the
encoder, which modality dims map to which feature columns, what the task
names were, and (when present) which participants the encoder learned
embeddings for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LoadedModel:
    """Everything a caller needs to make predictions or evaluate.

    ``model`` is a ``DownstreamRiskModel`` in eval mode. ``meta`` is the
    parsed sidecar. ``age_ref`` and ``feature_schema`` are populated when
    those optional sidecars exist next to the checkpoint.
    """

    model: Any                                # DownstreamRiskModel (torch)
    meta: dict[str, Any]
    age_ref: dict[str, float] | None = None
    feature_schema: dict[str, Any] | None = None
    checkpoint_path: Path | None = None

    @property
    def task_names(self) -> list[str]:
        return list(self.meta.get("task_names", []))

    @property
    def feature_columns(self) -> list[str]:
        return list(self.meta.get("feature_columns", []))

    @property
    def modality_slices(self) -> dict[str, tuple[int, int]]:
        return {k: tuple(v) for k, v in self.meta.get("modality_slices", {}).items()}

    @property
    def window_days(self) -> int:
        return int(self.meta.get("window_days", 14))


def load_downstream(
    checkpoint_path: Path | str,
    *,
    map_location: str = "cpu",
    allow_unknown_participants: bool = True,
) -> LoadedModel:
    """Load a saved downstream checkpoint and put it in eval mode.

    Parameters
    ----------
    checkpoint_path : path to the ``downstream.pt`` file. The sidecar
        ``downstream.meta.json`` must live next to it.
    map_location : forwarded to ``torch.load`` so callers can pull a
        GPU-saved checkpoint onto CPU (the common case for the API).
    allow_unknown_participants : if True, calls
        ``encoder.allow_unknown_participants()`` so inference on
        participants the model never saw at training maps to the mean
        embedding rather than raising IndexError. Default True because
        the API and dashboard need this; eval scripts can set it to
        False if they want strict mode.

    Raises FileNotFoundError if the checkpoint or its meta sidecar is
    missing. Doesn't try to recover from a corrupt sidecar -- if the
    JSON is malformed, the underlying ``json.JSONDecodeError`` propagates.
    """
    import torch
    from lhfm.models.downstream import DownstreamRiskModel
    from lhfm.models.encoder import MultimodalLongitudinalEncoder

    ckpt = Path(checkpoint_path)
    meta_path = ckpt.with_suffix(".meta.json")
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"meta sidecar not found: {meta_path}. "
            f"The checkpoint was saved without architecture metadata and "
            f"can't be reconstructed. Re-train and ensure scripts/train_model.py "
            f"writes the .meta.json sidecar alongside the .pt file."
        )

    meta = json.loads(meta_path.read_text())

    encoder = MultimodalLongitudinalEncoder(
        modality_dims={k: int(v) for k, v in meta["modality_dims"].items()},
        d_model=int(meta["d_model"]),
        n_heads=int(meta["n_heads"]),
        n_layers=int(meta["n_layers"]),
        max_seq_len=int(meta["max_seq_len"]),
        n_participants=int(meta.get("n_participants", 0)),
    )
    if allow_unknown_participants:
        encoder.allow_unknown_participants()

    model = DownstreamRiskModel(encoder=encoder, task_names=meta["task_names"])
    state = torch.load(ckpt, map_location=map_location, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    # Optional sidecars: nice-to-have for downstream code that wants to
    # re-standardize features against training-time stats.
    age_ref = _maybe_load_json(ckpt.parent / "age_reference.json")
    feature_schema = _maybe_load_json(ckpt.parent / "feature_schema.json")

    return LoadedModel(
        model=model,
        meta=meta,
        age_ref=age_ref,
        feature_schema=feature_schema,
        checkpoint_path=ckpt,
    )


def _maybe_load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        # A corrupt sidecar is rare and not worth crashing the API over.
        # The caller can still proceed with meta alone.
        return None
