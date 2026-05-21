"""Package a trained checkpoint into a release bundle.

Produces ``releases/<run_tag>/`` containing:

    downstream.pt            state-dict only (loadable with weights_only=True)
    downstream.meta.json     architecture + training metadata
    age_reference.json       train-only population stats for age standardization
    feature_columns.json     ordered list of input feature names
    model_card.md            human-readable card (a copy of paper/model_card.md
                             with the run-specific metadata appended)
    metrics_test.csv         per-task metrics + bootstrap CIs (if available)
    SHA256SUMS               integrity manifest
    README.md                short top-level note pointing at the model card

The bundle is everything a reviewer or downstream user needs to:
- Load the model and reproduce a prediction byte-identically.
- Verify the checkpoint hasn't been tampered with (SHA256SUMS).
- Understand training provenance (git SHA, config hash, seed).
- Decide whether to use it (model card + disclaimer).

This tool **does not** upload anywhere. It produces a tree you can
``zip`` and attach to a GitHub release or push to a HuggingFace /
Zenodo dataset by hand. The DOI step happens off-prem.

Usage::

    python scripts/export_release.py
    python scripts/export_release.py --checkpoint checkpoints/downstream-ema-blind.pt
    python scripts/export_release.py --output-dir releases/v0.1.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from lhfm.utils.config import load_config
from lhfm.utils.logging import get_logger

log = get_logger("export_release")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=str, default=None,
                   help="path to downstream.pt (default: checkpoints/downstream.pt)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="directory to write the bundle (default: releases/<run_tag>/)")
    p.add_argument("--metrics-csv", type=str, default=None,
                   help="optional metrics_test.csv to bundle (default: results/tables/metrics_test.csv)")
    return p.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_release_readme(bundle_dir: Path, meta: dict) -> None:
    text = f"""# LHFM release bundle

This directory is a self-contained release bundle for a single training run
of the Longitudinal Health Foundation Model (LHFM).

> **Research prototype. Not a medical device. Synthetic-data-trained.**
> See `model_card.md` and the project's `ACCEPTABLE_USE.md` before any
> downstream use.

## Provenance

- run tag      : `{meta.get("run_tag", "?")}`
- git sha      : `{meta.get("git_sha", "?")}`
- config hash  : `{meta.get("config_hash", "?")}`
- seed         : `{meta.get("training_config", {}).get("seed", "?")}`
- window days  : `{meta.get("window_days", "?")}`
- EMA-blind?   : `{meta.get("exclude_ema_features", False)}`
- params (trainable) : `{(meta.get("param_count") or {}).get("total", "?")}`

## Files

| file                    | purpose                                          |
| ----------------------- | ------------------------------------------------ |
| `downstream.pt`         | state_dict, load with `torch.load(weights_only=True)` |
| `downstream.meta.json`  | architecture + training provenance               |
| `age_reference.json`    | population age stats used for standardization    |
| `feature_columns.json`  | input feature schema                             |
| `metrics_test.csv`      | per-task metrics with participant-clustered 95% bootstrap CIs |
| `model_card.md`         | model card (intended use, limitations, risks)    |
| `SHA256SUMS`            | integrity manifest                               |

## How to load

```python
import json, torch
from lhfm.models.encoder import MultimodalLongitudinalEncoder
from lhfm.models.downstream import DownstreamRiskModel

meta = json.loads(open("downstream.meta.json").read())
enc = MultimodalLongitudinalEncoder(
    modality_dims=meta["modality_dims"],
    d_model=meta["d_model"], n_heads=meta["n_heads"],
    n_layers=meta["n_layers"], max_seq_len=meta["max_seq_len"],
    n_participants=meta["n_participants"],
)
enc.allow_unknown_participants()
model = DownstreamRiskModel(encoder=enc, task_names=meta["task_names"])
model.load_state_dict(torch.load("downstream.pt", map_location="cpu", weights_only=True))
model.eval()
```

## Citation

```bibtex
@software{{lhfm_2025,
  title  = {{Longitudinal Health Foundation Model (LHFM)}},
  author = {{LHFM Contributors}},
  year   = {{2025}},
  url    = {{https://github.com/EXAMPLE-OWNER/longitudinal-health-foundation-model}},
  note   = {{Research prototype. Synthetic data only. Not a medical device.}}
}}
```

For DOI assignment via Zenodo: configure your fork's Zenodo integration,
push a GitHub release with this bundle attached, and Zenodo will mint a
DOI automatically. Update `CITATION.cff` with the DOI once issued.
"""
    (bundle_dir / "README.md").write_text(text)


def _append_run_info_to_model_card(
    bundle_dir: Path, source_card: Path, meta: dict,
) -> None:
    """Copy the paper model card and append the run-specific provenance."""
    base = source_card.read_text() if source_card.exists() else "# Model card\n"
    addendum = f"""

---

## This release

- **Run tag**: `{meta.get("run_tag", "?")}`
- **Git SHA at training time**: `{meta.get("git_sha", "?")}`
- **Config hash**: `{meta.get("config_hash", "?")}`
- **Seed**: `{meta.get("training_config", {}).get("seed", "?")}`
- **SSL epochs**: `{meta.get("training_config", {}).get("epochs_ssl", "?")}`
- **Downstream epochs**: `{meta.get("training_config", {}).get("epochs_downstream", "?")}`
- **Trainable params**: `{(meta.get("param_count") or {}).get("total", "?")}`
- **EMA-blind**: `{meta.get("exclude_ema_features", False)}`. {"Recommended (primary) configuration." if meta.get("exclude_ema_features") else "EMA features included; numbers are subject to the target-leakage caveat documented in `paper/methods.md` §5.1."}
- **Train-cohort age reference**: mean = `{meta.get("age_ref_mean", "?")}`, std = `{meta.get("age_ref_std", "?")}`. The same constants must be re-applied at inference.

### Reproducibility

To reproduce the metrics that ship with this bundle, run on the same git
SHA with the same seed:

```bash
git checkout {meta.get("git_sha", "?")}
python scripts/run_pipeline.py --seed {meta.get("training_config", {}).get("seed", "?")}
python scripts/train_model.py --run-tag {meta.get("run_tag", "?")}
```

The `metrics_test.csv` in this bundle records the held-out test performance
under the participant-clustered bootstrap. Width of CIs is meaningfully
wider than a naive row-level bootstrap on longitudinal data; see
`paper/methods.md` §5.2.
"""
    (bundle_dir / "model_card.md").write_text(base + addendum)


def main() -> int:
    args = parse_args()
    cfg = load_config("default")

    ckpt = Path(args.checkpoint) if args.checkpoint else (
        Path(cfg["paths"]["checkpoint_dir"]) / "downstream.pt"
    )
    meta_path = ckpt.with_suffix(".meta.json")
    if not ckpt.exists() or not meta_path.exists():
        log.error("checkpoint or meta missing: %s, %s", ckpt, meta_path)
        return 2

    meta = json.loads(meta_path.read_text())
    run_tag = meta.get("run_tag") or ckpt.stem

    bundle_dir = Path(args.output_dir) if args.output_dir else (
        Path(cfg["paths"].get("releases_dir") or (ROOT / "releases")) / run_tag
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    log.info("staging release bundle to %s", bundle_dir)

    shutil.copy2(ckpt, bundle_dir / "downstream.pt")
    shutil.copy2(meta_path, bundle_dir / "downstream.meta.json")

    # Pull out the train-only age reference and the feature schema as
    # separate JSONs so downstream consumers don't need to parse the whole
    # meta blob.
    age_ref = {
        "age_ref_mean": meta.get("age_ref_mean"),
        "age_ref_std": meta.get("age_ref_std"),
        "note": (
            "Reference stats computed on the training split only (one row per "
            "participant). Re-apply identically at inference."
        ),
    }
    (bundle_dir / "age_reference.json").write_text(json.dumps(age_ref, indent=2))
    (bundle_dir / "feature_columns.json").write_text(
        json.dumps({
            "feature_columns": meta.get("feature_columns", []),
            "modality_slices": meta.get("modality_slices", {}),
        }, indent=2),
    )

    # Bundle the metrics CSV if it exists.
    metrics_csv = (
        Path(args.metrics_csv) if args.metrics_csv else
        Path(cfg["paths"]["results_dir"]) / "tables" / "metrics_test.csv"
    )
    if metrics_csv.exists():
        shutil.copy2(metrics_csv, bundle_dir / "metrics_test.csv")
    else:
        log.warning("no metrics CSV at %s; bundle will not include test metrics", metrics_csv)

    # Append run-specific info to a copy of the model card.
    source_card = ROOT / "paper" / "model_card.md"
    _append_run_info_to_model_card(bundle_dir, source_card, meta)

    _write_release_readme(bundle_dir, meta)

    # SHA256 manifest for everything in the bundle.
    sums_path = bundle_dir / "SHA256SUMS"
    lines = []
    for f in sorted(bundle_dir.iterdir()):
        if f.name == "SHA256SUMS" or f.is_dir():
            continue
        lines.append(f"{_sha256(f)}  {f.name}")
    sums_path.write_text("\n".join(lines) + "\n")

    log.info("release bundle ready: %s (%d files)",
             bundle_dir, sum(1 for _ in bundle_dir.iterdir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
