"""End-to-end data pipeline: source -> validated -> engineered -> on disk.

Usage::

    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --participants 100 --days 60 --seed 7

Real-data usage::

    python scripts/run_pipeline.py --adapter lifesnaps --raw-dir /data/lifesnaps
    python scripts/run_pipeline.py --adapter globem --raw-dir /data/globem
    python scripts/run_pipeline.py --adapter lifesnaps --raw-dir ... --preflight

Outputs:
    data/synthetic/cohort.csv           raw long-form cohort (synthetic only)
    data/synthetic/cohort.manifest.json small descriptive manifest
    data/processed/features.parquet     fully engineered feature table
    data/processed/features.csv         same, but as CSV for tools that hate parquet
    data/processed/pipeline_summary.json metadata + preflight report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `lhfm.*` importable when run as a script from any directory.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from lhfm.data.adapters import (                                            # noqa: E402
    AdapterConfig,
    get_adapter,
    list_adapters,
    preflight_report,
)
from lhfm.data.validation import validate_synthetic_dataframe               # noqa: E402
from lhfm.features import build_full_feature_table                          # noqa: E402
from lhfm.utils.config import load_config, set_global_seed                  # noqa: E402
from lhfm.utils.logging import get_logger                                   # noqa: E402


log = get_logger("pipeline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--adapter", default="synthetic",
        choices=list_adapters(),
        help="data source adapter (use 'synthetic' for the built-in generator)",
    )
    p.add_argument(
        "--raw-dir", type=str, default=None,
        help="path to the raw dataset directory (required for real-data adapters)",
    )
    p.add_argument("--participants", type=int, default=None,
                   help="synthetic only: number of participants")
    p.add_argument("--days", type=int, default=None,
                   help="synthetic only: number of days per participant")
    p.add_argument("--max-participants", type=int, default=None,
                   help="cap participants (useful for smoke runs on big real datasets)")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed (overrides config)")
    p.add_argument("--no-parquet", action="store_true",
                   help="skip writing the parquet copy (CSV only)")
    p.add_argument("--strict-validation", action="store_true",
                   help="treat plausibility warnings as errors")
    p.add_argument("--preflight", action="store_true",
                   help="run adapter + preflight report only; don't engineer features")
    p.add_argument("--no-weather", action="store_true",
                   help="skip weather enrichment (faster for offline testing)")
    p.add_argument("--default-lat", type=float, default=None,
                   help="study-site latitude when adapter lacks per-row coords")
    p.add_argument("--default-lon", type=float, default=None,
                   help="study-site longitude when adapter lacks per-row coords")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config("default")

    seed = args.seed if args.seed is not None else cfg["project"]["seed"]
    set_global_seed(seed)

    synthetic_dir = Path(cfg["paths"]["synthetic_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Resolve adapter config.
    raw_dir = Path(args.raw_dir) if args.raw_dir else synthetic_dir
    cache_dir = ROOT / "data" / "cache"
    adapter_cfg = AdapterConfig(
        raw_dir=raw_dir,
        cache_dir=cache_dir,
        enrich_weather=not args.no_weather,
        default_lat=args.default_lat,
        default_lon=args.default_lon,
        max_participants=args.max_participants,
    )

    AdapterCls = get_adapter(args.adapter)
    if args.adapter == "synthetic":
        n_participants = args.participants or cfg["data"]["n_participants"]
        n_days = args.days or cfg["data"]["n_days"]
        adapter = AdapterCls(
            adapter_cfg, n_participants=n_participants, n_days=n_days, seed=seed,
        )
        log.info("using synthetic adapter (n=%d × %d days, seed=%d)",
                 n_participants, n_days, seed)
    else:
        adapter = AdapterCls(adapter_cfg)
        log.info("using %s adapter on %s", args.adapter, raw_dir)

    raw = adapter.build()
    log.info("adapter produced: %d rows × %d cols, %d participants",
             len(raw), raw.shape[1], raw["participant_id"].nunique())

    # Preflight: report and stop.
    if args.preflight:
        pre = preflight_report(raw, min_days=cfg["training"]["window_days"] + 1)
        out_path = processed_dir / "preflight_report.json"
        out_path.write_text(json.dumps(pre, indent=2, default=str))
        log.info("preflight report written to %s", out_path)
        log.info("PREFLIGHT SUMMARY:\n%s", json.dumps(pre, indent=2, default=str))
        return 0

    # Save raw for downstream tools / reproducibility.
    raw_path = synthetic_dir / "cohort.csv"
    raw.to_csv(raw_path, index=False)
    log.info("wrote raw cohort to %s (%d rows)", raw_path, len(raw))

    # Validate.
    log.info("validating raw dataframe")
    report = validate_synthetic_dataframe(raw, strict=args.strict_validation)
    for w in report.warnings:
        log.warning("validation warning: %s", w)
    report.raise_if_failed()
    log.info("validation summary: %s", report.summary)

    # Engineer features.
    log.info("engineering features")
    feat = build_full_feature_table(raw, impute=True, add_targets=True)
    log.info("feature table: %d rows × %d columns", *feat.shape)

    csv_out = processed_dir / "features.csv"
    feat.to_csv(csv_out, index=False)
    log.info("wrote %s", csv_out)

    if not args.no_parquet:
        try:
            parquet_out = processed_dir / "features.parquet"
            feat.to_parquet(parquet_out, index=False)
            log.info("wrote %s", parquet_out)
        except Exception as exc:
            log.warning("parquet write failed (%s); CSV is still available", exc)

    # Tiny machine-readable summary so downstream tools can pick up metadata.
    pre = preflight_report(feat, min_days=cfg["training"]["window_days"] + 1)
    meta = {
        "adapter": args.adapter,
        "seed": seed,
        "n_rows": int(len(feat)),
        "n_columns": int(feat.shape[1]),
        "target_columns": [c for c in feat.columns if c.startswith("target_")],
        "validation_summary": report.summary,
        "preflight": pre,
    }
    (processed_dir / "pipeline_summary.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
