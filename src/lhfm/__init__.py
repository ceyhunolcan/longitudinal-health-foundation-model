"""Longitudinal Health Foundation Model (LHFM).

Self-supervised multimodal modeling of wearable, smartphone, and
environmental signals for personalized behavioral-health risk prediction.

Research prototype. Synthetic data + public-cohort adapters. Not a
medical device. See ``ACCEPTABLE_USE.md`` before using.

Quickstart::

    import lhfm

    # 1. Generate a synthetic cohort
    df = lhfm.generate_synthetic_cohort(n_participants=100, n_days=60)

    # 2. Or load a real cohort via an adapter
    df = lhfm.load_cohort("lifesnaps", raw_dir="data/raw/lifesnaps")

    # 3. Build the feature table
    features = lhfm.build_full_feature_table(df)

    # 4. From here on, see scripts/ for training, eval, and the CLI:
    #    `lhfm pipeline`, `lhfm train`, `lhfm evaluate`, ...

The version string lives here. ``pyproject.toml`` reads it via
``[tool.setuptools.dynamic]`` and ``CITATION.cff`` is kept in sync by
``scripts/bump_version.py`` (a tiny one-liner).
"""

from __future__ import annotations

__version__ = "0.2.0"


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------
# The functions below are the ones used in 90% of notebooks and scripts.
# Everything else is reachable via fully-qualified imports
# (``from lhfm.utils.fairness import run_fairness_audit``, etc.).

from lhfm.data.synthetic_generator import (
    GeneratorConfig,
    SyntheticCohortGenerator,
    generate_synthetic_cohort,
)
from lhfm.data.validation import validate_synthetic_dataframe
from lhfm.data.preprocessing import (
    build_windows,
    train_val_test_split_by_participant,
    binarize_targets,
)
from lhfm.features import build_full_feature_table
from lhfm.data.adapters import (
    AdapterConfig,
    get_adapter,
    list_adapters,
    preflight_report,
)


def load_downstream_checkpoint(checkpoint_path, **kwargs):
    """Load a trained downstream checkpoint. See ``lhfm.checkpoints.load_downstream``.

    Lazy-imports torch so ``import lhfm`` stays cheap for non-torch users
    (notebooks that only need the synthetic generator, lint jobs, docs).
    """
    from lhfm.checkpoints import load_downstream
    return load_downstream(checkpoint_path, **kwargs)


def load_cohort(adapter_name: str, raw_dir, **kwargs):
    """One-line cohort loader. Returns a long-form dataframe.

    >>> df = lhfm.load_cohort("synthetic", "/tmp/scratch", n_participants=50, n_days=30)
    >>> df = lhfm.load_cohort("lifesnaps", "data/raw/lifesnaps")
    >>> df = lhfm.load_cohort("globem", "data/raw/globem", enrich_weather=False)

    For anything beyond the defaults (custom AdapterConfig, weather CSV,
    site coordinates, etc.) instantiate the adapter directly:

        >>> from lhfm.data.adapters import AdapterConfig, get_adapter
        >>> cfg = AdapterConfig(raw_dir=..., default_lat=47.66, default_lon=-122.31)
        >>> df = get_adapter("globem")(cfg).build()
    """
    from pathlib import Path

    AdapterCls = get_adapter(adapter_name)
    # Split kwargs: AdapterConfig-known vs adapter-specific.
    cfg_fields = {"cache_dir", "enrich_weather", "weather_provider",
                  "weather_csv", "default_lat", "default_lon", "max_participants"}
    cfg_kwargs = {k: v for k, v in kwargs.items() if k in cfg_fields}
    other_kwargs = {k: v for k, v in kwargs.items() if k not in cfg_fields}
    cfg = AdapterConfig(raw_dir=Path(raw_dir), **cfg_kwargs)
    return AdapterCls(cfg, **other_kwargs).build()


__all__ = [
    "__version__",
    # cohort sources
    "GeneratorConfig",
    "SyntheticCohortGenerator",
    "generate_synthetic_cohort",
    "load_cohort",
    "AdapterConfig",
    "get_adapter",
    "list_adapters",
    "preflight_report",
    # preprocessing + features
    "validate_synthetic_dataframe",
    "build_windows",
    "train_val_test_split_by_participant",
    "binarize_targets",
    "build_full_feature_table",
    # trained-model loading
    "load_downstream_checkpoint",
]
