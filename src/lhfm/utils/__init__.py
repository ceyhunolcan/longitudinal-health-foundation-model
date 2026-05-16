"""Shared utilities: config loading, logging, metrics, plotting, fairness."""

from .climate_regimes import (
    CLIMATE_REGIMES,
    define_climate_regime,
    regime_summary,
    split_train_eval_by_regime,
)
from .config import load_config, resolve_device, set_global_seed
from .fairness import (
    check_fairness_thresholds,
    fairness_report_to_csv,
    run_fairness_audit,
)
from .logging import get_logger
from .metrics import (
    binary_classification_report,
    bootstrap_ci,
    expected_calibration_error,
    reliability_curve,
)

__all__ = [
    "load_config",
    "resolve_device",
    "set_global_seed",
    "get_logger",
    "binary_classification_report",
    "bootstrap_ci",
    "expected_calibration_error",
    "reliability_curve",
    "run_fairness_audit",
    "fairness_report_to_csv",
    "check_fairness_thresholds",
    "CLIMATE_REGIMES",
    "define_climate_regime",
    "regime_summary",
    "split_train_eval_by_regime",
]
