"""Pytest fixtures shared across the test modules."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make src.* importable regardless of how pytest is invoked.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def tiny_cohort():
    """Small synthetic dataframe used by most tests. Cached for speed."""
    from lhfm.data.synthetic_generator import generate_synthetic_cohort
    return generate_synthetic_cohort(n_participants=10, n_days=30, seed=123)


@pytest.fixture(scope="session")
def engineered(tiny_cohort):
    from lhfm.features import build_full_feature_table
    return build_full_feature_table(tiny_cohort, impute=True, add_targets=True)
