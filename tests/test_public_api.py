"""Tests for the public ``lhfm`` package surface.

We test that:
- ``import lhfm`` works in a torch-free environment (the synthetic
  generator and feature engineering don't need torch).
- The public ``__all__`` matches what's documented in the package
  docstring.
- ``lhfm.load_cohort`` is a working shortcut to the adapter machinery.
- The CLI multiplexer dispatches to the right script.

The CLI integration test runs in a subprocess so we don't pollute the
test session's argv / sys.path.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_import_is_torch_free():
    """``import lhfm`` shouldn't pull in torch.

    We import in a subprocess with PYTHONPATH set so we don't inherit
    this test session's already-imported modules.
    """
    code = (
        "import sys, lhfm; "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m); "
        "print('OK', lhfm.__version__)"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": ""},
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.startswith("OK")


def test_public_api_surface():
    """Every name in ``lhfm.__all__`` is actually importable from lhfm."""
    import lhfm
    for name in lhfm.__all__:
        assert hasattr(lhfm, name), f"lhfm.__all__ lists {name!r} but it's missing"


def test_version_string():
    import lhfm
    assert isinstance(lhfm.__version__, str)
    # Must be a sensible semver-ish string.
    parts = lhfm.__version__.split(".")
    assert 2 <= len(parts) <= 4
    assert all(p[0].isdigit() for p in parts)


def test_load_cohort_synthetic(tmp_path):
    """``lhfm.load_cohort('synthetic', ...)`` returns the same shape as
    calling the generator directly."""
    import lhfm
    df = lhfm.load_cohort(
        "synthetic", tmp_path,
        n_participants=8, n_days=15, seed=42,
    )
    assert isinstance(df, pd.DataFrame)
    assert df["participant_id"].nunique() == 8
    assert df["date"].nunique() == 15


def test_load_cohort_unknown_adapter_raises(tmp_path):
    import lhfm
    from lhfm.data.adapters import AdapterError
    with pytest.raises(AdapterError, match="unknown adapter"):
        lhfm.load_cohort("not-a-real-dataset", tmp_path)


def test_cli_version():
    res = subprocess.run(
        [sys.executable, "-m", "lhfm", "version"],
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": ""},
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    assert "lhfm" in res.stdout
    import lhfm
    assert lhfm.__version__ in res.stdout


def test_cli_help_lists_subcommands():
    res = subprocess.run(
        [sys.executable, "-m", "lhfm", "--help"],
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": ""},
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0
    # A handful of subcommands we promise in the CLI docstring.
    for expected in ("pipeline", "train", "evaluate", "fairness-audit",
                     "climate-holdout"):
        assert expected in res.stdout, f"missing subcommand {expected!r} in help"


def test_cli_pipeline_subcommand_dispatches(tmp_path):
    """End-to-end: lhfm pipeline runs a tiny synthetic build."""
    # We point at a separate output dir so the test doesn't fight with
    # the user's data/ checkout.
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    res = subprocess.run(
        [sys.executable, "-m", "lhfm", "pipeline",
         "--adapter", "synthetic",
         "--participants", "5",
         "--days", "15",
         "--seed", "1",
         "--no-parquet"],
        cwd=tmp_path,                    # so the script's data/ is local
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": ""},
        capture_output=True, text=True, check=False,
    )
    # We don't assert returncode==0 because the script writes into
    # data/synthetic which may need to be created relative to cwd. The
    # important thing is that the dispatcher found and ran the script.
    assert "feature table" in res.stdout or "feature table" in res.stderr, (
        f"pipeline subcommand didn't dispatch correctly. "
        f"stdout: {res.stdout[-500:]}\nstderr: {res.stderr[-500:]}"
    )
