"""Single-entry CLI for LHFM.

After ``pip install -e .`` the ``lhfm`` command becomes available::

    lhfm --help                            # list subcommands
    lhfm pipeline --adapter synthetic      # the data pipeline
    lhfm pipeline --adapter lifesnaps --raw-dir data/raw/lifesnaps
    lhfm train --participants 100
    lhfm evaluate                          # held-out test metrics
    lhfm fairness-audit                    # subgroup-stratified audit
    lhfm climate-holdout --holdout heat_wave
    lhfm scale-ablation                    # AUROC-vs-N curve
    lhfm export-release                    # bundle a checkpoint
    lhfm dashboard                         # launch streamlit
    lhfm version

Each subcommand dispatches to the corresponding ``scripts/*.py`` module's
``main()`` function. The scripts also work as direct ``python
scripts/X.py ...`` invocations -- the CLI just gives a friendlier surface.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

# Subcommand name -> (script module name, one-line help text)
_SUBCOMMANDS: dict[str, tuple[str, str]] = {
    "demo":            ("demo",                  "60-second end-to-end demo on a tiny synthetic cohort"),
    "pipeline":        ("run_pipeline",         "Build features from raw cohort data"),
    "train":           ("train_model",          "Pretrain SSL and fit downstream heads"),
    "evaluate":        ("evaluate_model",       "Held-out test metrics for a checkpoint"),
    "fairness-audit":  ("run_fairness_audit",   "Subgroup-stratified fairness audit"),
    "climate-holdout": ("run_climate_holdout",  "Climate-regime generalization eval"),
    "scale-ablation":  ("run_scale_ablation",   "Pretraining cohort-size sweep"),
    "export-release":  ("export_release",       "Bundle a checkpoint for release"),
    "dashboard":       ("launch_dashboard",     "Launch the Streamlit dashboard"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lhfm",
        description="Longitudinal Health Foundation Model CLI",
        # Hide the long subcommand-arg help here; each subcommand prints
        # its own --help and we want the top-level page to stay readable.
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_format_subcommand_list(),
    )
    parser.add_argument(
        "subcommand",
        nargs="?",
        choices=[*_SUBCOMMANDS, "version", "help"],
        help="subcommand to run (lhfm <subcommand> --help for details)",
    )
    parser.add_argument(
        "rest", nargs=argparse.REMAINDER,
        help="arguments forwarded to the subcommand",
    )

    args = parser.parse_args(argv)
    if args.subcommand in (None, "help"):
        parser.print_help()
        return 0
    if args.subcommand == "version":
        from lhfm import __version__
        print(f"lhfm {__version__}")
        return 0

    return _dispatch(args.subcommand, args.rest)


def _dispatch(subcommand: str, rest: list[str]) -> int:
    """Import the matching scripts/*.py and call its ``main()``."""
    script_name, _ = _SUBCOMMANDS[subcommand]

    # The scripts live outside the importable package -- they're a sibling
    # to ``src/``. We add ``scripts/`` to sys.path on the fly so the
    # ``import scripts.X`` succeeds whether the user is running from the
    # repo root, from inside a Docker container, or from an installed wheel
    # that doesn't ship the scripts (in which case we fail with a clear
    # message rather than mysteriously).
    repo_root = _find_repo_root()
    if repo_root is None:
        sys.stderr.write(
            "lhfm: cannot locate scripts/ directory. The CLI subcommands "
            "are thin wrappers around scripts/*.py; either run lhfm from "
            "inside a checkout of the repo, or invoke the scripts directly "
            "with `python -m`.\n"
        )
        return 2

    scripts_dir = repo_root / "scripts"
    if not scripts_dir.exists():
        sys.stderr.write(f"lhfm: no scripts/ directory at {scripts_dir}\n")
        return 2

    sys.path.insert(0, str(scripts_dir))
    try:
        module = importlib.import_module(script_name)
    except ImportError as exc:
        sys.stderr.write(f"lhfm: failed to import {script_name}: {exc}\n")
        return 2

    if not hasattr(module, "main"):
        sys.stderr.write(
            f"lhfm: {script_name}.py has no main() function to dispatch to\n"
        )
        return 2

    # Rebuild argv so argparse inside the script sees what it expects:
    # the script name as argv[0] and the rest of the arguments after it.
    saved_argv = sys.argv
    sys.argv = [script_name, *rest]
    try:
        result = module.main()
    finally:
        sys.argv = saved_argv
    return int(result) if isinstance(result, int) else 0


def _find_repo_root() -> Path | None:
    """Walk up from the package install location to find ``scripts/``.

    For ``pip install -e .`` this finds the source checkout. For ``pip
    install .`` from a wheel the scripts won't be there and the user
    should call them directly.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "scripts").is_dir() and (parent / "pyproject.toml").exists():
            return parent
    # Also try the current working directory as a last resort.
    cwd = Path.cwd()
    if (cwd / "scripts").is_dir() and (cwd / "pyproject.toml").exists():
        return cwd
    return None


def _format_subcommand_list() -> str:
    """Pretty list for the --help epilog."""
    width = max(len(name) for name in _SUBCOMMANDS)
    lines = ["", "subcommands:"]
    for name, (_, blurb) in _SUBCOMMANDS.items():
        lines.append(f"  {name:<{width}}   {blurb}")
    lines.append(f"  {'version':<{width}}   Print version and exit")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
