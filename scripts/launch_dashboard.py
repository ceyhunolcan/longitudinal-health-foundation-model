"""Convenience launcher for the Streamlit dashboard.

This is essentially a thin wrapper around::

    streamlit run src/lhfm/dashboard/app.py

but it lives at scripts/ so it's reachable next to the other entry points
and so it can fix up PYTHONPATH for users who haven't installed the
package in editable mode.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from subprocess import run

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "src" / "lhfm" / "dashboard" / "app.py"


def main() -> int:
    env = os.environ.copy()
    # Both ROOT and ROOT/src on PYTHONPATH so `from lhfm.*` and `from scripts.*`
    # both resolve regardless of where streamlit chdir'd us.
    extra = os.pathsep.join([str(ROOT / "src"), str(ROOT)])
    env["PYTHONPATH"] = extra + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP_PATH)]
    cmd.extend(sys.argv[1:])
    return run(cmd, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
