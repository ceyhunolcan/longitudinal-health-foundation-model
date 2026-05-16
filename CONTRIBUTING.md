# Contributing

Thanks for your interest. A few ground rules before you dive in.

## Scope

LHFM is a **research prototype**. We're happy to merge PRs that:

- fix bugs, especially anything that affects reproducibility or makes a
  metric reported in the README inaccurate,
- improve test coverage, particularly for edge cases (empty dataframes,
  all-NaN columns, single-participant inference),
- extend the synthetic generator to model phenomena currently absent
  (cold weather, hormonal cycles, medication effects, sensor noise floors),
- add real-data adapters (PhysioNet, GLOBEM, etc.) behind a clean interface,
- improve documentation, particularly the methods, ethics, and limitations
  pages.

We are **less likely to merge** PRs that:

- pursue clinical applications,
- add features that increase the project's surface area without
  documentation or tests,
- introduce heavy dependencies (anything that breaks `pip install -r
  requirements.txt` on a stock laptop is suspect).

## Workflow

1. Open an issue first for anything beyond a small bug fix. Briefly
   describe the change and the motivation.
2. Fork → branch → commit → PR. Conventional Commits (`feat:`, `fix:`,
   `docs:`, ...) preferred but not required.
3. Make sure `pytest -q` passes locally before opening the PR.
4. CI must be green before merge.

## Style

- Python 3.11+.
- Code formatting / linting via `ruff` (the config lives in `pyproject.toml`).
- Public functions and classes get docstrings. Internal helpers can skip
  them if the name is clearly self-explanatory.
- New numeric code lives behind a unit test. We're particularly strict
  about anything that could silently produce NaN.
- No real patient data. Period.

## Ethics

If your contribution materially expands the model's behavioural-monitoring
capabilities or its applicability to real people, please discuss with the
maintainers first and re-read `paper/ethics.md`. We will decline PRs that
push the project toward surveillance, deployment without consent, or use
on minors.

## Dev setup

```bash
git clone <your fork>
cd longitudinal-health-foundation-model

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-cov ruff

# fast checks
pytest -q
ruff check src tests scripts

# full pipeline smoke
python scripts/run_pipeline.py --participants 30 --days 45
python scripts/train_model.py --ssl-epochs 2 --downstream-epochs 2 --no-baselines
```
