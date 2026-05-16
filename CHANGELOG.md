# Changelog

All notable changes to this project will be documented here.

The format roughly follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Audit-6 (public-facing polish: demo, Colab, dashboard timeline, architecture doc)

Final pass before pushing to GitHub. Four targeted improvements aimed at
how the repo lands when a stranger opens it for the first time.

- **`make demo`** (60-second end-to-end on CPU). New `scripts/demo.py`
  walks generate → features → tiny SSL+downstream train → bootstrap-CI
  AUROC in four labelled stages, one screen of output. Hooked into the
  Makefile as the first pipeline target. This is what someone reads
  before they read anything else.
- **Mermaid architecture diagram** replaces the ASCII boxes in the
  README. Renders inline on GitHub with colour-coded data → feature →
  encoder → head → audit flow. Source at
  [`docs/figures/architecture.mmd`](docs/figures/architecture.mmd).
- **`docs/architecture.md`** — one-page rationale doc explaining the
  *why* of each design choice (why a transformer not RNN, why per-
  modality projectors, why attention-pool, why optional participant
  embeddings, why 14-day windows, why these four tasks), with a table
  mapping each decision to the file where it lives in source.
- **Colab quickstart notebook** (`notebooks/00_quickstart.ipynb`,
  22 cells). Self-contained: clones the repo, installs deps, runs the
  synthetic generator → feature pipeline → tiny train → bootstrap eval
  → integrated-gradients attribution plot → subgroup fairness audit.
  Linked from a "Open in Colab" badge in the README. ~3 min on free
  Colab CPU.
- **Dashboard interpretability timeline**: the dashboard now leads with
  a two-panel risk-and-driver visualisation per participant. Top panel
  is the per-day predicted-risk curve with bands shaded by risk level;
  bottom panel is a signed-attribution heatmap (rows are top features,
  cells coloured red when the feature pushed risk up, blue when it
  pushed it down). A focus-day marker and below-chart driver readout
  make the whole thing scrubbable. Falls back to rule-based attributions
  when no trained checkpoint is wired in, so the panel works on a fresh
  checkout. A task selector (low_mood / high_stress / sleep_disruption /
  climate_vulnerable) sits above the timeline.

#### README updates

- Top-of-page "Open in Colab" badge.
- New "60-second demo" section right under the warning banner, with the
  expected output as inline text so visitors know what to expect.
- ASCII architecture diagram swapped for the Mermaid version.

### Audit-5 (top-level API + CLI + consolidation)

Pre-push polish on the package surface. Three things you notice
immediately when opening the repo, all fixed:

- **Top-level API is now usable.** ``import lhfm`` exposes the 15
  functions people actually call from notebooks: ``generate_synthetic_cohort``,
  ``load_cohort``, ``build_full_feature_table``, ``build_windows``,
  ``train_val_test_split_by_participant``, ``binarize_targets``,
  ``validate_synthetic_dataframe``, the adapter registry, and
  ``load_downstream_checkpoint``. Submodules are still reachable by
  fully-qualified import for everything else. ``import lhfm`` stays
  torch-free; ``load_downstream_checkpoint`` lazy-imports torch only
  when called.
- **Single ``lhfm`` CLI command.** After ``pip install -e .`` the
  ``lhfm`` command is on PATH. ``lhfm pipeline``, ``lhfm train``,
  ``lhfm fairness-audit``, ``lhfm climate-holdout`` etc. dispatch to
  the existing scripts/ entries. Each script still works as
  ``python scripts/<name>.py`` for the people who prefer that.
- **Single checkpoint loader.** ``lhfm.checkpoints.load_downstream``
  consolidates the ~70 lines of "load .pt + meta sidecar + reconstruct
  encoder + reconstruct downstream + set eval mode" that used to be
  inlined separately in the API, the dashboard, and the eval scripts.
  Three drifted copies → one source of truth.

### Audit-5 fixes

- **Version drift across three files.** ``pyproject.toml`` said 0.1.0,
  ``CITATION.cff`` said 0.2.0, ``lhfm/__init__.py`` said 0.1.0. Now
  ``pyproject.toml`` reads dynamically from ``lhfm.__version__`` and
  ``CITATION.cff`` matches.
- **``LHFM Contributors`` placeholder** in ``pyproject.toml`` ``authors``
  field replaced with the real author.
- **Unused ``import json``** in ``api/main.py`` removed after the
  checkpoint-loading consolidation.

### Audit-5 tests

- ``tests/test_public_api.py``: 8 tests covering the public surface,
  the torch-free import guarantee, version-string format, the CLI
  multiplexer (subprocess-based so argv isolation is preserved), and
  ``load_cohort`` shortcut.

### Audit-4 (real-data adapters + pre-push cleanup)

Real-data infrastructure: a uniform adapter layer so plugging in LifeSnaps
or GLOBEM is a one-flag change, plus an Open-Meteo weather enricher with
on-disk caching. Also a humanizing pass over the recently-added modules
(less throat-clearing in docstrings) and a round of bug fixes caught while
re-reading everything before pushing to GitHub.

- **Adapter abstraction** (``src/lhfm/data/adapters/``): base class +
  registry, plus three adapters out of the box.
  - ``synthetic`` wraps the existing generator for CLI parity.
  - ``lifesnaps`` handles the Kaggle/Zenodo layout: column rename,
    sleep-minutes-to-hours, efficiency-%-to-decimal, EMA-1-5-to-1-7
    rescale, timezone-to-coordinates table for weather lookup.
  - ``globem`` handles the PhysioNet multi-institute-year layout:
    feature-file outer-join on (pid, date), screen-time-seconds-to-minutes,
    weekly EMA forward-filled into daily rows, BDI/PHQ outcome attachment
    as ``target_depressed``, institute coordinates for weather.
- **Weather enrichment** (``src/lhfm/data/weather.py``): Open-Meteo
  historical archive (ERA5 reanalysis, 1940-present) plus the separate
  air-quality endpoint with PM2.5 → US-EPA AQI conversion. JSON disk
  cache keyed by (lat, lon) so widening a cohort's date range only fetches
  the new tail, and an ``_AQ_RELIABLE_DAYS=90`` constant documents the
  air-quality archive's ~3-month reliable window (real-cohort AQI for
  older data will mostly be NaN — handled gracefully downstream).
- **Preflight command**: ``python scripts/run_pipeline.py --adapter X
  --preflight`` runs the adapter and reports participant count after the
  min-length filter, per-modality missingness, label balance, demographics.
  Catches "this won't train" in 30 seconds instead of 30 minutes.
- **Adapter tests** (``tests/test_adapters.py``): 17 tests covering the
  registry, schema enforcement, preflight, LifeSnaps and GLOBEM adapters
  on synthetic-shaped CSVs, and PM2.5 → AQI breakpoints.
- **Onboarding docs**: ``docs/adapters/lifesnaps.md`` and
  ``docs/adapters/globem.md`` with the CITI course link, PhysioNet
  credentialing steps, expected preflight values, and known issues.

#### Bugs fixed before push

- **``max_participants`` cap used alphabetic sort.** ``sorted(['P1','P10','P2'])[:2]``
  returned ``['P1','P10']`` not ``['P1','P2']``. Now uses a natural-sort
  key so small smoke runs pick the right participants.
- **Weather cache key conflated site and date range.** Re-running with a
  wider cohort window re-fetched the entire range from scratch. New cache
  is keyed by (rounded lat, rounded lon) only, with per-date union-merge,
  so widening a window fetches only the new days.
- **Weather cache path declared as parquet, written as JSON.** Confusing.
  Now consistently ``.json``.
- **AQI silently failed for old cohorts.** Open-Meteo's air-quality
  archive only reliably serves ~90 days back; for LifeSnaps (2021-2022)
  and GLOBEM (2018-2021) we'd otherwise spend a request per site to get
  empty data. Now skips with an informative log and leaves AQI NaN.
- **Adapter base raised ``IndexError`` on max_participants when ids
  weren't unique enough**: defensive ``.tolist()`` on ``unique()`` plus
  natural-sort.
- **GLOBEM and LifeSnaps adapters crashed on partial files.** Real
  GLOBEM ships some institute-years without ``ema_weekly.csv`` or
  ``dep_endterm.csv`` — the adapter now degrades gracefully (sets flag=1
  and continues) instead of crashing the whole load.
- **``cycle_phase`` was emitted before column-ordering**: the v0.2
  synthetic generator now stable-orders required columns first, extras
  alphabetically after — matches the adapter contract.

#### Cleanup pass

- README and ``CITATION.cff`` updated with real owner (``ceyhunolcan``)
  and removed placeholder note.
- ``.gitignore`` hardened: explicitly excludes ``data/cache``,
  ``*.safetensors``, ``results/tables/*.json`` (per-row arrays in metrics
  JSON are a re-identification risk on real data), ``.kaggle/``,
  ``.physionet/``, editor swap files, OS noise.
- Module docstrings in ``utils/fairness.py``, ``utils/climate_regimes.py``,
  ``interpretability.py``, both real-data adapters, and weather module
  rewritten in a more human voice: dropped the rhetorical "Why bother on
  synthetic data?" / "The intuition reviewers will press on" framings,
  cut em-dashes and bold-italic for emphasis, replaced bullet-heavy API
  preambles with plain prose, tightened the longest comments.
- ``CITATION.cff`` bumped to 0.2.0 with audit-3 + audit-4 abstract.

### Audit-3 (synthetic realism + fairness + climate generalization)

This pass closes everything from "what still needs improvement" *except*
real-data validation and DOI publication. It hardens the prototype enough
that the only thing standing between LHFM and an external benchmark is a
real cohort.

#### Synthetic generator (v0.2) — realism overhaul

The v0.1 generator was rich enough for the engineering tests but had
known limitations (within-person mood-sleep correlation ~0.05, no
medication / comorbidity / cycle / device-noise / cold-weather effects,
no subgroup metadata). v0.2 fixes those.

- **Medications**: SSRIs (~9% prevalence in young adults, ×5 with
  depression), beta-blockers (~5%, ×6 with cardio condition), sleep
  aids (~7%, ×3.5 with sleep disorder). Effect sizes match the
  digital-health meta-analytic ranges (β-blockers raise HRV ~12ms and
  lower RHR ~10bpm; SSRIs lower HRV ~4ms and lift mood floor; sleep aids
  reduce sleep-efficiency penalty).
- **Comorbidities**: anxiety (18%), depression (10%), sleep disorder
  (8%), cardiovascular condition (6%, age-scaled). Each pushes mood,
  stress, HRV, RHR, and missingness propensity in the appropriate
  direction.
- **Hormonal cycles**: 70% of eligible female participants (16-51)
  carry a synthetic 28-day cycle. Late-luteal HRV drop (-4.5ms),
  ovulation peak (+1.5ms), mild luteal mood dip (-0.25). Cycle phase
  emitted as ``cycle_phase`` column.
- **Device generation**: 30% of participants get an "older" device with
  6ms HRV measurement noise floor (vs 1.5ms for new). This is the
  canonical bottleneck in real wearable-validation studies.
- **Cold-weather effects**: T < 5°C now raises RHR via vasoconstriction,
  degrades sleep efficiency, and suppresses steps. New ``cold_snap_days``
  config knob and ``cold_snap`` climate track.
- **Wildfire smoke episodes**: trapezoidal AQI spikes 220-380 lasting
  2-5 days (2 episodes by default). New ``wildfire_smoke_days`` config.
- **Subgroup metadata**: ``race_ethnicity`` (5-way), ``ses_proxy``
  (low/middle/high), ``region`` (temperate/hot/cold). Independent draws
  — **no disparities baked in** by design. The fairness module needs
  axes to stratify on.
- **Stronger mood-sleep coupling**: sleep coefficient bumped 0.35→0.40
  + added 3-day rolling sleep_debt term (coef 0.18). Within-person
  mood-sleep correlation lands at **0.33** (target 0.25-0.35, was 0.05
  in v1). Verified empirically on a 50p×60d cohort.

Schema impact: 27 original columns preserved; 12 new metadata columns
appended (``race_ethnicity, ses_proxy, region, device_gen, has_anxiety,
has_depression, has_sleep_disorder, has_cardio_condition, on_ssri,
on_beta_blocker, on_sleep_aid, cycle_phase``). The
``validate_synthetic_dataframe`` required-columns set is unchanged.

#### Subgroup-stratified fairness audit

- New ``lhfm.utils.fairness`` module: ``run_fairness_audit``,
  ``fairness_report_to_csv``, ``check_fairness_thresholds``.
- Slices the held-out test set by sex, race/ethnicity, SES, region,
  device generation, depression status, anxiety status, age band.
  Computes per-subgroup AUROC with cluster-bootstrap CIs, AUPRC, ECE,
  Brier, FPR, FNR. Equalised-odds violation per axis = max-FPR-gap +
  max-FNR-gap.
- New ``scripts/run_fairness_audit.py`` CLI with ``--fail-on-violation``
  for CI gating. Loads the saved checkpoint via the meta sidecar, joins
  participant metadata back to windows, runs the audit per task, writes
  ``fairness_<task>.csv`` + ``fairness_<task>_equalised_odds.csv`` +
  ``fairness_summary.json``.
- Tests verify the audit catches a deliberately biased predictor and
  passes on balanced data.

#### Climate-regime generalization

- New ``lhfm.utils.climate_regimes`` module: ``define_climate_regime``
  (heat_wave / cold_snap / smoke_episode / normal),
  ``regime_summary``, ``split_train_eval_by_regime``.
- New ``scripts/run_climate_holdout.py`` CLI. Default mode: redacts
  held-out regime labels from train+val, fine-tunes the warm-started
  encoder, evaluates on held-out regime test windows. ``--no-retrain``
  diagnostic mode: per-regime metrics on the existing checkpoint.
- Writes ``results/tables/climate_regime_eval.csv`` with per-task,
  per-regime AUROC + bootstrap CI.

#### Tests added

- ``tests/test_fairness_and_regimes.py``: 11 tests across subgroup
  detection, AUROC-spread thresholds, equalised-odds violations, CSV
  export, regime mutual exclusivity, and NaN handling.
- The existing ``tests/test_training.py`` (torch-dependent) was already
  comprehensive after audit-2; it covers ``_compute_pos_weights``
  (caps + degenerate handling), ``train_downstream`` (state_dict
  shape, no-pickle saves), ``pretrain_ssl`` (decreasing loss,
  deterministic eval), ``evaluate_downstream`` (CI shape, cluster
  vs row-level bootstrap mode).

#### Documentation

- ``paper/data_card.md`` rewritten to v0.2 spec: medications,
  comorbidities, hormonal cycles, device noise, cold weather,
  wildfire smoke, subgroup metadata, tighter mood-sleep coupling.
- ``paper/methods.md`` §7 expanded with the fairness-audit + climate-
  holdout protocols.
- ``CHANGELOG.md`` audit-3 section (this).

### Audit-2 (senior-PI methodological pass)

- **Package rename ``src.*`` → ``lhfm.*``.** Standard src-layout.
- **Faithful interpretability.** ``lhfm.interpretability`` implements
  integrated gradients (Sundararajan, Taly & Yan 2017) over the
  multimodal encoder. The ``/predict`` endpoint returns IG-based
  explanations whenever a trained model is loaded; the rule-based panel
  remains as the no-model fallback.
- **Pretraining-scale ablation infrastructure.**
  ``scripts/run_scale_ablation.py`` sweeps cohort size (default
  20/50/100/200/400 × 2 seeds), trains SSL + downstream at each size,
  evaluates with participant-clustered bootstrap CIs, emits a tidy CSV
  and the canonical AUROC-vs-N figure.
- **Release bundling.** ``scripts/export_release.py`` packages a trained
  checkpoint into a self-contained ``releases/<run_tag>/`` with state-dict,
  meta sidecar, age-reference sidecar, feature-schema sidecar, metrics
  CSV, model-card addendum, integrity manifest (SHA256SUMS).
- **Encoder unseen-participant graceful fallback.** Strict at training
  (catches index bugs) but ``allow_unknown_participants()`` routes
  out-of-range indices to the mean of the learned embedding table. API
  flips into permissive mode at load.
- **``ACCEPTABLE_USE.md``** with forbidden uses (clinical decision-making,
  surveillance, employee monitoring, insurance underwriting,
  applications to minors, intimate-partner monitoring, re-identification,
  generative misuse).
- **Training-pipeline test coverage.** ``tests/test_training.py``
  exercises ``pretrain_ssl``, ``train_downstream``,
  ``_compute_pos_weights``, ``evaluate_downstream`` end-to-end on toy
  data. Regression test pins the deterministic-SSL-eval fix.

### Fixed
- Removed eager imports from ``src/lhfm/training/__init__.py`` so
  ``import lhfm.training`` works without torch (notebooks, lint jobs).

### Changed
- ``paper/methods.md`` gained §6 (interpretability), §7 (generalization
  studies).
- ``Makefile`` exposes ``make scale-ablation``, ``make climate-holdout``,
  ``make fairness-audit``, ``make release-bundle``.
- Dockerfile uses ``lhfm.api.main:app``.

### Methodological fixes (senior-PI audit pass)

These are the deeper issues caught on a second read by a senior reviewer.
None of them would have failed CI before, but each is the kind of thing a
biomedical-AI PI looks for first.

- **Temporal leakage in features.** Every within-person z-score in the
  wearable, smartphone, and climate modules now uses *expanding* (past-only)
  means and standard deviations. The prior version standardized against the
  full participant timeline so a day-`t` z-score depended on days `t+1..T` —
  fine for the synthetic data but invalid for real-time inference and a
  silent leak at training time.
- **Age standardization no longer leaks the population prior.** The
  hardcoded `AGE_REF_MEAN=34.0`, `AGE_REF_STD=11.0` constants happened to
  exactly match the synthetic generator. The training script now computes
  reference stats on the *training* split (one row per participant) and
  persists them to `downstream.meta.json`. The API reads them back at load
  time. The original constants remain as documented fallbacks.
- **Bootstrap CIs now use participant-level cluster resampling.** The
  prior implementation resampled windows uniformly, which on longitudinal
  data with multiple windows per person produces CIs that are roughly an
  order of magnitude too narrow. `bootstrap_ci` takes a `groups=` argument;
  `evaluate_downstream` passes `participant_idx` by default. Results CSV
  has a new `bootstrap_unit` column. The window-level mode remains for
  reproducing earlier numbers.
- **Target-leakage caveat documented + EMA-blind mode added.** Three of
  the four downstream tasks are thresholds on EMA fields that also appear
  as features. A model that simply autoregresses on `survey_mood` is the
  trivial baseline. `scripts/train_model.py --exclude-ema-features` drops
  the `survey_*` columns so the task becomes "predict tomorrow's mood from
  passive sensing alone". This is now documented as the primary protocol
  in the README and methods.

### Engineering improvements

- **Pad mask plumbed end to end.** `DownstreamRiskModel.forward` now
  accepts `pad_mask`; the API path passes one through whenever the request
  window is shorter than the model's training window. Zero-padded warmup
  days no longer contribute to attention pooling.
- **Encoder safety checks.** `MultimodalLongitudinalEncoder` now raises a
  clear `IndexError` when `participant_idx` exceeds the embedding table
  size, and `SinusoidalPositionalEncoding.forward` raises a clear
  `ValueError` when the sequence is longer than `max_seq_len`. The prior
  versions failed deep inside the embedding op with opaque shape errors.
- **Encoder parameter accounting.** `encoder.count_parameters()` returns a
  named breakdown (transformer, projectors, mask embeddings, pool,
  participant embedding). `train_model.py` logs the total at startup and
  persists it to the meta sidecar.
- **Provenance tagging.** Every training run logs and persists a
  `git_sha`, a `config_hash`, and an optional `run_tag`. The final
  checkpoint is also written as `downstream-<run_tag>.pt` so parallel
  sweeps don't overwrite each other.
- **Standalone evaluation CLI.** `scripts/evaluate_model.py` re-evaluates
  a saved checkpoint without retraining — useful for re-doing bootstrap
  with more resamples, switching the split, or pointing at a different
  feature CSV (the real-data adapter path).
- **K-way windowing collapsed to one pass.** `train_model.py` previously
  called `build_windows` once per task to assemble the multi-task label
  matrix; it now does one pass and joins labels back via a
  `(participant_id, target_date)` lookup.
- **Lazy training subpackage import.** `src.training.__init__` no longer
  eagerly pulls torch. `import src.training` works in torch-less envs
  (lint, docs, `--help` introspection); torch is imported when training
  code is actually called.
- **Makefile**, `.pre-commit-config.yaml`, and a full ruff + mypy +
  coverage configuration in `pyproject.toml`.

### Documentation

- `paper/methods.md` gained sections 5.1 (target leakage + EMA-blind),
  5.2 (clustered bootstrap as the unit of uncertainty), and 5.3 (train-
  data-only age standardization). The feature-engineering section
  explicitly calls out causal statistics.
- README has an explicit "target-leakage caveat" subsection telling
  readers to prefer the EMA-blind protocol.

## Audit-1 fixes (previous pass)

### Fixed
- `compute_baseline_features` now uses fixed population-reference statistics
  for `age_z` (fallback mean=34, std=11). The previous code computed
  these from the input dataframe, which broke single-participant inference
  in the API: a one-row input has zero variance and `age_z` came back as
  `NaN`. (Now further fixed in audit-2 to use train-data stats; see above.)
- `binarize_targets` no longer silently labels `target_climate_vulnerable`
  as positive when HRV is missing. NaN HRV (or NaN heat_index) now yields
  NaN target, which the downstream training loop correctly masks out.
- `_evaluate_ssl` now uses a deterministic mask generator, so SSL
  validation loss is comparable across epochs and early stopping is no
  longer noisy.
- API `_fallback_predict` and `_build_explanation` no longer replace
  legitimate zero values with their default fallbacks. The previous
  `value or default` short-circuit broke for things like `AQI = 0`.
- `build_windows` now skips windows that span date gaps by default
  (`require_consecutive_dates=True`), preventing silently-broken windows
  when a participant has a missing calendar day.
- Removed dead variables in the synthetic generator (`prev_sleep_dev`,
  `prev_hrv_dev`).
- Heat-index fallback now requires both `T >= 26.5°C` and `humidity >= 40%`
  to use the NWS Rothfusz regression. Outside that envelope we return
  ambient temperature, matching standard practice.

### Added
- Per-task `pos_weight` in BCE loss, computed once from the training labels
  and capped at 20.
- Bootstrap 95% confidence intervals for AUROC and AUPRC in
  `evaluate_downstream` (now upgraded to clustered bootstrap in audit-2).
- Brier score added to `binary_classification_report`.
- `bootstrap_ci` helper in `src/utils/metrics.py`.
- Participant-aware InfoNCE: same-participant non-paired windows are now
  masked out of the contrastive negative pool when participant indices are
  available.
- GitHub Actions CI: matrix-tested on Python 3.11 and 3.12, plus a tiny
  end-to-end smoke train and `ruff` lint.
- Calibration and confusion-matrix PNG figures saved to `results/figures/`
  after training.
- `CONTRIBUTING.md`, `CHANGELOG.md`, `CITATION.cff`.

### Changed
- API now uses the modern FastAPI `lifespan` context manager.
- Checkpoints are saved as a state-dict `.pt` plus a sibling
  `.meta.json` file; API loads with `torch.load(weights_only=True)`.
- API `_predict_with_model` reads `window_days` from the checkpoint meta
  instead of hard-coding 14.
- `ParticipantProfile.sex` accepts `F`, `M`, or `X`.
- Streamlit dashboard now adds the project root to `sys.path`.

## [0.1.0] - 2025

Initial release: synthetic cohort, feature engineering, multimodal
transformer encoder with three SSL objectives, four downstream risk heads,
FastAPI service, Streamlit dashboard, classical baselines, paper docs.
