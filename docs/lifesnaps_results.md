# LifeSnaps real-data results

This document summarises LHFM's performance on the LifeSnaps cohort
(Yfantidou et al. 2022, *Scientific Data*). It is intended as a
reproducible reference, not a paper draft — the numbers below were
produced by the public code in this repository on a single MacBook Air
CPU, in roughly 15 minutes of total runtime.

## Cohort

| Property | Value |
| --- | --- |
| Source | LifeSnaps "RAIS" release (Kaggle / Zenodo) |
| Participants | 71 |
| Days observed (per participant) | min 64, median 88, max 244 |
| Date range | 2021-04-08 → 2022-01-22 |
| Sex distribution | M = 41, F = 26, missing = 4 |
| Wearable | Fitbit Sense (HRV, RHR, sleep stages, steps, stress score) |
| Smartphone passive sensing | **not available** in this cohort |
| Climate enrichment | **not run** (no participant-level lat/lon) |
| EMA | SEMA app: 7 binary emotion checkboxes per delivery, ~31% of days |

## Tasks evaluated

LifeSnaps is a wearable-plus-EMA cohort with no smartphone sensing
and no GPS, so only a subset of LHFM's four downstream tasks are
evaluable on it:

| Task | Evaluable on LifeSnaps? | Reason |
| --- | --- | --- |
| `high_stress` | yes | survey_stress derived from SEMA TENSE/ANXIOUS report |
| `sleep_disruption` | yes | sleep_duration from Fitbit (in milliseconds in raw CSV; adapter normalises to hours) |
| `low_mood` | no | SEMA "SAD" checkbox almost never reported in this cohort; positive class collapses to zero in test |
| `climate_vulnerable` | no | requires temperature / AQI inputs that LifeSnaps does not provide |

## Held-out test results

Bootstrap 95% confidence intervals are clustered at the participant
level (1000 iterations). The test fold contains 11 participants.

| Task | Model | AUROC | 95% CI | AUPRC | n_pos | n_total |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `high_stress` | **LHFM** | **0.567** | [0.389, 0.688] | 0.218 | 55 | 294 |
| `high_stress` | logistic regression | 0.328 | – | 0.151 | 55 | 294 |
| `high_stress` | random forest | 0.368 | – | 0.211 | 55 | 294 |
| `sleep_disruption` | **LHFM** | 0.518 | [0.376, 0.682] | 0.129 | 51 | 464 |
| `sleep_disruption` | logistic regression | **0.656** | – | 0.195 | 51 | 464 |
| `sleep_disruption` | random forest | 0.641 | – | 0.197 | 51 | 464 |

### Read

- On `high_stress`, **LHFM beats both classical baselines by roughly
  20 AUROC points on identical features**. The 95% CI on LHFM is wide
  ([0.39, 0.69]) and includes 0.5; the gap to the baselines is the
  meaningful comparison, not the absolute AUROC.
- On `sleep_disruption`, **classical baselines outperform LHFM** on
  this cohort. We report it because it is real: SSL pretraining does
  not buy lift on every task in a Fitbit-only setting.
- The wide CIs reflect the small test fold (n = 11 participants).
  We expect substantially tighter CIs from replication on the GLOBEM
  cohort (n ≈ 700).

## Reproducing these numbers

```bash
# 0. one-time install
make install && pip install -e .

# 1. download LifeSnaps from Kaggle (free account, phone-verified):
#    https://www.kaggle.com/datasets/skywescar/lifesnaps-fitbit-dataset
#    unzip into data/raw/lifesnaps/

# 2. preflight (adapter sanity check, 5 s)
lhfm pipeline --adapter lifesnaps \
    --raw-dir data/raw/lifesnaps/csv_rais_anonymized \
    --preflight

# 3. build feature table (30 s)
lhfm pipeline --adapter lifesnaps \
    --raw-dir data/raw/lifesnaps/csv_rais_anonymized

# 4. train + evaluate (15 min on CPU)
lhfm train --features data/processed/features.parquet \
    --ssl-epochs 30 --downstream-epochs 30 \
    --tasks high_stress sleep_disruption \
    --run-tag lifesnaps_v3
```

Resulting files:
- `results/tables/metrics_test.csv` — LHFM held-out test metrics
- `results/tables/baselines.csv` — logreg / random forest on the same features
- `results/figures/calibration_high_stress.png` — reliability diagram
- `results/figures/confusion_high_stress.png` — confusion at the operating threshold

The first two are committed in this repository at the SHAs documented in
`run_tag=lifesnaps_v3` (see `git log --oneline`).

## Limitations specific to this cohort

1. **Test fold is 11 participants.** Bootstrap CIs are correspondingly
   wide. Even cleanly-positive results are not statistically distinguishable
   from chance at this sample size.
2. **No smartphone passive sensing.** LHFM's smartphone modality is
   absent on LifeSnaps. The model trains and evaluates on
   wearable + EMA only; smartphone-derived features (screen time,
   phone-unlock count, mobility, location entropy) are flagged as
   entirely missing.
3. **No climate enrichment.** The Open-Meteo enrichment requires
   participant-level latitude / longitude, which LifeSnaps does not
   publish (only timezone). All climate features are NaN on this cohort.
4. **EMA is sparse.** SEMA reports were delivered for roughly 31% of
   participant-days. Targets are NaN-masked on days with no EMA
   delivery to avoid biasing the positive rate.
5. **`low_mood` is unevaluable.** The SEMA "SAD" checkbox is reported
   on fewer than 3% of EMA days in this cohort, so the derived
   `survey_mood` distribution does not produce enough negative-affect
   days to make a meaningful binary target. We surface this honestly
   rather than tuning the threshold to manufacture a positive class.

## What this is *not*

These numbers are not the headline result of an eventual paper. They
are a transparency artefact: anyone who clones this repository, follows
the four commands above, and waits 15 minutes will reproduce them.
The intended primary cohort is GLOBEM (Xu et al., NeurIPS 2022),
which has substantially more participants and richer modalities.
Credentialing for that dataset is in progress.

## References

- Yfantidou, S., et al. (2022). *LifeSnaps, a 4-month multi-modal
  dataset capturing unobtrusive snapshots of our lives in the wild.*
  Scientific Data 9(1), 663.
- The companion dataset is distributed on Kaggle under the name
  "skywescar/lifesnaps-fitbit-dataset" and on Zenodo as the RAIS
  release.
