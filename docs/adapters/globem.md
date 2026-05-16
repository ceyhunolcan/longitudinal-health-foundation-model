# GLOBEM adapter — onboarding

GLOBEM (Generalization of LOngitudinal BEhavior Modeling) is the most
serious public benchmark for longitudinal passive sensing: n=497 across
4 institute-year cohorts, with published baselines for 18 algorithms on
depression detection.

Paper: Xu et al., *NeurIPS 2022 D&B Track*, <https://arxiv.org/abs/2211.02733>
Codebase: <https://github.com/UW-EXP/GLOBEM>
Dataset: <https://physionet.org/content/globem/1.1/>

## Why GLOBEM should be your primary dataset

- **Published benchmark.** 18 algorithms with reported numbers. Your
  paper has its baselines ready-made.
- **Cross-site generalisation built in.** Four institute-years (UW × 2
  years, CMU × 2 years) let you train on three and test on the fourth.
  That's the published canonical evaluation; LHFM's
  `--training-holdout institute_year` flag wires directly into it.
- **Reviewers recognise it.** Top-tier IMWUT, JMIR, npj Digital
  Medicine reviewers in this space know GLOBEM. Less explaining required.

What GLOBEM does NOT give you:

- **No RMSSD / HRV.** The cohort used Fitbit Charge series which don't
  expose RMSSD. `hrv_rmssd` and `stress_score` stay NaN.
- **No daily EMA.** GLOBEM uses weekly PHQ-4 + custom items. The
  adapter forward-fills the weekly value into the 7 day-rows downstream.
  Document this as a methodology decision; some reviewers will push back.
- **No race/ethnicity in the public release** (university IRB redacted
  it). The fairness audit can still slice on sex, age, depression
  status, institute-year, and device.

## Step-by-step

### 1. PhysioNet credentialing

This is the slow part. Start *today*.

1. **Create a PhysioNet account** at <https://physionet.org/register/>.
2. **Take the CITI Data or Specimens Only Research course.** It's free,
   online, takes ~2 hours, and is the credential PhysioNet requires for
   restricted datasets. Sign up at <https://about.citiprogram.org/>,
   pick your "Organizational Affiliation" (your university if you have
   one; otherwise "Independent Learner"), select the *Data or Specimens
   Only Research* course.
3. **Upload your CITI completion certificate** to your PhysioNet profile
   under "Credentialing".
4. **Wait for approval.** Usually 7-14 days. They check that you've
   listed an institutional affiliation and that the CITI certificate is
   valid. The PhysioNet team is friendly; if it takes longer than 2
   weeks, email them.
5. **Sign the GLOBEM DUA** at <https://physionet.org/content/globem/1.1/>
   once credentialed. This is a click-through, not an institutional
   contract — you can do it as an individual researcher.

### 2. Download

Once you've signed the DUA, PhysioNet gives you a URL list and a
wget-able file tree:

```bash
mkdir -p data/raw/globem
cd data/raw/globem
wget -r -N -c -np --user=<YOUR_PHYSIONET_USER> --ask-password \
    https://physionet.org/files/globem/1.1/
```

You should end up with:

```
data/raw/globem/
  INS-W_1/
    FeatureData/
      location.csv
      screen.csv
      steps.csv
      sleep.csv
      ...
    SurveyData/
      pre_survey.csv
      dep_endterm.csv
      ...
  INS-W_2/
  INS-W_3/
  INS-W_4/
```

(The actual structure may have a `physionet.org/files/globem/1.1/`
prefix; move the contents up if so.)

### 3. Preflight

```bash
python scripts/run_pipeline.py \
    --adapter globem \
    --raw-dir data/raw/globem \
    --preflight
```

You should see something like:

```
n_participants: ~497
n_participants_after_min_days_filter: ~480
n_days_median: ~70
sex_distribution: {'F': ~280, 'M': ~210, ...}
frac_missing_wearable_flag: ~0.30   # Fitbit Charge has wider gaps
frac_missing_phone_flag: ~0.15
frac_missing_survey_flag: ~0.85     # weekly EMA → most days have no fresh value
```

If `frac_missing_survey_flag` is below 0.50, the weekly forward-fill
likely went too far — check that your `--max-fill-gap` config is set to
7 (a week) and not larger.

### 4. Pipeline + train

```bash
python scripts/run_pipeline.py \
    --adapter globem \
    --raw-dir data/raw/globem

python scripts/train_model.py \
    --features data/processed/features.csv \
    --exclude-ema-features \
    --run-tag globem-emablind
```

### 5. Cross-site holdout (THE money figure)

The published GLOBEM benchmark evaluates by holding out one
(institute, year) pair at a time. Reproduce that as:

```bash
# Train on INS-W_1, INS-W_2, INS-W_3; test on INS-W_4
python scripts/run_climate_holdout.py \
    --features data/processed/features.csv \
    --holdout-column institute_year \
    --holdout-value INS-W_4 \
    --run-tag globem-cross-year
```

(Requires the `--holdout-column` flag I'll add to the climate-holdout
script once we have real GLOBEM data — it's a 10-line extension of the
existing regime-holdout logic.)

### 6. Fairness audit

```bash
python scripts/run_fairness_audit.py \
    --checkpoint checkpoints/downstream-globem-emablind.pt \
    --fail-on-violation \
    --max-auroc-gap 0.08
```

GLOBEM has sex, age, and institute-year as fairness axes. Real cohorts
typically show 5-15 percentage points of AUROC spread by sex on
depression-detection tasks — this is where the audit becomes *interesting*,
not just a pipeline check.

## Known issues

- **No HRV.** This means the `hrv_dev_from_baseline`, `recovery_score`,
  and `stress_burden_7d` features all degrade or zero out. The model
  will work, but two of LHFM's six wearable features become dead. The
  paper's "wearable physiology" framing needs to lean on sleep and RHR
  alone for GLOBEM.

- **Weekly EMA forward-filled to daily.** This violates the spirit of
  per-day prediction. Reviewers may push back. The defensible response:
  "we used the most recent weekly value as the *target* on each day
  within that week, mirroring the published GLOBEM evaluation protocol."
  Cite Xu et al. 2022 §4.2.

- **Column names may shift between PhysioNet versions.** GLOBEM v1.0 →
  v1.1 renamed several feature files. If the adapter raises
  `no matching cols`, run `head -1 <file>.csv` and update the
  `cols_map` dictionaries in `globem.py`.

- **Race/ethnicity stripped from public release.** The fairness audit
  can still slice on sex / age / device / institute-year / has_depression,
  but you cannot speak to race-based disparities on GLOBEM alone. Pair
  with LifeSnaps or a cohort that retains those columns.

## What goes in the paper

GLOBEM evaluations make three loadbearing arguments:

1. **Cross-site generalisation.** Train on 3 institute-years, test on the
   4th. Report AUROC + participant-clustered bootstrap CI per held-out
   year. Compare to the GLOBEM benchmark's 18 baselines.
2. **EMA-blind passive sensing.** Show that the foundation-model
   representation predicts weekly PHQ/BDI from passive sensing alone, at
   AUROC > 0.65 (the published GLOBEM ceiling for non-domain-adaptive
   methods).
3. **Fairness deltas.** Show that LHFM's per-subgroup AUROC spread on
   sex and institute-year is within the threshold you declare upfront.
