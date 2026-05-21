# Methods

## 1. Data

### 1.1 Synthetic cohort

To support fully reproducible methodological development without exposing
any real-patient data, we ship a synthetic longitudinal cohort generator
(`src/data/synthetic_generator.py`). The generator simulates 250 virtual
participants observed daily over a 90-day window. Each participant is
endowed with latent priors sampled from population-style distributions:
demographics (age, sex), chronotype, baseline sleep need, baseline HRV and
resting heart rate, mood and stress setpoints, and three sensitivity
parameters governing how heat exposure, air-pollution exposure, and overall
study-engagement modulate their daily signals.

Per-day signals fall into four blocks:

- **Wearable.** Daily steps, sleep duration, sleep efficiency, resting
  heart rate, HRV (RMSSD-style), and a manufacturer-style stress score.
- **Smartphone.** Phone unlocks, screen-time minutes, mobility radius, and
  location entropy (a proxy for behavioural variety).
- **Self-report (EMA).** Daily mood, energy, and stress ratings on a 1-7
  scale, modelled after the Photographic Affect Meter and similar
  short-form ecological-momentary-assessment instruments.
- **Environmental.** Ambient temperature, humidity, AQI, and a derived
  heat index. The temperature track includes seasonal trend plus two
  explicit heat-wave events; AQI includes intermittent pollution episodes.

Causal relationships are seeded so the dataset is non-trivially learnable:
sleep debt and elevated heat index degrade HRV and elevate resting HR
within-person; mood declines with high stress, poor sleep, poor AQI, and
extreme heat; mobility shrinks under heat stress and low mood.
**Missingness is informative**: low-mood days are flagged as missing for
the wearable and self-report modalities at elevated rates, mirroring the
well-documented "ghosting" effect in real passive-sensing studies.

A `validate_synthetic_dataframe` step enforces structural and plausibility
checks (column presence, value ranges, per-participant timeline length)
before downstream code consumes the data.

### 1.2 Preprocessing

Long stretches of missing values are forward-filled within participant for
up to three consecutive days. Remaining gaps are imputed with per-participant
medians, falling back to cohort medians. Splits are constructed at the
**participant** level (default 70/15/15) to prevent within-person leakage
across train, validation, and test sets.

### 1.3 Real-data cohorts and the adapter abstraction

The same model and training code are used unmodified on real data. The
key affordance is a thin **adapter abstraction**
(`src/lhfm/data/adapters/base.py`) that ingests a raw cohort directory
and emits a per-day dataframe in the schema the rest of the pipeline
expects: one row per `(participant_id, date)`, the union of expected
columns across wearable / smartphone / EMA / climate, and standard
sentinel handling for missing modalities. Three adapters are provided:

- `SyntheticAdapter`, a thin wrapper over the in-memory generator.
- `LifeSnapsAdapter`, which reads the public LifeSnaps "RAIS" release
  (Yfantidou et al. 2022) from Kaggle / Zenodo and emits the same
  per-day schema. The adapter handles real-world quirks of this
  release: age stored as range strings (`"<30"`, `"30-40"`), sex
  stored as full words (`"MALE"`, `"FEMALE"`, `"NB"`), sleep duration
  stored in milliseconds (the adapter detects the unit by inspecting
  the median and converts to hours), and SEMA emotion reports encoded
  as per-emotion one-hots (HAPPY, SAD, TIRED, ALERT, TENSE/ANXIOUS,
  RESTED/RELAXED, NEUTRAL), which the adapter collapses into the
  three-axis 1-7 valence / energy / stress scores used downstream.
- `GlobemAdapter`, which targets the GLOBEM longitudinal cohort
  (Xu et al. 2022, NeurIPS Datasets track). Implementation is staged;
  results pending credentialed access via PhysioNet.

When a cohort is missing entire modalities (LifeSnaps has no smartphone
sensing and no climate enrichment), the adapter sets the corresponding
columns to `NaN` and surfaces a *modality-missing flag*. The feature
engineering and model code treat these flags as first-class input,
which means a cohort can be evaluated even when it covers only a subset
of the modalities the model was pretrained on; the downstream heads
simply receive a zero-padded representation for the missing modality.

No participant-level data from real cohorts is redistributed in this
repository. Only aggregate metrics (per-task AUROC, AUPRC, ECE, Brier
score, baseline comparisons) and a small number of headline figures
(calibration plot, confusion matrix) are committed; row-level
predictions, raw cohort files, and per-participant attribution traces
are explicitly gitignored. This is enforced by `.gitignore` rules and
documented in the data-use compliance note of each adapter.

## 2. Feature engineering

Five feature modules produce a 25-dimensional per-day feature vector:

- **Wearable features** include sleep regularity index, 7-day rolling
  sleep duration, deviations of HRV and resting HR from each participant's
  baseline, a 7-day stress burden, and a composite recovery score.
- **Smartphone features** include within-person z-scores of screen time
  and unlock frequency, mobility radius, location entropy, and a 7-day
  behavioural-regularity index.
- **Climate features** include the raw heat index, a humidity-corrected
  variant, an estimated nighttime heat stress, 3-day heat exposure, and a
  within-person climate-stress composite.
- **Missingness features** explicitly model the *pattern* of dropout:
  consecutive-day streaks, 7-day missingness rate, and per-modality
  dropout entropy.
- **Baseline features** encode slowly varying personal context: age
  z-score, chronotype score, baseline HRV, baseline sleep need.

This explicit decomposition allows ablations by simply dropping a modality
column block in `configs/features.yaml`.

**Causal statistics.** Every within-person standardization uses *expanding*
(past-only) means and standard deviations rather than the participant's
full-timeline mean/std. The naive form leaks the future into past
features — at day 5 of training the z-score's denominator depends on days
6-90 — and gives mismatched distributions at inference time on partial
timelines. The expanding form means a day-`t` z-score uses only days
`1..t`.

## 3. Model architecture

The encoder (`MultimodalLongitudinalEncoder`) consists of:

1. **Modality projectors.** Each of the four modality blocks is projected
   independently through a two-layer MLP plus LayerNorm into a shared
   `d_model = 128` latent space.
2. **Missingness mask embeddings.** A learned vector per modality is added
   in proportion to the per-step imputed-fraction signal, giving the model
   a continuous awareness of how much of each timestep was reconstructed
   rather than measured.
3. **Participant embedding.** An optional embedding table indexed by
   participant id, projected up to `d_model` and added to each timestep,
   provides an explicit personalization axis.
4. **Sinusoidal positional encoding** plus a stack of three transformer
   encoder layers (`n_heads = 4`, `ff_dim = 256`, GELU activations,
   dropout 0.1) operating on the time axis.
5. **Attention-pooling head.** A learned-query multihead attention pool
   reduces the (B, T, d_model) sequence to a single (B, d_model) trajectory
   representation, which is what downstream heads consume.

## 4. Self-supervised pretraining

Pretraining minimises a weighted sum of three losses:

- **Masked feature reconstruction (weight 1.0).** A random 20% of timesteps
  in the wearable block are zeroed out and a per-step decoder is asked to
  recover them. Only masked positions contribute to the loss.
- **Next-day state prediction (weight 0.5).** Given the full window, a
  pooled-representation decoder predicts the final day's wearable feature
  vector. This pushes the encoder to learn forward-looking trajectory
  structure.
- **Participant-trajectory contrastive loss (weight 0.25).** Two
  differently-masked views of the same window are projected through a small
  MLP and compared with InfoNCE at temperature 0.1. The objective brings
  same-participant windows closer than across-participant windows.

We use AdamW (lr 5e-4, weight decay 1e-5), gradient clipping at L2 norm 1.0,
and early stopping on validation loss.

## 5. Downstream evaluation

Four binary downstream tasks are defined directly on the feature table:

- **Next-day low mood** (`survey_mood ≤ 3`)
- **Same-day high stress** (`survey_stress ≥ 5`)
- **Sleep disruption** (`sleep_efficiency < 0.80` or `sleep_duration ≤ 5h`)
- **Climate-vulnerable day** (`heat_index > 32°C` and HRV ≥ 10 ms below
  personal baseline)

We are deliberately explicit that all four are **rule-based weak labels
derived from passive-sensing signals**, not validated clinical instruments
or expert adjudications. They serve as scaffolding for methodological
evaluation; they should not be interpreted as proxies for clinical
diagnosis. The climate-vulnerability target in particular is a researcher-
defined heuristic combining heat exposure with a personalized autonomic
deviation — it is conceptually inspired by, but in no sense equivalent to,
clinical heat-illness criteria.

### 5.1 The target-leakage caveat (and the EMA-blind variant)

Three of the four targets are thresholds on EMA items (`survey_mood`,
`survey_stress`, `sleep_efficiency`) that are themselves present in the
feature table. Even with the *next-day* prediction formulation
(`y_{t+1} = f(x_{1..t})`), the model has direct access to the same scale's
value on every preceding day in the window. A model that simply learns
"tomorrow's mood looks like today's mood" will score well — this is
trivial autoregression, not a foundation-model contribution.

To make the methodological claim load-bearing we expose an **EMA-blind
training mode** (`scripts/train_model.py --exclude-ema-features`) which
removes all `survey_*` columns from the feature matrix. The resulting
benchmark asks the genuinely interesting question: *can passive-sensing
data alone (wearable, smartphone, climate) predict tomorrow's self-reported
mood?* We recommend always reporting both variants and treating the EMA-
blind numbers as the primary evidence.

### 5.2 Metrics and uncertainty

A small MLP head per task is trained on the pooled representation. NaN
labels (e.g. when the source EMA item was skipped, or HRV is missing on a
hot day) are masked out of the loss. Metrics reported on a held-out
*participant* split are AUROC, AUPRC, F1 at threshold 0.5, expected
calibration error (ECE) with 10 equal-width bins, Brier score, and
confusion matrices.

AUROC and AUPRC are accompanied by 95% percentile bootstrap confidence
intervals (1000 resamples). **The bootstrap unit is the participant, not
the window.** Multiple windows from the same person are not independent;
naively resampling rows treats them as if they were and produces CIs that
are typically an order of magnitude too narrow on longitudinal cohorts.
We resample whole participants with replacement, concatenate their rows,
and recompute the metric. This is the same `bootstrap_ci` helper but with
the `groups=` parameter supplied; the row-level variant is retained only
to reproduce earlier numbers.

### 5.3 Age standardization

The `age_z` feature uses population reference statistics (mean, std)
computed **on the training participants only**, one row per participant.
These reference stats are persisted in the checkpoint meta sidecar
(`downstream.meta.json`) so that inference re-applies the same
transformation. We previously used hardcoded constants matching the
synthetic generator's prior, which works but constitutes a leak of
population statistics across the train/test boundary on real data.

## 6. Interpretability

The deployed model returns faithful **integrated gradients** (IG)
attributions (Sundararajan, Taly & Yan, ICML 2017) for the highest-
probability task. IG was chosen over alternatives for three reasons:

- *Axiomatic completeness*: attributions sum to `f(x) - f(baseline)` in
  logit space, up to small Riemann-quadrature error. We surface that
  quadrature error (``convergence_delta``) alongside the attribution so
  the consumer can sanity-check it.
- *Implementation invariance*: two functionally identical networks produce
  identical attributions. Attention rollout, by contrast, is method-
  specific and not faithful in the sense that matters for clinical-style
  reporting.
- *Latency*: one forward pass plus `n_steps` (default 32) backward passes
  fits inside the API's per-request budget on CPU at our model size.

The baseline is the all-zeros input. This is defensible because every
upstream z-score and personal-baseline-deviation feature is already
centered at zero, so "all zeros" approximates a neutral / typical day.
A cohort-mean baseline is a defensible alternative and is supported by
passing it explicitly.

The rule-based explanation panel from earlier revisions remains in the
codebase as a no-model fallback (so the dashboard and the API still work
without a checkpoint) and as a sanity-check yardstick for new
attribution methods.

## 7. Generalization studies

Three scripts ship for testing the model beyond the standard
train/val/test split:

- ``scripts/run_scale_ablation.py`` sweeps pretraining cohort size over
  e.g. ``{20, 50, 100, 200, 400}`` participants, averaging over multiple
  seeds, and emits the canonical "AUROC vs. N" curve with participant-
  clustered bootstrap CIs. This is the figure that supports — or refutes —
  the foundation-model framing.
- ``scripts/run_climate_holdout.py`` redacts target labels on heat-wave
  (or cold-snap, or smoke-episode) days from the *training* split, fine-
  tunes the warm-started encoder, and evaluates the resulting model on
  those held-out climate-regime windows in addition to matched in-regime
  test sets. A small in-vs-out AUROC gap is evidence the model is
  learning physiology of climate stress rather than overfitting climate-
  specific shortcuts; a large gap is evidence of the opposite. The
  ``--no-retrain`` mode runs the same per-regime slicing against an
  already-saved checkpoint as a quick diagnostic.
- ``scripts/run_fairness_audit.py`` slices the held-out test split by
  participant-level attributes (sex, race/ethnicity, SES, geographic
  region, device generation, depression/anxiety status, age band) and
  reports per-subgroup AUROC with cluster-bootstrap CIs, AUPRC, ECE,
  Brier, and the per-axis equalised-odds violation (max FPR-gap +
  max FNR-gap). A ``--fail-on-violation`` flag exits non-zero when any
  axis exceeds the configured thresholds (default 10pp AUROC spread or
  20pp FPR+FNR drift), suitable as a CI gate. **The synthetic generator
  does not bake in subgroup-specific disparities**, so a properly trained
  model should pass this audit; the value of running it on synthetic
  data is methodological — it exercises every line of the audit pipeline
  before LHFM is pointed at a real cohort.

Both holdout and fairness scripts write tidy CSVs (under
``results/tables/``) and JSON summaries that can be diff'd across runs.

## 8. Baselines

For each task we report three classical baselines trained on a
window-flattened feature matrix (per-feature mean / std / last value /
linear slope across the 14-day window):

- Logistic regression with balanced class weights (`sklearn`)
- Random forest with 300 trees and balanced class weights
- XGBoost with `scale_pos_weight = #neg / #pos`, skipped gracefully if the
  package is not installed.

These baselines are run on every cohort the LHFM encoder is evaluated
on, with the same train/val/test participant splits and the same
held-out test bootstrap procedure. Their purpose is to isolate the
contribution of the self-supervised representation from the
contribution of the feature engineering: a fair comparison must give
the classical baseline the same engineered features the encoder sees.
LifeSnaps and (forthcoming) GLOBEM results report this comparison
side-by-side in the Results section.

## 9. Limitations

The cohort is synthetic; effect sizes were chosen for software-engineering
realism, not for clinical calibration. The encoder is not validated on real
patient data and must not be used clinically. The four downstream tasks are
*proxies*, not diagnoses; for example "low mood" here means a single
self-reported scale value below 4, not a depressive episode.
See `paper/limitations.md` and `paper/ethics.md` for a full discussion.

## 10. Selected references

The methodological choices above draw on a wider literature; this list is a
sample of the most directly relevant work, not an exhaustive bibliography.
A proper publication-grade version should expand this and add formal
citations.

1. Vaswani A, Shazeer N, Parmar N, et al. *Attention is all you need.*
   Advances in Neural Information Processing Systems, 2017. (Transformer
   encoder architecture.)
2. Devlin J, Chang M-W, Lee K, Toutanova K. *BERT: Pre-training of deep
   bidirectional transformers for language understanding.* NAACL, 2019.
   (Masked-feature reconstruction objective; the wearable-side analog used
   here.)
3. Chen T, Kornblith S, Norouzi M, Hinton G. *A simple framework for
   contrastive learning of visual representations.* ICML, 2020.
   (SimCLR-style InfoNCE; we use a participant-aware variant.)
4. Yuan H, et al. *Self-supervised learning for human activity recognition
   using 700,000 person-days of wearable data.* npj Digital Medicine, 2024.
   (Large-scale SSL on accelerometer streams.)
5. Xu X, et al. *GLOBEM: Cross-dataset generalization of longitudinal
   human behavior modeling.* NeurIPS Datasets, 2022. (Cross-cohort
   evaluation methodology and the real-data target we'd port to.)
6. Saeb S, et al. *Mobile phone sensor correlates of depressive symptom
   severity in daily-life behavior.* JMIR, 2015. (Original phone passive-
   sensing-to-mood evidence.)
7. Rothfusz LP. *The heat index "equation".* NWS Technical Attachment SR/SSD
   92-22, 1990. (Heat-index regression used in the climate module.)
8. Ebi KL, Capon A, Berry P, et al. *Hot weather and heat extremes: health
   risks.* The Lancet, 2021. (Climate-health framing for the climate-
   vulnerability target.)
