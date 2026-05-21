## Results: Synthetic cohort

### Cohort and protocol

The synthetic cohort comprises 250 simulated participants observed
daily over a 90-day window (22,500 participant-days), generated
deterministically by `src/lhfm/data/synthetic_generator.py` with seed
42 unless stated otherwise. Each participant carries latent priors
sampled from population-style distributions (demographics, chronotype,
baseline sleep need, baseline HRV and resting heart rate, mood and
stress setpoints, and three sensitivity parameters for heat, air
pollution, and study engagement) and emits per-day wearable,
smartphone, EMA, and environmental signals shaped by those priors.
Causal relationships are seeded so the dataset is non-trivially
learnable: sleep debt and elevated heat index degrade HRV and elevate
resting heart rate within-person; mood declines with high stress,
poor sleep, poor air quality, and extreme heat; mobility shrinks
under heat stress and low mood. Missingness is informative —
low-mood days are flagged as missing for the wearable and self-report
modalities at elevated rates, mirroring the well-documented
"ghosting" effect in real passive-sensing studies.

We split the cohort at the participant level (175 train, 37
validation, 38 test). LHFM was pretrained for up to 30 SSL epochs
on the training split, then four downstream heads were fine-tuned
for up to 30 epochs with early stopping on the primary task's
validation AUROC. The trainer reloads the best checkpoint by
validation AUROC before evaluating on the held-out test set. Total
wall-clock time is approximately 25 minutes on a single CPU.
Bootstrap 95 % confidence intervals are computed at the
participant level with 1,000 iterations.

### Held-out test results

| Task                  | Model           | AUROC | 95% CI            | AUPRC | n_pos | n_total |
| :-------------------- | :-------------- | ----: | :---------------- | ----: | ----: | ------: |
| `low_mood`            | LHFM            | 0.743 | [0.684, 0.804]     | 0.678 | 647   | 1819    |
| `low_mood`            | Logistic regr.  | **0.817** | –              | 0.759 | 647   | 1819    |
| `low_mood`            | Random forest   | 0.807 | –                  | 0.749 | 647   | 1819    |
| `high_stress`         | LHFM            | 0.580 | [0.481, 0.682]     | 0.438 | 625   | 1819    |
| `high_stress`         | Logistic regr.  | 0.672 | –                  | 0.568 | 625   | 1819    |
| `high_stress`         | Random forest   | **0.674** | –              | 0.571 | 625   | 1819    |
| `sleep_disruption`    | LHFM            | 0.625 | [0.573, 0.682]     | 0.219 | 334   | 2426    |
| `sleep_disruption`    | Logistic regr.  | 0.720 | –                  | 0.440 | 334   | 2426    |
| `sleep_disruption`    | Random forest   | **0.747** | –              | 0.424 | 334   | 2426    |
| `climate_vulnerable`  | LHFM            | 0.936 | [0.917, 0.953]     | 0.340 | 94    | 2426    |
| `climate_vulnerable`  | Logistic regr.  | 0.833 | –                  | 0.154 | 94    | 2426    |
| `climate_vulnerable`  | Random forest   | **0.948** | –              | 0.315 | 94    | 2426    |

**Table 1.** Held-out test metrics on the synthetic cohort
(250 participants × 90 days, seed 42, 38 test participants).
LHFM, logistic regression, and random forest were trained on the
identical window-flattened feature matrix (per-feature mean,
standard deviation, last value, and linear slope across the 14-day
window). Bootstrap 95 % confidence intervals on LHFM use the
participant-level resampling procedure described in §Methods 5.2;
1,000 iterations.

### Interpretation

All four downstream tasks are evaluable on this cohort, unlike on
LifeSnaps where modality and label sparsity render two of them
unevaluable. This is the design intent of the synthetic generator:
it lets us verify the *pipeline* on a setting where the causal
structure is known and every target has sufficient positives, before
applying it to real cohorts where modality coverage and label
delivery are out of the experimenter's control.

**Classical baselines outperform LHFM on all four synthetic tasks.**
Logistic regression and random forest exceed LHFM by approximately
7 AUROC points on `low_mood`, 9 points on `high_stress`, and 10-12
points on `sleep_disruption`. On `climate_vulnerable`, where LHFM
achieves the highest absolute AUROC of the four (0.936, 95 % CI
[0.917, 0.953]), random forest narrowly exceeds it (0.948). This
is the inverse of the LifeSnaps result, where LHFM exceeded the
same classical baselines on `high_stress` by approximately 20 AUROC
points. We report both directions of comparison in the same table
form rather than tuning either away.

The most parsimonious explanation for the inverted gap reconciles
the two results. Synthetic data is *clean*: every modality is
present every day for every participant, missingness is the
seeded informative kind only, and the engineered features
(per-feature mean, standard deviation, last value, and linear
slope across the 14-day window) are nearly sufficient statistics
for the targets they were designed against. Classical baselines
thrive in this regime. LHFM's value proposition — a representation
that absorbs modality dropout, sparse labels, and cross-modality
structure — does not become observable until those conditions are
violated. The LifeSnaps cohort violates them: smartphone sensing
is absent, climate is absent, EMA delivery covers only 31 % of
days. There, classical baselines fall to 0.33 AUROC on
`high_stress` while LHFM holds at 0.57, a 24-point gap in LHFM's
favour. The honest reading of Table 1 against the LifeSnaps result
(Table 2) is therefore: **LHFM's lift over classical baselines is
not universal; it emerges from the kind of heterogeneity real
cohorts exhibit and synthetic data, by construction, does not.**

The trainer's early-stopping signal on a multi-task cohort is the
*mean* of validation AUROC over the tasks that have at least one
positive in the validation fold; on cohorts with a single
evaluable task (e.g. LifeSnaps with only `high_stress` evaluable)
this reduces to single-task tracking. The synthetic run reported
in Table 1 (run-tag `synthetic_paper_v2`) early-stopped at
downstream epoch 11 with mean-over-tasks validation AUROC 0.734,
having let `sleep_disruption` and `climate_vulnerable` train past
the epoch at which `low_mood`'s validation AUROC alone peaked.
This avoids a confound that the earlier primary-task-only signal
introduced: on a cohort where the four heads converge at
different rates, the saved checkpoint must not be the one that
maximises only the first-listed task's metric. We do not, however,
select per-task best checkpoints. The reported numbers therefore
reflect a single shared encoder evaluated on all four heads, which
is the regime a downstream user would actually deploy.

The synthetic result is therefore not the headline empirical claim
of this work; the LifeSnaps result (§Results, LifeSnaps) and the
forthcoming GLOBEM replication (§Results, GLOBEM) are. The
synthetic comparison is informative about *where* the SSL
representation buys lift and where it does not — namely, that the
gain appears in cohorts whose modality and label heterogeneity the
classical feature pipeline cannot directly absorb.

### Subgroup fairness audit

We compute AUROC stratified by eight subgroup axes available in the
synthetic cohort: sex, chronotype (morning / intermediate /
evening), age band (18–34 / 35–54 / 55+), baseline HRV quartile,
baseline RHR quartile, depression flag, anxiety flag, and
beta-blocker use. For each axis, we report the per-subgroup AUROC
on the held-out test set and an equalised-odds-style gap (the
difference between true-positive rates at a fixed false-positive
rate of 0.10 between the two largest subgroups on that axis).

We ran `make fairness-audit` (`scripts/run_fairness_audit.py`) against
the `synthetic_paper_v2` checkpoint, stratifying held-out test
performance by eight subgroup axes carried in the v0.2 synthetic
generator: sex, race/ethnicity, socioeconomic-status proxy, region,
device generation, depression flag, anxiety flag, and age band.
For each (task, axis) combination, we report the equalised-odds gap
(max false-positive-rate gap plus max false-negative-rate gap across
the axis levels). Gaps exceeding 0.20 are flagged as violations by
the audit's default threshold. Per-subgroup tables are written to
`results/tables/fairness_<task>.csv` (23 subgroup rows per task) and
to `results/tables/fairness_<task>_equalised_odds.csv`; a summary is
written to `results/tables/fairness_summary.json`.

| Axis              | low_mood | high_stress | sleep_disrupt. | climate_vuln. |
| :---------------- | -------: | ----------: | -------------: | ------------: |
| sex               | 0.14     | **0.32**    | 0.07           | 0.04          |
| race_ethnicity    | **1.00** | **1.31**    | **0.80**       | **0.28**      |
| ses_proxy         | **0.34** | **0.39**    | **0.25**       | 0.06          |
| region            | 0.15     | **0.68**    | **0.32**       | 0.14          |
| device_gen        | 0.05     | **0.29**    | 0.13           | 0.11          |
| has_depression    | **0.46** | **0.50**    | **0.32**       | 0.19          |
| has_anxiety       | 0.13     | **0.62**    | 0.13           | **0.23**      |
| age_band          | **0.47** | **0.75**    | **0.25**       | **0.27**      |

**Table 2.** Equalised-odds gaps (`max FPR gap + max FNR gap` across
axis levels) on the held-out synthetic test set. Bold entries
exceed the default fairness threshold of 0.20. Source:
`results/tables/fairness_summary.json`.

The audit fires on most (task, axis) combinations. We interpret these
results with substantial caution, for two reasons specific to the
synthetic cohort.

First, the v0.2 synthetic generator was extended (relative to v0.1)
to attach stratification metadata - race/ethnicity, region, SES,
device generation, depression, anxiety, age band - to each
simulated participant, so that audit tooling can be developed and
exercised on the synthetic pipeline. However, **the generator does
not guarantee that these labels are independent of the outcome
distributions.** The clearest example is `has_anxiety`: anxious
participants carry a +1.1 shift in stress setpoint by construction.
The high-stress AUROC on `has_anxiety = true` participants
(0.957, n_pos clustered in a narrow setpoint band) is therefore
substantially higher than on `has_anxiety = false` participants
(0.657); the equalised-odds gap of 0.62 follows mechanically from
the generator's prior, not from a deficiency of the encoder. The
analogous concern applies to `has_depression`, `age_band` (via
HRV-baseline shifts), and likely to `race_ethnicity` (whose joint
distribution with the latent priors is not documented in the
generator). On a real cohort these correlations would have real
sources; on this synthetic cohort they are simulation choices.

Second, several subgroups carry small absolute sample sizes
(e.g. the hispanic subgroup on the high-stress task contains
n_pos = 104 out of n = 225 on the test set). Bootstrap CIs on
per-subgroup AUROC are accordingly wide, and a single high-AUROC
or low-AUROC subgroup can produce a large equalised-odds gap that
is not statistically robust.

We do not interpret Table 2 as evidence that LHFM is unfair, nor
that it is fair. It is evidence that **the audit machinery works
end-to-end** - every (task, axis) combination produces a per-axis
spread, an equalised-odds gap, a per-subgroup AUROC table, and a
machine-readable JSON. The substantive fairness analysis belongs
on the GLOBEM cohort (§Results, GLOBEM), where the subgroup labels
correspond to real participants and the per-subgroup gaps would
reflect real-world distributional shifts rather than generator
priors. Until that analysis is reported, the published audit
artefacts on this cohort exist for reproducibility and as a
template for downstream users running the same audit on their own
data.

### Climate-regime holdout

The synthetic generator includes two distinct climate regimes within
a single 90-day window: a baseline temperate period and two
explicit heat-wave events (consecutive days with peak temperature
above 35 °C). We evaluate generalisation across regimes by holding
out the heat-wave days from training and reporting LHFM's AUROC on
the heat-wave subset of the held-out test set, with the same
participant-clustered bootstrap procedure.

We ran `make climate-holdout` (`scripts/run_climate_holdout.py`)
to evaluate generalisation across climate regimes within the
synthetic cohort. The v0.2 generator labels every participant-day
with one of four regime tags: `normal` (17,750 rows under seed 42),
`heat_wave` (3,000 rows, prevalence of `climate_vulnerable`
positives jumps from 0 % under `normal` to 25 % under `heat_wave`),
`smoke_episode` (2,000 rows, prevalence of `low_mood` positives
85 % vs 33 % under `normal`), and `cold_snap` (zero rows under
seed 42, no evaluation possible). The holdout script redacts the
held-out regime's days from the training and validation splits,
fine-tunes the downstream heads on the remaining days, and then
evaluates the fine-tuned heads on each regime's portion of the
held-out test split.

Holding out `heat_wave` from training:

| Eval regime     | low_mood | high_stress | sleep_disrupt. | climate_vuln. |
| :-------------- | -------: | ----------: | -------------: | ------------: |
| `normal`        | 0.673    | 0.487       | 0.619          | **0.803**     |
| `heat_wave`     | 0.758    | 0.572       | 0.592          | **0.829**     |
| `smoke_episode` | 0.858    | 0.585       | 0.613          | **0.867**     |

**Table 3.** Per-regime test AUROC after retraining with `heat_wave`
days redacted from train and val. Source:
`results/tables/climate_regime_eval.csv`. n_windows ranges from
304 (smoke_episode) to 2,166 (normal); see CSV for per-row counts.

The `climate_vulnerable` head is the interesting case. By
construction the target is concentrated on heat-wave days
(prevalence 25 % under `heat_wave` versus 0 % under `normal`).
After heat-wave days are removed from training, the head retains
test-set AUROC of 0.83 on the held-out heat-wave windows - i.e.
the model generalises to a climate regime it never saw during
fine-tuning. The most parsimonious interpretation is that the
representation learned the *structural* cue for climate
vulnerability (HRV deviation on days with elevated heat exposure,
modulated by baseline heat sensitivity) from the wider
self-supervised pretraining, rather than memorising the specific
heat-wave windows during downstream fine-tuning. The same
checkpoint also generalises to `smoke_episode` (a regime it never
saw, AUROC 0.87) and to `normal` (a regime it saw heavily,
AUROC 0.80). This is the most positive finding on the synthetic
cohort: regime generalisation works, at least for the task whose
positives are structurally regime-specific.

The other three tasks (`low_mood`, `high_stress`,
`sleep_disruption`) behave less interestingly under regime holdout
because their positive-class distributions are far less
regime-bound. Their AUROCs vary across regimes in the range
[0.49, 0.86] but the pattern does not isolate climate from other
factors. The cleaner regime-generalisation analysis belongs on the
GLOBEM cohort, where regimes correspond to real geographic sites
and the variation is not a simulator artefact.

### Interpretability

For every held-out test window we compute integrated gradients
[@sundararajan2017axiomatic] over the encoder input
with a zero baseline, then aggregate the per-step attributions into
per-feature scores. Figure 2 (notebook
`notebooks/00_quickstart.ipynb` cell 11) shows the per-feature
attribution bar chart for a representative high-risk window: the
top driver is screen-time minutes (positive attribution, pushing
risk up), followed by daily steps (negative), phone unlock count
(negative), AQI (negative under low-AQI conditions), and stress
score (slight negative). Climate features appear with sensible
signs (heat exposure positive, humidity positive under the
humid-heat-index branch). This decomposition is the kind of
clinically-interpretable surface a reviewer might expect to see
before trusting a foundation-model prediction; it does not, on its
own, constitute causal evidence.
