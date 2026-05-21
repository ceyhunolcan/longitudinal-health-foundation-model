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
| `low_mood`            | LHFM            | 0.752 | [0.691, 0.814]     | 0.673 | 647   | 1819    |
| `low_mood`            | Logistic regr.  | **0.817** | –              | 0.759 | 647   | 1819    |
| `low_mood`            | Random forest   | 0.807 | –                  | 0.749 | 647   | 1819    |
| `high_stress`         | LHFM            | 0.604 | [0.504, 0.708]     | 0.451 | 625   | 1819    |
| `high_stress`         | Logistic regr.  | 0.672 | –                  | 0.568 | 625   | 1819    |
| `high_stress`         | Random forest   | **0.674** | –              | 0.571 | 625   | 1819    |
| `sleep_disruption`    | LHFM            | 0.596 | [0.500, 0.680]     | 0.212 | 334   | 2426    |
| `sleep_disruption`    | Logistic regr.  | 0.720 | –                  | 0.440 | 334   | 2426    |
| `sleep_disruption`    | Random forest   | **0.747** | –              | 0.424 | 334   | 2426    |
| `climate_vulnerable`  | LHFM            | 0.937 | [0.914, 0.957]     | 0.382 | 94    | 2426    |
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
6.5 AUROC points on `low_mood`, 7 points on `high_stress`, and
12-15 points on `sleep_disruption`. On `climate_vulnerable`, where
LHFM achieves the highest absolute AUROC of the four (0.937, 95 %
CI [0.914, 0.957]), random forest narrowly exceeds it (0.948). This
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

A second factor affecting the synthetic comparison is a
multi-task best-checkpoint subtlety in the training pipeline. The
trainer's early-stopping signal is the *primary* task's validation
AUROC (the first task in the configured list, `low_mood` on this
cohort). The remaining heads share the same encoder and may peak at
different epochs; on the synthetic run logged at run-tag
`synthetic_paper`, the `climate_vulnerable` head's validation AUROC
climbed from 0.67 at epoch 1 to 0.94 at epoch 6, but training
halted at epoch 6 because `low_mood`'s validation AUROC had not
improved past its epoch-1 value of 0.758. The reported test
numbers in Table 1 reflect the epoch-1 checkpoint for the non-
primary heads, which understates their per-task best performance.
On LifeSnaps this is the right design choice — only one task was
evaluable — but on a multi-task synthetic cohort the conservative
side-effect is visible. A per-task best-checkpoint variant is the
straightforward fix and is documented as a planned trainer
extension; we leave it out of the present comparison because
selectively maximising each head's checkpoint is the kind of
post-hoc choice we want the comparison to remain free of.

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

<<TODO: insert subgroup-stratified AUROC table once the
`make fairness-audit` target has been re-run against the
synthetic_paper checkpoint. The synthetic cohort is designed not to
contain systematic disparities, so we expect equalised-odds gaps
under 0.10 on every axis; any observed gap larger than that is a
model-fit artefact rather than a true demographic effect, and the
audit's purpose here is to document the format reviewers should
expect when LHFM is applied to a real cohort with real disparities.
>>

### Climate-regime holdout

The synthetic generator includes two distinct climate regimes within
a single 90-day window: a baseline temperate period and two
explicit heat-wave events (consecutive days with peak temperature
above 35 °C). We evaluate generalisation across regimes by holding
out the heat-wave days from training and reporting LHFM's AUROC on
the heat-wave subset of the held-out test set, with the same
participant-clustered bootstrap procedure.

<<TODO: insert climate-regime holdout AUROC once the
`make climate-holdout` target has been re-run against the
synthetic_paper checkpoint. This is the synthetic counterpart to
GLOBEM's geographic-site holdout (§Results, GLOBEM); the former
isolates climate variation within a single cohort, the latter
isolates climate-and-everything-else variation between cohorts.
>>

### Interpretability

For every held-out test window we compute integrated gradients
(Sundararajan et al. 2017)<<TODO: cite>> over the encoder input
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
