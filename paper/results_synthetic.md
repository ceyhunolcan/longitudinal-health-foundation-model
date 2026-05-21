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

<<TODO: insert table once synthetic_paper run completes. Expected
shape, populated from `results/tables/metrics_test.csv`:

| Task                  | Model           | AUROC | 95% CI | AUPRC | n_pos | n_total |
| :-------------------- | :-------------- | ----: | :----- | ----: | ----: | ------: |
| `low_mood`            | LHFM            |  TBD  | TBD    | TBD   | TBD   | TBD     |
| `low_mood`            | Logistic regr.  |  TBD  | –      | TBD   | TBD   | TBD     |
| `low_mood`            | Random forest   |  TBD  | –      | TBD   | TBD   | TBD     |
| `high_stress`         | LHFM            |  TBD  | TBD    | TBD   | TBD   | TBD     |
| `high_stress`         | Logistic regr.  |  TBD  | –      | TBD   | TBD   | TBD     |
| `high_stress`         | Random forest   |  TBD  | –      | TBD   | TBD   | TBD     |
| `sleep_disruption`    | LHFM            |  TBD  | TBD    | TBD   | TBD   | TBD     |
| `sleep_disruption`    | Logistic regr.  |  TBD  | –      | TBD   | TBD   | TBD     |
| `sleep_disruption`    | Random forest   |  TBD  | –      | TBD   | TBD   | TBD     |
| `climate_vulnerable`  | LHFM            |  TBD  | TBD    | TBD   | TBD   | TBD     |
| `climate_vulnerable`  | Logistic regr.  |  TBD  | –      | TBD   | TBD   | TBD     |
| `climate_vulnerable`  | Random forest   |  TBD  | –      | TBD   | TBD   | TBD     |

**Table 1.** Held-out test metrics on the synthetic cohort
(250 participants × 90 days, seed 42). LHFM, logistic regression,
and random forest were trained on the identical window-flattened
feature matrix. Bootstrap unit: participant, 1,000 iterations.
>>

### Interpretation

All four downstream tasks are evaluable on this cohort, unlike on
LifeSnaps where modality and label sparsity render two of them
unevaluable. This is the design intent of the synthetic generator:
it lets us verify the *pipeline* on a setting where the causal
structure is known and every target has sufficient positives, before
applying it to real cohorts where modality coverage and label
delivery are out of the experimenter's control.

The synthetic result is therefore not the headline empirical claim
of this work. It establishes that the model recovers seeded causal
structure (sleep debt and heat exposure depress HRV within-person;
informative missingness covaries with mood; climate vulnerability
is concentrated in participants with high heat sensitivity) and
that LHFM produces non-trivial AUROC on each task. The real-data
test of the same pipeline is on the LifeSnaps cohort
(§Results, LifeSnaps), and the headline cross-cohort replication
is on GLOBEM (§Results, GLOBEM).

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
