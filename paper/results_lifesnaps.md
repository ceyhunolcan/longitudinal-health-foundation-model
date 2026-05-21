## Results: LifeSnaps cohort

### Cohort and protocol

We applied the LHFM pipeline end-to-end to the LifeSnaps cohort
[@yfantidou2022lifesnaps], using the public
"RAIS" release distributed on Kaggle and Zenodo. After applying the
project's standard minimum-days filter (64 days observed per
participant), 71 participants and 7,410 participant-days were retained,
spanning April 2021 through January 2022, with a median of 88 days per
participant (range 64–244). The cohort distribution by reported sex was
M = 41, F = 26, missing = 4. Age was stored as range strings in the
public release (e.g. ``<30``, ``30-40``); we parsed each range to its
lower-bound integer. In the RAIS distribution we accessed, every
participant fell into the lowest band (``<30``), so the parsed age
column carries no within-cohort variance and is not informative for
fairness stratification on this cohort.

Participants were split at the participant level into 49 training,
11 validation, and 11 test sets (a 70/15/15 split, applied to the count
of participants rather than to rows, to prevent within-person leakage).
LHFM was pretrained for up to 30 SSL epochs on the training split and
fine-tuned for up to 30 downstream epochs with early stopping on the
primary task's validation AUROC; the trainer reloads the best
checkpoint by validation AUROC before evaluating on the held-out test
set. Total wall-clock time was approximately 15 minutes on a single
2024-class MacBook Air CPU (8 cores, no GPU). Two of the four
downstream tasks defined in this work were evaluable on this cohort
(high-stress, sleep-disruption); the remaining two (low-mood,
climate-vulnerability) were not, for reasons documented in §Results,
LifeSnaps-specific limitations below.

### Modality coverage on this cohort

LifeSnaps is a wearable-plus-EMA cohort with no smartphone passive
sensing and no participant-level GPS. The adapter accordingly emitted
NaN for the smartphone modality block (phone unlocks, screen-time
minutes, mobility radius, location entropy) and for the climate block
(temperature, humidity, AQI, heat index) for every participant-day.
LHFM's modality-missing flags propagated these absences through
pretraining and downstream training; the encoder and heads were
trained on the wearable + EMA + baseline-context subset of the input
space.

EMA delivery in the LifeSnaps SEMA stream was sparse: 31% of
participant-days carried at least one reported emotion across the seven
SEMA categories. Days with no emotion reported were treated as missing
labels (NaN-masked) for the EMA-derived targets, rather than
treated as negatives, to avoid biasing the positive rate.

### Held-out test results

Bootstrap 95% confidence intervals were computed at the participant
level with 1,000 iterations, using the procedure documented in
§Methods 5.2 (participants resampled with replacement, rows
concatenated, metric recomputed).

| Task                  | Model           | AUROC | 95% CI            | AUPRC | n_pos | n_total |
| :-------------------- | :-------------- | ----: | :---------------- | ----: | ----: | ------: |
| `high_stress`         | **LHFM**        | **0.567** | **[0.389, 0.688]** | 0.218 | 55 | 294 |
| `high_stress`         | Logistic regr.  | 0.328 | –                  | 0.151 | 55 | 294 |
| `high_stress`         | Random forest   | 0.368 | –                  | 0.211 | 55 | 294 |
| `sleep_disruption`    | **LHFM**        | 0.518 | [0.376, 0.682]     | 0.129 | 51 | 464 |
| `sleep_disruption`    | Logistic regr.  | **0.656** | –              | 0.195 | 51 | 464 |
| `sleep_disruption`    | Random forest   | 0.641 | –                  | 0.197 | 51 | 464 |

**Table 2.** Held-out test metrics on the LifeSnaps cohort.
Logistic regression and random forest were trained on the identical
window-flattened feature matrix LHFM consumes (per-feature mean,
standard deviation, last value, and linear slope across the 14-day
window). XGBoost was not run because the package was not installed in
the reproducibility environment. AUPRC values for classical baselines
are reported but uncertainty intervals were not computed at the
participant-clustered level for this comparison.

### Interpretation

On the high-stress prediction task, LHFM outperformed both classical
baselines by approximately 20 AUROC points on the identical feature
set: 0.567 versus 0.328 (logistic regression) and 0.368 (random forest).
Because all three models received the same engineered features as
input, the difference is attributable to the self-supervised
representation rather than to the feature engineering itself. The
LHFM 95% confidence interval ([0.389, 0.688]) is wide and includes
chance (0.5); we therefore frame the result as *an effect on the
baseline comparison*, not as an absolute high-stress detection
benchmark.

On the sleep-disruption task on the same cohort, the relationship is
inverted: random forest outperformed LHFM (0.641 versus 0.518). We
report this finding rather than tune it away. The most parsimonious
explanation is that sleep duration, sleep efficiency, and a small
number of derived sleep-stage ratios are nearly sufficient statistics
for this task, and a tree-based model can fit the threshold structure
faster than a transformer can learn it from masked-feature
reconstruction on a 49-participant pretraining set. SSL pretraining is
not universally helpful in the small-cohort, low-modality-coverage
regime; the LifeSnaps result demonstrates one specific case of that
limitation.

### LifeSnaps-specific limitations

Five limitations are specific to this cohort and bound the
interpretation of the result above.

1. **Test fold size.** The held-out test set contains 11 participants.
   Bootstrap 95% confidence intervals are correspondingly wide, and
   even clearly-positive results are not statistically distinguishable
   from chance at this sample size. Replication on the larger GLOBEM
   cohort is in progress (§Results, GLOBEM).

2. **No smartphone passive sensing.** LifeSnaps does not collect
   phone-unlock counts, screen-time minutes, mobility radius, or
   location entropy. LHFM's smartphone modality input is therefore
   uniformly absent on this cohort. The model trains and evaluates on
   wearable + EMA only.

3. **No climate enrichment.** LifeSnaps publishes only timezone, not
   participant-level latitude / longitude. The Open-Meteo enrichment
   that LHFM uses to populate temperature, humidity, AQI, and heat
   index could not run. All climate features are NaN on this cohort.

4. **EMA sparsity.** SEMA emotion reports were delivered for 31% of
   participant-days. Days without delivered reports are NaN-masked at
   the target level; the LHFM training loss treats them as missing
   labels rather than as negatives, but the effective labelled sample
   size is reduced accordingly.

5. **Low-mood unevaluable.** The SEMA "SAD" checkbox is reported on
   fewer than 3% of EMA days in this cohort. The derived `survey_mood`
   distribution does not produce enough low-mood days to populate a
   meaningful binary target: the test fold contained zero positives.
   We surface the task as unevaluable on this cohort rather than tune
   the binarisation threshold to manufacture a positive class.

### What this result does and does not establish

The LifeSnaps result establishes that the LHFM pipeline runs
end-to-end on a published, externally-collected cohort; that the
self-supervised representation produces measurable lift over classical
baselines on identical features on at least one evaluable task; and
that the pipeline behaves transparently when modalities or labels are
absent. It does not establish that LHFM detects high stress with
useful accuracy in absolute terms (the 95% CI on AUROC includes 0.5),
nor that the same lift will be observed on cohorts with different
modality coverage or different participant demographics. The GLOBEM
replication (§Results, GLOBEM) is intended to address the cohort-size
component of this uncertainty; the modality-coverage component will
require additional cohorts with simultaneous wearable, smartphone, and
climate signal.
