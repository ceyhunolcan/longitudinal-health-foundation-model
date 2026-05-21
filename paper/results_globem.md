## Results: GLOBEM cohort

> **Status.** This section is scaffolded against the GLOBEM cohort
> (Xu et al. 2022, NeurIPS Datasets and Benchmarks track)<<TODO: ref>>
> distributed via PhysioNet under credentialed access. Numbers and
> figures will be filled in once credentialed download is complete and
> the LHFM pipeline has been re-run on this cohort. The prose below
> describes the planned analysis; values marked `<<TODO>>` are pending.

### Cohort and protocol

GLOBEM is a four-year multi-institutional longitudinal study of college
students' behavioural-health signals, comprising passive-sensing data
from approximately <<TODO: confirm n>> participants observed daily over
academic terms (typically 10–14 weeks per participant). Each
participant contributes wearable, smartphone, and self-report data.
This cohort is approximately ten times the size of LifeSnaps in
participant count, and unlike LifeSnaps it includes smartphone passive
sensing (screen time, phone-unlock counts, location features), enabling
the full four-modality LHFM input space.

We applied the same protocol described in §Methods to GLOBEM: a
70/15/15 participant-level split, SSL pretraining on the training
participants only, downstream fine-tuning with early stopping on the
primary task's validation AUROC, best-checkpoint reload, and a
participant-clustered bootstrap on the held-out test set with 1,000
iterations. Where LifeSnaps modality coverage forced us to evaluate
two of four downstream tasks, GLOBEM permits evaluation of all four.

### Modality coverage on this cohort

| Modality | LifeSnaps | GLOBEM |
| :--- | :---: | :---: |
| Wearable (HRV, RHR, sleep) | ✓ | ✓ |
| Smartphone passive sensing | ✗ | ✓ |
| EMA self-report | ✓ (sparse) | ✓ |
| Climate enrichment | ✗ | <<TODO: confirm>> |

### Held-out test results

| Task                  | Model           | AUROC | 95% CI       | AUPRC | n_pos | n_total |
| :-------------------- | :-------------- | ----: | :----------- | ----: | ----: | ------: |
| `low_mood`            | LHFM            | <<TODO>> | <<TODO>>     | <<TODO>> | <<TODO>> | <<TODO>> |
| `low_mood`            | Logistic regr.  | <<TODO>> | –         | <<TODO>> | <<TODO>> | <<TODO>> |
| `low_mood`            | Random forest   | <<TODO>> | –         | <<TODO>> | <<TODO>> | <<TODO>> |
| `high_stress`         | LHFM            | <<TODO>> | <<TODO>>     | <<TODO>> | <<TODO>> | <<TODO>> |
| `high_stress`         | Logistic regr.  | <<TODO>> | –         | <<TODO>> | <<TODO>> | <<TODO>> |
| `high_stress`         | Random forest   | <<TODO>> | –         | <<TODO>> | <<TODO>> | <<TODO>> |
| `sleep_disruption`    | LHFM            | <<TODO>> | <<TODO>>     | <<TODO>> | <<TODO>> | <<TODO>> |
| `sleep_disruption`    | Logistic regr.  | <<TODO>> | –         | <<TODO>> | <<TODO>> | <<TODO>> |
| `sleep_disruption`    | Random forest   | <<TODO>> | –         | <<TODO>> | <<TODO>> | <<TODO>> |

**Table 3.** Held-out test metrics on the GLOBEM cohort. Same
participant-clustered bootstrap procedure as Table 2.

### Cross-cohort comparison

A direct LifeSnaps-versus-GLOBEM comparison on the two tasks evaluable
on both cohorts (`high_stress`, `sleep_disruption`) is reported in
Table <<TODO>>. The interesting empirical question is whether the
direction of the LHFM-versus-classical-baseline gap reverses across
cohorts: on LifeSnaps, LHFM beat the baselines on `high_stress` but
lost on `sleep_disruption`. If the gap on `sleep_disruption` closes or
reverses on GLOBEM, the most parsimonious explanation is that
smartphone modality input is doing the additional work, since that is
the modality LifeSnaps lacks.

### Climate-regime holdout

GLOBEM participants span <<TODO: confirm number>> distinct geographic
sites. We hold out one geographic site at a time, retrain the
downstream heads on the remaining sites' training participants, and
evaluate on the held-out site's held-out test participants. This
isolates *site shift* (climate, season, university culture, recruitment
demographics) from the within-cohort generalization measured in
Table 3. Results are summarized in Figure 4: <<TODO: AUROC by held-out
site for each task>>.

This procedure differs from the climate-regime holdout reported on the
synthetic cohort in §Results, Synthetic, in that the regimes here are
real geographic and seasonal differences rather than synthetic-generator
contrasts. The synthetic holdout was useful for verifying that the
model behaves sensibly under regime shift in the first place; the
GLOBEM holdout tests how large that effect is in practice.

### Limitations specific to GLOBEM

<<TODO: populate after credentialing. Expected items:
- Participant population is college students; results may not generalise
  to clinical, community, or workplace populations.
- Self-report instruments (PHQ-9, PSS, etc.) used by the GLOBEM team
  differ from the SEMA scales used in LifeSnaps; we apply a unified
  binarisation pipeline but the constructs being measured are not
  identical across cohorts.
- Some sites have substantially more participants than others; the
  climate-regime holdout therefore has uneven statistical power per
  site, which should be reported alongside the per-site AUROCs.
>>
