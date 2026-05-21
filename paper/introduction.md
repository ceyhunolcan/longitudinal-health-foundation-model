## Introduction

Continuous passive sensing through wearable devices and smartphones
has emerged over the past decade as a credible substrate for
behavioural-health research. Heart-rate variability, resting heart
rate, sleep duration and architecture, step counts, mobility radius,
and patterns of phone use can each be measured at minute or sub-minute
resolution across months at a time, with comparatively little
participant burden. A growing body of work has shown that these
signals carry information about momentary affect, sleep disruption,
and stress at the within-person level: Yfantidou et al. (2022)
released the LifeSnaps cohort and demonstrated that Fitbit Sense
streams predict daily SEMA-measured stress; Xu et al. (2022)
released the GLOBEM cohort spanning four years and multiple
institutions, with passive-sensing features predictive of depression
and anxiety symptoms; and several large industry efforts in 2024 and
2025 have reported foundation-model-scale pretraining on heart-rate
and wearable time series.<<TODO: cite Apple, Google PHM, Wu et al.>>
At the same time, climate exposure — heat, humidity, and air quality —
is increasingly recognised as a measurable driver of cardiovascular
and affective signals,<<TODO: cite Doerr 2018; Burke 2018>> opening
the possibility of integrated behavioural-and-environmental health
monitoring at scale.

Three persistent technical gaps separate this research direction from
clinical or population-health deployment. **First**, the existing
representation-learning methods for longitudinal passive sensing are
heterogeneous and largely cohort-specific: feature pipelines built
for LifeSnaps do not run on GLOBEM and vice versa, which prevents
the kind of cross-cohort replication that the broader
machine-learning-for-health community now expects as a minimum bar
for credibility.<<TODO: cite the Saeed/Goldstein call-to-arms>>
**Second**, where foundation-model-scale methods have been published,
code and weights are frequently withheld, evaluation protocols are
under-specified, and uncertainty intervals are reported at the
row-level rather than at the participant-level — producing
confidence intervals that are typically an order of magnitude
narrower than the participant-clustered alternative would yield.
**Third**, none of the existing efforts integrate climate exposure
as a first-class input modality, despite the documented physiological
and affective effects of heat and air pollution; climate is treated,
at best, as a covariate in regression, not as a stream the
representation should attend to.

We address these gaps with the **Longitudinal Health Foundation
Model (LHFM)**, a self-supervised multimodal transformer over
per-day wearable, smartphone, EMA, and climate signals, released
end-to-end under MIT license with synthetic data, both real-data
adapters (LifeSnaps and GLOBEM), training and evaluation code, and a
public reproducibility recipe. Our four contributions are:
(1) **A cohort-agnostic adapter abstraction** that lets the same
model and training code consume real cohorts with heterogeneous
modality coverage, including the LifeSnaps wearable + EMA cohort
(no smartphone, no climate) and the GLOBEM full-modality cohort.
(2) **A participant-clustered evaluation pipeline** that resamples
participants rather than rows under the bootstrap and reports
held-out test AUROC, AUPRC, ECE, and Brier scores with 95 %
confidence intervals as the default, against logistic-regression
and random-forest baselines trained on the identical engineered
features. (3) **A fairness audit and a climate-regime holdout** that
report subgroup-stratified AUROC and per-regime AUROC respectively,
both runnable from a single command. (4) **An integrated-gradients
interpretability pass** that decomposes any single-window prediction
into per-feature attributions, surfaced through a dashboard that any
user can run locally. We demonstrate the full pipeline on the
LifeSnaps cohort (n = 71), where LHFM exceeds the
logistic-regression and random-forest baselines on the same features
by approximately 20 AUROC points on high-stress prediction. A
companion replication on the larger GLOBEM cohort (n ≈ 700) is in
progress; the paper structure is intended to absorb that result as
it lands.

LHFM is not a medical device. We make no clinical claims. The
contribution of this work is methodological and infrastructural: a
reproducible scaffold against which other groups can fine-tune their
own cohorts without reimplementing the participant-clustered
evaluation, the fairness audit, the climate-regime holdout, or the
multimodal adapter abstraction.
