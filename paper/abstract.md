# Abstract

**Background.** Wearable, smartphone, and environmental signals collected
continuously across weeks and months hold considerable promise for
personalised behavioural-health monitoring. Realising that promise requires
representation-learning methods that can absorb high-dimensional,
multimodal, and heavily missing longitudinal streams while remaining
sensitive to within-person, day-to-day deviations rather than only
between-person differences. Existing foundation-model efforts in this space
either omit code and weights, blocking reproducibility, or are evaluated on
single cohorts with row-level rather than participant-level resampling,
producing optimistic uncertainty estimates.

**Methods.** We present the **Longitudinal Health Foundation Model (LHFM)**,
a self-supervised multimodal transformer that learns dense per-day
representations from wearable cardiovascular and sleep data, smartphone
passive-sensing features, and external climate exposure. The encoder is
pretrained with masked feature reconstruction, next-day state prediction,
and a participant-trajectory contrastive loss, then frozen for four
downstream binary risk heads (low-mood, high-stress, sleep-disruption, and
climate-vulnerability prediction). A cohort-agnostic adapter abstraction
ingests new datasets without changes to the model or training code; the
evaluation pipeline reports participant-clustered bootstrap confidence
intervals (1000 iterations) and benchmarks against logistic-regression and
random-forest baselines on identical engineered features. The full pipeline
is released as a reproducible research prototype with synthetic data
covering 250 simulated participants over 90 days, and applied end-to-end to
the LifeSnaps cohort (Yfantidou et al. 2022; n = 71).

**Results.** On synthetic data, LHFM recovers the seeded causal structure
across all four downstream tasks and surfaces clinically plausible feature
attributions via integrated gradients. On the LifeSnaps cohort, LHFM
achieves a high-stress AUROC of 0.567 (95 % CI [0.389, 0.688]) on
held-out test data (11 participants, 294 windows, 55 positives), compared
to logistic-regression (0.328) and random-forest (0.368) baselines on the
identical feature set — a roughly 20-AUROC-point lift attributable to the
self-supervised representation rather than the feature engineering. On the
sleep-disruption task on the same cohort, classical baselines outperform
LHFM (random forest 0.641 vs LHFM 0.518), an honest finding that we report
rather than tune away: SSL does not buy lift universally. Replication on
the larger GLOBEM cohort (Xu et al. 2022; n ≈ 700) is in progress and
will be reported in a forthcoming update.

**Conclusion.** LHFM is not a medical device. It is a reproducible
methodological scaffold for investigating how self-supervised foundation
models can be deployed safely on passive-sensing data in computational
psychiatry and climate-health research. Code, synthetic data, the LifeSnaps
adapter, and the held-out evaluation recipe are publicly available at
`https://github.com/ceyhunolcan/longitudinal-health-foundation-model`.
