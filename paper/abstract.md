# Abstract

Wearable, smartphone, and environmental signals collected continuously across
weeks and months hold considerable promise for personalized behavioral-health
monitoring. Realising that promise requires representation-learning methods
that can absorb high-dimensional, multimodal, and heavily missing longitudinal
streams while remaining sensitive to within-person, day-to-day deviations
rather than only between-person differences. We present the
**Longitudinal Health Foundation Model (LHFM)**, a self-supervised
multimodal transformer that learns dense per-day representations from
wearable cardiovascular and sleep data, smartphone passive-sensing features,
and external climate exposure. The encoder is pretrained with three
complementary objectives: masked feature reconstruction, next-day state
prediction, and a participant-trajectory contrastive loss. A small set of
binary risk heads then evaluates the learned representations on four
clinically motivated downstream proxies: next-day low-mood, high-stress,
sleep-disruption, and climate-vulnerability prediction.

The full pipeline is released as a reproducible research prototype with
synthetic, statistically realistic data covering 250 simulated participants
across 90 days. Synthetic data was generated under causal assumptions
consistent with the digital-health literature (sleep debt and heat exposure
degrade autonomic markers; informative missingness covaries with mood). The
prototype includes feature-engineering modules for wearable, smartphone,
climate, missingness-pattern, and personal-baseline features; FastAPI and
Streamlit interfaces for inference and exploration; and classical baselines
(logistic regression, random forest, gradient-boosted trees) for context.
LHFM is not a medical device; it is a methodological scaffold for
investigating how self-supervised foundation models can be deployed safely
and meaningfully on passive-sensing data in computational psychiatry and
climate-health research.
