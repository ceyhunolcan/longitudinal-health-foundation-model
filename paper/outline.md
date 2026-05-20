# Paper outline — Longitudinal Health Foundation Model

Working title: **A reproducible foundation model for longitudinal
behavioural health: multimodal self-supervised pretraining with
participant-clustered evaluation on wearable and smartphone cohorts.**

Target venue: npj Digital Medicine (Nature Portfolio).
Target length: ~5000 words + figures.
Authors: Ceyhun Olcan (Dartmouth College). <<TODO: confirm advisor inclusion>>

## Status legend

- ✅ done — text is drafted in the corresponding file
- 🚧 partial — scaffolded, awaits GLOBEM numbers or a final figure
- ❌ TODO — not yet started

## Section spine

### 1. Abstract — `paper/abstract.md`  🚧
Single-paragraph, structured-style (Background / Methods / Results /
Conclusion). Currently reads as synthetic-only; needs to be rewritten
to lead with the LifeSnaps baseline result and scaffold GLOBEM as
"replication on a larger cohort, n ≈ 700, reported below."

### 2. Introduction — `paper/introduction.md`  ❌
Three-paragraph build:
1. The clinical opportunity: continuous passive sensing for behavioural
   health (mood, stress, sleep) at scale, with citations to LifeSnaps,
   GLOBEM, the BiAffect project, and the Apple/Google heart-rate
   foundation models that appeared in 2024.
2. The technical gap: existing approaches either (a) are single-cohort,
   single-modality feature pipelines that don't transfer, or (b) are
   foundation-model-scale efforts that withhold code, weights, and
   evaluation protocols, blocking reproducibility.
3. Our contribution: an open, reproducible foundation-model pipeline
   that (i) handles real-world missingness and modality heterogeneity
   via an adapter abstraction, (ii) provides participant-clustered
   bootstrap evaluation by default, (iii) includes a fairness audit and
   climate-regime generalisation eval, and (iv) ships with end-to-end
   results on a published cohort (LifeSnaps), with GLOBEM replication
   in progress.

### 3. Methods — `paper/methods.md`  ✅ (existing) + 🚧 (additions)

The existing `methods.md` is solid for the model and synthetic-cohort
sections. Needs three additions:
- §1.2 Real-data cohorts: LifeSnaps adapter (RAIS release), GLOBEM
  adapter (PhysioNet, in progress), generic adapter abstraction.
- §3.4 Participant-clustered bootstrap: state the CI procedure
  explicitly (1000 iterations, resample participants, not rows).
- §3.5 Classical baselines: logreg + RF on the same engineered
  features, fit with identical train/test splits.

### 4. Results — split into two files

#### 4.1 Synthetic cohort — `paper/results_synthetic.md`  ❌
- Per-task AUROC + bootstrap CIs on the 60p × 90d synthetic cohort.
- Demonstrates: pipeline works, the model learns the seeded causal
  structure (sleep debt → HRV, heat → mood, etc.), and behaves
  sensibly under masked-modality ablation.
- Plus: fairness audit numbers (subgroup AUROC), climate-regime
  holdout (heat-wave AUROC), interpretability (integrated-gradients
  bar chart from `notebooks/00_quickstart.ipynb`).

#### 4.2 LifeSnaps replication — `paper/results_lifesnaps.md`  ❌
- Direct port of `docs/lifesnaps_results.md` into paper voice.
- Headline: high_stress AUROC 0.567 [0.389, 0.688] vs logreg 0.328 vs
  RF 0.368 — ~20-point lift over classical baselines.
- Honest finding on sleep_disruption (classical baselines win).
- Documented limitations on this cohort: no phone, no climate,
  n=11 test participants.

#### 4.3 GLOBEM replication — `paper/results_globem.md`  ❌ (scaffold)
- All numbers <<TODO: pending PhysioNet credentialing>>.
- Section structure ready: cohort table, per-task AUROC, baseline
  comparison, climate-regime holdout. Drafted as a template so we
  only need to fill in numbers when credentialing approves.

### 5. Discussion — `paper/discussion.md`  ❌
Four paragraphs:
1. What worked: SSL representation transfers a ~20 AUROC-point lift
   over classical baselines on the one high-positive-rate task we
   could evaluate (high_stress on LifeSnaps).
2. What didn't: sleep_disruption is better handled by RF on this
   cohort. The SSL prior is not universally helpful.
3. Why we trust the result: participant-clustered bootstrap, public
   adapter committed before evaluation, no per-row predictions
   committed (DUA-compliant).
4. What this enables: an open scaffold that other groups can fine-tune
   on their own cohorts without rebuilding the eval/fairness/climate
   pipelines. Specifically calls out that GLOBEM is in progress and
   that the same adapter abstraction will absorb it once credentialing
   approves.

### 6. Limitations — `paper/limitations.md`  ✅ (existing) + 🚧
Existing file is good. Add: LifeSnaps-specific limitations from
`docs/lifesnaps_results.md` (no phone, no climate, sparse EMA,
unevaluable low_mood/climate_vulnerable on this cohort). Add the
"cohort size, not model design, is the binding constraint" framing.

### 7. Ethics — `paper/ethics.md`  ✅
Existing file is solid. No edits needed.

### 8. Data and code availability — new section, can live in `methods.md`
- Code: `https://github.com/ceyhunolcan/longitudinal-health-foundation-model`
- License: MIT for code, CC-BY-4.0 for synthetic data
- LifeSnaps: Kaggle / Zenodo (cite Yfantidou 2022)
- GLOBEM: PhysioNet (cite Xu 2022); credentialing required
- No per-participant data redistributed (LifeSnaps DUA)
- Reproducibility recipe: `docs/lifesnaps_results.md`

### 9. Author contributions
<<TODO: solo authorship for now; add collaborators if recruited>>

### 10. Competing interests
None. (Confirm: not employed by any wearable manufacturer or
digital-health company commercialising this kind of model.)

### 11. References — `paper/references.bib`  ❌
Key citations to include:
- Yfantidou et al. 2022 (LifeSnaps)
- Xu et al. 2022 (GLOBEM, NeurIPS Datasets track)
- Saeed et al. 2023 (multimodal SSL for wearables)
- Apple heart-rate foundation model 2024 <<TODO: confirm exact ref>>
- Google PHM 2024 <<TODO: confirm exact ref>>
- Sundararajan, Taly, Yan 2017 (integrated gradients)
- ICMJE authorship guidelines (for the ethics note about AI tools)
- Twenge et al. 2018 (screen time + mood, used in interpretability)
- Lin et al. 2016 (smartphone use + depression)
- Doerr et al. 2018 (heat exposure + HRV)

## Figure plan

Total: 4 figures, 3 tables. npj DM has no hard figure limit but
appreciates restraint.

| # | Figure | Source | Status |
| --- | --- | --- | --- |
| F1 | Architecture diagram | `docs/figures/architecture.mmd` (Mermaid) | ✅ |
| F2 | Synthetic risk-driver attribution bar chart | Colab cell 11 | ✅ |
| F3 | LifeSnaps calibration plot for high_stress | `results/figures/calibration_high_stress.png` | ✅ |
| F4 | GLOBEM AUROC by climate regime | <<TODO: pending credentialing>> | ❌ |

| # | Table | Source | Status |
| --- | --- | --- | --- |
| T1 | Cohort characteristics (synthetic / LifeSnaps / GLOBEM) | this file | 🚧 |
| T2 | Held-out test AUROC, all cohorts × all tasks | this file | 🚧 |
| T3 | Fairness audit (subgroup-stratified AUROC) | `make fairness-audit` | ✅ |

## Submission readiness check

Before submission we need:
- [ ] GLOBEM results filled in (or removed; depends on credentialing timeline)
- [ ] Lit review fully populated (we've done 60% above)
- [ ] At least one Dartmouth-affiliated co-author or supervisor sign-off
- [ ] Final figure polish (consistent font, colour palette, legend style)
- [ ] Word-count compliance (~5000 words main text)
- [ ] CITATION.cff updated to reflect "paper accompanying this code"
- [ ] Preprint submitted to medRxiv first (npj Digital Medicine allows it)

## Drafting order tonight

1. ✅ Outline (this file)
2. Update `abstract.md` — rewrite to include LifeSnaps result
3. Draft `introduction.md`
4. Augment `methods.md` with real-data cohort + bootstrap + baselines
5. Draft `results_lifesnaps.md`
6. Scaffold `results_globem.md`
7. Draft `discussion.md`
8. Stop and commit. Synthetic-results section and final polish are
   for a fresh-head session.
