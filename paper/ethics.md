# Ethics statement

LHFM is a research prototype operating only on synthetic data. Even so,
the technical capability it demonstrates - dense, longitudinal modelling
of intimate behavioural and physiological signals - raises ethical
questions worth confronting before any real-data extension.

## Privacy and surveillance

Wearable and smartphone data is uniquely intimate. Even when stripped of
direct identifiers, a sufficiently long passive-sensing trace can be
re-identifying through gait, location pattern, or sleep schedule. Anyone
extending LHFM to real participants should consider:

- **Data minimization.** Collect only the streams the research question
  actually requires; storing more "in case" is a security debt.
- **On-device computation** where possible, rather than centralized
  aggregation.
- **Differential privacy** for any cohort-level statistics released.
- **Clear retention windows.** Data should age out.
- **Meaningful consent.** Long-form consent that explains *what passive
  sensing implies for a participant's daily life* matters more than
  legalese.

## Non-diagnostic status

The four downstream tasks (low mood, high stress, sleep disruption,
climate vulnerability) are coarse, EMA-derived **proxies**. They are not
diagnoses. They must not be presented to clinicians, employers, schools,
or insurers as evidence of a participant's mental or physical state.

The API response and the dashboard both carry an explicit disclaimer for
this reason.

## Algorithmic bias

The v0.2 synthetic generator attaches stratification metadata
(race/ethnicity, region, SES proxy, device generation, depression
and anxiety flags, age band) to each simulated participant so that
audit tooling can be developed and exercised against the synthetic
pipeline. However, these labels are not guaranteed independent of
the generator's seeded outcome distributions: for example, anxious
participants carry a higher stress setpoint by construction. Any
subgroup disparity observed on this cohort therefore conflates the
encoder's learned behaviour with the generator's priors, and
should not be interpreted as a real-world fairness signal. Anyone
moving to real data must:

- include subgroup metadata,
- evaluate per-subgroup AUROC / AUPRC / calibration,
- compare false-positive and false-negative rates across subgroups,
- and treat any disparity larger than a few percentage points as a model
  defect worth fixing or surfacing prominently.

## Climate-health framing

Climate-health is a load-bearing motivation for this project, but it is
also a topic where modeling decisions interact with health equity. Heat
vulnerability is not evenly distributed: it correlates strongly with
housing quality, air conditioning access, neighbourhood tree cover, and
occupational exposure - none of which are in the LHFM feature set. A
naive "climate-vulnerable day" prediction trained on real data could
end up encoding socioeconomic disadvantage as physiology. The current
model card states explicitly that LHFM is not appropriate for any
allocation or screening decision in this domain.

## Dual-use concerns

Dense behavioural monitoring is a dual-use capability. Plausible misuses
include workplace surveillance, school-based emotional monitoring of
minors, intimate-partner control, and immigration / carceral screening.
Researchers downstream of this codebase should adopt a clear, public
acceptable-use policy and decline collaborations that conflict with it.

## Children

The synthetic cohort includes virtual participants from age 18 upward.
LHFM has never been evaluated on minors, and the authors do not endorse
any extension that targets minors without a separate ethics review, child
assent protocol, and parental consent process that goes beyond standard
IRB practice.

## Mental-health context

If you are a researcher reading this who is in distress or considering
self-harm: please reach out to a trusted person or local crisis resource.
This project's outputs are not a substitute for human support.
