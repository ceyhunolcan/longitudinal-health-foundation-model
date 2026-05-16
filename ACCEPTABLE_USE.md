# Acceptable Use Policy

LHFM is released as a **research prototype** under the MIT license. Permissive
licensing is not a license to do whatever you want with passive-sensing
behavioural data. This document spells out what we — the authors and
maintainers — believe is and isn't an acceptable use of this codebase,
and what we will refuse to support.

This is a **soft norm**, not a legal restriction; the MIT license still
governs distribution. It exists because (a) we'd like contributors to know
the project's intended posture before they invest time in it, and (b) we'd
like third parties evaluating LHFM-derived products to be able to point at
something concrete when the conversation turns to misuse.

## Out of scope: do not use LHFM for any of the following

We will not collaborate on, write letters of support for, or merge code that
materially advances any of these uses:

- **Clinical decision-making of any kind.** LHFM is not a medical device.
  Outputs are not diagnostic. The four downstream tasks are rule-based
  weak labels on EMA scales, not validated clinical instruments. See
  `paper/model_card.md`.
- **Surveillance of employees, students, prisoners, or anyone else whose
  consent is structurally constrained.** Passive sensing of mood and
  stress in a workplace, school, immigration, or carceral context is a
  paradigmatic dual-use harm.
- **Insurance underwriting, employment screening, credit decisions, or
  other allocation decisions** that involve LHFM outputs.
- **Targeting of minors.** LHFM has been built and (synthetically) evaluated
  on adult cohorts. Any application directed at people under 18 requires
  a separate ethics review, child-assent protocol, and parental consent
  process beyond standard IRB practice. Even with all of those in place
  we expect strong methodological caution and decline to advise on
  surveillance-style monitoring of minors.
- **Intimate-partner monitoring.** Tools that let one adult monitor another
  adult's mood, location, sleep, or behavioural patterns without their
  ongoing, informed, revocable consent are an avenue for abuse. We will
  not contribute to such tools.
- **Re-identification or de-anonymization of any cohort**, including
  attempts to invert the model's representations to recover identifiable
  participant features.
- **Generative misuse**: producing synthetic time-series intended to
  impersonate a specific real person's behaviour for fraud, harassment,
  or manipulation.

## Use cases we actively support

- Methodological research on representation learning for longitudinal
  multimodal passive-sensing data.
- Teaching scaffolds for graduate courses on digital health, computational
  psychiatry, and climate-and-health.
- Real-data extensions conducted under IRB review with informed consent
  that explicitly covers (a) passive sensing, (b) secondary analysis,
  (c) the retention window, and (d) the participant's right to deletion.
- Fairness audits, calibration studies, and external validation against
  the methodological claims we make.
- Critical replication and adversarial probing. If LHFM is wrong about
  something we want to know.

## What we ask of redistributors

If you fork, extend, or re-publish a model trained from this codebase, we
ask (but do not legally require) that you:

1. **Preserve the disclaimer** "Research prototype. Not a medical device."
   on every user-facing surface (API responses, dashboards, downloads).
2. **Attach a model card** documenting the cohort the model was trained on,
   subgroup composition, calibration on held-out data, and known failure
   modes.
3. **Attach a data card** documenting consent, retention, deletion rights,
   and the specific scope of secondary analysis the consent covers.
4. **Do not strip the ethics and limitations documentation** from the
   repository. Updating them is welcome; removing them is not.
5. **Add yourself to `CITATION.cff`** as a fork maintainer and update the
   repository URL.

## Reporting concerns

If you become aware of a use of LHFM or its derivatives that falls into the
"out of scope" list above, please open an issue tagged `acceptable-use` or
contact the maintainers directly. We can't enforce any of this
unilaterally, but we can refuse to provide support, deny collaboration
requests, and publicly distance the project from misuse.

## A note to clinicians and digital-health teams considering LHFM

If you're reading this because someone proposed using LHFM in a clinical
workflow, the answer is no. Not "no for now, until we run a few more
experiments". No. The codebase is not what stops it — the absence of the
following is what stops it:

- prospective validation on a real cohort,
- a subgroup-stratified fairness audit on the actual deployment population,
- a faithful interpretability method (the current "explanation" is rule-
  based, not model-derived),
- a calibration study on the actual deployment population,
- regulatory clearance appropriate to the intended use,
- a human-factors study of how clinicians actually use the outputs.

Any one of those is a multi-year program of work. Several of them require
real-world consent regimes that this synthetic-only repository does not
even attempt to model.
