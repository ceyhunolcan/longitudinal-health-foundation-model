# Data card — LHFM synthetic cohort v0.2

## Why synthetic?

Real wearable + smartphone + EMA datasets that include the climate
covariates we want are scarce, often under restricted-access agreements,
and sometimes ethically fraught to redistribute. To make the LHFM pipeline
fully reproducible, this release uses **only synthetic data** generated
deterministically by ``src/lhfm/data/synthetic_generator.py``.

No HIPAA-regulated information, no identifiable participants, no real
sensor traces, and no real EMA responses are involved.

## Generator summary

- 250 simulated participants × 90 days = 22,500 person-days by default
- 4 input modalities (wearable, smartphone, EMA, environmental)
- Deterministic given the seed; default seed 42
- v0.2 adds clinical context (medications, comorbidities, hormonal
  cycles, device noise), additional climate regimes (cold snaps,
  wildfire smoke), and subgroup-stratification metadata. Schema is
  backward-compatible: the original 27 columns retain their names and
  meanings; new columns are appended.

## Latent participant priors

Each simulated participant is sampled from these distributions:

| field | distribution |
|---|---|
| age | N(34, 11), clipped to [18, 72] |
| sex | uniform {F, M} |
| chronotype | categorical (0.30, 0.50, 0.20) over morning / intermediate / evening |
| baseline_sleep_need | N(7.8, 0.6), clipped to [6.0, 9.5] (+0.4 if sleep disorder) |
| baseline_hrv (RMSSD-style) | N(55 − 0.35(age−30), 12), clipped to [18, 130]; +12ms β-blocker, −4ms SSRI, −8ms cardio cond. |
| baseline_rhr | N(64 + 0.10(age−30), 7), clipped to [45, 95]; −10bpm β-blocker, +5bpm cardio cond. |
| mood_setpoint (1-7) | N(5.0, 0.9), clipped to [2.0, 6.8] (−1.2 depression, −0.3 anxiety) |
| stress_setpoint (1-7) | N(3.2, 0.9), clipped to [1.0, 6.0] (+1.1 anxiety, +0.4 depression) |
| heat_sensitivity | N(1.0, 0.3), clipped to [0.3, 2.0] (×1.3 in cold-region residents) |
| aqi_sensitivity | N(1.0, 0.3), clipped to [0.3, 2.0] |
| missingness_propensity | N(1.0, 0.35), clipped to [0.3, 3.0] (×1.2-1.6 if depressed) |

### Comorbidities (independent draws)

| condition | prevalence |
|---|---|
| anxiety | 18% |
| depression | 10% |
| sleep disorder | 8% |
| cardiovascular condition | 6%, scaled up with age |

### Medications (conditional on comorbidity)

| medication | base prevalence | conditional bump |
|---|---|---|
| SSRI | 9% | ×5 with depression, ×1.6 with anxiety |
| β-blocker | 5% | ×6 with cardiovascular condition |
| sleep aid | 7% | ×3.5 with sleep disorder |

Each is capped (0.6 SSRI/BB, 0.4 sleep aid) so no group is unrealistically
saturated.

### Hormonal cycles

About 70% of female participants of reproductive age (16-51) carry a
synthetic 28-day cycle. The cycle modulates HRV (late-luteal drop of
~4.5 ms; ovulation peak of ~1.5 ms) and mood (mild luteal dip of
~0.25 points). Cycle phase is emitted as a categorical column
(``cycle_phase``: none/menses/follicular/ovulation/luteal).

### Device generation

30% of participants are assigned to an "older device" with a 6 ms HRV
measurement noise floor (vs. 1.5 ms for newer devices). This is the
canonical bottleneck that real wearable-validation studies report for
population-level HRV claims.

### Subgroup metadata (descriptive only)

| axis | levels | distribution |
|---|---|---|
| race_ethnicity | white / black / hispanic / asian / other | 0.59 / 0.13 / 0.18 / 0.06 / 0.04 |
| ses_proxy | low / middle / high | 0.30 / 0.50 / 0.20 |
| region | temperate / hot / cold | 0.55 / 0.25 / 0.20 |
| device_gen | new / old | 0.70 / 0.30 |

**These are independent draws.** The generator does not bake in
subgroup-specific outcome disparities. The columns exist so the
fairness audit module (``src/lhfm/utils/fairness.py``) has axes to
stratify on -- a sanity-check that the methodology pipeline works,
not a finding about disparities.

## Causal assumptions baked into the simulator

The simulator is deliberately *not* a noise generator. It instantiates a
small causal graph that mirrors qualitative findings from the digital-health
literature:

```
chronotype, baseline_sleep_need ----> sleep duration / efficiency
heat_index, humidity, cold_temp ----> sleep efficiency (penalty)
sleep_dev, climate_stress, prior_stress ----> stress_today
stress, sleep_dev, heat_index, cold ----> HRV, resting HR
cycle_phase, beta-blocker, SSRI ----> HRV, mood floor
stress, sleep, sleep_debt_3d, AQI, heat_index, cycle ----> next-day mood
prior_mood, mood -------------------> mobility radius, location entropy, screen time
low mood / low energy --------------> elevated probability of missing wearable + survey
```

The mood update intentionally uses both yesterday's sleep deviation
*and* a 3-day rolling sleep debt term. This was tuned in v0.2 to land
within-person mood-sleep correlation in the 0.25-0.35 range observed in
real EMA studies (the v0.1 generator produced ~0.05, which made the
next-day-mood task nearly impossible from passive sensing alone).

These are *qualitative* effect-direction claims, not clinically calibrated
effect sizes. The point is to produce a dataset that is non-trivially
learnable, not one that is faithful to any real cohort.

## Missingness model

- Base missingness rates: 12% wearable, 5% smartphone, 25% EMA survey.
- Multiplied by a per-person missingness propensity.
- Multiplied by ``(1 + mood_factor)`` where ``mood_factor = max(0, 4.5 - mood) / 4``,
  so low-mood days are more likely to be missing (informative missingness).
- Wearable also boosted on weekends (×1.4) to reflect the "left it on the
  charger" effect.
- Depressed participants have their base propensity scaled by ×1.2-1.6.

## Climate model

- Seasonal sinusoid with amplitude 6 °C around a 22 °C baseline.
- Two heat-wave events (days 25-31 and 60-64), each adding 3-5.5 °C.
- One cold-snap event (days 45-49), subtracting 4-8 °C.
- Three pollution episodes adding ~50 to AQI.
- Two wildfire-smoke spikes (days 35-39 and 70-73), reaching AQI 220-380
  for 4-day windows.
- Heat index computed via the standard NWS Rothfusz polynomial above ~27 °C.

## Validation

Every generated cohort is checked by ``validate_synthetic_dataframe`` for:

- column presence
- per-column plausibility ranges
- per-participant timeline length
- consistency between missingness flags and NaN values

## What this dataset **cannot** tell us

- Anything about real population effect sizes.
- **Anything about subgroup disparities.** The priors are independent of
  race, ethnicity, socioeconomic status, and geography, so by design no
  real-world disparities are present. A fairness audit on this data is
  exclusively a *pipeline* check, not a finding.
- Anything about deployment robustness or sensor drift beyond the simple
  device-generation noise model.
- Anything about how real people actually respond to climate extremes.

Anyone wishing to extend LHFM to real data must run an independent IRB
review, obtain informed consent including consent for secondary analysis,
and conduct an external fairness audit on representative subgroups.
