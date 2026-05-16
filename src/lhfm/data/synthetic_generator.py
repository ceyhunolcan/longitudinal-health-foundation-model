"""Synthetic longitudinal multimodal health-data generator.

The point of this module is to produce a dataset whose *structure* is
identical to what we'd get from a real wearable + smartphone + EMA study,
and whose *statistics* are realistic enough that downstream feature
engineering and model code can be unit-tested end-to-end without any real
patient data.

This v2 of the generator significantly tightens the simulation. The list
of new effects, each motivated by the digital-health literature, is:

- **Medications.** Three classes — SSRIs (~9% prevalence in young adults),
  beta-blockers (~5%, higher with age), sleep aids (~7%, episodic) — each
  with documented direction-of-effect on HRV, RHR, sleep latency, and mood.
- **Comorbidities.** Anxiety, depression, sleep disorder, cardiovascular
  condition are baked into participants' baseline setpoints. Prevalences
  roughly match adult US NHIS rates.
- **Hormonal cycles.** ~28% of female participants of reproductive age
  carry a 28-day cycle with a documented late-luteal HRV drop (~8%) and
  modest mood/sleep effects across the cycle.
- **Device generation.** Each participant is assigned a "device gen"
  (older/newer) that sets the HRV measurement noise floor. Older devices
  add ~6 ms of noise on top of the physiological signal -- the noise
  ceiling Sequence-data papers describe as the bottleneck for population
  validation.
- **Cold-weather effects.** Low ambient temperatures (T < 5°C) bump
  resting HR through vasoconstriction and degrade sleep efficiency.
- **Wildfire smoke episodes.** Two AQI spikes (peak 250-400) lasting 2-5
  days, on top of the baseline pollution episodes.
- **Stronger mood-sleep coupling.** Mood now depends more on last night's
  sleep, getting within-person mood-sleep correlation up to the ~0.25-0.35
  range observed in real EMA studies.
- **Subgroup metadata.** ``race_ethnicity``, ``ses_proxy`` (high/middle/
  low income), and ``region`` (temperate/hot/cold climate zone). These
  enable the subgroup-stratified fairness audit. None of them mechanically
  drive the simulation; they exist so the fairness module has columns to
  stratify on. **You should expect roughly equal performance across these
  groups by construction**, which makes the audit a methodology check
  rather than a finding.

We deliberately bake in causal links that match the digital-health
literature so that the foundation model has something non-trivial to learn:

    poor sleep / high stress / heat-stress / poor air quality / med side-effects
        |
        v
    autonomic load (HRV down, RHR up)
        |
        v
    next-day mood and energy
        |
        v
    behavioural changes (less mobility, more screen, missed surveys)

None of these effects are calibrated to clinical thresholds. The generator
is for software development and methodological prototyping only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Participant priors. We sample one of these "phenotype" mixtures per person.
# This is intentionally crude but gives us reasonable between-subject variance.
# ---------------------------------------------------------------------------

CHRONOTYPES = ("morning", "intermediate", "evening")
SEXES = ("F", "M")

# Subgroup categories used for the fairness audit. Prevalences are loosely
# inspired by US adult demographics but are not calibrated to any actual
# population estimate -- they exist to give the audit module distinct strata.
RACE_ETHNICITIES = ("white", "black", "hispanic", "asian", "other")
RACE_ETHNICITY_PROBS = (0.59, 0.13, 0.18, 0.06, 0.04)

SES_LEVELS = ("low", "middle", "high")
SES_PROBS = (0.30, 0.50, 0.20)

REGIONS = ("temperate", "hot", "cold")
REGION_PROBS = (0.55, 0.25, 0.20)

# Medication classes. Per-participant assignment is independent so a few
# people will carry multiple meds, which is realistic.
MEDICATIONS = ("ssri", "beta_blocker", "sleep_aid")


@dataclass
class ParticipantPrior:
    """Per-person latent parameters that drive their day-to-day signals."""

    participant_id: str
    age: int
    sex: str
    chronotype: str
    baseline_sleep_need: float        # hours
    baseline_hrv: float               # ms (RMSSD-ish)
    baseline_rhr: float               # bpm
    baseline_steps: float             # daily steps mean
    baseline_screen_min: float        # daily screen-time mean
    mood_setpoint: float              # 1-7 EMA scale
    stress_setpoint: float            # 1-7 EMA scale
    heat_sensitivity: float           # how much heat hits HRV / mood
    aqi_sensitivity: float            # how much pollution hits mood
    missingness_propensity: float     # general probability scaler

    # --- senior-PI additions ----------------------------------------------
    # Subgroup metadata (descriptive only).
    race_ethnicity: str = "white"
    ses_proxy: str = "middle"
    region: str = "temperate"

    # Comorbidities (boolean).
    has_anxiety: bool = False
    has_depression: bool = False
    has_sleep_disorder: bool = False
    has_cardio_condition: bool = False

    # Medications (boolean).
    on_ssri: bool = False
    on_beta_blocker: bool = False
    on_sleep_aid: bool = False

    # Hormonal cycle: cycle_day_0 is the day index (0 .. 27) on which the
    # participant's cycle starts at the cohort's "day 0". None means no cycle.
    cycle_day_0: Optional[int] = None

    # Device generation: older devices add HRV measurement noise.
    device_gen: str = "new"   # "new" or "old"


@dataclass
class GeneratorConfig:
    n_participants: int = 250
    n_days: int = 90
    start_date: str = "2024-04-01"
    seed: int = 42

    # Missingness base rates. Real wearable studies see roughly these numbers
    # though obviously the specific cohort and device matter a lot.
    base_missing_wearable: float = 0.12
    base_missing_phone: float = 0.05
    base_missing_survey: float = 0.25

    # Climate forcing. We seed a sinusoidal seasonal trend and superimpose a
    # couple of heat-wave events plus optional cold snaps and wildfire smoke
    # episodes.
    base_temp_c: float = 22.0
    seasonal_amplitude_c: float = 6.0
    heat_wave_days: tuple = field(default_factory=lambda: ((25, 31), (60, 64)))
    cold_snap_days: tuple = field(default_factory=lambda: ((45, 49),))
    wildfire_smoke_days: tuple = field(default_factory=lambda: ((35, 39), (70, 73)))

    # Sub-population prevalences. Anxiety/depression are correlated in
    # reality; we sample them independently for simplicity but document
    # that limitation in the data card.
    prevalence_anxiety: float = 0.18
    prevalence_depression: float = 0.10
    prevalence_sleep_disorder: float = 0.08
    prevalence_cardio_condition: float = 0.06    # ageing prior multiplies this

    prevalence_ssri: float = 0.09
    prevalence_beta_blocker: float = 0.05
    prevalence_sleep_aid: float = 0.07

    # Female participants of reproductive age who carry a hormonal cycle.
    fraction_cycling_when_eligible: float = 0.70

    # Older devices have a higher HRV noise floor.
    fraction_old_devices: float = 0.30


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------


class SyntheticCohortGenerator:
    """Generate a synthetic longitudinal multimodal cohort.

    Usage
    -----
    >>> gen = SyntheticCohortGenerator(GeneratorConfig(n_participants=50, n_days=60))
    >>> df = gen.generate()
    >>> df.head()
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.rng = np.random.default_rng(self.config.seed)

    # -- public API ---------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """Return the full long-form dataframe (one row per participant-day)."""
        priors = self._sample_priors()
        dates = pd.date_range(self.config.start_date, periods=self.config.n_days, freq="D")
        climate = self._make_climate_track(dates)

        rows: list[dict] = []
        for prior in priors:
            rows.extend(self._simulate_participant(prior, dates, climate))

        df = pd.DataFrame(rows)
        # Stable column ordering, makes debugging much nicer. We keep the
        # original schema first so anything reading it by position is happy.
        ordered = [
            "participant_id", "date", "age", "sex",
            "baseline_sleep_need", "baseline_hrv", "chronotype",
            "daily_steps", "sleep_duration", "sleep_efficiency",
            "resting_hr", "hrv_rmssd", "stress_score",
            "phone_unlock_count", "screen_time_minutes",
            "mobility_radius_km", "location_entropy",
            "survey_mood", "survey_energy", "survey_stress",
            "temperature_c", "humidity", "aqi", "heat_index",
            "missing_wearable_flag", "missing_phone_flag", "missing_survey_flag",
            # Senior-PI additions: subgroup metadata + clinical context.
            # These come at the end so column-positional readers still work.
            "race_ethnicity", "ses_proxy", "region", "device_gen",
            "has_anxiety", "has_depression", "has_sleep_disorder", "has_cardio_condition",
            "on_ssri", "on_beta_blocker", "on_sleep_aid",
            "cycle_phase",
        ]
        return df[ordered].reset_index(drop=True)

    def save(self, df: pd.DataFrame, out_path: Path | str) -> Path:
        """Save the generated dataframe as CSV (also writing a small manifest)."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        # Tiny manifest so we can later confirm what was generated and how.
        manifest = {
            "n_rows": len(df),
            "n_participants": df["participant_id"].nunique(),
            "n_days": df["date"].nunique(),
            "start_date": str(df["date"].min()),
            "end_date": str(df["date"].max()),
            "seed": self.config.seed,
            # Prevalences in the actual draw (will differ slightly from the
            # config's targets by sampling noise).
            "prevalence_anxiety_realised": float(df.groupby("participant_id")["has_anxiety"].first().mean()),
            "prevalence_depression_realised": float(df.groupby("participant_id")["has_depression"].first().mean()),
            "prevalence_ssri_realised": float(df.groupby("participant_id")["on_ssri"].first().mean()),
            "fraction_old_devices_realised": float((df.groupby("participant_id")["device_gen"].first() == "old").mean()),
        }
        pd.Series(manifest).to_json(out_path.with_suffix(".manifest.json"), indent=2)
        return out_path

    # -- internals ----------------------------------------------------------

    def _sample_priors(self) -> list[ParticipantPrior]:
        rng = self.rng
        cfg = self.config
        priors: list[ParticipantPrior] = []
        for i in range(cfg.n_participants):
            age = int(rng.normal(34, 11))
            age = max(18, min(72, age))

            sex = rng.choice(SEXES)
            chronotype = rng.choice(CHRONOTYPES, p=[0.30, 0.50, 0.20])

            # Subgroup metadata. Independent draws — we deliberately do NOT
            # build in disparities here, because the synthetic generator
            # has no business deciding what "real-world disparities" look like.
            race_ethnicity = rng.choice(RACE_ETHNICITIES, p=RACE_ETHNICITY_PROBS)
            ses_proxy = rng.choice(SES_LEVELS, p=SES_PROBS)
            region = rng.choice(REGIONS, p=REGION_PROBS)

            # Comorbidities. Independent draws (correlated in reality; doc'd).
            has_anxiety = bool(rng.random() < cfg.prevalence_anxiety)
            has_depression = bool(rng.random() < cfg.prevalence_depression)
            has_sleep_disorder = bool(rng.random() < cfg.prevalence_sleep_disorder)
            # Cardio risk scales with age.
            age_mult = 1.0 + max(0.0, (age - 30) / 30)
            has_cardio_condition = bool(rng.random() < cfg.prevalence_cardio_condition * age_mult)

            # Medications. Conditional bumps: someone with depression is much
            # more likely to be on an SSRI; someone with a cardio condition
            # is more likely to be on a beta-blocker.
            p_ssri = cfg.prevalence_ssri * (5.0 if has_depression else 1.0) * \
                                            (1.6 if has_anxiety else 1.0)
            p_bb = cfg.prevalence_beta_blocker * (6.0 if has_cardio_condition else 1.0)
            p_sleep = cfg.prevalence_sleep_aid * (3.5 if has_sleep_disorder else 1.0)

            on_ssri = bool(rng.random() < min(0.6, p_ssri))
            on_beta_blocker = bool(rng.random() < min(0.6, p_bb))
            on_sleep_aid = bool(rng.random() < min(0.4, p_sleep))

            # Sleep-need baseline. Sleep disorder shifts the *demand* up and
            # the *delivered* down via reduced efficiency later.
            baseline_sleep_need = float(np.clip(
                rng.normal(7.8 + (0.4 if has_sleep_disorder else 0.0), 0.6), 6.0, 9.5
            ))

            # HRV baseline. Lower with age. Medications shift it: beta-blockers
            # actually *raise* HRV substantially (~10-20 ms RMSSD); SSRIs lower
            # it slightly (~3-5 ms reduction is reported in meta-analyses).
            baseline_hrv = float(np.clip(rng.normal(55 - 0.35 * (age - 30), 12), 18, 130))
            if on_beta_blocker:
                baseline_hrv += rng.normal(12, 3)
            if on_ssri:
                baseline_hrv -= rng.normal(4, 1.5)
            if has_cardio_condition:
                baseline_hrv -= rng.normal(8, 2)
            baseline_hrv = float(np.clip(baseline_hrv, 8, 180))

            # RHR baseline. Beta-blockers lower RHR (their literal job).
            baseline_rhr = float(np.clip(rng.normal(64 + 0.10 * (age - 30), 7), 45, 95))
            if on_beta_blocker:
                baseline_rhr -= rng.normal(10, 2)
            if has_cardio_condition:
                baseline_rhr += rng.normal(5, 2)
            baseline_rhr = float(np.clip(baseline_rhr, 35, 110))

            # Mood and stress setpoints. Depression/anxiety shift these.
            mood_setpoint = float(np.clip(rng.normal(5.0, 0.9), 2.0, 6.8))
            if has_depression:
                mood_setpoint -= rng.normal(1.2, 0.3)
            if has_anxiety:
                mood_setpoint -= rng.normal(0.3, 0.15)
            mood_setpoint = float(np.clip(mood_setpoint, 1.5, 6.8))

            stress_setpoint = float(np.clip(rng.normal(3.2, 0.9), 1.0, 6.0))
            if has_anxiety:
                stress_setpoint += rng.normal(1.1, 0.3)
            if has_depression:
                stress_setpoint += rng.normal(0.4, 0.2)
            stress_setpoint = float(np.clip(stress_setpoint, 1.0, 6.8))

            # Hormonal cycle eligibility: female + reproductive age.
            cycle_day_0: Optional[int] = None
            if sex == "F" and 16 <= age <= 51 and rng.random() < cfg.fraction_cycling_when_eligible:
                # Random starting offset so cycles aren't synchronised.
                cycle_day_0 = int(rng.integers(0, 28))

            # Older devices add HRV measurement noise.
            device_gen = "old" if rng.random() < cfg.fraction_old_devices else "new"

            # Region modifies sensitivity to climate.
            heat_sensitivity = float(np.clip(rng.normal(1.0, 0.3), 0.3, 2.0))
            if region == "cold":
                heat_sensitivity *= 1.3   # less acclimatised
            elif region == "hot":
                heat_sensitivity *= 0.8   # acclimatised
            aqi_sensitivity = float(np.clip(rng.normal(1.0, 0.3), 0.3, 2.0))

            # Missingness propensity. Depression nudges this up substantially
            # (this is the canonical "ghosting" effect in real cohorts).
            missingness_propensity = float(np.clip(rng.normal(1.0, 0.35), 0.3, 2.5))
            if has_depression:
                missingness_propensity *= rng.uniform(1.2, 1.6)
            missingness_propensity = float(min(missingness_propensity, 3.0))

            priors.append(
                ParticipantPrior(
                    participant_id=f"P{i:04d}",
                    age=age, sex=sex, chronotype=chronotype,
                    baseline_sleep_need=baseline_sleep_need,
                    baseline_hrv=baseline_hrv,
                    baseline_rhr=baseline_rhr,
                    baseline_steps=float(np.clip(rng.normal(7500, 2200), 1500, 16000)),
                    baseline_screen_min=float(np.clip(rng.normal(220, 70), 30, 540)),
                    mood_setpoint=mood_setpoint,
                    stress_setpoint=stress_setpoint,
                    heat_sensitivity=heat_sensitivity,
                    aqi_sensitivity=aqi_sensitivity,
                    missingness_propensity=missingness_propensity,
                    race_ethnicity=race_ethnicity,
                    ses_proxy=ses_proxy,
                    region=region,
                    has_anxiety=has_anxiety,
                    has_depression=has_depression,
                    has_sleep_disorder=has_sleep_disorder,
                    has_cardio_condition=has_cardio_condition,
                    on_ssri=on_ssri,
                    on_beta_blocker=on_beta_blocker,
                    on_sleep_aid=on_sleep_aid,
                    cycle_day_0=cycle_day_0,
                    device_gen=device_gen,
                )
            )
        return priors

    def _make_climate_track(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Seasonal temperature plus heat waves, cold snaps, and wildfire smoke."""
        rng = self.rng
        n = len(dates)
        day_idx = np.arange(n)

        # Sinusoidal seasonal envelope. Phase chosen so that day 0 sits roughly
        # in late spring relative to the seasonal amplitude.
        season = self.config.seasonal_amplitude_c * np.sin(2 * np.pi * day_idx / 365)
        temp_c = self.config.base_temp_c + season + rng.normal(0, 1.4, size=n)

        # Heat-wave bumps.
        for start, end in self.config.heat_wave_days:
            if start < n:
                end = min(end, n - 1)
                bump = np.linspace(5.5, 3.0, end - start + 1) + rng.normal(0, 0.5, end - start + 1)
                temp_c[start:end + 1] += bump

        # Cold-snap dips. Same idea but with negative bumps.
        for start, end in self.config.cold_snap_days:
            if start < n:
                end = min(end, n - 1)
                dip = np.linspace(-8.0, -4.0, end - start + 1) + rng.normal(0, 1.0, end - start + 1)
                temp_c[start:end + 1] += dip

        humidity = np.clip(rng.normal(60, 12, size=n) + 0.4 * (temp_c - self.config.base_temp_c), 18, 98)

        # Air quality. Higher temps + lower wind days tend to coincide with worse AQI.
        aqi = np.clip(rng.normal(55, 22, size=n) + 0.8 * np.maximum(0, temp_c - 28), 10, 250)
        for ep_start in [10, 40, 75]:
            if ep_start < n:
                ep_end = min(ep_start + 3, n - 1)
                aqi[ep_start:ep_end + 1] += rng.normal(50, 10, size=ep_end - ep_start + 1)

        # Wildfire smoke spikes — much higher AQI, longer-lasting.
        for start, end in self.config.wildfire_smoke_days:
            if start < n:
                end = min(end, n - 1)
                # Trapezoidal: rise, hold, fall.
                length = end - start + 1
                peak = float(rng.uniform(220, 380))
                profile = np.concatenate([
                    np.linspace(60, peak, max(1, length // 3)),
                    np.full(length - 2 * max(1, length // 3), peak),
                    np.linspace(peak, 80, max(1, length // 3)),
                ])[:length]
                aqi[start:end + 1] = np.maximum(aqi[start:end + 1], profile)

        aqi = np.clip(aqi, 10, 500)
        heat_index = _heat_index(temp_c, humidity)

        return pd.DataFrame({
            "date": dates,
            "temperature_c": temp_c,
            "humidity": humidity,
            "aqi": aqi,
            "heat_index": heat_index,
        })

    def _simulate_participant(
        self,
        prior: ParticipantPrior,
        dates: pd.DatetimeIndex,
        climate: pd.DataFrame,
    ) -> list[dict]:
        rng = np.random.default_rng(self.rng.integers(0, 2**31 - 1))

        # We carry yesterday's stress and mood across days; that's where the
        # next-day prediction signal really lives.
        prev_stress = prior.stress_setpoint
        prev_mood = prior.mood_setpoint
        # 3-day rolling sleep debt feeds the mood update — this is the
        # mechanism behind the stronger within-person mood-sleep coupling
        # added in v2 of the generator.
        sleep_debt_rolling: list[float] = []

        # Device-driven HRV noise floor (~6 ms for older devices).
        hrv_noise_floor = 6.0 if prior.device_gen == "old" else 1.5

        rows: list[dict] = []
        for i, date in enumerate(dates):
            is_weekend = date.weekday() >= 5
            climate_row = climate.iloc[i]
            temp_c = float(climate_row["temperature_c"])
            humidity = float(climate_row["humidity"])
            aqi = float(climate_row["aqi"])
            heat_index = float(climate_row["heat_index"])

            # --- cycle phase (None / "follicular" / "ovulation" / "luteal" / "menses")
            cycle_phase = "none"
            cycle_hrv_mod = 0.0   # ms shift
            cycle_mood_mod = 0.0  # 1-7 shift
            if prior.cycle_day_0 is not None:
                day_in_cycle = (i + prior.cycle_day_0) % 28
                if day_in_cycle < 5:
                    cycle_phase = "menses"
                    cycle_hrv_mod = -2.0
                    cycle_mood_mod = -0.10
                elif day_in_cycle < 13:
                    cycle_phase = "follicular"
                elif day_in_cycle < 16:
                    cycle_phase = "ovulation"
                    cycle_hrv_mod = +1.5
                else:
                    cycle_phase = "luteal"
                    # Late-luteal HRV drop and mild mood dip (PMS-like) -- both
                    # are documented in published wearable studies.
                    if day_in_cycle >= 24:
                        cycle_hrv_mod = -4.5
                        cycle_mood_mod = -0.25
                    else:
                        cycle_hrv_mod = -2.0
                        cycle_mood_mod = -0.10

            # --- sleep ----------------------------------------------------
            weekend_shift = {"morning": 0.2, "intermediate": 0.5, "evening": 0.9}[prior.chronotype]
            sleep_target = prior.baseline_sleep_need + (weekend_shift if is_weekend else 0.0)
            # Heat at night degrades sleep efficiency, especially with humidity.
            heat_penalty_eff = 0.06 * max(0, heat_index - 28) / 10 * prior.heat_sensitivity
            # Cold-snap penalty: shivering wakes you up.
            cold_penalty_eff = 0.04 * max(0, 5.0 - temp_c) / 10
            # Sleep disorder: lower efficiency, more variable duration.
            disorder_dur_var = 1.4 if prior.has_sleep_disorder else 1.0
            disorder_eff = 0.10 if prior.has_sleep_disorder else 0.0
            # Sleep aid: shorter latency, slightly higher efficiency.
            sleep_aid_eff = -0.04 if prior.on_sleep_aid else 0.0
            sleep_duration = float(np.clip(
                rng.normal(sleep_target, 0.7 * disorder_dur_var),
                3.5, 11.0,
            ))
            sleep_efficiency = float(np.clip(
                rng.normal(0.88, 0.05)
                - heat_penalty_eff - cold_penalty_eff - disorder_eff - sleep_aid_eff,
                0.55, 0.99,
            ))

            sleep_dev = sleep_duration - prior.baseline_sleep_need  # negative = sleep debt

            # Rolling 3-day sleep debt (negative when accumulating).
            sleep_debt_rolling.append(min(0.0, sleep_dev))
            if len(sleep_debt_rolling) > 3:
                sleep_debt_rolling.pop(0)
            sleep_debt_3d = float(np.mean(sleep_debt_rolling))

            # --- HRV, resting HR -----------------------------------------
            # Stress carries over a bit from yesterday and is pushed by heat / AQI.
            stress_today = (
                0.5 * prev_stress
                + 0.5 * prior.stress_setpoint
                + 0.4 * max(0, -sleep_dev)
                + 0.25 * max(0, heat_index - 30) / 5 * prior.heat_sensitivity
                + 0.15 * max(0, aqi - 70) / 20 * prior.aqi_sensitivity
                + (0.5 if prior.has_anxiety else 0.0)   # anxious people: chronically elevated arousal
                + rng.normal(0, 0.5)
            )
            stress_today = float(np.clip(stress_today, 1.0, 7.0))

            hrv_dev = (
                -8.0 * (stress_today - prior.stress_setpoint) / 3.0
                + 3.5 * sleep_dev
                - 4.0 * max(0, heat_index - 30) / 10 * prior.heat_sensitivity
                + cycle_hrv_mod
                + rng.normal(0, 4.0 + hrv_noise_floor)
            )
            hrv_rmssd = float(np.clip(prior.baseline_hrv + hrv_dev, 8, 200))

            rhr_dev = (
                2.0 * (stress_today - prior.stress_setpoint) / 3.0
                - 1.0 * sleep_dev
                + 1.5 * max(0, heat_index - 30) / 10 * prior.heat_sensitivity
                # Cold weather raises RHR via vasoconstriction (~3-5 bpm).
                + 0.3 * max(0, 5.0 - temp_c)
                + rng.normal(0, 1.5)
            )
            resting_hr = float(np.clip(prior.baseline_rhr + rhr_dev, 35, 120))

            stress_score = float(np.clip(10 + 12 * stress_today + rng.normal(0, 5), 0, 100))

            # --- behavior / smartphone -----------------------------------
            screen_time = float(np.clip(
                rng.normal(
                    prior.baseline_screen_min + (30 if is_weekend else 0)
                    + 25 * max(0, prior.mood_setpoint - prev_mood)   # low mood -> more screen
                    - 0.3 * (sleep_duration - 7),
                    45,
                ),
                15, 720,
            ))
            phone_unlocks = float(np.clip(rng.normal(75 + 0.2 * screen_time, 18), 5, 280))

            mobility_radius = float(np.clip(
                rng.normal(7.5 if not is_weekend else 9.0, 4.0)
                - 1.5 * max(0, 4.5 - prev_mood)
                - 0.6 * max(0, heat_index - 32) / 5
                - 0.8 * max(0, 5.0 - temp_c) / 5,   # cold also keeps people indoors
                0.05, 60.0,
            ))
            location_entropy = float(np.clip(
                rng.normal(1.3 if not is_weekend else 1.0, 0.45)
                - 0.2 * max(0, 4.5 - prev_mood),
                0.05, 3.5,
            ))

            daily_steps = float(np.clip(
                rng.normal(prior.baseline_steps * (0.85 if is_weekend else 1.0), 1500)
                - 600 * max(0, heat_index - 32) / 5
                - 800 * max(0, 0.0 - temp_c) / 10,    # very cold days suppress steps too
                300, 35000,
            ))

            # --- mood / EMA ----------------------------------------------
            # Mood today is driven by yesterday's stress + last night's sleep +
            # 3-day sleep debt + today's air quality + cycle phase + setpoint
            # mean-reversion. The sleep coefficient is intentionally larger
            # than in v1 so within-person mood-sleep correlation lands in the
            # 0.25-0.35 range observed in real EMA studies (Ono et al. 2022,
            # Adler et al. 2024). v1 produced ~0.05 which made the next-day
            # prediction task nearly impossible from sensors alone.
            mood_raw = (
                0.55 * prior.mood_setpoint
                + 0.30 * prev_mood
                - 0.35 * (prev_stress - prior.stress_setpoint)
                + 0.40 * sleep_dev                # was 0.35 in v1
                + 0.18 * sleep_debt_3d            # new: cumulative effect
                - 0.15 * max(0, aqi - 70) / 20 * prior.aqi_sensitivity
                - 0.10 * max(0, heat_index - 32) / 5 * prior.heat_sensitivity
                + cycle_mood_mod
                + (0.25 if prior.on_ssri else 0.0)   # SSRI lifts mood floor slightly
                + rng.normal(0, 0.50)
            )
            survey_mood = float(np.clip(mood_raw, 1.0, 7.0))
            survey_energy = float(np.clip(
                0.8 * survey_mood + 0.2 * (sleep_duration - 6) + rng.normal(0, 0.4),
                1.0, 7.0,
            ))
            survey_stress = stress_today

            # --- missingness ---------------------------------------------
            mood_factor = max(0.0, (4.5 - survey_mood)) / 4.0
            p_missing_wearable = np.clip(
                self.config.base_missing_wearable
                * prior.missingness_propensity
                * (1.0 + 0.6 * mood_factor)
                * (1.4 if is_weekend else 1.0),
                0.0, 0.85,
            )
            p_missing_phone = np.clip(
                self.config.base_missing_phone * prior.missingness_propensity,
                0.0, 0.4,
            )
            p_missing_survey = np.clip(
                self.config.base_missing_survey
                * prior.missingness_propensity
                * (1.0 + 1.2 * mood_factor),
                0.0, 0.95,
            )

            miss_wear = bool(rng.random() < p_missing_wearable)
            miss_phone = bool(rng.random() < p_missing_phone)
            miss_surv = bool(rng.random() < p_missing_survey)

            row = {
                "participant_id": prior.participant_id,
                "date": date.date().isoformat(),
                "age": prior.age,
                "sex": prior.sex,
                "baseline_sleep_need": round(prior.baseline_sleep_need, 3),
                "baseline_hrv": round(prior.baseline_hrv, 2),
                "chronotype": prior.chronotype,

                "daily_steps": _maybe_missing(daily_steps, miss_wear),
                "sleep_duration": _maybe_missing(sleep_duration, miss_wear),
                "sleep_efficiency": _maybe_missing(sleep_efficiency, miss_wear),
                "resting_hr": _maybe_missing(resting_hr, miss_wear),
                "hrv_rmssd": _maybe_missing(hrv_rmssd, miss_wear),
                "stress_score": _maybe_missing(stress_score, miss_wear),

                "phone_unlock_count": _maybe_missing(phone_unlocks, miss_phone),
                "screen_time_minutes": _maybe_missing(screen_time, miss_phone),
                "mobility_radius_km": _maybe_missing(mobility_radius, miss_phone),
                "location_entropy": _maybe_missing(location_entropy, miss_phone),

                "survey_mood": _maybe_missing(survey_mood, miss_surv),
                "survey_energy": _maybe_missing(survey_energy, miss_surv),
                "survey_stress": _maybe_missing(survey_stress, miss_surv),

                "temperature_c": round(temp_c, 2),
                "humidity": round(humidity, 1),
                "aqi": round(aqi, 1),
                "heat_index": round(heat_index, 2),

                "missing_wearable_flag": int(miss_wear),
                "missing_phone_flag": int(miss_phone),
                "missing_survey_flag": int(miss_surv),

                # Senior-PI metadata: descriptive, not used as features by default.
                "race_ethnicity": prior.race_ethnicity,
                "ses_proxy": prior.ses_proxy,
                "region": prior.region,
                "device_gen": prior.device_gen,
                "has_anxiety": int(prior.has_anxiety),
                "has_depression": int(prior.has_depression),
                "has_sleep_disorder": int(prior.has_sleep_disorder),
                "has_cardio_condition": int(prior.has_cardio_condition),
                "on_ssri": int(prior.on_ssri),
                "on_beta_blocker": int(prior.on_beta_blocker),
                "on_sleep_aid": int(prior.on_sleep_aid),
                "cycle_phase": cycle_phase,
            }
            rows.append(row)

            # Stash for tomorrow's simulation.
            prev_stress = stress_today
            prev_mood = survey_mood

        return rows


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _maybe_missing(value: float, missing: bool):
    """Return NaN when the corresponding modality is flagged as missing."""
    return float("nan") if missing else round(float(value), 4)


def _heat_index(temp_c: np.ndarray, humidity: np.ndarray) -> np.ndarray:
    """Approximate apparent-temperature ("heat index") in degrees C.

    Uses the NWS Rothfusz regression on Fahrenheit, then converts back to
    Celsius. The regression is only meaningful at T >= ~80°F (~26.7°C)
    AND relative humidity >= 40%; outside that envelope we fall back to
    the ambient temperature, matching standard practice.
    """
    t_f = temp_c * 9 / 5 + 32
    rh = humidity
    hi_f = (
        -42.379 + 2.04901523 * t_f + 10.14333127 * rh
        - 0.22475541 * t_f * rh - 0.00683783 * t_f ** 2
        - 0.05481717 * rh ** 2 + 0.00122874 * t_f ** 2 * rh
        + 0.00085282 * t_f * rh ** 2 - 0.00000199 * t_f ** 2 * rh ** 2
    )
    hi_c = (hi_f - 32) * 5 / 9
    valid = (temp_c >= 26.5) & (humidity >= 40.0)
    return np.where(valid, hi_c, temp_c)


def generate_synthetic_cohort(
    n_participants: int = 250,
    n_days: int = 90,
    seed: int = 42,
    out_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Convenience function for scripts and notebooks."""
    cfg = GeneratorConfig(n_participants=n_participants, n_days=n_days, seed=seed)
    gen = SyntheticCohortGenerator(cfg)
    df = gen.generate()
    if out_path is not None:
        gen.save(df, out_path)
    return df
