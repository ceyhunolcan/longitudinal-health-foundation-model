# LifeSnaps adapter — onboarding

LifeSnaps is a 4-month wearable + EMA dataset (n=71, Fitbit Sense + SEMA app)
from Yfantidou et al., *Scientific Data* 2022.

DOI: <https://doi.org/10.5281/zenodo.6826682>
Paper: <https://www.nature.com/articles/s41597-022-01764-x>

## Why this adapter exists

LifeSnaps is the lowest-friction real dataset that gives LHFM something
to chew on:

- Fitbit Sense exposes RMSSD, automatic stress score, sleep stages —
  the wearable modality LHFM was built around.
- EMA covers daily mood, energy, stress on a 1-5 Likert scale (we
  rescale to 1-7).
- Public release with click-through licence (Kaggle) or short access
  form (Zenodo). No DUA review board, no institutional contract.

What LifeSnaps does NOT give you:

- **No smartphone passive sensing.** Phone unlocks, screen time,
  mobility, location entropy all stay NaN. The `missing_phone_flag`
  is set to 1 throughout, which the foundation-model encoder handles
  via its mask-embedding mechanism.
- **No climate data.** The adapter calls Open-Meteo with participant
  timezone as a location proxy (see `LIFESNAPS_TIMEZONE_COORDS` in
  `lifesnaps.py`).
- **No chronotype assessment.** Defaults to `"intermediate"`.

## Step-by-step

### 1. Get the data

**Option A — Kaggle (fastest, ~5 min):**

```bash
# Once you've set up Kaggle API credentials (~/.kaggle/kaggle.json):
mkdir -p data/raw/lifesnaps
cd data/raw/lifesnaps
kaggle datasets download skywescar/lifesnaps-fitbit-dataset
unzip lifesnaps-fitbit-dataset.zip
```

**Option B — Zenodo (official, restricted access form):**

1. Visit <https://zenodo.org/records/6832186> and click "Request access".
2. Fill out the short form (institution, intended use). Typically
   approved within a few days.
3. Download the bundle and unpack to `data/raw/lifesnaps/`.

You should end up with at minimum:

```
data/raw/lifesnaps/
  daily_fitbit_sema_df_unprocessed.csv
  hourly_fitbit_sema_df_unprocessed.csv
  sema_data.csv
  surveys.csv
  ... (and several others LHFM ignores for now)
```

### 2. Run the preflight

Before training anything, run the preflight to see what shape the data
takes after the adapter:

```bash
python scripts/run_pipeline.py \
    --adapter lifesnaps \
    --raw-dir data/raw/lifesnaps \
    --preflight
```

You should see something like:

```
n_participants: 71
n_participants_after_min_days_filter: 65   # 14-day minimum
n_days_min: 30, n_days_median: 110, n_days_max: 128
sex_distribution: {'F': 42, 'M': 28, 'X': 1}
age_mean: 32.5, age_std: 9.1
frac_missing_wearable_flag: 0.18    # missingness varies; this is realistic
frac_missing_phone_flag: 1.00       # by design — no phone sensing in LifeSnaps
frac_missing_survey_flag: 0.41
```

If `n_participants_after_min_days_filter` is below 30, something went
wrong with the date column — check it parsed correctly.

### 3. Run the full pipeline

```bash
python scripts/run_pipeline.py \
    --adapter lifesnaps \
    --raw-dir data/raw/lifesnaps
```

This will:

1. Load and normalise units (sleep minutes → hours, efficiency % → decimal,
   EMA 1-5 → 1-7).
2. Enrich with weather via Open-Meteo (requires `pip install requests`).
   Cached to `data/cache/weather_cache.json` so subsequent runs are fast.
3. Validate the schema.
4. Engineer features.
5. Write `data/processed/features.csv` (and `.parquet`).

Expected runtime: ~10 minutes for the first run (weather API calls take
~1 sec per unique site, of which LifeSnaps has ~10), seconds thereafter.

### 4. Train

```bash
python scripts/train_model.py \
    --features data/processed/features.csv \
    --run-tag lifesnaps-v1
```

For the methodologically honest run (predicting mood without using the
EMA features as inputs):

```bash
python scripts/train_model.py \
    --features data/processed/features.csv \
    --exclude-ema-features \
    --run-tag lifesnaps-emablind
```

### 5. Audit + generalisation

```bash
python scripts/run_fairness_audit.py \
    --checkpoint checkpoints/downstream-lifesnaps-emablind.pt
python scripts/run_climate_holdout.py \
    --holdout heat_wave \
    --checkpoint checkpoints/downstream-lifesnaps-emablind.pt
```

## Known issues

- **EMA scale rescaling is linear.** LifeSnaps mood is 1-5 ordinal;
  we map to 1-7 by `1 + (x-1) * 6/4`. This is a defensible simplification
  but it does mean that the "low mood" threshold (≤3 on the 1-7 scale)
  corresponds to ≤2.33 on the original 1-5, which is reasonable. Document
  this in your methods.

- **Timezone → city centroid is a crude location proxy.** Two participants
  in Athens both get the centroid of Athens. Real participants may be in
  the suburbs. For the climate-health story this is acceptable since
  intra-city temperature variation is small; for AQI it's a bigger
  approximation since AQI varies block-to-block in dense cities.

- **No outcome label other than what LHFM derives from EMA.** LifeSnaps
  ships STAI (anxiety inventory) baselines but those are participant-
  level, not day-level. If you want day-level anxiety prediction, you'll
  need to subclass `LifeSnapsAdapter.binarize_targets` and decide what
  binarisation rule applies.

## Quick comparison to GLOBEM

| | LifeSnaps | GLOBEM |
|---|---|---|
| Access barrier | Low (Kaggle) | Medium (PhysioNet credentialed) |
| n | 71 | 497 |
| Duration | ~4 months | ~10 weeks per term × 4 terms |
| Wearable | Fitbit Sense (full physiology) | Fitbit Charge (no HRV) |
| Smartphone | None | Rich (location, screen, calls) |
| EMA | Daily, multi-prompt | Weekly |
| Demographics | Self-reported | Self-reported + university |
| Outcome | STAI (participant-level) | PHQ/BDI (end-of-term) |
| Climate | Fetched from Open-Meteo | Fetched from Open-Meteo |

**For a strong paper**: train on GLOBEM, validate on LifeSnaps as an
out-of-distribution test. The cross-cohort transfer is the headline.
