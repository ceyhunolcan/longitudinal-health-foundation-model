# Architecture rationale

This page explains the *why* of LHFM's design choices for someone reading
the source code. The *what* is in [`paper/methods.md`](../paper/methods.md);
the *what to look at* is in the [architecture diagram](../README.md#architecture).

## Why a multimodal transformer and not a multimodal RNN/LSTM?

Three reasons specific to passive-sensing data:

- **Mask-aware attention** falls out naturally. Real cohorts have heavy,
  modality-specific missingness (you wore the watch but skipped EMA;
  the phone died at 3am). Attention masks let us tell the encoder "these
  timesteps don't carry signal" without padding with zeros and hoping.
- **Sequence lengths are short** (T = 14 days in our default windows),
  so transformer quadratic cost in T isn't a problem here. RNNs win on
  long sequences; we don't have long sequences.
- **Same machinery handles SSL and downstream** without architectural
  surgery. Mask the input → reconstruction objective. Pool the output →
  binary classifier. One encoder, three objectives.

## Why per-modality projectors (and not feature concatenation)?

Each modality has a different signal-to-noise profile and a different
scale. Wearable HRV varies on the order of ms; smartphone unlock counts
on the order of 100; AQI on the order of 100 but skewed. Throwing them
into a shared projection layer means the gradient updates are dominated
by whichever modality has the biggest numbers.

Per-modality projectors learn a per-modality scale and then sum into a
shared embedding space. This is a standard pattern (see e.g. PerceiverIO,
Florence-VL); it's not exotic.

It also means a new modality is a new projector — you don't have to
retrain the whole encoder when you add audio data from a smartwatch
microphone, say.

## Why an attention-pool over time (and not just the last timestep)?

The "next-day prediction" framing suggests using the last timestep's
representation as the prediction head's input. Two reasons to pool over
time instead:

- **Robustness to missingness on the prediction day.** If the participant
  didn't wear the watch yesterday, the last timestep is mostly mask
  tokens. Pooling over the window gives the model a way to lean on the
  rest of the week.
- **The SSL objective benefits.** Contrastive SSL between two augmented
  views of the same window only makes sense at the window level.

## Why participant embeddings, and why optional?

The encoder takes an optional `participant_idx` and adds a per-participant
embedding to the input. This handles between-person heterogeneity in
baselines (one participant's "elevated stress" is another's "Tuesday")
that the per-day baseline features can't fully capture.

Optional because:

- At inference on a *new* participant we won't have a learned embedding.
  The encoder's `allow_unknown_participants()` method falls back to the
  mean of the embedding table.
- At pretraining without labels we don't always have stable participant
  identities. Some cohorts re-index participants between deployment
  periods.

## Why the 14-day window?

Empirical pick from the digital-health literature:

- **Sleep regularity** stabilises in a ~1-week window.
- **Affective inertia** (today's mood depending on yesterday's stress
  depending on the day before's sleep) plays out over 3-5 days, so we
  want enough history to see the chain.
- **Weekly periodicity** (weekend vs weekday) needs at least 7 days for
  the model to see two of any day-of-week.
- **2 weeks** comfortably covers both, while keeping the transformer
  quadratic cost trivial on CPU.

The window length is the most important hyperparameter to revisit on
real data. Some cohorts (GLOBEM's weekly EMA) might prefer 21 or 28 days.

## Why these four downstream tasks?

The tasks are intentionally diverse along two axes:

| | EMA-derived | Behavioural |
|---|---|---|
| **Same-day predictable** | high_stress (proxies survey_stress) | sleep_disruption (proxies sleep efficiency) |
| **Next-day signal needed** | low_mood (mood depends on prior days' sleep + stress) | climate_vulnerable (HRV recovery from heat stress) |

This matters because if the model is good at *all four*, the foundation-
model story is plausible. If it's good at the EMA-derived tasks but not
the behavioural ones, it's learning EMA self-correlation, not physiology.
The EMA-blind training mode (`--exclude-ema-features`) is what really
tests the second claim.

## Where the choices are made in the source

| Design decision | File(s) |
|---|---|
| Modality slicing + projectors | [`src/lhfm/models/encoder.py`](../src/lhfm/models/encoder.py) — `MultimodalLongitudinalEncoder.__init__` |
| Mask-aware attention | [`src/lhfm/models/encoder.py`](../src/lhfm/models/encoder.py) — `forward()` builds `attn_mask` |
| SSL objectives | [`src/lhfm/models/self_supervised.py`](../src/lhfm/models/self_supervised.py) |
| Downstream heads | [`src/lhfm/models/downstream.py`](../src/lhfm/models/downstream.py) |
| Window construction | [`src/lhfm/data/preprocessing.py`](../src/lhfm/data/preprocessing.py) — `build_windows` |
| Target binarisation | [`src/lhfm/data/preprocessing.py`](../src/lhfm/data/preprocessing.py) — `binarize_targets` |
| Participant-clustered bootstrap | [`src/lhfm/utils/metrics.py`](../src/lhfm/utils/metrics.py) — `bootstrap_ci(groups=...)` |
| Integrated gradients | [`src/lhfm/interpretability.py`](../src/lhfm/interpretability.py) |

## What we *don't* do (and why)

- **Hierarchical attention** (within-day → across-day). Tried it on
  earlier prototypes; complexity didn't pay for itself at T=14.
- **Cross-modal attention** as a separate layer. The per-modality
  projector + summed input + transformer already learns cross-modal
  interactions through attention; a dedicated cross-modal layer was
  redundant in our ablations.
- **Time-2-Vec or learned positional encodings.** Sinusoidal is fine
  here; we don't have enough data to learn positional encodings from
  scratch.
- **A separate uncertainty head.** The integrated-gradients explanation
  + the calibration metrics (ECE, reliability curve) cover most of what
  a clinician cares about. A dropout-MC uncertainty head is on the
  roadmap, not in the prototype.
