# Forecasting C3DIR cloud fields with a ConvLSTM

C3DIR turns one GOES-19 scan into a 3D cloud field. Run it on consecutive scans and you get a
movie of the 3D cloud field. This package learns to continue that movie: from the previous
`x` frames (default 6, one hour) it predicts the next `k` (default 3, t+1 to t+3 = 30 min).

## Pipeline

```
trained C3DIR model  artifacts/models/c3dir/<version>/
        │   training_days(): the days it was trained on, read from its config.json
        ▼
sequences.py   build_frames()    every daytime 10-min GOES-19 scan of those days
        │                        → C3DIR predict_region() around the site
        │                        → data/c3dir_forecast/frames/<site>/<YYYYmmddTHHMM>.nc
        │      stack_frames()    → data/c3dir_forecast/<site>_sequence.nc  (time, level, lat, lon)
        ▼
dataset.py     to_array(), SequenceDataset    (x frames → next k frames), split by day
        ▼
network.py     ConvLSTMForecaster
forecaster.py  Forecaster: fit, predict, save / load versions
        │                        → artifacts/models/c3dir_forecast/<version>/
        ▼
evaluation.py  scores by lead time, next to persistence
plots.py       history, scores, example forecast
```

| File | What it does |
|---|---|
| `sequences.py` | `training_days`, `build_frames`, `stack_frames`, `load_sequence` |
| `dataset.py` | frames → `(channels, lat, lon)` arrays; gap-free sliding windows; train / val / test by day |
| `network.py` | `ConvLSTMCell`, `ConvLSTMForecaster`, `ForecastConfig` |
| `forecaster.py` | `Forecaster` (training loop, prediction, versioned save / load), `TrainConfig` |
| `evaluation.py` | `scores`, `compare` (ConvLSTM vs persistence) |
| `plots.py` | `plot_scores`, `plot_example` (cloud-top height maps), `plot_history` |

Scripts: `scripts/forecast_build_sequences.py` and `scripts/forecast_train.py`, each with its
settings at the top.

## Frames

Each frame is C3DIR's `predict_region` on one scan: a regular grid around the site with columns
normal to the surface. Settings in `build_frames` / the script:

| Setting | Default | |
|---|---|---|
| region | ± 1.28° latitude (wider in longitude by 1 / cos lat), 0.04° (~4 km) | 65 × 85 cells at Penn State |
| levels | the 80 C3DIR levels averaged 4 at a time | 20 levels of 1 km |
| fields | `prob_any`, ice / liquid / rain water content | |
| times | every 10 min with the sun within 80° of overhead at the site | C3DIR was trained on daytime only |
| days | every day from the first to the last C3DIR training (and validation) day | consecutive days, not only the days with an EarthCARE pass |
| GOES domain | CONUS (`"C"`), nearest scan within 4 min | about 10× smaller downloads than full disk |

Frames are cached, so an interrupted run carries on where it stopped. The raw GOES scans are
deleted once their frame is saved. Expect about 12 s per scan on a laptop (mostly the download):
a month of daytime frames at one site (about 30 × 75 scans) takes roughly 7–8 hours and
downloads about 130 GB in total, but only keeps ~1.5 MB per frame.

## Model

A 2D encoder–forecaster ConvLSTM (Shi et al., 2015). The levels and fields are stacked as
channels: 4 fields × 20 levels = 80.

- **Inputs:** probabilities as is; water contents scaled as in C3DIR, `log10(1 + w / 5e-5) / 4`.
- **Encoder:** two stride-2 3×3 convs (lat / lon / 4), then 2 stacked ConvLSTM layers (64 channels)
  read the `x` input frames.
- **Forecaster:** the same cells keep stepping, each step fed the model's own previous forecast.
  A decoder (two 2× transposed convs) turns the hidden state into a *change* to the last frame:
  probabilities shift in logit space, scaled water adds. The last layer starts at zero, so the
  untrained model is exactly persistence and training learns how clouds move and grow.
- **Loss:** binary cross-entropy on the probability channels (C3DIR's probabilities as soft targets)
  plus mean squared error on scaled water, over cells inside the GOES crop.
- **Training:** AdamW, cosine learning rate 1e-3 → 1e-6, random lat / lon flips, the weights with
  the lowest validation loss kept.
- **Longer horizons:** `predict(x, output_steps=n)` rolls forward past the trained `k`.

The split is by whole days: the last `TEST_DAYS` are test, the `VAL_DAYS` before them validation.
Windows never cross a gap (night, a missing scan).

## Evaluation

On the test days, for each lead time, compared with persistence (the last frame repeated):

| Score | Meaning |
|---|---|
| `csi`, `f1`, `accuracy` | cloud (`prob_any` ≥ 0.5) per voxel |
| `cover_accuracy` | cloudy column (any level cloudy) per cell |
| `prob_mae` | mean absolute error of `prob_any` |
| `*_water_mae` | mean absolute error of scaled water content |

Saved in `artifacts/models/c3dir_forecast/<version>/`: `model.pt`, `config.json` (settings,
C3DIR version, site, days in each split, git commit), `history.csv`, `metrics.csv`, and
`figures/` (history, scores by lead time, the cloudiest test example as cloud-top-height maps).

## Caveats

- The targets are C3DIR's own predictions, not observations, so the forecaster learns to
  continue C3DIR, errors included. Only the EarthCARE tracks are real 3D truth.
- One month at one site is not much data. Training several sites into one model, or more months,
  is the obvious next step.
- Daytime only, like C3DIR.

## Tested so far

There was no trained C3DIR model available when this was written, so the pipeline was checked
with stand-ins:
- real GOES-19 scans through a randomly initialised C3DIR → frames → sequence → arrays;
- a synthetic sequence of drifting cloud blobs: the ConvLSTM reaches a cloud CSI of 0.86 / 0.72 /
  0.60 at t+1 / t+2 / t+3, against 0.57 / 0.31 / 0.14 for persistence.
