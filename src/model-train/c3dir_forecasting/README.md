# C3DIR forecasting: ConvLSTM training (model-train)

A ConvLSTM that forecasts C3DIR's 3D cloud fields: from the `n` frames up to time t, the next `m`
frames (t+1 .. t+m, 10 min apart). Trained on the fields made by
[`data-processing/c3dir_forecasting`](../../data-processing/c3dir_forecasting/README.md) (GOES crops) and
[`model-deploy/c3dir_forecasting`](../../model-deploy/c3dir_forecasting/README.md) (C3DIR fields).
Separate from `c3dir_nowcasting`.

v1 is a baseline to get the pipeline working end to end, not a good model yet.

## Run

From the project root:

```bash
uv run --project src/model-train python -m c3dir_forecasting.train
```

Every setting at the top of `train.py` can be overridden by an environment variable of the same
name, e.g. `INPUT_STEPS=6 OUTPUT_STEPS=6 STEPS=1000`. With `PULL=true` the C3DIR fields are first
copied from the shared bucket, and with `PUSH=true` the new model version is copied to it; both
need the bucket named in `.env` (add `--env-file .env`).

| Setting | Default | |
|---|---|---|
| `INPUT_STEPS` | 3 | `n`: frames up to t (30 min) |
| `OUTPUT_STEPS` | 3 | `m`: frames forecast (30 min) |
| `TEST_DAYS` / `VAL_DAYS` | 1 / 0 | the last days held out for scoring / for picking the best weights. With no validation days, the best weights are picked on the training set |
| `STEPS`, `BATCH_SIZE`, `WIDTH` | 300, 8, 32 | training steps, batch size, network channels |
| `VERSION` | next `v00N` | the folder the model is saved to |
| `PULL`, `PUSH` | false | copy data from / the model to the shared bucket |

About 1.5 minutes on an M2 for the defaults on Aug 1–3 (5 sites, 225 frames).

## How it works

| File | Where | What it does |
|---|---|---|
| `frames.py` | model-core | one C3DIR field → one frame: `prob_any`, the 80 levels averaged 4 at a time (20 levels of 1 km), the central 100 × 100 cells (~400 km), the same shape at every site |
| `network.py` | model-core | encoder–forecaster ConvLSTM; predicts a change to the last frame, starting at zero, so the untrained model is persistence |
| `model.py` | model-core | `Forecaster`: predict, save / load versions |
| `dataset.py` | model-train | each site's frames as a time series; examples are `n + m` frames one scan apart at one site (never across days); whole days go to train / val / test |
| `training.py` | model-train | training loop (binary cross-entropy against C3DIR's probabilities, AdamW, cosine schedule, random east–west flips), scores by lead time |
| `plots.py` | model-train | training history, scores, an example forecast as cloud-top-height maps |
| `train.py` | model-train | the run (settings at the top) |

Shared code lives in model-core so a prediction service in model-deploy can load the same model
and build frames the same way.

## Outputs

`artifacts/models/c3dir_forecasting/<version>/`:

| File | Contents |
|---|---|
| `model.pt` | weights, network and frame settings, `n` and `m` |
| `config.json` | the same settings plus training settings, sites, days in each split, git commit |
| `history.csv` | loss during training |
| `metrics.csv` | test scores by lead time, ConvLSTM and persistence: cloudy-voxel CSI (P ≥ 0.5), P(cloud) mean absolute error, skill over persistence (0 = no better, 1 = perfect) |
| `figures/` | `history.png`, `scores.png`, `example.png` (the cloudiest test example) |
