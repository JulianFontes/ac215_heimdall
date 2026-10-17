# Cloud Forecasting

Small-scale 3D ConvLSTM nowcasting of cloud occurrence. From the previous 6 hours of ice, liquid and rain water content plus occurrence masks on an (altitude, latitude, longitude) grid, the model predicts the three occurrence masks one hour ahead and is compared against a persistence baseline.

Until real C3DIR data is available, training uses C3DIR-like volumes derived from ERA5 (`data/era5/boston_subset/c3dir_like_*.nc`).

## Run

From the top folder of the `heimdall` repo:

```bash
./src/model-train/c3dir_nowcasting/run_train.sh --help
```

This trains on a subset for a set number of epochs and writes to a user input directory. Extra arguments are passed through to `train.py` and override the defaults:

```bash
./src/model-train/c3dir_nowcasting/run_train.sh --epochs 5 --out training_runs/quick_test
```

To run on a different file, one can update the `run_train.sh` script or run the training module directly from the project root:

```bash
uv run --project src/model-train python -m c3dir_nowcasting.train \
  -p data/era5/boston_subset/c3dir_like_era5_20260601_20260603_N45.5_W74_S39.5_E68.nc \
  --epochs 20 --out training_runs/3day
```

| Argument | Default | Description |
| --- | --- | --- |
| `-p`, `--path` | required | Input `c3dir_like_*.nc` file |
| `--epochs` | 20 | Training epochs |
| `--batch` | 2 | Batch size |
| `--lr` | 1e-3 | AdamW learning rate |
| `--out` | `runs` | Output folder (created if missing) |

The device is picked automatically: CUDA, then MPS, then CPU. Fixed settings (input hours, altitude cutoff, hidden sizes, validation split, seed) are in `config.py`.

## Outputs

Written to the `--out` folder:

- `convlstm3d_best.pt`: best checkpoint (lowest validation loss)
- `history.csv`: per-epoch losses and validation CSI / F1
- `curves.png`: train/validation loss and validation CSI, with persistence shown as dashed lines
- `vs_persistence.png`: validation CSI of the best model next to persistence
- `forecast_height_lon/`: height-longitude forecast sections for each validation stretch

## Forecast for the frontend

`model-deploy/c3dir_nowcasting/predict.py` runs a checkpoint for K forecast hours and writes `voxels.geojson`, `points.csv` and `meta.json` to `heimdall-frontend/public/data/` (overwriting the sample data), so `./start_frontend.sh` shows the forecast:

```bash
uv run --project src/model-deploy python -m c3dir_nowcasting.predict \
  -p data/era5/boston_subset/c3dir_like_june2026_N45.5_W74_S39.5_E68.nc \
  -c training_runs/testing/convlstm3d_best.pt -k 6 --start "2026-06-20 00:00"
```

Each forecast hour is a one-hour-ahead prediction from the T_IN observed hours before it, as in `plot_forecasts.py`, so the volume must cover T_IN hours before `--start` and the K hours after. The files have the same columns as the frontend's sample data plus `cloud_type` (ice / liquid / rain, the most likely type in the bin). Every valid bin is exported (no thresholding; filter on `confidence` in Kepler instead), with `confidence` = that type's probability and `density` = 100 * P(any cloud type). `--start` defaults to the latest start the volume allows; `--out` writes elsewhere.

Only the forecast hours are exported by default. `--history N` also exports the N observed hours before `--start` (e.g. `--history 6`, the model's inputs), so the frontend timeline plays the observations into the forecast. Every row has a `source` column (`observed` / `forecast`); observed bins come from the ERA5 masks, so their `density` is 0 or 100, and clear-air bins have `cloud_type` `none`.

`meta.json` also records how the forecast was made under `"run"`: checkpoint and data paths, `-k` / `--start`, the forecast hours, whether they fall in the model's training period, the model settings and training arguments stored in the checkpoint, and when it was run.

## Files

`c3dir_nowcasting` is split over three projects in `src/`, each with its own `pyproject.toml` and `uv.lock`. They share the package name (namespace package, no `__init__.py`), so every module imports as `c3dir_nowcasting.<name>` whichever project it lives in.

| File | Contents |
| --- | --- |
| `model-core/.../config.py` | Constants and fixed settings |
| `model-core/.../preprocess.py` | `load_channels`: loading and channel construction, shared by training and prediction |
| `model-core/.../model.py` | `ConvLSTM3d` model |
| `model-train/.../loaders.py` | Sliding-window loaders |
| `model-train/.../evaluation.py` | Weighted BCE loss, CSI / F1, validation pass |
| `model-train/.../plots.py` | Training curves and forecast plots |
| `model-train/.../train.py` | Entry point and training loop |
| `model-train/.../plot_forecasts.py` | Height-longitude forecast plots from a saved checkpoint |
| `model-train/.../forecast_video.py` | MP4 of validation cases, observed vs forecast side by side (map + height-longitude section) |
| `model-train/.../run_train.sh` | Default training run (train, then plot forecasts) |
| `model-deploy/.../predict.py` | K-hour forecast from a saved checkpoint, exported for `heimdall-frontend` |

Change the architecture, constants or preprocessing only in `model-core`, so training and prediction can't drift apart. Entry points run with `uv run --project src/<model-train or model-deploy> python -m c3dir_nowcasting.<name>`.
