# GOES Nowcasting

Small-scale 2D ConvLSTM nowcasting of GOES-19 ABI imagery. From the previous `T_IN` frames of the bands in `config.BANDS` (5-minute CONUS scans, `MCMIPC` files), the model forecasts the next `T_OUT` frames of the target bands one after another and is compared against a persistence baseline (every forecast frame equals the last observed frame).

## Run

From the project root:

```bash
./src/model-train/goes_nowcasting/run_train.sh --help
```

Extra arguments are passed through to `train.py` and override the defaults:

```bash
./src/model-train/goes_nowcasting/run_train.sh --epochs 5 --out runs/quick_test
```

To run on different data, update `run_train.sh` or run the training module directly:

```bash
uv run --project src/model-train python -m goes_nowcasting.train -p data/goes19 --epochs 20 --out runs/baseline
```

| Argument | Default | Description |
| --- | --- | --- |
| `-p`, `--path` | required | One `MCMIP*.nc` file, a folder of them, or a glob pattern |
| `--epochs` | 20 | Training epochs |
| `--batch` | 2 | Batch size |
| `--lr` | 1e-3 | AdamW learning rate |
| `--out` | `runs` | Output folder (created if missing) |

The device is picked automatically: CUDA, then MPS, then CPU. Settings that change between runs (bands, input / forecast frames, hidden sizes, split) are in `config.py`.

The default windows (`T_IN = 6`, `T_OUT = 6`) need at least 12 consecutive frames per split. For a short file such as the 2-hour Cape Cod sample (24 frames), set `T_IN = 3`, `T_OUT = 2`, `VAL_FRAC = 0.25`, `TEST_FRAC = 0.25` in `config.py`.

## Data handling

- Files are concatenated along time, sorted, and de-duplicated. A time gap larger than 1.5 frames (`MAX_GAP_S` in `loaders.py`) breaks a sequence, so a window never spans an outage.
- The split is chronological (first 70% train, next 15% validation, last 15% test). Windows are built inside each split, so no frame is shared between splits.
- Each band is z-scored with the mean and std of the training frames only.
- Pixels that are NaN in any band are flagged invalid. They are set to the mean after normalizing and ignored by the loss and by every metric.
- Visible bands (1-6) change with sun angle and are near zero at night, so they are not in the default `BANDS`.

## Outputs

Written to the `--out` folder:

- `convlstm_best.pt`: best checkpoint (lowest validation loss), with the normalization statistics
- `history.csv`: per-epoch losses, and validation scores next to persistence (`val_*`, `base_*`): RMSE / MAE per band
- `curves.png`: train/validation loss, and the validation RMSE of every band on one axis: model solid, persistence dashed, same color per band (bands with a different unit, e.g. reflectance, get their own axis). A dotted vertical line marks the best epoch, the one saved as the checkpoint
- `vs_persistence.png`: test RMSE of persistence next to the model for every band in `config.TARGET_BANDS` (first and last lead time; infrared and visible bands in separate panels)
- `forecasts/forecast_CXX.png`, one per target band: truth, ConvLSTM, persistence and the error of both forecasts for one test window, with the RMSE of each error panel. The error colour scale is set per band
- `test_scores.json`: every test score by lead time, for the model and for persistence

Scores are RMSE / MAE in physical units, read from the `units` attribute of each band in the files (K for infrared bands, `1`, i.e. a dimensionless reflectance factor, for visible bands), computed per lead time and band over valid pixels only.

## Files

`goes_nowcasting` is split over three projects in `src/`, each with its own `pyproject.toml` and `uv.lock`. They share the package name (namespace package, no `__init__.py`), so every module imports as `goes_nowcasting.<name>` whichever project it lives in.

| File | Contents |
| --- | --- |
| `model-core/.../config.py` | Settings that change between runs |
| `model-core/.../preprocess.py` | `load_channels` and `scale`: loading and normalization, shared by training and prediction |
| `model-core/.../model.py` | `ConvLSTM` model (encoder-forecaster) |
| `model-train/.../loaders.py` | `prepare_data`: split, normalization and sliding-window loaders in one call |
| `model-train/.../evaluation.py` | Masked MSE loss, RMSE / MAE, validation pass |
| `model-train/.../plots.py` | Training curves and forecast plots |
| `model-train/.../train.py` | Entry point and training loop |
| `model-train/.../evaluate.py` | Scores and plots for a saved checkpoint |
| `model-train/.../goes_video.py` | MP4 of true colour + C13 imagery |
| `model-train/.../run_train.sh`, `run_eval.sh` | Default training / evaluation runs |
| `model-deploy/.../predict.py` | T_OUT-frame forecast from a saved checkpoint, written to netCDF |

Change the architecture, settings or preprocessing only in `model-core`, so training and prediction can't drift apart. Entry points run with `uv run --project src/<model-train or model-deploy> python -m goes_nowcasting.<name>`.

## Forecast

```bash
uv run --project src/model-deploy python -m goes_nowcasting.predict \
  -p data/noaa-goes19/aug2025/MCMIPC_G19_20250801T0000_20250808T0000_psu.nc \
  -c training_runs/pcu_week1/convlstm_best.pt --start "2025-08-07 18:00" --out goes_forecast.nc
```

Reads the `T_IN` frames ending at `--start` (default: the last frame), and writes the next `T_OUT` frames of every target band in physical units, with the input file's `goes_imager_projection`.
