# Cloud Forecasting

Small-scale 3D ConvLSTM nowcasting of cloud occurrence. From the previous 6 hours of ice, liquid and rain water content plus occurrence masks on an (altitude, latitude, longitude) grid, the model predicts the three occurrence masks one hour ahead and is compared against a persistence baseline.

Until real C3DIR data is available, training uses C3DIR-like volumes derived from ERA5 (`data/era5/boston_subset/c3dir_like_*.nc`).

## Run

From the top folder of the `heimdall` repo:

```bash
./src/heimdall/cloud_forecasting/run_train.sh --help
```

This trains on a subset for a set number of epochs and writes to a user input directory. Extra arguments are passed through to `forecasting.py` and override the defaults:

```bash
./src/heimdall/cloud_forecasting/run_train.sh --epochs 5 --out training_runs/quick_test
```

To run on a different file, one can update the `run_train.sh` script or call the script directly from the project root:

```bash
uv run python src/heimdall/cloud_forecasting/forecasting.py \
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

## Files

| File | Contents |
| --- | --- |
| `forecasting.py` | Entry point and training loop |
| `config.py` | Constants and fixed settings |
| `data.py` | Loading, channel construction, sliding-window loaders |
| `model.py` | `ConvLSTM3d` model |
| `evaluation.py` | Weighted BCE loss, CSI / F1, validation pass |
| `plots.py` | Training curves and forecast plots |
| `run_train.sh` | Default training run |
