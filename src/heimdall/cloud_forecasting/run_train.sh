#!/usr/bin/env bash
# Train/run the Bifrost forecaster on the C3DIR-like data derived from ERA5 subset,
# then save height-longitude forecast plots for the K in config.py.
# Usage: ./run_train.sh [extra args passed to forecasting.py]
set -euo pipefail

cd "$(dirname "$0")/../../.."  # project root

DATA=data/era5/boston_subset/c3dir_like_june2026_N45.5_W74_S39.5_E68.nc
OUT=training_runs/june2026_epoch100

uv run python src/heimdall/cloud_forecasting/forecasting.py \
  -p "$DATA" \
  --epochs 1 \
  --out "$OUT" \
  "$@"

K=$(cd src/heimdall/cloud_forecasting && uv run python -c "from config import K; print(K)")

uv run python src/heimdall/cloud_forecasting/plot_forecasts.py \
  -p "$DATA" \
  -c "$OUT/convlstm3d_best.pt" \
  --out "$OUT/forecast_height_lon_k$K"
