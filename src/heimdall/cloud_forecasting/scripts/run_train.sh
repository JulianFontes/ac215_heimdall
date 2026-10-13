#!/usr/bin/env bash
# Train/run the Bifrost forecaster on the C3DIR-like data derived from ERA5 subset,
# then save height-longitude forecast plots for the K in config.py.
# Usage: ./run_train.sh [extra args passed to train.py]
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"  # project root

DATA=data/era5/boston_subset/c3dir_like_june2026_N45.5_W74_S39.5_E68.nc
OUT=training_runs/test

uv run python -m heimdall.cloud_forecasting.scripts.train \
  -p "$DATA" \
  --epochs 1 \
  --out "$OUT" \
  "$@"

K=$(uv run python -c "from heimdall.cloud_forecasting.config import K; print(K)")

uv run python -m heimdall.cloud_forecasting.scripts.plot_forecasts \
  -p "$DATA" \
  -c "$OUT/convlstm3d_best.pt" \
  --out "$OUT/forecast_height_lon_k$K"
