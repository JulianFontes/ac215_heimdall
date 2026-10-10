#!/usr/bin/env bash
# Train/run the Bifrost forecaster on the C3DIR-like data derived from ERA5 subset.
# Usage: ./run_train.sh [extra args passed to forecasting.py]
set -euo pipefail

cd "$(dirname "$0")/../../.."  # project root

uv run python src/heimdall/cloud_forecasting/forecasting.py \
  -p data/era5/boston_subset/c3dir_like_june2026_N45.5_W74_S39.5_E68.nc \
  --epochs 100 \
  --out training_runs/june2026_epoch100 \
  "$@"