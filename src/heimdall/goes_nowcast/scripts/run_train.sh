#!/usr/bin/env bash
# Default training run. Extra arguments are passed through to scripts/train.py and override the defaults, e.g.
#   ./goes_nowcast/scripts/run_train.sh --epochs 5 --out runs/quick_test
set -e
cd "$(git rev-parse --show-toplevel)"  # project root

DATA=data/noaa-goes19/ABI-L2-MCMIPC/concat/MCMIPC_G19_20250701T1200_20250701T1400_capecod.nc
OUT=training_runs/goes_testing

uv run python -m heimdall.goes_nowcast.scripts.train \
    -p "$DATA" \
    --epochs 100 \
    --out "$OUT" \
    "$@"
