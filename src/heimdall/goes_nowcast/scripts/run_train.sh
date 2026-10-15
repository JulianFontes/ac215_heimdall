#!/usr/bin/env bash
# Default training run. Extra arguments are passed through to scripts/train.py and override the defaults, e.g.
#   ./goes_nowcast/scripts/run_train.sh --epochs 5 --out runs/quick_test
set -e
cd "$(git rev-parse --show-toplevel)"  # project root

DATA=data/noaa-goes19/MCMIPC_G19_20250701T1200_20250702T0000_capecod.nc
OUT=training_runs/capecod_12h

uv run python -m heimdall.goes_nowcast.scripts.train \
    -p "$DATA" \
    --epochs 100 \
    --out "$OUT" \
    "$@"
