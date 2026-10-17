#!/usr/bin/env bash
# Default training run. Extra arguments are passed through to scripts/train.py and override the defaults, e.g.
#   ./src/heimdall/heimdall-nowcasting/goes_nowcast/scripts/run_train.sh --epochs 5 --out runs/quick_test
set -e
cd "$(git rev-parse --show-toplevel)"  # project root

DATA=data/noaa-goes19/aug2025/MCMIPC_G19_20250801T0000_20250808T0000_psu.nc
OUT=training_runs/pcu_week1

uv run --project src/heimdall/heimdall-nowcasting python -m goes_nowcast.scripts.train \
    -p "$DATA" \
    --epochs 100 \
    --out "$OUT" \
    "$@"
