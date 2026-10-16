#!/usr/bin/env bash
# Evaluate a trained checkpoint on the validation split. Extra arguments are passed through to scripts/evaluate.py, e.g.
#   ./src/heimdall/heimdall-nowcasting/goes_nowcast/scripts/run_eval.sh --out training_runs/pcu_week1/eval_test
set -e
cd "$(git rev-parse --show-toplevel)"  # project root

DATA=data/noaa-goes19/aug2025/MCMIPC_G19_20250801T0000_20250808T0000_psu.nc
CKPT=training_runs/pcu_week1/convlstm_best.pt

uv run --project src/heimdall/heimdall-nowcasting python -m goes_nowcast.scripts.evaluate \
    -p "$DATA" \
    -c "$CKPT" \
    "$@"
