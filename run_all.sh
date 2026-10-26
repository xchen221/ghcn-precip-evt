#!/usr/bin/env bash
# Run the whole analysis end to end. Override the interpreter with PYTHON=...
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
mkdir -p logs
for step in 00_selfcheck 01_get_data 02_gev_fit 03_gev_gof 04a_threshold 04b_pot_fit 05_nonstationary; do
    echo "== $step"
    "$PYTHON" "src/$step.py" > "logs/$step.log" 2>&1
    tail -n 1 "logs/$step.log"
done
