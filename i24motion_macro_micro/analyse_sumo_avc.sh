#!/bin/bash
# Summary table and speed-oscillation time series for the demos written by simulate_sumo_avc.sh.
# No arguments; run inside the container after both simulations have finished.
#
#   arms      SUMO IDM (sumo in avc_original), Arm-1 = published AVC controller (trpo in
#             avc_original), Arm-2 = corridor-matched AVC controller (trpo in avc_corridor);
#             the first arm is the paired reference
#   window    summary metrics over 30-60 s of each episode
#   output    run_data/rl/analysis/stability/{tables,figures/timeseries} plus stability.log
set -euo pipefail
cd "$(dirname "$0")"

OUT=run_data/rl/analysis/stability
mkdir -p "$OUT"
python3 sim_rl_sumo_analysis_2.py \
  --arm "SUMO IDM=run_data/rl/analysis/avc_original:sumo" \
  --arm "Arm-1=run_data/rl/analysis/avc_original:trpo" \
  --arm "Arm-2=run_data/rl/analysis/avc_corridor:trpo" \
  --skip-s 30 \
  --oscillation-window-s 10 \
  --out "$OUT" \
  > "$OUT.log" 2>&1
