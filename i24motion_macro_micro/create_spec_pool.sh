#!/bin/bash
# Build the evaluation spec pools. No arguments; run inside the container from anywhere.
#
#   run_data/rl/spec_data/data_full_no_counterfactual.pkl.gz
#       every day in config/ (held-out days included) tiled into non-overlapping max_steps-long
#       episodes, specs only: no do-nothing replays are stored, because the demo runs its baselines
#       live on the same specs.
#   run_data/rl/spec_data/data_full_inband_050_070.pkl.gz
#       the subset whose INITIAL ground-truth density over cells 2-4 (200-500 m, where the bubble
#       starts), averaged over lanes, is 0.5-0.7 rho_j: congested and string-unstable for the corridor
#       IDM, and containing the ring controllers' training bands (AVC 0.57-0.67, EnduRL 0.57-0.70).
#
# Spec sampling reads EnvConfig's defaults (max_steps, warmup_s, band_span_s, ...) from
# sim_rl_sumo_training.py, so rebuild whenever those change. Both pools are overwritten.
set -euo pipefail
cd "$(dirname "$0")"

FULL=run_data/rl/spec_data/data_full_no_counterfactual.pkl.gz
INBAND=run_data/rl/spec_data/data_full_inband_057_068.pkl.gz

python3 sim_rl_spec_pool.py --out "$FULL"

python3 sim_rl_spec_pool.py \
    --filter-from "$FULL" \
    --out "$INBAND" \
    --filter density_band \
    --filter-kwargs '{"low": 0.57, "high": 0.68, "cells": [2, 3, 4]}'
