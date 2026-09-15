#!/bin/bash
# Published AVC ring controller (original_run, model-50) against SUMO's own car-following, on every
# in-band episode of the coupled I-24 simulation. No arguments; run inside the container.
#
#   pool      run_data/rl/spec_data/data_full_inband_050_070.pkl.gz (228 non-overlapping 30 s episodes
#             starting at 0.5-0.7 rho_j; build it with create_spec_pool.sh)
#   settings  0.1 s control step, the published observation normalisers (speed /10, gap /300), hero
#             minGap 0 as on the ring, +/-0.5 m/s^2 (EnvConfig default), policy mean action
#   output    claude_debug_junk_folder/avc_demo_smoke_inband_all/{episodes.csv,traces/,figures/,tables/}
#             plus claude_debug_junk_folder/avc_demo_smoke_inband_all.log
#
# ~25 min for 228 episodes x 2 controllers.
set -euo pipefail
cd "$(dirname "$0")"

OUT=run_data/rl/analysis/avc_original
ORIGINAL_RUN_MODEL=automatic_vehicular_control/run_data/original_run/models/model-50.zip
CORRIDOR_RUN_MODEL=automatic_vehicular_control/run_data/corridor_matched_v4/models/model-50.zip
SCENARIO_SPECS=run_data/rl/spec_data/data_full_inband_057_068.pkl.gz
mkdir -p $OUT
python3 sim_rl_sumo_demo.py \
  --run "$ORIGINAL_RUN_MODEL" --no-run-config \
  --controllers trpo sumo \
  --spec-pool "$SCENARIO_SPECS" --all-pool-specs --episodes 0 \
  --macro-dt 0.1 --obs-max-speed 10 --obs-max-dist 300 --hero-min-gap 0 --store \
  --out "$OUT" \
  > "$OUT.log" 2>&1

OUT=run_data/rl/analysis/avc_corridor
mkdir -p $OUT
python3 sim_rl_sumo_demo.py \
  --run "$CORRIDOR_RUN_MODEL" --no-run-config \
  --controllers trpo sumo \
  --spec-pool "$SCENARIO_SPECS" --all-pool-specs --episodes 0 \
  --macro-dt 0.1 --obs-max-speed 25.02031797294094 --obs-max-dist 300 --hero-min-gap 0 --store \
  --out "$OUT" \
  > "$OUT.log" 2>&1


