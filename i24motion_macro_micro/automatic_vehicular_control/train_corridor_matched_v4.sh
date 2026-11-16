#!/bin/bash
# Arm A'', the AVC-DENSITY-MATCHED ring: the corridor's fundamental diagram swept over
# exactly the density band AVC trained on.
#
# DIFFERS FROM v3 IN worker_kwargs ONLY.  Everything else -- the corridor IDM constants,
# max_speed, circumference_max, accel_penalty, schedule, TRPO/no-critic/global_reward,
# gamma, n_steps -- is byte-identical, so v4 vs v3 is a single-variable comparison of the
# training density distribution.  circumference_max deliberately stays 300 for that
# reason, even though the max swept circumference is now 498.
#
# THE SWEEP.  Density fraction is rho/rho_j = n_veh * jam_spacing / C, so matching AVC's
# band is a single scale factor, the ratio of jam spacings:
#     scale = 12.954218 / 7.000 = 1.850603
#     C = 230 * scale = 425.6   (rho/rho_j 0.6696, their densest)
#     C = 269 * scale = 497.8   (rho/rho_j 0.5725, their sparsest)
# Their 1 m steps become 1.85 m steps; all 40 values are distinct, so no worker is wasted.
# Measured: this band is where the damping prize actually is -- headroom (realised minus
# IDM equilibrium) is -0.34 m/s at C=400 and -0.31 at C=500, against -0.11 at C=600 and
# +0.01 above C=1000.  Every worker here is string unstable.
#
# WHAT THIS TESTS, and it is a sharp prediction rather than a hope.  Both the published
# AVC policy and v3 converge to the same functional form, a switching law
#     v_target = k * v_lead + c
# fitted to within 0.035 m/s.  k is what decides damping: below 1 the AV's target swings
# less than its leader (attenuating, string stable), above 1 it swings more (amplifying).
#   published AVC (original_run, step 50):  v_target = 0.691 * v_lead + 2.101   k = 0.691
#   corridor_matched_v3 (step 70):          v_target = 1.116 * v_lead + 1.944   k = 1.116
# v3 crossed the stability boundary because 20 of its 40 workers are quiescent: they are
# indifferent to k (no wave to amplify) while mildly preferring k > 1 (chase the leader,
# exploit the void behind the moving bottleneck), which dilutes the penalty on k >= 1.
# Here every worker is unstable, so k >= 1 is punished in every episode of every worker.
#
#   PREDICTION: k < 1 on this arm.  If it comes out below 1, the regime-dilution
#   mechanism is confirmed and the narrow band is what makes string stability learnable.
#   If it stays above 1, that explanation is wrong and the cause is elsewhere.
#
# Check it with claude_debug_junk_folder/follower_gain.py on any checkpoint -- it needs no
# rollouts and runs in seconds, so the prediction resolves long before 200 updates:
#   python3 claude_debug_junk_folder/follower_gain.py \
#       automatic_vehicular_control/run_data/corridor_matched_v4/models/model-20.zip \
#       25.02031797294094 300 16.8
# Note the SPEED CURVE WILL NOT SHOW THIS.  avc/speed has been flat within noise
# (sd ~1.0-1.5 m/s) in every run while the policy changed completely underneath it.
export PATH="/home/carla/.local/bin:$PATH"
mkdir -p run_data/corridor_matched_v4/
cd run_data/corridor_matched_v4/ || exit 1
PYTHONPATH="$(pwd)/../.." python3 ../../automatic_vehicular_control/ring.py . \
  "worker_kwargs=[{'circumference':426},{'circumference':427},{'circumference':429},{'circumference':431},{'circumference':433},{'circumference':435},{'circumference':437},{'circumference':439},{'circumference':440},{'circumference':442},{'circumference':444},{'circumference':446},{'circumference':448},{'circumference':450},{'circumference':452},{'circumference':453},{'circumference':455},{'circumference':457},{'circumference':459},{'circumference':461},{'circumference':463},{'circumference':465},{'circumference':466},{'circumference':468},{'circumference':470},{'circumference':472},{'circumference':474},{'circumference':476},{'circumference':477},{'circumference':479},{'circumference':481},{'circumference':483},{'circumference':485},{'circumference':487},{'circumference':489},{'circumference':490},{'circumference':492},{'circumference':494},{'circumference':496},{'circumference':498}]" \
  "n_veh=22" \
  "veh_length=5.249330172413793" \
  "lane_speed=25.02031797294094" \
  "idm=dict(tau=2.261378584805439,minGap=7.704888320362872,maxSpeed=25.02031797294094)" \
  "max_speed=25.02031797294094" \
  "circumference_max=300" \
  "accel_penalty=0" \
  "n_workers=40" "n_rollouts_per_step=40" "warmup_steps=2000" "skip_stat_steps=5000" \
  "horizon=5000" "global_reward=True" "n_steps=50" "alg='TRPO'" "use_critic=False" \
  "gamma=0.9995"
cd ../..
