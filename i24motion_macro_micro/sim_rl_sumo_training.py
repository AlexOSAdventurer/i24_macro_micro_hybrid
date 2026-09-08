"""Reinforcement-learning longitudinal control for the hybrid bubble's hero vehicle.

The hero of ``i24_motion_sumo_coupled.I24MotionSumoSimulationCoupled`` is the
vehicle the microscopic bubble is anchored on.  With ``hero_policy=None`` it is
an ordinary SUMO vehicle and the bubble drifts wherever the car-following model
takes it.  This module supplies a ``HeroPolicy`` driven by a learned controller
instead, wraps the whole macro/micro stack in a Gymnasium environment, and
trains it with stable-baselines3.

Why this environment is not just another car-following benchmark: the hero sits
inside a microscopic bubble embedded in the macroscopic CTM, so its observation
includes the *macroscopic* state downstream of the bubble -- density and
velocity in cells the hero could never see with on-board sensors.  That is the
lookahead an I-24 MOTION-informed controller would actually have, and it is what
makes the coupled simulator worth training in.  ``MacroLookaheadController``
below is the hand-written control law that uses the same signal, so a learned
policy has something honest to beat.

Layout
------
``EnvConfig`` / ``RewardConfig``   what an episode is and what it is worth.
``RLHeroPolicy``                   the ``HeroPolicy`` shim the engine calls each
                                   0.1 s substep; holds one macro-step action.
``I24SumoHeroEnv``                 ``gymnasium.Env`` over sim + coupler + bridge,
                                   one episode per bubble traversal.
``BaselineController`` subclasses  non-learned controllers for comparison.
``train``                          sb3-contrib RecurrentPPO (or plain PPO) over
                                   a vector of those environments.

Why recurrent by default
------------------------
This environment is a POMDP.  The hero observes its immediate leader and
follower, a handful of macroscopic cells and its own history of one step; it
does not observe SUMO's internal per-vehicle state, the vehicles beyond its
neighbours, or the empirical trajectories about to be injected at the bubble
boundary.  A feedforward policy can therefore only optimise over *memoryless*
policies, whose optimum can be arbitrarily worse than a history-dependent one,
and its critic is asked to regress ``V(o)`` for an observation whose value
genuinely depends on the belief state.  RecurrentPPO's LSTM carries a summary of
the history into both, which is the principled answer; ``--algorithm ppo``
keeps the memoryless version available as the ablation.

Running
-------
SUMO, gymnasium and stable-baselines3 all live inside the container, so training
runs there::

    docker exec <container> bash -lc \\
        'cd /workspaces/i24motion_macro_micro && python3.10 sim_rl_sumo_training.py \\
             --run-name ppo_smoothing --total-timesteps 100000 --envs 4'

An episode is one traversal of the bubble from ``initial_middle_s`` to
``max_middle_s`` -- roughly 40 macro steps at 1 Hz, about 30 s of wall clock
including the reset -- so budget accordingly: 100k environment steps is a couple
of days serially and a handful of hours across four environments.

Outputs land in ``run_data/rl/<run_name>/``: ``run_config.json``, SB3's
``progress.csv``, per-worker ``monitor_*.csv`` episode logs, and
``checkpoints/``.  ``sim_rl_sumo_demo.py`` replays a checkpoint and
``sim_rl_sumo_analysis.py`` turns these logs into figures and LaTeX tables.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
import zipfile
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulation import FundamentalDiagram, I24MicroMask, Simulation, GroundTruthStore, TriangularFD
from i24_micro_bridge import I24MicroSimBridge
from i24_sumo_coupler import I24SumoCoupler
from i24_motion_sumo_coupled import HeroPolicy

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG_FOLDER = os.path.join(HERE, "config")
DEFAULT_OUTPUT_ROOT = os.path.join(HERE, "run_data", "rl")

# Same triangular fit the SUMO runs in sim_runs.py / sim_demo.py use, so a
# trained controller sees the fundamental diagram those rollouts were produced
# with.  lambda_lc is a lane-change parameter and is not part of the FD.
DEFAULT_FD_PARAMS = {
    "v_f": 25.02031797294094,
    "w": 5.728460762748048,
    "rho_j": 0.07719493079089293,
}

# Dataset json read once per path; every episode re-parses the text so nothing
# downstream can mutate a shared config dict.
_CONFIG_TEXT_CACHE: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EnvConfig:
    """Everything that defines an episode of the coupled simulation."""

    config_folder: str = DEFAULT_CONFIG_FOLDER
    # Dataset json files (dates) episodes are drawn from.  Each one is a
    # different day of I-24 MOTION data, so this is the train/test split axis.
    datasets: Tuple[str, ...] = ("2022-11-30.json",)
    roads: Tuple[str, ...] = ("2",)
    lanes: Tuple[int, ...] = (-1, -2, -3, -4)

    # Bubble geometry, matching the SUMO runs in sim_runs.py.
    initial_middle_s: float = 350.0
    max_middle_s: float = 1300.0
    margin_s: float = 150.0
    visible_window: float = 150.0
    ghost_window: float = 0.0
    cell_length: float = 100.0

    # Episode start times are drawn from [time_origin + warmup_s, ...]; the
    # first hour of every day is skipped because the macro state is still
    # relaxing out of its initial condition.
    warmup_s: float = 3600.0
    # Micro trajectories are read out of a 1.6 GB parquet, so they are loaded one
    # time band at a time and several episodes are drawn from each band before
    # paying for the next read.
    band_span_s: float = 900.0
    episodes_per_band: int = 4

    # Micro engine.
    step_length: float = 0.1
    fd_params: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FD_PARAMS))
    gui: bool = False
    verbose: bool = False
    # SUMO's lane-change model would otherwise move the hero sideways, which a
    # longitudinal controller has no say over and cannot be credited for.
    lock_hero_lane: bool = True
    # SUMO vType every vehicle is inserted as.  "car" is deterministic IDM (the
    # control condition); "car_eidm" is Extended IDM with driver imperfection, so
    # the platoon is string unstable and stop-and-go waves form on their own.
    # Under "car" a wave-damping controller has nothing to damp -- measured:
    # FollowerStopper scores identically to plain IDM, and the reward's ceiling
    # over the do-nothing counterfactual is +0.0006/step.
    vehicle_type: str = "car_eidm"
    # SUMO speed mode for the hero, or None to leave its default (31, every
    # check on).  The default is the right setting for a wave-damping
    # controller: the commanded speed is clipped to the safe speed, so the hero
    # can always slow down strategically but never asks to exceed the prevailing
    # speed, which is exactly the one-sided authority a field-deployable
    # smoothing controller has.  A consequence worth knowing when reading the
    # demo's traces: in congestion every "go faster" controller is pinned to the
    # same safe speed and their trajectories coincide, so only the braking
    # decisions separate them.  Set 30 (safe-speed check off, acceleration
    # bounds kept) or 0 (full authority) for an ablation that gives the
    # controller the whole envelope; the reward's collision term stops being
    # decorative at that point.
    hero_speed_mode: Optional[int] = None

    # Episode limits.  The bubble normally retires on its own at max_middle_s
    # after ~40 steps; max_steps only catches a hero that crawls.
    max_steps: int = 120
    max_reset_attempts: int = 8

    # Action: one desired acceleration per macro step, held across the ten
    # 0.1 s substeps.  SUMO clips it to its own safe speed and bounds, so the
    # realised acceleration is what the reward is computed from.
    max_acceleration: float = 1.5
    max_deceleration: float = 2.0

    # Observation: how many macroscopic cells downstream of the bubble the hero
    # is allowed to see.  This is the part no on-board sensor could supply.
    macro_lookahead_cells: int = 1
    # Metrics: How many macroscopic cells upstream of the bubble do we evaluate our
    # macro rewards against. 
    macro_lookbehind_cells: int = 6
    # Vehicles within this distance behind the hero, in its own lane, are the
    # platoon whose smoothness the controller is judged on.
    platoon_window: float = 150.0

    def dataset_paths(self, dataset: str) -> Tuple[str, Dict[str, Any]]:
        config_path = os.path.join(self.config_folder, dataset)
        if (config_path not in _CONFIG_TEXT_CACHE):
            with open(config_path, "r") as handle:
                _CONFIG_TEXT_CACHE[config_path] = handle.read()
        # Parsed fresh on every call: the coupler keeps the dict it is handed,
        # so the cache holds the text rather than a shared object.
        config = json.loads(_CONFIG_TEXT_CACHE[config_path])
        dataset_dir = config["storage_locations"]["simulation_dataset"]
        if (not os.path.isabs(dataset_dir)):
            dataset_dir = os.path.join(HERE, dataset_dir)
        return dataset_dir, config

    def fundamental_diagram(self) -> TriangularFD:
        return TriangularFD(
            v_f=self.fd_params["v_f"], w=self.fd_params["w"], rho_j=self.fd_params["rho_j"]
        )

    @property
    def observation_size(self) -> int:
        return 15 + (2 * self.macro_lookahead_cells)


@dataclass
class RewardConfig:
    """Weights on the reward's components.

    Every component is written so that "more is better" before weighting, and
    each is normalised to roughly [-1, 1] per step, so the weights read as
    relative priorities rather than as unit conversions.  Components are always
    computed and reported in ``info``; a zero weight only removes one from the
    scalar the agent optimises, so the analysis can still show what a run that
    ignored it did to, say, energy.
    """

    platoon_speed: float = 0.0     # mean speed of the hero and its followers
    progress: float = 0.02         # one-sided floor: penalise *stalling* only
    speed_variance: float = 0.0    # penalise stop-and-go within the platoon
    acceleration: float = 0.0     # penalise realised |a|, a comfort/energy proxy
    jerk: float = 0.00             # penalise changes in realised a
    headway: float = 0.0          # penalise time headways below target_headway
    energy: float = 0.0            # penalise tractive energy (always reported)
    stopped: float = 0.0          # penalise standing still
    collision: float = 10.0        # one-off penalty, terminates the episode

    target_headway: float = 1.5    # seconds
    stopped_speed: float = 0.5     # m/s
    # Fraction of the prevailing speed the hero may drop to before the progress
    # term starts charging it.  A wave-damping controller works *by* slowing, so
    # this is a guardrail against the degenerate "stop dead" policy rather than
    # an objective: above the floor the term is flat and contributes no gradient
    # at all.  Keep it low -- a hero that stalls hard builds its own queue, which
    # crosses rho_c upstream and shows up in the macroscopic terms anyway.
    progress_floor: float = 0.02    # fraction of the reference speed
    # What "the prevailing speed" means for that floor.  "leader" is the vehicle
    # ahead in the hero's own lane, "macro_lookahead" the first CTM cell past the
    # front of the bubble, "follow_speed" SUMO's getFollowSpeed (an ablation --
    # see _reference_speed for why it is the wrong reference for a floor).
    progress_reference: str = "leader"
    # Seconds of reference speed to average over.  The instantaneous leader speed
    # is the phase of the wave one vehicle ahead, not the prevailing speed: in
    # stop-and-go the leader pulls out of a jam seconds before the hero can, so an
    # unsmoothed floor fires continuously on a hero that is doing nothing wrong.
    # 0.0 disables the smoothing.
    progress_reference_window: float = 10.0
    # Tractive energy per metre travelled that scores -1 before weighting.
    energy_scale: float = 2000.0   # J/m
    # "lane_behind" (the hero's own followers -- the only vehicles a longitudinal
    # controller actually influences), "all_behind", or "bubble".
    platoon_scope: str = "lane_behind"

    # What the upstream terms are scored against.  "baseline" replays the same
    # episode with the hero on SUMO's own car-following model and differences
    # against that, which asks "is this controller better than doing nothing".
    # "empirical" differences against the measured road instead, which asks "is
    # the simulation better than the road was" -- a question about the CTM, not
    # the policy, and one whose answer is dominated by model error the controller
    # cannot move.  Scoring against "empirical" is what left both upstream terms
    # reading simulator bias: a free positive on oscillation from the first-order
    # scheme's numerical diffusion, and a persistent negative on delay.
    # "baseline" costs one extra environment pass per episode.
    upstream_reference: str = "baseline"

    # Macroscopic Objectives
    rear_flux_smoothness: float = 0.00
    upstream_oscillation: float = 0.02
    upstream_delay: float = 0.98


@dataclass
class EnergyModel:
    """Tractive power proxy, integrated over the hero's 0.1 s speed trace.

    Not a calibrated fuel model -- it exists so runs can be compared on
    something with units, and so the analysis has an energy column whether or
    not energy was in the reward.
    """

    mass: float = 1500.0            # kg
    rolling_resistance: float = 0.01
    drag_coefficient: float = 0.32
    frontal_area: float = 2.4       # m^2
    air_density: float = 1.2        # kg/m^3
    gravity: float = 9.81

    def power(self, speed: float, acceleration: float) -> float:
        """Instantaneous tractive power in W, floored at zero (no regeneration)."""
        inertia = self.mass * acceleration * speed
        rolling = self.mass * self.gravity * self.rolling_resistance * speed
        drag = 0.5 * self.air_density * self.drag_coefficient * self.frontal_area * (speed ** 3)
        return max(0.0, inertia + rolling + drag)


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------


class GroundTruthCache:
    """One-entry cache over time-banded, road-filtered micro trajectories.

    ``GroundTruthStore.from_parquet`` reads the whole 1.6 GB micro parquet, which
    neither fits comfortably alongside several rollout workers nor is needed: the
    coupler only ever seeds from a few hundred metres over a few seconds.  Reading
    one band costs about two seconds and about 300 MB per road, so episodes are
    drawn in groups from the same band.
    """

    def __init__(self, pad_s: float = 120.0) -> None:
        self.pad_s = float(pad_s)
        self._key: Optional[Tuple[str, str, float, float]] = None
        self._store: Optional[GroundTruthStore] = None

    def get(
        self, dataset_dir: str, road: Optional[str], band_start: float, band_end: float
    ) -> GroundTruthStore:
        import pandas as pd

        key = (dataset_dir, str(road), float(band_start), float(band_end))
        if ((self._key == key) and (self._store is not None)):
            return self._store

        # Drop the previous store before reading the next one; two bands of
        # trajectories resident at once is the difference between fitting in RAM
        # and not when several environments are running.
        self._key = None
        self._store = None

        filters = [
            ("time", ">=", float(band_start) - self.pad_s),
            ("time", "<=", float(band_end) + self.pad_s),
        ]
        macro_filters = None
        if (road is not None):
            filters.append(("road_id", "==", str(road)))
            # The macro read used to take the whole file while the micro read was
            # filtered, which is 310 MB per worker for rows no episode on this road
            # can query -- _macro_density_lookup is keyed by (road_id, cell_id), so
            # the other road's entries are simply never looked up.
            macro_filters = [("road_id", "==", str(road))]
        micro_df = pd.read_parquet(os.path.join(dataset_dir, "micro.parquet"), filters=filters)
        macro_df = pd.read_parquet(os.path.join(dataset_dir, "macro.parquet"), filters=macro_filters)
        self._store = GroundTruthStore(micro_df, macro_df)
        self._key = key

        # Reading the band costs about three times the resident size of the frame
        # it produces, and pyarrow's pool keeps the difference: measured 2,470 MB
        # resident falling to 1,914 MB on release_unused() after five bands. The
        # frames above are already materialised into pandas, so nothing live is
        # being handed back.
        del micro_df, macro_df
        try:
            import pyarrow

            pyarrow.default_memory_pool().release_unused()
        except Exception:
            pass
        return self._store


# ---------------------------------------------------------------------------
# The hero policy the SUMO engine actually calls
# ---------------------------------------------------------------------------


class RLHeroPolicy(HeroPolicy):
    """Holds one macro-step action and applies it on every 0.1 s substep.

    The engine calls ``act`` once per substep, ten times per macro step, well
    inside ``Simulation.step()``.  Rather than invert control so the learner can
    be asked mid-step, the environment writes the action here before stepping and
    this class holds it -- the controller runs at the 1 Hz macro rate while the
    engine still integrates at 10 Hz.

    In passthrough mode it hands the hero back to SUMO's car-following model.
    That is not the same as ``hero_policy=None``: ``release_spawn_holds`` only
    releases the hero's spawn speed when there is no policy at all, so a policy
    that returned ``None`` would leave the hero pinned at its spawn speed
    forever.  Passthrough re-sends ``setSpeed(-1)`` instead.
    """

    def __init__(self, lock_hero_lane: bool = True, speed_mode: Optional[int] = None) -> None:
        self.engine = None
        self.commanded_acceleration: Optional[float] = None
        self.samples: List[Dict[str, float]] = []
        self.lock_hero_lane = bool(lock_hero_lane)
        self.speed_mode = speed_mode
        self._configured_id: Optional[str] = None

    # -- environment side ------------------------------------------------

    def set_acceleration(self, acceleration: Optional[float]) -> None:
        """``None`` returns the hero to SUMO's car-following model."""
        self.commanded_acceleration = None if (acceleration is None) else float(acceleration)

    def drain_samples(self) -> List[Dict[str, float]]:
        samples = self.samples
        self.samples = []
        return samples

    # -- engine side -----------------------------------------------------

    def reset(self, engine) -> None:
        self.engine = engine
        self._configured_id = None

    def _configure_hero(self) -> None:
        """Put the hero's lane-change and speed modes where this run wants them.

        Reapplied whenever the hero's SUMO id changes, because
        ``_ensure_hero_present`` reinserts it under a fresh record after a loss.
        """
        if ((self.engine is None) or (self.engine.hero_state is None)):
            return
        sumo_id = self.engine.hero_state["sumo_id"]
        if (sumo_id == self._configured_id):
            return
        try:
            if (self.lock_hero_lane):
                self.engine.conn.vehicle.setLaneChangeMode(sumo_id, 0)
            if (self.speed_mode is not None):
                self.engine.conn.vehicle.setSpeedMode(sumo_id, int(self.speed_mode))
            self._configured_id = sumo_id
        except Exception:
            pass

    def _get_follow_speed(self, fd: TriangularFD, eps_v = 1.0):
        if ((self.engine is None) or (self.engine.hero_state is None)):
            return None
        leader = self.engine.conn.vehicle.getLeader(self._configured_id)
        if (leader is None) or (leader[0] == ""):
            return fd.v_f
        leader_id, gap = leader
        ego_speed = self.engine.conn.vehicle.getSpeed(self._configured_id)
        leader_speed = self.engine.conn.vehicle.getSpeed(leader_id)
        leader_decel = self.engine.conn.vehicle.getDecel(leader_id)
        return max(self.engine.conn.vehicle.getFollowSpeed(self._configured_id, ego_speed, gap, leader_speed, leader_decel, leader_id), eps_v)

    def act(self, observation: dict) -> Optional[dict]:
        self._configure_hero()
        leader = observation.get("leader")
        self.samples.append({
            "time": float(observation["time"]),
            "speed": float(observation["speed"]),
            "commanded_acceleration": (
                float("nan") if (self.commanded_acceleration is None) else float(self.commanded_acceleration)
            ),
            "leader_gap": float("inf") if (leader is None) else float(leader["gap"]),
            "leader_speed": float("nan") if (leader is None) else float(leader["speed"]),
        })
        if (self.commanded_acceleration is None):
            return {"speed": -1.0}
        return {"acceleration": self.commanded_acceleration}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@dataclass
class EpisodeSpec:
    """The randomised part of an episode: which day, which road, which minute."""

    dataset: str
    road: str
    start_time: float
    band_start: float
    band_end: float
    sumo_seed: int


# Per-episode metrics carried in the final step's info dict.  Monitor is
# configured with these as info_keywords, so every episode lands in
# monitor_*.csv next to its return and length with no extra plumbing.
EPISODE_INFO_KEYS = (
    "dataset", "road", "start_time", "terminal_reason", "hero_mean_speed",
    "mean_progress_reward", "mean_rear_flux_smoothness", "mean_upstream_oscillation", "mean_upstream_delay",
    "hero_speed_std", "platoon_mean_speed", "platoon_speed_std",
    "acceleration_rms", "jerk_rms", "min_headway", "distance", "energy",
    "energy_per_metre", 
)


class I24SumoHeroEnv(gym.Env):
    """Gymnasium environment over the coupled macro/micro simulation.

    One episode is one traversal of the bubble along the corridor.  ``step``
    advances the macroscopic simulation by one time step (1 s), which internally
    rebuilds the four lane masks, runs the fluid solve, and advances SUMO by ten
    0.1 s substeps with the hero under this environment's action.

    The action is a scalar in [-1, 1], mapped to a desired acceleration
    (asymmetrically: -1 is ``max_deceleration``, +1 is ``max_acceleration``).
    SUMO applies it under its default speed mode, so it is clipped to the safe
    speed: the controller can slow down whenever it wants but never travels
    faster than the prevailing speed the car-following model would allow.  That
    is the authority a wave-damping controller actually has, and it means the
    controller cannot cause a rear-end collision -- so the reward's headway term
    is about comfort and smoothing rather than about crash avoidance.  See
    ``EnvConfig.hero_speed_mode`` to relax it for an ablation.

    ``reset(options={"spec": EpisodeSpec(...)})`` pins the episode instead of
    sampling one, which is how the demo and the analysis replay a fixed set of
    scenarios across controllers.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_config: Optional[EnvConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        energy_model: Optional[EnergyModel] = None,
        seed: int = 0,
        label_prefix: str = "rl",
        record_rollout: bool = False,
        ground_truth_cache: Optional[GroundTruthCache] = None,
    ) -> None:
        super().__init__()
        self.env_config = env_config or EnvConfig()
        self.reward_config = reward_config or RewardConfig()
        self.energy_model = energy_model or EnergyModel()
        self.rng = np.random.default_rng(seed)
        self.label_prefix = label_prefix
        self.record_rollout = bool(record_rollout)
        self.gt_cache = ground_truth_cache or GroundTruthCache()

        self.fd = self.env_config.fundamental_diagram()
        self.observation_size = self.env_config.observation_size
        # Every feature is scaled by v_f, rho_j or a window length, so the box is
        # a containment guarantee rather than a real range; observations are
        # clipped into it so a transient (an empty lookahead, a density above
        # rho_j) can never violate the space.
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.observation_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.sim: Optional[Simulation] = None
        self.coupler: Optional[I24SumoCoupler] = None
        self.bridge: Optional[I24MicroSimBridge] = None
        self.policy_shim: Optional[RLHeroPolicy] = None
        self.spec: Optional[EpisodeSpec] = None
        self.gt: Optional[GroundTruthStore] = None

        self._episode_counter = 0
        self._band: Optional[Tuple[str, str, float, float]] = None
        self._band_uses = 0
        self._step_index = 0
        self._previous_action = 0.0
        self._previous_acceleration = 0.0
        # Macro information.  Two steps of history: the flux term differences twice.
        self._previous_rear_flux: Optional[Dict[int, float]] = None
        self._previous_rear_flux_2: Optional[Dict[int, float]] = None
        self._macro_dt = 1.0
        self._reference_speed_history: Deque[float] = deque()
        # Base-cell densities, one snapshot per macro step, from the do-nothing
        # replay of the current episode.  Indexed by step, so the policy pass
        # compares against the counterfactual at the same *wall-clock time* --
        # the two runs diverge in position but advance in lockstep in time, and
        # the snapshot is keyed by cell so the lookup follows the bubble wherever
        # the policy has taken it.
        self._baseline_density_maps: List[Dict[Tuple[str, str], float]] = []
        self._baseline_rear_fluxes: List[Dict[int, float]] = []
        self._last_observation = np.zeros(self.observation_size, dtype=np.float32)
        self._episode_records: List[Dict[str, float]] = []
        self._terminal_reason: str = ""

    # ------------------------------------------------------------------
    # Episode sampling
    # ------------------------------------------------------------------

    def _sample_band(self, dataset: str) -> Tuple[float, float]:
        _, config = self.env_config.dataset_paths(dataset)
        origin = float(config["time_origin"])
        length = float(config["time_length"])
        first = origin + self.env_config.warmup_s
        # Leave room for the longest episode the step cap allows.
        budget = (self.env_config.max_steps * float(config["time_step"])) + 60.0
        last = origin + length - budget
        if (last <= first):
            raise ValueError(f"{dataset} is too short for warmup_s={self.env_config.warmup_s}")
        span = self.env_config.band_span_s
        band_count = max(1, int(math.floor((last - first) / span)))
        index = int(self.rng.integers(0, band_count))
        band_start = first + (index * span)
        return band_start, min(band_start + span, last)

    def _quantize_start_time(self, dataset: str, time_value: float) -> float:
        """Snap a start time onto the macroscopic snapshot grid.

        ``initialize_from_ground_truth`` looks the initial condition up in the
        macro store with a 1e-6 tolerance, so a start time drawn uniformly out of
        a band lands between snapshots and the episode fails to build.
        """
        _, config = self.env_config.dataset_paths(dataset)
        origin = float(config["time_origin"])
        step = float(config["time_step"])
        return origin + (round((float(time_value) - origin) / step) * step)

    def _sample_spec(self) -> EpisodeSpec:
        dataset = str(self.rng.choice(np.asarray(self.env_config.datasets)))
        road = str(self.rng.choice(np.asarray(self.env_config.roads)))
        reuse = (
            (self._band is not None)
            and (self._band[0] == dataset)
            and (self._band[1] == road)
            and (self._band_uses < self.env_config.episodes_per_band)
        )
        if (reuse):
            _, _, band_start, band_end = self._band
            self._band_uses += 1
        else:
            band_start, band_end = self._sample_band(dataset)
            self._band = (dataset, road, band_start, band_end)
            self._band_uses = 1
        return EpisodeSpec(
            dataset=dataset,
            road=road,
            start_time=self._quantize_start_time(dataset, self.rng.uniform(band_start, band_end)),
            band_start=band_start,
            band_end=band_end,
            sumo_seed=int(self.rng.integers(0, 2 ** 31 - 1)),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _build_episode(self, spec: EpisodeSpec) -> None:
        env_config = self.env_config
        # The coupler draws spawned vehicles' lengths with the global `random`
        # module (generate_vehicle_state_from_spawn), so pinning it here is what
        # makes an episode spec reproducible -- without it the same spec replayed
        # under two controllers gets two different sets of injected vehicles and
        # the paired comparison in the demo measures the spawn stream as much as
        # the controller.
        random.seed(spec.sumo_seed)
        dataset_dir, config = env_config.dataset_paths(spec.dataset)
        self._macro_dt = float(config["time_step"])
        gt = self.gt_cache.get(dataset_dir, spec.road, spec.band_start, spec.band_end)

        sim = Simulation.from_json(
            json_path=os.path.join(dataset_dir, "network.json"),
            time_resolution=config["time_step"],
            origin_time=spec.start_time,
            min_cell_length=env_config.cell_length,
        )
        # A rollout snapshot is a full deep copy of the base network per step;
        # training never looks at them, the demo does.
        sim.record_rollout = self.record_rollout
        sim.initialize_from_ground_truth(gt, time_value=spec.start_time)

        self._episode_counter += 1
        self.policy_shim = RLHeroPolicy(
            lock_hero_lane=env_config.lock_hero_lane, speed_mode=env_config.hero_speed_mode
        )
        coupler = I24SumoCoupler(
            gt,
            dt=config["time_step"],
            fd=self.fd,
            lanes=list(env_config.lanes),
            mapping=config,
            hero_road=spec.road,
            desired_time=spec.start_time,
            desired_s=env_config.initial_middle_s,
            visible_window=env_config.visible_window,
            ghost_window=env_config.ghost_window,
            vehicle_type=env_config.vehicle_type,
            step_length=env_config.step_length,
            seed=spec.sumo_seed,
            gui=env_config.gui,
            hero_policy=self.policy_shim,
            label=f"{self.label_prefix}_{os.getpid()}_{self._episode_counter}",
            verbose=env_config.verbose,
        )
        bridge = I24MicroSimBridge(
            sim=sim,
            road_id=spec.road,
            lanes=list(env_config.lanes),
            initial_middle_s=env_config.initial_middle_s,
            margin_s=env_config.margin_s,
            max_middle_s=env_config.max_middle_s,
            micro_coupler=coupler,
            bridge_callback_name="bridge_step",
        )
        self.sim, self.coupler, self.bridge, self.spec, self.gt = sim, coupler, bridge, spec, gt

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Start a new episode and return the first observation.

        The first macroscopic step is taken with the hero on SUMO's own
        car-following model.  That step is what brings the engine up and puts the
        first real mask into the network, so the macro cells the observation
        reads only exist afterwards -- and it hands the controller a settled
        vehicle rather than one still pinned at its spawn speed.
        """
        super().reset(seed=seed)
        if (seed is not None):
            self.rng = np.random.default_rng(seed)
        pinned = None if (options is None) else options.get("spec")

        self.close()
        last_error: Optional[Exception] = None
        for _ in range(self.env_config.max_reset_attempts):
            spec = pinned if (pinned is not None) else self._sample_spec()
            try:
                # The counterfactual has to be recorded before the pass that is
                # scored against it, and it builds and tears down its own episode.
                self._baseline_density_maps, self._baseline_rear_fluxes = (
                    self._baseline_pass(spec)
                    if (self.reward_config.upstream_reference == "baseline") else ([], [])
                )
                self._build_episode(spec)
                self._step_index = 0
                self._previous_action = 0.0
                self._previous_acceleration = 0.0
                self._previous_rear_flux = None
                self._previous_rear_flux_2 = None
                self._episode_records = []
                self._terminal_reason = ""
                self.policy_shim.set_acceleration(None)
                self.sim.step()
                if (not self.bridge.running):
                    raise RuntimeError("bubble retired on the warm-up step")
                if (self._retires_next_step()):
                    raise RuntimeError("bubble starts past max_middle_s")
                # Seed the previous acceleration from the warm-up step rather
                # than asserting zero.  It is both an observation feature and the
                # reference the first step's jerk penalty is measured against, so
                # a fabricated zero would charge the controller for a jerk it did
                # not cause -- the same fabricated-boundary-state problem as the
                # phantom terminal transition, at the other end of the episode.
                warmup = self._substep_metrics(
                    self.policy_shim.drain_samples(), float(self.coupler.hero_state["velocity"])
                )
                self._previous_acceleration = warmup["acceleration"]
                self._previous_rear_flux = self._rear_fluxes()
                # Seed the reference history from the warm-up step for the same
                # reason as the acceleration above: an empty window on step 1 would
                # score the floor against a single instantaneous reading, which is
                # exactly the jumpy quantity the smoothing exists to remove.
                self._reference_speed_history = deque(maxlen=self._reference_speed_window())
                self._reference_speed_history.append(self._instantaneous_reference_speed())
                self._last_observation = self._observation()
                return self._last_observation, {"spec": asdict(spec)}
            except Exception as exc:  # a minute with no usable hero, or a SUMO failure
                last_error = exc
                self.close()
                if (pinned is not None):
                    raise
        raise RuntimeError(
            f"could not start an episode in {self.env_config.max_reset_attempts} attempts: {last_error}"
        )

    def close(self) -> None:
        if ((self.bridge is not None) and self.bridge.running):
            try:
                self.bridge.destroy()  # also tears down the coupler and SUMO
            except Exception:
                pass
        elif (self.coupler is not None):
            # A build that failed after the engine came up but before the bridge
            # took ownership would otherwise leak a TraCI connection.
            try:
                self.coupler.destroy()
            except Exception:
                pass
        self.sim = None
        self.coupler = None
        self.bridge = None
        self.policy_shim = None

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _hero(self) -> Dict[str, Any]:
        return self.coupler.hero_state

    def _get_follow_speed(self) -> float:
        return self.policy_shim._get_follow_speed(self.fd)

    def _instantaneous_reference_speed(self) -> float:
        """The speed of the traffic the hero is measured against, right now.

        Deliberately *exogenous* to the hero.  ``getFollowSpeed`` is the obvious
        choice and the wrong one: it is computed from the current gap, so opening
        a gap raises it, and a floor measured against it therefore gets *harder*
        to satisfy the more room the hero makes -- it would charge the controller
        for the gap-opening that wave damping consists of.

        ``leader`` is the vehicle ahead in the hero's own lane.  Longitudinal
        control cannot influence it (only which vehicle it is, if a neighbour
        cuts into the gap), and it is local to the hero.  ``macro_lookahead`` is
        exogenous too, but the first cell past the front of the bubble sits at
        least ``margin_s`` ahead, so in stop-and-go it can describe traffic the
        hero is nowhere near.  ``follow_speed`` keeps the contaminated reference
        available as an ablation.
        """
        v_min = 0.1
        reference = self.reward_config.progress_reference
        if (reference == "follow_speed"):
            return max(v_min, float(self._get_follow_speed()))
        if (reference == "macro_lookahead"):
            return max(v_min, float(self._macro_lookahead(self._hero()["lane_id"])[0][1]))
        leader, _ = self._leader_and_follower()
        if (leader is None):
            return max(v_min, float(self.fd.v_f))
        return max(v_min, float(leader["velocity"]))

    def _reference_speed(self) -> float:
        """``_instantaneous_reference_speed`` averaged over the recent past.

        Read-only: the history is appended once per step by ``step``, so calling
        this twice in a step -- or from a diagnostic -- cannot shift the value.
        The averaging is what makes the floor mean "am I keeping up with the
        traffic around here" rather than "am I matching the vehicle in front at
        this instant", which in stop-and-go is a different and much jumpier
        question: a leader accelerating out of a wave leaves the hero at a small
        fraction of its speed for several seconds through no fault of its own.
        """
        if (len(self._reference_speed_history) == 0):
            return self._instantaneous_reference_speed()
        return max(0.1, float(np.mean(self._reference_speed_history)))

    def _reference_speed_window(self) -> int:
        """Length of the reference-speed history, in macro steps."""
        window = self.reward_config.progress_reference_window
        if (window <= 0.0):
            return 1
        return max(1, int(round(window / max(1e-9, self._macro_dt))))

    def _density_snapshot(self) -> Dict[Tuple[str, str], float]:
        """Every base cell's macroscopic density, keyed the way the mask's
        ``base_segments`` and the empirical snapshot are keyed."""
        snapshot: Dict[Tuple[str, str], float] = {}
        for road in self.sim.network.roads.values():
            for cell_id, cell in road.cells.items():
                length = cell.length if (cell.macro_length is None) else cell.macro_length
                if (length > 1e-9):
                    snapshot[(road.road_id, cell_id)] = float(cell.mass / length)
        return snapshot

    def _flux_curvature(
        self,
        current: Optional[Dict[int, float]],
        previous: Optional[Dict[int, float]],
        previous_2: Optional[Dict[int, float]],
    ) -> Optional[float]:
        """Mean |second difference| of the per-lane rear flux, in units of capacity.

        Clamped to [0, 1] so that differencing two of these stays in [-1, 1] like
        every other component.  ``None`` when there is not enough history.
        """
        if ((current is None) or (previous is None) or (previous_2 is None)):
            return None
        total = 0.0
        for lane in current:
            second_difference = current[lane] - (2.0 * previous[lane]) + previous_2[lane]
            total += abs(second_difference / self.fd.capacity)
        return min(1.0, (total / float(len(current))))

    def _baseline_pass(
        self, spec: EpisodeSpec
    ) -> Tuple[List[Dict[Tuple[str, str], float]], List[Dict[int, float]]]:
        """Replay this episode with the hero left to SUMO's car-following model.

        This is the counterfactual the upstream terms are scored against: what
        the road would have done over this exact episode had the controller not
        acted.  It is deterministic given the spec -- ``_build_episode`` reseeds
        the global RNG the spawn stream draws from -- so the policy pass sees the
        same traffic the baseline did, and the difference is attributable.

        Builds and tears down its own episode, so it must run before the pass
        that will be scored.  That is one extra environment pass per episode; on
        a distribution where specs essentially never repeat, it is a flat 2x on
        environment time.

        Returns base-cell densities and per-lane rear fluxes.  Element 0 of each
        is the warm-up step, so element ``k`` is the state after ``k`` scored
        steps -- which lines the fluxes up with the scored pass, whose own
        ``_previous_rear_flux`` is likewise seeded from its warm-up.
        """
        maps: List[Dict[Tuple[str, str], float]] = []
        fluxes: List[Dict[int, float]] = []
        self._build_episode(spec)
        try:
            self.sim.step()                      # the warm-up step the scored pass also takes
            maps.append(self._density_snapshot())
            fluxes.append(self._rear_fluxes())
            while (len(maps) <= self.env_config.max_steps):
                if ((not self.bridge.running) or self._retires_next_step()):
                    break
                self.policy_shim.set_acceleration(None)   # None == hand back to SUMO
                self.sim.step()
                maps.append(self._density_snapshot())
                fluxes.append(self._rear_fluxes())
        finally:
            self.close()
        return maps, fluxes

    def _rear_fluxes(self) -> Dict[int, float]:
        result = {}
        for lane in self.env_config.lanes:
            mask: I24MicroMask = self.sim.masking_cells[self.bridge._mask_id(lane)]
            result[lane] = mask.current_rear_flux
        return result

    def _neighbours_in_lane(self, lane: int) -> List[Dict[str, Any]]:
        return list(self.coupler.visible_state.get(lane, {}).values())

    def _leader_and_follower(self) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Nearest vehicle ahead of and behind the hero in its own lane.

        Read out of the coupler's visible state rather than out of TraCI so the
        same code works before the engine is up and for any micro engine; ``s``
        is the rear bumper on both sides, as everywhere else in the coupler.
        """
        hero = self._hero()
        leader = None
        follower = None
        for vehicle in self._neighbours_in_lane(hero["lane_id"]):
            if (vehicle["s"] > hero["s"]):
                if ((leader is None) or (vehicle["s"] < leader["s"])):
                    leader = vehicle
            elif (vehicle["s"] < hero["s"]):
                if ((follower is None) or (vehicle["s"] > follower["s"])):
                    follower = vehicle
        return leader, follower

    def _gap_to(self, hero: Dict[str, Any], leader: Optional[Dict[str, Any]]) -> float:
        if (leader is None):
            return float("inf")
        return float(leader["s"] - (hero["s"] + hero["length"]))

    def _macro_lookahead(self, lane: int) -> List[Tuple[float, float]]:
        """Density and velocity of the cells downstream of the mask, in order.

        This is the hero's over-the-horizon view: the CTM cells past the front of
        the bubble, which no microscopic sensor could reach.  Short chains are
        padded with their last entry so the observation keeps a fixed width.
        """
        count = self.env_config.macro_lookahead_cells
        profile: List[Tuple[float, float]] = []
        try:
            cell = self.sim.active.get_cell_with_mask(self.bridge._mask_id(lane))
            while ((cell is not None) and (len(profile) < count)):
                neighbours = cell.outflow_neighbors
                if (len(neighbours) == 0):
                    break
                cell = self.sim.active.active_cells[neighbours[0]]
                if (cell.kind == "mask"):
                    break
                density = float(cell.density)
                velocity = (
                    float(cell.velocity)
                    if (cell.velocity is not None)
                    else float(cell.fd.velocity_from_density(density))
                )
                profile.append((density, velocity))
        except Exception:
            pass
        if (len(profile) == 0):
            profile.append((self.fd.rho_c, self.fd.v_f))
        while (len(profile) < count):
            profile.append(profile[-1])
        return profile[:count]

    def _upstream_profiles(
        self, lane: int
    ) -> Optional[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]:
        """Simulated and empirical upstream state over *identical* base cells.

        This is the hero's behind-the-horizon view: the CTM cells upstream of
        the bubble, which no microscopic sensor could reach.  The simulated and
        empirical profiles are built in a single walk, entry for entry over the
        same ``(road_id, cell_id)`` keys, so the two cannot drift apart.  Built
        separately they did: one enumerated per *active* cell and the other per
        *base segment*, so the reward differenced two different stretches of
        road and scored a penalty even where the simulation matched the data.

        Base cells the mask covers only partially are skipped.  Their
        macroscopic mass is spread over ``macro_length`` rather than the whole
        cell -- the rest of it is held by the mask -- so their density is not
        comparable with the empirical snapshot, which is always a whole-cell
        value.

        Velocity comes from the fundamental diagram on both sides, since the
        empirical snapshot carries density only.  Under the first-order CTM
        ``ActiveCell.velocity`` is ``None`` anyway.  Note the consequence: the
        triangular FD is flat at ``v_f`` below ``rho_c``, so both profiles read
        exactly ``v_f`` in free flow and every velocity-based term differences
        to zero there.  That is deliberate -- there is no wave to damp -- but it
        means these profiles say nothing about free-flowing traffic, and a term
        that should stay live there has to use the densities instead.
        """
        count = self.env_config.macro_lookbehind_cells
        if (self.reward_config.upstream_reference == "baseline"):
            # ``step`` increments _step_index immediately after sim.step(), before
            # scoring, so the snapshot for the step being scored is one back.
            # Element 0 of the counterfactual is its warm-up, so the state after
            # the scored step (_step_index having already been incremented) is
            # element _step_index.
            if (self._step_index >= len(self._baseline_density_maps)):
                # The policy outlasted the counterfactual, so there is nothing to
                # compare against.  Score neutral rather than inventing a reference.
                return None
            empirical_snapshot = self._baseline_density_maps[self._step_index]
        else:
            empirical_snapshot = self.gt.get_empirical_densities_at_time(self.sim.current_time)
        sim_profile: List[Tuple[float, float]] = []
        empirical_profile: List[Tuple[float, float]] = []
        cell = self.sim.active.get_cell_with_mask(self.bridge._mask_id(lane))
        while ((cell is not None) and (len(sim_profile) < count)):
            neighbours = cell.inflow_neighbors
            if (len(neighbours) == 0):
                break
            cell = self.sim.active.active_cells[neighbours[0]]
            if (cell.kind == "mask"):
                break
            fd = cell.fd if (cell.fd is not None) else self.fd
            for ((road_id, cell_id), _, _) in cell.base_segments[::-1]:
                if (len(sim_profile) >= count):
                    break
                base_cell = self.sim.network.get_cell(road_id, cell_id)
                # None means "never masked", so the mass covers the whole cell.
                macro_length = (
                    base_cell.length if (base_cell.macro_length is None) else base_cell.macro_length
                )
                if (macro_length < (base_cell.length - 1e-9)):
                    continue
                empirical_density = empirical_snapshot.get((road_id, cell_id))
                if (empirical_density is None):
                    continue
                sim_density = float(base_cell.mass / macro_length)
                empirical_density = float(empirical_density)
                sim_profile.append((sim_density, float(fd.velocity_from_density(sim_density))))
                empirical_profile.append(
                    (empirical_density, float(fd.velocity_from_density(empirical_density)))
                )
        if (len(sim_profile) <= 1):
            return None
        return sim_profile[::-1], empirical_profile[::-1]

    @staticmethod
    def _profile_oscillation(profile: List[Tuple[float, float]], v_f: float) -> float:
        """Mean absolute velocity step between neighbouring cells, in units of v_f."""
        if (len(profile) < 2):
            return 0.0
        return sum(
            abs((profile[i + 1][1] - profile[i][1]) / v_f) for i in range(len(profile) - 1)
        ) / (len(profile) - 1)

    @staticmethod
    def _profile_delay(profile: List[Tuple[float, float]], v_f: float) -> float:
        """Mean shortfall below free flow, in units of v_f.

        One-sided: a cell running above ``v_f`` is not delayed, and scoring it
        as though it were (which ``abs`` did) charges free-flowing traffic for
        the gap between the environment's ``v_f`` and the cell's own.
        """
        if (len(profile) == 0):
            return 0.0
        return sum(max(0.0, (1.0 - (velocity / v_f))) for _, velocity in profile) / len(profile)

    def _macro_behind(self, lane: int) -> Tuple[float, float]:
        try:
            density = float(self.coupler.get_behind_lane_density(lane, self.fd.rho_c))
            velocity = float(self.coupler.get_behind_lane_velocity_macro(lane, self.fd.v_f))
        except Exception:
            density, velocity = self.fd.rho_c, self.fd.v_f
        return density, velocity

    def _platoon(self) -> List[Dict[str, Any]]:
        """The vehicles whose smoothness this controller is answerable for.

        A longitudinal controller only acts on the vehicles queued behind it in
        its own lane; crediting it with the speed of a neighbouring lane it
        cannot touch is just noise in the reward.  The wider scopes are there for
        ablations.
        """
        hero = self._hero()
        scope = self.reward_config.platoon_scope
        window = self.env_config.platoon_window
        members: List[Dict[str, Any]] = []
        lanes = [hero["lane_id"]] if (scope == "lane_behind") else list(self.env_config.lanes)
        for lane in lanes:
            for vehicle in self._neighbours_in_lane(lane):
                behind = (vehicle["s"] < hero["s"]) and ((hero["s"] - vehicle["s"]) <= window)
                if ((scope == "bubble") or behind):
                    members.append(vehicle)
        return members

    def _observation(self) -> np.ndarray:
        hero = self._hero()
        v_f = self.fd.v_f
        rho_j = self.fd.rho_j
        speed = float(hero["velocity"])
        leader, follower = self._leader_and_follower()
        leader_gap = self._gap_to(hero, leader)
        follower_gap = (
            float("inf") if (follower is None) else float(hero["s"] - (follower["s"] + follower["length"]))
        )
        gap_scale = self.env_config.visible_window
        headway = (leader_gap / speed) if (speed > 0.1) else 10.0

        platoon = self._platoon()
        platoon_speeds = np.asarray([speed] + [float(v["velocity"]) for v in platoon], dtype=float)
        ahead = [v for v in self._neighbours_in_lane(hero["lane_id"]) if (v["s"] > hero["s"])]

        lookahead = self._macro_lookahead(hero["lane_id"])
        behind_density, behind_velocity = self._macro_behind(hero["lane_id"])
        progress = (float(hero["s"]) - self.env_config.initial_middle_s) / max(
            1.0, (self.env_config.max_middle_s - self.env_config.initial_middle_s)
        )

        features = [
            speed / v_f,
            self._previous_acceleration / self.env_config.max_acceleration,
            min(leader_gap, gap_scale) / gap_scale,
            0.0 if (leader is None) else ((float(leader["velocity"]) - speed) / v_f),
            min(headway, 10.0) / 10.0,
            min(follower_gap, gap_scale) / gap_scale,
            0.0 if (follower is None) else ((float(follower["velocity"]) - speed) / v_f),
            float(np.mean(platoon_speeds)) / v_f,
            float(np.std(platoon_speeds)) / v_f,
            min(len(platoon), 20) / 20.0,
            min(len(ahead), 20) / 20.0,
            behind_density / rho_j,
            behind_velocity / v_f,
            self._previous_action,
            progress,
        ]
        for density, velocity in lookahead:
            features.append(density / rho_j)
            features.append(velocity / v_f)
        observation = np.asarray(features, dtype=np.float32)
        return np.clip(observation, self.observation_space.low, self.observation_space.high)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _substep_metrics(self, samples: Sequence[Dict[str, float]], final_speed: float) -> Dict[str, float]:
        """Realised acceleration, jerk, energy and minimum headway over one step.

        The policy shim logs the hero's speed before each of the ten substeps, so
        differencing that trace (closed with the speed the step ended at) gives
        the accelerations SUMO actually applied, which is generally not the
        commanded one -- it clips to the safe speed and to its own bounds.
        """
        step_length = self.env_config.step_length
        speeds = [float(s["speed"]) for s in samples] + [float(final_speed)]
        accelerations = [(speeds[i + 1] - speeds[i]) / step_length for i in range(len(speeds) - 1)]
        if (len(accelerations) == 0):
            accelerations = [0.0]
        energy = 0.0
        distance = 0.0
        for i, acceleration in enumerate(accelerations):
            speed = 0.5 * (speeds[i] + speeds[i + 1])
            energy += self.energy_model.power(speed, acceleration) * step_length
            distance += speed * step_length
        headways = []
        for sample in samples:
            gap = float(sample["leader_gap"])
            speed = float(sample["speed"])
            headways.append((gap / speed) if ((speed > 0.1) and math.isfinite(gap)) else 10.0)
        return {
            "acceleration": float(np.mean(accelerations)),
            "acceleration_rms": float(np.sqrt(np.mean(np.square(accelerations)))),
            "jerk_rms": (
                float(np.sqrt(np.mean(np.square(np.diff(accelerations)))) / step_length)
                if (len(accelerations) > 1) else 0.0
            ),
            "energy": float(energy),
            "distance": float(distance),
            "min_headway": float(min(headways)) if (len(headways) > 0) else 10.0,
        }

    def _reward(self, metrics: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        weights = self.reward_config
        v_f = self.fd.v_f
        speed = float(self._hero()["velocity"])
        platoon_speeds = np.asarray(
            [speed] + [float(v["velocity"]) for v in self._platoon()], dtype=float
        )

        jerk = metrics["acceleration"] - self._previous_acceleration
        shortfall = max(0.0, 1.0 - (metrics["min_headway"] / weights.target_headway))
        energy_per_metre = metrics["energy"] / max(1.0, metrics["distance"])

        # Rear-boundary flux, scored on the *second* difference rather than the
        # first.  The mask's rear boundary is anchored on the hero, so a hero that
        # decelerates at a constant rate ramps the flux through it at a constant
        # rate too: measured through a hard braking ramp, |dq| held at ~0.0175 per
        # step while |d2q| stayed near 0.0003, and the ramp alone accounted for
        # half the episode's total variation.  Charging |dq| therefore charges the
        # controller for decelerating -- the very thing wave damping consists of --
        # under the name of smoothness.  The second difference is blind to that
        # ramp and still sees the reversals a stop-and-go wave crossing the
        # boundary actually makes.
        #
        # Scored as a penalty rather than as ``1 - penalty``: the constant did
        # nothing for the gradient and was the largest single source of the
        # unconditional per-step reward, which gave the agent a stake in dragging
        # episodes out.  Zero now means "the flux was smooth", not "a step happened".
        # Under "baseline" the curvature is differenced against the same episode
        # driven by SUMO's own model, so the term reads "smoother than doing
        # nothing" rather than "smooth in absolute terms".  Absolute smoothness is
        # minimised by driving smoothly at the prevailing speed -- which is what
        # IDM does, and is exactly what the policy collapsed onto when this term
        # carried 62% of the cost with no counterfactual to beat.
        rear_fluxes = self._rear_fluxes()
        policy_curvature = self._flux_curvature(
            rear_fluxes, self._previous_rear_flux, self._previous_rear_flux_2
        )
        if (self.reward_config.upstream_reference != "baseline"):
            rear_flux_smoothness = 0.0 if (policy_curvature is None) else -policy_curvature
        else:
            index = self._step_index - 1
            baseline_fluxes = self._baseline_rear_fluxes
            baseline_curvature = (
                self._flux_curvature(
                    baseline_fluxes[index + 1], baseline_fluxes[index], baseline_fluxes[index - 1]
                )
                if ((index >= 1) and ((index + 1) < len(baseline_fluxes))) else None
            )
            rear_flux_smoothness = (
                0.0 if ((policy_curvature is None) or (baseline_curvature is None))
                else (baseline_curvature - policy_curvature)
            )

        # Upstream Calculations
        upstream_sim_oscillation = {}
        upstream_empirical_oscillation = {}
        upstream_sim_delay = {}
        upstream_empirical_delay = {}
        for lane in self.env_config.lanes:
            profiles = self._upstream_profiles(lane)
            if (profiles is None):
                # Not enough comparable upstream road yet -- near the start of the
                # corridor there may be only a cell or two behind the mask.  Score
                # both sides identically so the difference is exactly zero rather
                # than a bias in either direction.
                upstream_sim_oscillation[lane] = 1.0
                upstream_sim_delay[lane] = 1.0
                upstream_empirical_oscillation[lane] = 1.0
                upstream_empirical_delay[lane] = 1.0
                continue
            sim_profile, empirical_profile = profiles
            # v_f is only a scale here, and the same one on both sides, so it
            # cancels in the sim-minus-empirical difference below.
            upstream_sim_oscillation[lane] = self._profile_oscillation(sim_profile, self.fd.v_f)
            upstream_empirical_oscillation[lane] = self._profile_oscillation(
                empirical_profile, self.fd.v_f
            )
            upstream_sim_delay[lane] = self._profile_delay(sim_profile, self.fd.v_f)
            upstream_empirical_delay[lane] = self._profile_delay(empirical_profile, self.fd.v_f)

        upstream_oscillation_sim_reward = (1.0 - (sum([upstream_sim_oscillation[lane] for lane in upstream_sim_oscillation]) / len(upstream_sim_oscillation)))
        upstream_oscillation_empirical_reward = (1.0 - (sum([upstream_empirical_oscillation[lane] for lane in upstream_empirical_oscillation]) / len(upstream_empirical_oscillation))) 
        upstream_oscillation_reward = upstream_oscillation_sim_reward - upstream_oscillation_empirical_reward

        upstream_delay_sim_reward = (1.0 - (sum([upstream_sim_delay[lane] for lane in upstream_sim_delay]) / len(upstream_sim_delay)))
        upstream_delay_empirical_reward = (1.0 - (sum([upstream_empirical_delay[lane] for lane in upstream_empirical_delay]) / len(upstream_empirical_delay)))
        upstream_delay_reward = upstream_delay_sim_reward - upstream_delay_empirical_reward
        #print(f"Oscillation {upstream_oscillation_reward}, Delay {upstream_delay_reward}")

        # Progress, as a one-sided constraint rather than an objective.  Rewarding
        # speed/reference directly is identically "constant minus a penalty on
        # slowing", so it opposes wave damping at every step -- and since the hero
        # cannot exceed the safe speed, the term can only ever be spent, never
        # earned.  The floor form is flat wherever the hero is keeping up.
        floor = weights.progress_floor
        reference_speed = self._reference_speed()
        # If the traffic the hero is measured against has itself stopped, the floor
        # is vacuous: the hero being slow is not the hero stalling.  Scoring a ratio
        # of 1.0 leaves the term at exactly zero.
        progress_ratio = (
            (speed / reference_speed) if (reference_speed > weights.stopped_speed) else 1.0
        )
        progress = -((max(0.0, (floor - progress_ratio)) / floor))

        components = {
            "progress": progress,
            "rear_flux_smoothness": rear_flux_smoothness,
            "upstream_oscillation": upstream_oscillation_reward,
            "upstream_delay": upstream_delay_reward,
            "platoon_speed": float(np.mean(platoon_speeds)) / v_f,
            "speed_variance": -float(np.std(platoon_speeds)) / v_f,
            "acceleration": -((metrics["acceleration_rms"] / self.env_config.max_acceleration) ** 2),
            "jerk": -((jerk / self.env_config.max_acceleration) ** 2),
            "headway": -(shortfall ** 2),
            "energy": -(energy_per_metre / weights.energy_scale),
            "stopped": -1.0 if (speed < weights.stopped_speed) else 0.0,
            "collision": 0.0,
        }
        total = sum(components[name] * getattr(weights, name) for name in components)
        return float(total), components

    def _detect_collision(self) -> bool:
        """Whether the hero was involved in a collision on the last step.

        SUMO runs with ``--collision.action warn``, so a collision leaves the
        vehicles in place and only shows up in this list.  With the hero's
        ``setSpeed`` under the default speed mode it should never happen; it is
        checked because a controller that found a way to make it happen would
        otherwise be rewarded for it.
        """
        try:
            colliding = set(self.coupler.sumo_sim.conn.simulation.getCollidingVehiclesIDList())
        except Exception:
            return False
        return self.coupler.sumo_sim.hero_state["sumo_id"] in colliding

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def action_to_acceleration(self, action: float) -> float:
        action = float(np.clip(action, -1.0, 1.0))
        limit = self.env_config.max_acceleration if (action >= 0.0) else self.env_config.max_deceleration
        return action * limit

    def _retires_next_step(self) -> bool:
        """Whether the next macroscopic step would retire the bubble.

        Exactly predictable, and it has to be: ``I24MicroSimBridge._step`` sets
        ``middle_s = micro_coupler.step()``, and that returns the hero's ``s``
        from the *previous* poststep -- the value this environment is already
        holding.  Nothing between here and there moves the hero.

        Predicting it is what keeps the episode boundary honest.  Taking the
        retiring step instead would produce a transition whose observation is the
        previous one repeated and whose reward is a made-up zero, because the
        bridge tears the bubble down before the hero is advanced.  Under
        truncation SB3 adds ``gamma * V(next_obs)`` to that zero, and since
        ``next_obs`` is the same observation the transition started from, the
        critic gets regressed toward ``gamma * V(o)`` at its own input: a target
        that drags end-of-corridor values toward zero.  ``progress`` is in the
        observation, so the critic can identify precisely those states and the
        bias sticks -- reintroducing the "finishing the corridor is bad" pressure
        that bootstrapping truncated episodes exists to remove.
        """
        if ((self.bridge is None) or (self.coupler is None)):
            return True
        return float(self._hero()["s"]) >= self.env_config.max_middle_s

    def step(
        self, action, passthrough: bool = False
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance one macroscopic step under ``action``.

        The terminated/truncated distinction matters for the value bootstrap: the
        bubble retiring at ``max_middle_s`` is the corridor running out, not a
        failure, so those episodes are truncated and SB3 bootstraps their tail
        value rather than zeroing it.

        ``passthrough`` leaves the hero on SUMO's car-following model for this
        step and scores it anyway, which is how the uncontrolled baseline is
        measured on exactly the same reward as the learned controllers.
        """
        if ((self.bridge is None) or (not self.bridge.running)):
            raise RuntimeError("step() on a finished episode; call reset() first")

        action_value = float(np.asarray(action).reshape(-1)[0])
        acceleration = float("nan") if (passthrough) else self.action_to_acceleration(action_value)
        self.policy_shim.set_acceleration(None if (passthrough) else acceleration)
        self.sim.step()
        self._step_index += 1

        if (not self.bridge.running):
            # Defensive only: _retires_next_step() ends the episode before the
            # retiring step is ever taken, so this should be unreachable.  If the
            # bridge goes down for some other reason there is no new state to
            # score, and a zero-reward transition on a repeated observation would
            # be a fabricated one -- see the note there for why that matters.
            info: Dict[str, Any] = {
                "components": {}, "step": self._step_index, "terminal_reason": "bubble_lost",
            }
            info.update(self._episode_info("bubble_lost"))
            return self._last_observation, 0.0, False, True, info

        samples = self.policy_shim.drain_samples()
        hero = self._hero()
        metrics = self._substep_metrics(samples, float(hero["velocity"]))
        collided = self._detect_collision()
        # Append before scoring so this step's traffic is inside the window the
        # progress floor averages over.
        self._reference_speed_history.append(self._instantaneous_reference_speed())
        reward, components = self._reward(metrics)
        if (collided):
            components["collision"] = -1.0
            reward -= self.reward_config.collision

        self._previous_acceleration = metrics["acceleration"]
        self._previous_rear_flux_2 = self._previous_rear_flux
        self._previous_rear_flux = self._rear_fluxes()
        self._previous_action = 0.0 if (passthrough) else float(np.clip(action_value, -1.0, 1.0))
        lookahead = self._macro_lookahead(hero["lane_id"])
        record = {
            "step": self._step_index,
            "time": float(self.coupler.current_timestamp),
            "middle_s": float(self.bridge.middle_s),
            "hero_s": float(hero["s"]),
            "hero_speed": float(hero["velocity"]),
            "hero_lane": int(hero["lane_id"]),
            "action": self._previous_action,
            "commanded_acceleration": acceleration,
            "realised_acceleration": metrics["acceleration"],
            "acceleration_rms": metrics["acceleration_rms"],
            "jerk_rms": metrics["jerk_rms"],
            "energy": metrics["energy"],
            "distance": metrics["distance"],
            "min_headway": metrics["min_headway"],
            "platoon_size": len(self._platoon()),
            "platoon_mean_speed": components["platoon_speed"] * self.fd.v_f,
            "platoon_speed_std": -components["speed_variance"] * self.fd.v_f,
            "macro_density_ahead": lookahead[0][0],
            "macro_velocity_ahead": lookahead[0][1],
            "progress_reward": components["progress"],
            "rear_flux_smoothness": components["rear_flux_smoothness"],
            "upstream_oscillation": components["upstream_oscillation"],
            "upstream_delay": components["upstream_delay"],
            "reward": reward,
        }
        self._episode_records.append(record)
        self._last_observation = self._observation()

        # Both endings are truncations of an otherwise continuing process, so SB3
        # bootstraps their tail value; only a collision is a real termination.
        retiring = self._retires_next_step()
        truncated = retiring or (self._step_index >= self.env_config.max_steps)
        terminated = collided
        reason = (
            "collision" if (collided)
            else ("bubble_retired" if (retiring) else ("step_cap" if (truncated) else None))
        )
        info = {
            "components": components,
            "record": record,
            "step": self._step_index,
            "terminal_reason": reason,
        }
        if (terminated or truncated):
            info.update(self._episode_info(reason))
        return self._last_observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _episode_info(self, reason: Optional[str]) -> Dict[str, Any]:
        self._terminal_reason = reason or ""
        summary = self.episode_summary()
        # Monitor's info_keywords requires every declared key on the final step.
        return {key: summary.get(key, 0.0) for key in EPISODE_INFO_KEYS}

    def episode_summary(self) -> Dict[str, Any]:
        """Aggregate the finished episode into one row of per-episode metrics."""
        records = self._episode_records
        spec = self.spec
        summary: Dict[str, Any] = {
            "dataset": "" if (spec is None) else spec.dataset,
            "road": "" if (spec is None) else spec.road,
            "start_time": 0.0 if (spec is None) else spec.start_time,
            "terminal_reason": self._terminal_reason,
            "steps": len(records),
            "return": 0.0,
            "mean_reward": 0.0,
            "mean_progress_reward": 0.0,
            "mean_rear_flux_smoothness": 0.0,
            "mean_upstream_oscillation": 0.0,
            "mean_upstream_delay": 0.0,
            "hero_mean_speed": 0.0,
            "hero_speed_std": 0.0,
            "platoon_mean_speed": 0.0,
            "platoon_speed_std": 0.0,
            "acceleration_rms": 0.0,
            "jerk_rms": 0.0,
            "min_headway": 0.0,
            "distance": 0.0,
            "energy": 0.0,
            "energy_per_metre": 0.0,
        }
        if (len(records) == 0):
            return summary
        speeds = np.asarray([r["hero_speed"] for r in records], dtype=float)
        platoon = np.asarray([r["platoon_mean_speed"] for r in records], dtype=float)
        distance = float(np.sum([r["distance"] for r in records]))
        energy = float(np.sum([r["energy"] for r in records]))
        summary.update({
            "return": float(np.sum([r["reward"] for r in records])),
            "mean_reward": float(np.mean([r["reward"] for r in records])),
            "mean_progress_reward": float(np.mean([r["progress_reward"] for r in records])),
            "mean_rear_flux_smoothness": float(np.mean([r["rear_flux_smoothness"] for r in records])),
            "mean_upstream_oscillation": float(np.mean([r["upstream_oscillation"] for r in records])),
            "mean_upstream_delay": float(np.mean([r["upstream_delay"] for r in records])),
            "hero_mean_speed": float(np.mean(speeds)),
            "hero_speed_std": float(np.std(speeds)),
            "platoon_mean_speed": float(np.mean(platoon)),
            "platoon_speed_std": float(np.mean([r["platoon_speed_std"] for r in records])),
            "acceleration_rms": float(np.sqrt(np.mean(np.square([r["acceleration_rms"] for r in records])))),
            "jerk_rms": float(np.sqrt(np.mean(np.square([r["jerk_rms"] for r in records])))),
            "min_headway": float(np.min([r["min_headway"] for r in records])),
            "distance": distance,
            "energy": energy,
            "energy_per_metre": energy / max(1.0, distance),
        })
        return summary

    def episode_records(self) -> List[Dict[str, float]]:
        return list(self._episode_records)


# ---------------------------------------------------------------------------
# Non-learned controllers, for comparison
# ---------------------------------------------------------------------------


class BaselineController:
    """A controller with the same interface as a learned policy.

    ``act`` takes the environment's observation vector and returns an action in
    [-1, 1], or ``None`` for "hand the hero back to SUMO", which the environment
    turns into a passthrough step rather than into an acceleration.
    """

    name = "baseline"

    def reset(self) -> None:
        pass

    def act(self, observation: np.ndarray, env: "I24SumoHeroEnv") -> Optional[float]:
        raise NotImplementedError


class SumoBaseline(BaselineController):
    """The hero on SUMO's own car-following model -- the uncontrolled case."""

    name = "sumo"

    def act(self, observation: np.ndarray, env: "I24SumoHeroEnv") -> Optional[float]:
        return None


class IDMController(BaselineController):
    """Intelligent Driver Model on the hero's own leader.

    A microscopic controller with no view past its leader; the point of
    comparison for anything that uses the macroscopic lookahead.
    """

    name = "idm"

    def __init__(
        self,
        v_desired: float = 30.0,
        time_headway: float = 1.5,
        min_gap: float = 2.0,
        acceleration: float = 1.5,
        comfortable_deceleration: float = 2.0,
        delta: float = 4.0,
    ) -> None:
        self.v_desired = v_desired
        self.time_headway = time_headway
        self.min_gap = min_gap
        self.acceleration = acceleration
        self.comfortable_deceleration = comfortable_deceleration
        self.delta = delta

    def act(self, observation: np.ndarray, env: "I24SumoHeroEnv") -> Optional[float]:
        hero = env.coupler.hero_state
        speed = float(hero["velocity"])
        leader, _ = env._leader_and_follower()
        free = 1.0 - ((speed / max(1e-3, self.v_desired)) ** self.delta)
        interaction = 0.0
        if (leader is not None):
            gap = max(0.1, env._gap_to(hero, leader))
            approach = speed - float(leader["velocity"])
            desired_gap = (
                self.min_gap
                + (speed * self.time_headway)
                + ((speed * approach) / (2.0 * math.sqrt(self.acceleration * self.comfortable_deceleration)))
            )
            interaction = (max(0.0, desired_gap) / gap) ** 2
        return _acceleration_to_action(self.acceleration * (free - interaction), env)


class FollowerStopperController(BaselineController):
    """Speed-limited following in the spirit of the CIRCLES field controllers.

    Tracks a command speed that decays toward the leader's speed as the gap
    closes, which is the classic wave-damping heuristic this environment's
    smoothing reward is meant to capture.
    """

    name = "follower_stopper"

    def __init__(
        self,
        desired_speed: float = 25.0,
        boundaries: Tuple[float, float, float] = (4.5, 5.25, 6.0),
        decelerations: Tuple[float, float] = (1.5, 1.0),
        gain: float = 0.6,
    ) -> None:
        self.desired_speed = desired_speed
        self.boundaries = boundaries
        self.decelerations = decelerations
        self.gain = gain

    def act(self, observation: np.ndarray, env: "I24SumoHeroEnv") -> Optional[float]:
        hero = env.coupler.hero_state
        speed = float(hero["velocity"])
        leader, _ = env._leader_and_follower()
        if (leader is None):
            return _acceleration_to_action(self.gain * (self.desired_speed - speed), env)
        gap = max(0.1, env._gap_to(hero, leader))
        leader_speed = float(leader["velocity"])
        approach = min(0.0, leader_speed - speed)
        d1, d2 = self.decelerations
        x1, x2, x3 = self.boundaries
        boundary_1 = x1 + ((approach ** 2) / (2.0 * d1))
        boundary_2 = x2 + ((approach ** 2) / (2.0 * d2))
        boundary_3 = x3 + ((approach ** 2) / (2.0 * d2))
        reference = min(max(leader_speed, 0.0), self.desired_speed)
        if (gap <= boundary_1):
            command = 0.0
        elif (gap <= boundary_2):
            command = reference * ((gap - boundary_1) / max(1e-6, (boundary_2 - boundary_1)))
        elif (gap <= boundary_3):
            command = reference + (
                (self.desired_speed - reference) * ((gap - boundary_2) / max(1e-6, (boundary_3 - boundary_2)))
            )
        else:
            command = self.desired_speed
        return _acceleration_to_action(self.gain * (command - speed), env)


class MacroLookaheadController(BaselineController):
    """Hand-written use of the macroscopic downstream state.

    Tracks the equilibrium speed of the CTM cells ahead of the bubble, so a
    learned policy has to beat something that already reads the same lookahead
    the environment hands it.
    """

    name = "macro_lookahead"

    def __init__(self, gain: float = 0.5, safety_headway: float = 1.5) -> None:
        self.gain = gain
        self.safety_headway = safety_headway

    def act(self, observation: np.ndarray, env: "I24SumoHeroEnv") -> Optional[float]:
        hero = env.coupler.hero_state
        speed = float(hero["velocity"])
        target = float(np.mean([velocity for _, velocity in env._macro_lookahead(hero["lane_id"])]))
        leader, _ = env._leader_and_follower()
        if (leader is not None):
            target = min(target, max(0.1, env._gap_to(hero, leader)) / self.safety_headway)
        return _acceleration_to_action(self.gain * (target - speed), env)


def _acceleration_to_action(acceleration: float, env: "I24SumoHeroEnv") -> float:
    limit = env.env_config.max_acceleration if (acceleration >= 0.0) else env.env_config.max_deceleration
    return float(np.clip(acceleration / limit, -1.0, 1.0))


BASELINE_CONTROLLERS = {
    SumoBaseline.name: SumoBaseline,
    IDMController.name: IDMController,
    FollowerStopperController.name: FollowerStopperController,
    MacroLookaheadController.name: MacroLookaheadController,
}


def run_baseline_episode(
    env: I24SumoHeroEnv, controller: BaselineController, spec: Optional[EpisodeSpec] = None
) -> Tuple[Dict[str, Any], List[Dict[str, float]]]:
    """One episode under a non-learned controller. Returns (summary, records)."""
    options = None if (spec is None) else {"spec": spec}
    observation, _ = env.reset(options=options)
    controller.reset()
    while True:
        action = controller.act(observation, env)
        observation, _, terminated, truncated, _ = env.step(
            0.0 if (action is None) else action, passthrough=(action is None)
        )
        if (terminated or truncated):
            break
    return env.episode_summary(), env.episode_records()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """PPO hyperparameters and the shape of the run.

    ``algorithm`` defaults to ``recurrent_ppo`` (sb3-contrib's LSTM policy).
    This environment is a POMDP: the hero sees its immediate leader and
    follower, a few macroscopic cells, and nothing of SUMO's internal per-vehicle
    state or of the empirical trajectories about to be injected at the bubble
    boundary.  A feedforward policy can only optimise over memoryless policies,
    whose optimum can be arbitrarily worse than a history-dependent one, and its
    critic regresses ``V(o)`` for an observation whose value genuinely depends on
    the belief state.  Carrying an LSTM hidden state addresses both.  ``ppo``
    remains selectable as the memoryless ablation.
    """

    algorithm: str = "recurrent_ppo"  # "recurrent_ppo" or "ppo"
    total_timesteps: int = 1_000_000
    envs: int = 24
    # Steps per environment per update.  An episode is about 40 steps, so 128
    # keeps two or three episodes' worth of experience per environment in each
    # batch without letting the policy go stale.
    n_steps: int = 128
    batch_size: int = 128*3
    n_epochs: int = 10
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.97
    clip_range: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.05
    net_arch: Tuple[int, ...] = (64, 64)
    log_std_init: float = -0.5
    # Generalised State-Dependent Exploration.  Default PPO perturbs the action
    # with fresh iid noise every step, which integrates to a random walk in speed
    # and averages out over a trajectory -- it explores jitter, not manoeuvres.
    # Wave damping is a manoeuvre: brake, hold, release, timed against an
    # approaching shock, and it has to be held long enough to pay off.  gSDE
    # draws the perturbation as a function of state and holds it for
    # sde_sample_freq steps, so exploration is temporally correlated and whole
    # manoeuvres get tried.  That matters here because the counterfactual reward
    # scores baseline imitation at exactly 0 while any deviation costs
    # immediately, so the useful behaviour is on the far side of a moat that iid
    # noise will not cross.
    use_sde: bool = True
    sde_sample_freq: int = 8      # -1 samples once per rollout; 8-16 is a good range
    # Recurrent policy only. A separate critic LSTM (shared_lstm=False,
    # enable_critic_lstm=True) is sb3-contrib's default and the right one here:
    # the value function has to track the platoon's state over time, which is a
    # different summary of the history than the policy's.  The two are mutually
    # exclusive -- shared_lstm=True requires enable_critic_lstm=False.
    lstm_hidden_size: int = 128
    n_lstm_layers: int = 1
    shared_lstm: bool = False
    enable_critic_lstm: bool = True
    seed: int = 0
    # Checkpoint every this many environment steps (summed over environments).
    checkpoint_freq: int = 10_000
    # Periodic greedy evaluation.  Each evaluation episode costs a full bubble
    # traversal, so this is off unless asked for.
    eval_freq: int = 0
    eval_episodes: int = 5
    tensorboard: bool = False
    normalize: bool = False


def make_env(
    rank: int,
    env_config: EnvConfig,
    reward_config: RewardConfig,
    seed: int,
    monitor_dir: Optional[str] = None,
):
    """Factory for one monitored environment, for Dummy/SubprocVecEnv."""

    def _init():
        from stable_baselines3.common.monitor import Monitor

        env = I24SumoHeroEnv(
            env_config=env_config,
            reward_config=reward_config,
            seed=seed + rank,
            label_prefix=f"rl{rank}",
        )
        filename = None if (monitor_dir is None) else os.path.join(monitor_dir, f"monitor_{rank}")
        return Monitor(env, filename=filename, info_keywords=EPISODE_INFO_KEYS)

    return _init


def build_vec_env(
    env_config: EnvConfig,
    reward_config: RewardConfig,
    train_config: TrainConfig,
    output_dir: Optional[str],
    subprocess: bool = True,
):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    factories = [
        make_env(rank, env_config, reward_config, train_config.seed, output_dir)
        for rank in range(max(1, train_config.envs))
    ]
    if ((not subprocess) or (train_config.envs <= 1)):
        return DummyVecEnv(factories)
    # Spawn, not fork: a forked worker would inherit the parent's TraCI sockets
    # and pandas state, and SUMO does not survive that.
    return SubprocVecEnv(factories, start_method="spawn")


def _episode_metrics_callback():
    """Callback class that mirrors per-episode metrics into SB3's logger.

    Monitor already writes every episode to ``monitor_*.csv``; this puts the
    rolling means into ``progress.csv`` and tensorboard so a run can be watched
    while it trains.  Defined inside a function so that importing this module
    does not require stable-baselines3.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class EpisodeMetricsCallback(BaseCallback):
        tracked = (
            "hero_mean_speed", "platoon_mean_speed", "platoon_speed_std",
            "acceleration_rms", "jerk_rms", "min_headway", "energy_per_metre",
            "mean_progress_reward", "mean_rear_flux_smoothness", "mean_upstream_oscillation", "mean_upstream_delay"
        )

        def __init__(self, window: int = 20) -> None:
            super().__init__()
            self.window = int(window)
            self.history: Dict[str, List[float]] = {key: [] for key in self.tracked}
            self.terminal_reasons: List[str] = []

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                if ("episode" not in info):
                    continue  # Monitor only adds this on the final step.
                for key in self.tracked:
                    if (key in info):
                        self.history[key].append(float(info[key]))
                        self.history[key] = self.history[key][-self.window:]
                self.terminal_reasons.append(str(info.get("terminal_reason", "")))
                self.terminal_reasons = self.terminal_reasons[-self.window:]
            return True

        def _on_rollout_end(self) -> None:
            for key, values in self.history.items():
                if (len(values) > 0):
                    self.logger.record(f"episode/{key}", float(np.mean(values)))
            if (len(self.terminal_reasons) > 0):
                retired = sum(1 for reason in self.terminal_reasons if (reason == "bubble_retired"))
                self.logger.record("episode/fraction_completed", retired / len(self.terminal_reasons))

    return EpisodeMetricsCallback


def write_run_config(
    output_dir: str,
    env_config: EnvConfig,
    reward_config: RewardConfig,
    train_config: TrainConfig,
    resumed_from: Optional[str] = None,
    resumed_at_step: int = 0,
) -> None:
    """Record what this run was, so the demo and the analysis can rebuild it.

    ``resumed_from`` matters for more than provenance: a resumed run's
    ``monitor_*.csv`` restart their ``t`` at zero, so any analysis that pools
    episodes across both directories has to concatenate by run rather than sort
    by ``t``, and the step axis of this directory starts at ``resumed_at_step``.
    """
    with open(os.path.join(output_dir, "run_config.json"), "w") as handle:
        json.dump(
            {
                "env": asdict(env_config),
                "reward": asdict(reward_config),
                "train": asdict(train_config),
                "observation_size": env_config.observation_size,
                "created_at": time.time(),
                "resumed_from": resumed_from,
                "resumed_at_step": int(resumed_at_step),
            },
            handle,
            indent=2,
            default=str,
        )


def train(
    env_config: EnvConfig,
    reward_config: RewardConfig,
    train_config: TrainConfig,
    output_dir: str,
    subprocess: bool = True,
    resume_from: Optional[str] = None,
):
    """Run PPO (recurrent by default) over the environment; returns the model.

    ``resume_from`` continues a stopped run from one of its checkpoints. The
    checkpoint carries ``policy.optimizer.pth`` as well as the weights, so the
    Adam moments survive and this is a continuation rather than a warm restart.
    ``total_timesteps`` stays the *total* for the run: the remaining budget is
    computed from the checkpoint's own step count.

    Resuming always writes into a fresh ``output_dir``. Sharing the original
    would overwrite its ``monitor_*.csv`` and its ``progress.csv``, and the two
    cannot be pooled by sorting on ``t`` anyway because a new Monitor restarts
    that clock at zero.
    """
    import torch.nn as nn
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import VecNormalize

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    resume_checkpoint = None
    resume_step = 0
    if (resume_from is not None):
        resume_checkpoint = resolve_checkpoint(resume_from)
        resume_step = checkpoint_step_count(resume_checkpoint)
        if (os.path.abspath(os.path.dirname(resume_checkpoint)) == os.path.abspath(checkpoint_dir)):
            raise ValueError(
                "refusing to resume into the run being resumed from: pass a new --run-name, "
                "or the original run's monitor and progress files are overwritten"
            )
        # The spec stream is drawn from `seed + rank`, so reusing the original
        # seed would replay the exact episodes the run already trained on.
        train_config = replace(train_config, seed=train_config.seed + resume_step)

    write_run_config(
        output_dir, env_config, reward_config, train_config,
        resumed_from=resume_checkpoint, resumed_at_step=resume_step,
    )

    vec_env = build_vec_env(env_config, reward_config, train_config, output_dir, subprocess)
    if (train_config.normalize):
        # Observations are already scaled by v_f / rho_j by hand, so this is
        # about the return scale; the observation statistics stay off.
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=train_config.gamma)

    formats = ["stdout", "csv"]
    if (train_config.tensorboard):
        try:
            import tensorboard  # noqa: F401

            formats.append("tensorboard")
        except ImportError:
            print("WARNING: tensorboard is not installed; logging to CSV only.")
    logger = configure(output_dir, formats)

    algorithm, policy_name = algorithm_and_policy(train_config.algorithm)
    policy_kwargs: Dict[str, Any] = {
        "net_arch": {"pi": list(train_config.net_arch), "vf": list(train_config.net_arch)},
        "activation_fn": nn.Tanh,
        "log_std_init": train_config.log_std_init,
    }
    if (is_recurrent(train_config.algorithm)):
        if (train_config.shared_lstm and train_config.enable_critic_lstm):
            raise ValueError("shared_lstm and enable_critic_lstm are mutually exclusive")
        policy_kwargs.update({
            "lstm_hidden_size": train_config.lstm_hidden_size,
            "n_lstm_layers": train_config.n_lstm_layers,
            "shared_lstm": train_config.shared_lstm,
            "enable_critic_lstm": train_config.enable_critic_lstm,
        })

    if (resume_checkpoint is not None):
        # Hyperparameters come from the checkpoint, not from train_config: the
        # saved optimizer state belongs to the network the checkpoint holds, and
        # silently rebuilding it under different settings would not be a resume.
        model = algorithm.load(resume_checkpoint, env=vec_env, device="auto")
        print(
            f"resumed {train_config.algorithm} from {resume_checkpoint} "
            f"at {model.num_timesteps} steps (env seed offset to {train_config.seed})"
        )
    else:
        model = algorithm(
            policy_name,
            vec_env,
            learning_rate=train_config.learning_rate,
            n_steps=train_config.n_steps,
            batch_size=train_config.batch_size,
            n_epochs=train_config.n_epochs,
            gamma=train_config.gamma,
            gae_lambda=train_config.gae_lambda,
            clip_range=train_config.clip_range,
            ent_coef=train_config.ent_coef,
            vf_coef=train_config.vf_coef,
            max_grad_norm=train_config.max_grad_norm,
            target_kl=train_config.target_kl,
            use_sde=train_config.use_sde,
            sde_sample_freq=train_config.sde_sample_freq,
            seed=train_config.seed,
            policy_kwargs=policy_kwargs,
            verbose=0,
        )
    model.set_logger(logger)
    print(f"{train_config.algorithm} / {policy_name} over {max(1, train_config.envs)} environment(s)")

    callbacks: List[Any] = [_episode_metrics_callback()()]
    if (train_config.checkpoint_freq > 0):
        callbacks.append(
            CheckpointCallback(
                # CheckpointCallback counts steps per environment.
                save_freq=max(1, train_config.checkpoint_freq // max(1, train_config.envs)),
                save_path=checkpoint_dir,
                name_prefix="policy",
                save_vecnormalize=train_config.normalize,
            )
        )
    eval_env = None
    if (train_config.eval_freq > 0):
        eval_env = build_vec_env(
            env_config, reward_config, replace(train_config, envs=1), None, subprocess=False
        )
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=checkpoint_dir,
                log_path=output_dir,
                eval_freq=max(1, train_config.eval_freq // max(1, train_config.envs)),
                n_eval_episodes=train_config.eval_episodes,
                deterministic=True,
            )
        )

    # With reset_num_timesteps=False, SB3 adds the argument to the counter it
    # already holds, so this has to be the *remaining* budget rather than the
    # total (base_class._setup_learn: `total_timesteps += self.num_timesteps`).
    remaining = train_config.total_timesteps
    if (resume_checkpoint is not None):
        remaining = train_config.total_timesteps - model.num_timesteps
        if (remaining <= 0):
            raise ValueError(
                f"checkpoint is already at {model.num_timesteps} steps of a "
                f"{train_config.total_timesteps} budget; raise --total-timesteps to continue"
            )
        print(f"running {remaining} more steps to reach {train_config.total_timesteps}")

    try:
        model.learn(
            total_timesteps=remaining,
            callback=callbacks,
            progress_bar=False,
            reset_num_timesteps=(resume_checkpoint is None),
        )
    finally:
        model.save(os.path.join(checkpoint_dir, "policy_final"))
        if (train_config.normalize):
            vec_env.save(os.path.join(checkpoint_dir, "vecnormalize.pkl"))
        vec_env.close()
        if (eval_env is not None):
            eval_env.close()
    return model


# ---------------------------------------------------------------------------
# Loading a trained run back
# ---------------------------------------------------------------------------


def find_run_config(path: str) -> Optional[str]:
    """The run directory governing ``path``, or None if there is no config above it.

    ``path`` may be the run directory itself or anything inside it -- most
    usefully a checkpoint under ``checkpoints/``, so that evaluating one
    specific checkpoint still picks up the configuration it was trained under.
    Getting this wrong is not a small error: ``EnvConfig`` defaults differ from
    any real run's (``macro_lookahead_cells`` alone changes the observation
    width), so a caller that quietly falls back to defaults is evaluating the
    policy in an environment it never saw.
    """
    candidate = os.path.abspath(path)
    if (os.path.isfile(candidate)):
        candidate = os.path.dirname(candidate)
    while True:
        if (os.path.isfile(os.path.join(candidate, "run_config.json"))):
            return candidate
        parent = os.path.dirname(candidate)
        if (parent == candidate):
            return None
        candidate = parent


def load_run_config(run_dir: str) -> Tuple[EnvConfig, RewardConfig, Dict[str, Any]]:
    """Rebuild the configs a run was trained with from its run_config.json.

    Accepts the run directory or any path inside it; see ``find_run_config``.
    """
    resolved = find_run_config(run_dir)
    if (resolved is None):
        raise FileNotFoundError(f"no run_config.json at or above {run_dir}")
    with open(os.path.join(resolved, "run_config.json"), "r") as handle:
        payload = json.load(handle)
    return (
        env_config_from_dict(payload.get("env", {})),
        reward_config_from_dict(payload.get("reward", {})),
        payload,
    )


def env_config_from_dict(stored: Dict[str, Any], **overrides) -> EnvConfig:
    stored = dict(stored)
    stored["datasets"] = tuple(stored.get("datasets", ("2022-11-30.json",)))
    stored["roads"] = tuple(str(road) for road in stored.get("roads", ("2",)))
    stored["lanes"] = tuple(int(lane) for lane in stored.get("lanes", (-1, -2, -3, -4)))
    config = EnvConfig(**{k: v for k, v in stored.items() if (k in EnvConfig.__dataclass_fields__)})
    return replace(config, **overrides) if (overrides) else config


def reward_config_from_dict(stored: Dict[str, Any]) -> RewardConfig:
    stored = dict(stored)
    # ``hero_speed`` was the weight on speed/getFollowSpeed before that term became
    # the one-sided ``progress`` floor.  Carry the weight across so a run_config
    # written by an older checkpoint keeps the term switched on or off as it was --
    # but note the term itself now computes something different, so an old
    # checkpoint replayed under it is not being scored on the reward it trained on.
    if (("progress" not in stored) and ("hero_speed" in stored)):
        stored["progress"] = stored["hero_speed"]
    return RewardConfig(**{k: v for k, v in stored.items() if (k in RewardConfig.__dataclass_fields__)})


def resolve_checkpoint(path: str) -> str:
    """Accept a run directory, a checkpoint directory, or a model file."""
    if (os.path.isfile(path)):
        return path
    for candidate in (
        os.path.join(path, "checkpoints", "best_model.zip"),
        os.path.join(path, "checkpoints", "policy_final.zip"),
        os.path.join(path, "best_model.zip"),
        os.path.join(path, "policy_final.zip"),
    ):
        if (os.path.isfile(candidate)):
            return candidate
    # Otherwise take the highest-numbered periodic checkpoint.  Sorted by step
    # count, not by name: a plain sort is lexicographic, so "policy_9984_steps"
    # beats "policy_129792_steps" and a run without a policy_final would resume
    # from its *first* checkpoint.
    checkpoint_dir = path
    if (os.path.isdir(os.path.join(path, "checkpoints"))):
        checkpoint_dir = os.path.join(path, "checkpoints")
    candidates = [
        name for name in os.listdir(checkpoint_dir)
        if (name.startswith("policy") and name.endswith(".zip"))
    ]
    if (len(candidates) == 0):
        raise FileNotFoundError(f"no stable-baselines3 checkpoint under {path}")
    return max(
        (os.path.join(checkpoint_dir, name) for name in candidates),
        key=checkpoint_step_count,
    )


def checkpoint_step_count(name: str) -> int:
    """How many steps a checkpoint was saved at.

    Read out of the archive's own ``data`` blob rather than the filename, because
    ``policy_final.zip`` and ``best_model.zip`` carry no number and would
    otherwise report zero -- which would silently defeat the seed offset that
    stops a resumed run replaying the episodes it already trained on. Falls back
    to the ``policy_<n>_steps`` name for a path that is not a readable archive.
    """
    if (os.path.isfile(name)):
        try:
            with zipfile.ZipFile(name) as archive:
                stored = json.loads(archive.read("data").decode("utf-8"))
            steps = stored.get("num_timesteps")
            if (steps is not None):
                return int(steps)
        except Exception:
            pass
    match = re.search(r"policy_(\d+)_steps", os.path.basename(name))
    return int(match.group(1)) if (match is not None) else 0


RECURRENT_ALGORITHMS = ("recurrent_ppo", "ppo_lstm", "recurrentppo")


def is_recurrent(algorithm: str) -> bool:
    return str(algorithm).lower() in RECURRENT_ALGORITHMS


def algorithm_and_policy(algorithm: str):
    """Map an algorithm name onto its SB3 class and default policy name."""
    if (is_recurrent(algorithm)):
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO, "MlpLstmPolicy"
    if (str(algorithm).lower() != "ppo"):
        raise ValueError(f"unknown algorithm {algorithm!r}; expected 'recurrent_ppo' or 'ppo'")
    from stable_baselines3 import PPO

    return PPO, "MlpPolicy"


def detect_algorithm(checkpoint: str) -> str:
    """Work out which algorithm saved a checkpoint.

    A run's ``run_config.json`` is the authority when it is there.  Otherwise the
    saved policy class is read out of the zip, because loading a recurrent
    checkpoint with ``PPO`` (or the reverse) fails in a way that reads as a
    corrupt file rather than as the mismatch it is.
    """
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint)))
    config_path = os.path.join(run_dir, "run_config.json")
    if (os.path.isfile(config_path)):
        try:
            with open(config_path, "r") as handle:
                stored = json.load(handle).get("train", {})
            if ("algorithm" in stored):
                return str(stored["algorithm"])
        except Exception:
            pass
    try:
        from stable_baselines3.common.save_util import load_from_zip_file

        data, _, _ = load_from_zip_file(checkpoint, device="cpu", print_system_info=False)
        policy_class = str(data.get("policy_class", ""))
        if (("Lstm" in policy_class) or ("Recurrent" in policy_class)):
            return "recurrent_ppo"
    except Exception:
        pass
    return "ppo"


def load_model(path: str, device: str = "cpu", algorithm: Optional[str] = None):
    checkpoint = resolve_checkpoint(path)
    algorithm = algorithm or detect_algorithm(checkpoint)
    model_class, _ = algorithm_and_policy(algorithm)
    return model_class.load(checkpoint, device=device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


REWARD_WEIGHT_NAMES = (
    "platoon_speed", "progress", "speed_variance", "acceleration", "jerk",
    "headway", "energy", "stopped", "collision", 
    "rear_flux_smoothness", "upstream_oscillation", "upstream_delay"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-name", default=time.strftime("ppo_%Y%m%d_%H%M%S"))
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--resume", default=None,
        help="continue a stopped run from a checkpoint or run directory. Weights AND "
             "optimizer state are restored, so this is a continuation, not a warm restart. "
             "--total-timesteps stays the total for the whole run; the remainder is computed "
             "from the checkpoint. Writes to a new --run-name (the original's monitor and "
             "progress files are not touched) and offsets the env seed so the resumed half "
             "does not replay the same episode specs.",
    )
    parser.add_argument("--config-folder", default=DEFAULT_CONFIG_FOLDER)
    parser.add_argument("--datasets", nargs="+", default=["2022-11-21.json", "2022-11-22.json", "2022-11-23.json", "2022-11-24.json", "2022-11-25.json", "2022-11-28.json", "2022-12-01.json", "2022-12-02.json"], help="dataset json files to train on")
    parser.add_argument("--roads", nargs="+", default=["2"])
    parser.add_argument(
        "--algorithm", default="recurrent_ppo", choices=["recurrent_ppo", "ppo"],
        help="recurrent_ppo (sb3-contrib LSTM, the default) or ppo (memoryless ablation)",
    )
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--n-lstm-layers", type=int, default=2)
    parser.add_argument(
        "--shared-lstm", action="store_true",
        help="one LSTM for actor and critic (implies --no-critic-lstm)",
    )
    parser.add_argument(
        "--no-critic-lstm", dest="enable_critic_lstm", action="store_false",
        help="give the critic no recurrence of its own",
    )
    parser.add_argument("--envs", type=int, default=24, help="parallel environments; 1 runs in-process")
    parser.add_argument("--no-subprocess", action="store_true", help="use DummyVecEnv even with several environments")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=3*128)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.97)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=TrainConfig.ent_coef)
    parser.add_argument("--target-kl", type=float, default=0.0, help="0 disables the KL early stop")
    parser.add_argument(
        # BooleanOptionalAction, not store_true: store_true's default is always
        # False, which silently overrides TrainConfig's value on every CLI launch
        # and makes the dataclass default dead code.  This form defaults to the
        # dataclass and gives an explicit --no-use-sde to turn it off.
        "--use-sde", action=argparse.BooleanOptionalAction, default=TrainConfig.use_sde,
        help="state-dependent exploration: temporally correlated noise, so whole "
             "manoeuvres get explored rather than per-step jitter",
    )
    parser.add_argument(
        "--sde-sample-freq", type=int, default=TrainConfig.sde_sample_freq,
        help="steps a gSDE perturbation is held for (-1 = once per rollout; 8-16 is a good range)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--macro-lookahead-cells", type=int, default=4)
    parser.add_argument("--max-acceleration", type=float, default=1.5)
    parser.add_argument("--max-deceleration", type=float, default=2.0)
    parser.add_argument(
        "--hero-speed-mode", type=int, default=None,
        help="SUMO speed mode for the hero (default: SUMO's 31, safe-speed clipped; 30 drops the safe-speed check, 0 gives full authority)",
    )
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--eval-freq", type=int, default=0, help="0 disables periodic evaluation")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--normalize", action="store_true", help="wrap in VecNormalize (reward scaling)")
    parser.add_argument("--gui", action="store_true", help="run SUMO with a GUI (use --envs 1)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--check-env", action="store_true", help="run SB3's env checker and exit")
    for name in REWARD_WEIGHT_NAMES:
        parser.add_argument(f"--w-{name.replace('_', '-')}", type=float, default=None, dest=f"w_{name}")
    parser.add_argument("--target-headway", type=float, default=None)
    parser.add_argument("--platoon-scope", default=None, choices=["lane_behind", "all_behind", "bubble"])
    return parser


def configs_from_args(args) -> Tuple[EnvConfig, RewardConfig, TrainConfig]:
    env_config = EnvConfig(
        config_folder=args.config_folder,
        datasets=tuple(args.datasets),
        roads=tuple(args.roads),
        max_steps=args.max_steps,
        macro_lookahead_cells=args.macro_lookahead_cells,
        max_acceleration=args.max_acceleration,
        max_deceleration=args.max_deceleration,
        hero_speed_mode=args.hero_speed_mode,
        gui=args.gui,
        verbose=args.verbose,
    )
    reward_config = RewardConfig()
    for name in REWARD_WEIGHT_NAMES:
        value = getattr(args, f"w_{name}", None)
        if (value is not None):
            setattr(reward_config, name, value)
    if (args.target_headway is not None):
        reward_config.target_headway = args.target_headway
    if (args.platoon_scope is not None):
        reward_config.platoon_scope = args.platoon_scope
    train_config = TrainConfig(
        algorithm=args.algorithm,
        lstm_hidden_size=args.lstm_hidden_size,
        n_lstm_layers=args.n_lstm_layers,
        shared_lstm=args.shared_lstm,
        # sb3-contrib rejects the two together, and --shared-lstm is the explicit
        # request of the pair, so it wins.
        enable_critic_lstm=(args.enable_critic_lstm and (not args.shared_lstm)),
        total_timesteps=args.total_timesteps,
        envs=args.envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        target_kl=(args.target_kl if (args.target_kl > 0.0) else None),
        use_sde=args.use_sde,
        sde_sample_freq=args.sde_sample_freq,
        seed=args.seed,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        tensorboard=args.tensorboard,
        normalize=args.normalize,
    )
    return env_config, reward_config, train_config


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    env_config, reward_config, train_config = configs_from_args(args)

    if (args.check_env):
        from stable_baselines3.common.env_checker import check_env

        env = I24SumoHeroEnv(env_config=env_config, reward_config=reward_config, seed=args.seed)
        try:
            # An episode is only ~40 steps, so the checker's rollout is a real one.
            check_env(env, warn=True, skip_render_check=True)
            print("environment passed SB3's check_env")
        finally:
            env.close()
        return

    output_dir = os.path.join(args.output_root, args.run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"training {args.run_name} into {output_dir}")
    started = time.time()
    train(
        env_config=env_config,
        reward_config=reward_config,
        train_config=train_config,
        output_dir=output_dir,
        subprocess=(not args.no_subprocess),
        resume_from=args.resume,
    )
    print(f"done in {(time.time() - started) / 60.0:.1f} min; checkpoints in {output_dir}/checkpoints")


if __name__ == "__main__":
    main()
