"""Replay and inspect the trained hero controllers from sim_rl_sumo_training.

Runs one or more episodes of the coupled macro/micro simulation with the hero
under a chosen controller -- a stable-baselines3 checkpoint (RecurrentPPO or
plain PPO; the algorithm is read back from the run and the LSTM hidden state is
carried across the episode), or one of the non-learned baselines -- and then
lets you look at what happened:

  * a paired comparison table in the terminal, every controller replayed on the
    *same* episode specs (same day, same minute, same SUMO seed), which is the
    only way the differences mean anything;
  * per-step traces on disk for ``sim_rl_sumo_analysis.py``;
  * the Dash time-space viewer from ``sim_demo.run_app``;
  * optionally the rollout stored in the project's ``RolloutStore``, under a
    run_id ``sim_analysis.py`` already knows how to parse, so an RL run can be
    scored against the LWR / SUMO / CARLA rollouts on upstream density error.

Running (inside the container, where SUMO and SB3 live)::

    # paired comparison of a trained policy against the baselines
    docker exec <container> bash -lc \\
        'cd /workspaces/i24motion_macro_micro && python3.10 sim_rl_sumo_demo.py \\
             --run run_data/rl/ppo_smoothing --controllers ppo sumo idm macro_lookahead \\
             --episodes 5'

    # watch one episode in SUMO's GUI, then in the Dash viewer
    docker exec <container> bash -lc \\
        'cd /workspaces/i24motion_macro_micro && python3.10 sim_rl_sumo_demo.py \\
             --run run_data/rl/ppo_smoothing --controllers ppo --episodes 1 --gui --dash'

With ``--dash`` the viewer serves on port 8050; forward it with
``ssh -L 8050:localhost:8050 user@host`` to see it from a laptop.
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sim_rl_sumo_training import (
    BASELINE_CONTROLLERS,
    BaselineController,
    EnvConfig,
    EpisodeSpec,
    I24SumoHeroEnv,
    RewardConfig,
    detect_algorithm,
    env_config_from_dict,
    load_model,
    load_run_config,
    resolve_checkpoint,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# The corridor's heading, so the Dash viewer draws the road horizontally.  Same
# value the other demos in sim_demo.py pass.
ROTATION_DEG = 82.8192

DEFAULT_DATABASE = os.path.join(HERE, "run_data", "results_final.db")


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


class PolicyController(BaselineController):
    """A trained stable-baselines3 policy behind the baseline interface.

    Wrapping it this way means the learned controller and the hand-written ones
    go through exactly the same episode loop and are scored by exactly the same
    reward, which is what makes the comparison table honest.

    The LSTM hidden state is carried across the episode and cleared in ``reset``.
    That is not optional for a recurrent policy: evaluating one without threading
    its state back in silently turns it into a memoryless policy reading a
    zeroed hidden state every step, which is exactly the thing RecurrentPPO was
    chosen to avoid.  ``episode_start`` tells the policy when to zero it itself.
    A feedforward PPO model ignores both arguments and returns ``None`` as its
    state, so the same path serves both.
    """

    name = "ppo"

    def __init__(self, model, deterministic: bool = True, vec_normalize=None) -> None:
        self.model = model
        self.deterministic = bool(deterministic)
        self.vec_normalize = vec_normalize
        self.lstm_states = None
        self.episode_start = True

    def reset(self) -> None:
        self.lstm_states = None
        self.episode_start = True

    def act(self, observation: np.ndarray, env: I24SumoHeroEnv) -> Optional[float]:
        if (self.vec_normalize is not None):
            observation = self.vec_normalize.normalize_obs(observation)
        action, self.lstm_states = self.model.predict(
            observation,
            state=self.lstm_states,
            episode_start=np.array([self.episode_start]),
            deterministic=self.deterministic,
        )
        self.episode_start = False
        return float(np.asarray(action).reshape(-1)[0])


def load_policy_controller(run_or_checkpoint: str, deterministic: bool = True, device: str = "cpu"):
    """Load a checkpoint, plus its VecNormalize statistics when the run used them."""
    checkpoint = resolve_checkpoint(run_or_checkpoint)
    algorithm = detect_algorithm(checkpoint)
    model = load_model(checkpoint, device=device, algorithm=algorithm)
    vec_normalize = None
    statistics = os.path.join(os.path.dirname(checkpoint), "vecnormalize.pkl")
    if (os.path.isfile(statistics)):
        from stable_baselines3.common.vec_env import VecNormalize

        vec_normalize = VecNormalize.load(statistics, venv=None)
        vec_normalize.training = False
    print(f"loaded {algorithm} policy from {checkpoint}")
    return PolicyController(model, deterministic=deterministic, vec_normalize=vec_normalize)


def build_controller(name: str, args) -> BaselineController:
    if (name == "ppo"):
        if (args.run is None):
            raise ValueError("--controllers ppo needs --run pointing at a training run or checkpoint")
        return load_policy_controller(args.run, deterministic=(not args.stochastic), device=args.device)
    if (name not in BASELINE_CONTROLLERS):
        raise ValueError(f"unknown controller {name!r}; pick from ppo, {', '.join(BASELINE_CONTROLLERS)}")
    return BASELINE_CONTROLLERS[name]()


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


def build_env(args) -> Tuple[I24SumoHeroEnv, RewardConfig]:
    """Rebuild the environment a run was trained in, with demo overrides applied.

    Reading the configuration back out of ``run_config.json`` matters: a policy
    evaluated under a different reward, lookahead depth or bubble geometry than
    it was trained under is not the policy that was trained.  Only the things a
    demo legitimately changes -- which day, which road, the GUI, rollout
    recording -- are overridden here.
    """
    env_config = EnvConfig()
    reward_config = RewardConfig()
    if (args.run is not None) and (os.path.isfile(os.path.join(args.run, "run_config.json"))):
        env_config, reward_config, _ = load_run_config(args.run)
    overrides: Dict[str, Any] = {"gui": args.gui, "verbose": args.verbose}
    if (args.datasets):
        overrides["datasets"] = tuple(args.datasets)
    if (args.roads):
        overrides["roads"] = tuple(args.roads)
    if (args.max_steps is not None):
        overrides["max_steps"] = args.max_steps
    env_config = env_config_from_dict(asdict(env_config), **overrides)

    env = I24SumoHeroEnv(
        env_config=env_config,
        reward_config=reward_config,
        seed=args.seed,
        label_prefix="rl_demo",
        # Every step then keeps a snapshot of the network, which is what the
        # Dash viewer and the RolloutStore read; training turns this off.
        record_rollout=(args.dash or args.store),
    )
    return env, reward_config


def sample_specs(env: I24SumoHeroEnv, count: int) -> List[EpisodeSpec]:
    """Draw the episodes every controller will be replayed on."""
    return [env._sample_spec() for _ in range(count)]


def run_episode(
    env: I24SumoHeroEnv, controller: BaselineController, spec: EpisodeSpec
) -> Tuple[Dict[str, Any], List[Dict[str, float]]]:
    """One pinned episode under one controller. Returns (summary, per-step records)."""
    observation, _ = env.reset(options={"spec": spec})
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
# Output
# ---------------------------------------------------------------------------


SUMMARY_COLUMNS = [
    "controller", "episode", "dataset", "road", "start_time", "steps",
    "terminal_reason", "return", "mean_reward", "hero_mean_speed",
    "hero_speed_std", "platoon_mean_speed", "platoon_speed_std",
    "acceleration_rms", "jerk_rms", "min_headway", "distance", "energy",
    "energy_per_metre",
]


def write_summaries(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_COLUMNS})


def write_trace(path: str, records: Sequence[Dict[str, float]]) -> None:
    if (len(records) == 0):
        return
    columns = list(records[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def store_rollout(
    sim,
    controller_name: str,
    spec: EpisodeSpec,
    steps: int,
    database: str,
    dataset_config: Dict[str, Any],
) -> Optional[str]:
    """Put the rollout in the project's RolloutStore under a parseable run_id.

    The id follows the convention ``sim_analysis.extract_run_data`` expects --
    ``<sim_type>_road(..)_timeorigin(..)_episodelength(..)_<dataset>`` -- with
    hyphens inside the controller name because that parser splits on
    underscores.  The metadata carries the dataset config under the same
    ``config_folder`` key that parser reads for the empirical time origin, so an
    RL rollout sits alongside the LWR / SUMO / CARLA ones and is scored by the
    same upstream density metrics.
    """
    from simulation import RolloutRenderer
    from rollout_store import RolloutStore

    if (not sim.rollout_results):
        print("WARNING: nothing to store -- rerun with --store or --dash so rollouts are recorded")
        return None
    sim_type = f"rl-{controller_name.replace('_', '-')}"
    run_id = (
        f"{sim_type}_road({spec.road})_timeorigin({spec.start_time})"
        f"_episodelength({steps})_{spec.dataset}"
    )
    renderer = RolloutRenderer(sim)
    store = RolloutStore(database)
    run_id = store.put_renderer(
        renderer,
        run_id,
        metadata={
            "dataset_file": spec.dataset,
            "config_folder": dataset_config,
            "controller": controller_name,
            "spec": asdict(spec),
        },
        overwrite=True,
    )
    print(f"stored rollout as {run_id}")
    return run_id


def print_comparison(rows: Sequence[Dict[str, Any]]) -> None:
    """Per-controller means over the shared episode set."""
    controllers: List[str] = []
    for row in rows:
        if (row["controller"] not in controllers):
            controllers.append(row["controller"])

    header = (
        f"{'controller':<18}{'n':>3}{'return':>9}{'steps':>7}{'hero v':>8}{'platoon v':>11}"
        f"{'plt sd':>8}{'a rms':>8}{'jerk':>8}{'min hw':>8}{'J/m':>9}"
    )
    print()
    print(header)
    print("-" * len(header))
    for controller in controllers:
        subset = [row for row in rows if (row["controller"] == controller)]

        def mean(key: str) -> float:
            values = [float(row[key]) for row in subset if (row.get(key, "") != "")]
            return float(np.mean(values)) if (len(values) > 0) else float("nan")

        print(
            f"{controller:<18}{len(subset):>3}{mean('return'):>9.2f}{mean('steps'):>7.1f}"
            f"{mean('hero_mean_speed'):>8.2f}{mean('platoon_mean_speed'):>11.2f}"
            f"{mean('platoon_speed_std'):>8.2f}{mean('acceleration_rms'):>8.3f}"
            f"{mean('jerk_rms'):>8.3f}{mean('min_headway'):>8.2f}{mean('energy_per_metre'):>9.1f}"
        )
    print()
    print("hero v / platoon v in m/s, plt sd = mean within-platoon speed spread (lower is smoother),")
    print("a rms in m/s^2, jerk in m/s^3, min hw = smallest time headway in s, J/m = tractive energy.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", default=None, help="training run directory or checkpoint (.zip) for the ppo controller")
    parser.add_argument(
        "--controllers", nargs="+", default=["ppo", "sumo"],
        help="ppo plus any of: " + ", ".join(BASELINE_CONTROLLERS),
    )
    parser.add_argument("--episodes", type=int, default=3, help="episode specs, replayed by every controller")
    parser.add_argument("--seed", type=int, default=12345, help="seeds the episode sampling, so runs are repeatable")
    parser.add_argument("--datasets", nargs="+", default=["2022-11-29.json", "2022-11-30.json"], help="override the run's datasets (e.g. a held-out day)")
    parser.add_argument("--roads", nargs="+", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out", default=None, help="output directory (default: <run>/demo, else run_data/rl/demo)")
    parser.add_argument(
        "--stochastic", action="store_true",
        help="sample from the policy instead of taking its mean; worth reporting alongside the "
             "deterministic result, because on a POMDP the best memoryless policy can be strictly stochastic",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gui", action="store_true", help="run SUMO with its GUI")
    parser.add_argument("--dash", action="store_true", help="serve the last rollout in the Dash viewer")
    parser.add_argument("--dash-port", type=int, default=8050)
    parser.add_argument("--store", action="store_true", help="write rollouts into the RolloutStore database")
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    output_dir = args.out
    if (output_dir is None):
        output_dir = os.path.join(args.run, "demo") if (args.run and os.path.isdir(args.run)) else os.path.join(HERE, "run_data", "rl", "demo")
    os.makedirs(os.path.join(output_dir, "traces"), exist_ok=True)

    env, _ = build_env(args)
    controllers = [(name, build_controller(name, args)) for name in args.controllers]
    specs = sample_specs(env, args.episodes)
    print(f"{len(specs)} episode(s) x {len(controllers)} controller(s); writing to {output_dir}")
    for i, spec in enumerate(specs):
        print(f"  episode {i}: {spec.dataset} road {spec.road} at t={spec.start_time:.0f} seed={spec.sumo_seed}")

    rows: List[Dict[str, Any]] = []
    last_sim = None
    last_controller = None
    last_spec = None
    try:
        for name, controller in controllers:
            for index, spec in enumerate(specs):
                try:
                    summary, records = run_episode(env, controller, spec)
                except Exception as exc:
                    # One unusable minute of data should not cost the whole sweep.
                    print(f"WARNING: {name} episode {index} failed: {exc}")
                    continue
                row = dict(summary)
                row["controller"] = name
                row["episode"] = index
                rows.append(row)
                write_trace(os.path.join(output_dir, "traces", f"{name}_{index}.csv"), records)
                print(
                    f"{name:<18} ep {index}  steps {summary['steps']:>3}  "
                    f"return {summary['return']:>8.2f}  hero {summary['hero_mean_speed']:>5.2f} m/s  "
                    f"platoon {summary['platoon_mean_speed']:>5.2f} m/s  "
                    f"spread {summary['platoon_speed_std']:>4.2f}  {summary['terminal_reason']}"
                )
                if (args.store):
                    _, dataset_config = env.env_config.dataset_paths(spec.dataset)
                    store_rollout(
                        env.sim, name, spec, int(summary["steps"]), args.database, dataset_config
                    )
                last_sim, last_controller, last_spec = env.sim, name, spec

        if (len(rows) > 0):
            summary_path = os.path.join(output_dir, "episodes.csv")
            write_summaries(summary_path, rows)
            print_comparison(rows)
            print(f"wrote {summary_path} and {len(rows)} trace file(s) under {output_dir}/traces")
    finally:
        # Keep the last simulation alive for the viewer; only the engine goes.
        env.close()

    if (args.dash):
        if (last_sim is None) or (not last_sim.rollout_results):
            print("nothing to show in the viewer")
            return
        from sim_demo import run_app

        print(
            f"serving the {last_controller} rollout of {last_spec.dataset} "
            f"at t={last_spec.start_time:.0f} on port {args.dash_port}"
        )
        run_app(last_sim, rotation_deg=ROTATION_DEG, port=args.dash_port)


if __name__ == "__main__":
    main()
