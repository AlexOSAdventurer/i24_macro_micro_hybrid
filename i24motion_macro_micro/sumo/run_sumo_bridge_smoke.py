"""Smoke test for the SUMO-backed micro coupler.

Runs the macroscopic simulation with an I24SumoCoupler-driven bubble for a
short window and reports what the coupling actually did: how the bubble moved,
how many vehicles SUMO held, and how much mass the per-lane flux memories
carry.  Not a verification harness -- just enough to catch a broken coupling.

Run inside the container (SUMO lives there, not on the host):

    docker exec a3c093f073b7 bash -lc \
        'cd /workspaces/i24motion_macro_micro && python3.10 sumo/run_sumo_bridge_smoke.py --steps 60'
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from simulation import Simulation, GroundTruthStore  # noqa: E402
from i24_micro_bridge import I24MicroSimBridge  # noqa: E402
from i24_sumo_coupler import I24SumoCoupler  # noqa: E402

LANES = [-1, -2, -3, -4]


def total_system_mass(sim):
    """Continuum mass plus the vehicles the masks are holding.

    The mask cells live in ``sim.masking_cells``, not in the network, so summing
    the network alone undercounts by however many vehicles are inside the bubble.
    """
    total = 0.0
    for road_id, cell_id in sim.network.all_cell_keys():
        cell = sim.network.get_cell(road_id, cell_id)
        total += cell.mass + cell.mask_mass
    for mask in sim.masking_cells.values():
        total += mask.mass
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--config", default=os.path.join(ROOT, "i24_motion_to_dataset.json"))
    parser.add_argument("--road", default="2")
    parser.add_argument("--initial-middle-s", type=float, default=350.0)
    parser.add_argument("--max-middle-s", type=float, default=1300.0)
    parser.add_argument("--margin-s", type=float, default=150.0)
    parser.add_argument("--visible-window", type=float, default=150.0)
    parser.add_argument("--step-length", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as handle:
        config = json.load(handle)
    dataset = config["storage_locations"]["simulation_dataset"]
    if not os.path.isabs(dataset):
        dataset = os.path.join(ROOT, dataset)

    sim = Simulation.from_json(
        json_path=os.path.join(dataset, "network.json"),
        time_resolution=config["time_step"],
        origin_time=config["time_origin"],
        min_cell_length=100.0,
    )
    gt = GroundTruthStore.from_parquet(
        os.path.join(dataset, "micro.parquet"), os.path.join(dataset, "macro.parquet")
    )
    sim.initialize_from_ground_truth(gt, time_value=config["time_origin"])

    coupler = I24SumoCoupler(
        gt,
        dt=config["time_step"],
        lanes=LANES,
        mapping=config,
        hero_road=args.road,
        desired_time=config["time_origin"],
        desired_s=args.initial_middle_s,
        visible_window=args.visible_window,
        ghost_window=0.0,
        step_length=args.step_length,
        seed=args.seed,
        gui=args.gui,
        verbose=args.verbose,
    )
    bridge = I24MicroSimBridge(
        sim=sim,
        road_id=args.road,
        lanes=LANES,
        initial_middle_s=args.initial_middle_s,
        margin_s=args.margin_s,
        max_middle_s=args.max_middle_s,
        micro_coupler=coupler,
        bridge_callback_name="bridge_step",
    )

    header = (
        f"{'step':>5} {'middle_s':>10} {'anchor':>8} {'sumo':>6} {'vis':>5} "
        f"{'rear_flux':>10} {'front_flux':>11} {'net_mass':>10}"
    )
    print(header)
    print("-" * len(header))

    try:
        for i in range(args.steps):
            sim.step()
            if not bridge.running:
                print(f"bridge retired at step {i}")
                break
            live = len(coupler.sumo_sim.visible_states)
            visible = sum(len(coupler.visible_state[lane]) for lane in LANES)
            rear = sum(bridge.flow_memory_rear.values())
            front = sum(bridge.flow_memory_front.values())
            print(
                f"{i:>5} {bridge.middle_s:>10.2f} {bridge.anchor_speed:>8.2f} "
                f"{live:>6} {visible:>5} {rear:>10.3f} {front:>11.3f} "
                f"{total_system_mass(sim):>10.2f}"
            )
    finally:
        if bridge.running:
            bridge.destroy()

    print()
    print("per-lane flux memory at exit")
    for lane in LANES:
        print(
            f"  lane {lane:>3}  rear {bridge.flow_memory_rear[lane]:>9.4f}"
            f"   front {bridge.flow_memory_front[lane]:>9.4f}"
        )


if __name__ == "__main__":
    main()
