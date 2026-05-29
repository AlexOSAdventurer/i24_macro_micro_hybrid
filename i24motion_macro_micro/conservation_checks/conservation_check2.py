"""Check mass conservation more rigorously: compare dM to net boundary flux."""
import os, sys, json
sys.path.insert(0, ".")

import simulation as S
from simulation import Simulation, GroundTruthStore

def total_mass(network):
    return sum(c.mass for road in network.roads.values()
               for c in road.cells.values())

# Monkey-patch step() to capture per-step edge_flow and external in/outflow.
captured = {}
orig_compute = Simulation._compute_active_edge_flows
def patched_compute(self, active):
    edge_flow, ext_in, ext_out = orig_compute(self, active)
    # Snapshot BEFORE _step_active_network applies them
    captured["edge_flow"] = dict(edge_flow)
    captured["ext_in"]    = dict(ext_in)
    captured["ext_out"]   = dict(ext_out)
    captured["active"]    = active
    return edge_flow, ext_in, ext_out
Simulation._compute_active_edge_flows = patched_compute

# Also patch boundary GT overwrite to a no-op so we can compare cleanly.
S.GroundTruthStore.apply_density_snapshot_to_network_boundaries = lambda self, network, t, tolerance=1e-1: None

with open("./i24_motion_to_dataset.json") as f:
    config = json.load(f)

sim = Simulation.from_json(
    json_path=os.path.join(config["storage_locations"]["simulation_dataset"], "network.json"),
    time_resolution=config["time_step"],
    origin_time=config["time_origin"],
    min_cell_length=100.0,
)
gt = GroundTruthStore.from_parquet(
    os.path.join(config["storage_locations"]["simulation_dataset"], "micro.parquet"),
    os.path.join(config["storage_locations"]["simulation_dataset"], "macro.parquet"),
)
sim.initialize_from_ground_truth(gt, time_value=config["time_origin"])

dt = sim.time_resolution
print(f"{'step':>4} {'M_before':>12} {'M_after':>12} {'dM_actual':>12} {'flux_in*dt':>12} {'flux_out*dt':>12} {'expected_dM':>12} {'leak':>10}")
M_prev = total_mass(sim.network)
for i in range(1, 11):
    sim.step()
    M_now = total_mass(sim.network)

    # Internal edges: should net to zero across ALL cells (each q is added to v's rear, subtracted from u's front).
    # External in/out: dM_expected = (sum ext_in - sum ext_out) * dt
    sum_in  = sum(captured["ext_in"].values())  * dt
    sum_out = sum(captured["ext_out"].values()) * dt
    expected = sum_in - sum_out
    actual   = M_now - M_prev
    print(f"{i:>4} {M_prev:>12.4f} {M_now:>12.4f} {actual:>12.6f} {sum_in:>12.6f} {sum_out:>12.6f} {expected:>12.6f} {actual - expected:>10.6f}")
    M_prev = M_now
