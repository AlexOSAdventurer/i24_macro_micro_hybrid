"""With GT boundary overwrite enabled (production path) — isolate its effect."""
import os, sys, json
sys.path.insert(0, "/home/richarwa/SecondSSD/I24/i24_macroscopic")

from simulation import Simulation, GroundTruthStore

def total_mass(network):
    return sum(c.mass for road in network.roads.values()
               for c in road.cells.values())
def boundary_mass(network):
    return sum(c.mass for road in network.roads.values()
               for c in road.cells.values()
               if len(c.inflow_connections) == 0 or len(c.outflow_connections) == 0)

captured = {}
orig_compute = Simulation._compute_active_edge_flows
def patched_compute(self, active):
    e, i, o = orig_compute(self, active)
    captured["ext_in"]  = dict(i)
    captured["ext_out"] = dict(o)
    return e, i, o
Simulation._compute_active_edge_flows = patched_compute

# Snapshot boundary mass BEFORE the GT overwrite by patching step.
overwrite_deltas = []
import simulation as S
orig_overwrite = S.GroundTruthStore.apply_density_snapshot_to_network_boundaries
def patched_overwrite(self, network, t, tolerance=1e-1):
    before = boundary_mass(network)
    orig_overwrite(self, network, t, tolerance)
    after = boundary_mass(network)
    overwrite_deltas.append(after - before)
S.GroundTruthStore.apply_density_snapshot_to_network_boundaries = patched_overwrite

with open("/home/richarwa/SecondSSD/I24/i24_macroscopic/i24_motion_to_dataset.json") as f:
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
print(f"{'step':>4} {'dM_total':>10} {'flux_in*dt':>11} {'flux_out*dt':>12} {'GT_overwrite':>13} {'leak':>10}")
M_prev = total_mass(sim.network)
for i in range(1, 11):
    sim.step()
    M_now = total_mass(sim.network)
    sum_in  = sum(captured["ext_in"].values())  * dt
    sum_out = sum(captured["ext_out"].values()) * dt
    gt_delta = overwrite_deltas[-1]
    expected = sum_in - sum_out + gt_delta
    actual   = M_now - M_prev
    print(f"{i:>4} {actual:>10.4f} {sum_in:>11.4f} {sum_out:>12.4f} {gt_delta:>13.4f} {actual - expected:>10.4f}")
    M_prev = M_now
