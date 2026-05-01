"""Total-mass conservation across moving-mask cycle.

Total mass = sum(cell.mass) + sum(cell.mask_mass).
Expected change per step = ext_in*dt - ext_out*dt + GT_overwrite_delta_total.
The bridge can also add/remove vehicles (discrete domain), so we report that too.
"""
import os, sys, json
sys.path.insert(0, "/home/richarwa/SecondSSD/I24/i24_macroscopic")

import simulation as S
from simulation import Simulation, GroundTruthStore
from i24_trajectory_replayer import I24TrajectoryReplayer
from i24_micro_bridge import I24MicroSimBridge

def macro_mass(net):
    return sum(c.mass for r in net.roads.values() for c in r.cells.values())
def mask_mass_field(net):
    return sum(c.mask_mass for r in net.roads.values() for c in r.cells.values())
def total_mass(net):
    return macro_mass(net) + mask_mass_field(net)
def boundary_total_mass(net):
    return sum(c.mass + c.mask_mass for r in net.roads.values() for c in r.cells.values()
               if len(c.inflow_connections) == 0 or len(c.outflow_connections) == 0)
def vehicle_count(sim):
    return sum(len(m.vehicles) for m in sim.masking_cells.values())

captured = {}
orig_compute = Simulation._compute_active_edge_flows
def patched_compute(self, active):
    e, i, o, m = orig_compute(self, active)
    captured["in_internal"] = dict(e)
    captured["ext_in"]  = dict(i)
    captured["ext_out"] = dict(o)
    captured["mask_internal"] = dict(m)
    return e, i, o, m
Simulation._compute_active_edge_flows = patched_compute

overwrite_deltas = []
orig_overwrite = S.GroundTruthStore.apply_density_snapshot_to_network_boundaries
def patched_overwrite(self, network, t, tolerance=1e-1):
    before = boundary_total_mass(network)
    orig_overwrite(self, network, t, tolerance)
    overwrite_deltas.append(boundary_total_mass(network) - before)
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

replayer = I24TrajectoryReplayer(gt, dt=1.0, lanes=[-1, -2, -3, -4])
bridge = I24MicroSimBridge(
    sim=sim, road_id="2", lanes=[-1, -2, -3, -4],
    initial_middle_s=375.0, margin_s=150.0, max_middle_s=1450,
    update_micro_callback=replayer.step, bridge_callback_name="bridge_step",
)

dt = sim.time_resolution
print(f"{'step':>4} {'dM_total':>10} {'dM_macro':>10} {'dM_mask':>10} {'bridge_s':>10} {'anchor_speed':>13} {'flux_internal':>12} {'flux_in*dt':>11} {'flux_out*dt':>12} {'vehicles_in_mask':>17} {'GT_overwrite':>13} {'leak':>10}")
M_prev = total_mass(sim.network)
M_macro_prev = macro_mass(sim.network)
M_mask_prev = mask_mass_field(sim.network)
vehicles_prev = vehicle_count(sim)
for i in range(1, 85):
    bridge_s = bridge.middle_s
    anchor_speed = bridge.anchor_speed
    sim.step()
    M_now = total_mass(sim.network)
    M_macro_now = macro_mass(sim.network)
    M_mask_now = mask_mass_field(sim.network)
    M_macro_delta = M_macro_now - M_macro_prev
    M_mask_delta = M_mask_now - M_mask_prev
    sum_mask_internal = sum(captured["mask_internal"].values())  * dt if "mask_internal" in captured else 0.0
    sum_internal = sum(captured["in_internal"].values())  * dt if "in_internal" in captured else 0.0
    sum_in  = sum(captured["ext_in"].values())  * dt if "ext_in" in captured else 0.0
    sum_out = sum(captured["ext_out"].values()) * dt if "ext_out" in captured else 0.0
    vehicles = vehicle_count(sim)
    gt_delta = overwrite_deltas[-1]
    expected = sum_in - sum_out + gt_delta + sum_mask_internal
    actual   = M_macro_now - M_macro_prev
    print(f"{i:>4} {actual:>10.4f} {M_macro_delta:>10.4f} {M_mask_delta:>10.4f} {bridge_s:>10.4f} {anchor_speed:>13.4f} {sum_internal:>13.4f} {sum_in:>11.4f} {sum_out:>12.4f} {vehicles:>17.4f} {gt_delta:>13.4f} {actual - expected:>10.4f}")
    M_prev = M_now
    M_macro_prev = M_macro_now
    M_mask_prev = M_mask_now
    vehicles_prev = vehicles
