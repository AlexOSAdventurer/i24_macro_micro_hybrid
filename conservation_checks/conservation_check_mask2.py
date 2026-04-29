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
    e, i, o = orig_compute(self, active)
    captured["ext_in"]  = dict(i)
    captured["ext_out"] = dict(o)
    return e, i, o
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
    initial_middle_s=150.0, margin_s=150.0, max_middle_s=1450,
    update_micro_callback=replayer.step, bridge_callback_name="bridge_step",
)

dt = sim.time_resolution
print(f"{'step':>3} {'dM_macro':>9} {'dM_mask':>9} {'dM_tot':>9} {'in*dt':>7} {'out*dt':>7} {'GT_d':>7} {'d_vehic':>8} {'unexpl':>9}")

prev_macro = macro_mass(sim.network)
prev_mask  = mask_mass_field(sim.network)
prev_veh   = vehicle_count(sim)
for i in range(1, 16):
    sim.step()
    Mmac = macro_mass(sim.network)
    Mmsk = mask_mass_field(sim.network)
    Mtot = Mmac + Mmsk
    veh = vehicle_count(sim)
    sum_in  = sum(captured["ext_in"].values())  * dt
    sum_out = sum(captured["ext_out"].values()) * dt
    gt_d = overwrite_deltas[-1] if overwrite_deltas else 0.0
    dveh = veh - prev_veh
    # Expected (non-bridge) total change excluding bridge vehicle injection:
    expected = sum_in - sum_out + gt_d + dveh
    actual = Mtot - (prev_macro + prev_mask)
    unexplained = actual - expected
    print(f"{i:>3} {Mmac - prev_macro:>9.4f} {Mmsk - prev_mask:>9.4f} "
          f"{actual:>9.4f} {sum_in:>7.3f} {sum_out:>7.3f} {gt_d:>7.3f} "
          f"{dveh:>8d} {unexplained:>9.4f}")
    prev_macro, prev_mask, prev_veh = Mmac, Mmsk, veh
