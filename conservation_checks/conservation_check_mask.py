"""Verify conservation with a moving mask: dM_macro should equal
   external_in − external_out + GT_overwrite − mass_into_mask."""
import os, sys, json
sys.path.insert(0, "/home/richarwa/SecondSSD/I24/i24_macroscopic")

import simulation as S
from simulation import Simulation, GroundTruthStore
from i24_trajectory_replayer import I24TrajectoryReplayer
from i24_micro_bridge import I24MicroSimBridge

def total_macro_mass(network):
    """Sum of cell.mass — does NOT include mask_mass (which tracks the discrete domain)."""
    return sum(c.mass for road in network.roads.values() for c in road.cells.values())
def total_mask_mass_field(network):
    return sum(c.mask_mass for road in network.roads.values() for c in road.cells.values())
def boundary_macro_mass(network):
    return sum(c.mass for road in network.roads.values() for c in road.cells.values()
               if len(c.inflow_connections) == 0 or len(c.outflow_connections) == 0)

# Capture per-step external in/out
captured = {}
orig_compute = Simulation._compute_active_edge_flows
def patched_compute(self, active):
    e, i, o = orig_compute(self, active)
    captured["ext_in"]  = dict(i)
    captured["ext_out"] = dict(o)
    return e, i, o
Simulation._compute_active_edge_flows = patched_compute

# Capture GT overwrite delta on macro mass
overwrite_deltas = []
orig_overwrite = S.GroundTruthStore.apply_density_snapshot_to_network_boundaries
def patched_overwrite(self, network, t, tolerance=1e-1):
    before = boundary_macro_mass(network)
    orig_overwrite(self, network, t, tolerance)
    overwrite_deltas.append(boundary_macro_mass(network) - before)
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
print(f"{'step':>4} {'dM_macro':>10} {'ext_in*dt':>10} {'ext_out*dt':>11} {'GT_dM':>9} {'into_mask':>11} {'leak':>10} {'mask_mass':>10}")

M_prev = total_macro_mass(sim.network)
for i in range(1, 16):
    sim.step()
    M_now = total_macro_mass(sim.network)

    # Net flux INTO the masks this step. masking_cells dict is mutated by bridge,
    # so its values right now are the masks that LIVED THROUGH step i.
    net_into_mask = sum(m.rear_flow + m.front_flow for m in sim.masking_cells.values())

    sum_in  = sum(captured["ext_in"].values())  * dt
    sum_out = sum(captured["ext_out"].values()) * dt
    gt_d = overwrite_deltas[-1] if overwrite_deltas else 0.0
    expected = sum_in - sum_out + gt_d - net_into_mask
    actual = M_now - M_prev
    leak = actual - expected
    Mm = total_mask_mass_field(sim.network)
    print(f"{i:>4} {actual:>10.4f} {sum_in:>10.4f} {sum_out:>11.4f} {gt_d:>9.4f} {net_into_mask:>11.4f} {leak:>10.6f} {Mm:>10.4f}")
    M_prev = M_now
