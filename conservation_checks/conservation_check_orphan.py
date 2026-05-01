"""Verify the 'orphaned macro mass on partial->fully-masked base cells' hypothesis.

For each step, between (end of prev step active_to_base) and (start of this step's
base_to_active), check every base cell whose mask coverage went from
"partial (with non-zero mass on unmasked portion)" to "entirely masked".  Sum
that orphaned mass and compare against the macro-conservation leak.
"""
import os, sys, json
sys.path.insert(0, "/home/richarwa/SecondSSD/I24/i24_macroscopic")

import simulation as S
from simulation import Simulation, GroundTruthStore, ConservativeRemapper
from i24_trajectory_replayer import I24TrajectoryReplayer
from i24_micro_bridge import I24MicroSimBridge

captured = {}
orig_compute = Simulation._compute_active_edge_flows
def patched_compute(self, active):
    e, i, o, m = orig_compute(self, active)
    captured["mask_internal"] = dict(m); captured["ext_in"] = dict(i); captured["ext_out"] = dict(o)
    return e, i, o, m
Simulation._compute_active_edge_flows = patched_compute

overwrite_deltas = []
orig_overwrite = S.GroundTruthStore.apply_density_snapshot_to_network_boundaries
def boundary_total_mass(net):
    return sum(c.mass + c.mask_mass for r in net.roads.values() for c in r.cells.values()
               if len(c.inflow_connections) == 0 or len(c.outflow_connections) == 0)
def patched_overwrite(self, network, t, tolerance=1e-1):
    before = boundary_total_mass(network); orig_overwrite(self, network, t, tolerance)
    overwrite_deltas.append(boundary_total_mass(network) - before)
S.GroundTruthStore.apply_density_snapshot_to_network_boundaries = patched_overwrite

# Capture mask coverage at: A=end of prev step (= start of this step), B=after bridge_step+_update_masks
# Probe by patching base_to_active to record state JUST BEFORE it runs.
prev_state = {"base_mass": {}, "mask_cov": {}}  # mask_cov[(road,cell,lane)] = total mask overlap (m)
this_state = {"base_mass": {}, "mask_cov": {}}

def coverage_per_base(network, masking_cells):
    cov = {}
    for mask in masking_cells.values():
        for seg in mask.segments:
            for cell in network.roads[seg.road_id].cells_for_lane(seg.lane):
                ov = max(0.0, min(seg.end_s, cell.end_s) - max(seg.start_s, cell.start_s))
                if ov > 0:
                    k = (cell.road_id, cell.cell_id, cell.lane)
                    cov[k] = cov.get(k, 0.0) + ov
    return cov

def base_mass_per(network):
    return {(c.road_id, c.cell_id, c.lane): c.mass for r in network.roads.values() for c in r.cells.values()}

orig_b2a = ConservativeRemapper.base_to_active
def patched_b2a(simulation, network, active):
    # Snapshot state JUST BEFORE base_to_active (= post-bridge_step + post-_update_masks).
    this_state["base_mass"] = base_mass_per(network)
    this_state["mask_cov"]  = coverage_per_base(network, simulation.masking_cells)
    orig_b2a(simulation, network, active)
ConservativeRemapper.base_to_active = staticmethod(patched_b2a)

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

def macro_mass(net):
    return sum(c.mass for r in net.roads.values() for c in r.cells.values())
def cell_length(net, key):
    return net.get_cell(key[0], key[1]).length

dt = sim.time_resolution
M_macro_prev = macro_mass(sim.network)
print(f"{'step':>4} {'leak':>9} {'orphan_loss':>12} {'cells_lost'}")
for i in range(1, 85):
    # Save the END-OF-PREV-STEP state (this is what active_to_base of last step left)
    prev_state["base_mass"] = base_mass_per(sim.network)
    prev_state["mask_cov"]  = coverage_per_base(sim.network, sim.masking_cells)

    sim.step()  # this triggers the patched base_to_active which snapshots `this_state`

    # Find cells that went from partial-mask (with mass>0 on unmasked portion) to fully masked.
    orphan_loss = 0.0
    lost_cells = []
    for k, prev_mass in prev_state["base_mass"].items():
        cl = cell_length(sim.network, k)
        prev_cov = prev_state["mask_cov"].get(k, 0.0)
        new_cov  = this_state["mask_cov"].get(k, 0.0)
        prev_partial = (0.0 < prev_cov < cl - 1e-6)
        new_fully    = (new_cov  >= cl - 1e-6)
        if prev_partial and new_fully and prev_mass > 1e-6:
            orphan_loss += prev_mass
            lost_cells.append((k, prev_mass))

    M_macro_now = macro_mass(sim.network)
    sum_mask_internal = sum(captured["mask_internal"].values()) * dt if "mask_internal" in captured else 0.0
    sum_in  = sum(captured["ext_in"].values())  * dt if "ext_in" in captured else 0.0
    sum_out = sum(captured["ext_out"].values()) * dt if "ext_out" in captured else 0.0
    gt_delta = overwrite_deltas[-1]
    expected = sum_in - sum_out + gt_delta + sum_mask_internal
    leak = (M_macro_now - M_macro_prev) - expected
    M_macro_prev = M_macro_now

    if abs(leak) > 0.05 or orphan_loss > 1e-6:
        cells_str = ", ".join(f"{k[1]}:L{k[2]}={m:.3f}" for k,m in lost_cells)
        print(f"{i:>4} {leak:>9.4f} {orphan_loss:>12.4f}  {cells_str}")
