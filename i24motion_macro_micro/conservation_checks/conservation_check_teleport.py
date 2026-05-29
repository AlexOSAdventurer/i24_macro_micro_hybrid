"""Test the 'mask teleport between steps' hypothesis for sporadic leaks.

Hypothesis: at start of each step, the bridge replaces the mask object with
a new one centred at `bridge.middle_s + new_anchor*dt`, while the previous
step's `move_active_masks` shifted the mask geometry by `old_anchor*dt`.
When new_anchor != old_anchor, the mask geometry jumps discontinuously by
`(new_anchor - old_anchor) * dt` BETWEEN steps. base_to_active then
redistributes base.mass across new boundaries that don't match where
active_to_base last wrote — breaking macro conservation.

We instrument:
  - mask.segments[0] state captured at three points per step:
      A: end of previous step (= same as start-of-step before any callbacks)
      B: after bridge_step / _update_masks (mask object replaced)
      C: after move_active_masks (mask shifted by anchor*dt)
  - the macro-conservation leak (using mask2's ledger).
"""
import os, sys, json
sys.path.insert(0, ".")

import simulation as S
from simulation import Simulation, GroundTruthStore, ConservativeRemapper
from i24_trajectory_replayer import I24TrajectoryReplayer
from i24_micro_bridge import I24MicroSimBridge

# ---- Capture macro-conservation ledger (same as mask2) ----
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
def boundary_total_mass(net):
    return sum(c.mass + c.mask_mass for r in net.roads.values() for c in r.cells.values()
               if len(c.inflow_connections) == 0 or len(c.outflow_connections) == 0)
def patched_overwrite(self, network, t, tolerance=1e-1):
    before = boundary_total_mass(network)
    orig_overwrite(self, network, t, tolerance)
    overwrite_deltas.append(boundary_total_mass(network) - before)
S.GroundTruthStore.apply_density_snapshot_to_network_boundaries = patched_overwrite

# ---- Capture mask geometry before/after move_active_masks ----
geom_capture = {"before_move": None, "after_move": None}
orig_move = ConservativeRemapper.move_active_masks
def patched_move(simulation, active):
    geom_capture["before_move"] = {
        mid: (m.segments[0].start_s, m.segments[0].end_s, getattr(m, 'anchor_speed', 0.0))
        for mid, m in simulation.masking_cells.items()
    }
    orig_move(simulation, active)
    geom_capture["after_move"] = {
        mid: (m.segments[0].start_s, m.segments[0].end_s, getattr(m, 'anchor_speed', 0.0))
        for mid, m in simulation.masking_cells.items()
    }
ConservativeRemapper.move_active_masks = staticmethod(patched_move)

# ---- Setup ----
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

dt = sim.time_resolution
# Snapshot end-of-step state for each mask
prev_end = None  # dict mid -> (start_s, end_s, anchor) at end of previous step
M_macro_prev = macro_mass(sim.network)

print(f"{'step':>4} {'leak':>8} {'teleport_dx':>13} {'old_anchor':>11} {'new_anchor':>11} {'shift_dx':>10}  notes")
for i in range(1, 85):
    sim.step()
    # geom_capture['before_move'] = mask state JUST BEFORE move_active_masks (= post bridge_step + post _build_active_network)
    # geom_capture['after_move']  = post move_active_masks
    # 'prev_end' = state at end of last step (= state at start of this step's bridge_step)
    # Teleport between steps = before_move (post-bridge) - prev_end (post-prev-move)
    if prev_end is not None:
        for mid in geom_capture["before_move"]:
            if mid not in prev_end:
                continue
            old_s, _, old_a = prev_end[mid]
            new_s, _, new_a = geom_capture["before_move"][mid]
            shift_s, _, _ = geom_capture["after_move"][mid]
            teleport = new_s - old_s
            shift_dx = shift_s - new_s
            # Only print one mask (lane -1) to stay readable
            if mid == "micro_mask_2_lane-1":
                M_macro_now = macro_mass(sim.network)
                M_macro_delta = M_macro_now - M_macro_prev
                sum_mask_internal = sum(captured["mask_internal"].values()) * dt if "mask_internal" in captured else 0.0
                sum_in  = sum(captured["ext_in"].values())  * dt if "ext_in" in captured else 0.0
                sum_out = sum(captured["ext_out"].values()) * dt if "ext_out" in captured else 0.0
                gt_delta = overwrite_deltas[-1]
                expected = sum_in - sum_out + gt_delta + sum_mask_internal
                leak = M_macro_delta - expected
                M_macro_prev = M_macro_now
                marker = " <-- LEAK" if abs(leak) > 0.05 else ""
                print(f"{i:>4} {leak:>8.4f} {teleport:>13.4f} {old_a:>11.4f} {new_a:>11.4f} {shift_dx:>10.4f}{marker}")
    prev_end = dict(geom_capture["after_move"])
