"""Targeted probe for the catastrophic-leak phase (steps 88-92).

Two checks per step:
1) Per-mask discrete-vs-continuum ledger:
     dv_count = vehicle_count_after - vehicle_count_before
     flux_into_mask = -sum(mask_flow)*dt  (mask_flow is signed: positive = macro gains)
     discrepancy = dv_count - flux_into_mask
   If non-zero, the bridge/replayer is adding/removing vehicles without a
   matching continuum flux — that's the leak source.

2) Per-base-cell delta dump for cells touched by any mask, sorted by
   |dtotal| where dtotal = (cell.mass+cell.mask_mass)_after - same_before.
   Surfaces which cells silently lost (or gained) mass during the step.
"""
import os, sys, json
sys.path.insert(0, "/home/richarwa/SecondSSD/I24/i24_macroscopic")

import simulation as S
from simulation import Simulation, GroundTruthStore
from i24_trajectory_replayer import I24TrajectoryReplayer
from i24_micro_bridge import I24MicroSimBridge

# ----- Capture per-mask mask_flow integrated over the step -----
captured = {}
orig_compute = Simulation._compute_active_edge_flows
def patched_compute(self, active):
    e, i, o, m = orig_compute(self, active)
    per_mask = {}
    for (u, v), q in m.items():
        u_ac = active.active_cells[u]
        v_ac = active.active_cells[v]
        mid = u_ac.mask_id if u_ac.kind == "mask" else v_ac.mask_id
        per_mask[mid] = per_mask.get(mid, 0.0) + q
    captured["per_mask_flow"] = per_mask
    return e, i, o, m
Simulation._compute_active_edge_flows = patched_compute

# ----- Setup (mirrors conservation_check_mask2.py) -----
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

# ----- Helpers -----
def all_cells(sim):
    for road in sim.network.roads.values():
        for cell in road.cells.values():
            yield cell

def mask_length_in_cell(sim, base_cell):
    total = 0.0
    for mask in sim.masking_cells.values():
        for seg in mask.segments:
            if seg.road_id != base_cell.road_id or seg.lane != base_cell.lane:
                continue
            total += max(0.0, min(seg.end_s, base_cell.end_s) - max(seg.start_s, base_cell.start_s))
    return total

def cells_touched_by_any_mask(sim):
    out = set()
    for mask in sim.masking_cells.values():
        for seg in mask.segments:
            for cell in sim.network.roads[seg.road_id].cells_for_lane(seg.lane):
                if max(seg.start_s, cell.start_s) < min(seg.end_s, cell.end_s):
                    out.add((cell.road_id, cell.cell_id, cell.lane))
    return out

def snapshot(sim):
    return {
        "vehicles_per_mask": {mid: len(m.vehicles) for mid, m in sim.masking_cells.items()},
        "mass":      {(c.road_id, c.cell_id, c.lane): c.mass      for c in all_cells(sim)},
        "mask_mass": {(c.road_id, c.cell_id, c.lane): c.mask_mass for c in all_cells(sim)},
        "mlen":      {(c.road_id, c.cell_id, c.lane): mask_length_in_cell(sim, c) for c in all_cells(sim)},
        "clen":      {(c.road_id, c.cell_id, c.lane): c.length    for c in all_cells(sim)},
        "touched":   cells_touched_by_any_mask(sim),
        "bridge_s":  bridge.middle_s,
        "anchor":    bridge.anchor_speed,
    }

# ----- Run silently up to step 87 -----
for _ in range(1, 30):
    sim.step()

dt = sim.time_resolution

# ----- Instrument steps 30-92 -----
for step_i in range(30, 93):
    before = snapshot(sim)
    sim.step()
    after = snapshot(sim)

    print(f"\n========== STEP {step_i} ==========")
    print(f"  bridge_s {before['bridge_s']:.3f} -> {after['bridge_s']:.3f}   "
          f"anchor_speed {before['anchor']:.3f} -> {after['anchor']:.3f}")

    # ---- (1) Per-mask discrete-vs-continuum ledger ----
    print("  Per-mask vehicle change vs. continuum flux into mask:")
    print(f"    {'mask_id':<32s} {'dv':>5s} {'flux_in*dt':>12s} {'discrepancy':>12s}")
    mids = sorted(set(before["vehicles_per_mask"]) | set(after["vehicles_per_mask"]))
    total_dv = 0
    total_flux = 0.0
    for mid in mids:
        v_b = before["vehicles_per_mask"].get(mid, 0)
        v_a = after["vehicles_per_mask"].get(mid, 0)
        dv = v_a - v_b
        flux_in = -captured["per_mask_flow"].get(mid, 0.0) * dt
        disc = dv - flux_in
        print(f"    {mid:<32s} {dv:>5d} {flux_in:>12.4f} {disc:>12.4f}")
        total_dv += dv
        total_flux += flux_in
    print(f"    {'TOTAL':<32s} {total_dv:>5d} {total_flux:>12.4f} {total_dv - total_flux:>12.4f}")

    # ---- (2) Per-cell deltas for mask-touched cells ----
    touched = before["touched"] | after["touched"]
    rows = []
    for key in touched:
        m_b  = before["mass"].get(key, 0.0)
        m_a  = after["mass"].get(key, 0.0)
        mm_b = before["mask_mass"].get(key, 0.0)
        mm_a = after["mask_mass"].get(key, 0.0)
        ml_b = before["mlen"].get(key, 0.0)
        ml_a = after["mlen"].get(key, 0.0)
        cl   = before["clen"].get(key, after["clen"].get(key, 0.0))
        rows.append({
            "key": key,
            "dmass":      m_a - m_b,
            "dmask_mass": mm_a - mm_b,
            "dtotal":     (m_a + mm_a) - (m_b + mm_b),
            "mb": m_b, "ma": m_a,
            "kb": mm_b, "ka": mm_a,
            "mlen_b": ml_b, "mlen_a": ml_a,
            "clen": cl,
        })
    rows.sort(key=lambda r: -abs(r["dtotal"]))
    print("  Per-cell deltas (mask-touched cells, top 12 by |dtotal|):")
    print(f"    {'cell':<32s} {'dmass':>7s} {'dmask':>7s} {'dtotal':>7s} "
          f"{'mass_a':>7s} {'mask_a':>7s} {'mlen_b':>7s} {'mlen_a':>7s} {'clen':>7s}")
    for r in rows[:12]:
        ck = f"{r['key'][1]}:L{r['key'][2]}"
        print(f"    {ck:<32s} {r['dmass']:>7.3f} {r['dmask_mass']:>7.3f} {r['dtotal']:>7.3f} "
              f"{r['ma']:>7.3f} {r['ka']:>7.3f} {r['mlen_b']:>7.2f} {r['mlen_a']:>7.2f} {r['clen']:>7.2f}")
    sum_dtotal = sum(r["dtotal"] for r in rows)
    print(f"    SUM dtotal over mask-touched cells: {sum_dtotal:.4f}")
