import glob
import math
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# We assume all entries have the same time step
def generate_ground_truth_alignment_with_simulation_result(gt: pd.DataFrame, sim_macro: pd.DataFrame, sim_mask: pd.DataFrame, eps: float = 1e-8):
    mask_position = float(sim_mask.iloc[0]["x_start_position"])
    mask_length = float(sim_mask.iloc[0]["length"])
    sim_macro = sim_macro[(sim_macro["x_start_position"] - mask_position).abs() > eps]

    t = float(sim_macro.iloc[0]["time"])
    dt = float(sim_macro.iloc[0]["dt"])
    assert((gt["time"].nunique() == 1) and (float(gt["time"].iloc[0]) == t)), f"{t}, {gt}, {sim_macro}, {sim_mask}"
    assert((sim_macro["time"].nunique() == 1))
    assert((sim_mask["time"].nunique() == 1) and (float(sim_mask["time"].iloc[0]) == t))
    assert((sim_macro["dt"].nunique() == 1))
    assert((sim_mask["dt"].nunique() == 1) and (float(sim_mask["dt"].iloc[0]) == dt))

    time_array = []
    dt_array = []
    x_start_position_array = []
    length_array = []
    density_array = []
    for i in range(len(sim_macro)):
        macro_data = sim_macro.iloc[i]
        new_time = t
        new_dt = dt
        x_start_position_new = float(macro_data["x_start_position"])
        length_new = float(macro_data["length"])
        mass_new = 0.0
        for j in range(len(gt)):
            gt_data = gt.iloc[j]
            gt_position = float(gt_data["position"])
            gt_length = float(gt_data["length"])
            gt_density = float(gt_data["density"])
            overlap_start = max(x_start_position_new, gt_position)
            overlap_end = min(gt_position + gt_length, x_start_position_new + length_new)
            overlap_length = overlap_end - overlap_start
            if (overlap_length > eps):
                mass_new += (overlap_length * gt_density)
        density_new = mass_new / length_new
        time_array.append(new_time)
        dt_array.append(new_dt)
        x_start_position_array.append(x_start_position_new)
        length_array.append(length_new)
        density_array.append(density_new)

    sim_macro = sim_macro.reset_index(drop=True)
    return pd.DataFrame({
        "time": time_array,
        "dt": dt_array,
        "x_start_position": x_start_position_array,
        "length": length_array,
        "density": density_array
    }), sim_macro

def generate_ground_truth_and_sim_result(gt: pd.DataFrame, sim_macro: pd.DataFrame, sim_mask: pd.DataFrame, eps: float = 1e-8):
    gt = gt.sort_values(["time", "position"], kind="mergesort", ascending=True)
    sim_macro = sim_macro.sort_values(["time", "x_start_position"], kind="mergesort", ascending=True)
    sim_mask = sim_mask.sort_values(["time", "x_start_position"], kind="mergesort", ascending=True)
    # By default, remove the first and last x position from sim_macro and sim_mask - those are always being overwritten anyway
    x_positions = sorted(sim_macro["x_start_position"].unique().tolist())
    sim_macro = sim_macro[(sim_macro["x_start_position"] > x_positions[0]) & (sim_macro["x_start_position"] < x_positions[-1])]
    times = sim_macro["time"].unique().tolist()
    tasks = []
    for time in times:
        sim_macro_first_index = sim_macro["time"].searchsorted(time, side="left")
        sim_macro_last_index = sim_macro["time"].searchsorted(time, side="right")
        sim_mask_first_index = sim_mask["time"].searchsorted(time, side="left")
        sim_mask_last_index = sim_mask["time"].searchsorted(time, side="right")
        gt_first_index = gt["time"].searchsorted(time, side="left")
        gt_last_index = gt["time"].searchsorted(time, side="right")
        sim_macro_time = sim_macro[sim_macro_first_index:sim_macro_last_index]
        sim_mask_time = sim_mask[sim_mask_first_index:sim_mask_last_index]
        #sim_macro_time = sim_macro[(sim_macro["time"] - time).abs() <= eps]
        #sim_mask_time = sim_mask[(sim_mask["time"] - time).abs() <= eps]
        gt_time = gt[gt_first_index:gt_last_index]
        tasks.append({
            "gt": gt_time,
            "sim_macro": sim_macro_time,
            "sim_mask": sim_mask_time
        })
        print(time)
    df_results = Parallel(n_jobs=24)(delayed(generate_ground_truth_alignment_with_simulation_result)(gt=task["gt"], sim_macro=task["sim_macro"], sim_mask=task["sim_mask"]) for task in tasks)
    gt_step_list = []
    sim_macro_step_list = []
    for (gt_step, sim_macro_step) in df_results:
        gt_step_list.append(gt_step)
        sim_macro_step_list.append(sim_macro_step)
    gt_result = pd.concat(gt_step_list, axis=0, ignore_index=True)
    macro_result = pd.concat(sim_macro_step_list, axis=0, ignore_index=True)
    return gt_result, macro_result


def generate_convergence_order(ground_truth_file_coarse: str, ground_truth_file_fine: str, sim_macro_coarse_file: str, sim_mask_coarse_file: str, sim_macro_fine_file: str, sim_mask_fine_file: str, scaling: float):
    """Two-grid L1 convergence order.

    WARNING: this excludes the mask cell (see the filter in
    generate_ground_truth_alignment_with_simulation_result), so it is blind to any mass
    error living inside the micro bubble. That blindness inverted the case_m_4 model
    ranking and reported case_m_2 as exactly convergent when the complete norm is flat.
    Prefer convergence_table() below, which reports the macro and bubble terms separately
    and sums them. Kept because existing notebooks call it.
    """
    lwr_ground_truth_coarse = pd.read_csv(ground_truth_file_coarse)
    lwr_ground_truth_fine = pd.read_csv(ground_truth_file_fine)
    sim_macro_coarse = pd.read_csv(sim_macro_coarse_file)
    sim_mask_coarse = pd.read_csv(sim_mask_coarse_file)
    sim_macro_fine = pd.read_csv(sim_macro_fine_file)
    sim_mask_fine = pd.read_csv(sim_mask_fine_file)
    coarse_dt = float(sim_macro_coarse.iloc[0]["dt"])
    fine_dt = float(sim_macro_fine.iloc[0]["dt"])
    lwr_ground_truth_coarse_post, sim_macro_coarse_post = generate_ground_truth_and_sim_result(lwr_ground_truth_coarse, sim_macro_coarse, sim_mask_coarse)
    lwr_ground_truth_fine_post, sim_macro_fine_post = generate_ground_truth_and_sim_result(lwr_ground_truth_fine, sim_macro_fine, sim_mask_fine)
    lwr_ground_truth_coarse_post["mass"] = lwr_ground_truth_coarse_post["density"] * lwr_ground_truth_coarse_post["length"]
    lwr_ground_truth_fine_post["mass"] = lwr_ground_truth_fine_post["density"] * lwr_ground_truth_fine_post["length"]
    sim_macro_coarse_post["mass"] = sim_macro_coarse_post["density"] * sim_macro_coarse_post["length"]
    sim_macro_fine_post["mass"] = sim_macro_fine_post["density"] * sim_macro_fine_post["length"]
    coarse_l1_error = np.absolute(sim_macro_coarse_post["mass"].values - lwr_ground_truth_coarse_post["mass"].values).sum() * coarse_dt
    fine_l1_error = np.absolute(sim_macro_fine_post["mass"].values - lwr_ground_truth_fine_post["mass"].values).sum() * fine_dt
    #return coarse_l1_error, fine_l1_error, math.log(coarse_l1_error / fine_l1_error) / math.log(scaling)
    #coarse_mse_error = ((sim_macro_coarse_post["mass"] -  lwr_ground_truth_coarse_post["mass"]) ** 2).sum() / (sim_macro_coarse_post["length"].sum())
    #fine_mse_error = ((sim_macro_fine_post["mass"] -  lwr_ground_truth_fine_post["mass"]) ** 2).sum() / (sim_macro_fine_post["length"].sum())
    #coarse_raw_error = (sim_macro_coarse_post["mass"] - lwr_ground_truth_coarse_post["mass"]).abs().sum()
    #fine_raw_error = (sim_macro_fine_post["mass"] - lwr_ground_truth_fine_post["mass"]).abs().sum()
    #return coarse_raw_error, fine_raw_error
    return lwr_ground_truth_coarse_post, sim_macro_coarse_post, lwr_ground_truth_fine_post, sim_macro_fine_post, coarse_l1_error, fine_l1_error


#analysis_helpers.generate_convergence_order("verification_config/case_m_2_dx_128.0_dt_1.0/vanilla_lwr.csv", "verification_config/case_m_2_dx_64.0_dt_0.5/vanilla_lwr.csv", "verification_results/case_m_2_dx_128.0_dt_1.0/newell_case_macro.csv", "verification_results/case_m_2_dx_128.0_dt_1.0/newell_case_mask.csv", "verification_results/case_m_2_dx_64.0_dt_0.5/newell_case_macro.csv", "verification_results/case_m_2_dx_64.0_dt_0.5/newell_case_mask.csv", 2.0)
# (np.float64(427.0002880439491), np.float64(827.9486403238341))
#(np.float64(427.0002880439491), np.float64(413.97432016191703))
#>>> analysis_helpers.generate_convergence_order("verification_config/case_m_2_dx_128.0_dt_1.0/vanilla_lwr.csv", "verification_config/case_m_2_dx_64.0_dt_0.5/vanilla_lwr.csv", "verification_results/case_m_2_dx_128.0_dt_1.0/newell_case_macro.csv", "verification_results/case_m_2_dx_128.0_dt_1.0/newell_case_mask.csv", "verification_results/case_m_2_dx_64.0_dt_0.5/newell_case_macro.csv", "verification_results/case_m_2_dx_64.0_dt_0.5/newell_case_mask.csv", 2.0)
# analysis_helpers.generate_convergence_order("verification_config/case_m_2_dx_128.0_dt_1.0/micro_case_b_1.csv", "verification_config/case_m_2_dx_64.0_dt_0.5/micro_case_b_1.csv", "verification_results/case_m_2_dx_128.0_dt_1.0/micro_case_b_1_macro.csv", "verification_results/case_m_2_dx_128.0_dt_1.0/micro_case_b_1_mask.csv", "verification_results/case_m_2_dx_64.0_dt_0.5/micro_case_b_1_macro.csv", "verification_results/case_m_2_dx_64.0_dt_0.5/micro_case_b_1_mask.csv", 2.0)
# analysis_helpers.generate_convergence_order("verification_config/case_m_2_dx_128.0_dt_1.0/micro_case_b_1.csv", "verification_config/case_m_2_dx_8.0_dt_0.0625/micro_case_b_1.csv", "verification_results/case_m_2_dx_128.0_dt_1.0/micro_case_b_1_macro.csv", "verification_results/case_m_2_dx_128.0_dt_1.0/micro_case_b_1_mask.csv", "verification_results/case_m_2_dx_8.0_dt_0.0625/micro_case_b_1_macro.csv", "verification_results/case_m_2_dx_8.0_dt_0.0625/micro_case_b_1_mask.csv", 2.0)


# ======================================================================================
# Convergence tables
#
# Replaces the pairwise generate_convergence_order() above with a whole-sweep table. Four
# differences, each of which changed a published conclusion at some point:
#
#   1. THE BUBBLE TERM. The macro L1 excludes the mask cell, so mass error inside the
#      micro domain is invisible to it. That systematically rewards models which leak
#      mass OUT of the bubble: on case_m_4 it hid a 2.7-vehicle deficit in IDM/ARZ while
#      charging Newell in full for mass it was correctly holding, inverting the ranking.
#      `L1 bubble` = |vehicles in mask - GT integrated over the mask window| * dt.
#
#   2. THE MASK IS LOCATED BY LENGTH, NOT BY THE MASK LOG'S POSITION. The mask CSV's
#      x_start_position lags the active mesh by anchor_speed*dt, because the bridge only
#      refreshes middle_s on the NEXT step. The mask cell is the one whose length is
#      2*margin_s rather than dx, so match on that instead.
#
#   3. VECTORISED OVERLAP. The exact piecewise-constant overlap integral is done with two
#      clipped broadcasts rather than a Python double loop over (cell, gt_region).
#
#   4. STREAMING. Macro logs reach 7 GB at dx=4/dt=0.03125. Rows are written in step
#      order, so time groups are contiguous and can be consumed chunk by chunk.
# ======================================================================================

GRID_SWEEP = ("128.0_dt_1.0", "64.0_dt_0.5", "32.0_dt_0.25",
              "16.0_dt_0.125", "8.0_dt_0.0625", "4.0_dt_0.03125")

MACRO_CHUNK_ROWS = 2_000_000


def gt_mass_on_intervals(gt_starts, gt_ends, gt_densities, cell_starts, cell_ends):
    """Exact overlap integral of a piecewise-constant field onto arbitrary intervals.

    The GT regions need not tile the domain: every prescribed b-case leaves a genuine gap
    where the mask sits. Integrate over each cell's own [start, end] rather than over
    consecutive edges -- edges built from consecutive starts span that gap, and spanning
    it charges the full GT mass lying under the mask to the neighbouring cell.
    """
    def cumulative(x):
        clipped = np.clip(x[:, None], gt_starts[None, :], gt_ends[None, :])
        return ((clipped - gt_starts[None, :]) * gt_densities[None, :]).sum(axis=1)
    return cumulative(cell_ends) - cumulative(cell_starts)


def _iter_macro_time_groups(path, chunksize=MACRO_CHUNK_ROWS):
    """Yield (time, frame) for a macro log without holding the file in memory.

    Logger writes every cell for one step before moving to the next, so a time group is
    contiguous and the time column is non-decreasing. The trailing group of each chunk may
    be incomplete, so it is carried into the next chunk rather than yielded.
    """
    columns = ["time", "x_start_position", "length", "density"]
    carry = None
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize):
        if carry is not None and len(carry):
            chunk = pd.concat([carry, chunk], ignore_index=True)
        carry = None
        times = chunk["time"].values
        cut = np.searchsorted(times, times[-1], side="left")
        if cut == 0:
            carry = chunk          # the whole chunk is one step; need more rows
            continue
        head = chunk.iloc[:cut]
        carry = chunk.iloc[cut:].reset_index(drop=True)
        for t, group in head.groupby("time", sort=False):
            yield float(t), group
    if carry is not None and len(carry):
        for t, group in carry.groupby("time", sort=False):
            yield float(t), group


def default_gt_case(micro_case):
    """Prescribed b-cases are scored against their own exact solution; the genuine
    car-following models are scored against the mask-free LWR solution."""
    return micro_case if micro_case.startswith("micro_case_b_") else "vanilla_lwr"


def _analyse_mask_free_run(grid, macro_path, gt_path):
    """Macro-only error for a run with no micro domain (the mask-free control).

    Returns the same keys as analyse_run so the two are interchangeable in a table; the
    bubble and cumulative-count terms are NaN because there is no micro region to measure,
    not because they failed to compute.
    """
    gt = pd.read_csv(gt_path)
    gt_groups = {float(t): g for t, g in gt.groupby("time")}
    dt = None
    macro = upstream = 0.0
    for t, step in _iter_macro_time_groups(macro_path):
        gt_step = gt_groups.get(t)
        if gt_step is None:
            continue
        if dt is None:
            times = sorted(gt_groups)
            dt = (times[1] - times[0]) if len(times) > 1 else 1.0
        gt_starts = gt_step["position"].values
        gt_ends = gt_starts + gt_step["length"].values
        step = step.sort_values("x_start_position").iloc[1:-1]
        if not len(step):
            continue
        positions = step["x_start_position"].values
        lengths = step["length"].values
        error = np.abs(step["density"].values * lengths
                       - gt_mass_on_intervals(gt_starts, gt_ends,
                                              gt_step["density"].values,
                                              positions, positions + lengths))
        macro += error.sum() * dt
        upstream += error.sum() * dt
    return {
        "grid": grid, "macro": macro, "upstream": upstream, "downstream": 0.0,
        "bubble": float("nan"), "total": macro,
        "moskowitz": float("nan"), "moskowitz_mean": float("nan"),
    }


def analyse_run(macro_case, micro_case, grid, results_folder,
                gt_case=None, config_folder="verification_config", tolerance=1e-6,
                bubble_source="vehicle_count"):
    """Error of one run: macro (upstream/downstream), bubble, and cumulative-count.

    Returns a dict. `bubble`, `moskowitz` and `moskowitz_mean` are NaN when the ground
    truth deliberately leaves a gap under the mask, which every prescribed b-case does --
    there is nothing to compare against, and charging the whole window would be fiction.

    `macro + bubble` (reported as `total`) is a single consistent functional: both terms
    are cell-wise lumped mass errors, sum_cells |integral over cell of (rho_sim - rho_gt)|,
    and the mask is simply one large cell. Since that sum tends to the true L1 density
    error as cells shrink, `total` is a legitimate L1 measure.

    Its one structural caveat: resolution is non-uniform and stays that way. Outside the
    mask, cells shrink with dx and cancellation within a cell vanishes; the mask window
    stays 2*margin_s wide at every grid, so misarrangement inside the bubble stays
    invisible. The bubble is therefore under-penalised relative to the fluid region, and
    increasingly so as dx falls. Fixing that needs vehicle positions (Edie density over
    the window), which Logger does not currently record.

    `moskowitz` is the alternative with uniform treatment -- see the note at its
    computation below -- at the cost of being a weaker norm.

    bubble_source selects what counts as "mass in the bubble":
      "vehicle_count" (default) -- len(vehicles) from the mask log. Always an integer, and
          the honest answer to how much traffic the micro domain is actually holding.
      "fluid" -- the mask cell's density*length from the macro log. The fluid side's
          bookkeeping of the same quantity, which carries an O(dt) ALE operator-splitting
          residual: measured against case_m_4/newell, the gap has max |.| of 6.461, 3.231,
          1.615 at dt = 1.0, 0.5, 0.25, halving exactly with dt. Use it only to reproduce
          results produced before the vehicle_count column existed.
    Older logs have no vehicle_count column, and fall back to "fluid" automatically.
    """
    if bubble_source not in ("vehicle_count", "fluid"):
        raise ValueError(f"unknown bubble_source {bubble_source!r}")
    gt_case = gt_case or default_gt_case(micro_case)
    run_dir = os.path.join(results_folder, f"{macro_case}_dx_{grid}")
    macro_path = os.path.join(run_dir, f"{micro_case}_macro.csv")
    mask_path = os.path.join(run_dir, f"{micro_case}_mask.csv")
    gt_path = os.path.join(config_folder, f"{macro_case}_dx_{grid}", f"{gt_case}.csv")

    # The mask-free control writes no mask log at all. There is no bubble and no mask cell
    # to exclude, so every term but the macro one is undefined; dt comes off the macro log.
    has_mask = os.path.exists(mask_path)
    if not has_mask:
        return _analyse_mask_free_run(grid, macro_path, gt_path)

    mask_log = pd.read_csv(mask_path)
    mask_length = float(mask_log["length"].iloc[0])
    dt = float(mask_log["dt"].iloc[0])
    # Logger gained a vehicle_count column partway through the study; derive it from the
    # mask cell's mass when reading an older log, since mask.mass == len(vehicles).
    counts = (mask_log.set_index("time")["vehicle_count"].to_dict()
              if (bubble_source == "vehicle_count"
                  and "vehicle_count" in mask_log.columns) else None)
    last_mask_position = mask_log.set_index("time")["x_start_position"].to_dict()

    gt = pd.read_csv(gt_path)
    gt_groups = {float(t): g for t, g in gt.groupby("time")}

    macro = upstream = downstream = bubble = 0.0
    moskowitz = measure = 0.0
    bubble_comparable = True
    for t, step in _iter_macro_time_groups(macro_path):
        gt_step = gt_groups.get(t)
        if gt_step is None:
            continue
        gt_starts = gt_step["position"].values
        gt_ends = gt_starts + gt_step["length"].values
        gt_densities = gt_step["density"].values

        # Drop the road's two end cells: they are overwritten by the boundary conditions
        # every step, so they measure the boundary treatment rather than the scheme.
        step = step.sort_values("x_start_position").iloc[1:-1]
        if not len(step):
            continue
        positions = step["x_start_position"].values
        lengths = step["length"].values
        # Every cell's mass on both sides, mask included. The cumulative-count metric
        # below needs the mask in the running total, so it is computed before the mask is
        # split out for the macro term.
        sim_masses = step["density"].values * lengths
        gt_masses = gt_mass_on_intervals(gt_starts, gt_ends, gt_densities,
                                         positions, positions + lengths)

        is_mask = np.abs(lengths - mask_length) < tolerance
        if is_mask.any():
            index = int(np.argmax(is_mask))
            low = float(positions[index])
            high = low + mask_length
            covered = float(gt_mass_on_intervals(
                gt_starts, gt_ends, np.ones_like(gt_densities),
                np.array([low]), np.array([high]))[0])
            if covered < 0.99 * mask_length:
                bubble_comparable = False
            else:
                if counts is not None and t in counts:
                    held = float(counts[t])
                else:
                    held = float(sim_masses[index])
                bubble += abs(held - gt_masses[index]) * dt
                sim_masses[index] = held
        else:
            # The bridge is torn down once the window reaches max_middle_s; nothing to
            # exclude, but still split about its last logged position.
            low = float(last_mask_position.get(t, np.inf))
            high = low + mask_length

        # --- cumulative-count (Moskowitz) metric --------------------------------------
        # N(x, t) = vehicles between the upstream edge of the analysed region and x. LWR
        # is the Hamilton-Jacobi equation for N, and N is exactly defined on both sides of
        # the coupling -- continuum by integrating density, micro by literal vehicle count
        # -- so |N_sim - N_gt| compares the two halves in one norm rather than summing a
        # pointwise density error against a lumped mass error. Both curves are anchored to
        # zero at the same upstream edge, so this measures redistribution, not the offset
        # inherited from the dropped boundary cell.
        delta_n = np.cumsum(sim_masses - gt_masses)
        moskowitz += (np.abs(delta_n) * lengths).sum() * dt
        measure += lengths.sum() * dt

        if is_mask.any():
            keep = ~is_mask
            step, positions, lengths = step[keep], positions[keep], lengths[keep]
            gt_masses, sim_masses = gt_masses[keep], sim_masses[keep]

        error = np.abs(sim_masses - gt_masses)
        macro += error.sum() * dt
        upstream += error[positions < low].sum() * dt
        downstream += error[positions >= high].sum() * dt

    return {
        "grid": grid,
        "macro": macro,
        "upstream": upstream,
        "downstream": downstream,
        "bubble": bubble if bubble_comparable else float("nan"),
        "total": macro + (bubble if bubble_comparable else 0.0),
        # Raw units are vehicles*m*s, which depend on road length and run duration; the
        # mean is the interpretable one: the average number of vehicles by which the
        # hybrid's cumulative-count surface is displaced from LWR's.
        "moskowitz": moskowitz if bubble_comparable else float("nan"),
        "moskowitz_mean": (moskowitz / measure
                           if (bubble_comparable and measure > 0.0) else float("nan")),
    }


def convergence_table(macro_case, micro_case, results_folder,
                      gt_case=None, grids=GRID_SWEEP, scaling=2.0,
                      config_folder="verification_config",
                      bubble_source="vehicle_count"):
    """Convergence table across a grid sweep, with observed orders.

    `p_macro` is the order of the macro term alone, `p_total` the order of macro+bubble.
    Report p_total: the macro half converges at the scheme's contact/shock rate in every
    scenario, but the bubble term is a grid-independent floor set by the micro model, so
    p_total decays once the floor dominates. That is a genuine property of coupling a
    continuum to a fixed number of discrete vehicles, not a defect.
    """
    rows = []
    for grid in grids:
        try:
            rows.append(analyse_run(macro_case, micro_case, grid, results_folder,
                                    gt_case=gt_case, config_folder=config_folder,
                                    bubble_source=bubble_source))
        except FileNotFoundError:
            continue
    table = pd.DataFrame(rows)
    if table.empty:
        return table

    def orders(values):
        out = [float("nan")]
        for previous, current in zip(values[:-1], values[1:]):
            usable = (previous > 0.0 and current > 0.0
                      and np.isfinite(previous) and np.isfinite(current))
            out.append(math.log(previous / current) / math.log(scaling) if usable
                       else float("nan"))
        return out

    table["p_macro"] = orders(table["macro"].tolist())
    table["p_total"] = orders(table["total"].tolist())
    table["p_moskowitz"] = orders(table["moskowitz"].tolist())
    table.attrs["macro_case"] = macro_case
    table.attrs["micro_case"] = micro_case
    table.attrs["gt_case"] = gt_case or default_gt_case(micro_case)
    return table


def format_convergence_table(table):
    """Fixed-width rendering, with n/a where the bubble term is not comparable."""
    if table.empty:
        return "(no runs found)"
    header = (f"{table.attrs.get('micro_case', '?')}  vs GT "
              f"{table.attrs.get('gt_case', '?')}  /  {table.attrs.get('macro_case', '?')}")
    lines = [header,
             f"{'grid':>16} {'L1 macro':>12} {'L1 upstream':>12} {'L1 downstr':>12}"
             f" {'L1 bubble':>11} {'L1 ALL':>12} {'p_mac':>7} {'p_all':>7}"
             f" {'mean|dN|':>9} {'p_N':>7}"]
    for row in table.itertuples():
        bubble = "        n/a" if math.isnan(row.bubble) else f"{row.bubble:11.4f}"
        total = "         n/a" if math.isnan(row.bubble) else f"{row.total:12.4f}"
        mean_dn = ("      n/a" if math.isnan(row.moskowitz_mean)
                   else f"{row.moskowitz_mean:9.4f}")
        p_macro = "" if math.isnan(row.p_macro) else f"{row.p_macro:7.3f}"
        p_total = "" if math.isnan(row.p_total) else f"{row.p_total:7.3f}"
        p_n = "" if math.isnan(row.p_moskowitz) else f"{row.p_moskowitz:7.3f}"
        lines.append(f"{row.grid:>16} {row.macro:12.4f} {row.upstream:12.4f}"
                     f" {row.downstream:12.4f} {bubble} {total} {p_macro:>7} {p_total:>7}"
                     f" {mean_dn:>9} {p_n:>7}")
    lines.append("  L1 macro  = mask cell EXCLUDED (blind to error inside the bubble)")
    lines.append("  L1 bubble = |vehicles in mask - GT over the mask window| * dt")
    lines.append("  L1 ALL    = macro + bubble; cell-wise lumped mass error, both terms the same")
    lines.append("              functional. Resolution is dx outside the mask but stays 2*margin_s")
    lines.append("              inside it, so the bubble is under-penalised at fine grids.")
    lines.append("  mean|dN|  = mean cumulative-count displacement, in vehicles (single norm, mask included)")
    lines.append("  n/a       = GT leaves a gap under the mask (all prescribed b-cases)")
    return "\n".join(lines)


def discover_micro_cases(results_folder, macro_case, grid=GRID_SWEEP[0]):
    """Micro cases present for a macro scenario, read off the result folder."""
    pattern = os.path.join(results_folder, f"{macro_case}_dx_{grid}", "*_macro.csv")
    return sorted(os.path.basename(p)[:-len("_macro.csv")] for p in glob.glob(pattern))


def sweep_tables(results_folder, macro_cases=("case_m_1", "case_m_2", "case_m_3", "case_m_4"),
                 micro_cases=None, grids=GRID_SWEEP, scaling=2.0, n_jobs=1, verbose=True):
    """Convergence tables for every (macro case, micro case) pair in the sweep.

    n_jobs > 1 parallelises across pairs. Each worker streams one macro log at a time, so
    peak memory is roughly n_jobs * (one chunk + one ground truth), not n_jobs * 7 GB.
    """
    pairs = []
    for macro_case in macro_cases:
        for micro_case in (micro_cases or discover_micro_cases(results_folder, macro_case)):
            pairs.append((macro_case, micro_case))

    def one(pair):
        return pair, convergence_table(pair[0], pair[1], results_folder,
                                       grids=grids, scaling=scaling)

    if n_jobs == 1:
        results = [one(pair) for pair in pairs]
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(one)(pair) for pair in pairs)

    tables = {}
    for pair, table in results:
        tables[pair] = table
        if verbose:
            print(format_convergence_table(table))
            print()
    return tables


def bubble_decomposition(macro_case, micro_case, results_folder, grids=GRID_SWEEP,
                         gt_case=None, config_folder="verification_config",
                         tolerance=1e-6):
    """Split the bubble term into the vehicle count and the GT window mass.

    Written to diagnose a non-monotonic bubble term: `held` is grid-independent (the micro
    representation does not refine), so any grid dependence in |held - gt_window| must come
    from the reference. That is how the case_m_3/case_m_4 initial interface was found to
    sit one cell too far downstream, which made |.| fold into a spurious V as gt_window
    swept across the fixed vehicle count.
    """
    gt_case = gt_case or default_gt_case(micro_case)
    rows = []
    for grid in grids:
        run_dir = os.path.join(results_folder, f"{macro_case}_dx_{grid}")
        macro_path = os.path.join(run_dir, f"{micro_case}_macro.csv")
        mask_path = os.path.join(run_dir, f"{micro_case}_mask.csv")
        gt_path = os.path.join(config_folder, f"{macro_case}_dx_{grid}", f"{gt_case}.csv")
        if not (os.path.exists(macro_path) and os.path.exists(gt_path)):
            continue
        mask_log = pd.read_csv(mask_path)
        mask_length = float(mask_log["length"].iloc[0])
        dt = float(mask_log["dt"].iloc[0])
        gt = pd.read_csv(gt_path)
        gt_groups = {float(t): g for t, g in gt.groupby("time")}

        held_series, window_series = [], []
        for t, step in _iter_macro_time_groups(macro_path):
            gt_step = gt_groups.get(t)
            if gt_step is None:
                continue
            lengths = step["length"].values
            is_mask = np.abs(lengths - mask_length) < tolerance
            if not is_mask.any():
                continue
            index = int(np.argmax(is_mask))
            low = float(step["x_start_position"].values[index])
            gt_starts = gt_step["position"].values
            gt_ends = gt_starts + gt_step["length"].values
            bounds = (np.array([low]), np.array([low + mask_length]))
            covered = float(gt_mass_on_intervals(gt_starts, gt_ends,
                                                 np.ones(len(gt_starts)), *bounds)[0])
            if covered < 0.99 * mask_length:
                continue
            held_series.append(float(step["density"].values[index] * lengths[index]))
            window_series.append(float(gt_mass_on_intervals(
                gt_starts, gt_ends, gt_step["density"].values, *bounds)[0]))
        if not held_series:
            continue
        held = np.array(held_series)
        window = np.array(window_series)
        rows.append({
            "grid": grid,
            "held_mean": held.mean(),
            "held_max": held.max(),
            "gt_window_mean": window.mean(),
            "gt_window_max": window.max(),
            "signed_mean": (held - window).mean(),
            "L1_bubble": (np.abs(held - window) * dt).sum(),
        })
    return pd.DataFrame(rows)


# ======================================================================================
# Publication figures
#
# Static matplotlib output for LaTeX -- no hover layer, no dark mode; those apply to
# on-screen charts, not print.
#
# Colour is assigned by the job it does:
#   * model identity (newell/idm/arz) -> the categorical palette's first three slots,
#     which are the documented all-pairs-validated set (worst CVD dE 9.2, normal-vision
#     24.0 on a light surface). Three is exactly the documented cap for forms where any
#     pair may be compared; a fourth model would have to fold to "Other" or facet.
#   * wave type (shock/contact) -> the first two slots, same reason.
#   * density magnitude -> one hue, light->dark. Never a rainbow.
#   * the exact/reference solution is NEVER a series colour -- it is neutral ink, because
#     it is not one of the things being compared.
#
# Every series also carries a linestyle and a marker, so the figures survive greyscale
# printing and photocopying, where hue alone would collapse. Aqua sits below 3:1 on a
# light surface, so the relief rule applies: curves are direct-labelled, not colour-only.
# ======================================================================================

SURFACE_LIGHT = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"

SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"
SERIES_AQUA = "#1baf7a"

# (colour, linestyle, marker) -- redundant encoding for greyscale robustness.
MODEL_STYLE = {
    "newell_case": (SERIES_BLUE, "-", "o"),
    "idm_case": (SERIES_ORANGE, "--", "s"),
    "arz_case": (SERIES_AQUA, "-.", "^"),
}
MODEL_LABEL = {"newell_case": "Newell", "idm_case": "IDM", "arz_case": "ARZ"}

# Which Riemann structure each prescribed boundary case drives. Read off the measured
# orders: these four converge at ~1 (shock) and those four at ~1/2 (contact), which is
# the textbook split for a first-order monotone scheme and is the point of the figure.
B_CASE_WAVE = {
    "micro_case_b_1": "shock", "micro_case_b_3": "shock",
    "micro_case_b_7": "shock", "micro_case_b_8": "shock",
    "micro_case_b_2": "contact", "micro_case_b_4": "contact",
    "micro_case_b_5": "contact", "micro_case_b_6": "contact",
}
WAVE_STYLE = {"shock": (SERIES_BLUE, "-", "o"), "contact": (SERIES_ORANGE, "--", "s")}

# Sequential blue ramp, steps 100->700 of the same hue as series slot 1.
SEQUENTIAL_BLUE = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                   "#256abf", "#184f95", "#0d366b")

# LaTeX column widths in inches.
WIDTH_SINGLE = 3.4
WIDTH_DOUBLE = 7.0


def _pyplot():
    """Import matplotlib lazily with a headless backend.

    Kept out of module scope so importing analysis_helpers for table work (including
    inside joblib workers) does not pay for matplotlib or touch a display.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE_LIGHT,
        "axes.facecolor": SURFACE_LIGHT,
        "savefig.facecolor": SURFACE_LIGHT,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.labelcolor": INK_PRIMARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        # Recessive frame: no top/right spines, hairline axes, faint grid.
        "axes.edgecolor": INK_MUTED,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#e2e1dc",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    return plt


def dx_of_grid(grid):
    return float(grid.split("_dt_")[0])


# --------------------------------------------------------------------------------------
# Figure data: computed once, cached, so figure iteration does not re-stream 60 GB.
# --------------------------------------------------------------------------------------

def build_figure_data(results_folder, grids=None, cache_path="figure_data.csv",
                      n_jobs=3, macro_cases=("case_m_1", "case_m_2", "case_m_3", "case_m_4")):
    """Compute every convergence table once and write a tidy CSV.

    Streaming the macro logs is the expensive part of every figure, so it happens here
    and nowhere else. Figure functions read the cache.
    """
    grids = grids or GRID_SWEEP
    tables = sweep_tables(results_folder, macro_cases=macro_cases, grids=grids,
                          n_jobs=n_jobs, verbose=False)
    frames = []
    for (macro_case, micro_case), table in tables.items():
        if table.empty:
            continue
        frame = table.copy()
        frame["macro_case"] = macro_case
        frame["micro_case"] = micro_case
        frame["dx"] = frame["grid"].map(dx_of_grid)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(cache_path, index=False)
    return data


def load_figure_data(cache_path="figure_data.csv"):
    return pd.read_csv(cache_path)


def _reference_slope(ax, slope, x_range, y_at_left, label):
    """Draw a reference power law as neutral ink -- it is an annotation, not a series."""
    x0, x1 = x_range
    ax.plot([x0, x1], [y_at_left, y_at_left * (x1 / x0) ** slope],
            color=INK_MUTED, linewidth=0.9, linestyle=(0, (4, 2)), zorder=1)
    ax.annotate(label, xy=(x1, y_at_left * (x1 / x0) ** slope),
                xytext=(2, -1), textcoords="offset points",
                color=INK_SECONDARY, fontsize=7, va="center")


MACHINE_ZERO = 1e-6


def _label_right(ax, x, y, text, used, colour=INK_SECONDARY, tolerance=0.08,
                 on_collision="skip"):
    """Direct-label a curve at its right (finest-grid) end, resolving collisions.

    Curves that lie on top of each other would otherwise stamp their labels in the same
    place and render as an unreadable overstrike. `used` accumulates the y positions
    already claimed on this axes.

    on_collision="skip"   -- drop the label. Correct when the curves are genuinely
        indistinguishable (m_3's three bubble floors all sit on ~600), where a second
        label would imply a distinction the data does not contain.
    on_collision="offset" -- keep the label, nudged vertically. Correct when the curves
        are distinct cases that merely run close (b_2 and b_6 on m_4 differ by ~5%);
        suppressing one there loses identity, which is what it did on the first render.
    """
    collisions = sum(1 for claimed in used
                     if claimed > 0 and abs(math.log10(max(y, 1e-300) / claimed)) < tolerance)
    if collisions and on_collision == "skip":
        return False
    used.append(y)
    ax.annotate(text, xy=(x, y), xytext=(4, -8.5 * collisions), textcoords="offset points",
                fontsize=6.5, color=colour, va="center", zorder=5,
                annotation_clip=False)
    return True


def figure_boundary_orders(data, out_path=None, floor=MACHINE_ZERO):
    """F1 -- soundness of the coupling boundary itself.

    One panel per Riemann structure, because the result IS the split: shocks land on slope
    1, contacts on slope 1/2, and the two families sit four decades apart in absolute
    error. Sharing one axes (the first attempt) let the near-exact cases stretch the range
    to 10^-9 and flattened every slope into an unreadable band.

    Points at or below `floor` are machine zero, not measurements: b_2/case_m_2 is exact at
    every grid and b_1/case_m_2 is exact at dx=128 by cell alignment. They cannot go on a
    log axis; the function returns the list of what it dropped so the caption can say so
    rather than silently losing them.
    """
    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(WIDTH_DOUBLE, 2.8))
    prescribed = data[data.micro_case.str.startswith("micro_case_b_")].copy()
    prescribed["wave"] = prescribed["micro_case"].map(B_CASE_WAVE)
    dropped = []

    # Colour encodes the wave type (one per panel); MARKER encodes the individual case.
    # Direct labels were tried first and do not work here: five same-coloured curves
    # converge at the fine end, so the labels either overstrike each other or spill off
    # the axis. A marker plus a legend carries identity without fighting for space.
    case_markers = ("o", "s", "^", "D", "v", "P")

    for ax, wave, slope, slope_label in ((axes[0], "shock", 1.0, r"$\Delta x^{1}$"),
                                         (axes[1], "contact", 0.5, r"$\Delta x^{1/2}$")):
        colour, linestyle, _ = WAVE_STYLE[wave]
        subset = prescribed[prescribed["wave"] == wave]
        drawn = []
        for index, ((macro_case, micro_case), group) in enumerate(
                subset.groupby(["macro_case", "micro_case"])):
            group = group.sort_values("dx", ascending=False)     # coarse -> fine
            usable = group[group["macro"] > floor]
            if len(usable) < 2:
                dropped.append(f"{micro_case} on {macro_case}")
                continue
            if len(usable) < len(group):
                dropped.append(f"{micro_case} on {macro_case} (coarsest point only)")
            ax.plot(usable["dx"], usable["macro"], color=colour, linestyle=linestyle,
                    marker=case_markers[index % len(case_markers)],
                    markerfacecolor=SURFACE_LIGHT, markeredgewidth=1.0,
                    label=f"{micro_case.replace('micro_case_', '')} / "
                          f"{macro_case.replace('case_', '')}", zorder=3)
            drawn.append(usable)
        if drawn:
            everything = pd.concat(drawn)
            dx_hi, dx_lo = everything["dx"].max(), everything["dx"].min()
            _reference_slope(ax, slope, (dx_hi, dx_lo),
                             everything["macro"].max() * 1.8, slope_label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(r"cell length $\Delta x$ (m)")
        ax.set_title("Shocks: expected order 1" if wave == "shock"
                     else "Contacts: expected order 1/2")
        ax.grid(True, which="major", zorder=0)
        ax.margins(x=0.12)
        ax.legend(loc="lower left", fontsize=6, ncol=1, handlelength=2.4,
                  borderpad=0.2, labelspacing=0.25)
    axes[0].set_ylabel(r"$L_1$ mass error (veh$\cdot$s)")
    figure.suptitle("Prescribed boundary cases converge at the wave-type rate", y=1.03)
    figure.dropped_cases = dropped
    if out_path:
        figure.savefig(out_path)
    return figure


def figure_two_term(data, macro_cases=("case_m_3", "case_m_4"), out_path=None):
    """F2 -- the thesis figure: a converging macro term against a flat micro floor.

    Solid = macro (mask excluded), dashed = bubble. The crossing of a descending line and
    a horizontal one is the whole argument, so both terms must be on the same axes;
    plotting their sum alone would hide it.
    """
    plt = _pyplot()
    figure, axes = plt.subplots(1, len(macro_cases), figsize=(WIDTH_DOUBLE, 2.9),
                                sharey=True)
    axes = np.atleast_1d(axes)
    for ax, macro_case in zip(axes, macro_cases):
        subset = data[(data.macro_case == macro_case)
                      & (data.micro_case.isin(MODEL_STYLE))]
        used = []
        for micro_case, group in subset.groupby("micro_case"):
            group = group.sort_values("dx")
            colour, _, marker = MODEL_STYLE[micro_case]
            positive = group[group["macro"] > MACHINE_ZERO]
            order = list(MODEL_STYLE).index(micro_case)
            ax.plot(positive["dx"], positive["macro"], color=colour, linestyle="-",
                    marker=marker, markerfacecolor=SURFACE_LIGHT, markeredgewidth=1.0,
                    linewidth=2.2 - 0.5 * order, label=f"{MODEL_LABEL[micro_case]} macro",
                    zorder=3 + order)
            ax.plot(group["dx"], group["bubble"], color=colour, linestyle="--",
                    marker=marker, markerfacecolor=colour, markeredgewidth=0,
                    alpha=0.85, label=f"{MODEL_LABEL[micro_case]} bubble", zorder=2)
            # Label the bubble curve at the finest grid. On m_3 all three floors sit on
            # ~600 and only one label is drawn; on m_4 Newell separates and both appear.
            finest = group.sort_values("dx").iloc[0]
            _label_right(ax, finest["dx"], finest["bubble"], MODEL_LABEL[micro_case],
                         used)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(r"cell length $\Delta x$ (m)")
        ax.set_title(f"scenario $m_{{{macro_case.rsplit('_', 1)[-1]}}}$")
        ax.grid(True, which="major", zorder=0)
    axes[0].set_ylabel(r"$L_1$ mass error (veh$\cdot$s)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3,
                  bbox_to_anchor=(0.5, -0.16))
    figure.suptitle("The macro term converges; the micro term is a floor", y=1.02)
    if out_path:
        figure.savefig(out_path)
    return figure


def figure_observed_order(data, macro_case="case_m_4", out_path=None):
    """F3 -- coupled models fall below the prescribed-boundary ceiling.

    Observed order between successive grids. The prescribed cases hold 1/2; the genuine
    car-following models decay below it. They differ in exactly one respect -- prescribed
    boundary data versus data inferred through the FD inversion -- so this is the
    controlled comparison.
    """
    plt = _pyplot()
    figure, ax = plt.subplots(figsize=(WIDTH_SINGLE, 2.7))
    subset = data[data.macro_case == macro_case]
    for micro_case, group in subset.groupby("micro_case"):
        group = group.sort_values("dx", ascending=False)
        orders = group.dropna(subset=["p_macro"])
        if orders.empty:
            continue
        if micro_case in MODEL_STYLE:
            colour, linestyle, marker = MODEL_STYLE[micro_case]
            label = MODEL_LABEL[micro_case]
            width, zorder = 1.4, 3
        else:
            colour, linestyle, marker = INK_MUTED, ":", "x"
            label = micro_case.replace("micro_case_", "prescribed ")
            width, zorder = 1.0, 2
        ax.plot(orders["dx"], orders["p_macro"], color=colour, linestyle=linestyle,
                marker=marker, linewidth=width, markerfacecolor=SURFACE_LIGHT,
                markeredgewidth=1.0, label=label, zorder=zorder)
    ax.axhline(0.5, color=INK_MUTED, linewidth=0.9, linestyle=(0, (4, 2)), zorder=1)
    ax.annotate("theoretical contact rate 1/2", xy=(0.02, 0.5), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", fontsize=6.5,
                color=INK_SECONDARY)
    ax.set_xscale("log", base=2)
    ax.invert_xaxis()
    ax.set_xlabel(r"cell length $\Delta x$ (m)")
    ax.set_ylabel("observed order $p$")
    ax.set_title("Coupled models decay below the prescribed ceiling")
    ax.grid(True, which="major", axis="y", zorder=0)
    ax.legend(loc="lower left", ncol=1)
    if out_path:
        figure.savefig(out_path)
    return figure


# --------------------------------------------------------------------------------------
# Visual figures -- read the run logs directly rather than the convergence cache.
# --------------------------------------------------------------------------------------

def bubble_timeseries(macro_case, micro_case, grid, results_folder,
                      gt_case=None, config_folder="verification_config", tolerance=1e-6):
    """Per-step vehicles held in the bubble against the exact mass over the same window.

    Mask geometry comes from the macro log rather than the mask log: the latter's
    x_start_position lags the active mesh by anchor_speed*dt, which would bias the window
    the ground truth is integrated over.
    """
    gt_case = gt_case or default_gt_case(micro_case)
    run_dir = os.path.join(results_folder, f"{macro_case}_dx_{grid}")
    mask_log = pd.read_csv(os.path.join(run_dir, f"{micro_case}_mask.csv"))
    mask_length = float(mask_log["length"].iloc[0])
    counts = (mask_log.set_index("time")["vehicle_count"].to_dict()
              if "vehicle_count" in mask_log.columns else {})
    gt = pd.read_csv(os.path.join(config_folder, f"{macro_case}_dx_{grid}",
                                  f"{gt_case}.csv"))
    gt_groups = {float(t): g for t, g in gt.groupby("time")}

    rows = []
    for t, step in _iter_macro_time_groups(
            os.path.join(run_dir, f"{micro_case}_macro.csv")):
        gt_step = gt_groups.get(t)
        if gt_step is None:
            continue
        lengths = step["length"].values
        is_mask = np.abs(lengths - mask_length) < tolerance
        if not is_mask.any():
            continue
        index = int(np.argmax(is_mask))
        low = float(step["x_start_position"].values[index])
        starts = gt_step["position"].values
        ends = starts + gt_step["length"].values
        bounds = (np.array([low]), np.array([low + mask_length]))
        covered = float(gt_mass_on_intervals(starts, ends, np.ones(len(starts)),
                                             *bounds)[0])
        if covered < 0.99 * mask_length:
            continue
        rows.append({
            "time": t,
            "mask_rear_s": low,
            "held": float(counts.get(t, step["density"].values[index] * lengths[index])),
            "gt_window": float(gt_mass_on_intervals(starts, ends,
                                                    gt_step["density"].values,
                                                    *bounds)[0]),
        })
    return pd.DataFrame(rows)


def figure_bubble_occupancy(results_folder, macro_case="case_m_4", grid="16.0_dt_0.125",
                            models=("newell_case", "idm_case", "arz_case"),
                            out_path=None, skip_seconds=5.0):
    """F4 -- the floor, made mechanical.

    Vehicles actually held in the bubble against the exact LWR demand over the same
    window. This is the same quantity the bubble L1 integrates, shown before integration,
    so a reader can see WHY one model's floor is five times another's.
    """
    plt = _pyplot()
    figure, ax = plt.subplots(figsize=(WIDTH_SINGLE, 2.6))
    demand = None
    used = []
    for micro_case in models:
        series = bubble_timeseries(macro_case, micro_case, grid, results_folder)
        if series.empty:
            continue
        # The opening seconds are the initial seeding transient, before each model relaxes
        # to its own equilibrium occupancy; it compresses the informative range.
        series = series[series["time"] >= skip_seconds]
        colour, linestyle, _ = MODEL_STYLE[micro_case]
        ax.plot(series["time"], series["held"], color=colour, linestyle=linestyle,
                label=MODEL_LABEL[micro_case], zorder=3)
        # IDM and ARZ both settle on exactly 1 vehicle, so only one of them gets a label.
        tail = series.iloc[-1]
        _label_right(ax, tail["time"], tail["held"], MODEL_LABEL[micro_case], used,
                     tolerance=0.02)
        demand = series if demand is None else demand
    if demand is not None:
        # Neutral ink: the exact solution is the reference, not one of the compared things.
        ax.plot(demand["time"], demand["gt_window"], color=INK_PRIMARY,
                linestyle=(0, (4, 2)), linewidth=1.1, label="exact LWR", zorder=5)
        _label_right(ax, demand["time"].iloc[-1], demand["gt_window"].iloc[-1],
                     "exact LWR", used, colour=INK_PRIMARY, tolerance=0.02)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("vehicles in the micro window")
    ax.set_title("What each model holds versus what LWR demands")
    ax.grid(True, axis="y", zorder=0)
    ax.legend(loc="upper right", ncol=2)
    if out_path:
        figure.savefig(out_path)
    return figure


def density_grid(macro_path, mask_length, n_time=400, n_space=600, road_length=40000.0,
                 tolerance=1e-6):
    """Bin a macro log onto a fixed (time, space) raster for a heatmap.

    The logs are 300 MB at dx=16 and 7 GB at dx=4, and no display resolves 2500 cells, so
    binning happens while streaming rather than after loading. Returns the raster, the
    time edges, and the mask window per binned row.
    """
    accumulated = np.zeros((n_time, n_space))
    weights = np.zeros((n_time, n_space))
    mask_rear = np.full(n_time, np.nan)
    times = []
    for t, step in _iter_macro_time_groups(macro_path):
        times.append(t)
    t_min, t_max = min(times), max(times)
    edges = np.linspace(t_min, t_max, n_time + 1)

    for t, step in _iter_macro_time_groups(macro_path):
        row = min(int((t - t_min) / (t_max - t_min + 1e-12) * n_time), n_time - 1)
        positions = step["x_start_position"].values
        lengths = step["length"].values
        densities = step["density"].values
        columns = np.clip((positions / road_length * n_space).astype(int), 0, n_space - 1)
        np.add.at(accumulated, (row, columns), densities * lengths)
        np.add.at(weights, (row, columns), lengths)
        is_mask = np.abs(lengths - mask_length) < tolerance
        if is_mask.any():
            mask_rear[row] = positions[int(np.argmax(is_mask))]
    with np.errstate(invalid="ignore", divide="ignore"):
        raster = np.where(weights > 0, accumulated / np.maximum(weights, 1e-12), np.nan)
    return raster, edges, mask_rear


def figure_space_time(results_folder, macro_case="case_m_4", micro_case="newell_case",
                      grid="16.0_dt_0.125", out_path=None, road_length=40000.0):
    """F5 -- where the bubble sits relative to the wave structure.

    One hue, light to dark, for a magnitude field. The mask window is drawn over it as a
    neutral outline so the reader can see the micro domain travelling through the jam.
    """
    plt = _pyplot()
    from matplotlib.colors import LinearSegmentedColormap
    colourmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)

    run_dir = os.path.join(results_folder, f"{macro_case}_dx_{grid}")
    mask_log = pd.read_csv(os.path.join(run_dir, f"{micro_case}_mask.csv"))
    mask_length = float(mask_log["length"].iloc[0])
    raster, edges, mask_rear = density_grid(
        os.path.join(run_dir, f"{micro_case}_macro.csv"), mask_length,
        road_length=road_length)

    figure, ax = plt.subplots(figsize=(WIDTH_SINGLE, 2.8))
    image = ax.imshow(raster, aspect="auto", origin="lower", cmap=colourmap,
                      extent=(0.0, road_length / 1000.0, edges[0], edges[-1]))
    rows = np.linspace(edges[0], edges[-1], len(mask_rear))
    finite = np.isfinite(mask_rear)
    ax.plot(mask_rear[finite] / 1000.0, rows[finite], color=INK_PRIMARY,
            linewidth=0.8, linestyle="-", zorder=3)
    ax.plot((mask_rear[finite] + mask_length) / 1000.0, rows[finite], color=INK_PRIMARY,
            linewidth=0.8, linestyle="-", zorder=3)
    ax.annotate("micro window", xy=(mask_rear[finite][len(mask_rear[finite]) // 2] / 1000.0,
                                    rows[finite][len(rows[finite]) // 2]),
                xytext=(6, 0), textcoords="offset points", fontsize=6.5,
                color=INK_PRIMARY, va="center", zorder=4)
    bar = figure.colorbar(image, ax=ax, pad=0.02)
    bar.set_label(r"density $\rho$ (veh/m)", fontsize=7)
    bar.ax.tick_params(labelsize=6)
    bar.outline.set_visible(False)
    ax.set_xlabel("position (km)")
    ax.set_ylabel("time (s)")
    ax.set_title(f"{MODEL_LABEL.get(micro_case, micro_case)}: micro window in the wave field")
    if out_path:
        figure.savefig(out_path)
    return figure


def figure_trajectories(figure_run_folder="figure_runs", macro_case="case_m_4",
                        grid="16.0_dt_0.125",
                        models=("newell_case", "idm_case", "arz_case"),
                        out_path=None, frame="relative", mask_length=None,
                        results_folder=None):
    """F6 -- vehicle trajectories inside the bubble.

    Small multiples rather than one axes: three models' trajectories overlaid would be
    unreadable, and the comparison is between panels, not within one.

    `frame="relative"` plots position within the moving window. Lines come out FLAT, which
    surprises people expecting trajectories: the window travels at the anchor's speed and
    in free flow so does every vehicle, so relative positions do not change. Read it as a
    spacing diagram, not a trajectory diagram. `frame="absolute"` gives the conventional
    rising trajectories if that is what a reader expects.

    The measured content is the shaded band: no vehicle is EVER ahead of the anchor
    (0 of ~16k logged rows), because the window is centred on the ego and nobody can
    overtake on one lane, so the leading margin_s of the mask is structurally empty.

    Needs Logger(vehicle_path=...), which the sweep does not enable; see figure_runs.py.
    """
    plt = _pyplot()
    figure, axes = plt.subplots(1, len(models), figsize=(WIDTH_DOUBLE, 2.5),
                                sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    run_dir = os.path.join(figure_run_folder, f"{macro_case}_dx_{grid}")
    if mask_length is None:
        # 2*margin_s, read off whichever mask log is to hand rather than hardcoded.
        for folder in (run_dir, os.path.join(results_folder or "", f"{macro_case}_dx_{grid}")):
            candidate = os.path.join(folder, f"{models[0]}_mask.csv")
            if os.path.exists(candidate):
                mask_length = float(pd.read_csv(candidate, usecols=["length"],
                                                nrows=1)["length"].iloc[0])
                break
    for ax, micro_case in zip(axes, models):
        path = os.path.join(run_dir, f"{micro_case}_vehicles.csv")
        if not os.path.exists(path):
            continue
        vehicles = pd.read_csv(path)
        colour, _, _ = MODEL_STYLE[micro_case]
        for _, track in vehicles.groupby("vehicle_id"):
            track = track.sort_values("time")
            y = (track["s"] - track["mask_rear_s"]) if frame == "relative" else track["s"] / 1000.0
            ax.plot(track["time"], y, color=colour, linewidth=0.7, alpha=0.85, zorder=3)
        held = vehicles.groupby("time").size().mean()
        ax.set_title(f"{MODEL_LABEL[micro_case]}  ({held:.2f} veh held)")
        ax.set_xlabel("time (s)")
        ax.grid(True, axis="y", zorder=0)
        if frame == "relative" and mask_length:
            ax.axhspan(mask_length / 2.0, mask_length, color=INK_MUTED, alpha=0.11,
                       zorder=0, linewidth=0)
            if micro_case == models[0]:
                ax.annotate("never occupied", xy=(0.5, 0.80), xycoords="axes fraction",
                            fontsize=6.5, color=INK_SECONDARY, ha="center")
            # Pin to the full window. Autoscaling stops at the frontmost vehicle, which
            # hides the fact that the leading half of the bubble is empty in every model
            # -- the most informative thing in the panel. The dashed line is the anchor,
            # at margin_s; no vehicle is ever ahead of it.
            ax.set_ylim(0.0, mask_length)
            ax.axhline(mask_length / 2.0, color=INK_MUTED, linewidth=0.7,
                       linestyle=(0, (2, 2)), zorder=1)
    axes[0].set_ylabel("position within the micro window (m)"
                       if frame == "relative" else "position (km)")
    figure.suptitle("The leading half of the micro domain is never seeded", y=1.04)
    if out_path:
        figure.savefig(out_path)
    return figure


def save_publication_figures(results_folder, out_dir="figures",
                             cache_path="figure_data.csv", grids=None,
                             figure_run_folder="figure_runs",
                             visual_grid="16.0_dt_0.125", formats=("pdf", "png"),
                             table_dir="tables", tables=True):
    """Build every figure and the LaTeX convergence tables from one cached dataset.

    Figures and tables answer different questions and the paper needs both: the tables
    carry the convergence rates (where overplotted curves and machine-zero values defeat a
    log axis), the figures carry mechanism and space-time structure. Both read the same
    `cache_path`, so they cannot drift apart.
    """
    plt = _pyplot()
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(cache_path):
        data = load_figure_data(cache_path)
    else:
        data = build_figure_data(results_folder, grids=grids, cache_path=cache_path)

    builders = {
        "f1_boundary_orders": lambda: figure_boundary_orders(data),
        "f2_two_term": lambda: figure_two_term(data),
        "f3_observed_order": lambda: figure_observed_order(data),
        "f4_bubble_occupancy": lambda: figure_bubble_occupancy(
            results_folder, grid=visual_grid),
        "f5_space_time": lambda: figure_space_time(results_folder, grid=visual_grid),
        "f6_trajectories": lambda: figure_trajectories(
            figure_run_folder, grid=visual_grid),
    }
    written = []
    for name, build in builders.items():
        try:
            figure = build()
        except Exception as error:                       # keep going; report at the end
            print(f"  SKIP {name}: {type(error).__name__}: {error}")
            continue
        for extension in formats:
            path = os.path.join(out_dir, f"{name}.{extension}")
            figure.savefig(path)
            written.append(path)
        plt.close(figure)
    for path in written:
        print(f"  wrote {path}")

    if tables:
        _, standalone = write_latex_tables(cache_path=cache_path, out_dir=table_dir)
        written.append(standalone)
        pdf = render_latex_pdf(standalone)
        if pdf:
            written.append(pdf)
    return written


# ======================================================================================
# LaTeX convergence tables
#
# A convergence study is a claim about numbers, and the field presents it as a table: the
# reader checks that the order column behaves, and can redo the arithmetic. Tables also
# sidestep three problems the plotted versions have -- the three models' macro errors on
# case_m_4 differ by ~1% and overplot to the point of hiding one series entirely; ten
# same-coloured curves collide when direct-labelled; and b_2 on case_m_2 is machine zero
# (~1.5e-10), which cannot be placed on a log axis at all but prints fine in a cell.
#
# booktabs only -- siunitx is not installed on this machine, so alignment is done by
# formatting the numbers rather than by S columns.
# ======================================================================================

LATEX_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in,landscape]{geometry}
\usepackage{booktabs}
\begin{document}
\pagestyle{empty}
"""

MODEL_ORDER = ("newell_case", "idm_case", "arz_case")


def _fmt_error(value):
    """Errors span 1e-10 to 1e4 across this study, so no single format serves."""
    if value != value:
        return "---"
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e5:
        mantissa, exponent = f"{value:.2e}".split("e")
        return f"${mantissa}\\times 10^{{{int(exponent)}}}$"
    if abs(value) < 1.0:
        return f"{value:.4f}"
    if abs(value) < 100.0:
        return f"{value:.2f}"
    return f"{value:.0f}"


def _fmt_order(value):
    return "---" if value != value else f"{value:.3f}"


def latex_case_block(data, macro_case, micro_cases, columns=("macro", "p_macro"),
                     headers=("$L_1$", "$p$"), label=None, caption=None):
    """One booktabs table: rows are grids, column groups are cases.

    `columns` selects which per-case quantities to show, so the same function emits the
    prescribed-boundary table (error and order) and the model table (macro error, order,
    and the bubble floor beside it).
    """
    present = [c for c in micro_cases
               if not data[(data.macro_case == macro_case) & (data.micro_case == c)].empty]
    if not present:
        return ""
    width = len(columns)
    spec = "r" + "".join(" " + "r" * width for _ in present)
    lines = [r"\begin{table}[htbp]", r"\centering"]
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{spec}}}")
    lines.append(r"\toprule")

    group_header = [r"$\Delta x$ (m)"]
    rules = []
    start = 2
    for case in present:
        pretty = MODEL_LABEL.get(case, case.replace("micro_case_", "").replace("_", r"\_"))
        group_header.append(rf"\multicolumn{{{width}}}{{c}}{{{pretty}}}")
        rules.append(rf"\cmidrule(lr){{{start}-{start + width - 1}}}")
        start += width
    lines.append(" & ".join(group_header) + r" \\")
    lines.append("".join(rules))
    lines.append(" & ".join([""] + list(headers) * len(present)) + r" \\")
    lines.append(r"\midrule")

    frame = data[data.macro_case == macro_case]
    for dx in sorted(frame["dx"].unique(), reverse=True):
        row = [f"{dx:g}"]
        for case in present:
            match = frame[(frame.micro_case == case) & (frame.dx == dx)]
            for column in columns:
                if match.empty:
                    row.append("---")
                    continue
                value = float(match[column].iloc[0])
                row.append(_fmt_order(value) if column.startswith("p_")
                           else _fmt_error(value))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_convergence_tables(data, macro_cases=("case_m_1", "case_m_2",
                                                "case_m_3", "case_m_4")):
    """Both table families: prescribed boundary cases, then the car-following models."""
    blocks = [r"\section*{Prescribed boundary cases}"]
    for macro_case in macro_cases:
        prescribed = sorted(
            data[(data.macro_case == macro_case)
                 & (data.micro_case.str.startswith("micro_case_b_"))]["micro_case"].unique())
        blocks.append(latex_case_block(
            data, macro_case, prescribed,
            columns=("macro", "p_macro"), headers=("$L_1$", "$p$"),
            label=f"tab:prescribed-{macro_case}",
            caption=(f"Scenario {macro_case.replace('case_', '').replace('_', '')}: "
                     r"prescribed boundary cases. $L_1$ mass error (veh$\cdot$s) and "
                     r"observed order $p$ between successive grids.")))
    blocks.append(r"\clearpage")
    blocks.append(r"\section*{Car-following models}")
    for macro_case in macro_cases:
        blocks.append(latex_case_block(
            data, macro_case, MODEL_ORDER,
            columns=("macro", "p_macro", "bubble"),
            headers=("$L_1$ macro", "$p$", "$L_1$ bubble"),
            label=f"tab:models-{macro_case}",
            caption=(f"Scenario {macro_case.replace('case_', '').replace('_', '')}: "
                     r"car-following models. The macro term converges; the bubble term is "
                     r"a grid-independent floor set by the model.")))
    return "\n\n".join(b for b in blocks if b)


def write_latex_tables(cache_path="figure_data.csv", out_dir="tables",
                       basename="convergence_tables"):
    """Write both a bare fragment (for \\input) and a standalone document (for rendering)."""
    data = load_figure_data(cache_path)
    body = latex_convergence_tables(data)
    os.makedirs(out_dir, exist_ok=True)
    fragment = os.path.join(out_dir, f"{basename}.tex")
    with open(fragment, "w") as handle:
        handle.write(body + "\n")
    standalone = os.path.join(out_dir, f"{basename}_standalone.tex")
    with open(standalone, "w") as handle:
        handle.write(LATEX_PREAMBLE + body + "\n" + r"\end{document}" + "\n")
    print(f"  wrote {fragment}")
    print(f"  wrote {standalone}")
    return fragment, standalone


def render_latex_pdf(standalone_path, engine=None):
    """Typeset the standalone table document, if a LaTeX engine is on this machine.

    Returns the PDF path, or None when no engine is installed -- the analysis runs inside a
    container that has no TeX at all, while the host has pdflatex, so this has to degrade
    rather than fail. The .tex output is the real deliverable; the PDF is a convenience.
    """
    engine = engine or next((e for e in ("pdflatex", "lualatex", "xelatex", "tectonic")
                             if shutil.which(e)), None)
    if engine is None:
        print("  (no LaTeX engine on this machine; skipped PDF, .tex still written)")
        return None
    directory = os.path.dirname(os.path.abspath(standalone_path)) or "."
    name = os.path.basename(standalone_path)
    result = subprocess.run(
        [engine, "-interaction=nonstopmode", "-halt-on-error", name],
        cwd=directory, capture_output=True, text=True)
    pdf = os.path.join(directory, os.path.splitext(name)[0] + ".pdf")
    if result.returncode != 0 or not os.path.exists(pdf):
        # Surface the first real TeX error rather than the whole log.
        complaint = next((ln for ln in result.stdout.splitlines() if ln.startswith("! ")),
                         f"{engine} exited {result.returncode}")
        print(f"  LaTeX failed: {complaint}")
        return None
    print(f"  wrote {pdf}")
    return pdf
