import glob
import math
import os

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
