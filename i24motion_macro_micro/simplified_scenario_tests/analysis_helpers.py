import pandas as pd

# We assume all entries have the same time step
def generate_ground_truth_alignment_with_simulation_result(gt: pd.DataFrame, sim_macro: pd.DataFrame, sim_mask: pd.DataFrame, eps: float = 1e-8):
    mask_position = float(sim_mask.iloc[0]["x_start_position"])
    mask_length = float(sim_mask.iloc[0]["length"])
    sim_macro = sim_macro[(sim_macro["x_start_position"] - mask_position).abs() > eps]

    t = float(sim_macro.iloc[0]["time"])
    dt = float(sim_macro.iloc[0]["dt"])
    assert((gt["time"].nunique() == 1) and (float(gt["time"].iloc[0]) == t))
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
            overlap_length = overlap_end - overlap_end
            if (overlap_length > eps):
                mass_new += (overlap_length * gt_density)
        density_new = mass_new / length_new
        time_array.append(new_time)
        dt_array.append(new_dt)
        x_start_position_array.append(x_start_position_new)
        length_array.append(length_new)
        density_array.append(density_new)

    return pd.DataFrame({
        "time": time_array,
        "dt": dt_array,
        "x_start_position": x_start_position_array,
        "length": length_array,
        "density": density_array
    }), sim_macro

def generate_ground_truth_and_sim_result(gt: pd.DataFrame, sim_macro: pd.DataFrame, sim_mask: pd.DataFrame, eps: float = 1e-8):
    gt_result = pd.DataFrame()
    macro_result = pd.DataFrame()
    # By default, remove the first and last x position from sim_macro and sim_mask - those are always being overwritten anyway
    x_positions = sorted(sim_macro["x_start_position"].unique().tolist())
    sim_macro = sim_macro[(sim_macro["x_start_position"] > x_positions[0]) | (sim_macro["x_start_position"] < x_positions[-1])]
    times = sim_macro["time"].unique().tolist()
    for time in times:
        sim_macro_time = sim_macro[(sim_macro["time"] - time).abs() <= eps]
        sim_mask_time = sim_mask[(sim_mask["time"] - time).abs() <= eps]
        gt_time = gt[(gt["time"] - time).abs() <= eps]
        gt_step, sim_macro_step = generate_ground_truth_alignment_with_simulation_result(gt_time, sim_macro_time, sim_mask_time, eps)
        gt_result = pd.concat([gt_result, gt_step], axis=0)
        macro_result = pd.concat([macro_result, sim_macro_step], axis=0)
        print(time)
    gt_result = gt_result.sort_values(["time", "x_start_position"], kind="mergesort", ascending=True)
    macro_result = macro_result.sort_values(["time", "x_start_position"], kind="mergesort", ascending=True)
    return gt_result, macro_result
