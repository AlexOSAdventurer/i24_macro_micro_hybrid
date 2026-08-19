import sys
sys.path.append("..")
import argparse
from functools import partial
import json
import multiprocessing
import time
import traceback
from simulation import Simulation, RolloutRenderer, GroundTruthStore, TriangularFD
from bridge_coupler import SimplifiedSimBridge, NewellModel, IDMModel, IIDMModel, MicroscopicARZVehicleModel
from prescribed_bridge_coupler import PrescribedSimBridge, boundary_case_1, boundary_case_2, boundary_case_3, boundary_case_4, boundary_case_5, boundary_case_6, boundary_case_7, boundary_case_8
from simulation_dataset_verification import micro_length
from logger import Logger
import os
import glob

def spawn_density_function_linear_interpolation(start_density, end_density, s, max_s):
    end_frac = s / max_s
    start_frac = 1.0 - end_frac
    return (start_density * start_frac) + (end_density * end_frac)

# We assume the mask is placed exactly on top of the discontinuity
# as in, the middle of the boundary is exactly at the discontinuity
def spawn_density_function_discontinuity(start_density, end_density, s, max_s):
    end_frac = s / max_s
    if (end_frac >= 0.5):
        return end_density
    return start_density

def lwr_triangular_fd():
    fd = TriangularFD(
        v_f=49.7,
        w=5.7,
        rho_j=0.13
    )
    return fd

def micro_case_b_1(sim: Simulation, config_file_name: str):
    with open(config_file_name, "r") as f:
        config = json.load(f)
    road_length = config["road_data"]["1"]["road_length"]
    micro_middle_position = (0.2 * road_length)# + (micro_length / 2.0)
    micro_max_position = road_length - (micro_length / 2.0)
    return PrescribedSimBridge(sim=sim, 
                              road_id="1",
                              initial_middle_s=micro_middle_position,
                              max_middle_s=micro_max_position,
                              margin_s=(micro_length / 2.0),
                              fd=lwr_triangular_fd(),
                              config_file_name=config_file_name,
                              boundary_function=boundary_case_1,
                              spawn_density_function=partial(spawn_density_function_discontinuity, get_rear_density(sim, micro_middle_position - micro_length), get_front_density(sim, micro_middle_position + micro_length)))

def micro_case_b_2(sim: Simulation, config_file_name: str):
    with open(config_file_name, "r") as f:
        config = json.load(f)
    road_length = config["road_data"]["1"]["road_length"]
    micro_middle_position = (0.2 * road_length)# + (micro_length / 2.0)
    micro_max_position = road_length - (micro_length / 2.0)
    return PrescribedSimBridge(sim=sim, 
                              road_id="1",
                              initial_middle_s=micro_middle_position,
                              max_middle_s=micro_max_position,
                              margin_s=(micro_length / 2.0),
                              fd=lwr_triangular_fd(),
                              config_file_name=config_file_name,
                              boundary_function=boundary_case_2,
                              spawn_density_function=partial(spawn_density_function_discontinuity, get_rear_density(sim, micro_middle_position - micro_length), get_front_density(sim, micro_middle_position + micro_length)))

def micro_case_b_3(sim: Simulation, config_file_name: str):
    with open(config_file_name, "r") as f:
        config = json.load(f)
    road_length = config["road_data"]["1"]["road_length"]
    micro_middle_position = (0.2 * road_length)# + (micro_length / 2.0)
    micro_max_position = road_length - (micro_length / 2.0)
    return PrescribedSimBridge(sim=sim, 
                              road_id="1",
                              initial_middle_s=micro_middle_position,
                              max_middle_s=micro_max_position,
                              margin_s=(micro_length / 2.0),
                              fd=lwr_triangular_fd(),
                              config_file_name=config_file_name,
                              boundary_function=boundary_case_3,
                              spawn_density_function=partial(spawn_density_function_discontinuity, get_rear_density(sim, micro_middle_position - micro_length), get_front_density(sim, micro_middle_position + micro_length)))

def micro_case_b_4(sim: Simulation, config_file_name: str):
    with open(config_file_name, "r") as f:
        config = json.load(f)
    road_length = config["road_data"]["1"]["road_length"]
    micro_middle_position = (0.2 * road_length)# + (micro_length / 2.0)
    micro_max_position = road_length - (micro_length / 2.0)
    return PrescribedSimBridge(sim=sim, 
                              road_id="1",
                              initial_middle_s=micro_middle_position,
                              max_middle_s=micro_max_position,
                              margin_s=(micro_length / 2.0),
                              fd=lwr_triangular_fd(),
                              config_file_name=config_file_name,
                              boundary_function=boundary_case_4,
                              spawn_density_function=partial(spawn_density_function_discontinuity, get_rear_density(sim, micro_middle_position - micro_length), get_front_density(sim, micro_middle_position + micro_length)))

def micro_case_b_5(sim: Simulation, config_file_name: str,):
    with open(config_file_name, "r") as f:
        config = json.load(f)
    road_length = config["road_data"]["1"]["road_length"]
    micro_middle_position = (0.2 * road_length)# + (micro_length / 2.0)
    micro_max_position = road_length - (micro_length / 2.0)
    return PrescribedSimBridge(sim=sim, 
                              road_id="1",
                              initial_middle_s=micro_middle_position,
                              max_middle_s=micro_max_position,
                              margin_s=(micro_length / 2.0),
                              fd=lwr_triangular_fd(),
                              config_file_name=config_file_name,
                              boundary_function=boundary_case_5,
                              spawn_density_function=partial(spawn_density_function_discontinuity, get_rear_density(sim, micro_middle_position - micro_length), get_front_density(sim, micro_middle_position + micro_length)))

def micro_case_b_6(sim: Simulation, config_file_name: str):
    with open(config_file_name, "r") as f:
        config = json.load(f)
    road_length = config["road_data"]["1"]["road_length"]
    micro_middle_position = (0.2 * road_length)# + (micro_length / 2.0)
    micro_max_position = road_length - (micro_length / 2.0)
    return PrescribedSimBridge(sim=sim, 
                              road_id="1",
                              initial_middle_s=micro_middle_position,
                              max_middle_s=micro_max_position,
                              margin_s=(micro_length / 2.0),
                              fd=lwr_triangular_fd(),
                              config_file_name=config_file_name,
                              boundary_function=boundary_case_6,
                              spawn_density_function=partial(spawn_density_function_discontinuity, get_rear_density(sim, micro_middle_position - micro_length), get_front_density(sim, micro_middle_position + micro_length)))

def micro_case_b_7(sim: Simulation, config_file_name: str):
    with open(config_file_name, "r") as f:
        config = json.load(f)
    road_length = config["road_data"]["1"]["road_length"]
    micro_middle_position = (0.2 * road_length)# + (micro_length / 2.0)
    micro_max_position = road_length - (micro_length / 2.0)
    return PrescribedSimBridge(sim=sim, 
                              road_id="1",
                              initial_middle_s=micro_middle_position,
                              max_middle_s=micro_max_position,
                              margin_s=(micro_length / 2.0),
                              fd=lwr_triangular_fd(),
                              config_file_name=config_file_name,
                              boundary_function=boundary_case_7,
                              spawn_density_function=partial(spawn_density_function_discontinuity, get_rear_density(sim, micro_middle_position - micro_length), get_front_density(sim, micro_middle_position + micro_length)))

def micro_case_b_8(sim: Simulation, config_file_name: str):
    with open(config_file_name, "r") as f:
        config = json.load(f)
    road_length = config["road_data"]["1"]["road_length"]
    micro_middle_position = (0.2 * road_length)# + (micro_length / 2.0)
    micro_max_position = road_length - (micro_length / 2.0)
    return PrescribedSimBridge(sim=sim, 
                              road_id="1",
                              initial_middle_s=micro_middle_position,
                              max_middle_s=micro_max_position,
                              margin_s=(micro_length / 2.0),
                              fd=lwr_triangular_fd(),
                              config_file_name=config_file_name,
                              boundary_function=boundary_case_8,
                              spawn_density_function=partial(spawn_density_function_discontinuity, get_rear_density(sim, micro_middle_position - micro_length), get_front_density(sim, micro_middle_position + micro_length)))

def get_rear_density(sim: Simulation, rear_boundary_s: float):
    sim_road_1 = sim.network.roads["1"]
    lane_cells = [c for c in sim_road_1.cells_for_lane(-1) if c.start_s <= rear_boundary_s]
    return lane_cells[-1].density

def get_front_density(sim: Simulation, front_boundary_s: float):
    sim_road_1 = sim.network.roads["1"]
    lane_cells = [c for c in sim_road_1.cells_for_lane(-1) if c.end_s >= front_boundary_s]
    return lane_cells[0].density

def newell_case(sim: Simulation, config_file_name: str):
    with open(config_file_name, "r") as f:
        config = json.load(f)
    road_length = config["road_data"]["1"]["road_length"]
    micro_middle_position = (0.2 * road_length)# + (micro_length / 2.0)
    micro_max_position = road_length - (micro_length / 2.0)
    v_f = 49.7
    w = 5.7
    rho_j = 0.13
    still_gap = 1.0
    jam_spacing = 1.0 / rho_j
    vehicle_length = jam_spacing - still_gap
    time_headway = 1.0 / (w * rho_j)
    newell_model = NewellModel(v_f=v_f, jam_spacing=jam_spacing, time_gap=time_headway)
    fd = TriangularFD(v_f=v_f, w=w, rho_j=rho_j)
    bridge = SimplifiedSimBridge(sim=sim, 
                              road_id="1",
                              initial_middle_s=micro_middle_position,
                              max_middle_s=micro_max_position,
                              margin_s=(micro_length / 2.0),
                              fd=fd,
                              ftl_model=newell_model,
                              spawn_density_function=partial(spawn_density_function_discontinuity, get_rear_density(sim, micro_middle_position - micro_length), get_front_density(sim, micro_middle_position + micro_length)))
    bridge.spawn_length = vehicle_length
    bridge.min_spawn_distance = still_gap
    return bridge

def idm_case(sim: Simulation, config_file_name: str):
    with open(config_file_name, "r") as f:
        config = json.load(f)
    road_length = config["road_data"]["1"]["road_length"]
    micro_middle_position = (0.2 * road_length)# + (micro_length / 2.0)
    micro_max_position = road_length - (micro_length / 2.0)
    v_f = 49.7
    w = 5.7
    rho_j = 0.13
    still_gap = 1.0
    jam_spacing = 1.0 / rho_j
    vehicle_length = jam_spacing - still_gap
    time_headway = 1.0 / (w * rho_j)
    acceleration_exponent = 4.0
    max_accel = 1.5
    max_decel = 10.0
    idm_model = IDMModel(v_f=v_f, vehicle_length=vehicle_length, still_gap=still_gap, time_headway=time_headway, acceleration_exponent=acceleration_exponent, max_accel=max_accel, max_decel=max_decel)
    fd = TriangularFD(v_f=v_f, w=w, rho_j=rho_j)
    bridge = SimplifiedSimBridge(sim=sim, 
                              road_id="1",
                              initial_middle_s=micro_middle_position,
                              max_middle_s=micro_max_position,
                              margin_s=(micro_length / 2.0),
                              fd=fd,
                              ftl_model=idm_model,
                              spawn_density_function=partial(spawn_density_function_discontinuity, get_rear_density(sim, micro_middle_position - micro_length), get_front_density(sim, micro_middle_position + micro_length)))
    bridge.spawn_length = vehicle_length
    bridge.min_spawn_distance = still_gap
    return bridge

def arz_case(sim: Simulation, config_file_name: str):
    with open(config_file_name, "r") as f:
        config = json.load(f)
    road_length = config["road_data"]["1"]["road_length"]
    micro_middle_position = (0.2 * road_length)# + (micro_length / 2.0)
    micro_max_position = road_length - (micro_length / 2.0)
    v_f = 49.7
    w = 5.7
    rho_j = 0.13
    still_gap = 1.0
    jam_spacing = 1.0 / rho_j
    vehicle_length = jam_spacing - still_gap
    traffic_pressure = w / v_f
    driver_relaxation_time = 1.5
    arz_model = MicroscopicARZVehicleModel(v_max=v_f, rho_max=rho_j, gamma=traffic_pressure, tau=driver_relaxation_time, vehicle_length=vehicle_length)
    fd = TriangularFD(v_f=v_f, w=w, rho_j=rho_j)
    bridge = SimplifiedSimBridge(sim=sim, 
                              road_id="1",
                              initial_middle_s=micro_middle_position,
                              max_middle_s=micro_max_position,
                              margin_s=(micro_length / 2.0),
                              fd=fd,
                              ftl_model=arz_model,
                              spawn_density_function=partial(spawn_density_function_discontinuity, get_rear_density(sim, micro_middle_position - micro_length), get_front_density(sim, micro_middle_position + micro_length)))
    bridge.spawn_length = vehicle_length
    bridge.min_spawn_distance = still_gap
    return bridge


def vanilla_lwr_case(sim: Simulation, config_file_name: str):
    """Mask-free control: the pure fluid solver, no bridge and no micro domain.

    Returning None is the whole implementation -- Logger treats a null bridge as "no mask"
    and writes only the macro log. This isolates the base scheme's own convergence rate, so
    coupler error can be told apart from discretisation error. The case name matches the
    ground-truth file name, so `analysis_helpers.default_gt_case` already resolves it.
    """
    return None


verification_config = {
    "result_folder": "verification_results/",
    "macro_cases" : {
        "m_1": {
            "micro_cases": {
                "micro_case_b_4": micro_case_b_4,
                "micro_case_b_5": micro_case_b_5,
                "newell_case": newell_case,
                "idm_case": idm_case,
                "arz_case": arz_case,
                "vanilla_lwr": vanilla_lwr_case
            }
        },
        "m_2": {
            "micro_cases": {
                "micro_case_b_1": micro_case_b_1,
                "micro_case_b_2": micro_case_b_2,
                "micro_case_b_3": micro_case_b_3,
                "newell_case": newell_case,
                "idm_case": idm_case,
                "arz_case": arz_case,
                "vanilla_lwr": vanilla_lwr_case
            }
        },
        "m_3": {
            "micro_cases": {
                "micro_case_b_1": micro_case_b_1,
                "micro_case_b_7": micro_case_b_7,
                "micro_case_b_8": micro_case_b_8,
                "newell_case": newell_case,
                "idm_case": idm_case,
                "arz_case": arz_case,
                "vanilla_lwr": vanilla_lwr_case
            }
        },
        "m_4": {
            "micro_cases": {
                "micro_case_b_2": micro_case_b_2,
                "micro_case_b_6": micro_case_b_6,
                "newell_case": newell_case,
                "idm_case": idm_case,
                "arz_case": arz_case,
                "vanilla_lwr": vanilla_lwr_case
            }
        }
    }
}

def load_sim_verification_demo(config):
    sim = Simulation.from_json(
        json_path=os.path.join(config["storage_locations"]["simulation_dataset"], "network.json"),
        time_resolution=config["time_step"],
        origin_time=config["time_origin"],
        min_cell_length=config["cell_length"]
    )

    gt = GroundTruthStore.from_parquet(None, os.path.join(config["storage_locations"]["simulation_dataset"], "macro.parquet"))
    sim.initialize_from_ground_truth(gt, time_value=config["time_origin"])
    #v_f = 49.73562026160161
    #w = 5.697835695812354
    #rho_j = 0.1304577157114563
    v_f = 49.7
    w = 5.7
    rho_j = 0.13
    still_gap = 1.0
    jam_spacing = 1.0 / rho_j
    vehicle_length = jam_spacing - still_gap
    time_headway = 1.0 / (w * rho_j)
    acceleration_exponent = 4.0
    max_accel = 1.5
    max_decel = 10.0
    traffic_pressure = w / v_f
    driver_relaxation_time = 1.5

    """
    newell_model = NewellModel(v_f=v_f, jam_spacing=jam_spacing, time_gap=time_headway)
    idm_model = IDMModel(v_f=v_f, vehicle_length=vehicle_length, still_gap=still_gap, time_headway=time_headway, acceleration_exponent=acceleration_exponent, max_accel=max_accel, max_decel=max_decel)
    arz_model = MicroscopicARZVehicleModel(v_max=v_f, rho_max=rho_j, gamma=traffic_pressure, tau=driver_relaxation_time, vehicle_length=vehicle_length)
    bridge = SimplifiedSimBridge(
        sim=sim,
        road_id="1",
        initial_middle_s=15000.0,
        margin_s=4000.0,
        max_middle_s=35000.0,
        fd=TriangularFD(v_f=v_f, w=w, rho_j=rho_j),
        ftl_model=arz_model,
        bridge_callback_name="bridge_step"
    )
    """
    newell_model = NewellModel(v_f=v_f, jam_spacing=jam_spacing, time_gap=time_headway)
    idm_model = IDMModel(v_f=v_f, vehicle_length=vehicle_length, still_gap=still_gap, time_headway=time_headway, acceleration_exponent=acceleration_exponent, max_accel=max_accel, max_decel=max_decel)
    arz_model = MicroscopicARZVehicleModel(v_max=v_f, rho_max=rho_j, gamma=traffic_pressure, tau=driver_relaxation_time, vehicle_length=vehicle_length)
    fd = TriangularFD(v_f=v_f, w=w, rho_j=rho_j)
    bridge = SimplifiedSimBridge(
        sim=sim,
        road_id="1",
        initial_middle_s=15000.0,
        margin_s=250.0,
        max_middle_s=35000.0,
        fd=fd,
        ftl_model=idm_model,
        bridge_callback_name="bridge_step",
        spawn_density_function=partial(spawn_density_function_linear_interpolation, fd.rho_c / 2.0, fd.rho_c / 2.0)
    )
    bridge.spawn_length = vehicle_length
    bridge.min_spawn_distance = still_gap
    logger = Logger(sim, bridge, "test_macro.csv", "test_mask.csv")

    return sim, bridge, logger

def load_sim_verification_config_demo():
    with open("config_demo.json", "r") as f:
        config = json.load(f)
    return config

def run_demo_simplified():
    config = load_sim_verification_config_demo()
    sim, bridge, logger = load_sim_verification_demo(config)
    sim.run(config["time_length"])
    logger.destroy()

def load_sim_verification_config(config_file: str):
    with open(config_file, "r") as f:
        config = json.load(f)
    return config

def load_sim_verification(config_path: str, config: dict, micro_case_function: callable, micro_case_macro_result_path: str, micro_case_mask_result_path: str):
    sim = Simulation.from_json(
        json_path=os.path.join(config["storage_locations"]["simulation_dataset"], "network.json"),
        time_resolution=config["time_step"],
        origin_time=config["time_origin"],
        min_cell_length=config["cell_length"]
    )

    gt = GroundTruthStore.from_parquet(None, os.path.join(config["storage_locations"]["simulation_dataset"], "macro.parquet"))
    sim.initialize_from_ground_truth(gt, time_value=config["time_origin"])

    bridge = micro_case_function(sim, config_path)
    # A None bridge is the mask-free control; Logger then needs the FD explicitly, since it
    # normally reads the velocity relation off the bridge.
    logger = Logger(sim, bridge, micro_case_macro_result_path, micro_case_mask_result_path,
                    fd=None if bridge is not None else lwr_triangular_fd())

    return sim, bridge, logger

# --------------------------------------------------------------------------------------
# Sweep driver
#
# One run per process, because peak RSS is dominated by the fluid mesh and the ground
# truth store and neither is released until the process exits. Concurrency is capped per
# grid rather than globally: cost scales with cell count, so the fine grids need a much
# tighter cap than the coarse ones. Measured on the 125 GB host, a single run peaks around
# 2 GB at dx=32 but 14.5 GB at dx=4, climbing past 20 GB late in the run - six concurrent
# dx=4 runs exhausted RAM.
# --------------------------------------------------------------------------------------

DEFAULT_MAX_WORKERS = 6

# (dx_at_or_below, worker_cap), ascending. Anything finer than the first entry uses its
# cap; anything coarser than the last uses the global ceiling.
WORKER_CAPS = [(4.0, 3), (8.0, 4), (16.0, 6)]


def _grid_of(config_file):
    """"verification_config/case_m_1_dx_128.0_dt_1.0.json" -> "128.0_dt_1.0"."""
    base = os.path.splitext(os.path.basename(config_file))[0]
    marker = "_dx_"
    return base[base.find(marker) + len(marker):]


def _dx_of(grid):
    return float(grid.split("_dt_")[0])


def _workers_for(grid, ceiling):
    dx = _dx_of(grid)
    for threshold, cap in WORKER_CAPS:
        if dx <= threshold:
            return max(1, min(cap, ceiling))
    return max(1, ceiling)


def _last_logged_time(path, tail_bytes=65536):
    """Time value on the final row of a macro log, without reading the file.

    The logs reach 7 GB at dx=4, so completion is checked by seeking to the end rather
    than parsing a column. Logger writes rows in step order, so the last row carries the
    largest time, and `time` is the first column of Logger.MACRO_COLUMNS.
    """
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - tail_bytes))
        lines = [ln for ln in f.read().split(b"\n") if ln.strip()]
    if not lines:
        return None
    try:
        return float(lines[-1].split(b",")[0])
    except ValueError:
        # Landed inside the header, or a partial line from a killed writer.
        return None


def writes_mask_log(macro_case, micro_case):
    """False for the mask-free control, which has no bridge and so no mask log."""
    macro_key = macro_case.replace("case_", "")
    factory = verification_config["macro_cases"][macro_key]["micro_cases"][micro_case]
    return factory is not vanilla_lwr_case


def is_complete(result_folder, macro_case, micro_case, grid):
    """A run counts as complete only if its logs exist and the macro log reaches
    time_length. Logger streams, so a killed job leaves a short file rather than none."""
    run_dir = os.path.join(result_folder, f"{macro_case}_dx_{grid}")
    macro_path = os.path.join(run_dir, f"{micro_case}_macro.csv")
    mask_path = os.path.join(run_dir, f"{micro_case}_mask.csv")
    # The control writes no mask log at all; requiring one would mark it permanently
    # incomplete and re-run it on every invocation.
    needed = [macro_path] + ([mask_path] if writes_mask_log(macro_case, micro_case) else [])
    if not all(os.path.exists(p) for p in needed):
        return False, "missing"
    try:
        config = load_sim_verification_config(
            f"verification_config/{macro_case}_dx_{grid}.json")
        want = float(config["time_length"])
    except (OSError, ValueError, KeyError) as e:
        return False, f"unreadable-config ({type(e).__name__})"
    got = _last_logged_time(macro_path)
    if got is None:
        return False, "empty"
    if abs(got - want) > 1e-6:
        return False, f"truncated (t={got} of {want})"
    return True, "ok"


def enumerate_jobs(result_folder, macro_filter=None, micro_filter=None, grid_filter=None):
    """All (macro_case, micro_case, grid) triples the config implies.

    Grids come from globbing verification_config rather than a hardcoded list, so adding
    a new dx/dt pair there is picked up without touching this file.
    """
    jobs = []
    for macro_key, info in verification_config["macro_cases"].items():
        macro_case = f"case_{macro_key}"
        if macro_filter and macro_key not in macro_filter and macro_case not in macro_filter:
            continue
        for config_file in sorted(glob.glob(f"verification_config/*{macro_key}*.json")):
            grid = _grid_of(config_file)
            if grid_filter and grid not in grid_filter:
                continue
            for micro_case in info["micro_cases"]:
                if micro_filter and micro_case not in micro_filter:
                    continue
                jobs.append((macro_case, micro_case, grid))
    return jobs


def run_single(macro_case, micro_case, grid, result_folder):
    """Execute one verification run. Top-level so it is importable by pool workers."""
    macro_key = macro_case.replace("case_", "")
    micro_case_function = verification_config["macro_cases"][macro_key]["micro_cases"][micro_case]
    config_file = f"verification_config/{macro_case}_dx_{grid}.json"

    run_dir = os.path.join(result_folder, f"{macro_case}_dx_{grid}")
    os.makedirs(run_dir, exist_ok=True)
    macro_path = os.path.join(run_dir, f"{micro_case}_macro.csv")
    mask_path = os.path.join(run_dir, f"{micro_case}_mask.csv")

    config = load_sim_verification_config(config_file)
    sim, bridge, logger = load_sim_verification(
        config_file, config, micro_case_function, macro_path, mask_path)
    # Simulation.record_rollout defaults to True, appending a full active-network snapshot
    # every step for the renderer. Verification never renders, and at dx=4/dt=0.03125 that
    # is 9600 snapshots of ~10000 cells - tens of GB of dead weight. Purely observational,
    # so disabling it cannot change results.
    sim.record_rollout = False
    sim.run(config["time_length"])
    logger.destroy()


def _worker(job):
    """Pool entry point. Returns the job plus a traceback string, or None on success."""
    macro_case, micro_case, grid, result_folder = job
    started = time.time()
    try:
        run_single(macro_case, micro_case, grid, result_folder)
        return job, time.time() - started, None
    except Exception:
        return job, time.time() - started, traceback.format_exc()


def run_verification(result_folder=None, max_workers=DEFAULT_MAX_WORKERS, force=False,
                     macro_filter=None, micro_filter=None, grid_filter=None):
    """Run the verification sweep, skipping runs that already completed.

    Returns the list of jobs that failed. Runs are grouped by grid and each group gets
    its own worker cap (see WORKER_CAPS); groups run coarsest-first so cheap failures
    surface before the expensive grids start.
    """
    result_folder = result_folder or verification_config["result_folder"]
    jobs = enumerate_jobs(result_folder, macro_filter, micro_filter, grid_filter)

    todo, skipped = [], 0
    for macro_case, micro_case, grid in jobs:
        if not force:
            complete, _ = is_complete(result_folder, macro_case, micro_case, grid)
            if complete:
                skipped += 1
                continue
        todo.append((macro_case, micro_case, grid, result_folder))

    print(f"{result_folder}: {len(jobs)} jobs, {skipped} already complete, "
          f"{len(todo)} to run", file=sys.stderr)
    if not todo:
        return []

    by_grid = {}
    for job in todo:
        by_grid.setdefault(job[2], []).append(job)

    failures = []
    done = 0
    # maxtasksperchild=1 so every run gets a fresh interpreter; a reused worker would
    # carry the previous run's mesh and store into the next one.
    ctx = multiprocessing.get_context("spawn")
    for grid in sorted(by_grid, key=_dx_of, reverse=True):
        group = by_grid[grid]
        workers = _workers_for(grid, max_workers)
        print(f"  grid {grid}: {len(group)} runs, {workers} workers", file=sys.stderr)
        with ctx.Pool(processes=workers, maxtasksperchild=1) as pool:
            for job, elapsed, error in pool.imap_unordered(_worker, group):
                done += 1
                tag = f"[{done}/{len(todo)}]"
                if error is None:
                    print(f"{tag} OK   {job[0]} {job[1]} {job[2]} ({elapsed:.1f}s)",
                          file=sys.stderr)
                else:
                    failures.append(job)
                    print(f"{tag} FAIL {job[0]} {job[1]} {job[2]}\n{error}",
                          file=sys.stderr)

    print(f"{result_folder}: {done - len(failures)} succeeded, {len(failures)} failed",
          file=sys.stderr)
    return failures


def print_status(result_folder, macro_filter=None, micro_filter=None, grid_filter=None):
    jobs = enumerate_jobs(result_folder, macro_filter, micro_filter, grid_filter)
    missing, reasons, by_grid = [], {}, {}
    for macro_case, micro_case, grid in jobs:
        complete, why = is_complete(result_folder, macro_case, micro_case, grid)
        if not complete:
            missing.append((macro_case, micro_case, grid))
            reasons.setdefault(why.split(" ")[0], []).append(grid)
            by_grid[grid] = by_grid.get(grid, 0) + 1
    print(f"{result_folder}: {len(jobs) - len(missing)}/{len(jobs)} complete, "
          f"{len(missing)} to run")
    for why, hits in sorted(reasons.items()):
        print(f"  {why}: {len(hits)}")
    for grid in sorted(by_grid, key=_dx_of, reverse=True):
        print(f"    {grid}: {by_grid[grid]}")
    return missing


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Run the hybrid verification sweep.")
    parser.add_argument("--results", default=None,
                        help="output folder (default: verification_config['result_folder'])")
    parser.add_argument("--jobs", type=int, default=DEFAULT_MAX_WORKERS,
                        help=f"max concurrent runs (default {DEFAULT_MAX_WORKERS}); the "
                             "per-grid caps in WORKER_CAPS lower this on fine grids")
    parser.add_argument("--macro", action="append",
                        help="restrict to a macro case, e.g. m_3 (repeatable)")
    parser.add_argument("--micro", action="append",
                        help="restrict to a micro case, e.g. newell_case (repeatable)")
    parser.add_argument("--grid", action="append",
                        help="restrict to a grid, e.g. 4.0_dt_0.03125 (repeatable)")
    parser.add_argument("--force", action="store_true",
                        help="re-run even where complete output already exists")
    parser.add_argument("--status", action="store_true",
                        help="report completion and exit without running anything")
    parser.add_argument("--demo", action="store_true", help="run the single demo case")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.demo:
        run_demo_simplified()
        return 0
    result_folder = args.results or verification_config["result_folder"]
    if args.status:
        print_status(result_folder, args.macro, args.micro, args.grid)
        return 0
    failures = run_verification(
        result_folder=result_folder, max_workers=args.jobs, force=args.force,
        macro_filter=args.macro, micro_filter=args.micro, grid_filter=args.grid)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())