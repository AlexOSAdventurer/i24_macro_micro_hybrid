import sys
sys.path.append("..")
from functools import partial
import json
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
    micro_middle_position = (0.2 * road_length) + (micro_length / 2.0)
    micro_max_position = road_length - (micro_length / 2.0)
    return PrescribedSimBridge(sim=sim, 
                              road_id="1",
                              initial_middle_s=micro_middle_position,
                              max_middle_s=micro_max_position,
                              margin_s=(micro_length / 2.0),
                              fd=lwr_triangular_fd(),
                              config_file_name=config_file_name,
                              boundary_function=boundary_case_1)

def micro_case_b_2(sim: Simulation, config_file_name: str):
    with open(config_file_name, "r") as f:
        config = json.load(f)
    road_length = config["road_data"]["1"]["road_length"]
    micro_middle_position = (0.2 * road_length) + (micro_length / 2.0)
    micro_max_position = road_length - (micro_length / 2.0)
    return PrescribedSimBridge(sim=sim, 
                              road_id="1",
                              initial_middle_s=micro_middle_position,
                              max_middle_s=micro_max_position,
                              margin_s=(micro_length / 2.0),
                              fd=lwr_triangular_fd(),
                              config_file_name=config_file_name,
                              boundary_function=boundary_case_2)

def micro_case_b_3(sim: Simulation, config_file_name: str, gt: GroundTruthStore):
    with open(config_file_name, "r") as f:
        config = json.load(f)
    road_length = config["road_data"]["1"]["road_length"]
    micro_middle_position = (0.2 * road_length) + (micro_length / 2.0)
    micro_max_position = road_length - (micro_length / 2.0)
    return PrescribedSimBridge(sim=sim, 
                              road_id="1",
                              initial_middle_s=micro_middle_position,
                              max_middle_s=micro_max_position,
                              margin_s=(micro_length / 2.0),
                              fd=lwr_triangular_fd(),
                              config_file_name=config_file_name,
                              boundary_function=boundary_case_3)

def micro_case_b_4(sim: Simulation, config_file_name: str):
    with open(config_file_name, "r") as f:
        config = json.load(f)
    road_length = config["road_data"]["1"]["road_length"]
    micro_middle_position = (0.2 * road_length) + (micro_length / 2.0)
    micro_max_position = road_length - (micro_length / 2.0)
    return PrescribedSimBridge(sim=sim, 
                              road_id="1",
                              initial_middle_s=micro_middle_position,
                              max_middle_s=micro_max_position,
                              margin_s=(micro_length / 2.0),
                              fd=lwr_triangular_fd(),
                              config_file_name=config_file_name,
                              boundary_function=boundary_case_4)

def micro_case_b_5(sim: Simulation, config_file_name: str,):
    with open(config_file_name, "r") as f:
        config = json.load(f)
    road_length = config["road_data"]["1"]["road_length"]
    micro_middle_position = (0.2 * road_length) + (micro_length / 2.0)
    micro_max_position = road_length - (micro_length / 2.0)
    return PrescribedSimBridge(sim=sim, 
                              road_id="1",
                              initial_middle_s=micro_middle_position,
                              max_middle_s=micro_max_position,
                              margin_s=(micro_length / 2.0),
                              fd=lwr_triangular_fd(),
                              config_file_name=config_file_name,
                              boundary_function=boundary_case_5)

def micro_case_b_6(sim: Simulation, config_file_name: str):
    with open(config_file_name, "r") as f:
        config = json.load(f)
    road_length = config["road_data"]["1"]["road_length"]
    micro_middle_position = (0.2 * road_length) + (micro_length / 2.0)
    micro_max_position = road_length - (micro_length / 2.0)
    return PrescribedSimBridge(sim=sim, 
                              road_id="1",
                              initial_middle_s=micro_middle_position,
                              max_middle_s=micro_max_position,
                              margin_s=(micro_length / 2.0),
                              fd=lwr_triangular_fd(),
                              config_file_name=config_file_name,
                              boundary_function=boundary_case_6)

def micro_case_b_7(sim: Simulation, config_file_name: str):
    with open(config_file_name, "r") as f:
        config = json.load(f)
    road_length = config["road_data"]["1"]["road_length"]
    micro_middle_position = (0.2 * road_length) + (micro_length / 2.0)
    micro_max_position = road_length - (micro_length / 2.0)
    return PrescribedSimBridge(sim=sim, 
                              road_id="1",
                              initial_middle_s=micro_middle_position,
                              max_middle_s=micro_max_position,
                              margin_s=(micro_length / 2.0),
                              fd=lwr_triangular_fd(),
                              config_file_name=config_file_name,
                              boundary_function=boundary_case_7)

def micro_case_b_8(sim: Simulation, config_file_name: str):
    with open(config_file_name, "r") as f:
        config = json.load(f)
    road_length = config["road_data"]["1"]["road_length"]
    micro_middle_position = (0.2 * road_length) + (micro_length / 2.0)
    micro_max_position = road_length - (micro_length / 2.0)
    return PrescribedSimBridge(sim=sim, 
                              road_id="1",
                              initial_middle_s=micro_middle_position,
                              max_middle_s=micro_max_position,
                              margin_s=(micro_length / 2.0),
                              fd=lwr_triangular_fd(),
                              config_file_name=config_file_name,
                              boundary_function=boundary_case_8)

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
    micro_middle_position = (0.2 * road_length) + (micro_length / 2.0)
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
                              spawn_density_function=partial(spawn_density_function_linear_interpolation, get_rear_density(sim, micro_middle_position - micro_length), get_front_density(sim, micro_middle_position + micro_length)))
    bridge.spawn_length = vehicle_length
    bridge.min_spawn_distance = still_gap
    return bridge

def idm_case(sim: Simulation, config_file_name: str):
    with open(config_file_name, "r") as f:
        config = json.load(f)
    road_length = config["road_data"]["1"]["road_length"]
    micro_middle_position = (0.2 * road_length) + (micro_length / 2.0)
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
                              spawn_density_function=partial(spawn_density_function_linear_interpolation, get_rear_density(sim, micro_middle_position - micro_length), get_front_density(sim, micro_middle_position + micro_length)))
    bridge.spawn_length = vehicle_length
    bridge.min_spawn_distance = still_gap
    return bridge

def arz_case(sim: Simulation, config_file_name: str):
    with open(config_file_name, "r") as f:
        config = json.load(f)
    road_length = config["road_data"]["1"]["road_length"]
    micro_middle_position = (0.2 * road_length) + (micro_length / 2.0)
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
                              spawn_density_function=partial(spawn_density_function_linear_interpolation, get_rear_density(sim, micro_middle_position - micro_length), get_front_density(sim, micro_middle_position + micro_length)))
    bridge.spawn_length = vehicle_length
    bridge.min_spawn_distance = still_gap
    return bridge


verification_config = {
    "result_folder": "verification_results/",
    "macro_cases" : {
        "m_1": {
            "micro_cases": {
                "micro_case_b_4": micro_case_b_4,
                "micro_case_b_5": micro_case_b_5,
                "newell_case": newell_case,
                "idm_case": idm_case,
                "arz_case": arz_case
            }
        },
        "m_2": {
            "micro_cases": {
                "micro_case_b_1": micro_case_b_1,
                "micro_case_b_2": micro_case_b_2,
                "micro_case_b_3": micro_case_b_3,
                "newell_case": newell_case,
                "idm_case": idm_case,
                "arz_case": arz_case
            }
        },
        "m_3": {
            "micro_cases": {
                "micro_case_b_1": micro_case_b_1,
                "micro_case_b_7": micro_case_b_7,
                "micro_case_b_8": micro_case_b_8,
                "newell_case": newell_case,
                "idm_case": idm_case,
                "arz_case": arz_case
            }
        },
        "m_4": {
            "micro_cases": {
                "micro_case_b_2": micro_case_b_2,
                "micro_case_b_6": micro_case_b_6,
                "newell_case": newell_case,
                "idm_case": idm_case,
                "arz_case": arz_case
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
    logger = Logger(sim, bridge, micro_case_macro_result_path, micro_case_mask_result_path)

    return sim, bridge, logger

def run_verification():
    result_folder = verification_config["result_folder"]
    for macro_case in verification_config["macro_cases"]:
        macro_case_info = verification_config["macro_cases"][macro_case]
        for config_file in glob.glob(f"verification_config/*{macro_case}*.json"):
            macro_case_result_folder = os.path.join(result_folder, os.path.splitext(os.path.basename(config_file))[0])
            os.makedirs(macro_case_result_folder, exist_ok=True)
            for micro_case in macro_case_info["micro_cases"]:
                print(macro_case, config_file, macro_case_result_folder, micro_case)
                micro_case_macro_result_path = os.path.join(macro_case_result_folder, f"{micro_case}_macro.csv")
                micro_case_mask_result_path = os.path.join(macro_case_result_folder, f"{micro_case}_mask.csv")
                micro_case_function = macro_case_info["micro_cases"][micro_case]
                config = load_sim_verification_config(config_file)
                sim, bridge, logger = load_sim_verification(config_file, config, micro_case_function, micro_case_macro_result_path, micro_case_mask_result_path)
                sim.run(config["time_length"])
                logger.destroy()
            

if __name__ == "__main__":
    #run_demo_simplified()
    run_verification()