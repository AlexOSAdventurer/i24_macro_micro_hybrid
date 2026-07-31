import sys
sys.path.append("..")
from functools import partial
import json
from simulation import Simulation, RolloutRenderer, GroundTruthStore, TriangularFD
from bridge_coupler import SimplifiedSimBridge, NewellModel, IDMModel, IIDMModel, MicroscopicARZVehicleModel
from logger import Logger
import os

def spawn_density_function_linear_interpolation(start_density, end_density, s, max_s):
    end_frac = s / max_s
    start_frac = 1.0 - end_frac
    return (start_density * start_frac) + (end_density * end_frac)

def load_sim_demo():
    with open("config_demo.json", "r") as f:
        config = json.load(f)

    sim = Simulation.from_json(
        json_path=os.path.join(config["storage_locations"]["simulation_dataset"], "network.json"),
        time_resolution=config["time_step"],
        origin_time=config["time_origin"],
        min_cell_length=100.0
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
        margin_s=4000.0,
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

def run_demo_simplified():
    sim, bridge, logger = load_sim_demo()
    sim.run(2400.0)
    logger.destroy()

if __name__ == "__main__":
    run_demo_simplified()