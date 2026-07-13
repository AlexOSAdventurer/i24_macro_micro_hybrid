from __future__ import annotations

from simulation import Simulation, GroundTruthStore, I24WestAndEastNetwork, TriangularFD, RolloutRenderer
from i24_trajectory_replayer import I24TrajectoryReplayer
from i24_micro_bridge import I24MicroSimBridge
from simulation_dataset import I24SimulationData
from i24_carla_coupler import I24CarlaCoupler
import optuna
import json
import os

with open("i24_motion_to_dataset.json", "r") as f:
    config = json.load(f)

gt = GroundTruthStore.from_parquet(os.path.join(config["storage_locations"]["simulation_dataset"], "micro.parquet"), os.path.join(config["storage_locations"]["simulation_dataset"], "macro.parquet"))

def run_calibration(trial):
    #{'v_f': 38.00311919090539, 'rho_j': 0.09305952185516403, 'lambda_lc': 0.09997312297982588, 'w': 6.98263909134981}
    #  {'v_f': 36.84345956738889, 'rho_j': 0.11911626274100441, 'lambda_lc': 0.07329849477541688, 'w': 5.5232290087351155}
    #v_f = trial.suggest_float("v_f", 30.0, 50.0)
    #rho_j = trial.suggest_float("rho_j", 0.07, 0.15)
    #lambda_lc = trial.suggest_float("lambda_lc", 0.05, 0.2)
    #w = trial.suggest_float("w", 5.0, 8.0)
    v_f=49.816011505539535
    w=6.053452290089522
    rho_j=0.12998583138493472
    lambda_lc=0.10076621081371681
    min_spawn_length = trial.suggest_float("min_spawn_length", 4.0, 15.0)
    min_spawn_distance = trial.suggest_float("min_spawn_distance", 1.0, 10.0)
    min_spawn_length += min_spawn_distance
    sim_data = I24SimulationData()
    sim_data.network_generator = I24WestAndEastNetwork(TriangularFD(v_f, w, rho_j), lambda_lc)
    sim_data.network_generator.create_network(sim_data.config["road_data"]["2"]["road_length"], sim_data.config["road_data"]["2"]["cell_length"], sim_data.config["road_data"]["2"]["lanes"], lane_width=sim_data.config["road_data"]["2"]["lane_width"])
    sim_data.network_generator.save_network(sim_data.network_path)

    with open("i24_motion_to_dataset.json", "r") as f:
        config = json.load(f)

    sim = Simulation.from_json(
        json_path=os.path.join(config["storage_locations"]["simulation_dataset"], "network.json"),
        time_resolution=config["time_step"],
        origin_time=config["time_origin"],
        min_cell_length=100.0
    )

    sim.initialize_from_ground_truth(gt, time_value=config["time_origin"])
    coupler = I24CarlaCoupler(gt, dt=1.0, lanes=[-1, -2, -3, -4], mapping=config, hero_road="2", desired_time=config["time_origin"], desired_s=350.0, visible_window=150.0, ghost_window=0.0)
    coupler.min_spawn_length = min_spawn_length
    coupler.min_spawn_distance = min_spawn_distance
    bridge = I24MicroSimBridge(
        sim=sim,
        road_id="2",
        lanes=[-1, -2, -3, -4],
        initial_middle_s=350.0,
        margin_s=150.0,
        max_middle_s=1300,
        micro_coupler=coupler,
        bridge_callback_name="bridge_step"
    )
    bridge_time_window = 90.0
    bridge_time_step = 1800.0
    current_bridge_iteration = 1.0

    rear_flux_entry = []
    front_flux_entry = []
    bridge_active = True
    
    def update_bridge_callback(current_time, resolution):
        nonlocal bridge
        nonlocal bridge_time_window
        nonlocal current_bridge_iteration
        nonlocal sim
        nonlocal rear_flux_entry
        nonlocal front_flux_entry
        nonlocal bridge_active
        if ((current_time - sim.origin_time) >= ((bridge_time_step * (current_bridge_iteration - 1)) + bridge_time_window)) and (bridge_active):
            print(bridge.middle_s, bridge.max_middle_s, bridge_time_window, current_bridge_iteration)
            print("Rear: ", [bridge.flow_memory_rear[lane] for lane in bridge.flow_memory_rear])
            print("Front: ", [bridge.flow_memory_front[lane] for lane in bridge.flow_memory_front])
            rear_flux_entry.append(sum([bridge.flow_memory_rear[lane] for lane in bridge.flow_memory_rear]))
            front_flux_entry.append(sum([bridge.flow_memory_front[lane] for lane in bridge.flow_memory_front]))
            bridge_active = False
            if bridge.running:
                bridge.destroy()

        if ((current_time - sim.origin_time) >= (bridge_time_step * (current_bridge_iteration))):
            current_bridge_iteration += 1
            coupler = I24CarlaCoupler(gt, dt=1.0, lanes=[-1, -2, -3, -4], mapping=config, hero_road="2", desired_time=sim.current_time, desired_s=350.0, visible_window=150.0, ghost_window=0.0)
            coupler.min_spawn_length = min_spawn_length
            coupler.min_spawn_distance = min_spawn_distance
            bridge = I24MicroSimBridge(
                sim=sim,
                road_id="2",
                lanes=[-1, -2, -3, -4],
                initial_middle_s=350.0,
                margin_s=150.0,
                max_middle_s=1300,
                micro_coupler=coupler,
                bridge_callback_name="bridge_step"
            )
            bridge_active = True
    
    sim.register_step_callback(update_bridge_callback, "bridge_restart")
    for i in range(3599):
        sim.step()
    renderer = RolloutRenderer(sim)
    lanes = [-1, -2, -3, -4]
    jam_threshold = 15.0
    metric = 0.0
    for lane in lanes:
        sim_velocity_data = renderer.ts_data[("2", lane, "sim")]["velocity"]
        empirical_velocity_data = renderer.ts_data[("2", lane, "empirical")]["velocity"][1:]
        sim_in_jam = (sim_velocity_data < jam_threshold).astype(int).reshape(-1)
        empirical_in_jam = (empirical_velocity_data < jam_threshold).astype(int).reshape(-1)
        sim_in_free = (sim_velocity_data >= jam_threshold).astype(int).reshape(-1)
        empirical_in_free = (empirical_velocity_data >= jam_threshold).astype(int).reshape(-1)
        combined_flow_in_jam = -(sim_in_jam * empirical_in_jam).sum() / (empirical_in_jam.sum() + 1e-12)
        combined_flow_in_free = -(sim_in_free * empirical_in_free).sum() / (empirical_in_free.sum() + 1e-12)
        metric += (combined_flow_in_jam + combined_flow_in_free) / float(len(lanes))
        print(combined_flow_in_jam, combined_flow_in_free)
    print(metric)
    return metric
    #net_flux_delta = [abs(rear - front) for (rear, front) in zip(rear_flux_entry, front_flux_entry)]
    #return sum(net_flux_delta)

if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(run_calibration, n_trials=100, n_jobs=1)
    print(study.best_params)  # E.g. {'x': 2.002108042}
