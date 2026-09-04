from __future__ import annotations
import optuna
import random
from simulation import Simulation, GroundTruthStore, I24WestAndEastNetwork, TriangularFD, RolloutRenderer
from i24_trajectory_replayer import I24TrajectoryReplayer
from i24_micro_bridge import I24MicroSimBridge
from simulation_dataset import I24SimulationData
import json
import os
import numpy as np

datasets = os.listdir("config/")
configs = {}
gts = {}
for dataset in datasets:
    config_path = os.path.join("config/", dataset)
    with open(config_path, "r") as f:
        config = json.load(f)
    configs[dataset] = config
    gt = GroundTruthStore.from_parquet(os.path.join(config["storage_locations"]["simulation_dataset"], "micro.parquet"), os.path.join(config["storage_locations"]["simulation_dataset"], "macro.parquet"))
    gts[dataset] = gt
    print(dataset)

def run_calibration(trial):

    #{'v_f': 38.00311919090539, 'rho_j': 0.09305952185516403, 'lambda_lc': 0.09997312297982588, 'w': 6.98263909134981}
    #  {'v_f': 36.84345956738889, 'rho_j': 0.11911626274100441, 'lambda_lc': 0.07329849477541688, 'w': 5.5232290087351155}
    selected_dataset = random.choice(datasets)
    config = configs[selected_dataset]
    gt = gts[selected_dataset]

    print(f"Creating sim data from {selected_dataset}!")
    v_f = trial.suggest_float("v_f", 25.0, 35.0)
    rho_j = trial.suggest_float("rho_j", 0.06, 0.10)
    lambda_lc = trial.suggest_float("lambda_lc", 0.01, 0.2)
    #v_f = trial.suggest_float("v_f", 30.0, 40.0)
    #rho_j = trial.suggest_float("rho_j", 0.06, 0.13)
    #lambda_lc = trial.suggest_float("lambda_lc", 0.01, 0.1)
    w = trial.suggest_float("w", 3.0, 6.0)
    network_generator = I24WestAndEastNetwork(TriangularFD(v_f, w, rho_j), lambda_lc)
    network_generator.create_network(config["road_data"]["2"]["road_length"], config["road_data"]["2"]["cell_length"], config["road_data"]["2"]["lanes"], lane_width=config["road_data"]["2"]["lane_width"])
    network_generator.save_network(os.path.join(config["storage_locations"]["simulation_dataset"], "network.json"))

    print("Creating sim itself!")
    sim = Simulation.from_json(
        json_path=os.path.join(config["storage_locations"]["simulation_dataset"], "network.json"),
        time_resolution=config["time_step"],
        origin_time=config["time_origin"],
        min_cell_length=100.0
    )

    sim.initialize_from_ground_truth(gt, time_value=config["time_origin"])
    print("Running sim!")
    sim.run(14399.0)
    print("Sim done!")
    renderer = RolloutRenderer(sim)
    lanes = [-1, -2, -3, -4]
    jam_threshold = 15.0
    metric = 0.0
    def l1_error(sim, emp):
        #    return np.abs(sim - emp).sum() / ((sim.reshape(-1).shape[0]) * emp.reshape(-1).max())
        return np.abs(sim - emp).sum() / (sim.reshape(-1).shape[0])

    for lane in lanes:
        renderer._ensure_ts_lane("1", lane)
        renderer._ensure_ts_lane("2", lane)
        sim_velocity_data = renderer.ts_data[("2", lane, "sim")]["velocity"]
        empirical_velocity_data = renderer.ts_data[("2", lane, "empirical")]["velocity"][1:]
        
        sim_in_jam = (sim_velocity_data < jam_threshold).astype(int).reshape(-1)
        empirical_in_jam = (empirical_velocity_data < jam_threshold).astype(int).reshape(-1)
        sim_in_free = (sim_velocity_data >= jam_threshold).astype(int).reshape(-1)
        empirical_in_free = (empirical_velocity_data >= jam_threshold).astype(int).reshape(-1)
        combined_flow_in_jam = -(sim_in_jam * empirical_in_jam).sum() / (empirical_in_jam.sum() + 1e-12)
        combined_flow_in_free = -(sim_in_free * empirical_in_free).sum() / (empirical_in_free.sum() + 1e-12)
        #metric += (combined_flow_in_jam + combined_flow_in_free) / float(len(lanes))
        l1_current = (l1_error(sim_velocity_data, empirical_velocity_data) / (2.0 * float(len(lanes))))
        metric += l1_current
        print("Road 2 ", combined_flow_in_jam, combined_flow_in_free, l1_current)

        sim_velocity_data = renderer.ts_data[("1", lane, "sim")]["velocity"]
        empirical_velocity_data = renderer.ts_data[("1", lane, "empirical")]["velocity"][1:]
        sim_in_jam = (sim_velocity_data < jam_threshold).astype(int).reshape(-1)
        empirical_in_jam = (empirical_velocity_data < jam_threshold).astype(int).reshape(-1)
        sim_in_free = (sim_velocity_data >= jam_threshold).astype(int).reshape(-1)
        empirical_in_free = (empirical_velocity_data >= jam_threshold).astype(int).reshape(-1)
        combined_flow_in_jam = -(sim_in_jam * empirical_in_jam).sum() / (empirical_in_jam.sum() + 1e-12)
        combined_flow_in_free = -(sim_in_free * empirical_in_free).sum() / (empirical_in_free.sum() + 1e-12)

        #metric += (combined_flow_in_jam + combined_flow_in_free) / float(len(lanes))
        l1_current = (l1_error(sim_velocity_data, empirical_velocity_data) / (2.0 * float(len(lanes))))
        metric += l1_current
        print("Road 1 ", combined_flow_in_jam, combined_flow_in_free, l1_current)
    print(metric)
    return metric
    #net_flux_delta = [abs(rear - front) for (rear, front) in zip(rear_flux_entry, front_flux_entry)]
    #return sum(net_flux_delta)

if __name__ == "__main__":
    random.seed(42)
    study = optuna.create_study()
    study.optimize(run_calibration, n_trials=250, n_jobs=1)
    print(study.best_params)  # E.g. {'x': 2.002108042}
