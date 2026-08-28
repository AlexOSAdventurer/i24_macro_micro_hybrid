from __future__ import annotations

from simulation import Simulation, GroundTruthStore, I24WestAndEastNetwork, TriangularFD, RolloutRenderer
from i24_trajectory_replayer import I24TrajectoryReplayer
from i24_micro_bridge import I24MicroSimBridge
from simulation_dataset import I24SimulationData
import optuna
import json
import os

with open("config/2022-11-22.json", "r") as f:
    config = json.load(f)

gt = GroundTruthStore.from_parquet(os.path.join(config["storage_locations"]["simulation_dataset"], "micro.parquet"), os.path.join(config["storage_locations"]["simulation_dataset"], "macro.parquet"))

def run_calibration(trial):
    #{'v_f': 38.00311919090539, 'rho_j': 0.09305952185516403, 'lambda_lc': 0.09997312297982588, 'w': 6.98263909134981}
    #  {'v_f': 36.84345956738889, 'rho_j': 0.11911626274100441, 'lambda_lc': 0.07329849477541688, 'w': 5.5232290087351155}
    print("Creating sim data!")
    v_f = trial.suggest_float("v_f", 30.0, 50.0)
    rho_j = trial.suggest_float("rho_j", 0.07, 0.15)
    lambda_lc = trial.suggest_float("lambda_lc", 0.01, 0.2)
    w = trial.suggest_float("w", 4.5, 8.0)
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
    for lane in lanes:
        renderer._ensure_ts_lane("2", lane)
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
