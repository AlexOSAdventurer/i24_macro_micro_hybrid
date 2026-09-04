"""Dash viewer for Simulation rollouts.

Usage
-----
From a notebook or script::

    from simulation import Simulation
    from sim_demo import run_app

    sim = ...  # Simulation with rollout_results populated
    run_app(sim)

Then open http://localhost:8050 in your browser.
For remote machines, SSH port-forward first::

    ssh -L 8050:localhost:8050 user@host

Standalone (requires a pickled Simulation at demo_sim.pickle)::

    python sim_demo.py
"""
from __future__ import annotations
import sim_calibration_metanet
from simulation import Simulation, RolloutRenderer, GroundTruthStore, I24WestAndEastNetworkCollapsed, METANETModel, METANETParams, TriangularFD
from rollout_store import RolloutStore
from i24_trajectory_replayer import I24TrajectoryReplayer
from i24_carla_coupler import I24CarlaCoupler
from i24_sumo_coupler import I24SumoCoupler
from i24_micro_bridge import I24MicroSimBridge
from dash import Dash, dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import json
import os
import numpy as np

database_file = "run_data/results_final.db"

def run_lwr_triangular():
    config_folder = "config/"
    datasets = os.listdir("config")
    for dataset in datasets:
        config_path = os.path.join(config_folder, dataset)
        with open(config_path, "r") as f:
            config = json.load(f)

        sim = Simulation.from_json(
            json_path=os.path.join(config["storage_locations"]["simulation_dataset"], "network.json"),
            time_resolution=config["time_step"],
            origin_time=config["time_origin"],
            min_cell_length=config["cell_length"]
        )

        gt = GroundTruthStore.from_parquet(os.path.join(config["storage_locations"]["simulation_dataset"], "micro.parquet"), os.path.join(config["storage_locations"]["simulation_dataset"], "macro.parquet"))
        sim.initialize_from_ground_truth(gt, time_value=config["time_origin"])
        for i in range(int(config["time_length"] - 1.0)):
            sim.step()
        renderer = RolloutRenderer(sim)
        store = RolloutStore(database_file)
        run_id = f"lwr_triangular_{dataset}"
        run_id = store.put_renderer(renderer, run_id, metadata={"dataset_file": dataset, "config_folder": config, "config_path": config_path})
        print(f"{dataset} LWR run stored as {run_id}!")

def run_sumo():
    config_folder = "config/"
    datasets = os.listdir("config")
    for dataset in datasets:
        for road in ["1", "2"]:
            config_path = os.path.join(config_folder, dataset)
            with open(config_path, "r") as f:
                config = json.load(f)
            start_time = config["time_origin"] + 3600.0
            end_time = config["time_origin"] + config["time_length"]
            episode_step = 360.0
            for current_start_time in np.arange(start_time, end_time, episode_step):
                current_episode_step = min(episode_step, (end_time - current_start_time)) - 1.0
                try:
                    sim = Simulation.from_json(
                        json_path=os.path.join(config["storage_locations"]["simulation_dataset"], "network.json"),
                        time_resolution=config["time_step"],
                        origin_time=current_start_time,
                        min_cell_length=config["cell_length"]
                    )

                    gt = GroundTruthStore.from_parquet(os.path.join(config["storage_locations"]["simulation_dataset"], "micro.parquet"), os.path.join(config["storage_locations"]["simulation_dataset"], "macro.parquet"))
                    sim.initialize_from_ground_truth(gt, time_value=current_start_time)
                    params = {'v_f': 25.02031797294094, 'rho_j': 0.07719493079089293, 'lambda_lc': 0.13384902076274044, 'w': 5.728460762748048}
                    triangular_fd = TriangularFD(v_f=params['v_f'], w=params['w'], rho_j=params['rho_j'])

                    coupler = I24SumoCoupler(
                        gt,
                        dt=config["time_step"],
                        fd=triangular_fd,
                        lanes=[-1, -2, -3, -4],
                        mapping=config,
                        hero_road=road,
                        desired_time=current_start_time,
                        desired_s=350.0,
                        visible_window=150.0,
                        ghost_window=0.0,
                        step_length=0.1,
                        seed=42,
                        gui=False,
                        verbose=True,
                    )
                    #I24CarlaCoupler(gt, dt=1.0, lanes=[-1, -2, -3, -4], mapping=config, hero_road="2", desired_time=config["time_origin"], desired_s=350.0, visible_window=150.0, ghost_window=0.0, bev_video_path="carla_camera_1_low_congestion")
                    bridge = I24MicroSimBridge(
                        sim=sim,
                        road_id=road,
                        lanes=[-1, -2, -3, -4],
                        initial_middle_s=350.0,
                        margin_s=150.0,
                        max_middle_s=1300,
                        micro_coupler=coupler,
                        bridge_callback_name="bridge_step"
                    )
                    current_bridge_iteration = 1
                    def update_bridge_callback(current_time, resolution):
                        nonlocal bridge
                        nonlocal current_bridge_iteration
                        nonlocal sim
                        if (not bridge.running):
                            current_bridge_iteration += 1
                            coupler = I24SumoCoupler(
                                gt,
                                dt=config["time_step"],
                                fd=triangular_fd,
                                lanes=[-1, -2, -3, -4],
                                mapping=config,
                                hero_road=road,
                                desired_time=sim.current_time,
                                desired_s=350.0,
                                visible_window=150.0,
                                ghost_window=0.0,
                                step_length=0.1,
                                seed=42,
                                gui=False,
                                verbose=True,
                            )
                            
                            bridge = I24MicroSimBridge(
                                sim=sim,
                                road_id=road,
                                lanes=[-1, -2, -3, -4],
                                initial_middle_s=350.0,
                                margin_s=150.0,
                                max_middle_s=1300,
                                micro_coupler=coupler,
                                bridge_callback_name="bridge_step"
                            )
                            print("Bridge reset!")
                    #sim.register_poststep_callback(update_bridge_callback, "bridge_restart")
                    for i in range(int(current_episode_step)):
                        sim.step()
                    renderer = RolloutRenderer(sim)
                    store = RolloutStore(database_file)
                    run_id = f"sumo_road({road})_timeorigin({current_start_time})_episodelength({current_episode_step})_{dataset}"
                    run_id = store.put_renderer(renderer, run_id, metadata={"dataset_file": dataset, "config_folder": config, "config_path": config_path})
                    print(f"{dataset} SUMO run stored as {run_id}!")
                except Exception as e:
                    print(f"Exception {e} raised during running {dataset}, {road}, {current_start_time}, {current_episode_step}, range ending in {end_time}. Skipping it.")


def run_carla():
    config_folder = "config/"
    datasets = os.listdir("config")
    for dataset in datasets:
        for road in ["1", "2"]:
            config_path = os.path.join(config_folder, dataset)
            with open(config_path, "r") as f:
                config = json.load(f)
            start_time = config["time_origin"] + 3600.0
            end_time = config["time_origin"] + config["time_length"]
            episode_step = 360.0
            for current_start_time in np.arange(start_time, end_time, episode_step):
                current_episode_step = min(episode_step, (end_time - current_start_time)) - 1.0
                try:
                    sim = Simulation.from_json(
                        json_path=os.path.join(config["storage_locations"]["simulation_dataset"], "network.json"),
                        time_resolution=config["time_step"],
                        origin_time=current_start_time,
                        min_cell_length=config["cell_length"]
                    )

                    gt = GroundTruthStore.from_parquet(os.path.join(config["storage_locations"]["simulation_dataset"], "micro.parquet"), os.path.join(config["storage_locations"]["simulation_dataset"], "macro.parquet"))
                    sim.initialize_from_ground_truth(gt, time_value=current_start_time)
                    params = {'v_f': 25.02031797294094, 'rho_j': 0.07719493079089293, 'lambda_lc': 0.13384902076274044, 'w': 5.728460762748048}
                    triangular_fd = TriangularFD(v_f=params['v_f'], w=params['w'], rho_j=params['rho_j'])

                    coupler = I24CarlaCoupler(
                        gt, 
                        dt=1.0, 
                        fd=triangular_fd, 
                        lanes=[-1, -2, -3, -4], 
                        mapping=config, 
                        hero_road=road, 
                        desired_time=current_start_time, 
                        desired_s=350.0, 
                        visible_window=150.0, 
                        ghost_window=0.0)

                    bridge = I24MicroSimBridge(
                        sim=sim,
                        road_id=road,
                        lanes=[-1, -2, -3, -4],
                        initial_middle_s=350.0,
                        margin_s=150.0,
                        max_middle_s=1300,
                        micro_coupler=coupler,
                        bridge_callback_name="bridge_step"
                    )
                    for i in range(int(current_episode_step)):
                        sim.step()
                    renderer = RolloutRenderer(sim)
                    store = RolloutStore(database_file)
                    run_id = f"carla_road({road})_timeorigin({current_start_time})_episodelength({current_episode_step})_{dataset}"
                    run_id = store.put_renderer(renderer, run_id, metadata={"dataset_file": dataset, "config_folder": config, "config_path": config_path})
                    print(f"{dataset} CARLA run stored as {run_id}!")
                except Exception as e:
                    print(f"Exception {e} raised during running {dataset}, {road}, {current_start_time}, {current_episode_step}, range ending in {end_time}. Skipping it.")

if __name__ == "__main__":
    #run_lwr_triangular()
    #run_sumo()
    run_carla()
