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

database_file = "run_data/results2.db"

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
        for i in range(int(config["time_length"])):
            sim.step()
        renderer = RolloutRenderer(sim)
        store = RolloutStore(database_file)
        run_id = f"lwr_triangular_{dataset}"
        run_id = store.put_renderer(renderer, run_id, metadata={"dataset_file": dataset, "config_folder": config, "config_path": config_path})
        print(f"{dataset} LWR run stored as {run_id}!")


if __name__ == "__main__":
    run_lwr_triangular()
