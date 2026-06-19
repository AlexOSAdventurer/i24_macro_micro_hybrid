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

from simulation import Simulation, RolloutRenderer, GroundTruthStore
from i24_trajectory_replayer import I24TrajectoryReplayer
from i24_carla_coupler import I24CarlaCoupler
from i24_micro_bridge import I24MicroSimBridge
from dash import Dash, dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import json
import os


def run_app(
    sim: Simulation,
    rotation_deg: float = 0.0,
    port: int = 8050,
    host: str = "0.0.0.0",
    debug: bool = False,
) -> None:
    renderer = RolloutRenderer(sim, rotation_deg)
    N = renderer.N_frames
    step_marks = {
        i: str(int(renderer.sim_times[i]))
        for i in range(0, N, max(1, N // 10))
    }
    # ms per frame at 1x (one simulated second per real second)
    dt_ms = int((renderer.sim_times[1] - renderer.sim_times[0]) * 1000) if N > 1 else 1000
    speed_options = [
        {"label": "Stop", "value": 0},
        {"label": "1x",   "value": 1},
        {"label": "5x",   "value": 5},
        {"label": "10x",  "value": 10},
    ]
    mask_options = [
        {"label": "Hide Masks", "value": False},
        {"label": "Show Masks",   "value": True}
    ]

    # Build road/lane options from ts_data keys
    road_lane_options = [
        {"label": f"Road {rid} · Lane {lane}", "value": f"{rid}:{lane}"}
        for rid, lane, version in sorted(renderer.ts_data.keys()) if version == "sim"
    ]
    default_road_lane = road_lane_options[0]["value"] if road_lane_options else ""
    default_rid, default_lane = default_road_lane.split(":") if default_road_lane else ("", 0)

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.Div(
                [
                    dcc.RadioItems(
                        id="quantity",
                        options=[{"label": q.capitalize(), "value": q} for q in renderer.QUANTITIES],
                        value="density",
                        inline=True,
                        style={"marginRight": "30px"},
                    ),
                    dcc.RadioItems(
                        id="network-view",
                        options=[
                            {"label": "Base network", "value": "base"},
                            {"label": "Active network", "value": "active"},
                        ],
                        value="base",
                        inline=True,
                        style={"marginRight": "30px"},
                    ),
                    dcc.RadioItems(
                        id="speed",
                        options=speed_options,
                        value=0,
                        inline=True,
                    ),
                    dcc.RadioItems(
                        id="render_masks",
                        options=mask_options,
                        value=False,
                        inline=True,
                    )
                ],
                style={"display": "flex", "padding": "10px", "alignItems": "center"},
            ),
            dcc.Graph(
                id="sim-graph",
                figure=renderer.get_figure(0),
                style={"height": "70vh"},
            ),
            dcc.Slider(
                id="step-slider",
                min=0,
                max=N - 1,
                step=1,
                value=0,
                marks=step_marks,
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            dcc.Interval(id="play-interval", interval=dt_ms, disabled=True),
            html.Hr(),
            html.Div(
                [
                    html.Label("Time-space diagram:", style={"marginRight": "10px", "fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="ts-road-lane",
                        options=road_lane_options,
                        value=default_road_lane,
                        clearable=False,
                        style={"width": "220px"},
                    ),
                ],
                style={"display": "flex", "padding": "10px", "alignItems": "center"},
            ),
            dcc.Graph(
                id="ts-graph-sim",
                figure=renderer.get_ts_figure(default_rid, int(default_lane), version="sim"),
                style={"height": "40vh"},
            ),
            dcc.Graph(
                id="ts-graph-empirical",
                figure=renderer.get_ts_figure(default_rid, int(default_lane), version="empirical"),
                style={"height": "40vh"},
            ),
        ]
    )

    @app.callback(
        Output("sim-graph", "figure"),
        Input("step-slider", "value"),
        Input("quantity", "value"),
        Input("network-view", "value"),
    )
    def update(step_idx: int, quantity: str, network_view: str) -> go.Figure:
        return renderer.get_figure(step_idx, show_base=(network_view == "base"), quantity=quantity)

    @app.callback(
        Output("play-interval", "disabled"),
        Output("play-interval", "interval"),
        Input("speed", "value"),
    )
    def configure_interval(speed: int):
        if speed == 0:
            return True, dt_ms
        return False, dt_ms

    @app.callback(
        Output("ts-graph-sim", "figure"),
        Input("ts-road-lane", "value"),
        Input("quantity", "value"),
        Input("render_masks", "value")
    )
    def update_ts_sim(road_lane: str, quantity: str, render_masks: bool) -> go.Figure:
        rid, lane_str = road_lane.split(":")
        return renderer.get_ts_figure(rid, int(lane_str), quantity, version="sim", render_masks=render_masks)
    
    @app.callback(
        Output("ts-graph-empirical", "figure"),
        Input("ts-road-lane", "value"),
        Input("quantity", "value"),
    )
    def update_ts_empirical(road_lane: str, quantity: str) -> go.Figure:
        rid, lane_str = road_lane.split(":")
        return renderer.get_ts_figure(rid, int(lane_str), quantity, version="empirical")

    @app.callback(
        Output("step-slider", "value"),
        Input("play-interval", "n_intervals"),
        State("step-slider", "value"),
        State("speed", "value"),
    )
    def advance_slider(_, step_idx: int, speed: int) -> int:
        return (step_idx + speed) % N

    app.run(host=host, port=port, debug=debug)

"""
with open("i24_motion_to_dataset.json", "r") as f:
    config = json.load(f)

sim = Simulation.from_json(
    json_path=os.path.join(config["storage_locations"]["simulation_dataset"], "network.json"),
    time_resolution=config["time_step"],
    origin_time=config["time_origin"],
    min_cell_length=100.0
)

gt = GroundTruthStore.from_parquet(os.path.join(config["storage_locations"]["simulation_dataset"], "micro.parquet"), os.path.join(config["storage_locations"]["simulation_dataset"], "macro.parquet"))
sim.initialize_from_ground_truth(gt, time_value=config["time_origin"])
replayer = I24TrajectoryReplayer(gt, dt=1.0, lanes=[-1, -2, -3, -4])
bridge = I24MicroSimBridge(
    sim=sim,
    road_id="2",
    lanes=[-1, -2, -3, -4],
    initial_middle_s=150.0,
    margin_s=150.0,
    max_middle_s=1450,
    update_micro_callback=replayer.step,
    bridge_callback_name="bridge_step"
)
"""

def run_demo_open_loop():
    with open("i24_motion_to_dataset.json", "r") as f:
        config = json.load(f)

    sim = Simulation.from_json(
        json_path=os.path.join(config["storage_locations"]["simulation_dataset"], "network.json"),
        time_resolution=config["time_step"],
        origin_time=config["time_origin"],
        min_cell_length=100.0
    )

    gt = GroundTruthStore.from_parquet(os.path.join(config["storage_locations"]["simulation_dataset"], "micro.parquet"), os.path.join(config["storage_locations"]["simulation_dataset"], "macro.parquet"))
    sim.initialize_from_ground_truth(gt, time_value=config["time_origin"])
    replayer = I24TrajectoryReplayer(gt, dt=1.0, lanes=[-1, -2, -3, -4])
    
    bridge = I24MicroSimBridge(
        sim=sim,
        road_id="2",
        lanes=[-1, -2, -3, -4],
        initial_middle_s=350.0,
        margin_s=150.0,
        max_middle_s=1300,
        micro_coupler=replayer,
        bridge_callback_name="bridge_step"
    )
    bridge_time_window = 600.0 #360.0 #1080.0
    current_bridge_iteration = 1.0
    def update_bridge_callback(current_time, resolution):
        nonlocal bridge
        nonlocal bridge_time_window
        nonlocal current_bridge_iteration
        nonlocal sim
        if ((current_time - sim.origin_time) >= (bridge_time_window * current_bridge_iteration)):
            print("Resetting bridge!")
            bridge.destroy()
            bridge = I24MicroSimBridge(
                sim=sim,
                road_id="2",
                lanes=[-1, -2, -3, -4],
                initial_middle_s=350.0,
                margin_s=150.0,
                max_middle_s=1450,
                micro_coupler=replayer,
                bridge_callback_name="bridge_step"
            )
            #bridge._step(sim.current_time, sim.time_resolution)
            print("Bridge reset!")
            current_bridge_iteration += 1
    sim.register_step_callback(update_bridge_callback, "bridge_restart")
    for i in range(3599):
        sim.step()
    run_app(sim, rotation_deg=82.8192)

def run_demo_carla():
    with open("i24_motion_to_dataset.json", "r") as f:
        config = json.load(f)

    sim = Simulation.from_json(
        json_path=os.path.join(config["storage_locations"]["simulation_dataset"], "network.json"),
        time_resolution=config["time_step"],
        origin_time=config["time_origin"],
        min_cell_length=100.0
    )

    gt = GroundTruthStore.from_parquet(os.path.join(config["storage_locations"]["simulation_dataset"], "micro.parquet"), os.path.join(config["storage_locations"]["simulation_dataset"], "macro.parquet"))
    sim.initialize_from_ground_truth(gt, time_value=config["time_origin"])
    coupler = I24CarlaCoupler(gt, dt=1.0, lanes=[-1, -2, -3, -4], mapping=config, hero_road="2", desired_time=config["time_origin"], desired_s=350.0, visible_window=150.0, ghost_window=0.0, bev_video_path="carla_camera_bev_view_1.mp4")
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
    bridge_time_step = 3600.0 #360.0 #1080.0
    bridge_time_window = 90.0
    current_bridge_iteration = 1
    def update_bridge_callback(current_time, resolution):
        nonlocal bridge
        nonlocal bridge_time_step
        nonlocal current_bridge_iteration
        nonlocal sim
        if ((current_time - sim.origin_time) >= ((bridge_time_step * (current_bridge_iteration - 1)) + bridge_time_window)) and (bridge.running):
            print("Resetting bridge!")
            if (bridge.running):
                bridge.destroy()
        if ((current_time - sim.origin_time) >= (bridge_time_step * (current_bridge_iteration))):
            current_bridge_iteration += 1
            coupler = I24CarlaCoupler(gt, dt=1.0, lanes=[-1, -2, -3, -4], mapping=config, hero_road="2", desired_time=sim.current_time, desired_s=350.0, visible_window=150.0, ghost_window=0.0, bev_video_path=f"carla_camera_bev_view_{current_bridge_iteration}.mp4")
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
            #bridge._step(sim.current_time, sim.time_resolution)
            print("Bridge reset!")
    sim.register_step_callback(update_bridge_callback, "bridge_restart")
    for i in range(3599):
        sim.step()
    run_app(sim, rotation_deg=82.8192)

if __name__ == "__main__":
    run_demo_carla()
