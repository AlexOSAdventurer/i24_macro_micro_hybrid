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


import sys
sys.path.append("..")
import sim_verification
from simulation import Simulation, RolloutRenderer, GroundTruthStore, TriangularFD
from bridge_coupler import SimplifiedSimBridge, NewellModel, IDMModel, IIDMModel, MicroscopicARZVehicleModel
from dash import Dash, dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import json
import os
from functools import partial


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

    # Build road/lane options (time-space data is built lazily on selection)
    road_lane_options = [
        {"label": f"Road {rid} · Lane {lane}", "value": f"{rid}:{lane}"}
        for rid, lane in sorted(renderer.ts_road_lanes)
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

def spawn_density_function_linear_interpolation(start_density, end_density, s, max_s):
    end_frac = s / max_s
    start_frac = 1.0 - end_frac
    return (start_density * start_frac) + (end_density * end_frac)

def load_sim_demo(config):
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
        ftl_model=newell_model,
        bridge_callback_name="bridge_step",
        spawn_density_function=partial(spawn_density_function_linear_interpolation, fd.rho_c / 2.0, fd.rho_c / 2.0)
    )
    bridge.spawn_length = vehicle_length
    bridge.min_spawn_distance = still_gap

    return sim, bridge

def get_config():
    with open("config_demo.json", "r") as f:
        config = json.load(f)
    return config

def run_demo_simplified():
    config = get_config()
    sim, bridge = load_sim_demo(config)
    sim.run(config["time_length"])
    run_app(sim, rotation_deg=0.0, port=8052)

def run_demo_verification(macro_name_full: str, macro_name_short: str, micro_case: str):
    config_main_folder = "verification_config/"
    config_case_folder = config_main_folder + macro_name_full
    config_file_path = config_main_folder + f"{macro_name_full}.json"
    config_selection = sim_verification.verification_config["macro_cases"][macro_name_short]["micro_cases"][micro_case]
    config = sim_verification.load_sim_verification_config(config_file_path)
    sim, bridge, logger = sim_verification.load_sim_verification(config_file_path, config, config_selection, "test_macro.csv", "test_mask.csv")
    sim.run(config["time_length"])
    run_app(sim, rotation_deg=0.0, port=8052)

if __name__ == "__main__":
    #run_demo_simplified()
    run_demo_verification("case_m_4_dx_8.0_dt_0.0625", "m_4", "idm_case")
