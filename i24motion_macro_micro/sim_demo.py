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
from i24_trajectory_replayer import I24TrajectoryReplayer
from i24_carla_coupler import I24CarlaCoupler
from i24_sumo_coupler import I24SumoCoupler
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

    # Build road/lane options from ts_road_lanes, NOT from ts_data: the time-space
    # entries are built lazily (only for the lane being viewed), so ts_data is still
    # empty at this point and the dropdown would come up with no options at all.
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
        if not road_lane or ":" not in road_lane:
            return go.Figure()
        rid, lane_str = road_lane.split(":")
        return renderer.get_ts_figure(rid, int(lane_str), quantity, version="sim", render_masks=render_masks)
    
    @app.callback(
        Output("ts-graph-empirical", "figure"),
        Input("ts-road-lane", "value"),
        Input("quantity", "value"),
    )
    def update_ts_empirical(road_lane: str, quantity: str) -> go.Figure:
        if not road_lane or ":" not in road_lane:
            return go.Figure()
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

def run_demo_lwr_triangular():
    with open("config/2022-11-30.json", "r") as f:
        config = json.load(f)

    sim = Simulation.from_json(
        json_path=os.path.join(config["storage_locations"]["simulation_dataset"], "network.json"),
        time_resolution=config["time_step"],
        origin_time=config["time_origin"],
        min_cell_length=100.0
    )

    gt = GroundTruthStore.from_parquet(os.path.join(config["storage_locations"]["simulation_dataset"], "micro.parquet"), os.path.join(config["storage_locations"]["simulation_dataset"], "macro.parquet"))
    sim.initialize_from_ground_truth(gt, time_value=config["time_origin"])
    for i in range(14399):
        sim.step()
    run_app(sim, rotation_deg=82.8192)

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
    with open("config/2022-11-30.json", "r") as f:
        config = json.load(f)

    sim = Simulation.from_json(
        json_path=os.path.join(config["storage_locations"]["simulation_dataset"], "network.json"),
        time_resolution=config["time_step"],
        origin_time=config["time_origin"]+2750.0,
        min_cell_length=100.0
    )

    gt = GroundTruthStore.from_parquet(os.path.join(config["storage_locations"]["simulation_dataset"], "micro.parquet"), os.path.join(config["storage_locations"]["simulation_dataset"], "macro.parquet"))
    sim.initialize_from_ground_truth(gt, time_value=config["time_origin"]+2750.0)
    params = {'v_f': 25.02031797294094, 'rho_j': 0.07719493079089293, 'lambda_lc': 0.13384902076274044, 'w': 5.728460762748048}
    #params = {'v_f': 30.019341712559083, 'rho_j': 0.09411385875052518, 'lambda_lc': 0.16914326352090997, 'w': 4.573064692495487} #. Best is trial 71 with value: -1.6869119514065773.
    triangular_fd = TriangularFD(v_f=params['v_f'], w=params['w'], rho_j=params['rho_j'])

    coupler = I24CarlaCoupler(gt, dt=1.0, fd=triangular_fd, lanes=[-1, -2, -3, -4], mapping=config, hero_road="2", desired_time=config["time_origin"]+2750.0, desired_s=350.0, visible_window=150.0, ghost_window=0.0, bev_video_path="carla_camera_1_low_congestion")
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
    bridge_time_step = 600.0 #1200.0 #360.0 #1080.0
    bridge_time_window = 150.0
    current_bridge_iteration = 1
    def update_bridge_callback(current_time, resolution):
        nonlocal bridge
        nonlocal bridge_time_step
        nonlocal current_bridge_iteration
        nonlocal sim
        if ((current_time - sim.origin_time) >= (bridge_time_step * (current_bridge_iteration))):
            if (bridge.running):
                bridge.destroy()
            current_bridge_iteration += 1
            coupler = I24CarlaCoupler(gt, dt=1.0, fd=triangular_fd, lanes=[-1, -2, -3, -4], mapping=config, hero_road="2", desired_time=sim.current_time, desired_s=350.0, visible_window=150.0, ghost_window=0.0, bev_video_path=f"carla_camera_{current_bridge_iteration}_low_congestion")
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
    sim.register_poststep_callback(update_bridge_callback, "bridge_restart")
    for i in range(10799):
        sim.step()
    run_app(sim, rotation_deg=82.8192)

def run_demo_sumo():
    with open("config/2022-11-30.json", "r") as f:
        config = json.load(f)

    sim = Simulation.from_json(
        json_path=os.path.join(config["storage_locations"]["simulation_dataset"], "network.json"),
        time_resolution=config["time_step"],
        origin_time=config["time_origin"]+3600.0,
        min_cell_length=100.0
    )

    gt = GroundTruthStore.from_parquet(os.path.join(config["storage_locations"]["simulation_dataset"], "micro.parquet"), os.path.join(config["storage_locations"]["simulation_dataset"], "macro.parquet"))
    sim.initialize_from_ground_truth(gt, time_value=config["time_origin"]+3600.0)
    params = {'v_f': 25.02031797294094, 'rho_j': 0.07719493079089293, 'lambda_lc': 0.13384902076274044, 'w': 5.728460762748048}
    #params = {'v_f': 30.019341712559083, 'rho_j': 0.09411385875052518, 'lambda_lc': 0.16914326352090997, 'w': 4.573064692495487} #. Best is trial 71 with value: -1.6869119514065773.
    triangular_fd = TriangularFD(v_f=params['v_f'], w=params['w'], rho_j=params['rho_j'])

    coupler = I24SumoCoupler(
        gt,
        dt=config["time_step"],
        fd=triangular_fd,
        lanes=[-1, -2, -3, -4],
        mapping=config,
        hero_road="2",
        desired_time=config["time_origin"]+3600.0,
        desired_s=350.0,
        visible_window=150.0,
        ghost_window=0.0,
        step_length=0.1,
        seed=42,
        gui=True,
        verbose=True,
    )
    #I24CarlaCoupler(gt, dt=1.0, lanes=[-1, -2, -3, -4], mapping=config, hero_road="2", desired_time=config["time_origin"], desired_s=350.0, visible_window=150.0, ghost_window=0.0, bev_video_path="carla_camera_1_low_congestion")
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
    bridge_time_step = 600.0 #1200.0 #360.0 #1080.0
    bridge_time_window = 90.0
    current_bridge_iteration = 1
    def update_bridge_callback(current_time, resolution):
        nonlocal bridge
        nonlocal bridge_time_step
        nonlocal current_bridge_iteration
        nonlocal sim
        """
        if ((current_time - sim.origin_time) >= ((bridge_time_step * (current_bridge_iteration - 1)) + bridge_time_window)) and (bridge.running):
            print("Resetting bridge!")
            if (bridge.running):
                bridge.destroy()
        """
        if ((current_time - sim.origin_time) >= (bridge_time_step * (current_bridge_iteration))):
            if (bridge.running):
                bridge.destroy()
            current_bridge_iteration += 1
            coupler = I24SumoCoupler(
                gt,
                dt=config["time_step"],
                fd=triangular_fd,
                lanes=[-1, -2, -3, -4],
                mapping=config,
                hero_road="2",
                desired_time=sim.current_time,
                desired_s=350.0,
                visible_window=150.0,
                ghost_window=0.0,
                step_length=0.1,
                seed=42,
                gui=False,
                verbose=True,
            )
            #I24CarlaCoupler(gt, dt=1.0, lanes=[-1, -2, -3, -4], mapping=config, hero_road="2", desired_time=sim.current_time, desired_s=350.0, visible_window=150.0, ghost_window=0.0, bev_video_path=f"carla_camera_{current_bridge_iteration}_low_congestion")
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
    sim.register_poststep_callback(update_bridge_callback, "bridge_restart")
    for i in range(10799):
        sim.step()
        print(i)
    run_app(sim, rotation_deg=82.8192)

def run_demo_metanet():
    """
    -0.7499999999999981 -0.8339045886961388
    1.0478862366201542 0.44127095672850625
    -1.583904588696137
    [I 2026-08-19 09:21:27,102] Trial 922 finished with value: -1.583904588696137 and parameters: {'tau_s': 47.86898722920049, 'eta_km2_per_h': 54.498074688599296, 'kappa_veh_per_km_lane': 45.95143563683463, 'v_free_kmh': 134.4172710870525, 'rho_crit_veh_per_km_lane': 24.27930712745335, 'alpha': 4.7917591228885, 'jam_threshold': 5.003871736480674}. Best is trial 922 with value: -1.583904588696137.
    """

    with open("i24_motion_to_dataset.json", "r") as f:
        config = json.load(f)
    road_config = config["road_data"]["2"]
    # The FD is decorative under METANET (the model supplies demand and supply
    # itself); it is kept only so the cells carry one for plotting. The collapsed
    # generator scales its rho_j by the lane count.
    generator = I24WestAndEastNetworkCollapsed(
        fd=TriangularFD(v_f=49.73562026160161, w=5.697835695812354, rho_j=0.1304577157114563),
        lambda_lc=0.0,  # unused: one lane per road leaves no pair to exchange across
    )
    generator.create_network(
        road_config["road_length"],
        road_config["cell_length"],
        road_config["lanes"],
        lane_width=road_config["lane_width"],
    )

    lane_counts = set(generator.lanes_per_road.values())
    if len(lane_counts) != 1:
        raise ValueError(
            f"Roads have differing lane counts {generator.lanes_per_road}; a single "
            f"shared METANETParams cannot describe them. Use per_cell_params."
        )

    """
    param_dict = {'tau_s': 36.346679071778006, 
                  'eta_km2_per_h': 21.832917579225892, 
                  'kappa_veh_per_km_lane': 53.30974608164321, 
                  'v_free_kmh': 73.12670738279476, 
                  'rho_crit_veh_per_km_lane': 56.0129859172635, 
                  'alpha': 3.437269967637944}
    """
    param_dict = {'tau_s': 30.376601542684984, 
                  'eta_km2_per_h': 50.49894847234014, 
                  'kappa_veh_per_km_lane': 17.004542409940868, 
                  'v_free_kmh': 107.36993061424273, 
                  'rho_crit_veh_per_km_lane': 18.562273787100544, 
                  'alpha': 2.1228564728742834}
    #{'jam_recall': 0.7997352927753183, 'free_recall': 0.8387589013224821, 'metric_interior': -1.6210698664883636, 'velocity_rmse': 3.88641774859314, 'density_rmse': 0.03729489497405928}
    params = METANETParams.from_paper_units(tau_h=param_dict["tau_s"] / 3600.0,
        eta_km2_per_h=param_dict["eta_km2_per_h"],
        kappa_veh_per_km_lane=param_dict["kappa_veh_per_km_lane"],
        v_free_kmh=param_dict["v_free_kmh"],
        rho_crit_veh_per_km_lane=param_dict["rho_crit_veh_per_km_lane"],
        alpha=param_dict["alpha"],
        lanes=4
    )

    model = METANETModel(params)
    # No masks, so the active mesh is just the base cells: leave min_cell_length at
    # its default rather than passing the cell length and risking a merge.
    sim = Simulation(
        network=generator.network,
        time_resolution=config["time_step"],
        origin_time=config["time_origin"],
        macro_model=model,
    )
    sim.record_rollout = True

    t0 = float(config["time_origin"])
    sim.initialize_from_ground_truth(sim_calibration_metanet.gt_collapsed, time_value=t0, drives_boundaries=False)
    # Shared with the calibration rather than copied: this used to be an inline
    # duplicate, and it drifted into prescribing the discharge through
    # outflow_boundary_map. That map is applied as min(demand, q_gt), a one-sided cap
    # that buried 1038 undischarged vehicles in road 1's 100 m terminal cell over an
    # hour and drove it to 2.1x the measured density.
    sim_calibration_metanet.install_boundary_conditions(
        sim, model, sim_calibration_metanet.gt_collapsed
    )
    for i in range(3599):
        sim.step()
    run_app(sim, rotation_deg=82.8192)

if __name__ == "__main__":
    run_demo_carla()
