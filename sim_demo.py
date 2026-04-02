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
        return False, 1.0

    @app.callback(
        Output("step-slider", "value"),
        Input("play-interval", "n_intervals"),
        State("step-slider", "value"),
        State("speed", "value"),
    )
    def advance_slider(_, step_idx: int, speed: int) -> int:
        return (step_idx + speed) % N

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    with open("i24_motion_to_dataset.json", "r") as f:
        config = json.load(f)

    sim = Simulation.from_json(
        json_path=os.path.join(config["storage_locations"]["simulation_dataset"], "network.json"),
        time_resolution=config["time_step"],
        origin_time=config["time_origin"]+60.0
    )

    gt = GroundTruthStore.from_parquet(os.path.join(config["storage_locations"]["simulation_dataset"], "micro.parquet"), os.path.join(config["storage_locations"]["simulation_dataset"], "macro.parquet"))
    sim.initialize_from_ground_truth(gt, time_value=config["time_origin"]+60.0)
    replayer = I24TrajectoryReplayer(gt, dt=1.0, lanes=[-1, -2, -3, -4])
    bridge = I24MicroSimBridge(
        sim=sim,
        road_id="2",
        lanes=[-1, -2, -3, -4],
        initial_middle_s=600.0,
        margin_s=50.0,
        update_micro_callback=replayer.step,
    )
    sim.run(duration=60.0)
    run_app(sim, rotation_deg=90.0)
