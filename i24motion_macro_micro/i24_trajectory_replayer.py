"""I24TrajectoryReplayer — loads empirical I-24 vehicle trajectories and drives
the microscopic window inside I24MicroSimBridge.

This is a pure data replayer: it has no simulation logic of its own.
It queries the I24MotionData DuckDB store for vehicles present in the
current window, updates the bridge's vehicle dict, and returns the next
window centre position derived from the observed vehicle speeds.

Usage
-----
    from i24_motion_data import I24MotionData
    from i24_trajectory_replayer import I24TrajectoryReplayer
    from i24_micro_bridge import I24MicroSimBridge

    motion_data = I24MotionData(
        road_id=2,
        timestamp_min=origin_time,
        timestamp_max=origin_time + 3600.0,
        s_min=0.0,
        s_max=1600.0,
    )
    replayer = I24TrajectoryReplayer(motion_data, dt=1.0)
    bridge = I24MicroSimBridge(
        sim=sim,
        road_id="road_2",
        lanes=[-1, -2, -3, -4],
        initial_middle_s=800.0,
        margin_s=200.0,
        update_micro_callback=replayer.step,
    )
    sim.run(duration=3600.0)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from simulation import GroundTruthStore
from i24_micro_bridge import Vehicle

if TYPE_CHECKING:
    from i24_micro_bridge import I24MicroSimBridge


class I24TrajectoryReplayer:
    def __init__(self, motion_data: GroundTruthStore, dt: float, lanes: List[int]) -> None:
        self.motion_data = motion_data
        self.lanes = lanes
        self.dt = float(dt)
        self.bridge = None # Set by the bridge itself
        self.current_anchor_speed = None
        self.initialized = False

    def get_lane_dfs(self, timestamp_min, timestamp_max, s_min, s_max):
        df = self.motion_data.micro_df
        window = df[
            (df["time"] >= timestamp_min) &
            (df["time"] <= timestamp_max) &
            (df["s"] >= s_min) &
            (df["s"] <= s_max)
        ]
        return {lane: window[window["lane_id"] == lane] for lane in self.lanes}

    # ------------------------------------------------------------------
    # Bridge callback — called each step with the bridge as argument
    # ------------------------------------------------------------------

    def step(self) -> float:
        """Query empirical vehicles for this timestep, update the bridge,
        and return the new window centre position.

        The window centre tracks the vehicle currently closest to middle_s.
        That vehicle's speed advances the window; the next step then finds
        the closest vehicle to the new position and repeats.

        This method is passed directly as bridge.update_micro_callback.
        """
        sim_time = self.bridge.sim.current_time
        middle_s = self.bridge.middle_s
        margin_s = self.bridge.margin_s
        s_min = middle_s - margin_s
        s_max = middle_s + margin_s

        lane_dfs = self.get_lane_dfs(sim_time, sim_time + self.dt, s_min, s_max)
        """lane_dfs = self.motion_data.queryEdieBoxSubset(
            timestamp_min=sim_time,
            timestamp_max=sim_time + self.dt,
            s_min=s_min,
            s_max=s_max,
        )"""

        vehicles: Dict[str, Vehicle] = {}
        anchor_speed: float | None = None
        anchor_dist = float("inf")

        for lane, df in lane_dfs.items():
            if df.empty:
                continue

            for vid, group in df.groupby("id"):
                if len(group) == 1:
                    continue
                group = group.sort_values("time")
                row0 = group.iloc[0]
                row1 = group.iloc[1]
                s_abs = float(row0["s"])
                s_next_abs = float(row1["s"])
                velocity_estimated = (s_next_abs - s_abs) / (row1["time"] - row0["time"])

                vehicles[str(vid)] = Vehicle(
                    length=float(row0["length"]),
                    width=float(row0["width"]),
                    s=s_abs - s_min,   # relative to rear face of window
                    t=float(row0["t"]),
                    lane=lane,
                    s_dt=velocity_estimated
                )

                # Track the vehicle closest to middle_s as the window anchor
                dist = abs(s_abs - middle_s)
                if dist < anchor_dist and len(group) >= 2 and (s_abs > middle_s):
                    ds = float(group["s"].iloc[-1]) - float(group["s"].iloc[0])
                    dt_obs = float(group["time"].iloc[-1]) - float(group["time"].iloc[0])
                    if dt_obs > 1e-9:
                        anchor_dist = dist
                        anchor_speed = ds / dt_obs

        self.bridge.update_vehicles(vehicles)
        self.bridge.anchor_speed = anchor_speed
        result = self._compute_next_middle_s(middle_s, self.current_anchor_speed if self.current_anchor_speed is not None else anchor_speed)
        self.current_anchor_speed = anchor_speed
        if self.initialized:
            return result
        else:
            self.initialized = True
            return middle_s

    # ------------------------------------------------------------------

    def _compute_next_middle_s(
        self, middle_s: float, anchor_speed: float | None
    ) -> float:
        """Advance the window centre by the anchor vehicle's speed × dt.

        Returns middle_s unchanged if no anchor vehicle was found.
        """
        if anchor_speed is None:
            return middle_s
        return middle_s + anchor_speed * self.dt

    def destroy(self):
        pass