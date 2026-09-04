"""CARLA-backed microscopic coupler for I24MicroSimBridge.

Sibling of ``I24SumoCoupler``: identical bridge contract, identical empirical
seeding, identical macro-flux-driven spawn logic -- the vehicles inside the
bubble are simulated by CARLA's traffic manager rather than by SUMO's
car-following and lane-change models.

    from i24_carla_coupler import I24CarlaCoupler

    coupler = I24CarlaCoupler(
        gt, dt=1.0, fd=triangular_fd, lanes=[-1, -2, -3, -4], mapping=config,
        hero_road="2", desired_time=config["time_origin"], desired_s=350.0,
        visible_window=150.0, ghost_window=0.0,
    )
    bridge = I24MicroSimBridge(sim=sim, road_id="2", lanes=[-1, -2, -3, -4],
                               initial_middle_s=350.0, margin_s=150.0,
                               max_middle_s=1300, micro_coupler=coupler,
                               bridge_callback_name="bridge_step")

Requires CARLA and the ``i24motion_to_carla`` package, both of which live inside
the container at /workspaces/.
"""
from __future__ import annotations

from typing import List
import sys

from simulation import FundamentalDiagram, GroundTruthStore
from i24_micro_coupler_base import I24MicroCouplerBase

sys.path.append("/workspaces/")
from i24motion_to_carla import I24MotionCarlaSimulationCoupled

__all__ = ["I24CarlaCoupler"]


class I24CarlaCoupler(I24MicroCouplerBase):
    """Bridge coupler whose microscopic engine is CARLA."""

    def __init__(
        self,
        motion_data: GroundTruthStore,
        dt: float,
        fd: FundamentalDiagram,
        lanes: List[int],
        mapping,
        hero_road: str,
        desired_time: float,
        desired_s: float,
        visible_window: float,
        ghost_window: float,
        host: str = "localhost",
        port: int = 2000,
        record_videos: bool = False,
        bev_video_path: str = "carla_camera_bev_view.mp4",
    ) -> None:
        super().__init__(
            motion_data=motion_data,
            dt=dt,
            fd=fd,
            lanes=lanes,
            mapping=mapping,
            hero_road=hero_road,
            desired_time=desired_time,
            desired_s=desired_s,
            visible_window=visible_window,
            ghost_window=ghost_window,
        )
        # Built only after super().__init__ has seeded the hero: the CARLA side
        # reads getHeroData()["time"] inside its own constructor to fix its time
        # origin, so the hero has to exist before this line runs.
        self.carla_sim = I24MotionCarlaSimulationCoupled(
            host, port, self, self.mapping["road_data"], bev_video_path, record_videos
        )
        self.carla_sim.hero_vehicle_speed = fd.v_f * self.carla_sim.mps_to_kph

    # ------------------------------------------------------------------
    # Engine interface
    # ------------------------------------------------------------------

    def initialize_engine(self):
        self.carla_sim.initialize_simulation()

    def sync_engine_with_visible_state(self):
        self.carla_sim.update_simulation()

    def advance_engine(self, dt):
        return self.carla_sim.run_simulation_over(dt)

    def destroy_engine(self):
        self.carla_sim.destroy_simulation()