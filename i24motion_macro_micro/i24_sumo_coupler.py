"""SUMO-backed microscopic coupler for I24MicroSimBridge.

Drop-in alternative to ``I24CarlaCoupler``: same bridge contract, same empirical
seeding, same macro-flux-driven spawn logic -- but the vehicles inside the
bubble are simulated by SUMO's own car-following and lane-change models rather
than by CARLA's traffic manager.

    from i24_sumo_coupler import I24SumoCoupler

    coupler = I24SumoCoupler(
        gt, dt=1.0, lanes=[-1, -2, -3, -4], mapping=config, hero_road="2",
        desired_time=config["time_origin"], desired_s=350.0,
        visible_window=150.0, ghost_window=0.0,
    )
    bridge = I24MicroSimBridge(sim=sim, road_id="2", lanes=[-1, -2, -3, -4],
                               initial_middle_s=350.0, margin_s=150.0,
                               max_middle_s=1300, micro_coupler=coupler,
                               bridge_callback_name="bridge_step")

Requires SUMO, which lives inside the container -- see sumo/README notes and
``sumo/build_i24_network.py`` for how ``i24_corridor.net.xml`` is produced.
"""
from __future__ import annotations

from typing import List, Optional

from simulation import FundamentalDiagram, GroundTruthStore
from i24_micro_coupler_base import I24MicroCouplerBase
from i24_motion_sumo_coupled import (
    DEFAULT_ADDITIONAL_FILE,
    DEFAULT_NET_FILE,
    HeroPolicy,
    I24MotionSumoSimulationCoupled,
)

__all__ = ["I24SumoCoupler", "HeroPolicy"]


class I24SumoCoupler(I24MicroCouplerBase):
    """Bridge coupler whose microscopic engine is SUMO, driven over TraCI."""

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
        net_file: str = DEFAULT_NET_FILE,
        additional_file: Optional[str] = DEFAULT_ADDITIONAL_FILE,
        vehicle_type: str = "car",
        step_length: float = 0.1,
        use_libsumo: bool = False,
        gui: bool = False,
        seed: Optional[int] = None,
        lead_speed_control: bool = True,
        hero_policy: Optional[HeroPolicy] = None,
        hero_min_gap: Optional[float] = None,
        label: str = "i24_bridge",
        extra_args: Optional[List[str]] = None,
        verbose: bool = False,
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
        self.sumo_sim = I24MotionSumoSimulationCoupled(
            coupler=self,
            mapping=self.mapping["road_data"],
            net_file=net_file,
            additional_file=additional_file,
            vehicle_type=vehicle_type,
            step_length=step_length,
            use_libsumo=use_libsumo,
            gui=gui,
            seed=seed,
            lead_speed_control=lead_speed_control,
            hero_policy=hero_policy,
            hero_min_gap=hero_min_gap,
            label=label,
            extra_args=extra_args,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Engine interface
    # ------------------------------------------------------------------

    def initialize_engine(self):
        self.sumo_sim.initialize_simulation()

    def sync_engine_with_visible_state(self):
        self.sumo_sim.update_simulation()

    def advance_engine(self, dt):
        return self.sumo_sim.run_simulation_over(dt)

    def destroy_engine(self):
        self.sumo_sim.destroy_simulation()
