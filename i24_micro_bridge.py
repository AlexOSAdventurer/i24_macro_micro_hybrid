"""MicroSimBridge — couples empirical I-24 microscopic trajectory data to the
macroscopic CTM simulation via four per-lane I24MicroMask instances.

Usage
-----
    bridge = MicroSimBridge(
        sim=sim,
        road_id="road_2",
        lanes=[-1, -2, -3, -4],
        initial_middle_s=800.0,
        margin_s=200.0,
    )
    sim.run(duration=3600.0)

The bridge registers itself as a step callback on construction.  Each step it:
  1. Advances the window position (placeholder — implement _compute_next_middle_s).
  2. Rebuilds one I24MicroMask per lane with the new position.
  3. Replaces the masks in sim.masking_cells in-place.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Dict, List

from simulation import Simulation, I24MicroMask

@dataclass
class Vehicle:
    length: float # Meters
    width: float # Meters
    s: float # Longitudinal position. Relative to the rear of the microscopic bubble.
    t: float # Lateral position. Decreases as one moves to the right.
    lane: int = 0 # Lane index (same convention as Network lanes)

class I24MicroSimBridge:
    def __init__(
        self,
        sim: Simulation,
        road_id: str,
        lanes: List[int],
        initial_middle_s: float,
        max_middle_s: float,
        margin_s: float,
        update_micro_callback=None
    ) -> None:
        self.sim = sim
        self.road_id = road_id
        self.lanes = lanes
        self.middle_s = float(initial_middle_s)
        self.max_middle_s = float(max_middle_s)
        self.running = True
        self.margin_s = float(margin_s)
        self.flow_memory_rear = {
            lane: 0.0 for lane in lanes
        }
        self.flow_memory_front = {
            lane: 0.0 for lane in lanes
        }
        self.vehicles: Dict[str, Vehicle] = {}
        self.anchor_speed = 0.0

        # Create one mask per lane and register with the simulation
        for lane in self.lanes:
            mask = I24MicroMask(
                mask_id=self._mask_id(lane),
                network=sim.network,
                road_id=road_id,
                lane=lane,
                middle_s=self.middle_s,
                margin_s=self.margin_s,
                anchor_speed=0.0
            )
            sim.add_masking_cell(mask)

        sim.register_step_callback(partial(I24MicroSimBridge._step, self))
        self.update_micro_callback = update_micro_callback


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mask_id(self, lane: int) -> str:
        return f"micro_mask_{self.road_id}_lane{lane}"

    def _step(self, sim_time: float, dt: float) -> None:
        """Called by Simulation.step() before _update_masks().

        Advances the window and rebuilds all four lane masks in-place.
        """
        if not self.running:
            return
        lane_cells = {
            lane: self.sim.masking_cells[self._mask_id(lane)] for lane in self.lanes
        }
        for lane in lane_cells:
            self.flow_memory_rear[lane] += lane_cells[lane].rear_flow
            self.flow_memory_front[lane] += lane_cells[lane].front_flow

        self.middle_s = self.update_micro_callback(self)
        if self.middle_s >= self.max_middle_s:
            self.running = False
            for lane in self.lanes:
                del self.sim.masking_cells[self._mask_id(lane)]
            return
            
        for lane in self.lanes:
            new_mask = I24MicroMask(
                mask_id=self._mask_id(lane),
                network=self.sim.network,
                road_id=self.road_id,
                lane=lane,
                middle_s=self.middle_s,
                margin_s=self.margin_s,
                anchor_speed=self.anchor_speed
            )
            new_mask.vehicles = {vehicle: self.vehicles[vehicle] for vehicle in self.vehicles if self.vehicles[vehicle].lane == lane}
            self.sim.masking_cells[self._mask_id(lane)] = new_mask

    def update_vehicles(self, vehicles: Dict[str, Vehicle]):
        self.vehicles = vehicles
