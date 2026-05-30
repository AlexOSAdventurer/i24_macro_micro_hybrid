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
    s_dt: float = 0 # Vehicle velocity (absolute, not relative to anchor)

class I24MicroSimBridge:
    def __init__(
        self,
        sim: Simulation,
        road_id: str,
        lanes: List[int],
        initial_middle_s: float,
        max_middle_s: float,
        margin_s: float,
        micro_coupler=None,
        bridge_callback_name=None
    ) -> None:
        self.sim = sim
        self.road_id = road_id
        self.lanes = lanes
        self.middle_s = float(initial_middle_s)
        self.max_middle_s = float(max_middle_s)
        self.running = True
        self.margin_s = float(margin_s)
        self.initialized = False
        self.flow_memory_rear = {
            lane: 0.0 for lane in lanes
        }
        self.flow_memory_front = {
            lane: 0.0 for lane in lanes
        }
        self.vehicles: Dict[str, Vehicle] = {}
        self.anchor_speed = 0.0

        self.bridge_callback_name = bridge_callback_name
        sim.register_step_callback(partial(I24MicroSimBridge._step, self), bridge_callback_name)
        self.micro_coupler = micro_coupler
        self.micro_coupler.bridge = self

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
        if self.initialized:
            lane_cells = {
                lane: self.sim.masking_cells[self._mask_id(lane)] for lane in self.lanes
            }
            for lane in lane_cells:
                self.flow_memory_rear[lane] = lane_cells[lane].rear_flux_memory
                self.flow_memory_front[lane] = lane_cells[lane].front_flux_memory
        else:
            self.initialized = True

        self.middle_s = self.micro_coupler.step()
        if self.middle_s >= self.max_middle_s:
            self.destroy()
            return
                    
        for lane in self.lanes:
            new_mask = I24MicroMask(
                mask_id=self._mask_id(lane),
                network=self.sim.network,
                road_id=self.road_id,
                lane=lane,
                middle_s=self.middle_s,
                margin_s=self.margin_s,
                anchor_speed=self.anchor_speed,
                rear_flux_memory=self.flow_memory_rear[lane],
                front_flux_memory=self.flow_memory_front[lane]
            )
            new_mask.vehicles = {vehicle: self.vehicles[vehicle] for vehicle in self.vehicles if self.vehicles[vehicle].lane == lane}
            self.sim.masking_cells[self._mask_id(lane)] = new_mask

    def update_vehicles(self, vehicles: Dict[str, Vehicle]):
        self.vehicles = vehicles

    def destroy(self):
        if self.running:
            self.running = False
            for lane in self.lanes:
                del self.sim.masking_cells[self._mask_id(lane)]
            for road_id, cell_id in self.sim.network.all_cell_keys():
                cell = self.sim.network.get_cell(road_id, cell_id)
                cell.mass += cell.mask_mass
                cell.mask_mass = 0
            self.sim.unregister_step_callback(self.bridge_callback_name)
            self.micro_coupler.destroy()
