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

The bridge registers a step and a poststep callback on construction, splitting
each macroscopic step around the fluid solve.  Before it:
  1. Asks the coupler for the window position (micro_coupler.step()).
  2. Rebuilds one I24MicroMask per lane with that position.
  3. Replaces the masks in sim.masking_cells in-place.
After it:
  4. Reads each mask's flux memories and totals back off the fluid step.
  5. Advances the micro engine (micro_coupler.poststep()).
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
        # Pure cumulative flux, never debited by spawn/despawn. See I24MicroMask.
        self.flow_total_rear = {
            lane: 0.0 for lane in lanes
        }
        self.flow_total_front = {
            lane: 0.0 for lane in lanes
        }
        self.vehicles: Dict[str, Vehicle] = {}
        self.anchor_speed = 0.0

        self.bridge_callback_name = bridge_callback_name
        sim.register_step_callback(partial(I24MicroSimBridge._step, self), bridge_callback_name)
        sim.register_poststep_callback(partial(I24MicroSimBridge._poststep, self), bridge_callback_name)
        self.micro_coupler = micro_coupler
        self.micro_coupler.bridge = self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mask_id(self, lane: int) -> str:
        return f"micro_mask_{self.road_id}_lane{lane}"

    def _step(self, sim_time: float, dt: float) -> None:
        """Called by Simulation.step() before _update_masks().

        The pre-fluid half: the coupler settles its bubble and hands over a
        vehicle set, then all four lane masks are rebuilt around it.  The micro
        engine is advanced afterwards, in _poststep, so everything the mask is
        built from comes from a single hero snapshot.
        """
        if not self.running:
            return

        self.middle_s = self.micro_coupler.step()
        self.initialized = True
        if self.middle_s >= self.max_middle_s:
            print("bridge memories: ", self.flow_memory_front, self.flow_memory_rear)
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
                front_flux_memory=self.flow_memory_front[lane],
                rear_flux_total=self.flow_total_rear[lane],
                front_flux_total=self.flow_total_front[lane]
            )
            new_mask.vehicles = {vehicle: self.vehicles[vehicle] for vehicle in self.vehicles if self.vehicles[vehicle].lane == lane}
            self.sim.masking_cells[self._mask_id(lane)] = new_mask

    def _poststep(self, sim_time: float, dt: float) -> None:
        """Called by Simulation.step() after the fluid step has run.

        Reads back what the fluid step did to each mask, then advances the micro
        engine.  That order is load-bearing: the coupler's engine can lose
        vehicles during its advance and hands their mass straight back via
        credit_lost_vehicle, so the read has to happen first or those credits are
        overwritten by it and the mass is destroyed.
        """
        if not self.running:
            return
        for lane in self.lanes:
            lane_cell = self.sim.masking_cells[self._mask_id(lane)]
            self.flow_memory_rear[lane] = lane_cell.rear_flux_memory
            self.flow_memory_front[lane] = lane_cell.front_flux_memory
            self.flow_total_rear[lane] = lane_cell.rear_flux_total
            self.flow_total_front[lane] = lane_cell.front_flux_total

        # I24CarlaCoupler and I24TrajectoryReplayer are standalone couplers that
        # still run the whole cycle inside step(); they have no poststep half.
        poststep = getattr(self.micro_coupler, "poststep", None)
        if poststep is not None:
            poststep()

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
                # `mass` now covers the whole cell again, so the remap state that
                # said otherwise has to go with it. macro_length is the *unmasked*
                # length the mask left behind; leaving it set makes the next
                # base_to_active compute mass / sliver instead of mass / length,
                # spiking a partially-masked cell by length/sliver (and creating
                # that mass for real, since active_to_base writes it back).
                cell.macro_length = None
            self.sim.unregister_step_callback(self.bridge_callback_name)
            # Both halves have to go: _step can destroy the bridge mid-step, and a
            # surviving _poststep would then index masking_cells for a mask that
            # was just deleted.
            self.sim.unregister_poststep_callback(self.bridge_callback_name)
            self.micro_coupler.destroy()
