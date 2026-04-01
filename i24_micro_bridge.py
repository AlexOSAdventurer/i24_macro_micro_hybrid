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

from functools import partial
from typing import List

from simulation import Simulation, I24MicroMask


class MicroSimBridge:
    def __init__(
        self,
        sim: Simulation,
        road_id: str,
        lanes: List[int],
        initial_middle_s: float,
        margin_s: float,
    ) -> None:
        self.sim = sim
        self.road_id = road_id
        self.lanes = lanes
        self.middle_s = float(initial_middle_s)
        self.margin_s = float(margin_s)

        # Create one mask per lane and register with the simulation
        for lane in self.lanes:
            mask = I24MicroMask(
                mask_id=self._mask_id(lane),
                network=sim.network,
                road_id=road_id,
                lane=lane,
                middle_s=self.middle_s,
                margin_s=self.margin_s,
            )
            sim.add_masking_cell(mask)

        sim.register_step_callback(partial(MicroSimBridge._step, self))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mask_id(self, lane: int) -> str:
        return f"micro_mask_{self.road_id}_lane{lane}"

    def _step(self, sim_time: float, dt: float) -> None:
        """Called by Simulation.step() before _update_masks().

        Advances the window and rebuilds all four lane masks in-place.
        """
        self.middle_s = self._compute_next_middle_s(sim_time, dt)

        for lane in self.lanes:
            self.sim.masking_cells[self._mask_id(lane)] = I24MicroMask(
                mask_id=self._mask_id(lane),
                network=self.sim.network,
                road_id=self.road_id,
                lane=lane,
                middle_s=self.middle_s,
                margin_s=self.margin_s,
            )

    def _compute_next_middle_s(self, sim_time: float, dt: float) -> float:
        """Return the window centre position at sim_time + dt.

        Placeholder — replace with empirical trajectory lookup.
        """
        return self.middle_s
