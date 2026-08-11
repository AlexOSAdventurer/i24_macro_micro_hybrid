from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Dict, List
import math
import numpy
import json

from simulation import Simulation, I24MicroMask, TriangularFD

@dataclass
class Vehicle:
    length: float # Meters
    width: float # Meters
    s: float # Longitudinal position. Relative to the rear of the microscopic bubble.
    t: float # Lateral position. Decreases as one moves to the right.
    lane: int = 0 # Lane index (same convention as Network lanes)
    s_dt: float = 0 # Vehicle velocity (absolute, not relative to anchor)
    rho: float | None = None # Prescribed boundary density, if this is a boundary vehicle.
                             # Carried explicitly because it cannot be recovered from s_dt
                             # on the free-flow branch (every v_f maps back to rho_c).

def boundary_case_1(bridge: PrescribedSimBridge):
    bridge.anchor_speed = 0.0
    bridge.vehicles = {}
    bridge.vehicles["1"] = Vehicle(
        length=bridge.spawn_length,
        width=bridge.spawn_width,
        s=bridge.middle_s,
        t=bridge.vehicle_t_position,
        lane=bridge.lane_id,
        s_dt=bridge.anchor_speed
    )
    bridge.ego_id = "1"
    bridge.add_boundary_vehicles(density_rear=bridge.fd.rho_j)

def boundary_case_2(bridge: PrescribedSimBridge):
    bridge.anchor_speed = bridge.fd.v_f
    bridge.vehicles = {}
    bridge.vehicles["1"] = Vehicle(
        length=bridge.spawn_length,
        width=bridge.spawn_width,
        s=bridge.middle_s,
        t=bridge.vehicle_t_position,
        lane=bridge.lane_id,
        s_dt=bridge.anchor_speed
    )
    bridge.ego_id = "1"
    bridge.add_boundary_vehicles(density_rear=bridge.fd.rho_c * 0.05)

def boundary_case_3(bridge: PrescribedSimBridge):
    w = bridge.fd.w
    j = bridge.fd.rho_j
    c = bridge.fd.rho_c
    bridge.anchor_speed = w * ((j / (2.0 * c)) - 1.0)
    bridge.vehicles = {}
    bridge.vehicles["1"] = Vehicle(
        length=bridge.spawn_length,
        width=bridge.spawn_width,
        s=bridge.middle_s,
        t=bridge.vehicle_t_position,
        lane=bridge.lane_id,
        s_dt=bridge.anchor_speed
    )
    bridge.ego_id = "1"
    bridge.add_boundary_vehicles(density_rear=bridge.fd.rho_c * 2.0)

def boundary_case_4(bridge: PrescribedSimBridge):
    w = bridge.fd.w
    j = bridge.fd.rho_j
    c = bridge.fd.rho_c
    bridge.anchor_speed = w * ((j / (2.0 * c)) - 1.0)
    bridge.vehicles = {}
    bridge.vehicles["1"] = Vehicle(
        length=bridge.spawn_length,
        width=bridge.spawn_width,
        s=bridge.middle_s,
        t=bridge.vehicle_t_position,
        lane=bridge.lane_id,
        s_dt=bridge.anchor_speed
    )
    bridge.ego_id = "1"
    bridge.add_boundary_vehicles(density_rear=bridge.fd.rho_c * 0.05)

def boundary_case_5(bridge: PrescribedSimBridge):
    bridge.anchor_speed = 0.0 if ((bridge.current_timestamp - bridge.time_origin) <= 10) else bridge.fd.v_f
    bridge.vehicles = {}
    bridge.vehicles["1"] = Vehicle(
        length=bridge.spawn_length,
        width=bridge.spawn_width,
        s=bridge.middle_s,
        t=bridge.vehicle_t_position,
        lane=bridge.lane_id,
        s_dt=bridge.anchor_speed
    )
    bridge.ego_id = "1"
    bridge.add_boundary_vehicles(density_rear=bridge.fd.rho_j)

def boundary_case_6(bridge: PrescribedSimBridge):
    bridge.anchor_speed = bridge.fd.v_f
    bridge.vehicles = {}
    bridge.vehicles["1"] = Vehicle(
        length=bridge.spawn_length,
        width=bridge.spawn_width,
        s=bridge.middle_s,
        t=bridge.vehicle_t_position,
        lane=bridge.lane_id,
        s_dt=bridge.anchor_speed
    )
    bridge.ego_id = "1"
    bridge.add_boundary_vehicles(density_rear=(bridge.fd.rho_j if (bridge.current_timestamp - bridge.time_origin) <= 10.0 else (bridge.fd.rho_c * 0.05)))


def boundary_case_7(bridge: PrescribedSimBridge):
    bridge.anchor_speed = bridge.fd.v_f if ((bridge.current_timestamp - bridge.time_origin) <= 10) else 0.0
    bridge.vehicles = {}
    bridge.vehicles["1"] = Vehicle(
        length=bridge.spawn_length,
        width=bridge.spawn_width,
        s=bridge.middle_s,
        t=bridge.vehicle_t_position,
        lane=bridge.lane_id,
        s_dt=bridge.anchor_speed
    )
    bridge.ego_id = "1"
    bridge.add_boundary_vehicles(density_rear=bridge.fd.rho_c * 0.05)

def boundary_case_8(bridge: PrescribedSimBridge):
    bridge.anchor_speed = 0.0
    bridge.vehicles = {}
    bridge.vehicles["1"] = Vehicle(
        length=bridge.spawn_length,
        width=bridge.spawn_width,
        s=bridge.middle_s,
        t=bridge.vehicle_t_position,
        lane=bridge.lane_id,
        s_dt=bridge.anchor_speed
    )
    bridge.ego_id = "1"
    bridge.add_boundary_vehicles(density_rear=(bridge.fd.rho_c * 0.05 if (bridge.current_timestamp - bridge.time_origin) <= 10.0 else bridge.fd.rho_j))
    

class PrescribedSimBridge:
    spawn_length = 4.0 #6.8725979813165115 + 4.418460070966603
    spawn_width = 2.0
    spawn_threshold = 4.5 # Meters
    vehicle_spawn_limit = 10.0 # 5 cars per tick allowed
    spawn_lookahead = 300.0

    def __init__(
        self,
        sim: Simulation,
        road_id: str,
        initial_middle_s: float,
        max_middle_s: float,
        margin_s: float,
        fd: TriangularFD,
        config_file_name,
        boundary_function=None,
        spawn_density_function=None,
        bridge_callback_name=None
    ) -> None:
        with open(config_file_name, "r") as f:
            self.config = json.load(f)
            self.road_length = self.config["road_data"]["1"]["road_length"]
            self.time_origin = self.config["time_origin"]
        self.sim = sim
        self.current_timestamp = sim.current_time
        self.ego_id = None
        self.road_id = road_id
        self.fd = fd
        #self.spawn_length = (1.0 / fd.rho_j) - 0.1
        self.spawn_length = 4.0
        self.transition_region_size = 1.0 / fd.rho_c
        self.vehicle_t_position = -self.spawn_width / 2.0
        self.lane_id = -1
        self.middle_s = float(initial_middle_s)
        self.max_middle_s = float(max_middle_s)
        self.running = True
        self.margin_s = float(margin_s)
        self.initialized = False
        self.flow_memory_rear = 0.0
        self.flow_memory_front = 0.0
        # Pure cumulative flux, never debited by spawn/despawn. See I24MicroMask.
        self.flow_total_rear = 0.0
        self.flow_total_front = 0.0
        self.vehicles: Dict[str, Vehicle] = {}
        self.anchor_speed = 0.0
        self.masking_cell: I24MicroMask = None
        self.boundary_function = boundary_function
        self.spawn_density_function = spawn_density_function

        self.bridge_callback_name = bridge_callback_name
        sim.register_step_callback(partial(PrescribedSimBridge._step, self), bridge_callback_name)
        sim.register_poststep_callback(partial(PrescribedSimBridge._poststep, self), bridge_callback_name)

    def get_current_visible_window(self):
        return self.vehicles[self.ego_id].s - self.margin_s, self.vehicles[self.ego_id].s + self.margin_s

    def _mask_id(self, lane: int) -> str:
        return f"micro_mask_{self.road_id}_lane{lane}"

    def _step(self, sim_time: float, dt: float) -> None:
        """Called by Simulation.step() before _update_masks().

        Advances the window and rebuilds all four lane masks in-place.
        """
        lane_id = self.lane_id
        if not self.running:
            return
        self.boundary_function(self)
        self.initialized = True

        if self.middle_s >= self.max_middle_s:
            print("bridge memories: ", self.flow_memory_front, self.flow_memory_rear)
            self.destroy()
            return
        new_mask = I24MicroMask(
            mask_id=self._mask_id(lane_id),
            network=self.sim.network,
            road_id=self.road_id,
            lane=lane_id,
            middle_s=self.middle_s,
            margin_s=self.margin_s,
            anchor_speed=self.anchor_speed,
            rear_flux_memory=self.flow_memory_rear,
            front_flux_memory=self.flow_memory_front,
            rear_flux_total=self.flow_total_rear,
            front_flux_total=self.flow_total_front
        )
        self.masking_cell = new_mask
        new_mask.vehicles = self.collate_vehicles()#{vehicle: self.vehicles[vehicle] for vehicle in self.vehicles}
        self.sim.masking_cells[self._mask_id(lane_id)] = new_mask

    def _poststep(self, sim_time: float, dt: float):
        lane_id = self.lane_id
        lane_cell = self.sim.masking_cells[self._mask_id(lane_id)]
        # Do vehicle processing logic here
        self.advance_vehicles_and_boundaries()
        self.current_timestamp += self.sim.time_resolution
        print(self.middle_s, self.anchor_speed, self.current_timestamp, self.sim.time_resolution, len(self.vehicles), float(len(self.vehicles)) / (2 * self.margin_s))
        self.flow_memory_rear = lane_cell.rear_flux_memory
        self.flow_memory_front = lane_cell.front_flux_memory
        self.flow_total_rear = lane_cell.rear_flux_total
        self.flow_total_front = lane_cell.front_flux_total

    def add_boundary_vehicles(self, density_rear: float):
        # Rear Vehicle - carries the case's prescribed rear boundary density.
        self.vehicles["2"] = Vehicle(
            length=self.spawn_length,
            width=self.spawn_width,
            s=(self.middle_s - self.margin_s + self.spawn_length),
            t=self.vehicle_t_position,
            lane=self.lane_id,
            s_dt=self.fd.velocity_from_density(density_rear),
            rho=density_rear
        )
        # Front Vehicle - every b-case pairs its MovingBoundary rear with a SilentBoundary
        # front, i.e. a front that emits exactly the downstream equilibrium flux and so
        # leaves the downstream state untouched. micro_mass_check spells this out: it
        # evaluates both the front demand and the front supply at the downstream density.
        # That requires rho_interior == rho_exterior at the front, which is why the
        # density is carried explicitly rather than recovered from the vehicle's speed.
        #
        # It is taken from the case's prescribed downstream state rather than read back
        # off the adjacent macro cell. Reading the cell makes the front neutrally stable:
        # it re-injects whatever that cell currently holds, so the seam transient created
        # when the mask is first inserted is locked in forever instead of washing out, and
        # the b-cases stop converging. A SilentBoundary in the exact solver does not
        # observe the solution either - it is boundary *data*. Note the front is still not
        # blind to the macro state: the supply term in front_boundary_flux uses the real
        # exterior density, so genuine downstream congestion still throttles the front.
        density_front = self.spawn_density_function(2.0 * self.margin_s, 2.0 * self.margin_s)
        self.vehicles["3"] = Vehicle(
            length=self.spawn_length,
            width=self.spawn_width,
            s=(self.middle_s + self.margin_s - self.spawn_length),
            t=self.vehicle_t_position,
            lane=self.lane_id,
            s_dt=self.fd.velocity_from_density(density_front),
            rho=density_front
        )
        
    # This is meant for the upper level fluid simulator. Thus we have to convert road ids to strings and restructure it to play nice with that code.
    def collate_vehicles(self):
        result = {}
        min_s, max_s = self.get_current_visible_window()
        for vehicle in self.vehicles:
            vehicle_data = self.vehicles[vehicle]
            result[str(vehicle)] = Vehicle(length=vehicle_data.length, width=vehicle_data.width, s=vehicle_data.s - min_s, t=vehicle_data.t, lane=vehicle_data.lane, s_dt=vehicle_data.s_dt, rho=vehicle_data.rho)
        return result

    def update_vehicles(self, vehicles: Dict[str, Vehicle]):
        self.vehicles = vehicles

    def advance_vehicles_and_boundaries(self):
        self.middle_s += (self.anchor_speed * self.sim.time_resolution)
        for vehicle in self.vehicles:
            self.vehicles[vehicle].s += (self.vehicles[vehicle].s_dt * self.sim.time_resolution)

    def get_ahead_lane_velocity_macro(self, default_speed):
        mask_cell = self.sim.active.get_cell_with_mask(self._mask_id(self.lane_id))
        front_cell = self.sim.active.active_cells[mask_cell.outflow_neighbors[0]] if len(mask_cell.outflow_neighbors) > 0 else None
        if front_cell is None:
            return default_speed
        fd = front_cell.fd
        mass = front_cell.mass
        cell_length = front_cell.end_s - front_cell.start_s
        if front_cell.kind == "mask":
            return default_speed # We currently don't bother connecting masks together.
        return fd.velocity_from_density(mass / cell_length)
    
    def get_behind_lane_velocity_macro(self, default_speed):
        mask_cell = self.sim.active.get_cell_with_mask(self._mask_id(self.lane_id))
        behind_cell = self.sim.active.active_cells[mask_cell.inflow_neighbors[0]] if len(mask_cell.inflow_neighbors) > 0 else None
        if behind_cell is None:
            return default_speed
        fd = behind_cell.fd
        mass = behind_cell.mass
        cell_length = behind_cell.end_s - behind_cell.start_s
        if behind_cell.kind == "mask":
            return default_speed # We currently don't bother connecting masks together.
        return fd.velocity_from_density(mass / cell_length)
    
    def spawn_ego_vehicle(self):
        self.ego_id = self.generate_next_vehicle_id()
        self.vehicles[self.ego_id] = Vehicle(self.spawn_length, self.spawn_width, self.middle_s, self.vehicle_t_position, self.lane_id, self.fd.v_f)

    def destroy(self):
        if self.running:
            self.running = False
            del self.sim.masking_cells[self._mask_id(self.lane_id)]
            for road_id, cell_id in self.sim.network.all_cell_keys():
                cell = self.sim.network.get_cell(road_id, cell_id)
                cell.mass += cell.mask_mass
                cell.mask_mass = 0
            self.sim.unregister_step_callback(self.bridge_callback_name)
            self.sim.unregister_poststep_callback(self.bridge_callback_name)
