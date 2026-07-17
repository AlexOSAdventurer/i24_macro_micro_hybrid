from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Dict, List
import math

from simulation import Simulation, I24MicroMask, TriangularFD

@dataclass
class Vehicle:
    length: float # Meters
    width: float # Meters
    s: float # Longitudinal position. Relative to the rear of the microscopic bubble.
    t: float # Lateral position. Decreases as one moves to the right.
    lane: int = 0 # Lane index (same convention as Network lanes)
    s_dt: float = 0 # Vehicle velocity (absolute, not relative to anchor)

class MicroscopicFTLVehicleModel:

    # Override in children classes
    def __init__(self):
        pass

    def generate_velocity(self, current_s: float, leader_s: float, current_v: float, leader_v: float, dt: float):
        pass
    
    def generate_acceleration(self, current_s: float, leader_s: float, current_v: float, leader_v: float, dt: float):
        pass

class NewellModel(MicroscopicFTLVehicleModel):
    def __init__(self, v_f: float, jam_spacing: float, time_gap: float):
        self.v_f = v_f
        self.jam_spacing = jam_spacing
        self.time_gap = time_gap

    def generate_velocity(self, current_s: float, leader_s: float, current_v: float, leader_v: float, dt: float):
        return min(self.v_f, (leader_s -  current_s) / dt)
    
    def generate_acceleration(self, current_s: float, leader_s: float, current_v: float, leader_v: float, dt: float):
        return (self.generate_velocity(current_s, leader_s, current_v, leader_v, dt) - current_v) / dt

class SimplifiedSimBridge:
    spawn_length = 4.0 #6.8725979813165115 + 4.418460070966603
    spawn_width = 2.0
    spawn_threshold = 4.5 # Meters
    vehicle_spawn_limit = 5 # 5 cars per tick allowed

    def __init__(
        self,
        sim: Simulation,
        road_id: str,
        initial_middle_s: float,
        max_middle_s: float,
        margin_s: float,
        fd: TriangularFD,
        ftl_model: MicroscopicFTLVehicleModel,
        vehicle_t_position: float = 1.8288,
        bridge_callback_name=None
    ) -> None:
        self.sim = sim
        self.current_timestamp = sim.current_time
        self.ego_id = None
        self.road_id = road_id
        self.fd = fd
        self.transition_region_size = 1.0 / fd.rho_c
        self.ftl_model = ftl_model
        self.vehicle_t_position = vehicle_t_position
        self.lane_id = -1
        self.middle_s = float(initial_middle_s)
        self.max_middle_s = float(max_middle_s)
        self.running = True
        self.margin_s = float(margin_s)
        self.initialized = False
        self.flow_memory_rear = 0.0
        self.flow_memory_front = 0.0
        self.vehicles: Dict[str, Vehicle] = {}
        self.anchor_speed = 0.0
        self.masking_cell = None
        self.next_vehicle_id = 1

        self.bridge_callback_name = bridge_callback_name
        sim.register_step_callback(partial(SimplifiedSimBridge._step, self), bridge_callback_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        if self.initialized:
            lane_cell = self.sim.masking_cells[self._mask_id(lane_id)]
            self.flow_memory_rear = lane_cell.rear_flux_memory
            self.flow_memory_front = lane_cell.front_flux_memory
            # Do vehicle processing logic here
            self.advance_and_update_vehicles()
            self.current_timestamp += self.sim.time_resolution
            print(self.middle_s, self.anchor_speed, self.current_timestamp, self.sim.time_resolution)
            #print(self.vehicles)
        else:
            self.spawn_ego_vehicle()
            self.initialized = True

        self.middle_s, self.anchor_speed = self.vehicles[self.ego_id].s, self.vehicles[self.ego_id].s_dt
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
            front_flux_memory=self.flow_memory_front
        )
        new_mask.vehicles = {vehicle: self.vehicles[vehicle] for vehicle in self.vehicles}
        self.sim.masking_cells[self._mask_id(lane_id)] = new_mask

    def update_vehicles(self, vehicles: Dict[str, Vehicle]):
        self.vehicles = vehicles

    def get_ahead_lane_velocity_micro(self, default_speed, min_cell_size=25.0):
        vehicles = self.vehicles
        front_most_vehicle = None
        for vehicle in vehicles:
            if (front_most_vehicle is None) or (vehicles[vehicle].s > front_most_vehicle.s):
                front_most_vehicle = vehicles[vehicle]
        if front_most_vehicle is None:
            return self.get_ahead_lane_velocity_macro(default_speed, min_cell_size)
        return front_most_vehicle.s_dt
    
    def get_behind_lane_velocity_micro(self, default_speed, min_cell_size=25.0):
        vehicles = self.vehicles
        rear_most_vehicle = None
        for vehicle in vehicles:
            if (rear_most_vehicle is None) or (vehicles[vehicle].s < rear_most_vehicle.s):
                rear_most_vehicle = vehicles[vehicle]
        if rear_most_vehicle is None:
            return self.get_behind_lane_velocity_macro(default_speed)
        return rear_most_vehicle.s_dt

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
    
    def get_ahead_lane_density(self, default_density):
        mask_cell = self.sim.active.get_cell_with_mask(self._mask_id(self.lane_id))
        front_cell = self.sim.active.active_cells[mask_cell.outflow_neighbors[0]] if len(mask_cell.outflow_neighbors) > 0 else None
        if front_cell is None:
            return default_density
        mass = front_cell.mass
        cell_length = front_cell.end_s - front_cell.start_s
        if front_cell.kind == "mask":
            return default_density # We currently don't bother connecting masks together.
        return mass / cell_length
    
    def get_behind_lane_density(self, default_density):
        mask_cell = self.sim.active.get_cell_with_mask(self._mask_id(self.lane_id))
        behind_cell = self.sim.active.active_cells[mask_cell.inflow_neighbors[0]] if len(mask_cell.inflow_neighbors) > 0 else None
        if behind_cell is None:
            return default_density
        mass = behind_cell.mass
        cell_length = behind_cell.end_s - behind_cell.start_s
        if behind_cell.kind == "mask":
            return default_density # We currently don't bother connecting masks together.
        return mass / cell_length
    
    def spawn_ego_vehicle(self):
        self.ego_id = self.generate_next_vehicle_id()
        self.vehicles[self.ego_id] = Vehicle(self.spawn_length, self.spawn_width, self.middle_s, self.vehicle_t_position, self.lane_id, self.fd.v_f)

    def get_immediate_leader_vehicle(self, vehicle: Vehicle):
        current_leader_key = None
        for vehicle_key in self.vehicles:
            new_vehicle = self.vehicles[vehicle_key]
            if (current_leader_key is None) or (self.vehicles[current_leader_key].s > new_vehicle.s):
                if (new_vehicle.s > vehicle.s):
                    current_leader_key = vehicle_key
        if (current_leader_key is not None) and (self.vehicles[current_leader_key].s >= (self.middle_s + self.margin_s - self.transition_region_size)):
            return current_leader_key, self.vehicles[current_leader_key]
        return None, None

    def get_front_leader_vehicle(self):
        current_leader_key = None
        for vehicle_key in self.vehicles:
            new_vehicle = self.vehicles[vehicle_key]
            if (current_leader_key is None) or (self.vehicles[current_leader_key].s < new_vehicle.s):
                current_leader_key = vehicle_key
        if (current_leader_key is not None) and (self.vehicles[current_leader_key].s >= (self.middle_s + self.margin_s - self.transition_region_size)):
            return current_leader_key, self.vehicles[current_leader_key]
        return None, None
    
    def generate_next_vehicle_id(self):
        new_id = str(self.next_vehicle_id)
        self.next_vehicle_id += 1
        return new_id
    
    # behind_or_in_front is either "behind" or "front"
    def generate_vehicle_state_from_spawn(self, s_min, s_max, behind_or_in_front="behind"):
        if (behind_or_in_front != "behind") and (behind_or_in_front != "front"):
            return None # Force failure upstream. Hacky but whatevs. We can improve all of this later.
        estimated_velocity = self.get_behind_lane_velocity_micro(0.0) if (behind_or_in_front == "behind") else self.get_ahead_lane_velocity_micro(0.0)
        estimated_width = 3.5 # We're hardcoding this for now. We'll need to add lane/road cross-referencing lookup later
        new_id = self.generate_next_vehicle_id()
        self.vehicles[new_id] = Vehicle(
            length = float(s_max - s_min),
            width = estimated_width,
            s = float(s_min),
            t = -estimated_width / 2.0,
            lane = self.lane_id,
            s_dt = estimated_velocity
        )
        return new_id

    # behind_or_in_front is either "behind" or "front"
    def _create_vehicle_spawns_in_srange(self, vehicle_count, s_min, s_max, behind_or_in_front, toprint=False):
        density = self.get_behind_lane_density(0.001) if (behind_or_in_front == "behind") else self.get_ahead_lane_density(0.001)
        spawn_lengths = (1.0 / density)
        spawn_distance = spawn_lengths - self.spawn_length
        vehicle_count = min(math.floor((s_max - s_min) / spawn_lengths), vehicle_count)
        if (behind_or_in_front == "behind"):
            s_max = s_min + (spawn_lengths * vehicle_count)
        else:
            s_min = s_max - (spawn_lengths * vehicle_count)
        if toprint:
            print("density: ", density)
            print("s_max, s_min: ", s_max, s_min)
            print("spawn_lengths: ", spawn_lengths)
            print("vehicle count: ", vehicle_count)
        for i in range(vehicle_count):
#        for start_position in numpy.arange(s_min, s_max, spawn_lengths):
            start_position = s_min + (i * spawn_lengths)
            start_position_calculated = None
            end_position_calculated = None
            if (behind_or_in_front == "behind"):
                start_position_calculated = start_position
                end_position_calculated = start_position + spawn_lengths - spawn_distance
            elif (behind_or_in_front == "front"):
                start_position_calculated = start_position + spawn_distance
                end_position_calculated = start_position + spawn_lengths

            new_vehicle_id = self.generate_vehicle_state_from_spawn(start_position_calculated, end_position_calculated, behind_or_in_front)
            #print(f"Spawned Vehicle Data {new_vehicle_id}:", self.vehicles[new_vehicle_id])
            #print("New states ", self.vehicles)
        return vehicle_count

    def _spawn_vehicles_in_rear(self, s_availability):
        rear_flux_memory = self.flow_memory_rear
        visible_window = self.get_current_visible_window()
        if (rear_flux_memory > 0.0):
            # Perform a poisson draw to determine the number of vehicles to create
            vehicle_count = min(math.floor(rear_flux_memory), self.vehicle_spawn_limit) # min(numpy.random.poisson(rear_flux_memory), self.vehicle_spawn_limit)
            if (vehicle_count > 0):
                #print("window: ", visible_window)
                #print("availability: ", s_availability)
                #print("spawn_length info: ")
                vehicle_count = self._create_vehicle_spawns_in_srange(vehicle_count, visible_window[0], visible_window[0] + s_availability, "behind", False)
                self.flow_memory_rear -= vehicle_count

    def _spawn_vehicles_in_front(self, s_availability):
        front_flux_memory = self.flow_memory_front
        visible_window = self.get_current_visible_window()
        if (front_flux_memory < 0.0):
            # Perform a poisson draw to determine the number of vehicles to create
            vehicle_count = min(math.floor(-front_flux_memory), self.vehicle_spawn_limit) # min(numpy.random.poisson(-front_flux_memory), self.vehicle_spawn_limit)
            if (vehicle_count > 0):
                vehicle_count = self._create_vehicle_spawns_in_srange(vehicle_count, visible_window[1] - s_availability, visible_window[1], "front", True)
                self.flow_memory_front += vehicle_count

    def inject_vehicles_from_macro(self):
        visible_window = self.get_current_visible_window()
        # Spawn Rear Boundary Vehicles
        # Get rearmost s position
        rear_s = None
        rear_velocity = None
        for vehicle_id in self.vehicles:
            vehicle_data = self.vehicles[vehicle_id]
            if (rear_s is None) or (rear_s > vehicle_data.s):
                rear_s = vehicle_data.s
                rear_velocity = vehicle_data.s_dt
        if rear_s is None:
            s_availability = (visible_window[1] - visible_window[0])
        else:
            desired_macro_velocity = self.get_behind_lane_velocity_macro(0.0)
            closing_rate = desired_macro_velocity - rear_velocity
            #rear_s = min(rear_s, rear_s - (closing_rate * self.spawn_ttc))
            s_availability = rear_s - visible_window[0]
        s_availability = min(s_availability, self.transition_region_size)
        # Mandate a certain distance threshold of the rearmost vehicle for spawning in new stuff
        if (s_availability > self.spawn_threshold):
            self._spawn_vehicles_in_rear(s_availability)
        # Spawn Front Boundary Vehicles
        front_s = None
        front_velocity = None
        for vehicle_id in self.vehicles:
            vehicle_data = self.vehicles[vehicle_id]
            if (front_s is None) or (front_s < vehicle_data.s):
                front_s = vehicle_data.s + vehicle_data.length
                front_velocity = vehicle_data.s_dt
        if front_s is None:
            s_availability = (visible_window[1] - visible_window[0])
        else:
            desired_macro_velocity = self.get_ahead_lane_velocity_macro(0.0)
            closing_rate = front_velocity - desired_macro_velocity
            #front_s = max(front_s, front_s + (closing_rate * self.spawn_ttc))
            s_availability = visible_window[1] - front_s
        s_availability = min(s_availability, self.transition_region_size)
        # Mandate a certain distance threshold of the frontmost vehicle for spawning in new stuff
        if (s_availability > self.spawn_threshold):
            self._spawn_vehicles_in_front(s_availability)

    def remove_vehicles_from_macro(self):
        vehicles_to_remove = []
        visible_window = self.get_current_visible_window()
        for vehicle in self.vehicles:
            vehicle_data = self.vehicles[vehicle]
            if (vehicle_data.s > (2 * self.margin_s)):
                vehicles_to_remove.append(vehicle)
                self.flow_memory_front -= 1.0
            elif (vehicle_data.s < 0):
                vehicles_to_remove.append(vehicle)
                self.flow_memory_rear += 1.0
        for vehicle in vehicles_to_remove:
            self.vehicles.pop(vehicle, None)

    def spawn_and_despawn_vehicles(self):
        self.inject_vehicles_from_macro()
        self.remove_vehicles_from_macro()

    def update_vehicle_velocities(self):
        front_leader_vehicle_key, _ = self.get_front_leader_vehicle()
        for vehicle in self.vehicles:
            immediate_leader_vehicle_key, immediate_leader_vehicle_object = self.get_immediate_leader_vehicle(self.vehicles[vehicle])
            if (front_leader_vehicle_key is not None) and (front_leader_vehicle_key == vehicle):
                self.vehicles[vehicle].s_dt = self.get_ahead_lane_velocity_macro(self.fd.v_f)
            else:
                if (immediate_leader_vehicle_key is None):
                    leader_s = math.inf
                    leader_v = math.inf
                else:
                    leader_s = immediate_leader_vehicle_object.s
                    leader_v = immediate_leader_vehicle_object.s_dt
                self.vehicles[vehicle].s_dt += self.ftl_model.generate_acceleration(self.vehicles[vehicle].s, leader_s, self.vehicles[vehicle].s_dt, leader_v, self.sim.time_resolution)

    def move_vehicles(self):
        for vehicle in self.vehicles:
            self.vehicles[vehicle].s += self.vehicles[vehicle].s_dt

    def advance_and_update_vehicles(self):
        self.spawn_and_despawn_vehicles()
        self.update_vehicle_velocities()
        self.move_vehicles()

    def destroy(self):
        if self.running:
            self.running = False
            del self.sim.masking_cells[self._mask_id(self.lane_id)]
            for road_id, cell_id in self.sim.network.all_cell_keys():
                cell = self.sim.network.get_cell(road_id, cell_id)
                cell.mass += cell.mask_mass
                cell.mask_mass = 0
            self.sim.unregister_step_callback(self.bridge_callback_name)
