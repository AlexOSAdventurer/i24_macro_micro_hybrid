from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Dict, List
import math
import numpy

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

    # Number of integration sub-steps the bridge should take per macro time step.
    # First-order (velocity-based) models are stable at any dt and use 1; acceleration
    # models (e.g. IDM) need a finer step to stay collision-free, so they raise this.
    substeps = 1

    # Override in children classes
    def __init__(self):
        pass

    def generate_velocity(self, current_s: float, leader_s: float, current_v: float, leader_v: float, dt: float):
        pass

    def generate_acceleration(self, current_s: float, leader_s: float, current_v: float, leader_v: float, dt: float):
        pass

    def step(self, current_s: float, leader_s: float, current_v: float, leader_v: float, dt: float):
        """Advance one vehicle by a single integration step of size dt.

        Returns (new_velocity, distance_advanced). Keeping the integration inside the
        model lets each model choose its own (safe) discretization; the bridge only
        decides how many sub-steps to take via `substeps`.
        """
        raise NotImplementedError

class NewellModel(MicroscopicFTLVehicleModel):
    substeps = 50
    # The textbook Newell model that 1-for-1 matches LWR with a Triangular FD.
    # Parameters:
    # v_f: Free flow velocity in meters per second. Same as the LWR/Triangular v_f.
    # jam_spacing: max density in vehicles per meter. Same as LWR/Triangular rho_j.
    # time_gap: Time headway in seconds per vehicle. Derive it from LWR/Triangular via 1/(w * rho_j). 
    def __init__(self, v_f: float, jam_spacing: float, time_gap: float):
        self.v_f = v_f
        self.jam_spacing = jam_spacing
        self.time_gap = time_gap

    def generate_velocity(self, current_s: float, leader_s: float, current_v: float, leader_v: float, dt: float):
        headway = leader_s - current_s if leader_s is not None else None
        v_leader_constrained = 0.0
        if (headway is None):
            v_leader_constrained = self.v_f
        elif (headway > self.jam_spacing):
            v_leader_constrained = (headway - self.jam_spacing) / self.time_gap
        target_v = min(self.v_f, v_leader_constrained)

        return max(0.0, target_v)

    def generate_acceleration(self, current_s: float, leader_s: float, current_v: float, leader_v: float, dt: float):
        return (self.generate_velocity(current_s, leader_s, current_v, leader_v, dt) - current_v)

    def step(self, current_s: float, leader_s: float, current_v: float, leader_v: float, dt: float):
        # Newell sets velocity directly from the gap (safe at any dt), then advances at it.
        new_v = self.generate_velocity(current_s, leader_s, current_v, leader_v, dt)
        return new_v, new_v * dt

class IDMModel(MicroscopicFTLVehicleModel):
    # IDM is only provably collision-free in continuous time. Integrated with a coarse
    # macro dt (e.g. 1 s) it overshoots and cars overlap, so sub-step the ODE.
    substeps = 10

    def __init__(self, v_f: float, vehicle_length: float, still_gap: float, time_headway: float, acceleration_exponent: float, max_accel: float, max_decel: float):
        self.v_f = v_f
        self.vehicle_length = vehicle_length
        self.still_gap = still_gap
        self.time_headway = time_headway
        self.acceleration_exponent = acceleration_exponent
        self.max_accel = max_accel
        self.max_decel = max_decel

    def generate_velocity(self, current_s: float, leader_s: float, current_v: float, leader_v: float, dt: float):
        accel = self.generate_acceleration(current_s, leader_s, current_v, leader_v, dt)
        new_v = current_v + (accel * dt)

        return max(0.0, new_v)

    def generate_acceleration(self, current_s: float, leader_s: float, current_v: float, leader_v: float, dt: float):
        if (leader_v is None):
            leader_s = math.inf
            leader_v = self.v_f
        s = leader_s - current_s - self.vehicle_length
        dv = current_v - leader_v

        if (s <= self.still_gap):
            return -self.max_decel
        
        s_star = self.still_gap + (current_v * self.time_headway) + ((current_v * dv) / (2.0 * (self.max_accel * self.max_decel) ** 0.5))
        accel = self.max_accel * (1.0 - ((current_v / self.v_f) ** self.acceleration_exponent) - ((s_star / s) ** 2.0))

        bounded_accel = max(-self.max_decel, min(accel, self.max_accel))
        return bounded_accel

    def step(self, current_s: float, leader_s: float, current_v: float, leader_v: float, dt: float):
        # Ballistic (kinematic) update: advance position with the average velocity over
        # the step rather than the end-of-step velocity. This halves the discretization
        # error versus plain Euler and, together with sub-stepping, keeps IDM collision-free.
        accel = self.generate_acceleration(current_s, leader_s, current_v, leader_v, dt)
        new_v = current_v + (accel * dt)
        if (new_v < 0.0):
            # The vehicle would reverse within the step; instead stop it at the point where
            # v reaches 0 (distance = v^2 / 2|a|) so it never travels backwards.
            distance = -(current_v ** 2) / (2.0 * accel) if (accel < 0.0) else 0.0
            new_v = 0.0
        else:
            distance = (current_v * dt) + (0.5 * accel * (dt ** 2))
        new_v = min(new_v, self.v_f)
        return new_v, distance

class IIDMModel(IDMModel):
    # Improved IDM (Treiber & Kesting, "Traffic Flow Dynamics", 2013). Same parameters and
    # same ballistic sub-stepped integration as IDMModel -- only the acceleration law is
    # replaced. The IIDM (a) clamps the dynamic part of the desired gap to be non-negative
    # and (b) reformulates the free/interaction terms so acceleration never exceeds max_accel
    # and the model stays collision-free in stop-and-go, the regime where plain IDM overlaps.
    substeps = 10

    def generate_acceleration(self, current_s: float, leader_s: float, current_v: float, leader_v: float, dt: float):
        if (leader_v is None):
            leader_s = math.inf
            leader_v = self.v_f
        s = leader_s - current_s - self.vehicle_length
        dv = current_v - leader_v

        # Guard the gap so the ratio z = s_star / s is well-defined even at/through contact;
        # a vanishing gap drives z high, which the interaction term turns into hard braking.
        s = max(s, 1e-6)

        # Desired dynamic gap, with the velocity-difference term clamped to >= 0 (IIDM form).
        dynamic_gap = (current_v * self.time_headway) + ((current_v * dv) / (2.0 * (self.max_accel * self.max_decel) ** 0.5))
        s_star = self.still_gap + max(0.0, dynamic_gap)
        z = s_star / s

        # Free-road acceleration: the term that survives when the leader is far away.
        if (current_v <= self.v_f):
            accel_free = self.max_accel * (1.0 - ((current_v / self.v_f) ** self.acceleration_exponent))
        else:
            accel_free = -self.max_decel * (1.0 - ((self.v_f / current_v) ** (self.max_accel * self.acceleration_exponent / self.max_decel)))

        # Combine the free and interaction terms with the IIDM's piecewise blend.
        if (current_v <= self.v_f):
            if (z >= 1.0):
                accel = self.max_accel * (1.0 - (z ** 2.0))
            elif (accel_free > 1e-8):
                accel = accel_free * (1.0 - (z ** (2.0 * self.max_accel / accel_free)))
            else:
                accel = accel_free
        else:
            if (z >= 1.0):
                accel = accel_free + (self.max_accel * (1.0 - (z ** 2.0)))
            else:
                accel = accel_free

        bounded_accel = max(-self.max_decel, min(accel, self.max_accel))
        return bounded_accel
    
class MicroscopicARZVehicleModel(MicroscopicFTLVehicleModel):
    # Acceleration models need a finer time step to remain numerically stable 
    # and prevent vehicle overlapping during rapid deceleration.
    substeps = 10

    def __init__(self, v_max: float, rho_max: float, gamma: float, tau: float, vehicle_length: float = 5.0):
        """
        Initializes the microscopic ARZ (Aw-Rascle-Zhang) follow-the-leader model.

        Parameters:
        -----------
        v_max : float
            Maximum free-flow velocity (m/s).
        rho_max : float
            Maximum traffic density (vehicles/meter). 1/rho_max represents the jam spacing.
        gamma : float
            Traffic pressure exponent (dimensionless parameter tuning driver anticipation).
        tau : float
            Relaxation time constant (seconds). Lower values mean faster adaptation to equilibrium.
        vehicle_length : float
            Physical length of the vehicle (meters) used to calculate clear headway spacing.
        """
        super().__init__()
        self.v_max = v_max
        self.rho_max = rho_max
        self.gamma = gamma
        self.tau = tau
        self.vehicle_length = vehicle_length
        
        # Jam spacing (minimum distance between front bumpers of consecutive cars)
        self.s_min = 1.0 / self.rho_max

    def _calculate_local_density(self, current_s: float, leader_s: float) -> float:
        """Calculates the micro-density experienced by the follower."""
        # Headway distance between front bumpers
        headway = leader_s - current_s
        
        # Guard against zero or negative spacing to prevent division by zero
        if headway <= self.s_min:
            return self.rho_max
            
        return 1.0 / headway

    def _equilibrium_velocity(self, rho: float) -> float:
        """Standard Greenshields-type or density-dependent equilibrium velocity function."""
        if rho >= self.rho_max:
            return 0.0
        # Example using a standard power-law density decay matching ARZ pressure forms
        return self.v_max * (1.0 - (rho / self.rho_max) ** self.gamma)

    def generate_velocity(self, current_s: float, leader_s: float, current_v: float, leader_v: float, dt: float) -> float:
        """Calculates the expected velocity after a small time step dt."""
        accel = self.generate_acceleration(current_s, leader_s, current_v, leader_v, dt)
        new_v = current_v + accel * dt
        return max(0.0, min(self.v_max, new_v))  # Bound velocity between 0 and v_max

    def generate_acceleration(self, current_s: float, leader_s: float, current_v: float, leader_v: float, dt: float) -> float:
        """
        Computes acceleration based on the microscopic ARZ formulation:
        dv/dt = (V(rho) - v) / tau  +  (C * (leader_v - current_v)) / (leader_s - current_s)^2
        """
        if leader_s is None:
            leader_s = math.inf
            leader_v = self.v_max

        rho = self._calculate_local_density(current_s, leader_s)
        v_eq = self._equilibrium_velocity(rho)
        
        # 1. Relaxation Term: Tendency to adapt to the equilibrium velocity
        relaxation = (v_eq - current_v) / self.tau
        
        # 2. Anticipation/Pressure Term: Reaction to the relative speed of the leader
        headway = leader_s - current_s
        if headway <= self.s_min:
            # If closer than jam spacing, trigger max emergency braking
            return -9.81 
            
        # Macroscopic pressure p(rho) derivative equivalent mapped to micro-spacing
        # C proportional factor derived from the pressure function p(rho) = rho^gamma
        c_factor = self.gamma * (rho / self.rho_max) ** self.gamma
        anticipation = (c_factor * (leader_v - current_v)) / headway
        
        return relaxation + anticipation

    def step(self, current_s: float, leader_s: float, current_v: float, leader_v: float, dt: float) -> tuple[float, float]:
        """
        Advances the vehicle by a macro time step `dt` using internal sub-stepping.
        
        Returns:
        --------
        tuple (new_velocity, distance_advanced)
        """

        if leader_s is None:
            leader_s = math.inf
            leader_v = self.v_max

        sub_dt = dt / self.substeps
        sim_s = current_s
        sim_v = current_v
        
        # Linearly interpolate leader trajectory variables across the sub-steps assuming constant velocity
        leader_sub_v = leader_v
        sim_leader_s = leader_s

        for _ in range(self.substeps):
            # Compute current acceleration
            accel = self.generate_acceleration(sim_s, sim_leader_s, sim_v, leader_sub_v, sub_dt)
            
            # Update micro state using Forward Euler integration
            sim_v = sim_v + accel * sub_dt
            sim_v = max(0.0, min(self.v_max, sim_v)) # Keep physical bounds
            
            sim_s = sim_s + sim_v * sub_dt
            
            # Advance the leader's proxy position forward for the next sub-step evaluation
            sim_leader_s += leader_sub_v * sub_dt

        distance_advanced = sim_s - current_s
        return sim_v, distance_advanced

class SimplifiedSimBridge:
    spawn_length = 4.0 #6.8725979813165115 + 4.418460070966603
    min_spawn_distance = 1.0
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
        ftl_model: MicroscopicFTLVehicleModel,
        bridge_callback_name=None,
        spawn_density_function=None,
    ) -> None:
        self.sim = sim
        self.current_timestamp = sim.current_time
        self.ego_id = None
        self.road_id = road_id
        self.fd = fd
        #self.spawn_length = (1.0 / fd.rho_j) - 0.1
        #self.spawn_length = 4.0
        self.transition_region_size = 1.0 / fd.rho_c
        self.ftl_model = ftl_model
        self.vehicle_t_position = -self.spawn_width / 2.0
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
        self.masking_cell: I24MicroMask = None
        self.next_vehicle_id = 1
        self.spawn_density_function = spawn_density_function

        self.bridge_callback_name = bridge_callback_name
        sim.register_step_callback(partial(SimplifiedSimBridge._step, self), bridge_callback_name)
        sim.register_poststep_callback(partial(SimplifiedSimBridge._poststep, self), bridge_callback_name)

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
            self.inject_vehicles_from_macro()
            self.remove_vehicles_from_macro()
        else:
            self.spawn_initial_vehicles()
            print(self.vehicles)
            print(self.ego_id)
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
        self.masking_cell = new_mask
        new_mask.vehicles = self.collate_vehicles()#{vehicle: self.vehicles[vehicle] for vehicle in self.vehicles}
        self.sim.masking_cells[self._mask_id(lane_id)] = new_mask

    def _poststep(self, sim_time: float, dt: float):
        lane_id = self.lane_id
        lane_cell = self.sim.masking_cells[self._mask_id(lane_id)]
        # Do vehicle processing logic here
        self.advance_and_update_vehicles()
        self.current_timestamp += self.sim.time_resolution
        print(self.middle_s, self.anchor_speed, self.current_timestamp, self.sim.time_resolution, len(self.vehicles), float(len(self.vehicles)) / (2 * self.margin_s))
        self.flow_memory_rear = lane_cell.rear_flux_memory
        self.flow_memory_front = lane_cell.front_flux_memory
    
    # This is meant for the upper level fluid simulator. Thus we have to convert road ids to strings and restructure it to play nice with that code.
    def collate_vehicles(self):
        result = {}
        min_s, max_s = self.get_current_visible_window()
        for vehicle in self.vehicles:
            vehicle_data = self.vehicles[vehicle]
            result[str(vehicle)] = Vehicle(length=vehicle_data.length, width=vehicle_data.width, s=vehicle_data.s - min_s, t=vehicle_data.t, lane=vehicle_data.lane, s_dt=vehicle_data.s_dt)
        return result

    def update_vehicles(self, vehicles: Dict[str, Vehicle]):
        self.vehicles = vehicles

    def get_ahead_lane_velocity_micro(self, default_speed, min_cell_size=25.0):
        vehicles = self.vehicles
        min_s, max_s = self.get_current_visible_window()
        front_most_vehicle = None
        for vehicle in vehicles:
            if (front_most_vehicle is None) or (vehicles[vehicle].s > front_most_vehicle.s):
                front_most_vehicle = vehicles[vehicle]
        if (front_most_vehicle is None) or (front_most_vehicle.s < (max_s - self.spawn_lookahead)):
            return self.get_ahead_lane_velocity_macro(default_speed)
        return front_most_vehicle.s_dt
    
    def get_behind_lane_velocity_micro(self, default_speed, min_cell_size=25.0):
        vehicles = self.vehicles
        min_s, max_s = self.get_current_visible_window()
        rear_most_vehicle = None
        for vehicle in vehicles:
            if (rear_most_vehicle is None) or (vehicles[vehicle].s < rear_most_vehicle.s):
                rear_most_vehicle = vehicles[vehicle]
        if (rear_most_vehicle is None) or (rear_most_vehicle.s > (min_s + self.spawn_lookahead)):
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
    
    def spawn_initial_vehicles(self, dx=0.01, eps=1e-8, spawn_ego_exact=True):
        #self.vehicles[self.ego_id] = Vehicle(self.spawn_length, self.spawn_width, self.middle_s, self.vehicle_t_position, self.lane_id,)
        # Spawn vehicles behind and in front of the ego vehicle and keep density consistent as we go
        min_s = self.middle_s - self.margin_s
        current_s = min_s
        max_s = self.middle_s + self.margin_s
        closest_to_center_vehicle_id: str = None
        closest_to_center_vehicle: Vehicle = None
        previous_vehicle: Vehicle = None
        current_mass = 0
        while (current_s < max_s):
            density = self.spawn_density_function(current_s - min_s, 2.0 * self.margin_s)
            estimated_velocity = self.fd.velocity_from_density(density)
            spawn_here = False
            if (spawn_ego_exact and (abs(current_s - self.middle_s) < (dx - eps))):
                spawn_here = True
                spawn_ego_exact = False
            elif (current_mass >= 1.0) and ((previous_vehicle is None) or ((previous_vehicle.s + previous_vehicle.length) < current_s)):
                if (spawn_ego_exact and (((current_s + self.spawn_length) < self.middle_s) or (current_s > (self.middle_s + self.spawn_length)))) or (not spawn_ego_exact):
                    spawn_here = True

            if spawn_here:
                previous_vehicle = Vehicle(self.spawn_length, self.spawn_width, current_s, self.vehicle_t_position, self.lane_id, estimated_velocity)
                self.vehicles[self.generate_next_vehicle_id()] = previous_vehicle
                current_mass -= 1.0
            
            current_mass += (density * dx)
            current_s += dx
        for vehicle in self.vehicles:
            vehicle_data = self.vehicles[vehicle]
            if (closest_to_center_vehicle is None) or (abs(closest_to_center_vehicle.s - self.middle_s) > abs(vehicle_data.s - self.middle_s)):
                closest_to_center_vehicle = vehicle_data
                closest_to_center_vehicle_id = vehicle

        self.ego_id = closest_to_center_vehicle_id
        self.middle_s = closest_to_center_vehicle.s

    def get_immediate_leader_vehicle(self, vehicle: Vehicle):
        current_leader_key = None
        for vehicle_key in self.vehicles:
            new_vehicle = self.vehicles[vehicle_key]
            if (current_leader_key is None) or (self.vehicles[current_leader_key].s > new_vehicle.s):
                if (new_vehicle.s > vehicle.s):
                    current_leader_key = vehicle_key
        if (current_leader_key is not None):
            return current_leader_key, self.vehicles[current_leader_key]
        return None, None

    def get_front_leader_vehicle(self):
        current_leader_key = None
        min_s, max_s = self.get_current_visible_window()
        for vehicle_key in self.vehicles:
            new_vehicle = self.vehicles[vehicle_key]
            if (current_leader_key is None) or (self.vehicles[current_leader_key].s < new_vehicle.s):
                current_leader_key = vehicle_key
        if (current_leader_key is not None) and (self.vehicles[current_leader_key].s >= (max_s - self.transition_region_size)):
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
        estimated_width = self.spawn_width # We're hardcoding this for now. We'll need to add lane/road cross-referencing lookup later
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
        #spawn_lengths = min(1.0 / density, self.transition_region_size - 1e-5)
        estimated_meters_per_vehicle = (s_max - s_min) / vehicle_count
        spawn_lengths = max(min(estimated_meters_per_vehicle, self.transition_region_size - 1e-5), self.spawn_length + self.min_spawn_distance)
        spawn_distance = spawn_lengths - self.spawn_length
        vehicle_count = min(math.floor((s_max - s_min) / spawn_lengths), vehicle_count)
        if (behind_or_in_front == "behind"):
            s_min = s_max - (spawn_lengths * vehicle_count)
        else:
            s_max = s_min + (spawn_lengths * vehicle_count)
        if toprint:
            print("density: ", density)
            print("s_max, s_min: ", s_max, s_min)
            print("spawn_lengths: ", spawn_lengths)
            print("vehicle count: ", vehicle_count)
            front_s = None
            front_velocity = None
            for vehicle_id in self.vehicles:
                vehicle_data = self.vehicles[vehicle_id]
                if (front_s is None) or (front_s < vehicle_data.s):
                    front_s = vehicle_data.s + vehicle_data.length
                    front_velocity = vehicle_data.s_dt
            print("frontmost vehicle position and speed and length: ", front_s, front_velocity, self.spawn_length)
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
        min_s, max_s = self.get_current_visible_window()
        if (rear_flux_memory > 0.0):
            # Perform a poisson draw to determine the number of vehicles to create
            vehicle_count = min(math.floor(rear_flux_memory), self.vehicle_spawn_limit)
            #vehicle_count = min(numpy.random.poisson(rear_flux_memory), self.vehicle_spawn_limit)
            if (vehicle_count > 0):
                vehicle_count = self._create_vehicle_spawns_in_srange(vehicle_count, min_s, min_s + s_availability, "behind", False)
                self.flow_memory_rear -= vehicle_count

    def _spawn_vehicles_in_front(self, s_availability):
        front_flux_memory = self.flow_memory_front
        min_s, max_s = self.get_current_visible_window()
        if (front_flux_memory < 0.0):
            # Perform a poisson draw to determine the number of vehicles to create
            vehicle_count = min(math.floor(-front_flux_memory), self.vehicle_spawn_limit)
            #vehicle_count = min(numpy.random.poisson(-front_flux_memory), self.vehicle_spawn_limit)
            if (vehicle_count > 0):
                vehicle_count = self._create_vehicle_spawns_in_srange(vehicle_count, max_s - s_availability, max_s, "front", False)
                self.flow_memory_front += vehicle_count

    def inject_vehicles_from_macro(self):
        min_s, max_s = self.get_current_visible_window()
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
            s_availability = (max_s - min_s)
        else:
            desired_macro_velocity = self.get_behind_lane_velocity_macro(0.0)
            closing_rate = desired_macro_velocity - rear_velocity
            #rear_s = min(rear_s, rear_s - (closing_rate * self.spawn_ttc))
            s_availability = rear_s - min_s
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
            s_availability = (max_s - min_s)
        else:
            desired_macro_velocity = self.get_ahead_lane_velocity_macro(0.0)
            closing_rate = front_velocity - desired_macro_velocity
            #front_s = max(front_s, front_s + (closing_rate * self.spawn_ttc))
            s_availability = max_s - front_s
        s_availability = min(s_availability, self.transition_region_size)
        # Mandate a certain distance threshold of the frontmost vehicle for spawning in new stuff
        if (s_availability > self.spawn_threshold):
            self._spawn_vehicles_in_front(s_availability)

    def remove_vehicles_from_macro(self):
        min_s, max_s = self.get_current_visible_window()
        vehicles_to_remove = []
        for vehicle in self.vehicles:
            vehicle_data = self.vehicles[vehicle]
            if (vehicle_data.s > max_s):
                vehicles_to_remove.append(vehicle)
                self.flow_memory_front -= 1.0
            elif (vehicle_data.s < min_s):
                vehicles_to_remove.append(vehicle)
                self.flow_memory_rear += 1.0
        for vehicle in vehicles_to_remove:
            self.vehicles.pop(vehicle, None)

    def spawn_and_despawn_vehicles(self):
        self.inject_vehicles_from_macro()
        self.remove_vehicles_from_macro()

    def _integration_substep(self, dt):
        # Advance every micro vehicle by one sub-step of size dt. Velocities and positions
        # are read from the start-of-substep snapshot and committed together afterwards, so
        # a follower and its leader are integrated against a consistent state.
        front_leader_vehicle_key, _ = self.get_front_leader_vehicle()
        new_state = {}
        for vehicle in self.vehicles:
            vehicle_data = self.vehicles[vehicle]
            if (front_leader_vehicle_key is not None) and (front_leader_vehicle_key == vehicle):
                # The front boundary vehicle is driven by the macro flow, not the car-following model.
                new_v = self.get_ahead_lane_velocity_macro(self.fd.v_f)
                new_state[vehicle] = (new_v, vehicle_data.s + (new_v * dt))
                continue
            immediate_leader_vehicle_key, immediate_leader_vehicle_object = self.get_immediate_leader_vehicle(vehicle_data)
            if (immediate_leader_vehicle_key is None):
                leader_s = None
                leader_v = None
            else:
                leader_s = immediate_leader_vehicle_object.s
                leader_v = immediate_leader_vehicle_object.s_dt
            new_v, distance = self.ftl_model.step(vehicle_data.s, leader_s, vehicle_data.s_dt, leader_v, dt)
            new_state[vehicle] = (min(max(new_v, 0.0), self.fd.v_f), vehicle_data.s + distance)
        for vehicle, (new_v, new_s) in new_state.items():
            self.vehicles[vehicle].s_dt = new_v
            self.vehicles[vehicle].s = new_s

    def move_vehicles(self):
        # Integrate the micro vehicles across the macro step. Acceleration models sub-step
        # the ODE (model.substeps > 1) to stay collision-free; first-order models use 1 step.
        n = max(1, int(getattr(self.ftl_model, "substeps", 1)))
        dt_sub = self.sim.time_resolution / n
        for _ in range(n):
            self._integration_substep(dt_sub)

    def advance_and_update_vehicles(self):
        #print(self.vehicles, "\n---------------")
        #print(self.vehicles, "\n---------------")
        self.move_vehicles()
        #self.spawn_and_despawn_vehicles()
        #print(self.vehicles, "\n---------------")

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
