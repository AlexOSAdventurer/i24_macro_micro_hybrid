"""Engine-agnostic base for the microscopic couplers driven by I24MicroSimBridge.

The bridge asks the micro-coupler
for one thing each macroscopic step::

    middle_s = micro_coupler.step()

and expects the coupler to have pushed a fresh vehicle dictionary via
bridge.update_vehicles and to have settled up with the per-lane flux
memories bridge.flow_memory_rear / bridge.flow_memory_front before
returning.  Everything needed to honour that contract *except* the microscopic
engine itself lives here:
  1. empirical seeding of the hero and of the initial visible vehicles;
  2. the moving visible/ghost windows anchored on the hero;
  3. the macroscopic queries against the neighbouring macro cells;
  4. the bookkeeping that manages the flux-memory bookkeeping when vehicles spawn or leave

A concrete coupler supplies the engine by implementing the four hooks in the
"engine interface" section below.  ``I24SumoCoupler`` does this with TraCI.

"""
from __future__ import annotations

from typing import List
import copy
import math

import pandas

from simulation import GroundTruthStore
from i24_micro_bridge import Vehicle

class I24MicroCouplerBase:
    """Shared machinery for a bridge ``micro_coupler`` backed by a micro engine."""

    # Spawn geometry, calibrated against CARLA in sim_calibration_carla.py.
    min_spawn_length = 6.8725979813165115 + 4.418460070966603
    min_spawn_distance = 4.418460070966603
    spawn_ttc = 0.5504990436241721
    visible_time_max_difference = 0.1  # Seconds
    ghost_time_max_difference = 1.0  # Seconds
    desired_s_max_difference = 50.0  # Meters
    spawn_threshold = 0.0  # Meters
    spawn_region = 75.0  # Meters
    vehicle_spawn_limit = 5.0  # 5 cars per tick allowed

    def __init__(
        self,
        motion_data: GroundTruthStore,
        dt: float,
        lanes: List[int],
        mapping,
        hero_road: str,
        desired_time: float,
        desired_s: float,
        visible_window: float,
        ghost_window: float,
    ) -> None:
        self.motion_data = motion_data
        self.lanes = lanes
        self.dt = float(dt)
        self.bridge = None  # Set by the bridge itself
        self.current_anchor_speed = None
        self.visible_window = visible_window
        self.ghost_window = ghost_window
        self.initialized = False
        self.mapping = mapping
        self.vehicles_to_completely_ignore = []
        self.hero_road = int(hero_road)
        self.hero_lane = None
        self.hero_state = None
        self.visible_state = None
        self.ghost_state = None
        self.current_timestamp = None
        self.next_vehicle_id = 0
        self.loadHero(int(hero_road), desired_time, desired_s)
        self.loadVisible()
        self.loadGhosts()

    # ------------------------------------------------------------------
    # Engine interface — implemented by the concrete coupler
    # ------------------------------------------------------------------

    def initializeEngine(self):
        """Bring the microscopic engine up and populate it from the seeded
        hero/visible state.  Called once, lazily, on the first step()."""
        raise NotImplementedError

    def syncEngineWithVisibleState(self):
        """Reconcile the engine's vehicle population with self.visible_state:
        insert whatever the macro flux just spawned, remove whatever left."""
        raise NotImplementedError

    def advanceEngine(self, dt):
        """Run the engine forward by dt seconds.

        Returns (hero_state, visible_states) where hero_state is a
        vehicle-state dict and visible_states maps lane_id -> {id: state}
        using each vehicle's *current* lane, so a lane change inside the engine
        shows up as a move between the returned buckets.
        """
        raise NotImplementedError

    def destroyEngine(self):
        """Tear the engine down.  Called by the bridge when the bubble retires."""
        raise NotImplementedError

    def initialize(self):
        self.initializeEngine()
        self.initialized = True

    def destroy(self):
        self.destroyEngine()

    # ------------------------------------------------------------------
    # Bridge callback
    # ------------------------------------------------------------------

    def step(self) -> float:
        """Advance the micro engine one macroscopic step and return the new
        window centre.

        The first call only stands the engine up; the window does not move
        until the engine is live, which matches the CARLA coupler's behaviour.
        """
        hero = self.getHeroData()
        anchor_speed = hero["velocity"]
        middle_s = hero["s"]

        if not self.initialized:
            self.initialize()
            return middle_s

        self.visible_state = self.injectVehiclesFromMacro(self.getVisibleData())
        self.syncEngineWithVisibleState()
        new_hero_state, new_visible_states = self.advanceEngine(self.dt)
        self.updateHeroVehicleFromEngine(new_hero_state)
        self.updateVisibleVehiclesFromEngine(new_visible_states)
        vehicles = self.collateVisibleAndHeroVehicles(self.bridge)

        self.bridge.update_vehicles(vehicles)
        self.bridge.anchor_speed = anchor_speed
        result = self._compute_next_middle_s(
            middle_s,
            self.current_anchor_speed if self.current_anchor_speed is not None else anchor_speed,
        )
        self.current_anchor_speed = anchor_speed
        self.current_timestamp += self.dt
        return result

    def _compute_next_middle_s(self, middle_s: float, anchor_speed: float | None) -> float:
        """Advance the window centre by the anchor vehicle's speed x dt."""
        if anchor_speed is None:
            return middle_s
        return middle_s + anchor_speed * self.dt

    # ------------------------------------------------------------------
    # Empirical data access
    # ------------------------------------------------------------------

    def get_lane_dfs(self, timestamp_min, timestamp_max, s_min, s_max):
        df = self.motion_data.micro_df
        window = df[
            (df["time"] >= timestamp_min)
            & (df["time"] <= timestamp_max)
            & (df["s"] >= s_min)
            & (df["s"] <= s_max)
            & (df["road_id"] == str(self.hero_road))
        ]
        return {lane: window[window["lane_id"] == lane] for lane in self.lanes}

    def getVehicleTrajectoryFromReal(self, id, lane):
        df = self.motion_data.micro_df
        window = df[(df["id"] == id) & (df["lane_id"] == lane)]
        return window

    def estimateVehicleVelocityFromReal(self, id, lane, current_time):
        trajectory = self.getVehicleTrajectoryFromReal(id, lane)
        trajectory_time_sorted = trajectory.sort_values(by=["time"], ascending=True)
        if len(trajectory_time_sorted) < 2:
            return 0.0
        current_index = int(trajectory_time_sorted["time"].searchsorted(current_time))
        if current_index >= len(trajectory_time_sorted):
            current_index = len(trajectory_time_sorted) - 1
        if current_index > 0:
            return float(
                trajectory_time_sorted.iloc[current_index]["s"]
                - trajectory_time_sorted.iloc[current_index - 1]["s"]
            ) / float(
                trajectory_time_sorted.iloc[current_index]["time"]
                - trajectory_time_sorted.iloc[current_index - 1]["time"]
            )
        return float(
            trajectory_time_sorted.iloc[current_index + 1]["s"]
            - trajectory_time_sorted.iloc[current_index]["s"]
        ) / float(
            trajectory_time_sorted.iloc[current_index + 1]["time"]
            - trajectory_time_sorted.iloc[current_index]["time"]
        )

    def estimateGhostVehicleVelocity(self, vehicle_data):
        # Replay its velocity for as long as the trajectory data lasts; once that
        # expires the ghost is discarded and whatever replaces it is used instead.
        return self.estimateVehicleVelocityFromReal(
            vehicle_data["id"], vehicle_data["lane_id"], self.current_timestamp
        )

    # ------------------------------------------------------------------
    # Vehicle state construction
    # ------------------------------------------------------------------

    def generateNextVehicleID(self):
        new_id = self.next_vehicle_id
        self.next_vehicle_id += 1
        return new_id

    def generateVehicleStateFromRow(self, row, lane, current_timestamp=None):
        if current_timestamp is None:
            current_timestamp = self.current_timestamp
        estimated_velocity = self.estimateVehicleVelocityFromReal(
            int(row["id"]), lane, float(row["time"])
        )
        estimated_s = float(row["s"]) + ((current_timestamp - float(row["time"])) * estimated_velocity)
        return {
            "id": self.generateNextVehicleID(),
            "class": str(row["class"]),
            "length": float(row["length"]),
            "width": float(row["width"]),
            "time": current_timestamp,
            "s": estimated_s,
            "t": float(row["t"]),
            "velocity": estimated_velocity,
            "lane_id": int(lane),
            "road_id": self.hero_road,
        }

    # behind_or_in_front is either "behind" or "front"
    def generateVehicleStateFromSpawn(self, lane, s_min, s_max, behind_or_in_front="behind"):
        if (behind_or_in_front != "behind") and (behind_or_in_front != "front"):
            return None  # Force failure upstream.
        new_time = self.current_timestamp
        estimated_velocity = (
            self.getBehindLaneVelocityMicro(lane, 0.0)
            if (behind_or_in_front == "behind")
            else self.getAheadLaneVelocityMicro(lane, 0.0)
        )
        estimated_width = 3.5  # Hardcoded pending a lane/road cross-reference lookup.
        return {
            "id": self.generateNextVehicleID(),
            "class": "spawned",
            "length": float(s_max - s_min),
            "width": estimated_width,
            "time": new_time,
            "s": float(s_min),
            "t": (lane * estimated_width) + (estimated_width / 2),
            "velocity": estimated_velocity,
            "lane_id": lane,
            "road_id": self.hero_road,
        }

    def generateUpdatedVehicleStateFromEngine(self, vehicle_data):
        new_time = self.current_timestamp
        return {
            "id": int(vehicle_data["id"]),
            "class": str(vehicle_data["class"]),
            "length": float(vehicle_data["length"]),
            "width": float(vehicle_data["width"]),
            "time": new_time,
            "s": float(vehicle_data["s"]),
            "t": float(vehicle_data["t"]),
            "velocity": vehicle_data["velocity"],
            "lane_id": int(vehicle_data["lane_id"]),
            "road_id": self.hero_road,
        }

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def loadHero(self, hero_road, desired_time, desired_s):
        potential_heroes = pandas.concat(
            list(
                (
                    self.get_lane_dfs(
                        desired_time - self.visible_time_max_difference,
                        desired_time + self.visible_time_max_difference,
                        desired_s - self.desired_s_max_difference,
                        desired_s + self.desired_s_max_difference,
                    )
                ).values()
            )
        )
        if len(potential_heroes) == 0:
            raise Exception(f"No avaiable heroes with {hero_road} and {desired_time} and {desired_s}")
        potential_heroes["time_delta"] = (potential_heroes["time"] - desired_time).abs()
        potential_heroes["s_delta"] = (potential_heroes["s"] - desired_s).abs()
        potential_heroes_sorted = potential_heroes.sort_values(
            by=["time_delta", "s_delta"], ascending=True
        )
        selected_hero = potential_heroes_sorted.iloc[0]
        hero_lane = selected_hero.lane_id
        original_hero_state = self.generateVehicleStateFromRow(
            selected_hero, hero_lane, float(selected_hero["time"])
        )
        self.hero_state = original_hero_state
        self.hero_lane = hero_lane
        self.current_timestamp = original_hero_state["time"]
        self.anchor_speed = self.hero_state["velocity"]

    def loadVisible(self):
        self.visible_state = {}
        for lane in self.getLanes():
            self.visible_state[lane] = {}
        min_timestamp, max_timestamp, min_s, max_s = self.getCurrentVisibleWindow()
        potential_visibles = self.get_lane_dfs(min_timestamp, max_timestamp, min_s, max_s)
        for lane in self.getLanes():
            potential_visibles_lane_sorted = potential_visibles[lane].sort_values(
                by=["time"], ascending=True
            )
            uniques = list(potential_visibles_lane_sorted["id"].unique())
            for unique in uniques:
                if unique != self.hero_state["id"]:
                    unique_vehicle_data = potential_visibles_lane_sorted[
                        potential_visibles_lane_sorted["id"] == unique
                    ].copy()
                    unique_vehicle_data["time_delta"] = (
                        unique_vehicle_data["time"] - self.current_timestamp
                    ).abs()
                    unique_vehicle_data_sorted = unique_vehicle_data.sort_values(
                        by=["time_delta"], ascending=True
                    )
                    unique_vehicle_data_row = unique_vehicle_data_sorted.iloc[0]
                    candidate = self.generateVehicleStateFromRow(unique_vehicle_data_row, lane)
                    self.registerNewVisibleVehicle(candidate)

    def loadGhosts(self, ignore_ids=None):
        # The ghost region is 0 m on each side in the current configuration, so
        # there is nothing to load. Kept for symmetry with the visible state.
        self.ghost_state = {"behind": {}, "ahead": {}}
        for lane in self.getLanes():
            self.ghost_state["behind"][lane] = {}
            self.ghost_state["ahead"][lane] = {}

    # ------------------------------------------------------------------
    # Windows and accessors
    # ------------------------------------------------------------------

    def getCurrentVisibleWindow(self):
        return (
            self.current_timestamp - self.visible_time_max_difference,
            self.current_timestamp + self.visible_time_max_difference,
            self.hero_state["s"] - self.visible_window,
            self.hero_state["s"] + self.visible_window,
        )

    def getCurrentBehindGhostWindow(self):
        return (
            self.current_timestamp - self.ghost_time_max_difference,
            self.current_timestamp + self.ghost_time_max_difference,
            self.hero_state["s"] - self.visible_window - self.ghost_window,
            self.hero_state["s"] - self.visible_window,
        )

    def getCurrentAheadGhostWindow(self):
        return (
            self.current_timestamp - self.ghost_time_max_difference,
            self.current_timestamp + self.ghost_time_max_difference,
            self.hero_state["s"] + self.visible_window,
            self.hero_state["s"] + self.visible_window + self.ghost_window,
        )

    def getLanes(self):
        return self.lanes

    def getVisibleIds(self):
        result = {}
        for lane in self.getLanes():
            result[lane] = [id for id in self.visible_state[lane]]
        return result

    def getVisibleIdsFlat(self):
        visible_ids = self.getVisibleIds()
        return sum([visible_ids[lane] for lane in self.getLanes()], [])

    def getGhostIds(self):
        result = {"behind": {}, "ahead": {}}
        for position in self.ghost_state:
            for lane in self.getLanes():
                result[position][lane] = [id for id in self.ghost_state[position][lane]]
        return result

    def getVisibleData(self):
        return self.visible_state

    def getGhostData(self):
        return self.ghost_state

    def getHeroData(self):
        return self.hero_state

    # This is meant for the upper level fluid simulator, which wants string road
    # ids and bubble-relative positions.
    def collateVisibleAndHeroVehicles(self, bridge):
        result = {}
        visible_data = self.getVisibleData()
        hero_data = self.getHeroData()
        min_timestamp, max_timestamp, min_s, max_s = self.getCurrentVisibleWindow()
        for lane in visible_data:
            visible_data_copied = copy.deepcopy(visible_data[lane])
            for i in visible_data_copied:
                visible_data_copied[i]["road_id"] = str(visible_data_copied[i]["road_id"])
                result[str(visible_data_copied[i]["id"])] = Vehicle(
                    length=visible_data_copied[i]["length"],
                    width=visible_data_copied[i]["width"],
                    s=visible_data_copied[i]["s"] - min_s,
                    t=visible_data_copied[i]["t"],
                    lane=visible_data_copied[i]["lane_id"],
                    s_dt=visible_data_copied[i]["velocity"],
                )
        hero_copied = copy.deepcopy(hero_data)
        hero_copied["road_id"] = str(hero_data["road_id"])
        result[str(hero_copied["id"])] = Vehicle(
            length=hero_copied["length"],
            width=hero_copied["width"],
            s=hero_copied["s"] - min_s,
            t=hero_copied["t"],
            lane=hero_copied["lane_id"],
            s_dt=hero_copied["velocity"],
        )
        return result

    def getLowestBehindGhostVehicle(self, lane):
        return self._extremum(self.ghost_state["behind"][lane], lowest=True)

    def getHighestAheadGhostVehicle(self, lane):
        return self._extremum(self.ghost_state["ahead"][lane], lowest=False)

    def getLowestAheadGhostVehicle(self, lane):
        return self._extremum(self.ghost_state["ahead"][lane], lowest=True)

    def getLowestBehindVisibleVehicle(self, lane):
        return self._extremum(self.visible_state[lane], lowest=True)

    def getHighestAheadVisibleVehicle(self, lane):
        return self._extremum(self.visible_state[lane], lowest=False)

    @staticmethod
    def _extremum(lane_data, lowest):
        vehicle_ids = list(lane_data.keys())
        if len(vehicle_ids) == 0:
            return None
        selected = lane_data[vehicle_ids[0]]
        for entry in vehicle_ids[1:]:
            current_entry = lane_data[entry]
            if (current_entry["s"] < selected["s"]) if lowest else (current_entry["s"] > selected["s"]):
                selected = current_entry
        return selected

    # ------------------------------------------------------------------
    # Macroscopic queries against the cells flanking the mask
    # ------------------------------------------------------------------

    def _neighbor_cell(self, lane_id, side):
        mask_cell = self.bridge.sim.active.get_cell_with_mask(self.bridge._mask_id(lane_id))
        neighbors = mask_cell.outflow_neighbors if side == "ahead" else mask_cell.inflow_neighbors
        if len(neighbors) == 0:
            return None
        cell = self.bridge.sim.active.active_cells[neighbors[0]]
        if cell.kind == "mask":
            return None  # Masks are not connected to one another.
        return cell

    def getAheadLaneVelocityMacro(self, lane_id, default_speed, min_cell_size=25.0):
        cell = self._neighbor_cell(lane_id, "ahead")
        if cell is None:
            return default_speed
        return cell.fd.velocity_from_density(cell.mass / (cell.end_s - cell.start_s))

    def getBehindLaneVelocityMacro(self, lane_id, default_speed):
        cell = self._neighbor_cell(lane_id, "behind")
        if cell is None:
            return default_speed
        return cell.fd.velocity_from_density(cell.mass / (cell.end_s - cell.start_s))

    def getAheadLaneDensity(self, lane_id, default_density):
        cell = self._neighbor_cell(lane_id, "ahead")
        if cell is None:
            return default_density
        return cell.mass / (cell.end_s - cell.start_s)

    def getBehindLaneDensity(self, lane_id, default_density):
        cell = self._neighbor_cell(lane_id, "behind")
        if cell is None:
            return default_density
        return cell.mass / (cell.end_s - cell.start_s)

    def getAheadLaneVelocityMicro(self, lane_id, default_speed, min_cell_size=25.0):
        front_most_vehicle = self.getHighestAheadVisibleVehicle(lane_id)
        if front_most_vehicle is None:
            return self.getAheadLaneVelocityMacro(lane_id, default_speed, min_cell_size)
        return front_most_vehicle["velocity"]

    def getBehindLaneVelocityMicro(self, lane_id, default_speed, min_cell_size=25.0):
        rear_most_vehicle = self.getLowestBehindVisibleVehicle(lane_id)
        if rear_most_vehicle is None:
            return self.getBehindLaneVelocityMacro(lane_id, default_speed)
        return rear_most_vehicle["velocity"]

    # ------------------------------------------------------------------
    # Macro -> micro injection
    # ------------------------------------------------------------------

    # behind_or_in_front is either "behind" or "front"
    def _createVehicleSpawnsInSRange(
        self, new_visible_states, lane, vehicle_count, s_min, s_max, behind_or_in_front, toprint=False
    ):
        density = (
            self.getBehindLaneDensity(lane, 0.001)
            if (behind_or_in_front == "behind")
            else self.getAheadLaneDensity(lane, 0.001)
        )
        spawn_lengths = min(
            max(1.0 / density, self.min_spawn_length), max(s_max - s_min, self.min_spawn_length)
        )
        vehicle_count = min(math.floor((s_max - s_min) / spawn_lengths), vehicle_count)
        if behind_or_in_front == "behind":
            s_max = s_min + (spawn_lengths * vehicle_count)
        else:
            s_min = s_max - (spawn_lengths * vehicle_count)
        if toprint:
            print("density: ", density)
            print("s_max, s_min: ", s_max, s_min)
            print("spawn_lengths: ", spawn_lengths)
            print("vehicle count: ", vehicle_count)
        for i in range(vehicle_count):
            start_position = s_min + (i * spawn_lengths)
            if behind_or_in_front == "behind":
                start_position_calculated = start_position
                end_position_calculated = start_position + spawn_lengths - self.min_spawn_distance
            else:
                start_position_calculated = start_position + self.min_spawn_distance
                end_position_calculated = start_position + spawn_lengths

            new_vehicle_data = self.generateVehicleStateFromSpawn(
                lane, start_position_calculated, end_position_calculated, behind_or_in_front
            )
            new_visible_states[lane][new_vehicle_data["id"]] = new_vehicle_data
        return new_visible_states, vehicle_count

    def _spawnVehiclesInRear(self, new_visible_states, lane, s_availability):
        rear_flux_memory = self.bridge.flow_memory_rear[lane]
        visible_window = self.getCurrentVisibleWindow()
        if rear_flux_memory > 0.0:
            vehicle_count = min(math.floor(rear_flux_memory), self.vehicle_spawn_limit)
            if vehicle_count > 0:
                new_visible_states, vehicle_count = self._createVehicleSpawnsInSRange(
                    new_visible_states,
                    lane,
                    vehicle_count,
                    visible_window[2],
                    visible_window[2] + s_availability,
                    "behind",
                )
                self.bridge.flow_memory_rear[lane] -= vehicle_count
        return new_visible_states

    def _spawnVehiclesInFront(self, new_visible_states, lane, s_availability):
        front_flux_memory = self.bridge.flow_memory_front[lane]
        visible_window = self.getCurrentVisibleWindow()
        if front_flux_memory < 0.0:
            vehicle_count = min(math.floor(-front_flux_memory), self.vehicle_spawn_limit)
            if vehicle_count > 0:
                new_visible_states, vehicle_count = self._createVehicleSpawnsInSRange(
                    new_visible_states,
                    lane,
                    vehicle_count,
                    visible_window[3] - s_availability,
                    visible_window[3],
                    "front",
                )
                self.bridge.flow_memory_front[lane] += vehicle_count
        return new_visible_states

    def injectVehiclesFromMacro(self, new_visible_states):
        visible_window = self.getCurrentVisibleWindow()
        for lane in self.getLanes():
            # Rear boundary: how much clear road is there behind the rearmost car?
            rear_s = None
            rear_velocity = None
            for vehicle_id in new_visible_states[lane]:
                vehicle_data = new_visible_states[lane][vehicle_id]
                if (rear_s is None) or (rear_s > vehicle_data["s"]):
                    rear_s = vehicle_data["s"]
                    rear_velocity = vehicle_data["velocity"]
            if rear_s is None:
                s_availability = visible_window[3] - visible_window[2]
            else:
                desired_macro_velocity = self.getBehindLaneVelocityMacro(lane, 0.0)
                closing_rate = desired_macro_velocity - rear_velocity
                rear_s = min(rear_s, rear_s - (closing_rate * self.spawn_ttc))
                s_availability = rear_s - visible_window[2]
            s_availability = min(s_availability, self.spawn_region)
            if s_availability > self.spawn_threshold:
                new_visible_states = self._spawnVehiclesInRear(new_visible_states, lane, s_availability)

            # Front boundary: same, ahead of the frontmost car.
            front_s = None
            front_velocity = None
            for vehicle_id in new_visible_states[lane]:
                vehicle_data = new_visible_states[lane][vehicle_id]
                if (front_s is None) or (front_s < vehicle_data["s"]):
                    front_s = vehicle_data["s"] + vehicle_data["length"]
                    front_velocity = vehicle_data["velocity"]
            if front_s is None:
                s_availability = visible_window[3] - visible_window[2]
            else:
                desired_macro_velocity = self.getAheadLaneVelocityMacro(lane, 0.0)
                closing_rate = front_velocity - desired_macro_velocity
                front_s = max(front_s, front_s + (closing_rate * self.spawn_ttc))
                s_availability = visible_window[3] - front_s
            s_availability = min(s_availability, self.spawn_region)
            if s_availability > self.spawn_threshold:
                new_visible_states = self._spawnVehiclesInFront(new_visible_states, lane, s_availability)

        return new_visible_states

    # ------------------------------------------------------------------
    # Micro -> macro readback
    # ------------------------------------------------------------------

    def updateHeroVehicleFromEngine(self, new_hero_state):
        hero_state_processed = self.generateUpdatedVehicleStateFromEngine(new_hero_state)
        hero_state_processed["velocity"] = float(new_hero_state["velocity"])
        self.hero_state = hero_state_processed

    def updateVisibleVehiclesFromEngine(self, new_visible_states):
        """Rebuild ``visible_state`` from what the engine reports.

        ``new_visible_states`` is keyed by each vehicle's *current* lane, so a
        vehicle that changed lanes inside the engine is credited to the lane it
        ended up in.  That lets per-lane flux memory disagree with the macro
        per-lane density -- which is expected and permitted -- while the total
        over all lanes still balances.
        """
        self.visible_state = {}
        for lane in self.getLanes():
            self.visible_state[lane] = {}
        for lane in self.getLanes():
            for vehicle_id in new_visible_states[lane]:
                vehicle_data = new_visible_states[lane][vehicle_id]
                visible_window = self.getCurrentVisibleWindow()
                if vehicle_data["s"] < visible_window[2]:
                    self.bridge.flow_memory_rear[lane] += 1
                elif vehicle_data["s"] > visible_window[3]:
                    self.bridge.flow_memory_front[lane] -= 1
                else:
                    new_data = self.generateUpdatedVehicleStateFromEngine(vehicle_data)
                    self.registerNewVisibleVehicle(new_data, True)

    def creditLostVehicle(self, lane_id, side):
        """Return a vanished vehicle's mass to the macro flux memory.

        The engine calls this when a vehicle it was tracking disappears for a
        reason the window logic cannot see: an insertion that never took, an
        engine-side removal, or a vehicle driving off the coupled edge.  Without
        it that vehicle's mass would simply be destroyed.  ``side`` is "rear" or
        "front" and matches the sign convention of the flux memories.
        """
        if side == "rear":
            self.bridge.flow_memory_rear[lane_id] += 1
        elif side == "front":
            self.bridge.flow_memory_front[lane_id] -= 1
        else:
            raise ValueError(f"side must be 'rear' or 'front', got {side!r}")

    # ------------------------------------------------------------------
    # Ghost promotion (inactive while ghost_window == 0)
    # ------------------------------------------------------------------

    def updateVisibleVehiclesWithGhostSelection(self, ghost_data):
        visible_window = self.getCurrentVisibleWindow()
        for lane in self.getLanes():
            for id in ghost_data[lane]:
                candidate = ghost_data[lane][id]
                candidate_new_s = candidate["s"] + (
                    candidate["velocity"] * (self.current_timestamp - candidate["time"])
                )
                if (candidate_new_s > visible_window[2]) and (candidate_new_s < visible_window[3]):
                    if self.checkIfCandidateVisibleNoOverlapWithCurrentVisible(lane, candidate):
                        candidate["s"] = candidate_new_s
                        self.registerNewVisibleVehicle(candidate)

    def updateVisibleVehiclesViaGhosts(self):
        self.updateVisibleVehiclesWithGhostSelection(self.ghost_state["behind"])
        self.updateVisibleVehiclesWithGhostSelection(self.ghost_state["ahead"])

    # ------------------------------------------------------------------
    # Geometry checks and registration
    # ------------------------------------------------------------------

    def checkVehicleBoundingBoxNoOverlap(self, vehicle1, vehicle2):
        vehicle_first = vehicle1 if (vehicle1["s"] < vehicle2["s"]) else vehicle2
        vehicle_second = vehicle1 if (vehicle1["s"] > vehicle2["s"]) else vehicle2
        vehicle_first_min = vehicle_first["s"]
        vehicle_first_max = vehicle_first["s"] + vehicle_first["length"] + self.min_spawn_distance
        vehicle_second_min = vehicle_second["s"]

        return (vehicle_first_min < vehicle_second_min) and (vehicle_first_max < vehicle_second_min)

    def checkIfCandidateGhostNoOverlapWithCurrentGhosts(self, ghost_data, candidate):
        for id in ghost_data:
            if not self.checkVehicleBoundingBoxNoOverlap(ghost_data[id], candidate):
                return False
        return True

    def checkIfInitVisibleNoOverlapWithCurrentVisible(self, lane, candidate):
        for id in self.visible_state[lane]:
            if not self.checkVehicleBoundingBoxNoOverlap(self.visible_state[lane][id], candidate):
                return False
        return True

    def checkIfCandidateVisibleNoOverlapWithCurrentVisible(self, lane, candidate):
        # 1d bounding box over the lane, from the backmost visible vehicle to the
        # frontmost one. A candidate may not breach it.
        lowest_vehicle = self.getLowestBehindVisibleVehicle(lane)
        highest_vehicle = self.getHighestAheadVisibleVehicle(lane)
        if lowest_vehicle is None:
            return True
        return ((candidate["s"] + candidate["length"]) < lowest_vehicle["s"]) or (
            candidate["s"] > (highest_vehicle["s"] + highest_vehicle["length"])
        )

    def registerNewVisibleVehicle(self, vehicle_data, ignore_invalid_visible_cell_position=False):
        if vehicle_data["id"] in self.vehicles_to_completely_ignore:
            print(f"WARNING: Threw away visible {vehicle_data} because it was marked as a vehicle to ignore.")
            return
        visible_window = self.getCurrentVisibleWindow()
        if (vehicle_data["s"] > visible_window[2]) and (vehicle_data["s"] < visible_window[3]):
            if ignore_invalid_visible_cell_position or self.checkIfInitVisibleNoOverlapWithCurrentVisible(
                vehicle_data["lane_id"], vehicle_data
            ):
                self.visible_state[vehicle_data["lane_id"]][vehicle_data["id"]] = vehicle_data

    def registerNewGhostVehicle(self, vehicle_data, ignore_invalid_ghost_cell_position=False):
        # With a 0 m ghost region on each side there is nothing to hold onto, so
        # vehicles that slip out are simply dropped here.
        pass
