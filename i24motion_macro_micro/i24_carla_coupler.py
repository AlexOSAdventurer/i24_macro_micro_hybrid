from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List
import json
import pandas

import copy
from simulation import GroundTruthStore
from i24_micro_bridge import Vehicle
import sys
sys.path.append("/workspaces/")
from i24motion_to_carla import I24MotionCarlaSimulationCoupled

if TYPE_CHECKING:
    from i24_micro_bridge import I24MicroSimBridge


class I24CarlaCoupler:
    visible_time_max_difference = 0.1 # Seconds
    ghost_time_max_difference = 1.0 # Seconds
    desired_s_max_difference = 20.0 # Meters

    def __init__(self, motion_data: GroundTruthStore, dt: float, lanes: List[int], mapping, hero_road: str, desired_time: float, desired_s: float, visible_window: float, ghost_window: float) -> None:
        self.motion_data = motion_data
        self.lanes = lanes
        self.dt = float(dt)
        self.bridge = None # Set by the bridge itself
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
        self.loadHero(int(hero_road), desired_time, desired_s)
        self.loadVisible()
        self.loadGhosts()
        self.carla_sim = I24MotionCarlaSimulationCoupled("localhost", 2000, self, self.mapping["road_data"], "result_new_road_2_lane_1.csv", "result_bev_road_2_lane_1.mp4")

    def get_lane_dfs(self, timestamp_min, timestamp_max, s_min, s_max):
        df = self.motion_data.micro_df
        window = df[
            (df["time"] >= timestamp_min) &
            (df["time"] <= timestamp_max) &
            (df["s"] >= s_min) &
            (df["s"] <= s_max)
        ]
        return {lane: window[window["lane_id"] == lane] for lane in self.lanes}
    
    def initialize(self):
        self.carla_sim.initializeSimulation()
        self.initialized = True

    # ------------------------------------------------------------------
    # Bridge callback — called each step with the bridge as argument
    # ------------------------------------------------------------------

    def step(self) -> float:
        """Query empirical vehicles for this timestep, update the bridge,
        and return the new window centre position.

        The window centre tracks the vehicle currently closest to middle_s.
        That vehicle's speed advances the window; the next step then finds
        the closest vehicle to the new position and repeats.

        This method is passed directly as bridge.update_micro_callback.
        """
        
        if self.initialized:
            self.carla_sim.updateSimulation()
            new_hero_state, new_visible_states = self.carla_sim.runSimulationOver(self.dt)
            self.updateHeroVehicleViaCARLA(new_hero_state)
            self.updateVisibleVehiclesViaCARLA(new_visible_states)
        else:
            self.initialize(self.bridge)

        #lane_dfs = self.get_lane_dfs(sim_time, sim_time + self.dt, s_min, s_max)
        """lane_dfs = self.motion_data.queryEdieBoxSubset(
            timestamp_min=sim_time,
            timestamp_max=sim_time + self.dt,
            s_min=s_min,
            s_max=s_max,
        )
        
        self.updateHeroVehicleViaCARLA(new_hero_state)
        self.updateVisibleVehiclesViaCARLA(new_visible_states)
        self.updateVisibleVehiclesViaGhosts()
        """

        hero = self.getHeroData()
        anchor_speed = hero["velocity"]
        middle_s = hero["s"]
        vehicles = self.collateVisibleAndHeroVehicles(self.bridge)

        self.bridge.update_vehicles(vehicles)
        self.bridge.anchor_speed = anchor_speed
        result = self._compute_next_middle_s(middle_s, self.current_anchor_speed if self.current_anchor_speed is not None else anchor_speed)
        self.current_anchor_speed = anchor_speed
        return result

    def _compute_next_middle_s(
        self, middle_s: float, anchor_speed: float | None
    ) -> float:
        """Advance the window centre by the anchor vehicle's speed × dt.

        Returns middle_s unchanged if no anchor vehicle was found.
        """
        if anchor_speed is None:
            return middle_s
        return middle_s + anchor_speed * self.dt
    
    def getAheadLaneVelocity(self, lane_id):
        pass
    
    def destroy(self):
        self.carla_sim.destroySimulation()
    
    def generateVehicleStateFromRow(self, row, lane, current_timestamp=None):
        if current_timestamp is None:
            current_timestamp = self.current_timestamp
        estimated_velocity = self.estimateVehicleVelocityFromReal(int(row["id"]), lane, float(row["time"]))
        estimated_s = float(row["s"]) + ((current_timestamp - float(row["time"])) * estimated_velocity)
        return {
            "id": int(row["id"]),
            "class": int(row["class"]),
            "length": float(row["length"]),
            "width": float(row["width"]),
            "time": current_timestamp,
            "s": estimated_s,
            "t": float(row["t"]),
            "velocity": estimated_velocity,
            "lane_id": lane,
            "road_id": self.hero_road
        }
    
    def generateUpdatedVehicleStateFromCARLA(self, vehicle_data):
        new_time = self.current_timestamp
        estimated_velocity = self.estimateVehicleVelocityFromReal(int(vehicle_data["id"]), int(vehicle_data["lane_id"]), new_time)
        return {
            "id": int(vehicle_data["id"]),
            "class": int(vehicle_data["class"]),
            "length": float(vehicle_data["length"]),
            "width": float(vehicle_data["width"]),
            "time": new_time,
            "s": float(vehicle_data["s"]),
            "t": float(vehicle_data["t"]),
            "velocity": estimated_velocity,
            "lane_id": int(vehicle_data["lane_id"]),
            "road_id": self.hero_road
        }
    
    def getVehicleTrajectoryFromReal(self, id, lane):
        df = self.motion_data.micro_df
        window = df[(df["id"] == id) & (df["lane_id"] == lane)]
        return window
    
    def estimateVehicleVelocityFromReal(self, id, lane, current_time):
        trajectory = self.getVehicleTrajectoryFromReal(id, lane)
        trajectory_time_sorted = trajectory.sort_values(by=['time'], ascending=True)
        if len(trajectory_time_sorted) < 2:
            return 0.0
        current_index = int(trajectory_time_sorted['time'].searchsorted(current_time))
        if (current_index >= len(trajectory_time_sorted)):
            current_index = len(trajectory_time_sorted) - 1
        if (current_index > 0):
            return float(trajectory_time_sorted.iloc[current_index]["s"] - trajectory_time_sorted.iloc[current_index - 1]["s"]) / float(trajectory_time_sorted.iloc[current_index]["time"] - trajectory_time_sorted.iloc[current_index - 1]["time"])
        return float(trajectory_time_sorted.iloc[current_index + 1]["s"] - trajectory_time_sorted.iloc[current_index]["s"]) / float(trajectory_time_sorted.iloc[current_index + 1]["time"] - trajectory_time_sorted.iloc[current_index]["time"])
    
    def estimateGhostVehicleVelocity(self, vehicle_data):
        # Replay its velocity for as long as we have it from the trajectory data - when that expires, we will simply discard the ghost and assume whatever its replacement becomes
        # This keeps us as data driven as possible
        return self.estimateVehicleVelocityFromReal(self, vehicle_data["id"], vehicle_data["lane_id"], self.current_timestamp)

    def loadHero(self, hero_road, desired_time, desired_s):
        potential_heroes = pandas.concat(list((self.get_lane_dfs(desired_time - self.visible_time_max_difference, desired_time + self.visible_time_max_difference, desired_s - self.desired_s_max_difference, desired_s + self.desired_s_max_difference)).values()))
        #potential_heroes = self.real_data.queryEdieBoxSubset(desired_time - self.visible_time_max_difference, desired_time + self.visible_time_max_difference, desired_s - self.desired_s_max_difference, desired_s + self.desired_s_max_difference)[hero_lane]
        if len(potential_heroes) == 0:
            raise Exception(f"No avaiable heroes with {hero_road} and {desired_time} and {desired_s}")
        potential_heroes["time_delta"] = (potential_heroes["time"] - desired_time).abs()
        potential_heroes["s_delta"] = (potential_heroes["s"] - desired_s).abs()
        potential_heroes_sorted = potential_heroes.sort_values(by=['time_delta', 's_delta'], ascending=True)
        selected_hero = potential_heroes_sorted.iloc[0]
        hero_lane = selected_hero.lane_id
        original_hero_state = self.generateVehicleStateFromRow(selected_hero, hero_lane, float(selected_hero["time"]))
        self.hero_state = original_hero_state
        self.hero_lane = hero_lane
        self.current_timestamp = original_hero_state["time"]
        self.anchor_speed = self.hero_state["velocity"]

    def getCurrentVisibleWindow(self):
        #print(self.hero_state["s"] - self.visible_window, self.hero_state["s"] + self.visible_window)
        return self.current_timestamp - self.visible_time_max_difference, self.current_timestamp + self.visible_time_max_difference, self.hero_state["s"] - self.visible_window, self.hero_state["s"] + self.visible_window
    
    def getCurrentBehindGhostWindow(self):
        return self.current_timestamp - self.ghost_time_max_difference, self.current_timestamp + self.ghost_time_max_difference, self.hero_state["s"] - self.visible_window - self.ghost_window, self.hero_state["s"] - self.visible_window
    
    def getCurrentAheadGhostWindow(self):
        return self.current_timestamp - self.ghost_time_max_difference, self.current_timestamp + self.ghost_time_max_difference, self.hero_state["s"] + self.visible_window, self.hero_state["s"] + self.visible_window + self.ghost_window
    
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
        result = {
            "behind": {},
            "ahead": {}
        }
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
    
    # This is meant for the upper level fluid simulator. Thus we have to convert road ids to strings and restructure it to play nice with that code.
    # Long story short, I have some tech debt here
    def collateVisibleAndHeroVehicles(self, bridge):
        result = {}
        visible_data = self.getVisibleData()
        hero_data = self.getHeroData()
        min_timestamp, max_timestamp, min_s, max_s = self.getCurrentVisibleWindow()
        for lane in visible_data:
            visible_data_copied = copy.deepcopy(visible_data[lane])
            for i in visible_data_copied:
                visible_data_copied[i]["road_id"] = str(visible_data_copied[i]["road_id"])
                result[str(visible_data_copied[i]["id"])] = Vehicle(length=visible_data_copied[i]["length"], width=visible_data_copied[i]["width"], s=visible_data_copied[i]["s"] - min_s, t=visible_data_copied[i]["t"], lane=visible_data_copied[i]["lane_id"], s_dt=visible_data_copied[i]["velocity"]) 
        hero_copied = copy.deepcopy(hero_data)
        hero_copied["road_id"] = str(hero_data["road_id"])
        result[str(hero_copied["id"])] = Vehicle(length=hero_copied["length"], width=hero_copied["width"], s=hero_copied["s"] - min_s, t=hero_copied["t"], lane=hero_copied["lane_id"], s_dt=hero_copied["velocity"])
        return result
    
    def getLowestBehindGhostVehicle(self, lane):
        lane_data = self.ghost_state["behind"][lane]
        vehicle_ids = list(lane_data.keys())
        if len(vehicle_ids) == 0:
            return None
        lowest_vehicle = lane_data[vehicle_ids[0]]
        for entry in vehicle_ids[1:]:
            current_entry = lane_data[entry]
            if (current_entry["s"] < lowest_vehicle["s"]):
                lowest_vehicle = current_entry
        return lowest_vehicle
    
    def getHighestAheadGhostVehicle(self, lane):
        lane_data = self.ghost_state["ahead"][lane]
        vehicle_ids = list(lane_data.keys())
        if len(vehicle_ids) == 0:
            return None
        highest_vehicle = lane_data[vehicle_ids[0]]
        for entry in vehicle_ids[1:]:
            current_entry = lane_data[entry]
            if (current_entry["s"] > highest_vehicle["s"]):
                highest_vehicle = current_entry
        return highest_vehicle
    
    def getLowestAheadGhostVehicle(self, lane):
        lane_data = self.ghost_state["ahead"][lane]
        vehicle_ids = list(lane_data.keys())
        if len(vehicle_ids) == 0:
            return None
        lowest_vehicle = lane_data[vehicle_ids[0]]
        for entry in vehicle_ids[1:]:
            current_entry = lane_data[entry]
            if (current_entry["s"] < lowest_vehicle["s"]):
                lowest_vehicle = current_entry
        return lowest_vehicle
    
    def getLowestBehindVisibleVehicle(self, lane):
        lane_data = self.visible_state[lane]
        vehicle_ids = list(lane_data.keys())
        if len(vehicle_ids) == 0:
            return None
        lowest_vehicle = lane_data[vehicle_ids[0]]
        for entry in vehicle_ids[1:]:
            current_entry = lane_data[entry]
            if (current_entry["s"] < lowest_vehicle["s"]):
                lowest_vehicle = current_entry
        return lowest_vehicle
    
    def getHighestAheadVisibleVehicle(self, lane):
        lane_data = self.visible_state[lane]
        vehicle_ids = list(lane_data.keys())
        if len(vehicle_ids) == 0:
            return None
        highest_vehicle = lane_data[vehicle_ids[0]]
        for entry in vehicle_ids[1:]:
            current_entry = lane_data[entry]
            if (current_entry["s"] > highest_vehicle["s"]):
                highest_vehicle = current_entry
        return highest_vehicle

    def loadVisible(self):
        self.visible_state = {}
        for lane in self.getLanes():
            self.visible_state[lane] = {}
        min_timestamp, max_timestamp, min_s, max_s = self.getCurrentVisibleWindow()
        potential_visibles = self.get_lane_dfs(min_timestamp, max_timestamp, min_s, max_s)
        #potential_visibles = self.real_data.queryEdieBoxSubset(min_timestamp, max_timestamp, min_s, max_s, [self.hero_state["id"]])
        for lane in self.getLanes():
            potential_visibles_lane_sorted = potential_visibles[lane].sort_values(by=["time"], ascending=True)
            uniques = list(potential_visibles_lane_sorted["id"].unique())
            for unique in uniques:
                print(self.hero_state["id"], unique)
                if unique != self.hero_state["id"]:
                    unique_vehicle_data = potential_visibles_lane_sorted[potential_visibles_lane_sorted["id"] == unique].copy()
                    unique_vehicle_data["time_delta"] = (unique_vehicle_data["time"] - self.current_timestamp).abs()
                    unique_vehicle_data_sorted = unique_vehicle_data.sort_values(by=["time_delta"], ascending=True)
                    unique_vehicle_data_row = unique_vehicle_data_sorted.iloc[0]
                    #self.visible_state[lane][int(unique)] = self.generateVehicleStateFromRow(unique_vehicle_data_row, lane)
                    candidate = self.generateVehicleStateFromRow(unique_vehicle_data_row, lane)
                    self.registerNewVisibleVehicle(candidate)
    
    def loadGhosts(self, ignore_ids=None):
        # In this version we have no ghost vehicles, and assume the ghost region size is 0 in front and behind.
        # Thus, lets just return an empty list!
        self.ghost_state = {
            "behind": {},
            "ahead": {}
        }
        for lane in self.getLanes():
            self.ghost_state["behind"][lane] = {}
            self.ghost_state["ahead"][lane] = {}

    def updateHeroVehicleViaCARLA(self, new_hero_state):
        hero_state_processed = self.generateUpdatedVehicleStateFromCARLA(new_hero_state)
        hero_state_processed["velocity"] = float(new_hero_state["velocity"])
        self.hero_state = hero_state_processed

    def updateVisibleVehiclesViaCARLA(self, new_visible_states):
        self.visible_state = {}
        for lane in self.getLanes():
            self.visible_state[lane] = {}
        for lane in self.getLanes():
            for vehicle_id in new_visible_states[lane]:
                vehicle_data = new_visible_states[lane][vehicle_id]
                # Did we slip past the behind ghost cell?
                behind_ghost_window = self.getCurrentBehindGhostWindow()
                ahead_ghost_window = self.getCurrentAheadGhostWindow()
                if (vehicle_data["s"] < behind_ghost_window[2]):
                    print(f"WARNING: Threw away {vehicle_data} because it was visible but then slipped behind the behind ghost cell!")
                    #self.vehicles_to_completely_ignore.append(vehicle_id)
                    #continue # Throw away this vehicle from now on
                """
                # Did we slip into the behind ghost cell?
                elif (vehicle_data["s"] < behind_ghost_window[3]):
                    self.registerNewGhostVehicle(vehicle_data)
                # Did we slip into the ahead ghost cell?
                elif (vehicle_data["s"] > ahead_ghost_window[2]):
                    self.registerNewGhostVehicle(vehicle_data)
                else:
                    new_data = self.generateUpdatedVehicleStateFromCARLA(vehicle_data)
                    self.registerNewVisibleVehicle(new_data, True)
                """
                new_data = self.generateUpdatedVehicleStateFromCARLA(vehicle_data)
                self.registerNewVisibleVehicle(new_data, True)

    def updateVisibleVehiclesWithGhostSelection(self, ghost_data):
        visible_window = self.getCurrentVisibleWindow()
        for lane in self.getLanes():
            for id in ghost_data[lane]:
                candidate = ghost_data[lane][id]
                candidate_new_s = candidate["s"] + (candidate["velocity"] * (self.current_timestamp - candidate["time"]))
                print(f"Candidate visible {candidate} which is a ghost has a projected {candidate_new_s} position with this window {visible_window}")
                if (candidate_new_s > visible_window[2]) and (candidate_new_s < visible_window[3]):
                    if (self.checkIfCandidateVisibleNoOverlapWithCurrentVisible(lane, candidate)):
                        candidate["s"] = candidate_new_s
                        self.registerNewVisibleVehicle(candidate)
                    else:
                        print(f"WARNING: Threw away {candidate} visible vehicle because it overlapped with the other visible vehicles!")
                        # No need to remove. Will be dealt with when we reload the ghost data.

    def updateVisibleVehiclesViaGhosts(self):
        # For each ghost vehicle, we will estimate its projected future position with a simple change to s.
        # Then, we will see if they fall under the visible region. If so, attempt to admit them, as long as geometry permits it.
        # Check behind vehicles
        self.updateVisibleVehiclesWithGhostSelection(self.ghost_state["behind"])
        # Check ahead vehicles
        self.updateVisibleVehiclesWithGhostSelection(self.ghost_state["ahead"])

    def checkVehicleBoundingBoxNoOverlap(self, vehicle1, vehicle2):
        vehicle_first = vehicle1 if (vehicle1["s"] < vehicle2["s"]) else vehicle2
        vehicle_second = vehicle1 if (vehicle1["s"] > vehicle2["s"]) else vehicle2 
        vehicle_first_min, vehicle_first_max = vehicle_first["s"], vehicle_first["s"] + vehicle_first["length"]
        vehicle_second_min, vehicle_second_max = vehicle_second["s"], vehicle_second["s"] + vehicle_second["length"]

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
        # Create a 1d bounding box for the lane that covers the backmost visible vehicle to the frontmost one. We cannot breach this.
        lowest_vehicle = self.getLowestBehindVisibleVehicle(lane)
        highest_vehicle = self.getHighestAheadVisibleVehicle(lane)
        if lowest_vehicle is None:
            return True
        result = ((candidate["s"] + candidate["length"]) < lowest_vehicle["s"]) or (candidate["s"] > (highest_vehicle["s"] + highest_vehicle["length"]))
        if not result:
            print(f"WARNING: Threw away {candidate} because it was in an invalid visible position with respect to {lowest_vehicle} and {highest_vehicle}.\n")
        return result
    
    def registerNewVisibleVehicle(self, vehicle_data, ignore_invalid_visible_cell_position=False):
        if vehicle_data["id"] in self.vehicles_to_completely_ignore:
            print(f"WARNING: Threw away visible {vehicle_data} because it was marked as a vehicle to ignore.")
            return
        visible_window = self.getCurrentVisibleWindow()
        # Are we inside?
        if (vehicle_data["s"] > visible_window[2]) and (vehicle_data["s"] < visible_window[3]):
            if ignore_invalid_visible_cell_position or self.checkIfInitVisibleNoOverlapWithCurrentVisible(vehicle_data["lane_id"], vehicle_data):
                self.visible_state[vehicle_data["lane_id"]][vehicle_data["id"]] = vehicle_data
            else:
                print(f"WARNING: Threw away {vehicle_data} because it was in an invalid visible position")
                #self.vehicles_to_completely_ignore.append(vehicle_data["id"]) # Permanently throw away
        else:
            print(f"WARNING: Threw away {vehicle_data} because it wasn't in a valid visible position")
            #self.vehicles_to_completely_ignore.append(vehicle_data["id"]) # Permanently throw away

    def registerNewGhostVehicle(self, vehicle_data, ignore_invalid_ghost_cell_position=False):
        # When we get ghost vehicles in our case, it is solely from cars slipping out.
        # And we have the ghost region be 0 on each side. Thus, we simply destroy any vehicles that come in here.
        # We might have ghost vehicles later though - just so you know.
        pass
