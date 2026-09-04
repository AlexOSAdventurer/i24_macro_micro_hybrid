"""Engine-agnostic base for the microscopic couplers driven by I24MicroSimBridge.

The bridge splits each macroscopic step around the fluid solve and asks the
micro-coupler for one thing on each side of it::

    middle_s = micro_coupler.step()      # before the fluid step
    ...                                  # Simulation advances the continuum
    micro_coupler.poststep()             # after the fluid step

``step`` settles the bubble and hands over a fresh vehicle dictionary via
bridge.update_vehicles, having settled up with the per-lane flux memories
bridge.flow_memory_rear / bridge.flow_memory_front; the mask the bridge then
builds is anchored entirely on the hero state as of that moment.  ``poststep``
advances the micro engine, so the hero state ``step`` reads is always the one
the previous advance produced and never a mid-step extrapolation of it.

Everything needed to honour that contract *except* the microscopic engine
itself lives here:
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
import random

import pandas

from simulation import FundamentalDiagram, GroundTruthStore
from i24_micro_bridge import Vehicle

class I24MicroCouplerBase:
    """Shared machinery for a bridge ``micro_coupler`` backed by a micro engine."""

    # Spawn geometry, calibrated against CARLA in sim_calibration_carla.py.
    #min_spawn_length = 6.8725979813165115
    visible_time_max_difference = 0.1  # Seconds
    ghost_time_max_difference = 1.0  # Seconds
    desired_s_max_difference = 50.0  # Meters
    
    vehicle_spawn_limit = 5.0  # 5 cars per tick allowed
    estimated_vehicle_width = 3.5 # Expected vehicle width on spawn

    vehicle_blueprints_bounds = {
        'vehicle.audi.a2': {
            "xmin": -1.8526612520217896,
            "xmax": 1.8527082204818726,
            "ymin": -0.8940306901931763,
            "ymax": 0.8946478366851807
        },  
        'vehicle.mercedes.coupe_2020': {
            "xmin": -2.339423894882202,
            "xmax": 2.334214925765991,
            "ymin": -0.9059094786643982,
            "ymax": 0.9059030413627625
        },  
        'vehicle.dodge.charger_police': {
            "xmin": -2.46346116065979,
            "xmax": 2.5107829570770264,
            "ymin": -1.0186110734939575,
            "ymax": 1.0197900533676147
        },  
        'vehicle.audi.tt': {
            "xmin": -2.0909781455993652,
            "xmax": 2.0902318954467773,
            "ymin": -0.9970653057098389,
            "ymax": 0.9970518350601196
        },  
        'vehicle.jeep.wrangler_rubicon': {
            "xmin": -1.9328800439834595,
            "xmax": 1.9333406686782837,
            "ymin": -0.9515625834465027,
            "ymax": 0.9536339640617371
        },  
        'vehicle.mini.cooper_s': {
            "xmin": -1.9028857946395874,
            "xmax": 1.9029144048690796,
            "ymin": -0.9852210283279419,
            "ymax": 0.985054612159729
        },  
        'vehicle.mercedes.coupe': {
            "xmin": -2.5134027004241943,
            "xmax": 2.513374090194702,
            "ymin": -1.0765365362167358,
            "ymax": 1.0750097036361694
        },  
        'vehicle.dodge.charger_2020': {
            "xmin": -2.5092527866363525,
            "xmax": 2.498572587966919,
            "ymin": -0.9408262968063354,
            "ymax": 0.9407956600189209
        },  
        'vehicle.ford.ambulance': {
            "xmin": -3.471363067626953,
            "xmax": 2.894279956817627,
            "ymin": -1.1736520528793335,
            "ymax": 1.1775223016738892
        },  
        'vehicle.lincoln.mkz_2020': {
            "xmin": -2.4525094032287598,
            "xmax": 2.4398722648620605,
            "ymin": -0.9183293581008911,
            "ymax": 0.9183839559555054
        },   
        'vehicle.mini.cooper_s_2021': {
            "xmin": -2.307612657546997,
            "xmax": 2.245086431503296,
            "ymin": -1.048503041267395,
            "ymax": 1.0485690832138062
        },  
        'vehicle.ford.crown': {
            "xmin": -2.4974451065063477,
            "xmax": 2.8682336807250977,
            "ymin": -0.9003520607948303,
            "ymax": 0.9003720879554749
        },  
        'vehicle.toyota.prius': {
            "xmin": -2.2548015117645264,
            "xmax": 2.258721113204956,
            "ymin": -1.0037702322006226,
            "ymax": 1.0030442476272583
        },  
        'vehicle.carlamotors.european_hgv': {
            "xmin": -3.9551165103912354,
            "xmax": 3.9805939197540283,
            "ymin": -1.4402295351028442,
            "ymax": 1.45085871219635
        },  
        'vehicle.carlamotors.carlacola': {
            "xmin": -2.601931571960449,
            "xmax": 2.6019067764282227,
            "ymin": -1.3134857416152954,
            "ymax": 1.3135038614273071
        },  
        'vehicle.nissan.patrol_2021': {
            "xmin": -2.7545599937438965,
            "xmax": 2.8112688064575195,
            "ymin": -1.0749708414077759,
            "ymax": 1.0749961137771606
        },  
        'vehicle.dodge.charger_police_2020': {
            "xmin": -2.5092551708221436,
            "xmax": 2.728259325027466,
            "ymin": -0.9648845195770264,
            "ymax": 0.9648749828338623
        },  
        'vehicle.mercedes.sprinter': {
            "xmin": -2.9681336879730225,
            "xmax": 2.947056531906128,
            "ymin": -0.9900767803192139,
            "ymax": 0.9983561038970947
        },  
        'vehicle.audi.etron': {
            "xmin": -2.4314372539520264,
            "xmax": 2.42427134513855,
            "ymin": -1.016370177268982,
            "ymax": 1.0163863897323608
        },  
        'vehicle.volkswagen.t2_2021': {
            "xmin": -2.104430913925171,
            "xmax": 2.3377530574798584,
            "ymin": -0.8876193761825562,
            "ymax": 0.8869459629058838
        },  
        'vehicle.carlamotors.firetruck': {
            "xmin": -4.4874677658081055,
            "xmax": 3.9805736541748047,
            "ymin": -1.440228819847107,
            "ymax": 1.4508594274520874
        },  
        'vehicle.ford.mustang': {
            "xmin": -2.326622724533081,
            "xmax": 2.390902280807495,
            "ymin": -0.9474157094955444,
            "ymax": 0.9474111795425415
        },  
        'vehicle.volkswagen.t2': {
            "xmin": -2.2388687133789062,
            "xmax": 2.241568088531494,
            "ymin": -1.0352046489715576,
            "ymax": 1.0341105461120605
        },  
        'vehicle.mitsubishi.fusorosa': {
            "xmin": -5.581599235534668,
            "xmax": 4.6910858154296875,
            "ymin": -2.0929453372955322,
            "ymax": 1.8512064218521118
        },  
        'vehicle.tesla.model3': {
            "xmin": -2.3666815757751465,
            "xmax": 2.425097942352295,
            "ymin": -1.0817289352416992,
            "ymax": 1.0817210674285889
        },  
        'vehicle.tesla.cybertruck': {
            "xmin": -3.136770248413086,
            "xmax": 3.1367831230163574,
            "ymin": -1.1948214769363403,
            "ymax": 1.19475257396698
        },  
        'vehicle.lincoln.mkz_2017': {
            "xmin": -2.4467976093292236,
            "xmax": 2.454885721206665,
            "ymin": -1.0641733407974243,
            "ymax": 1.0641509294509888
        }, 
        'vehicle.nissan.patrol': {
            "xmin": -2.3596363067626953,
            "xmax": 2.244873523712158,
            "ymin": -0.9656212329864502,
            "ymax": 0.9659717082977295
        },  
        'vehicle.nissan.micra': {
            "xmin": -1.8166637420654297,
            "xmax": 1.8167121410369873,
            "ymin": -0.9216124415397644,
            "ymax": 0.9235013127326965
        }
    }

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
    ) -> None:
        self.motion_data = motion_data
        self.lanes = lanes
        self.dt = float(dt)
        self.fd = fd
        self.bridge = None  # Set by the bridge itself
        self.transition_region_size = 1.0 / fd.rho_c
        self.min_spawn_length = max([self.vehicle_blueprints_bounds[bp]["xmax"] - self.vehicle_blueprints_bounds[bp]["xmin"] for bp in self.vehicle_blueprints_bounds])
        self.average_spawn_length = sum([self.vehicle_blueprints_bounds[bp]["xmax"] - self.vehicle_blueprints_bounds[bp]["xmin"] for bp in self.vehicle_blueprints_bounds]) / len(self.vehicle_blueprints_bounds)
        self.min_spawn_distance = (1.0 / fd.rho_j) - self.average_spawn_length
        self.spawn_threshold = self.min_spawn_length + self.min_spawn_distance
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
        self.load_hero(int(hero_road), desired_time, desired_s)
        self.load_visible()
        self.load_ghosts()

    # ------------------------------------------------------------------
    # Engine interface — implemented by the concrete coupler
    # ------------------------------------------------------------------

    def initialize_engine(self):
        """Bring the microscopic engine up and populate it from the seeded
        hero/visible state.  Called once, lazily, on the first step()."""
        raise NotImplementedError

    def sync_engine_with_visible_state(self):
        """Reconcile the engine's vehicle population with self.visible_state:
        insert whatever the macro flux just spawned, remove whatever left."""
        raise NotImplementedError

    def advance_engine(self, dt):
        """Run the engine forward by dt seconds.

        Returns (hero_state, visible_states) where hero_state is a
        vehicle-state dict and visible_states maps lane_id -> {id: state}
        using each vehicle's *current* lane, so a lane change inside the engine
        shows up as a move between the returned buckets.
        """
        raise NotImplementedError

    def destroy_engine(self):
        """Tear the engine down.  Called by the bridge when the bubble retires."""
        raise NotImplementedError

    def initialize(self):
        self.initialize_engine()
        self.initialized = True

    def destroy(self):
        self.destroy_engine()

    # ------------------------------------------------------------------
    # Bridge callback
    # ------------------------------------------------------------------

    def step(self) -> float:
        """The pre-fluid half of a macroscopic step.  Returns the window centre.

        Nothing here advances the engine, so the mask the bridge is about to
        build sees one consistent snapshot -- the hero state left by the previous
        poststep().  ``middle_s``, ``bridge.anchor_speed`` and the coordinate
        origin used by collate_visible_and_hero_vehicles all read the same
        ``self.hero_state``, which is what SimplifiedSimBridge gets out of its
        single ``self.vehicles[self.ego_id]`` read.
        """
        if not self.initialized:
            # The engine comes up already populated from the seeded hero/visible
            # state, so the first mask is a real one rather than an empty bubble
            # anchored to a stationary (anchor_speed == 0) frame.
            self.initialize()
        else:
            self.inject_vehicles_from_macro()
            self.retire_departed_vehicles()
            self.sync_engine_with_visible_state()

        self.bridge.update_vehicles(self.collate_visible_and_hero_vehicles(self.bridge))
        self.bridge.anchor_speed = self.hero_state["velocity"]
        return self.hero_state["s"]

    def poststep(self) -> None:
        """The post-fluid half: advance the micro engine and adopt its state.

        Running after the fluid step means the macro state the engine consults
        mid-advance (apply_lead_vehicle_speeds) is the state that step just
        produced -- the same phase at which the simplified bridge's
        car-following model queries the macro cells from its own poststep.

        The bridge reads its flux memories off the mask *before* calling this so
        that any credit_lost_vehicle the engine raises during the advance lands
        on top of that read instead of being overwritten by it.
        """
        new_hero_state, new_visible_states = self.advance_engine(self.dt)
        self.update_hero_vehicle_from_engine(new_hero_state)
        self.adopt_visible_vehicles_from_engine(new_visible_states)
        self.current_timestamp += self.dt

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

    def get_vehicle_trajectory_from_real(self, id, lane):
        df = self.motion_data.micro_df
        window = df[(df["id"] == id) & (df["lane_id"] == lane)]
        return window

    def estimate_vehicle_velocity_from_real(self, id, lane, current_time):
        trajectory = self.get_vehicle_trajectory_from_real(id, lane)
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

    def estimate_ghost_vehicle_velocity(self, vehicle_data):
        # Replay its velocity for as long as the trajectory data lasts; once that
        # expires the ghost is discarded and whatever replaces it is used instead.
        return max(min(self.estimate_vehicle_velocity_from_real(
            vehicle_data["id"], vehicle_data["lane_id"], self.current_timestamp
        ), self.fd.v_f), 0.0)

    # ------------------------------------------------------------------
    # Vehicle state construction
    # ------------------------------------------------------------------

    def generate_next_vehicle_id(self):
        new_id = self.next_vehicle_id
        self.next_vehicle_id += 1
        return new_id

    '''
        Obtain vehicle blueprints that are smaller or equal to length in length.
    '''
    def get_eligible_vehicle_blueprints(self, length: float):
        blueprints = []
        for blueprint_key in self.vehicle_blueprints_bounds:
            blueprint = self.vehicle_blueprints_bounds[blueprint_key]
            current_length = blueprint["xmax"] - blueprint["xmin"]
            if (current_length <= length):
                blueprints.append(blueprint_key)
        return blueprints

    '''
        Obtain vehicle blueprint that is closest in length while being smaller or equal to it in length.
    '''
    def get_closest_vehicle_blueprint(self, length: float):
        selected_blueprint = None
        selected_blueprint_key = None
        for blueprint_key in self.vehicle_blueprints_bounds:
            blueprint = self.vehicle_blueprints_bounds[blueprint_key]
            current_length = blueprint["xmax"] - blueprint["xmin"]
            if ((selected_blueprint is None) or ((selected_blueprint["xmax"] - selected_blueprint["xmin"]) < current_length)) and (current_length <= length):
                selected_blueprint = blueprint
                selected_blueprint_key = blueprint_key
        if (selected_blueprint_key is None):
            # This happens if the length is far too small. Simply select the smallest possible one.
            for blueprint_key in self.vehicle_blueprints_bounds:
                blueprint = self.vehicle_blueprints_bounds[blueprint_key]
                current_length = blueprint["xmax"] - blueprint["xmin"]
                if ((selected_blueprint is None) or ((selected_blueprint["xmax"] - selected_blueprint["xmin"]) > current_length)):
                    selected_blueprint = blueprint
                    selected_blueprint_key = blueprint_key
        return selected_blueprint_key

    def generate_vehicle_state_from_row(self, row, lane, current_timestamp=None):
        if current_timestamp is None:
            current_timestamp = self.current_timestamp
        estimated_velocity = self.estimate_vehicle_velocity_from_real(
            int(row["id"]), lane, float(row["time"])
        )
        estimated_velocity = max(min(estimated_velocity, self.fd.v_f), 0.0)
        estimated_s = float(row["s"]) + ((current_timestamp - float(row["time"])) * estimated_velocity)
        selected_blueprint = self.vehicle_blueprints_bounds[self.get_closest_vehicle_blueprint(float(row["length"]))]
        new_length = selected_blueprint["xmax"] - selected_blueprint["xmin"] if (selected_blueprint is not None) else float(row["length"])
        print(new_length)
        return {
            "id": self.generate_next_vehicle_id(),
            "class": str(row["class"]),
            "length": new_length,
            "width": self.estimated_vehicle_width,
            "time": current_timestamp,
            "s": estimated_s,
            "t": float(row["t"]),
            "velocity": estimated_velocity,
            "lane_id": int(lane),
            "road_id": self.hero_road,
        }

    # behind_or_in_front is either "behind" or "front"
    def generate_vehicle_state_from_spawn(self, lane, s_min, s_max, behind_or_in_front="behind"):
        if (behind_or_in_front != "behind") and (behind_or_in_front != "front"):
            return None  # Force failure upstream.
        new_time = self.current_timestamp
        estimated_velocity = (
            self.get_behind_lane_velocity_micro(lane, 0.0)
            if (behind_or_in_front == "behind")
            else self.get_ahead_lane_velocity_micro(lane, 0.0)
        )
        length = float(s_max - s_min)
        eligible_blueprints = self.get_eligible_vehicle_blueprints(length)
        new_length = length
        if (len(eligible_blueprints) > 0):
            blueprint = self.vehicle_blueprints_bounds[random.choice(eligible_blueprints)]
            new_length = blueprint["xmax"] - blueprint["xmin"]
        print(new_length)
        return {
            "id": self.generate_next_vehicle_id(),
            "class": "spawned",
            "length": new_length,
            "width": self.estimated_vehicle_width,
            "time": new_time,
            "s": float(s_min),
            "t": (lane * self.estimated_vehicle_width) + (self.estimated_vehicle_width / 2),
            "velocity": estimated_velocity,
            "lane_id": lane,
            "road_id": self.hero_road,
        }

    def generate_updated_vehicle_state_from_engine(self, vehicle_data):
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

    def load_hero(self, hero_road, desired_time, desired_s):
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
        original_hero_state = self.generate_vehicle_state_from_row(
            selected_hero, hero_lane, float(selected_hero["time"])
        )
        self.hero_state = original_hero_state
        self.hero_lane = hero_lane
        self.current_timestamp = original_hero_state["time"]
        self.anchor_speed = self.hero_state["velocity"]

    def load_visible(self):
        self.visible_state = {}
        for lane in self.get_lanes():
            self.visible_state[lane] = {}
        min_timestamp, max_timestamp, min_s, max_s = self.get_current_visible_window()
        potential_visibles = self.get_lane_dfs(min_timestamp, max_timestamp, min_s, max_s)
        for lane in self.get_lanes():
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
                    candidate = self.generate_vehicle_state_from_row(unique_vehicle_data_row, lane)
                    self.register_new_visible_vehicle(candidate)

    def load_ghosts(self, ignore_ids=None):
        # The ghost region is 0 m on each side in the current configuration, so
        # there is nothing to load. Kept for symmetry with the visible state.
        self.ghost_state = {"behind": {}, "ahead": {}}
        for lane in self.get_lanes():
            self.ghost_state["behind"][lane] = {}
            self.ghost_state["ahead"][lane] = {}

    # ------------------------------------------------------------------
    # Windows and accessors
    # ------------------------------------------------------------------

    def get_current_visible_window(self):
        return (
            self.current_timestamp - self.visible_time_max_difference,
            self.current_timestamp + self.visible_time_max_difference,
            self.hero_state["s"] - self.visible_window,
            self.hero_state["s"] + self.visible_window,
        )

    def get_current_behind_ghost_window(self):
        return (
            self.current_timestamp - self.ghost_time_max_difference,
            self.current_timestamp + self.ghost_time_max_difference,
            self.hero_state["s"] - self.visible_window - self.ghost_window,
            self.hero_state["s"] - self.visible_window,
        )

    def get_current_ahead_ghost_window(self):
        return (
            self.current_timestamp - self.ghost_time_max_difference,
            self.current_timestamp + self.ghost_time_max_difference,
            self.hero_state["s"] + self.visible_window,
            self.hero_state["s"] + self.visible_window + self.ghost_window,
        )

    def get_lanes(self):
        return self.lanes

    def get_visible_ids(self):
        result = {}
        for lane in self.get_lanes():
            result[lane] = [id for id in self.visible_state[lane]]
        return result

    def get_visible_ids_flat(self):
        visible_ids = self.get_visible_ids()
        return sum([visible_ids[lane] for lane in self.get_lanes()], [])

    def get_ghost_ids(self):
        result = {"behind": {}, "ahead": {}}
        for position in self.ghost_state:
            for lane in self.get_lanes():
                result[position][lane] = [id for id in self.ghost_state[position][lane]]
        return result

    def get_visible_data(self):
        return self.visible_state

    def get_ghost_data(self):
        return self.ghost_state

    def get_hero_data(self):
        return self.hero_state

    # This is meant for the upper level fluid simulator, which wants string road
    # ids and bubble-relative positions.
    def collate_visible_and_hero_vehicles(self, bridge):
        result = {}
        visible_data = self.get_visible_data()
        hero_data = self.get_hero_data()
        min_timestamp, max_timestamp, min_s, max_s = self.get_current_visible_window()
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

    def get_lowest_behind_ghost_vehicle(self, lane):
        return self._extremum(self.ghost_state["behind"][lane], None, lowest=True)

    def get_highest_ahead_ghost_vehicle(self, lane):
        return self._extremum(self.ghost_state["ahead"][lane], None, lowest=False)

    def get_lowest_ahead_ghost_vehicle(self, lane):
        return self._extremum(self.ghost_state["ahead"][lane], None, lowest=True)

    def get_lowest_behind_visible_vehicle(self, lane):
        return self._extremum(self.visible_state[lane], self.hero_state if self.hero_state["lane_id"] == lane else None, lowest=True)

    def get_highest_ahead_visible_vehicle(self, lane):
        return self._extremum(self.visible_state[lane], self.hero_state if self.hero_state["lane_id"] == lane else None, lowest=False)

    @staticmethod
    def _extremum(lane_data, hero_vehicle, lowest):
        vehicle_ids = list(lane_data.keys())
        selected = None
        if len(vehicle_ids) > 0:
            selected = lane_data[vehicle_ids[0]]
            for entry in vehicle_ids[1:]:
                current_entry = lane_data[entry]
                if (current_entry["s"] < selected["s"]) if lowest else (current_entry["s"] > selected["s"]):
                    selected = current_entry
        if (hero_vehicle is not None):
            if (selected is None) or ((hero_vehicle["s"] < selected["s"]) if lowest else (hero_vehicle["s"] > selected["s"])):
                selected = hero_vehicle
            
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

    def get_ahead_lane_velocity_macro(self, lane_id, default_speed, min_cell_size=25.0):
        cell = self._neighbor_cell(lane_id, "ahead")
        if cell is None:
            return default_speed
        return cell.fd.velocity_from_density(cell.mass / (cell.end_s - cell.start_s))

    def get_behind_lane_velocity_macro(self, lane_id, default_speed):
        cell = self._neighbor_cell(lane_id, "behind")
        if cell is None:
            return default_speed
        return cell.fd.velocity_from_density(cell.mass / (cell.end_s - cell.start_s))

    def get_ahead_lane_density(self, lane_id, default_density):
        cell = self._neighbor_cell(lane_id, "ahead")
        if cell is None:
            return default_density
        return cell.mass / (cell.end_s - cell.start_s)

    def get_behind_lane_density(self, lane_id, default_density):
        cell = self._neighbor_cell(lane_id, "behind")
        if cell is None:
            return default_density
        return cell.mass / (cell.end_s - cell.start_s)

    def get_ahead_lane_velocity_micro(self, lane_id, default_speed, min_cell_size=25.0):
        front_most_vehicle = self.get_highest_ahead_visible_vehicle(lane_id)
        if front_most_vehicle is None:
            return self.get_ahead_lane_velocity_macro(lane_id, default_speed, min_cell_size)
        return front_most_vehicle["velocity"]

    def get_behind_lane_velocity_micro(self, lane_id, default_speed, min_cell_size=25.0):
        rear_most_vehicle = self.get_lowest_behind_visible_vehicle(lane_id)
        if rear_most_vehicle is None:
            return self.get_behind_lane_velocity_macro(lane_id, default_speed)
        return rear_most_vehicle["velocity"]

    # ------------------------------------------------------------------
    # Macro -> micro injection
    # ------------------------------------------------------------------

    # behind_or_in_front is either "behind" or "front"
    def _create_vehicle_spawns_in_s_range(
        self, new_visible_states, lane, vehicle_count, s_min, s_max, behind_or_in_front, toprint=False
    ):
        density = (
            self.get_behind_lane_density(lane, 0.001)
            if (behind_or_in_front == "behind")
            else self.get_ahead_lane_density(lane, 0.001)
        )
        estimated_meters_per_vehicle = (s_max - s_min) / vehicle_count
        spawn_lengths = max(min(estimated_meters_per_vehicle, self.transition_region_size - 1e-5), self.min_spawn_length + self.min_spawn_distance)
        vehicle_count = min(math.floor((s_max - s_min) / spawn_lengths), vehicle_count)
        if behind_or_in_front == "behind":
            s_min = s_max - (spawn_lengths * vehicle_count)
        else:
            s_max = s_min + (spawn_lengths * vehicle_count)
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

            new_vehicle_data = self.generate_vehicle_state_from_spawn(
                lane, start_position_calculated, end_position_calculated, behind_or_in_front
            )
            new_visible_states[lane][new_vehicle_data["id"]] = new_vehicle_data
        return new_visible_states, vehicle_count

    def _spawn_vehicles_in_rear(self, new_visible_states, lane, s_availability):
        rear_flux_memory = self.bridge.flow_memory_rear[lane]
        visible_window = self.get_current_visible_window()
        if rear_flux_memory > 0.0:
            vehicle_count = min(math.floor(rear_flux_memory), self.vehicle_spawn_limit)
            if vehicle_count > 0:
                new_visible_states, vehicle_count = self._create_vehicle_spawns_in_s_range(
                    new_visible_states,
                    lane,
                    vehicle_count,
                    visible_window[2],
                    visible_window[2] + s_availability,
                    "behind",
                )
                self.bridge.flow_memory_rear[lane] -= vehicle_count
        return new_visible_states

    def _spawn_vehicles_in_front(self, new_visible_states, lane, s_availability):
        front_flux_memory = self.bridge.flow_memory_front[lane]
        visible_window = self.get_current_visible_window()
        if front_flux_memory < 0.0:
            vehicle_count = min(math.floor(-front_flux_memory), self.vehicle_spawn_limit)
            if vehicle_count > 0:
                new_visible_states, vehicle_count = self._create_vehicle_spawns_in_s_range(
                    new_visible_states,
                    lane,
                    vehicle_count,
                    visible_window[3] - s_availability,
                    visible_window[3],
                    "front",
                )
                self.bridge.flow_memory_front[lane] += vehicle_count
        return new_visible_states

    def inject_vehicles_from_macro(self):
        visible_window = self.get_current_visible_window()
        for lane in self.get_lanes():
            # Rear boundary: how much clear road is there behind the rearmost car?
            rear_vehicle = self.get_lowest_behind_visible_vehicle(lane)
            if (rear_vehicle is None):
                s_availability = visible_window[3] - visible_window[2]
            else:
                s_availability = rear_vehicle["s"] - visible_window[2]

            s_availability = min(s_availability, self.transition_region_size)
            if s_availability > self.spawn_threshold:
                self.visible_state = self._spawn_vehicles_in_rear(self.visible_state, lane, s_availability)

            # Front boundary: same, ahead of the frontmost car.
            front_vehicle = self.get_highest_ahead_visible_vehicle(lane)
            if (front_vehicle is None):
                s_availability = visible_window[3] - visible_window[2]
            else:
                s_availability = visible_window[3] - front_vehicle["s"] - front_vehicle["length"]

            s_availability = min(s_availability, self.transition_region_size)
            if s_availability > self.spawn_threshold:
                self.visible_state = self._spawn_vehicles_in_front(self.visible_state, lane, s_availability)

    # ------------------------------------------------------------------
    # Micro -> macro readback
    # ------------------------------------------------------------------

    def update_hero_vehicle_from_engine(self, new_hero_state):
        hero_state_processed = self.generate_updated_vehicle_state_from_engine(new_hero_state)
        hero_state_processed["velocity"] = float(new_hero_state["velocity"])
        self.hero_state = hero_state_processed

    def adopt_visible_vehicles_from_engine(self, new_visible_states):
        """Rebuild ``visible_state`` from what the engine reports.

        ``new_visible_states`` is keyed by each vehicle's *current* lane, so a
        vehicle that changed lanes inside the engine is credited to the lane it
        ended up in.  That lets per-lane flux memory disagree with the macro
        per-lane density -- which is expected and permitted -- while the total
        over all lanes still balances.

        No window test happens here; this adopts the engine's truth wholesale,
        the counterpart of SimplifiedSimBridge.move_vehicles committing new
        positions.  Vehicles that have left the window are retired on the far
        side of the fluid step, in retire_departed_vehicles.
        """
        self.visible_state = {}
        for lane in self.get_lanes():
            self.visible_state[lane] = {}
        for lane in self.get_lanes():
            for vehicle_id in new_visible_states[lane]:
                vehicle_data = new_visible_states[lane][vehicle_id]
                if vehicle_data["id"] in self.vehicles_to_completely_ignore:
                    print(f"WARNING: Threw away visible {vehicle_data} because it was marked as a vehicle to ignore.")
                    continue
                new_data = self.generate_updated_vehicle_state_from_engine(vehicle_data)
                self.visible_state[lane][new_data["id"]] = new_data

    def retire_departed_vehicles(self):
        """Drop vehicles that have left the visible window, returning their mass
        to the macro flux memory.

        Called *after* inject_vehicles_from_macro, matching
        SimplifiedSimBridge.remove_vehicles_from_macro: a vehicle that departed
        during the last advance is still in visible_state while injection
        measures the room at each boundary, so the boundary does not spawn into
        the gap that vehicle left in the same step it left it.
        """
        visible_window = self.get_current_visible_window()
        for lane in self.get_lanes():
            departed = []
            for vehicle_id in self.visible_state[lane]:
                vehicle_data = self.visible_state[lane][vehicle_id]
                if vehicle_data["s"] < visible_window[2]:
                    departed.append(vehicle_id)
                    self.bridge.flow_memory_rear[lane] += 1
                elif vehicle_data["s"] > visible_window[3]:
                    departed.append(vehicle_id)
                    self.bridge.flow_memory_front[lane] -= 1
            for vehicle_id in departed:
                self.visible_state[lane].pop(vehicle_id, None)

    def credit_lost_vehicle(self, lane_id, side):
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

    def update_visible_vehicles_with_ghost_selection(self, ghost_data):
        visible_window = self.get_current_visible_window()
        for lane in self.get_lanes():
            for id in ghost_data[lane]:
                candidate = ghost_data[lane][id]
                candidate_new_s = candidate["s"] + (
                    candidate["velocity"] * (self.current_timestamp - candidate["time"])
                )
                if (candidate_new_s > visible_window[2]) and (candidate_new_s < visible_window[3]):
                    if self.check_if_candidate_visible_no_overlap_with_current_visible(lane, candidate):
                        candidate["s"] = candidate_new_s
                        self.register_new_visible_vehicle(candidate)

    def update_visible_vehicles_via_ghosts(self):
        self.update_visible_vehicles_with_ghost_selection(self.ghost_state["behind"])
        self.update_visible_vehicles_with_ghost_selection(self.ghost_state["ahead"])

    # ------------------------------------------------------------------
    # Geometry checks and registration
    # ------------------------------------------------------------------

    def check_vehicle_bounding_box_no_overlap(self, vehicle1, vehicle2):
        vehicle_first = vehicle1 if (vehicle1["s"] < vehicle2["s"]) else vehicle2
        vehicle_second = vehicle1 if (vehicle1["s"] > vehicle2["s"]) else vehicle2
        vehicle_first_min = vehicle_first["s"]
        vehicle_first_max = vehicle_first["s"] + vehicle_first["length"] + self.min_spawn_distance
        vehicle_second_min = vehicle_second["s"]

        return (vehicle_first_min < vehicle_second_min) and (vehicle_first_max < vehicle_second_min)

    def check_if_candidate_ghost_no_overlap_with_current_ghosts(self, ghost_data, candidate):
        for id in ghost_data:
            if not self.check_vehicle_bounding_box_no_overlap(ghost_data[id], candidate):
                return False
        return True

    def check_if_init_visible_no_overlap_with_current_visible(self, lane, candidate):
        for id in self.visible_state[lane]:
            if not self.check_vehicle_bounding_box_no_overlap(self.visible_state[lane][id], candidate):
                return False
        return self.check_vehicle_bounding_box_no_overlap(self.hero_state, candidate) if (self.hero_state is not None) and (self.hero_state["lane_id"] == lane) else True

    def check_if_candidate_visible_no_overlap_with_current_visible(self, lane, candidate):
        # 1d bounding box over the lane, from the backmost visible vehicle to the
        # frontmost one. A candidate may not breach it.
        lowest_vehicle = self.get_lowest_behind_visible_vehicle(lane)
        highest_vehicle = self.get_highest_ahead_visible_vehicle(lane)
        if lowest_vehicle is None:
            return True
        return ((candidate["s"] + candidate["length"]) < lowest_vehicle["s"]) or (
            candidate["s"] > (highest_vehicle["s"] + highest_vehicle["length"])
        )

    def register_new_visible_vehicle(self, vehicle_data, ignore_invalid_visible_cell_position=False):
        if vehicle_data["id"] in self.vehicles_to_completely_ignore:
            print(f"WARNING: Threw away visible {vehicle_data} because it was marked as a vehicle to ignore.")
            return
        visible_window = self.get_current_visible_window()
        if (vehicle_data["s"] >= visible_window[2]) and (vehicle_data["s"] <= visible_window[3]):
            if ignore_invalid_visible_cell_position or self.check_if_init_visible_no_overlap_with_current_visible(
                vehicle_data["lane_id"], vehicle_data
            ):
                self.visible_state[vehicle_data["lane_id"]][vehicle_data["id"]] = vehicle_data

    def register_new_ghost_vehicle(self, vehicle_data, ignore_invalid_ghost_cell_position=False):
        # With a 0 m ghost region on each side there is nothing to hold onto, so
        # vehicles that slip out are simply dropped here.
        pass
