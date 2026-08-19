"""SUMO engine wrapper for the hybrid macro/micro bridge.

This is the SUMO counterpart of ``I24MotionCarlaSimulationCoupled``: it owns the
microscopic engine and speaks the coupler's vehicle-state dictionaries, while
``I24SumoCoupler`` owns the macro-facing bookkeeping.

Coordinate contract, verified against the converted network
(see ``sumo/verify_i24_network.py``):

  * OpenDRIVE road ``N`` is SUMO edge ``"-N"``, and SUMO lane position along
    that edge equals OpenDRIVE ``s`` -- same origin, same direction.
  * ``sumo_lane_index = n_lanes + opendrive_lane_id``, so lane ``-1`` (innermost)
    is index 3 and lane ``-4`` is index 0.
  * The coupler's ``s`` is the vehicle's **rear** bumper, matching the CARLA
    path; SUMO's ``getLanePosition`` is the **front** bumper, so the two differ
    by the vehicle length.

Vehicles run on SUMO's own car-following and lane-change models.  The coupler
influences them in exactly two places: it inserts and removes vehicles at the
bubble boundaries, and it pushes the downstream macroscopic velocity onto the
leading vehicle of each lane with ``setSpeed``.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_NET_FILE = os.path.join(HERE, "sumo", "i24_corridor.net.xml")
DEFAULT_ADDITIONAL_FILE = os.path.join(HERE, "sumo", "i24_corridor_coupled.add.xml")


class HeroPolicy:
    """Optional external controller for the hero (anchor) vehicle.

    With no policy the hero is an ordinary SUMO vehicle: its speed comes from
    the car-following model, and the bubble's motion is therefore emergent.
    Supplying a policy lets an arbitrary controller drive it instead -- an RL
    agent, an MPC, or a replay of the empirical trajectory.

    ``act`` receives an observation dict and returns either ``None`` (leave the
    hero on SUMO's car-following model this step) or a dict with any of:

      ``speed``         target speed in m/s; ``-1`` hands control back to SUMO
      ``acceleration``  desired acceleration in m/s^2, applied over one substep
      ``lane``          target SUMO lane index to change to
      ``lane_duration`` seconds to hold that lane (default: one substep)
    """

    def reset(self, engine) -> None:
        pass

    def act(self, observation: dict) -> Optional[dict]:
        return None


class I24MotionSumoSimulationCoupled:
    """Drives a SUMO instance over TraCI on behalf of an I24 micro coupler."""

    # A SUMO vehicle shorter than this is not physically meaningful; empirical
    # lengths are honoured exactly above it.
    min_vehicle_length = 1.0
    min_vehicle_width = 0.5

    # Substeps a freshly inserted vehicle is held at its spawn speed before
    # being released to the car-following model. Mirrors CARLA's spawn cooldown.
    spawn_speed_hold_substeps = 1

    # Substeps a vehicle may go unseen after insertion before it is written off
    # as an insertion failure and its mass refunded to the macro flux memory.
    insertion_grace_substeps = 2

    def __init__(
        self,
        coupler,
        mapping,
        net_file: str = DEFAULT_NET_FILE,
        additional_file: Optional[str] = DEFAULT_ADDITIONAL_FILE,
        vehicle_type: str = "car",
        step_length: float = 0.1,
        use_libsumo: bool = False,
        gui: bool = False,
        seed: Optional[int] = None,
        end_time: float = 1e6,
        lead_speed_control: bool = True,
        hero_policy: Optional[HeroPolicy] = None,
        label: str = "i24_bridge",
        extra_args: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> None:
        self.coupler = coupler
        self.mapping = mapping  # config["road_data"]
        self.net_file = net_file
        self.additional_file = additional_file
        self.vehicle_type = vehicle_type
        self.step_length = float(step_length)
        self.use_libsumo = use_libsumo
        self.gui = gui
        self.seed = seed
        self.end_time = float(end_time)
        self.lead_speed_control = lead_speed_control
        self.hero_policy = hero_policy
        self.label = label
        self.extra_args = list(extra_args) if extra_args else []
        self.verbose = verbose

        self.conn = None
        self.started = False
        self.current_timestamp = None

        # cosim id -> tracking record
        self.visible_states: Dict[int, dict] = {}
        self.hero_state: Optional[dict] = None
        self.route_ids: Dict[str, str] = {}
        self.lead_controlled: Dict[int, bool] = {}

        self._traci = None
        self._sumolib = None

    # ------------------------------------------------------------------
    # Identifiers and geometry
    # ------------------------------------------------------------------

    def edgeID(self, road_id) -> str:
        return "-" + str(road_id)

    def roadIDFromEdge(self, edge_id: str) -> Optional[str]:
        if not edge_id.startswith("-"):
            return None
        return edge_id[1:]

    def laneCount(self, road_id) -> int:
        return int(self.mapping[str(road_id)]["lanes"])

    def laneIndex(self, road_id, lane_id) -> int:
        """OpenDRIVE lane id -> SUMO lane index. Lane -1 is the innermost."""
        return self.laneCount(road_id) + int(lane_id)

    def openDriveLane(self, road_id, lane_index) -> int:
        return int(lane_index) - self.laneCount(road_id)

    def laneID(self, road_id, lane_id) -> str:
        return f"{self.edgeID(road_id)}_{self.laneIndex(road_id, lane_id)}"

    def vehicleID(self, cosim_id) -> str:
        return f"v{int(cosim_id)}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _importSumo(self):
        if self.use_libsumo:
            import libsumo

            self._traci = libsumo
        else:
            import traci

            self._traci = traci
        import sumolib

        self._sumolib = sumolib

    def _binary(self) -> str:
        name = "sumo-gui" if self.gui else "sumo"
        try:
            return self._sumolib.checkBinary(name)
        except Exception:
            return name

    def _buildCommand(self) -> List[str]:
        command = [
            self._binary(),
            "-n",
            self.net_file,
            "--step-length",
            str(self.step_length),
            "--step-method.ballistic",
            "--begin",
            "0",
            "--end",
            str(self.end_time),
            # The coupler owns vehicle removal; SUMO must not quietly teleport or
            # delete vehicles out from under the flux bookkeeping.
            "--time-to-teleport",
            "-1",
            "--collision.action",
            "warn",
            "--no-step-log",
            "true",
        ]
        if self.additional_file:
            command += ["--additional-files", self.additional_file]
        if self.seed is not None:
            command += ["--seed", str(int(self.seed))]
        if not self.verbose:
            command += ["--no-warnings", "true"]
        command += self.extra_args
        return command

    def _registerRoutes(self):
        """One route per coupled road: the mainline edge plus whatever follows it.

        The bubble never reaches the end of the mainline in the configured runs,
        but giving vehicles a downstream continuation keeps SUMO from treating
        them as about to arrive.
        """
        net = self._sumolib.net.readNet(self.net_file)
        for road_id in self.mapping:
            edge_id = self.edgeID(road_id)
            try:
                edge = net.getEdge(edge_id)
            except KeyError:
                continue
            edges = [edge_id] + [e.getID() for e in edge.getOutgoing()][:1]
            route_id = f"route_{road_id}"
            self.conn.route.add(route_id, edges)
            self.route_ids[str(road_id)] = route_id

    def connectToHost(self):
        self._importSumo()
        command = self._buildCommand()
        if self.use_libsumo:
            self._traci.start(command)
            self.conn = self._traci
        else:
            self._traci.start(command, label=self.label)
            self.conn = self._traci.getConnection(self.label)
        self.started = True

    def initializeSimulation(self):
        self.connectToHost()
        self._registerRoutes()
        self.current_timestamp = self.coupler.current_timestamp
        if self.hero_policy is not None:
            self.hero_policy.reset(self)
        self.spawnHeroVehicle()
        self.spawnAndDespawnVisibleVehiclesFromCoSIM()
        # Deliberately no simulationStep() here. Stepping now would advance every
        # vehicle by one substep that the coupler's clock never sees, putting the
        # seeded positions permanently out of step with self.current_timestamp.
        # The first runSimulationOver() steps before it reads anything back, so
        # the forced insertions are in the network by the time they are sampled.

    def destroySimulation(self):
        if not self.started:
            return
        try:
            self.conn.close()
        except Exception as exc:  # pragma: no cover - teardown must not raise
            print(f"WARNING: SUMO connection did not close cleanly: {exc}")
        finally:
            self.started = False
            self.conn = None

    # ------------------------------------------------------------------
    # Spawning and despawning
    # ------------------------------------------------------------------

    def _trackingRecord(self, cosim_data, veh_id, spawn_side):
        return {
            "sumo_id": veh_id,
            "cosim_data": cosim_data,
            "spawn_side": spawn_side,
            "seen": False,
            "age_substeps": 0,
            "hold_substeps": self.spawn_speed_hold_substeps,
            "last_s": float(cosim_data["s"]),
            "last_lane_id": int(cosim_data["lane_id"]),
        }

    def _spawnSideFor(self, cosim_data) -> str:
        """Which bubble boundary this vehicle entered through.

        Used to refund the right flux memory if the insertion turns out to have
        failed, so a rejected spawn exactly undoes the decrement that paid for it.
        """
        window = self.coupler.getCurrentVisibleWindow()
        centre = 0.5 * (window[2] + window[3])
        return "rear" if float(cosim_data["s"]) < centre else "front"

    def spawnVehicleFromCoSIM(self, cosim_data) -> Optional[str]:
        road_id = str(cosim_data["road_id"])
        route_id = self.route_ids.get(road_id)
        if route_id is None:
            print(f"WARNING: no SUMO route registered for road {road_id}; cannot spawn {cosim_data['id']}")
            return None

        veh_id = self.vehicleID(cosim_data["id"])
        lane_index = self.laneIndex(road_id, cosim_data["lane_id"])
        lane_id = self.laneID(road_id, cosim_data["lane_id"])
        length = max(float(cosim_data["length"]), self.min_vehicle_length)
        width = max(float(cosim_data["width"]), self.min_vehicle_width)
        speed = max(0.0, float(cosim_data["velocity"]))
        # cosim s is the rear bumper; SUMO positions the front bumper.
        front_pos = float(cosim_data["s"]) + length

        try:
            self.conn.vehicle.add(
                veh_id,
                route_id,
                typeID=self.vehicle_type,
                depart="now",
                departLane=str(lane_index),
                departPos=str(front_pos),
                departSpeed=str(speed),
            )
        except Exception as exc:
            print(f"WARNING: SUMO rejected vehicle.add for {veh_id}: {exc}")
            return None

        # Honour the empirical geometry exactly rather than quantising it to a
        # fixed vehicle model, which the CARLA path has to do.
        try:
            self.conn.vehicle.setLength(veh_id, length)
            self.conn.vehicle.setWidth(veh_id, width)
            # Force the vehicle in at the requested position. Without this SUMO
            # would defer to its own insertion check and silently drop it when
            # the gap looks tight, which the spawn geometry has already sized.
            self.conn.vehicle.moveTo(veh_id, lane_id, front_pos)
            self.conn.vehicle.setSpeed(veh_id, speed)
        except Exception as exc:
            print(f"WARNING: could not place {veh_id} at {lane_id}@{front_pos:.2f}: {exc}")
        return veh_id

    def spawnHeroVehicle(self):
        cosim_data = self.coupler.getHeroData()
        veh_id = self.spawnVehicleFromCoSIM(cosim_data)
        if veh_id is None:
            raise RuntimeError(f"Failed to insert the hero vehicle into SUMO: {cosim_data}")
        self.hero_state = self._trackingRecord(cosim_data, veh_id, self._spawnSideFor(cosim_data))
        try:
            self.conn.vehicle.setColor(veh_id, (255, 0, 0, 255))
        except Exception:
            pass

    def spawnVisibleVehicle(self, cosim_data):
        veh_id = self.spawnVehicleFromCoSIM(cosim_data)
        if veh_id is None:
            # Nothing entered the engine, so give the mass straight back.
            self.coupler.creditLostVehicle(int(cosim_data["lane_id"]), self._spawnSideFor(cosim_data))
            return
        self.visible_states[cosim_data["id"]] = self._trackingRecord(
            cosim_data, veh_id, self._spawnSideFor(cosim_data)
        )

    def despawnVehicle(self, record):
        try:
            self.conn.vehicle.remove(record["sumo_id"])
        except Exception:
            pass  # Already gone from SUMO's side.

    def despawnVisibleVehicle(self, record):
        self.despawnVehicle(record)
        self.visible_states.pop(record["cosim_data"]["id"], None)
        self.lead_controlled.pop(record["cosim_data"]["id"], None)

    def resetVisibleStates(self):
        self.visible_states = {}

    def spawnAndDespawnVisibleVehiclesFromCoSIM(self):
        cosim_visible_vehicles = self.coupler.getVisibleData()
        self.resetVisibleStates()
        for lane in cosim_visible_vehicles:
            for id in cosim_visible_vehicles[lane]:
                self.spawnVisibleVehicle(cosim_visible_vehicles[lane][id])

    def updateVisibleVehiclesFromCoSIM(self):
        """Reconcile SUMO's population with the coupler's visible state."""
        cosim_visible_ids = set(self.coupler.getVisibleIdsFlat())
        for cosim_id in list(self.visible_states.keys()):
            if cosim_id not in cosim_visible_ids:
                self.despawnVisibleVehicle(self.visible_states[cosim_id])
        cosim_visible_vehicles = self.coupler.getVisibleData()
        for lane in cosim_visible_vehicles:
            for cosim_id in cosim_visible_vehicles[lane]:
                if cosim_id not in self.visible_states:
                    self.spawnVisibleVehicle(cosim_visible_vehicles[lane][cosim_id])

    def updateSimulation(self):
        self.updateVisibleVehiclesFromCoSIM()

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def _defaultAheadSpeed(self, road_id) -> float:
        try:
            return float(self.conn.lane.getMaxSpeed(self.laneID(road_id, -1)))
        except Exception:
            return 30.0

    def applyLeadVehicleSpeeds(self):
        """Push the downstream macroscopic velocity onto each lane's leader.

        This is the only channel by which the macro state ahead of the bubble
        reaches the microscopic vehicles.  ``setSpeed`` is left under SUMO's
        default speed mode, so the target is still clipped to a safe speed --
        the leader is pulled toward the macro velocity but will not drive into
        anything to get there.
        """
        if not self.lead_speed_control:
            return
        road_id = str(self.coupler.hero_road)
        leaders: Dict[int, Optional[dict]] = {lane: None for lane in self.coupler.getLanes()}
        for cosim_id, record in self.visible_states.items():
            lane = record["last_lane_id"]
            if lane not in leaders:
                continue
            if (leaders[lane] is None) or (record["last_s"] > leaders[lane]["last_s"]):
                leaders[lane] = record

        new_lead_ids = set()
        for lane, record in leaders.items():
            if record is None:
                continue
            ghost_lead = self.coupler.getLowestAheadGhostVehicle(lane)
            if ghost_lead is not None:
                target_speed = float(ghost_lead["velocity"])
            else:
                target_speed = float(
                    self.coupler.getAheadLaneVelocityMacro(lane, self._defaultAheadSpeed(road_id))
                )
            cosim_id = record["cosim_data"]["id"]
            new_lead_ids.add(cosim_id)
            try:
                self.conn.vehicle.setSpeed(record["sumo_id"], max(0.0, target_speed))
            except Exception:
                pass
            self.lead_controlled[cosim_id] = True

        # Anything that stopped being a leader goes back to car-following.
        for cosim_id in list(self.lead_controlled.keys()):
            if cosim_id in new_lead_ids:
                continue
            record = self.visible_states.get(cosim_id)
            self.lead_controlled.pop(cosim_id, None)
            if record is None:
                continue
            try:
                self.conn.vehicle.setSpeed(record["sumo_id"], -1)
            except Exception:
                pass

    def releaseSpawnHolds(self):
        """Hand freshly inserted vehicles over to the car-following model."""
        for cosim_id, record in self.visible_states.items():
            if record["hold_substeps"] > 0:
                record["hold_substeps"] -= 1
                if record["hold_substeps"] == 0 and cosim_id not in self.lead_controlled:
                    try:
                        self.conn.vehicle.setSpeed(record["sumo_id"], -1)
                    except Exception:
                        pass
        if self.hero_state is not None and self.hero_state["hold_substeps"] > 0:
            self.hero_state["hold_substeps"] -= 1
            if self.hero_state["hold_substeps"] == 0 and self.hero_policy is None:
                try:
                    self.conn.vehicle.setSpeed(self.hero_state["sumo_id"], -1)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Hero control
    # ------------------------------------------------------------------

    def buildHeroObservation(self) -> dict:
        veh_id = self.hero_state["sumo_id"]
        lane_id = self.hero_state["last_lane_id"]
        leader = None
        try:
            leader_result = self.conn.vehicle.getLeader(veh_id, 200.0)
            if leader_result is not None and leader_result[0] != "":
                leader = {"sumo_id": leader_result[0], "gap": float(leader_result[1])}
                leader["speed"] = float(self.conn.vehicle.getSpeed(leader_result[0]))
        except Exception:
            leader = None
        return {
            "time": self.current_timestamp,
            "dt": self.step_length,
            "s": self.hero_state["last_s"],
            "speed": float(self.hero_state["cosim_data"]["velocity"]),
            "lane_id": lane_id,
            "lane_index": self.laneIndex(self.coupler.hero_road, lane_id),
            "leader": leader,
            "macro_velocity_ahead": self.coupler.getAheadLaneVelocityMacro(
                lane_id, self._defaultAheadSpeed(str(self.coupler.hero_road))
            ),
            "macro_density_ahead": self.coupler.getAheadLaneDensity(lane_id, 0.0),
            "visible_window": self.coupler.getCurrentVisibleWindow(),
        }

    def applyHeroPolicy(self):
        """Let an external policy override the hero, if one was supplied.

        With ``hero_policy=None`` the hero is an ordinary SUMO vehicle and the
        bubble's motion is emergent, which is the default.
        """
        if self.hero_policy is None or self.hero_state is None:
            return
        action = self.hero_policy.act(self.buildHeroObservation())
        if not action:
            return
        veh_id = self.hero_state["sumo_id"]
        try:
            if "speed" in action:
                self.conn.vehicle.setSpeed(veh_id, float(action["speed"]))
            if "acceleration" in action:
                current = float(self.conn.vehicle.getSpeed(veh_id))
                target = max(0.0, current + float(action["acceleration"]) * self.step_length)
                self.conn.vehicle.setSpeed(veh_id, target)
            if "lane" in action:
                self.conn.vehicle.changeLane(
                    veh_id, int(action["lane"]), float(action.get("lane_duration", self.step_length))
                )
        except Exception as exc:
            print(f"WARNING: hero policy action {action} failed: {exc}")

    # ------------------------------------------------------------------
    # Readback
    # ------------------------------------------------------------------

    def _refreshRecord(self, record) -> bool:
        """Pull one vehicle's state out of SUMO. False if it is no longer usable."""
        veh_id = record["sumo_id"]
        try:
            edge_id = self.conn.vehicle.getRoadID(veh_id)
        except Exception:
            return False
        if edge_id.startswith(":"):
            # On an internal junction lane; keep the previous sample for a step.
            return True
        road_id = self.roadIDFromEdge(edge_id)
        if road_id is None or road_id != str(self.coupler.hero_road):
            return False  # Left the coupled road.
        length = float(self.conn.vehicle.getLength(veh_id))
        lane_index = int(self.conn.vehicle.getLaneIndex(veh_id))
        lane_id = self.openDriveLane(road_id, lane_index)
        if lane_id not in self.coupler.getLanes():
            return False  # Changed into a lane the macro side does not model.
        record["last_s"] = float(self.conn.vehicle.getLanePosition(veh_id)) - length
        record["last_lane_id"] = lane_id
        record["cosim_data"]["velocity"] = float(self.conn.vehicle.getSpeed(veh_id))
        record["cosim_data"]["length"] = length
        record["cosim_data"]["width"] = float(self.conn.vehicle.getWidth(veh_id))
        return True

    def rebuildCoSIMVehicleState(self, record) -> dict:
        cosim_data = record["cosim_data"]
        lane_id = record["last_lane_id"]
        return {
            "id": cosim_data["id"],
            "class": cosim_data["class"],
            "length": float(cosim_data["length"]),
            "width": float(cosim_data["width"]),
            "time": self.current_timestamp,
            "s": float(record["last_s"]),
            # Same lateral kludge the CARLA path uses, so the two engines produce
            # comparable t values.
            "t": (lane_id * 3.5) + 1.75,
            "velocity": float(cosim_data["velocity"]),
            "lane_id": lane_id,
            "road_id": self.coupler.hero_road,
        }

    def rebuildCoSIMHeroState(self) -> dict:
        return self.rebuildCoSIMVehicleState(self.hero_state)

    def rebuildCoSIMVisibleState(self) -> dict:
        cosim_visible_states = {lane: {} for lane in self.coupler.getLanes()}
        for record in self.visible_states.values():
            rebuilt = self.rebuildCoSIMVehicleState(record)
            # _refreshRecord drops anything that wandered into an unmodelled
            # lane, so every surviving record lands in a bucket here.
            cosim_visible_states[rebuilt["lane_id"]][rebuilt["id"]] = rebuilt
        return cosim_visible_states

    def updateCoSIM(self):
        return self.rebuildCoSIMHeroState(), self.rebuildCoSIMVisibleState()

    # ------------------------------------------------------------------
    # Vehicle loss accounting
    # ------------------------------------------------------------------

    def _lossSideFor(self, record, arrived_ids) -> str:
        if record["sumo_id"] in arrived_ids:
            return "front"  # Drove off the downstream end of the network.
        window = self.coupler.getCurrentVisibleWindow()
        centre = 0.5 * (window[2] + window[3])
        return "rear" if record["last_s"] < centre else "front"

    def _reapVanishedVehicles(self):
        """Account for vehicles SUMO no longer has.

        A vehicle can leave the engine without crossing a bubble boundary: an
        insertion that never took, a collision removal, or driving off the
        coupled edge.  The window logic in the coupler cannot see any of those,
        so their mass is handed back to the macro flux memory here.  Otherwise
        the bubble would quietly destroy vehicles.
        """
        try:
            live_ids = set(self.conn.vehicle.getIDList())
            arrived_ids = set(self.conn.simulation.getArrivedIDList())
        except Exception:
            return

        for cosim_id in list(self.visible_states.keys()):
            record = self.visible_states[cosim_id]
            record["age_substeps"] += 1
            if record["sumo_id"] in live_ids:
                record["seen"] = True
                if self._refreshRecord(record):
                    continue
                # Still in SUMO but off the coupled road: remove it deliberately.
                self.despawnVehicle(record)
                self.visible_states.pop(cosim_id, None)
                self.lead_controlled.pop(cosim_id, None)
                self.coupler.creditLostVehicle(record["last_lane_id"], "front")
                continue

            if not record["seen"]:
                if record["age_substeps"] < self.insertion_grace_substeps:
                    continue  # Still pending departure.
                side = record["spawn_side"]
            else:
                side = self._lossSideFor(record, arrived_ids)
            self.visible_states.pop(cosim_id, None)
            self.lead_controlled.pop(cosim_id, None)
            self.coupler.creditLostVehicle(record["last_lane_id"], side)

        self._ensureHeroPresent(live_ids)

    def _ensureHeroPresent(self, live_ids):
        if self.hero_state is None:
            return
        if self.hero_state["sumo_id"] in live_ids:
            self.hero_state["seen"] = True
            if self._refreshRecord(self.hero_state):
                return
        elif not self.hero_state["seen"]:
            self.hero_state["age_substeps"] += 1
            if self.hero_state["age_substeps"] < self.insertion_grace_substeps:
                return
        # The bubble is anchored on the hero, so losing it would strand the
        # bridge. Put it back where it was and keep going, loudly.
        print(
            f"WARNING: hero vehicle {self.hero_state['sumo_id']} left SUMO at "
            f"s={self.hero_state['last_s']:.1f}; reinserting at its last state."
        )
        self.despawnVehicle(self.hero_state)
        cosim_data = self.hero_state["cosim_data"]
        cosim_data["s"] = self.hero_state["last_s"]
        cosim_data["lane_id"] = self.hero_state["last_lane_id"]
        veh_id = self.spawnVehicleFromCoSIM(cosim_data)
        if veh_id is None:
            raise RuntimeError("Lost the hero vehicle and could not reinsert it.")
        self.hero_state = self._trackingRecord(cosim_data, veh_id, self.hero_state["spawn_side"])

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def runSimulationOver(self, t=1.0, eps=1e-9):
        """Advance SUMO by ``t`` seconds in ``step_length`` substeps."""
        substeps = max(1, int(round(t / self.step_length)))
        for _ in range(substeps):
            self.releaseSpawnHolds()
            self.applyLeadVehicleSpeeds()
            self.applyHeroPolicy()
            self.conn.simulationStep()
            self.current_timestamp += self.step_length
            self._reapVanishedVehicles()
        return self.updateCoSIM()
