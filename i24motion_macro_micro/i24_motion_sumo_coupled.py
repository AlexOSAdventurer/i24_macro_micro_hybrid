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

import numpy as np

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

    # Acceleration Constants.  These are AVC's (Yan et al., T-ASE 2022) ring-road
    # values, adopted so the corridor's human drivers are the same car-following
    # model the ring controllers were trained against; the previous corridor
    # values were max_accel = 1.5, max_decel = 2.0.  Not adopted from AVC:
    # tau and minGap/length, which are pinned by the calibrated triangular FD
    # (tau = 1/(w * rho_j) = 2.26 s, length + minGap = 1/rho_j) and are what keep
    # the microscopic steady state on the CTM's fundamental diagram -- taking
    # AVC's tau=1.0 would imply w = 12.95 m/s, 2.26x the calibrated backward wave
    # speed, and 1.83x the CTM's capacity.  speedFactor needs no change: the
    # add.xml's normc(1.0,0.1,0.8,1.2) is exactly AVC's speedFactor=1.0 with
    # speedDev=0.1 at SUMO's default two-deviation cutoff.
    #max_accel = 1.5
    #max_decel = 2.0
    max_accel = 1.0
    max_decel = 1.5

    # Amplitude of the human-driver acceleration noise, in m/s^2.  AVC's ring
    # gets its string instability from a patched SUMO (ZhongxiaYan/sumo, branch
    # 1.1.0) whose IDM adds randNorm(0, sigma) to the computed acceleration every
    # step, with sigma = 0.2 for human vehicles and 0 for the RL vehicle.  The
    # stock SUMO here has no such hook and its `sigma` attribute is a documented
    # no-op under IDM, so the noise is injected over TraCI instead; see
    # apply_idm_noise.  Set to 0 to recover deterministic IDM.
    idm_sigma = 0.2

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
        hero_min_gap: Optional[float] = None,
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
        # None keeps the class default (the FD-derived minGap); 0.0 opts into the ring's
        # AV authority.  See the hero_min_gap class attribute.
        if hero_min_gap is not None:
            self.hero_min_gap = float(hero_min_gap)
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
        self._tc = None

        # TraCI is a request/response protocol over TCP, so per-vehicle getters
        # cost a socket round-trip each and the substep loop is latency-bound
        # rather than compute-bound.  Reading the same variables through a
        # subscription collapses the whole population into one round-trip per
        # substep, which is the difference between ~600 round-trips per macro
        # step and a handful.
        self._subscribed: set = set()
        self._sub_results: Dict[str, dict] = {}
        # Geometry is written once at spawn and never changes, so it is cached
        # rather than re-read every substep for every vehicle.
        self._vehicle_geometry: Dict[str, tuple] = {}
        self._lane_max_speed: Dict[str, float] = {}
        # Driver noise is drawn PER VEHICLE, from a generator seeded by (run seed,
        # cosim id), rather than from one shared stream.  That makes the noise common
        # random numbers across runs that differ only in the hero's actions, which is
        # what the paired policy-vs-baseline comparison needs: a shared stream desyncs
        # as soon as the hero changes how many vehicles are lane leaders or on spawn
        # holds, so the two arms would see different noise realisations and the paired
        # difference would carry variance unrelated to the controller.  Kept separate
        # from SUMO's own RNG (--seed above) for the same reason.
        self._noise_seed = seed
        self._noise_rngs: Dict[int, np.random.Generator] = {}

    # ------------------------------------------------------------------
    # Identifiers and geometry
    # ------------------------------------------------------------------

    def edge_id(self, road_id) -> str:
        return "-" + str(road_id)

    def road_id_from_edge(self, edge_id: str) -> Optional[str]:
        if not edge_id.startswith("-"):
            return None
        return edge_id[1:]

    def lane_count(self, road_id) -> int:
        return int(self.mapping[str(road_id)]["lanes"])

    def lane_index(self, road_id, lane_id) -> int:
        """OpenDRIVE lane id -> SUMO lane index. Lane -1 is the innermost."""
        return self.lane_count(road_id) + int(lane_id)

    def open_drive_lane(self, road_id, lane_index) -> int:
        return int(lane_index) - self.lane_count(road_id)

    def lane_id(self, road_id, lane_id) -> str:
        return f"{self.edge_id(road_id)}_{self.lane_index(road_id, lane_id)}"

    def vehicle_id(self, cosim_id) -> str:
        return f"v{int(cosim_id)}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _import_sumo(self):
        if self.use_libsumo:
            import libsumo

            self._traci = libsumo
        else:
            import traci

            self._traci = traci
        try:
            self._tc = self._traci.constants
        except AttributeError:  # older libsumo exposes the constants separately
            import traci.constants as tc

            self._tc = tc
        import sumolib

        self._sumolib = sumolib

    def _binary(self) -> str:
        name = "sumo-gui" if self.gui else "sumo"
        try:
            return self._sumolib.checkBinary(name)
        except Exception:
            return name

    def _build_command(self) -> List[str]:
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

    def _register_routes(self):
        """One route per coupled road: the mainline edge plus whatever follows it.

        The bubble never reaches the end of the mainline in the configured runs,
        but giving vehicles a downstream continuation keeps SUMO from treating
        them as about to arrive.
        """
        net = self._sumolib.net.readNet(self.net_file)
        for road_id in self.mapping:
            edge_id = self.edge_id(road_id)
            try:
                edge = net.getEdge(edge_id)
            except KeyError:
                continue
            edges = [edge_id] + [e.getID() for e in edge.getOutgoing()][:1]
            route_id = f"route_{road_id}"
            self.conn.route.add(route_id, edges)
            self.route_ids[str(road_id)] = route_id

    def connect_to_host(self):
        self._import_sumo()
        command = self._build_command()
        if self.use_libsumo:
            self._traci.start(command)
            self.conn = self._traci
        else:
            self._traci.start(command, label=self.label)
            self.conn = self._traci.getConnection(self.label)
        self.started = True

    def initialize_simulation(self):
        self.connect_to_host()
        self._register_routes()
        self.current_timestamp = self.coupler.current_timestamp
        if self.hero_policy is not None:
            self.hero_policy.reset(self)
        self.spawn_hero_vehicle()
        self.spawn_and_despawn_visible_vehicles_from_co_sim()
        # Deliberately no simulationStep() here. Stepping now would advance every
        # vehicle by one substep that the coupler's clock never sees, putting the
        # seeded positions permanently out of step with self.current_timestamp.
        # The first run_simulation_over() steps before it reads anything back, so
        # the forced insertions are in the network by the time they are sampled.

    def destroy_simulation(self):
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

    def _tracking_record(self, cosim_data, veh_id, spawn_side):
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

    def _spawn_side_for(self, cosim_data) -> str:
        """Which bubble boundary this vehicle entered through.

        Used to refund the right flux memory if the insertion turns out to have
        failed, so a rejected spawn exactly undoes the decrement that paid for it.
        """
        window = self.coupler.get_current_visible_window()
        centre = 0.5 * (window[2] + window[3])
        return "rear" if float(cosim_data["s"]) < centre else "front"

    def spawn_vehicle_from_co_sim(self, cosim_data) -> Optional[str]:
        road_id = str(cosim_data["road_id"])
        route_id = self.route_ids.get(road_id)
        if route_id is None:
            print(f"WARNING: no SUMO route registered for road {road_id}; cannot spawn {cosim_data['id']}")
            return None

        veh_id = self.vehicle_id(cosim_data["id"])
        lane_index = self.lane_index(road_id, cosim_data["lane_id"])
        lane_id = self.lane_id(road_id, cosim_data["lane_id"])
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
            self.conn.vehicle.setMinGap(veh_id, self.coupler.min_spawn_distance)
            self.conn.vehicle.setMaxSpeed(veh_id, self.coupler.fd.v_f)
            self.conn.vehicle.setAccel(veh_id, self.max_accel)
            self.conn.vehicle.setDecel(veh_id, self.max_decel)
            self.conn.vehicle.setTau(veh_id, 1.0 / (self.coupler.fd.w * self.coupler.fd.rho_j))
        except Exception as exc:
            print(f"WARNING: could not place {veh_id} at {lane_id}@{front_pos:.2f}: {exc}")
        # The clamped values actually written, not the raw empirical ones, so the
        # cache reads back exactly what getLength/getWidth would have returned.
        self._vehicle_geometry[veh_id] = (length, width)
        # Deliberately NOT subscribed here.  vehicle.add() only queues a
        # departure: SUMO does not insert until the next step, and the insertion
        # can fail outright (hence insertion_grace_substeps).  Subscribing to an
        # id SUMO does not yet know makes it answer every variable with an error
        # on every step, forever.  _refresh_record subscribes instead, once the
        # vehicle has actually been read back.
        return veh_id

    def spawn_hero_vehicle(self):
        cosim_data = self.coupler.get_hero_data()
        veh_id = self.spawn_vehicle_from_co_sim(cosim_data)
        if veh_id is None:
            raise RuntimeError(f"Failed to insert the hero vehicle into SUMO: {cosim_data}")
        self.hero_state = self._tracking_record(cosim_data, veh_id, self._spawn_side_for(cosim_data))
        self._apply_hero_min_gap(veh_id)
        try:
            self.conn.vehicle.setColor(veh_id, (255, 0, 0, 255))
        except Exception:
            pass

    # Override for the designated AV slot's minGap.  DEFAULT None: the hero keeps the
    # FD-derived minGap (1/rho_j - average_spawn_length) that every other vehicle gets,
    # so the jam spacing stays consistent for it too.
    #
    # Pass 0.0 to OPT IN to ring-equivalent authority.  ring.py:86 zeroes the RL
    # vehicle's minGap for its controlled phase, so a ring-trained policy learned with
    # the freedom to close a gap completely; evaluating it here behind a 7.705 m buffer
    # applies a constraint it never trained with, and the mismatch flatters it, because
    # pressing against a buffer looks better behaved than the tailgating its training
    # rewarded.  Note the same line means very different things in the two setups: AVC's
    # minGap is 2 m, so zeroing frees 2 m, where ours is 7.705 m at a ~17 m mean gap.
    #
    # Whatever the value, it is applied to the hero whether or not a policy is driving,
    # so a paired policy-vs-baseline comparison differs in CONTROL ONLY rather than also
    # in vehicle physics.
    hero_min_gap: Optional[float] = None

    def _apply_hero_min_gap(self, veh_id):
        if self.hero_min_gap is None:
            return
        try:
            self.conn.vehicle.setMinGap(veh_id, float(self.hero_min_gap))
        except Exception as exc:
            print(f"WARNING: could not set hero minGap on {veh_id}: {exc}")

    def spawn_visible_vehicle(self, cosim_data):
        veh_id = self.spawn_vehicle_from_co_sim(cosim_data)
        if veh_id is None:
            # Nothing entered the engine, so give the mass straight back.
            self.coupler.credit_lost_vehicle(int(cosim_data["lane_id"]), self._spawn_side_for(cosim_data))
            return
        self.visible_states[cosim_data["id"]] = self._tracking_record(
            cosim_data, veh_id, self._spawn_side_for(cosim_data)
        )

    def despawn_vehicle(self, record):
        try:
            self.conn.vehicle.remove(record["sumo_id"])
        except Exception:
            pass  # Already gone from SUMO's side.
        self._forget_vehicle(record["sumo_id"])

    def despawn_visible_vehicle(self, record):
        self.despawn_vehicle(record)
        self.visible_states.pop(record["cosim_data"]["id"], None)
        self.lead_controlled.pop(record["cosim_data"]["id"], None)

    def reset_visible_states(self):
        self.visible_states = {}

    def spawn_and_despawn_visible_vehicles_from_co_sim(self):
        cosim_visible_vehicles = self.coupler.get_visible_data()
        self.reset_visible_states()
        for lane in cosim_visible_vehicles:
            for id in cosim_visible_vehicles[lane]:
                self.spawn_visible_vehicle(cosim_visible_vehicles[lane][id])

    def update_visible_vehicles_from_co_sim(self):
        """Reconcile SUMO's population with the coupler's visible state."""
        cosim_visible_ids = set(self.coupler.get_visible_ids_flat())
        for cosim_id in list(self.visible_states.keys()):
            if cosim_id not in cosim_visible_ids:
                self.despawn_visible_vehicle(self.visible_states[cosim_id])
        cosim_visible_vehicles = self.coupler.get_visible_data()
        for lane in cosim_visible_vehicles:
            for cosim_id in cosim_visible_vehicles[lane]:
                if cosim_id not in self.visible_states:
                    self.spawn_visible_vehicle(cosim_visible_vehicles[lane][cosim_id])

    def update_simulation(self):
        self.update_visible_vehicles_from_co_sim()

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def _default_ahead_speed(self, road_id) -> float:
        # A lane's speed limit is network geometry: fixed for the whole run, but
        # this was being fetched twice per substep.
        key = str(road_id)
        cached = self._lane_max_speed.get(key)
        if (cached is not None):
            return cached
        try:
            value = float(self.conn.lane.getMaxSpeed(self.lane_id(road_id, -1)))
        except Exception:
            value = 30.0
        self._lane_max_speed[key] = value
        return value

    def get_current_visible_window_substep(self):
        return self.hero_state["last_s"] - self.coupler.visible_window, self.hero_state["last_s"] + self.coupler.visible_window, 

    def apply_lead_vehicle_speeds(self):
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
        leaders: Dict[int, Optional[dict]] = {lane: None for lane in self.coupler.get_lanes()}
        rear_edge, front_edge = self.get_current_visible_window_substep()
        for cosim_id, record in self.visible_states.items():
            lane = record["last_lane_id"]
            if lane not in leaders:
                continue
            elif (record["last_s"] >= (front_edge - self.coupler.transition_region_size)) and ((leaders[lane] is None) or (record["last_s"] > leaders[lane]["last_s"])):
                leaders[lane] = record

        new_lead_ids = set()
        for lane, record in leaders.items():
            if record is None:
                continue
            ghost_lead = self.coupler.get_lowest_ahead_ghost_vehicle(lane)
            if ghost_lead is not None:
                target_speed = float(ghost_lead["velocity"])
            else:
                target_speed = float(
                    self.coupler.get_ahead_lane_velocity_macro(lane, self._default_ahead_speed(road_id))
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

    def apply_idm_noise(self):
        """Perturb every human driver's acceleration by N(0, idm_sigma) m/s^2.

        AVC's ring adds the draw inside IDM; TraCI cannot reach into the model, so
        the equivalent is done from outside.  ``setPreviousSpeed`` overwrites the
        speed SUMO integrates from, and IDM's update is
        ``v(t+dt) = v(t) + a(v(t), gap, v_lead) * dt``, so telling it the vehicle
        is at ``v + N(0, sigma) * dt`` shifts the next speed by exactly the same
        amount an acceleration perturbation of ``N(0, sigma)`` would -- to first
        order in dt, which is what the discrete model actually integrates.  This
        is the same emulation used on the ring side (``TrafficState.apply_idm_noise``
        in automatic_vehicular_control/env.py), so the two stacks share one noise
        process rather than two different ones.

        Three classes of vehicle are skipped.  The hero, because AVC's ``rl`` vType
        carries sigma=0 and its controller should not be fighting its own actuation
        noise.  Lane leaders, because ``apply_lead_vehicle_speeds`` has already
        forced their speed to the macroscopic velocity ahead -- they are not
        car-following at all, so a perturbation could not change their motion, only
        the acceleration SUMO reports for them.  Freshly spawned vehicles still on
        their insertion hold, for the same reason.

        Cost: one TraCI command per noised vehicle per substep.  There is no
        batched setter, and the call must happen after the readback and before
        ``simulationStep`` because it consumes the current speed.  Measured at ~5%
        of wall time on the socket client.

        On common random numbers: the per-vehicle generators are deliberately NOT
        reaped when a vehicle leaves, so a vehicle that exits and re-enters the bubble
        resumes its own sequence rather than restarting it.  The residual imperfection
        is that a vehicle present for a different NUMBER of substeps between two arms
        advances its stream a different number of times; perfect synchronisation would
        need a fresh generator keyed by (seed, id, absolute step) per draw, which costs
        a Generator construction per vehicle per substep.  The population-level desync
        this replaces was far larger: it shifted every vehicle's draw whenever the hero
        changed how many vehicles were lane leaders or on spawn holds.
        """
        if not self.idm_sigma:
            return
        dt = self.step_length
        for cosim_id, record in self.visible_states.items():
            if (cosim_id in self.lead_controlled) or (record["hold_substeps"] > 0):
                continue # under setSpeed, so a perturbation cannot move it
            veh_id = record["sumo_id"]
            sample = self._vehicle_sample(veh_id)
            if sample is None:
                continue
            # One generator per vehicle, advanced once per substep.  A vehicle's noise
            # sequence therefore depends only on its own id and the run seed -- not on
            # how many other vehicles happen to be present, nor on iteration order.
            rng = self._noise_rngs.get(cosim_id)
            if rng is None:
                rng = np.random.default_rng(
                    (self._noise_seed, int(cosim_id)) if (self._noise_seed is not None)
                    else (int(cosim_id),)
                )
                self._noise_rngs[cosim_id] = rng
            speed = float(sample[self._tc.VAR_SPEED])
            try:
                self.conn.vehicle.setPreviousSpeed(
                    veh_id, max(0.0, speed + float(rng.normal(0.0, self.idm_sigma)) * dt)
                )
            except Exception:
                pass

    def release_spawn_holds(self):
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

    def build_hero_observation(self) -> dict:
        veh_id = self.hero_state["sumo_id"]
        lane_id = self.hero_state["last_lane_id"]
        leader = None
        try:
            leader_result = self.conn.vehicle.getLeader(veh_id, 200.0)
            if leader_result is not None and leader_result[0] != "":
                leader = {"sumo_id": leader_result[0], "gap": float(leader_result[1])}
                leader_sample = self._vehicle_sample(leader_result[0])
                leader["speed"] = (
                    float(leader_sample[self._tc.VAR_SPEED]) if (leader_sample is not None)
                    else float(self.conn.vehicle.getSpeed(leader_result[0]))
                )
        except Exception:
            leader = None
        return {
            "time": self.current_timestamp,
            "dt": self.step_length,
            "s": self.hero_state["last_s"],
            "speed": float(self.hero_state["cosim_data"]["velocity"]),
            "lane_id": lane_id,
            "lane_index": self.lane_index(self.coupler.hero_road, lane_id),
            "leader": leader,
            "macro_velocity_ahead": self.coupler.get_ahead_lane_velocity_macro(
                lane_id, self._default_ahead_speed(str(self.coupler.hero_road))
            ),
            "macro_density_ahead": self.coupler.get_ahead_lane_density(lane_id, 0.0),
            "visible_window": self.coupler.get_current_visible_window(),
        }

    def apply_hero_policy(self):
        """Let an external policy override the hero, if one was supplied.

        With ``hero_policy=None`` the hero is an ordinary SUMO vehicle and the
        bubble's motion is emergent, which is the default.
        """
        if self.hero_policy is None or self.hero_state is None:
            return
        action = self.hero_policy.act(self.build_hero_observation())
        if not action:
            return
        veh_id = self.hero_state["sumo_id"]
        try:
            if "speed" in action:
                self.conn.vehicle.setSpeed(veh_id, float(action["speed"]))
            if "acceleration" in action:
                # The subscription was refreshed after the last simulationStep,
                # so this is the same value getSpeed would return right now.
                sample = self._vehicle_sample(veh_id)
                current = (
                    float(sample[self._tc.VAR_SPEED]) if (sample is not None)
                    else float(self.conn.vehicle.getSpeed(veh_id))
                )
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

    def _subscribe_vehicle(self, veh_id: str) -> None:
        """Register the variables ``_refresh_record`` needs, once per vehicle.

        Everything read per substep goes through this; geometry is deliberately
        excluded because it is set at spawn and cached in ``_vehicle_geometry``.
        """
        if (self._tc is None) or (veh_id in self._subscribed):
            return
        try:
            self.conn.vehicle.subscribe(
                veh_id,
                [
                    self._tc.VAR_ROAD_ID,
                    self._tc.VAR_LANE_INDEX,
                    self._tc.VAR_LANEPOSITION,
                    self._tc.VAR_SPEED,
                ],
            )
            self._subscribed.add(veh_id)
        except Exception:
            pass  # Fall back to direct getters for this vehicle.

    def _forget_vehicle(self, veh_id: str) -> None:
        if (veh_id in self._subscribed):
            try:
                # Without this the subscription outlives the vehicle and SUMO
                # answers every subscribed variable with an error, every step.
                self.conn.vehicle.unsubscribe(veh_id)
            except Exception:
                pass
        self._subscribed.discard(veh_id)
        self._vehicle_geometry.pop(veh_id, None)
        self._sub_results.pop(veh_id, None)

    def _refresh_subscriptions(self) -> None:
        """One round-trip for the whole population, straight after a step."""
        try:
            self._sub_results = self.conn.vehicle.getAllSubscriptionResults()
        except Exception:
            self._sub_results = {}

    def _vehicle_sample(self, veh_id: str) -> Optional[dict]:
        """Subscribed values for one vehicle, or None to fall back to getters.

        A vehicle inserted this substep has no results until the next step, and
        a subscription can fail outright, so every caller must cope with None.
        """
        sample = self._sub_results.get(veh_id)
        if (sample is None) or (self._tc is None):
            return None
        if (self._tc.VAR_ROAD_ID not in sample) or (self._tc.VAR_SPEED not in sample):
            return None
        return sample

    def _refresh_record(self, record) -> bool:
        """Pull one vehicle's state out of SUMO. False if it is no longer usable."""
        veh_id = record["sumo_id"]
        sample = self._vehicle_sample(veh_id)
        try:
            if sample is not None:
                edge_id = sample[self._tc.VAR_ROAD_ID]
            else:
                edge_id = self.conn.vehicle.getRoadID(veh_id)
        except Exception:
            return False
        # Reaching here means SUMO answered for this id, so it exists and can
        # safely carry a subscription.  This is the only place that subscribes:
        # a vehicle is registered exactly once, after it has been read back.
        self._subscribe_vehicle(veh_id)
        if edge_id.startswith(":"):
            # On an internal junction lane; keep the previous sample for a step.
            return True
        road_id = self.road_id_from_edge(edge_id)
        if road_id is None or road_id != str(self.coupler.hero_road):
            return False  # Left the coupled road.
        # Geometry is whatever was written at spawn, so it is read back from the
        # cache rather than fetched; SUMO cannot have changed it.
        geometry = self._vehicle_geometry.get(veh_id)
        if (geometry is not None):
            length, width = geometry
        else:
            length = float(self.conn.vehicle.getLength(veh_id))
            width = float(self.conn.vehicle.getWidth(veh_id))
        if sample is not None:
            lane_index = int(sample[self._tc.VAR_LANE_INDEX])
            lane_position = float(sample[self._tc.VAR_LANEPOSITION])
            speed = float(sample[self._tc.VAR_SPEED])
        else:
            lane_index = int(self.conn.vehicle.getLaneIndex(veh_id))
            lane_position = float(self.conn.vehicle.getLanePosition(veh_id))
            speed = float(self.conn.vehicle.getSpeed(veh_id))
        lane_id = self.open_drive_lane(road_id, lane_index)
        if lane_id not in self.coupler.get_lanes():
            return False  # Changed into a lane the macro side does not model.
        record["last_s"] = lane_position - length
        record["last_lane_id"] = lane_id
        record["cosim_data"]["velocity"] = speed
        record["cosim_data"]["length"] = length
        record["cosim_data"]["width"] = width
        return True

    def rebuild_co_sim_vehicle_state(self, record) -> dict:
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

    def rebuild_co_sim_hero_state(self) -> dict:
        return self.rebuild_co_sim_vehicle_state(self.hero_state)

    def rebuild_co_sim_visible_state(self) -> dict:
        cosim_visible_states = {lane: {} for lane in self.coupler.get_lanes()}
        for record in self.visible_states.values():
            rebuilt = self.rebuild_co_sim_vehicle_state(record)
            # _refreshRecord drops anything that wandered into an unmodelled
            # lane, so every surviving record lands in a bucket here.
            cosim_visible_states[rebuilt["lane_id"]][rebuilt["id"]] = rebuilt
        return cosim_visible_states

    def update_co_sim(self):
        return self.rebuild_co_sim_hero_state(), self.rebuild_co_sim_visible_state()

    # ------------------------------------------------------------------
    # Vehicle loss accounting
    # ------------------------------------------------------------------

    def _loss_side_for(self, record, arrived_ids) -> str:
        if record["sumo_id"] in arrived_ids:
            return "front"  # Drove off the downstream end of the network.
        centre = self.hero_state["last_s"]
        return "rear" if record["last_s"] < centre else "front"

    def _reap_vanished_vehicles(self):
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
                if self._refresh_record(record):
                    continue
                # Still in SUMO but off the coupled road: remove it deliberately.
                self.despawn_vehicle(record)
                self.visible_states.pop(cosim_id, None)
                self.lead_controlled.pop(cosim_id, None)
                self.coupler.credit_lost_vehicle(record["last_lane_id"], "front")
                continue

            if not record["seen"]:
                if record["age_substeps"] < self.insertion_grace_substeps:
                    continue  # Still pending departure.
                side = record["spawn_side"]
            else:
                side = self._loss_side_for(record, arrived_ids)
            self.visible_states.pop(cosim_id, None)
            self.lead_controlled.pop(cosim_id, None)
            self._forget_vehicle(record["sumo_id"])
            self.coupler.credit_lost_vehicle(record["last_lane_id"], side)

        self._ensure_hero_present(live_ids)

    def _ensure_hero_present(self, live_ids):
        if self.hero_state is None:
            return
        if self.hero_state["sumo_id"] in live_ids:
            self.hero_state["seen"] = True
            if self._refresh_record(self.hero_state):
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
        self.despawn_vehicle(self.hero_state)
        cosim_data = self.hero_state["cosim_data"]
        cosim_data["s"] = self.hero_state["last_s"]
        cosim_data["lane_id"] = self.hero_state["last_lane_id"]
        veh_id = self.spawn_vehicle_from_co_sim(cosim_data)
        if veh_id is None:
            raise RuntimeError("Lost the hero vehicle and could not reinsert it.")
        self.hero_state = self._tracking_record(cosim_data, veh_id, self.hero_state["spawn_side"])
        # The reinserted vehicle is a fresh SUMO id, so it came in with the FD-derived
        # minGap that spawn_vehicle_from_co_sim applies to everything; re-grant the AV's.
        self._apply_hero_min_gap(veh_id)

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def run_simulation_over(self, t=1.0, eps=1e-9):
        """Advance SUMO by ``t`` seconds in ``step_length`` substeps."""
        substeps = max(1, int(round(t / self.step_length)))
        for _ in range(substeps):
            self._ensure_hero_present(set(self.conn.vehicle.getIDList()))
            self.release_spawn_holds()
            self.apply_lead_vehicle_speeds()
            self.apply_hero_policy()
            # Last, so it reads the speeds the step will actually start from.
            self.apply_idm_noise()
            self.conn.simulationStep()
            self.current_timestamp += self.step_length
            # One round-trip for the whole population, before anything reads it.
            self._refresh_subscriptions()
            self._reap_vanished_vehicles()
        return self.update_co_sim()
