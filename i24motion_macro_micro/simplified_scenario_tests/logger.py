import pandas
from bridge_coupler import SimplifiedSimBridge
from prescribed_bridge_coupler import PrescribedSimBridge
from simulation import Simulation

class Logger:
    """Streams the macro field and mask state to CSV as the simulation runs.

    Rows are buffered and appended in chunks rather than held for the whole run.
    The previous accumulate-everything-then-write design needed ~6 GB per process at
    dx=8/dt=0.0625 (~24M rows) and made anything finer impossible; it also lost the
    entire log if a run was killed. Peak memory is now bounded by `flush_rows`
    regardless of run length or cell count.

    Output is byte-identical to the unbuffered version.
    """

    # Buffered macro rows before a chunk is appended. Bounds memory independently of
    # how many active cells the mesh has.
    flush_rows = 250_000

    MACRO_COLUMNS = (
        "time", "dt", "x_start_position", "length", "density", "velocity", "flow",
    )
    MASK_COLUMNS = (
        "time", "dt", "x_start_position", "length",
        "rear_density", "rear_velocity", "rear_flow",
        "front_density", "front_velocity", "front_flow",
        # Cumulative integral of the actual coupling flux, sum(net_flux * dt). Distinct from
        # rear_flow/front_flow above, which are just q(rho_interior) and ignore the exterior
        # state and the anchor speed. *_flux_memory is a running BALANCE that spawn/despawn
        # also debit and credit; *_flux_total is the pure integral. Reconcile against the
        # totals, never the memories.
        "rear_flux_memory", "front_flux_memory",
        "rear_flux_total", "front_flux_total",
        "vehicle_count",
    )
    # Per-vehicle trajectories. OFF unless vehicle_path is given, because the sweep does
    # not need them and the format of the two logs above must not change underneath the
    # analysis code. `s` is the ABSOLUTE road coordinate: bridge.vehicles holds absolute
    # positions, and collate_vehicles() is what converts to mask-relative for the fluid
    # side, so this reads the bridge rather than masking_cell.vehicles.
    VEHICLE_COLUMNS = (
        "time", "dt", "vehicle_id", "s", "s_dt", "length", "mask_rear_s",
    )

    def __init__(self, sim: Simulation, bridge: SimplifiedSimBridge | PrescribedSimBridge = None, sim_macro_path="sim_data.csv", mask_path="mask_data.csv", callback_name="logger", vehicle_path=None):
        self.sim_macro_data = {c: [] for c in self.MACRO_COLUMNS}
        self.mask_data = {c: [] for c in self.MASK_COLUMNS}
        self.vehicle_data = {c: [] for c in self.VEHICLE_COLUMNS}
        self.sim = sim
        self.bridge = bridge
        self.sim_macro_path = sim_macro_path
        self.mask_path = mask_path
        self.vehicle_path = vehicle_path
        self.callback_name = callback_name

        # Opened for the life of the run; truncates any previous output up front so a
        # killed run leaves a short file rather than a stale complete one.
        self._macro_handle = open(sim_macro_path, "w", newline="")
        self._mask_handle = open(mask_path, "w", newline="")
        self._vehicle_handle = open(vehicle_path, "w", newline="") if vehicle_path else None
        self._headers_written = False
        self._closed = False

        sim.register_poststep_callback(self._step, callback_name)

    def _step(self, sim_time: float, dt: float):
        for cell_id in self.bridge.sim.active.active_cells:
            cell = self.bridge.sim.active.active_cells[cell_id]
            cell_length = cell.length
            cell_x_start_position = cell.start_s
            cell_density = cell.mass / cell_length
            cell_velocity = self.bridge.fd.velocity_from_density(cell_density)
            cell_flow = cell_density * cell_velocity
            self.sim_macro_data["time"].append(sim_time)
            self.sim_macro_data["dt"].append(dt)
            self.sim_macro_data["x_start_position"].append(cell_x_start_position)
            self.sim_macro_data["length"].append(cell_length)
            self.sim_macro_data["density"].append(cell_density)
            self.sim_macro_data["velocity"].append(cell_velocity)
            self.sim_macro_data["flow"].append(cell_flow)

        self.mask_data["time"].append(sim_time)
        self.mask_data["dt"].append(dt)
        # Read the mask window off the active mesh rather than off bridge.middle_s.
        # move_active_masks() shifts the mesh mask by anchor_speed*dt during the step,
        # and SimplifiedSimBridge only refreshes middle_s on the *next* _step, so the
        # bridge value lags the mesh by one anchor step. Downstream analysis uses this
        # position to exclude the mask cell, so it has to be the mesh's own geometry.
        mask_cell = self.sim.active.get_cell_with_mask(self.bridge._mask_id(self.bridge.lane_id))
        if mask_cell is not None:
            mask_rear_s = float(mask_cell.start_s)
            mask_length = float(mask_cell.end_s - mask_cell.start_s)
        else:
            # Bridge already destroyed: no mask cell to exclude, fall back to its last geometry.
            mask_rear_s = self.bridge.middle_s - self.bridge.margin_s
            mask_length = self.bridge.margin_s * 2.0
        self.mask_data["x_start_position"].append(mask_rear_s)
        self.mask_data["length"].append(mask_length)
        rear_density = self.bridge.masking_cell.get_rear_density(self.bridge.fd)
        rear_velocity = self.bridge.fd.velocity_from_density(rear_density)
        rear_flow = rear_density * rear_velocity
        front_density = self.bridge.masking_cell.get_front_density(self.bridge.fd)
        front_velocity = self.bridge.fd.velocity_from_density(front_density)
        front_flow = front_density * front_velocity
        self.mask_data["rear_density"].append(rear_density)
        self.mask_data["rear_velocity"].append(rear_velocity)
        self.mask_data["rear_flow"].append(rear_flow)
        self.mask_data["front_density"].append(front_density)
        self.mask_data["front_velocity"].append(front_velocity)
        self.mask_data["front_flow"].append(front_flow)
        self.mask_data["rear_flux_memory"].append(self.bridge.masking_cell.rear_flux_memory)
        self.mask_data["front_flux_memory"].append(self.bridge.masking_cell.front_flux_memory)
        self.mask_data["rear_flux_total"].append(self.bridge.masking_cell.rear_flux_total)
        self.mask_data["front_flux_total"].append(self.bridge.masking_cell.front_flux_total)
        self.mask_data["vehicle_count"].append(len(self.bridge.masking_cell.vehicles))

        if self._vehicle_handle is not None:
            # getattr: PrescribedSimBridge and a torn-down bridge may not carry a vehicle
            # dict, and a missing trajectory log must never abort a run.
            for vehicle_id, vehicle in getattr(self.bridge, "vehicles", {}).items():
                self.vehicle_data["time"].append(sim_time)
                self.vehicle_data["dt"].append(dt)
                self.vehicle_data["vehicle_id"].append(vehicle_id)
                self.vehicle_data["s"].append(vehicle.s)
                self.vehicle_data["s_dt"].append(vehicle.s_dt)
                self.vehicle_data["length"].append(vehicle.length)
                self.vehicle_data["mask_rear_s"].append(mask_rear_s)

        if len(self.sim_macro_data["time"]) >= self.flush_rows:
            self._write_chunk()

    def _write_chunk(self):
        """Append the buffered rows to disk and clear the buffers.

        Both files are advanced together so the header is written exactly once each,
        even on a step where one buffer happens to be empty.
        """
        if self._closed:
            return
        header = not self._headers_written
        pandas.DataFrame(self.sim_macro_data, columns=list(self.MACRO_COLUMNS)).to_csv(
            self._macro_handle, header=header, index=False)
        pandas.DataFrame(self.mask_data, columns=list(self.MASK_COLUMNS)).to_csv(
            self._mask_handle, header=header, index=False)
        if self._vehicle_handle is not None:
            pandas.DataFrame(self.vehicle_data, columns=list(self.VEHICLE_COLUMNS)).to_csv(
                self._vehicle_handle, header=header, index=False)
            self._vehicle_handle.flush()
        self._headers_written = True
        self._macro_handle.flush()
        self._mask_handle.flush()
        for buf in (self.sim_macro_data, self.mask_data, self.vehicle_data):
            for column in buf:
                buf[column].clear()

    def flush(self):
        """Write any remaining buffered rows and close the files."""
        if self._closed:
            return
        self._write_chunk()
        self._macro_handle.close()
        self._mask_handle.close()
        if self._vehicle_handle is not None:
            self._vehicle_handle.close()
        self._closed = True

    def destroy(self):
        self.flush()
        self.sim.unregister_poststep_callback(self.callback_name)
