import pandas
from bridge_coupler import SimplifiedSimBridge
from prescribed_bridge_coupler import PrescribedSimBridge
from simulation import Simulation

class Logger:
    def __init__(self, sim: Simulation, bridge: SimplifiedSimBridge | PrescribedSimBridge = None, sim_macro_path="sim_data.csv", mask_path="mask_data.csv", callback_name="logger"):
        self.sim_macro_data = {
            "time": [],
            "dt": [],
            "x_start_position": [],
            "length": [],
            "density": [],
            "velocity": [],
            "flow": []
        }
        self.mask_data = {
            "time": [],
            "dt": [],
            "x_start_position": [],
            "length": [],
            "rear_density": [],
            "rear_velocity": [],
            "rear_flow": [],
            "front_density": [],
            "front_velocity": [],
            "front_flow": []
        }
        self.sim = sim
        self.bridge = bridge
        self.sim_macro_path = sim_macro_path
        self.mask_path = mask_path
        self.callback_name = callback_name
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
        mask_middle_s = self.bridge.middle_s
        mask_rear_s = mask_middle_s - self.bridge.margin_s
        mask_length = self.bridge.margin_s * 2.0
        self.mask_data["x_start_position"].append(mask_rear_s)
        self.mask_data["length"].append(mask_length)
        rear_density = self.bridge.masking_cell.get_rear_density(self.bridge.fd)
        rear_velocity = self.bridge.fd.velocity_from_density(rear_density)
        rear_flow = rear_density * rear_velocity
        front_density = self.bridge.masking_cell.get_rear_density(self.bridge.fd)
        front_velocity = self.bridge.fd.velocity_from_density(rear_density)
        front_flow = rear_density * rear_velocity
        self.mask_data["rear_density"].append(rear_density)
        self.mask_data["rear_velocity"].append(rear_velocity)
        self.mask_data["rear_flow"].append(rear_flow)
        self.mask_data["front_density"].append(front_density)
        self.mask_data["front_velocity"].append(front_velocity)
        self.mask_data["front_flow"].append(front_flow)

    def flush(self):
        final_sim_macro_df = pandas.DataFrame(self.sim_macro_data)
        final_mask_data_df = pandas.DataFrame(self.mask_data)

        final_sim_macro_df.to_csv(self.sim_macro_path, index=False)
        final_mask_data_df.to_csv(self.mask_path, index=False)

    def destroy(self):
        self.flush()
        self.sim.unregister_poststep_callback(self.callback_name)