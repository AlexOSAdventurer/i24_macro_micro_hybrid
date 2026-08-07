import sys
sys.path.append("..")
import i24_motion_macro
import simulation
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import os
import math

class SimplifiedSimulationData:
    def __init__(self, config_path="config.json", network_path="network.json", macro_path="macro.parquet"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.simulation_path = self.config["storage_locations"]["simulation_dataset"]
        self.network_path = os.path.join(self.simulation_path, network_path)
        self.macro_final_path = os.path.join(self.simulation_path, macro_path)
        os.makedirs(self.simulation_path, exist_ok=True)

    def create_network_file(self):
        config = self.config["road_data"]["1"]
        triangular_fd = simulation.TriangularFD(v_f=self.config["fd"]["v_f"], w=self.config["fd"]["w"], rho_j=self.config["fd"]["rho_j"])
        self.network_generator = simulation.SimplifiedOneLaneRoadNetwork(fd=triangular_fd, lambda_lc=self.config["fd"]["lambda_lc"])
        self.network_generator.create_network(config["road_length"], config["cell_length"], lane_width=config["lane_width"])
        self.network_generator.save_network(self.network_path)

    def generate_macro_data(self):
        road_str = "1"
        lane_str = "-1"
        road_config = self.config["road_data"][road_str]
        road_length = road_config["road_length"]
        cell_length = road_config["cell_length"]
        time_origin = road_config["time_origin"]
        time_length = road_config["time_length"]
        time_step = road_config["time_step"]

        total_cells = math.ceil(road_length / cell_length)
        total_time = math.ceil(time_length / time_step) + 1

        network = simulation.Network.from_json(self.network_path)
        fd : simulation.TriangularFD = network.roads["1"].cells["road_1_cell_-1_step_0"].fd

        time_list = np.empty((total_cells * total_time), dtype=float)
        timelength_list = np.empty((total_cells * total_time), dtype=float)
        road_id_list = np.empty((total_cells * total_time), dtype=np.dtypes.StringDType())
        cell_id_list = np.empty((total_cells * total_time), dtype=np.dtypes.StringDType())
        density_list = np.empty((total_cells * total_time), dtype=float)
        velocity_list = np.empty((total_cells * total_time), dtype=float)
        cell_id_base_reference = [f"road_1_cell_{lane_str}_step_{i}" for i in range(total_cells)]
        for i in range(total_cells):
            for j in range(total_time):
                if ((float(i) / float(total_cells)) < 0.5): # Half way down the road or no?
                    density = fd.rho_c / 2.0
                    velocity = fd.velocity_from_density(density)
                else:
                    density = fd.rho_j
                    velocity = 0.0
                new_index = (i * total_time) + j
                time_list[new_index] = (j * time_step) + time_origin
                timelength_list[new_index] = time_step
                road_id_list[new_index] = road_str
                cell_id_list[new_index] = cell_id_base_reference[i]
                density_list[new_index] = density
                velocity_list[new_index] = velocity
                print(i, j)

        final_dataframe = pd.DataFrame({
            "time": time_list,
            "time_length": timelength_list,
            "road_id": road_id_list,
            "cell_id": cell_id_list,
            "density": density_list,
            "velocity": velocity_list
        })
        
        final_dataframe_sorted = final_dataframe.sort_values(["time", "road_id", "cell_id"], kind="mergesort", ascending=True)
        sorting_columns = [
            pq.SortingColumn(column_index=0, descending=False, nulls_first=True),
            pq.SortingColumn(column_index=2, descending=False, nulls_first=True),
            pq.SortingColumn(column_index=3, descending=False, nulls_first=True),
        ]
        table = pa.Table.from_pandas(final_dataframe_sorted, preserve_index=False)
        pq.write_table(table, self.macro_final_path, compression="zstd", row_group_size=5000, sorting_columns=sorting_columns)

if __name__ == "__main__":
    sim_data = SimplifiedSimulationData(config_path="config_demo.json")
    sim_data.create_network_file()
    sim_data.generate_macro_data()