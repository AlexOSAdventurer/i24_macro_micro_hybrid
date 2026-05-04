import i24_motion_data
import i24_motion_macro
import simulation
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import os

class I24SimulationData:
    def __init__(self, config_path="i24_motion_to_dataset.json", network_path="network.json", macro_path="macro.parquet", micro_path="micro.parquet"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.preprocessing_path = self.config["storage_locations"]["preprocessing_data"]
        self.simulation_path = self.config["storage_locations"]["simulation_dataset"]
        self.network_path = os.path.join(self.simulation_path, network_path)
        self.macro_final_path = os.path.join(self.simulation_path, macro_path)
        self.micro_final_path = os.path.join(self.simulation_path, micro_path)
        self.micro_source_data = {}
        self.macro_source_data = {}
        os.makedirs(self.simulation_path, exist_ok=True)
        self.load_source_data()

    def create_network_file(self):
        # For now both roads have the same length, so just grab one of them for config.
        config = self.config["road_data"]["2"]
        triangular_fd = simulation.TriangularFD(v_f=49.73439485591916, w=5.427947465556459, rho_j=0.13498786841497915)
        self.network_generator = simulation.I24WestAndEastNetwork(fd=triangular_fd, lambda_lc=0.20197165527692704)
        #greenshields_fd = simulation.GreenshieldsFD(v_f=42.634906282620406, rho_j=0.06711901063946596)
        #self.network_generator = simulation.I24WestAndEastNetwork(fd=greenshields_fd, lambda_lc=0.28911022854691354)
        self.network_generator.create_network(config["road_length"], config["cell_length"], config["lanes"], lane_width=config["lane_width"])
        self.network_generator.save_network(self.network_path)

    def load_source_data(self):
        for road in self.config["road_data"]:
            self.micro_source_data[int(road)] = i24_motion_data.I24MotionData(int(road), self.config["road_data"][road]["time_origin"], self.config["road_data"][road]["time_origin"] + self.config["road_data"][road]["time_length"], 0.0, self.config["road_data"][road]["road_length"])
            self.macro_source_data[int(road)] = i24_motion_macro.I24MotionMacro(self.micro_source_data[int(road)], int(road), f"road_{road}")

    def generate_micro_data(self):
        feet_to_meters = 0.3048
        final_dataframe = pd.DataFrame()
        for road in self.micro_source_data:
            print(f"Generating micro data for road {road}")
            road_str = str(road)
            road_config = self.config["road_data"][road_str]
            all_vehicles = self.micro_source_data[road].queryEdieBoxBatch([road_config["time_origin"]], [road_config["time_origin"] + road_config["time_length"]], [0.0], [road_config["road_length"]])[0]
            for lane in all_vehicles:
                all_vehicles_selected_columns = all_vehicles[lane][["id", "time", "class", "s", "t", "length", "width", "height"]].copy()
                all_vehicles_selected_columns["length"] *= feet_to_meters
                all_vehicles_selected_columns["width"] *= feet_to_meters
                all_vehicles_selected_columns["height"] *= feet_to_meters
                all_vehicles_selected_columns["road_id"] = road_str
                all_vehicles_selected_columns["lane_id"] = lane
                final_dataframe = pd.concat([final_dataframe, all_vehicles_selected_columns])
        final_dataframe_sorted = final_dataframe.sort_values(["time", "road_id", "s", "id"], kind="mergesort", ascending=True)
        sorting_columns = [
            pq.SortingColumn(column_index=1, descending=False, nulls_first=True),
            pq.SortingColumn(column_index=8, descending=False, nulls_first=True),
            pq.SortingColumn(column_index=3, descending=False, nulls_first=True),
            pq.SortingColumn(column_index=0, descending=False, nulls_first=True),
        ]
        table = pa.Table.from_pandas(final_dataframe_sorted, preserve_index=False)
        pq.write_table(table, self.micro_final_path, compression="zstd", row_group_size=1000, sorting_columns=sorting_columns)

    def generate_macro_data(self):
        final_dataframe = pd.DataFrame()
        for road in self.macro_source_data:
            print(f"Generating macro data for road {road}")
            road_str = str(road)
            road_config = self.config["road_data"][road_str]
            time_origin = road_config["time_origin"]
            time_length = road_config["time_length"]
            time_step = road_config["time_step"]
            macro_data = self.macro_source_data[road].loadProcessedMacroData()

            time_list = []
            timelength_list = []
            road_id_list = []
            cell_id_list = []
            density_list = []
            velocity_list = []
            for lane in macro_data:
                lane_str = str(lane)
                macro_lane_data = macro_data[lane]
                velocity_data = np.nan_to_num(macro_lane_data["velocity"])
                density_data = np.nan_to_num(macro_lane_data["density"])
                t_shape, x_shape = velocity_data.shape[0], velocity_data.shape[1]
                for t in range(t_shape):
                    for x in range(x_shape):
                        cell_id = f"road_{road}_cell_{lane}_step_{x}"
                        current_time = (t * time_step) + time_origin
                        density = float(density_data[t][x])
                        velocity = float(velocity_data[t][x])
                        time_list.append(current_time)
                        timelength_list.append(time_step)
                        road_id_list.append(road_str)
                        cell_id_list.append(cell_id)
                        density_list.append(density)
                        velocity_list.append(velocity)
            current_dataframe = pd.DataFrame({
                "time": time_list,
                "time_length": timelength_list,
                "road_id": road_id_list,
                "cell_id": cell_id_list,
                "density": density_list,
                "velocity": velocity_list
            })
            final_dataframe = pd.concat([final_dataframe, current_dataframe])
        final_dataframe_sorted = final_dataframe.sort_values(["time", "road_id", "cell_id"], kind="mergesort", ascending=True)
        sorting_columns = [
            pq.SortingColumn(column_index=0, descending=False, nulls_first=True),
            pq.SortingColumn(column_index=2, descending=False, nulls_first=True),
            pq.SortingColumn(column_index=3, descending=False, nulls_first=True),
        ]
        table = pa.Table.from_pandas(final_dataframe_sorted, preserve_index=False)
        pq.write_table(table, self.macro_final_path, compression="zstd", row_group_size=1000, sorting_columns=sorting_columns)

if __name__ == "__main__":
    sim_data = I24SimulationData()
    sim_data.create_network_file()
    #sim_data.generate_micro_data()
    #sim_data.generate_macro_data()