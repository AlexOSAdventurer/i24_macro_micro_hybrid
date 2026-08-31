import i24_motion_data
import i24_motion_macro
import simulation
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import os
"""
[I 2026-06-05 03:16:21,985] Trial 99 finished with value: -1.4247049284076403 and parameters: {'v_f': 36.833188698970794, 'rho_j': 0.13426972640115076, 'lambda_lc': 0.12607510767478558, 'w': 5.244198786834116}. Best is trial 41 with value: -1.4780850809769102.
{'v_f': 38.82661919086127, 'rho_j': 0.13273852897987434, 'lambda_lc': 0.12801469008205105, 'w': 5.52936062831337}

"""

class I24SimulationData:
    def __init__(self, config_path="i24_motion_to_dataset.json", network_path="network.json", network_metanet_path="network_metanet.json", macro_path="macro.parquet", macro_metanet_path="macro_metanet.parquet", micro_path="micro.parquet"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.config_path = config_path
        self.preprocessing_path = self.config["storage_locations"]["preprocessing_data"]
        self.simulation_path = self.config["storage_locations"]["simulation_dataset"]
        self.network_path = os.path.join(self.simulation_path, network_path)
        self.network_metanet_path = os.path.join(self.simulation_path, network_metanet_path)
        self.macro_final_path = os.path.join(self.simulation_path, macro_path)
        self.macro_metanet_path = os.path.join(self.simulation_path, macro_metanet_path)
        self.micro_final_path = os.path.join(self.simulation_path, micro_path)
        self.micro_source_data = {}
        self.macro_source_data = {}
        os.makedirs(self.simulation_path, exist_ok=True)
        self.load_source_data()

    def create_network_file(self):
        # For now both roads have the same length, so just grab one of them for config.
        config = self.config["road_data"]["2"]
        params = {'v_f': 30.019341712559083, 'rho_j': 0.09411385875052518, 'lambda_lc': 0.16914326352090997, 'w': 4.573064692495487} #. Best is trial 71 with value: -1.6869119514065773.
        #params = {'v_f': 44.628171971976364, 'rho_j': 0.10161810757336569, 'lambda_lc': 0.12233207260740411, 'w': 4.939175703300891}
        #params = {'v_f': 43.906521944351404, 'rho_j': 0.13154571531898504, 'lambda_lc': 0.0789091812007707, 'w': 6.054218545965104}
        #params_original = {'v_f': 49.73562026160161, "rho_j": 0.1304577157114563, "lambda_lc": 0.13951724538021434, 'w': 5.697835695812354}
        triangular_fd = simulation.TriangularFD(v_f=params['v_f'], w=params['w'], rho_j=params['rho_j'])
        self.network_generator = simulation.I24WestAndEastNetwork(fd=triangular_fd, lambda_lc=params['lambda_lc'])
        #triangular_fd = simulation.TriangularFD(v_f=49.816011505539535, w=6.053452290089522, rho_j=0.12998583138493472)
        #self.network_generator = simulation.I24WestAndEastNetwork(fd=triangular_fd, lambda_lc=0.10076621081371681)
        #greenshields_fd = simulation.GreenshieldsFD(v_f=42.634906282620406, rho_j=0.06711901063946596)
        #self.network_generator = simulation.I24WestAndEastNetwork(fd=greenshields_fd, lambda_lc=0.28911022854691354)
        self.network_generator.create_network(config["road_length"], config["cell_length"], config["lanes"], lane_width=config["lane_width"])
        self.network_generator.save_network(self.network_path)

    def create_network_file_metanet(self):
        # Same corridor as create_network_file, but with each carriageway's lanes
        # collapsed into a single cell per longitudinal step -- the lane-aggregated
        # mesh METANET is written on. Cell density is then the total across lanes, so
        # METANETParams keeps the paper's PER-LANE rho_crit/kappa and carries the lane
        # count in `lanes` (recorded here as metanet_lanes_per_road; it is also
        # config["road_data"][road]["lanes"], since the saved network only sees one
        # lane per road).
        config = self.config["road_data"]["2"]
        # The FD is decorative under METANET (the model supplies its own demand and
        # supply), but it still drives the rollout colouring and any first-order run
        # on this mesh; the collapsed generator scales its rho_j by the lane count.
        triangular_fd = simulation.TriangularFD(v_f=49.73562026160161, w=5.697835695812354, rho_j=0.1304577157114563)
        # lambda_lc is unused: a single lane per road leaves no adjacent pair for the
        # lane-change model, so the collapsed cells carry NoLaneChange.
        self.network_generator_metanet = simulation.I24WestAndEastNetworkCollapsed(fd=triangular_fd, lambda_lc=0.13951724538021434)
        self.network_generator_metanet.create_network(config["road_length"], config["cell_length"], config["lanes"], lane_width=config["lane_width"])
        self.metanet_lanes_per_road = self.network_generator_metanet.lanes_per_road
        self.network_generator_metanet.save_network(self.network_metanet_path)

    def load_source_data(self):
        for road in self.config["road_data"]:
            self.micro_source_data[int(road)] = i24_motion_data.I24MotionData(int(road), self.config["road_data"][road]["time_origin"], self.config["road_data"][road]["time_origin"] + self.config["road_data"][road]["time_length"], 0.0, self.config["road_data"][road]["road_length"], config_path=self.config_path)
            self.macro_source_data[int(road)] = i24_motion_macro.I24MotionMacro(self.micro_source_data[int(road)], int(road), f"road_{road}", config_path=self.config_path)

    def generate_micro_data(self):
        feet_to_meters = 0.3048
        final_dataframe = pd.DataFrame()
        for road in self.micro_source_data:
            print(f"Generating micro data for road {road}")
            road_str = str(road)
            road_config = self.config["road_data"][road_str]
            all_vehicles = self.micro_source_data[road].query_edie_box_batch([road_config["time_origin"]], [road_config["time_origin"] + road_config["time_length"]], [0.0], [road_config["road_length"]])[0]
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
            macro_data = self.macro_source_data[road].load_processed_macro_data()

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

    def generate_macro_data_metanet(self):
        # Ground truth on the collapsed mesh of create_network_file_metanet: one row per
        # (time, road, longitudinal station) instead of one per lane, with cell ids
        # matching I24WestAndEastNetworkCollapsed's `road_{road}_cell_collapsed_step_{x}`.
        #
        # Density SUMS over lanes, because the collapsed cell's density is likewise a
        # total across lanes. Mean speed is FLOW-weighted, Sum(rho*v) / Sum(rho), since
        # speed is intensive, not additive; a station with no vehicles in any lane falls
        # back to the unweighted mean over lanes. Outage cells are zero in the source
        # data (nan_to_num, as in generate_macro_data), so they aggregate to zero here.
        final_dataframe = pd.DataFrame()
        for road in self.macro_source_data:
            print(f"Generating collapsed macro data for road {road}")
            road_str = str(road)
            road_config = self.config["road_data"][road_str]
            time_origin = road_config["time_origin"]
            time_step = road_config["time_step"]
            macro_data = self.macro_source_data[road].load_processed_macro_data()

            density_total = None
            flow_total = None
            velocity_sum = None
            lane_count = 0
            for lane in macro_data:
                macro_lane_data = macro_data[lane]
                # float64 accumulation: the source arrays are float32, and summing four
                # lanes (plus the rho*v weighting) in float32 costs ~1e-5 m/s of speed.
                velocity_data = np.nan_to_num(macro_lane_data["velocity"]).astype(np.float64)
                density_data = np.nan_to_num(macro_lane_data["density"]).astype(np.float64)
                if density_total is None:
                    density_total = np.zeros_like(density_data)
                    flow_total = np.zeros_like(density_data)
                    velocity_sum = np.zeros_like(velocity_data)
                elif density_data.shape != density_total.shape:
                    raise ValueError(
                        f"Road {road} lane {lane} macro grid {density_data.shape} does not "
                        f"match {density_total.shape}: lanes cannot be collapsed."
                    )
                density_total += density_data
                flow_total += density_data * velocity_data
                velocity_sum += velocity_data
                lane_count += 1

            if lane_count == 0:
                continue
            safe = np.where(density_total > 1e-12, density_total, 1.0)
            velocity_total = np.where(
                density_total > 1e-12, flow_total / safe, velocity_sum / float(lane_count)
            )

            t_shape, x_shape = density_total.shape[0], density_total.shape[1]
            time_list = []
            timelength_list = []
            road_id_list = []
            cell_id_list = []
            density_list = []
            velocity_list = []
            for t in range(t_shape):
                for x in range(x_shape):
                    time_list.append((t * time_step) + time_origin)
                    timelength_list.append(time_step)
                    road_id_list.append(road_str)
                    cell_id_list.append(f"road_{road}_cell_collapsed_step_{x}")
                    density_list.append(float(density_total[t][x]))
                    velocity_list.append(float(velocity_total[t][x]))
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
        pq.write_table(table, self.macro_metanet_path, compression="zstd", row_group_size=1000, sorting_columns=sorting_columns)

if __name__ == "__main__":
    #sim_data = I24SimulationData()
    #sim_data.create_network_file()
    #sim_data.create_network_file_metanet()
    #sim_data.generate_micro_data()
    #sim_data.generate_macro_data_metanet()
    #sim_data.generate_macro_data()
    config_folder = "config/"
    datasets = os.listdir(config_folder)
    for dataset in datasets:
        print(dataset)
        config_path = os.path.join(config_folder, dataset)
        with open(config_path, "r") as f:
            config = json.load(f)
        sim_data = I24SimulationData(config_path=config_path)
        sim_data.create_network_file()
        sim_data.create_network_file_metanet()
        #sim_data.generate_micro_data()
        #sim_data.generate_macro_data_metanet()
        #sim_data.generate_macro_data()