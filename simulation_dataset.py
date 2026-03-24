import i24_motion_data
import i24_motion_macro
import simulation
import numpy as np
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
        self.load_source_data()

    def create_network_file(self):
        self.network_generator = simulation.I24WestAndEastNetwork()
        self.network_generator.create_network()
        self.network_generator.save_network(self.network_path)

    def load_source_data(self):
        for road in self.config["road_data"]:
            self.micro_source_data[int(road)] = i24_motion_data.I24MotionData(int(road), self.config["road_data"][road]["time_origin"], self.config["road_data"][road]["time_origin"] + self.config["road_data"][road]["time_length"], 0.0, self.config["road_data"][road]["road_length"])
            self.macro_source_data[int(road)] = i24_motion_macro.I24MotionMacro(self.micro_source_data[int(road)], int(road), f"road_{road}")
