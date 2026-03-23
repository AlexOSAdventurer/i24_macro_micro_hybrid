import rtree
import pandas
import json
import os
import pyarrow as pa
import pyarrow.parquet as pq
import gc

with open("./11-30-2022/6386d89efb3ff533c12df167__post10.json", "r") as f:
    full_data = json.load(f)

database_name = "road_data"
os.makedirs("road_data", exist_ok=True)
os.chdir(database_name)

roads = [1, 2] # Eastbound, Westbound
lanes = [-1, -2, -3, -4] # Innermost lanes to outermost lanes

with open("../i24_motion_to_carla_mapping_adjusted_origin.json", "r") as f:
    mapping_data = json.load(f)

mile_to_feet = 5280
feet_to_meters = 0.3048

road_lane_data = {
    1: {-1: [-24.0, -12.0], -2: [-36.0, -24.0], -3: [-48.0, -36.0], -4: [-60.0, -48.0]},
    2: {-1: [12.0, 24.0], -2: [24.0, 36.0], -3: [36.0, 48.0], -4: [48.0, 60.0]}
}

# We assume all the lanes are on the right of the reference line
def tConversionFunction(y_position):
    return ((abs(y_position) - 12.0) * feet_to_meters) * -1

def convertXPositionToMarker(x_position):
    return x_position / mile_to_feet

def westboundMarkerToS(marker):
    return (((mapping_data["2"]["origin_marker"] - marker) * mile_to_feet) * feet_to_meters) + mapping_data["2"]["origin_meter"]

def eastboundMarkerToS(marker):
    return (((marker - mapping_data["1"]["origin_marker"]) * mile_to_feet) * feet_to_meters) + mapping_data["1"]["origin_meter"]

def convertYPositionToRoadAndLaneAndT(y_position):
    t_position = tConversionFunction(y_position)
    for road in road_lane_data:
        for lane in road_lane_data[road]:
            bounds = road_lane_data[road][lane]
            if (y_position > bounds[0]) and (y_position < bounds[1]):
                return t_position, road, lane

def processDataIntoDicts(source_data, roads, lanes, unique_id_count=1):
    road_lane_trajectory_data = {}
    for road in roads:
        road_lane_trajectory_data[road] = {}
        for lane in lanes:
            road_lane_trajectory_data[road][lane] = {
                "time": [],
                "x": [],
                "y": [],
                "length": [],
                "width": [],
                "height": [],
                "class": [],
                "id": [],
                "s": [],
                "t": []
            }
    for i, entry in enumerate(source_data):
        # Create separate entry for each
        print(i, unique_id_count)
        for j in range(len(entry["x_position"])):
            new_obj_time = entry["timestamp"][j]
            new_obj_x = entry["x_position"][j]
            new_obj_y = entry["y_position"][j]
            new_obj_length = entry["length"]
            new_obj_width = entry["width"]
            new_obj_height = entry["height"]
            new_obj_class = entry["coarse_vehicle_class"]

            marker = convertXPositionToMarker(new_obj_x)
            res = convertYPositionToRoadAndLaneAndT(new_obj_y)
            if (res == None):
                continue
            new_obj_t, new_obj_road, new_obj_lane = res
            new_obj_s = None
            #Eastbound
            if new_obj_road == 1:
                new_obj_s = eastboundMarkerToS(marker)
            # Westbound
            elif new_obj_road == 2:
                new_obj_s = westboundMarkerToS(marker)

            new_obj_id = unique_id_count
            road_lane_trajectory_data[new_obj_road][new_obj_lane]["time"].append(new_obj_time)
            road_lane_trajectory_data[new_obj_road][new_obj_lane]["x"].append(new_obj_x)
            road_lane_trajectory_data[new_obj_road][new_obj_lane]["y"].append(new_obj_y)
            road_lane_trajectory_data[new_obj_road][new_obj_lane]["length"].append(new_obj_length)
            road_lane_trajectory_data[new_obj_road][new_obj_lane]["width"].append(new_obj_width)
            road_lane_trajectory_data[new_obj_road][new_obj_lane]["height"].append(new_obj_height)
            road_lane_trajectory_data[new_obj_road][new_obj_lane]["class"].append(new_obj_class)
            road_lane_trajectory_data[new_obj_road][new_obj_lane]["s"].append(new_obj_s)
            road_lane_trajectory_data[new_obj_road][new_obj_lane]["t"].append(new_obj_t)
            road_lane_trajectory_data[new_obj_road][new_obj_lane]["id"].append(new_obj_id)
        unique_id_count += 1
    return road_lane_trajectory_data

road_lane_trajectory_data = processDataIntoDicts(full_data, roads, lanes)

def generator_function(road_and_lane_data):
    print(len(road_and_lane_data["id"]))
    for i in range(len(road_and_lane_data["id"])):
        new_obj = {
            "time": road_and_lane_data["time"][i],
            "x": road_and_lane_data["x"][i],
            "y": road_and_lane_data["y"][i],
            "length": road_and_lane_data["length"][i],
            "width": road_and_lane_data["width"][i],
            "height": road_and_lane_data["height"][i],
            "class": road_and_lane_data["class"][i],
            "id": road_and_lane_data["id"][i],
            "s": road_and_lane_data["s"][i],
            "t": road_and_lane_data["t"][i]
        }
        yield (road_and_lane_data["id"][i], (new_obj["time"], new_obj["s"], new_obj["time"], new_obj["s"]), new_obj)

road_data_indices = {}
for road in roads:
    road_data_indices[road] = {}
    for lane in lanes:
        print(road, lane)
        # Build the index
        p = rtree.index.Property()
        p.dimension = 2 # Time and longitudinal position
        idx = rtree.index.Index(filename=f"road{road}lane{abs(lane)}", stream=generator_function(road_lane_trajectory_data[road][lane]), properties=p)
        road_data_indices[road][lane] = idx

del mapping_data
for road in roads:
    for lane in lanes:
        road_lane_trajectory = road_lane_trajectory_data[road][lane]
        print("Dataframe being made!")
        pandas_df = pandas.DataFrame(road_lane_trajectory)
        print("Dataframe created!")
        del road_lane_trajectory_data[road][lane]
        print("Sorting dataframe...")
        pandas_df_sorted = pandas_df.sort_values(["time", "s"], kind="mergesort", ascending=True)
        print("Dataframe sorted...")
        table = pa.Table.from_pandas(pandas_df_sorted, preserve_index=False)
        print("Parquet table made!")
        del pandas_df_sorted
        del pandas_df
        gc.collect()
        sorting_columns = [
            pq.SortingColumn(column_index=0, descending=False, nulls_first=True),
            pq.SortingColumn(column_index=8, descending=False, nulls_first=True)
        ]
        pq.write_table(table, f"road{road}lane{lane}.parquet", compression="zstd", row_group_size=1000, sorting_columns=sorting_columns)
        print("Written ", road, " and lane ", lane)