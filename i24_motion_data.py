import pyarrow.parquet as pq
import duckdb
import pandas
import os
import sys
import json

class I24MotionData:
    trajectories_all_db_prefix = "trajectories_all"
    trajectories_subset_db_prefix = "trajectories_subset"
    road_lane_lookup = {
        1: [-1, -2, -3, -4],
        2: [-1, -2, -3, -4]
    }
    def __init__(self, road_id, timestamp_min, timestamp_max, s_min, s_max, data_folder="road_data/", config_path="i24_motion_to_dataset.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.data_folder = os.path.join(self.config["storage_locations"]["preprocessing_data"], data_folder)
        self.road_id = road_id
        self.timestamp_min = timestamp_min
        self.timestamp_max = timestamp_max
        self.s_min = s_min
        self.s_max = s_max
        self.conn = duckdb.connect()
        self.preloadAllData()
        self.preloadSubset()

    def preloadAllData(self):
        for lane in self.road_lane_lookup[self.road_id]:
            lane_str = str(lane) if lane >= 0 else "neg"+str(abs(lane))
            ref = f"{self.trajectories_all_db_prefix}_{self.road_id}_{lane_str}"
            file_path = os.path.join(self.data_folder, f"road{self.road_id}lane{lane}.parquet")
            self.conn.execute(f""" 
                CREATE VIEW {ref} AS
                SELECT * FROM parquet_scan('{file_path}')             
            """)

    def preloadSubset(self):
        for lane in self.road_lane_lookup[self.road_id]:
            lane_str = str(lane) if lane >= 0 else "neg"+str(abs(lane))
            source_ref = f"{self.trajectories_all_db_prefix}_{self.road_id}_{lane_str}"
            dest_ref = f"{self.trajectories_subset_db_prefix}_{self.road_id}_{lane_str}"
            self.directQueryEdieBoxDB(source_ref, dest_ref, self.timestamp_min, self.timestamp_max, self.s_min, self.s_max)

    def directQueryEdieBoxDF(self, source_ref, timestamp_min, timestamp_max, s_min, s_max):
        q = f"""
        SELECT time, x, y, length, width, height, class, id, s, t
        FROM {source_ref}
        WHERE time BETWEEN ? AND ?
        AND s BETWEEN ? AND ?
        """
        return self.conn.execute(q, [timestamp_min, timestamp_max, s_min, s_max]).fetch_df()
    
    def directQueryEdieBoxDB(self, source_ref, dest_ref, timestamp_min, timestamp_max, s_min, s_max):
        q = f"""
        CREATE TABLE {dest_ref} AS
        SELECT time, x, y, length, width, height, class, id, s, t
        FROM {source_ref}
        WHERE time BETWEEN ? AND ?
        AND s BETWEEN ? AND ?
        """
        return self.conn.execute(q, [timestamp_min, timestamp_max, s_min, s_max])
    
    def getVehicleTrajectoryDF(self, source_ref, id, start_time=None):
        if start_time is None:
            q = f"""
            SELECT time, x, y, length, width, height, class, id, s, t
            FROM {source_ref}
            WHERE id = ?
            ORDER BY time ASC
            """
            return self.conn.execute(q, [id]).fetch_df()
        else:
            q = f"""
            SELECT time, x, y, length, width, height, class, id, s, t
            FROM {source_ref}
            WHERE id = ?
            AND time >= ?
            ORDER BY time ASC
            """
            return self.conn.execute(q, [id, start_time]).fetch_df() 
    
    def getVehicleTrajectory(self, lane, id, start_time=None):
        lane_str = str(lane) if lane >= 0 else "neg"+str(abs(lane))
        source_ref = f"{self.trajectories_subset_db_prefix}_{self.road_id}_{lane_str}"
        return self.getVehicleTrajectoryDF(source_ref, id, start_time=start_time)
    
    def queryEdieBoxSubset(self, timestamp_min, timestamp_max, s_min, s_max, ignore_ids = None):
        result = {}
        for lane in self.road_lane_lookup[self.road_id]:
            lane_str = str(lane) if lane >= 0 else "neg"+str(abs(lane))
            source_ref = f"{self.trajectories_subset_db_prefix}_{self.road_id}_{lane_str}"
            result[lane] = self.directQueryEdieBoxDF(source_ref, timestamp_min, timestamp_max, s_min, s_max)
            # Each unique id needs at least two rows corresponding to it so we can estimate velocity and justifiably say it's a reliable track
            unique_ids = list(result[lane]["id"].unique())
            for id in unique_ids:
                subset = result[lane][result[lane]["id"] == id]
                if (len(subset) < 2):
                    result[lane] = result[lane][result[lane]["id"] != id]
            if ignore_ids is not None:
                for ignore_id in ignore_ids:
                    result[lane] = result[lane][result[lane]["id"] != ignore_id]
        return result
    
    def queryEdieBoxBatch(self, timestamp_min, timestamp_max, s_min, s_max):
        query_ids = list(range(len(timestamp_min)))
        windows = pandas.DataFrame({
            "query_id": query_ids,
            "timestamp_min": timestamp_min,
            "timestamp_max": timestamp_max,
            "s_min": s_min,
            "s_max": s_max
        })
        self.conn.register("windows", windows)
        result = {}
        for query in query_ids:
            result[query] = {}
        for lane in self.road_lane_lookup[self.road_id]:
            #print(lane)
            lane_str = str(lane) if lane >= 0 else "neg"+str(abs(lane))
            source_ref = f"{self.trajectories_subset_db_prefix}_{self.road_id}_{lane_str}"
            q = f"""
            SELECT w.query_id AS query_id,
            LIST(p.time ORDER BY p.time) AS time,
            LIST(p.x ORDER BY p.time) AS x,
            LIST(p.y ORDER BY p.time) AS y,
            LIST(p.length ORDER BY p.time) AS length,
            LIST(p.width ORDER BY p.time) AS width,
            LIST(p.height ORDER BY p.time) AS height,
            LIST(p.class ORDER BY p.time) AS class,
            LIST(p.id ORDER BY p.time) AS id,
            LIST(p.s ORDER BY p.time) AS s,
            LIST(p.t ORDER BY p.time) AS t
            FROM {source_ref} p
            JOIN windows w
            ON p.time BETWEEN w.timestamp_min AND w.timestamp_max
            AND p.s BETWEEN w.s_min AND w.s_max 
            GROUP BY w.query_id
            """
            intermediate_result = self.conn.execute(q).fetch_df()
            #print(len(intermediate_result))
            for i in range(len(intermediate_result)):
                row = intermediate_result.iloc[i]
                query_id = int(row["query_id"])
                local_dataframe = pandas.DataFrame({
                    "time": row["time"],
                    "x": row["x"],
                    "y": row["y"],
                    "length": row["length"],
                    "width": row["width"],
                    "height": row["height"],
                    "class": row["class"],
                    "id": row["id"],
                    "s": row["s"],
                    "t": row["t"]
                })
                result[query_id][lane] = local_dataframe
        return result