import pyarrow.parquet as pq
import duckdb
import numpy
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
    road_lane_asm_lookup = {
        1: {
            -1: {
                "init_delta": 140.0,
                "init_tau": 9.27,
                "init_c_cong": -5.48,
                "init_c_free": 22.53,
                "init_v_thr": 22.16, 
                "init_v_delta": 4.52,
            },
            -2: {
                "init_delta": 140.0,
                "init_tau": 7.86,
                "init_c_cong": -5.61,
                "init_c_free": 22.26,
                "init_v_thr": 18.78, 
                "init_v_delta": 3.44,
            },
            -3: {
                "init_delta": 140.0,
                "init_tau": 8.25,
                "init_c_cong": -5.84,
                "init_c_free": 22.62,
                "init_v_thr": 18.31, 
                "init_v_delta": 3.66,
            },
            -4: {
                "init_delta": 160.0,
                "init_tau": 11.56,
                "init_c_cong": -5.71,
                "init_c_free": 26.82,
                "init_v_thr": 16.51, 
                "init_v_delta": 3.94
                ,
            },
        },
        2: {
            -1: {
                "init_delta": 140.0,
                "init_tau": 9.27,
                "init_c_cong": -5.48,
                "init_c_free": 22.53,
                "init_v_thr": 22.16, 
                "init_v_delta": 4.52,
            },
            -2: {
                "init_delta": 140.0,
                "init_tau": 7.86,
                "init_c_cong": -5.61,
                "init_c_free": 22.26,
                "init_v_thr": 18.78, 
                "init_v_delta": 3.44,
            },
            -3: {
                "init_delta": 140.0,
                "init_tau": 8.25,
                "init_c_cong": -5.84,
                "init_c_free": 22.62,
                "init_v_thr": 18.31, 
                "init_v_delta": 3.66,
            },
            -4: {
                "init_delta": 160.0,
                "init_tau": 11.56,
                "init_c_cong": -5.71,
                "init_c_free": 26.82,
                "init_v_thr": 16.51, 
                "init_v_delta": 3.94
                ,
            },
        }
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
        self.conn = duckdb.connect(config={'memory_limit': '160GB', 'threads': 48})
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
    
    @staticmethod
    def uniformBoxGrid(timestamp_min, timestamp_max, s_min, s_max, tol=1e-6):
        """Recognise a batch of Edie windows as a regular tiling of (time, s).

        The batches `I24MotionMacro.computeEdieBoxQueries` builds are always a
        uniform grid walked in `itertools.product(times, positions)` order, which
        means the box a sample belongs to is arithmetic rather than a search. The
        general range join cannot know that and costs O(points * windows); the
        binned form costs one pass. Returns the grid as
        `(t_origin, dt, n_t, s_origin, dx, n_x)` when the batch qualifies, so that
        `query_id == ti * n_x + xi`, and None when it does not, in which case the
        caller must fall back to the range join.
        """
        n = len(timestamp_min)
        if (n < 2):
            # A single window is already cheap to join and, unlike a tiling, has no
            # neighbour to hand its upper edge to. Leave it on the general path.
            return None
        t_lo = numpy.asarray(timestamp_min, dtype=numpy.float64)
        t_hi = numpy.asarray(timestamp_max, dtype=numpy.float64)
        s_lo = numpy.asarray(s_min, dtype=numpy.float64)
        s_hi = numpy.asarray(s_max, dtype=numpy.float64)

        t_vals = numpy.unique(t_lo)
        s_vals = numpy.unique(s_lo)
        n_t, n_x = len(t_vals), len(s_vals)
        if (n_t * n_x != n):
            return None

        dt, dx = float(t_hi[0] - t_lo[0]), float(s_hi[0] - s_lo[0])
        if (dt <= 0.0) or (dx <= 0.0):
            return None
        # every box the same size...
        if numpy.any(numpy.abs((t_hi - t_lo) - dt) > tol) or numpy.any(numpy.abs((s_hi - s_lo) - dx) > tol):
            return None
        # ...abutting its neighbours exactly, so flooring lands in the right box...
        if (n_t > 1) and numpy.any(numpy.abs(numpy.diff(t_vals) - dt) > tol):
            return None
        if (n_x > 1) and numpy.any(numpy.abs(numpy.diff(s_vals) - dx) > tol):
            return None
        # ...and laid out in product order, so the flat index below matches query_id
        if numpy.any(numpy.abs(t_lo - numpy.repeat(t_vals, n_x)) > tol):
            return None
        if numpy.any(numpy.abs(s_lo - numpy.tile(s_vals, n_t)) > tol):
            return None
        return float(t_vals[0]), dt, n_t, float(s_vals[0]), dx, n_x

    @staticmethod
    def collectBoxRows(intermediate_result, result, lane):
        """Fan the one-row-per-box query result out into per-box DataFrames."""
        columns = ["time", "x", "y", "length", "width", "height", "class", "id", "s", "t"]
        query_ids = intermediate_result["query_id"].to_numpy()
        lists = {column: intermediate_result[column].to_list() for column in columns}
        for i in range(len(query_ids)):
            local_dataframe = pandas.DataFrame({column: lists[column][i] for column in columns})
            result[int(query_ids[i])][lane] = local_dataframe

    def queryEdieBoxGrid(self, grid):
        """Batch Edie box query for a uniform tiling, binning instead of joining.

        Note the box bounds are half open here, [lo, hi), where the range join in
        `queryEdieBoxBatch` uses inclusive BETWEEN on both ends. A sample sitting
        exactly on a shared edge therefore lands in one box rather than being
        counted in both, and one sitting on the outer edge of the tiled domain is
        dropped rather than folded into the last box.
        """
        t_origin, dt, n_t, s_origin, dx, n_x = grid
        result = {}
        for query in range(n_t * n_x):
            result[query] = {}
        for lane in self.road_lane_lookup[self.road_id]:
            lane_str = str(lane) if lane >= 0 else "neg"+str(abs(lane))
            source_ref = f"{self.trajectories_subset_db_prefix}_{self.road_id}_{lane_str}"
            q = f"""
            WITH binned AS (
                SELECT CAST(floor((p.time - {t_origin!r}) / {dt!r}) AS BIGINT) AS ti,
                CAST(floor((p.s - {s_origin!r}) / {dx!r}) AS BIGINT) AS xi,
                p.time, p.x, p.y, p.length, p.width, p.height, p.class, p.id, p.s, p.t
                FROM {source_ref} p
            )
            SELECT b.ti * {n_x} + b.xi AS query_id,
            LIST(b.time ORDER BY b.time) AS time,
            LIST(b.x ORDER BY b.time) AS x,
            LIST(b.y ORDER BY b.time) AS y,
            LIST(b.length ORDER BY b.time) AS length,
            LIST(b.width ORDER BY b.time) AS width,
            LIST(b.height ORDER BY b.time) AS height,
            LIST(b.class ORDER BY b.time) AS class,
            LIST(b.id ORDER BY b.time) AS id,
            LIST(b.s ORDER BY b.time) AS s,
            LIST(b.t ORDER BY b.time) AS t
            FROM binned b
            WHERE b.ti >= 0 AND b.ti < {n_t} AND b.xi >= 0 AND b.xi < {n_x}
            GROUP BY query_id
            """
            self.collectBoxRows(self.conn.execute(q).fetch_df(), result, lane)
        return result

    def queryEdieBoxBatch(self, timestamp_min, timestamp_max, s_min, s_max, use_grid_fast_path=True):
        if use_grid_fast_path:
            grid = self.uniformBoxGrid(timestamp_min, timestamp_max, s_min, s_max)
            if grid is not None:
                return self.queryEdieBoxGrid(grid)
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
            self.collectBoxRows(self.conn.execute(q).fetch_df(), result, lane)
        return result