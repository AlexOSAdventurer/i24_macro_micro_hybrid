import i24_motion_data
import itertools
import numpy
import pickle
import os

import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def fft_four_convs(Dp, Mp, k_cong, k_free, eps=1e-12, use_ortho=True):
    """
    Compute via FFT:
        sum_cong = conv2d(Dp, k_cong)
        sum_free = conv2d(Dp, k_free)
        N_cong   = conv2d(Mp, k_cong)
        N_free   = conv2d(Mp, k_free)
    Inputs:
      Dp, Mp:   (B, C, H, W)
      k_cong,:  (F, C, Kh, Kw)
      k_free:   (F, C, Kh, Kw)
    Returns:
      sum_cong, N_cong, sum_free, N_free each of shape (B, F, H-Kh+1, W-Kw+1)
    """
    #  sanitize inputs ———
    Dp = torch.nan_to_num(Dp, nan=0.0, posinf=0.0, neginf=0.0)
    Mp = torch.nan_to_num(Mp, nan=0.0, posinf=0.0, neginf=0.0)

    B, C, H, W        = Dp.shape
    F, _, Kh, Kw      = k_cong.shape
    Fh, Fw            = H + Kh - 1, W + Kw - 1
    device, dtype     = Dp.device, Dp.dtype

    # ——— pad inputs ———
    print(B, C, Fh, Fw)
    Dp_pad = torch.zeros(B, C, Fh, Fw, device=device, dtype=dtype)
    Mp_pad = torch.zeros(B, C, Fh, Fw, device=device, dtype=dtype)
    Dp_pad[..., :H, :W] = Dp
    Mp_pad[..., :H, :W] = Mp

    # ——— pad kernels ———
    k1_pad = torch.zeros(F, C, Fh, Fw, device=device, dtype=dtype)
    k2_pad = torch.zeros(F, C, Fh, Fw, device=device, dtype=dtype)
    k1_pad[..., :Kh, :Kw] = k_cong
    k2_pad[..., :Kh, :Kw] = k_free

    # choose normalization
    norm = "ortho" if use_ortho else None

    # ——— FFT both inputs and kernels ———
    Df  = torch.fft.rfftn(Dp_pad, dim=(-2, -1), s=(Fh, Fw), norm=norm)
    Mf  = torch.fft.rfftn(Mp_pad, dim=(-2, -1), s=(Fh, Fw), norm=norm)
    Kf1 = torch.fft.rfftn(k1_pad, dim=(-2, -1), s=(Fh, Fw), norm=norm)
    Kf2 = torch.fft.rfftn(k2_pad, dim=(-2, -1), s=(Fh, Fw), norm=norm)

    # ——— pointwise multiply in freq domain ———
    Y1 = Df * Kf1    # for sum_cong
    Y2 = Df * Kf2    # for sum_free
    Z1 = Mf * Kf1    # for N_cong
    Z2 = Mf * Kf2    # for N_free

    # ——— inverse FFT back to real ———
    y1 = torch.fft.irfftn(Y1, dim=(-2, -1), s=(Fh, Fw), norm=norm)
    y2 = torch.fft.irfftn(Y2, dim=(-2, -1), s=(Fh, Fw), norm=norm)
    z1 = torch.fft.irfftn(Z1, dim=(-2, -1), s=(Fh, Fw), norm=norm)
    z2 = torch.fft.irfftn(Z2, dim=(-2, -1), s=(Fh, Fw), norm=norm)

    # ——— crop “valid” region ———
    oh, ow = H - Kh + 1, W - Kw + 1
    sum_cong = y1[..., Kh-1:Kh-1+oh, Kw-1:Kw-1+ow]
    sum_free = y2[..., Kh-1:Kh-1+oh, Kw-1:Kw-1+ow]
    N_cong   = z1[..., Kh-1:Kh-1+oh, Kw-1:Kw-1+ow]
    N_free   = z2[..., Kh-1:Kh-1+oh, Kw-1:Kw-1+ow]

    # ——— optional epsilon to counts to avoid zero division downstream ———
    N_cong = N_cong + eps
    N_free = N_free + eps

    return sum_cong, N_cong, sum_free, N_free

class AdaptiveSmoothing(nn.Module):
    def __init__(self,
                 kernel_time_window: float,
                 kernel_space_window: float,
                 dx: float,
                 dt: float,
                 init_delta: float = 0.02, # mile
                 init_tau: float = 4.0, # seconds
                 init_c_cong: float = 12.0,
                 init_c_free: float = -45.0,
                 init_v_thr: float = 40.0,
                 init_v_delta: float = 10.0,
                 high_is_congestion=False):
        super().__init__()
        self.size_t = int(kernel_time_window / dt)
        self.size_x = int(kernel_space_window / dx)
        self.dt = dt
        self.dx = dx

        t_offs = torch.arange(-self.size_t, self.size_t + 1) * dt
        # print(t_offs)
        x_offs = torch.arange(-self.size_x, self.size_x + 1) * dx
        # print(x_offs)
        #X, T = torch.meshgrid(x_offs, t_offs, indexing='ij')
        T, X = torch.meshgrid(t_offs, x_offs, indexing='ij')
        self.register_buffer('T_offsets', T.float())
        self.register_buffer('X_offsets', X.float())

        self.delta   = nn.Parameter(torch.tensor(init_delta))
        self.tau     = nn.Parameter(torch.tensor(init_tau))
        self.c_cong  = nn.Parameter(torch.tensor(init_c_cong))
        self.c_free  = nn.Parameter(torch.tensor(init_c_free))
        self.v_thr   = nn.Parameter(torch.tensor(init_v_thr))
        self.v_delta = nn.Parameter(torch.tensor(init_v_delta))
        self.high_is_congestion = high_is_congestion

    def forward(self, raw_data: torch.Tensor, w: torch.Tensor = None):
        # Ensure input is 4D: (B, C, T, X)
        
        if raw_data.ndim == 2:
            raw_data = raw_data.unsqueeze(0).unsqueeze(0)
        elif raw_data.ndim == 3:
            raw_data = raw_data.unsqueeze(1)

        mask = (~raw_data.isnan()).float()
        data = torch.nan_to_num(raw_data, nan=0.0)

        c_cong_s = self.c_cong #/ 3600.0 Already at meters per second
        c_free_s = self.c_free #/ 3600.0 Already at meters per second
        t_cong = self.T_offsets - self.X_offsets / c_cong_s
        t_free = self.T_offsets - self.X_offsets / c_free_s

        k_cong = torch.exp(-(t_cong.abs() / self.tau + self.X_offsets.abs() / self.delta))
        # size of k_cong
        k_free = torch.exp(-(t_free.abs() / self.tau + self.X_offsets.abs() / self.delta))

        k_cong = k_cong.unsqueeze(0).unsqueeze(0)  # (1,1,Kt,Kx)
        k_free = k_free.unsqueeze(0).unsqueeze(0)

        #pad = (self.size_t, self.size_t, self.size_x, self.size_x) # to deal with the edge effects
        pad = (self.size_x, self.size_x, self.size_t, self.size_t) # to deal with the edge effects
        Dp = F.pad(data, pad, value=0.0)
        Mp = F.pad(mask, pad, value=0.0)

        sum_cong = F.conv2d(Dp, k_cong)
        N_cong   = F.conv2d(Mp, k_cong)
        sum_free = F.conv2d(Dp, k_free)
        N_free   = F.conv2d(Mp, k_free)
        # use FFT to compute the convolutions
        #sum_cong, N_cong, sum_free, N_free = fft_four_convs(Dp, Mp, k_cong, k_free)

        v_cong = sum_cong / N_cong
        v_free = sum_free / N_free

        if (w is None):
            if (self.high_is_congestion):
                v_max = torch.max(v_cong, v_free)
                w = 0.5 * (1 + torch.tanh((v_max - self.v_thr) / self.v_delta))
            else:
                v_min = torch.min(v_cong, v_free)
                w = 0.5 * (1 + torch.tanh((self.v_thr - v_min) / self.v_delta))
        
        v = w * v_cong + (1 - w) * v_free
        valid_cong = (N_cong > 0).float()
        valid_free = (N_free > 0).float()
        # if no cong data → use free; if no free data → use cong
        v = valid_cong*valid_free*v + (1-valid_cong)*v_free + (1-valid_free)*v_cong
        # check if there's nan if so print
        if torch.isnan(v).any():
            print("Warning! NaN detected in output")
            print(N_cong)
        # print size of v
        return v.squeeze(1), w

class I24MotionMacro:
    def __init__(self, data_source, road_id, data_folder, config_path="i24_motion_to_dataset.json", longitudinal_cell_size=100.0, time_delta=1.0, max_velocity=45.0, min_velocity=-15.0):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.data_source = data_source
        self.road_id = road_id
        self.longitudinal_cell_size = longitudinal_cell_size
        self.time_delta = time_delta
        self.data_folder = os.path.join(self.config["storage_locations"]["preprocessing_data"], data_folder)
        self.max_velocity = max_velocity
        self.min_velocity = min_velocity
        os.makedirs(self.data_folder, exist_ok=True)
        self.lanes = i24_motion_data.I24MotionData.road_lane_lookup[road_id]
        self.lane_asm_calibrations = i24_motion_data.I24MotionData.road_lane_asm_lookup[road_id]

    def computeBox(self, time_index, long_cell_index):
        return {
            "time": {"min": (time_index * self.time_delta) + self.data_source.timestamp_min, "max": ((time_index + 1) * self.time_delta) + self.data_source.timestamp_min},
            "position": {"min": long_cell_index * self.longitudinal_cell_size, "max": (long_cell_index + 1) * self.longitudinal_cell_size}
        }
    
    @staticmethod
    def vehiclesInBox(vehicles):
        vehicles_sorted_by_time = vehicles.sort_values(by='time', ascending=True)
        vehicle_ids = vehicles_sorted_by_time["id"].unique().tolist()
        result = {}
        for id in vehicle_ids:
            result[id] = vehicles_sorted_by_time[vehicles_sorted_by_time["id"] == id]
        return result
    
    def computeEdieBoxQueries(self):
        timestamp_mins = numpy.arange(self.data_source.timestamp_min, self.data_source.timestamp_max, self.time_delta)
        s_mins = numpy.arange(self.data_source.s_min, self.data_source.s_max, self.longitudinal_cell_size)
        queries = {"timestamp_min": [], "timestamp_max": [], "s_min": [], "s_max": []}
        for r in itertools.product(timestamp_mins, s_mins):
            queries["timestamp_min"].append(r[0])
            queries["timestamp_max"].append(r[0] + self.time_delta)
            queries["s_min"].append(r[1])
            queries["s_max"].append(r[1] + self.longitudinal_cell_size)
        return queries
    
    def queryEdieBoxes(self, queries):
        return self.data_source.queryEdieBoxBatch(queries["timestamp_min"], queries["timestamp_max"], queries["s_min"], queries["s_max"])
    
    def pickleEdieBoxResults(self, query_results):
        path = os.path.join(self.data_folder, "raw_trajectories.pickle")
        with open(path, "wb") as file:
            pickle.dump(query_results, file)

    def unpickleEdieBoxResults(self):
        path = os.path.join(self.data_folder, "raw_trajectories.pickle")
        with open(path, "rb") as file:
            return pickle.load(file)
        
    def pickleRawMacroData(self, lane, raw_macro):
        folder = os.path.join(self.data_folder, str(lane))
        data_file = os.path.join(folder, "raw_macro.pickle")
        os.makedirs(folder, exist_ok=True)
        with open(data_file, "wb") as file:
            pickle.dump(raw_macro, file)

    def unpickleRawMacroData(self, lane):
        folder = os.path.join(self.data_folder, str(lane))
        data_file = os.path.join(folder, "raw_macro.pickle")
        with open(data_file, "rb") as file:
            return pickle.load(file)
        
    def pickleProcessedMacroData(self, lane, processed_macro):
        folder = os.path.join(self.data_folder, str(lane))
        data_file = os.path.join(folder, "processed_macro.pickle")
        os.makedirs(folder, exist_ok=True)
        with open(data_file, "wb") as file:
            pickle.dump(processed_macro, file)

    def unpickleProcessedMacroData(self, lane):
        folder = os.path.join(self.data_folder, str(lane))
        data_file = os.path.join(folder, "processed_macro.pickle")
        with open(data_file, "rb") as file:
            return pickle.load(file)
    
    @staticmethod
    def edieTotals(vehicles, time_delta, sample_delta=None):
        """Edie's total time spent (veh*s) and total distance travelled (veh*m) in one box.

        `vehicles` maps vehicle id -> that vehicle's samples inside the box, ordered by
        time. Two things matter for the totals to be unbiased:

        - Each sample stands for one sampling period of presence, so a vehicle's time
          in the box is its sample span PLUS one period. Taking the bare span drops up
          to a period per vehicle (~4% of a 1 s box at 25 Hz), and a vehicle caught by
          a single sample would come out at zero.
        - Distance is measured along `s`, the coordinate the box is cut in, from first
          to last sample rather than as a path length. Endpoint displacement is what
          the old estimator used and it does not accumulate per-sample position noise.

        Together these give q = TTD/(L*T) = rho*v exactly, which counting vehicle ids
        did not: `len(vehicles)/L` weights every vehicle as if it occupied the cell for
        the whole window, over-estimating density by roughly (L + v*T)/L -- about 35%
        at 34.5 m/s on a 100 m x 1 s box, and worst exactly where speeds are highest.
        """
        total_time = 0.0
        total_distance = 0.0
        periods = []
        clipped = 0
        for vehicle_id in vehicles:
            entries = vehicles[vehicle_id]
            times = entries["time"].to_numpy(dtype=numpy.float64)
            if len(times) < 2:
                # One sample carries no displacement, so it can contribute time but no
                # distance, which would drag the mean speed down. Settle these below at
                # the box's own mean speed instead of dropping them from the density.
                clipped += 1
                continue
            span = float(times[-1] - times[0])
            if span < 1e-8:
                clipped += 1
                continue
            period = (float(numpy.median(numpy.diff(times))) if sample_delta is None
                      else float(sample_delta))
            periods.append(period)
            # A vehicle cannot be inside the box for longer than the box lasts; the
            # query bounds are inclusive at both ends, so span can already equal T.
            occupancy = min(span + period, float(time_delta))
            positions = entries["s"].to_numpy(dtype=numpy.float64)
            speed = abs(float(positions[-1] - positions[0])) / span
            total_time += occupancy
            total_distance += speed * occupancy
        if clipped and total_time > 1e-8:
            period = float(numpy.median(periods)) if periods else float(sample_delta or 0.0)
            mean_speed = total_distance / total_time
            total_time += clipped * period
            total_distance += clipped * period * mean_speed
        return total_time, total_distance

    @staticmethod
    def density(total_time, longitudinal_cell_size, time_delta):
        """Edie's generalized density: TTS / (L*T)."""
        return float(total_time) / (float(longitudinal_cell_size) * float(time_delta))

    @staticmethod
    def velocity(total_time, total_distance, max_velocity, min_velocity):
        """Edie's space-mean speed: TTD / TTS.

        This is a time-in-box weighted mean of the vehicles' speeds. The old unweighted
        mean of per-vehicle speeds gave a vehicle that merely clipped the corner of the
        box the same weight as one that sat in it for the whole window, which biases
        the estimate toward the fast vehicles (they are the ones passing through).
        """
        if (total_time < 1e-8):
            return float('nan')
        return max(min(float(total_distance) / float(total_time), max_velocity), min_velocity)

    @staticmethod
    def flow(density, velocity):
        if (math.isfinite(velocity) and math.isfinite(density)):
            return density * velocity
        else:
            return float('nan')
    
    @staticmethod
    def computeMacroData(queries, query_results, lane, longitudinal_cell_size, time_delta, max_velocity, min_velocity):
        density_result = numpy.zeros(len(query_results), dtype=numpy.float32)
        velocity_result = numpy.zeros(len(query_results), dtype=numpy.float32)
        flow_result = numpy.zeros(len(query_results), dtype=numpy.float32)
        for i, (timestamp_min, timestamp_max, s_min, s_max) in enumerate(zip(queries["timestamp_min"], queries["timestamp_max"], queries["s_min"], queries["s_max"])):
            if (lane in query_results[i]):
                lane_data = query_results[i][lane]
                vehicles = I24MotionMacro.vehiclesInBox(lane_data)
                total_time, total_distance = I24MotionMacro.edieTotals(vehicles, time_delta)
                density_result[i] = I24MotionMacro.density(total_time, longitudinal_cell_size, time_delta)
                velocity_result[i] = I24MotionMacro.velocity(total_time, total_distance, max_velocity, min_velocity)
                flow_result[i] = I24MotionMacro.flow(density_result[i], velocity_result[i])
            if ((i % 1000) == 0):
                print(i)
        return density_result, velocity_result, flow_result
    
    def createRawMacroData(self):
        queries = self.computeEdieBoxQueries()
        #query_results = self.unpickleEdieBoxResults()
        query_results = self.queryEdieBoxes(queries)
        self.pickleEdieBoxResults(query_results)
        raw_macro_data = {}
        for lane in self.lanes:
            raw_macro_data[lane] = {}
            (lane_density, lane_velocity, lane_flow) = I24MotionMacro.computeMacroData(queries, query_results, lane, self.longitudinal_cell_size, self.time_delta, self.max_velocity, self.min_velocity)
            raw_macro_data[lane]["density"] = lane_density
            raw_macro_data[lane]["velocity"] = lane_velocity
            raw_macro_data[lane]["flow"] = lane_flow
            self.pickleRawMacroData(lane, raw_macro_data[lane])

    def loadRawMacroData(self):
        raw_macro_data = {}
        for lane in self.lanes:
            raw_macro_data[lane] = {}
            raw_macro_data[lane] = self.unpickleRawMacroData(lane)
        return raw_macro_data
    
    def createProcessedMacroData(self):
        raw_macro_data = self.loadRawMacroData()
        print("Raw data loaded!")
        s_size = self.data_source.s_max - self.data_source.s_min
        t_size = self.data_source.timestamp_max - self.data_source.timestamp_min
        device_count = torch.cuda.device_count()
        outage_locations = self.config["road_data"][str(self.road_id)]["outage_locations"]
        with torch.no_grad():
            outage_mask = []
            queries = self.computeEdieBoxQueries()
            for i, entry in enumerate(zip(queries['s_min'], queries['s_max'])):
                s_min, s_max = entry[0], entry[1]
                outage_mask.append(False)
                for outage in outage_locations:
                    if (s_min >= outage[0]) and (s_min <= outage[1]):
                        outage_mask[-1] = True
            
            for lane in self.lanes:
                asm_velocity = AdaptiveSmoothing(t_size, s_size, dx=self.longitudinal_cell_size, dt=self.time_delta, 
                                        init_delta=self.lane_asm_calibrations[lane]["init_delta"], 
                                        init_tau=self.lane_asm_calibrations[lane]["init_tau"],
                                        init_c_cong=self.lane_asm_calibrations[lane]["init_c_cong"],
                                        init_c_free=self.lane_asm_calibrations[lane]["init_c_free"],
                                        init_v_thr=self.lane_asm_calibrations[lane]["init_v_thr"],
                                        init_v_delta=self.lane_asm_calibrations[lane]["init_v_delta"]).to(device=f"cuda:{0 % device_count}")
                asm_density = AdaptiveSmoothing(t_size, s_size, dx=self.longitudinal_cell_size, dt=self.time_delta, 
                                        init_delta=self.lane_asm_calibrations[lane]["init_delta"], 
                                        init_tau=self.lane_asm_calibrations[lane]["init_tau"],
                                        init_c_cong=self.lane_asm_calibrations[lane]["init_c_cong"],
                                        init_c_free=self.lane_asm_calibrations[lane]["init_c_free"],
                                        init_v_thr=self.lane_asm_calibrations[lane]["init_v_thr"],
                                        init_v_delta=self.lane_asm_calibrations[lane]["init_v_delta"]).to(device=f"cuda:{1 % device_count}")
                print(f"Processing lane {lane}")
                processed_macro_data = {}
                raw_macro_data_lane = raw_macro_data[lane]
                x_shape = int(round(s_size / self.longitudinal_cell_size, 0))
                t_shape = int(round(t_size / self.time_delta, 0))

                velocity_asm_input = raw_macro_data_lane["velocity"].reshape(-1).copy()
                velocity_asm_input[outage_mask] = numpy.nan
                velocity_asm_input = velocity_asm_input.reshape((1, 1, t_shape, x_shape)).copy()
                velocity_asm_input = torch.from_numpy(velocity_asm_input).to(torch.float32).to(device=f"cuda:{0 % device_count}")

                density_asm_input = raw_macro_data_lane["density"].reshape(-1).copy()
                density_asm_input[outage_mask] = numpy.nan
                density_asm_input = density_asm_input.reshape((1, 1, t_shape, x_shape)).copy()
                density_asm_input = torch.from_numpy(density_asm_input).to(torch.float32).to(device=f"cuda:{1 % device_count}")
                print(f"Lane data loaded!")

                velocity_asm_output, w = asm_velocity(velocity_asm_input)
                w = w.cpu().to(device=f"cuda:{1 % device_count}") # Send to density CUDA device
                print("Velocity done!")
                # We use the weights from the previous velocity reconstruction
                density_asm_output, w = asm_density(density_asm_input, w)
                print("Density done!")
                flow_asm_output = velocity_asm_output * density_asm_output.to(device=f"cuda:{0 % device_count}")
                print("Flow done!")

                velocity_asm_output = velocity_asm_output.cpu().detach().numpy().reshape((t_shape, x_shape))
                density_asm_output = density_asm_output.cpu().detach().numpy().reshape((t_shape, x_shape))
                flow_asm_output = flow_asm_output.cpu().detach().numpy().reshape((t_shape, x_shape))

                processed_macro_data["velocity"] = velocity_asm_output
                processed_macro_data["density"] = density_asm_output
                processed_macro_data["flow"] = flow_asm_output
                print("Pickling processed data!")
                self.pickleProcessedMacroData(lane, processed_macro_data)
                print(f"Picked lane {lane}!")

    def loadProcessedMacroData(self):
        processed_macro_data = {}
        for lane in self.lanes:
            processed_macro_data[lane] = {}
            processed_macro_data[lane] = self.unpickleProcessedMacroData(lane)
        return processed_macro_data
        
if __name__ == "__main__":
    """
    print("Loading data source...")
    data_source = i24_motion_data.I24MotionData(2, 1669812350, 1669812350+3600, 0, 1600)
    print("Creating macro processing object...")
    macro = I24MotionMacro(data_source, 2, "road_2")
    print("Creating raw macro data and saving it...")
    #macro.createRawMacroData()
    print("Creating processed macro and saving it...")
    macro.createProcessedMacroData()
    print("Loading data source...")
    data_source = i24_motion_data.I24MotionData(1, 1669812350, 1669812350+3600, 0, 1600)
    print("Creating macro processing object...")
    macro = I24MotionMacro(data_source, 1, "road_1")
    print("Creating raw macro data and saving it...")
    macro.createRawMacroData()
    print("Creating processed macro and saving it...")
    macro.createProcessedMacroData()
    """
    config_folder = "config/"
    datasets = os.listdir(config_folder)
    for dataset in datasets:
        config_path = os.path.join(config_folder, dataset)
        with open(config_path, "r") as f:
            config = json.load(f)
        time_origin = config["time_origin"]
        time_length = config["time_length"]
        road_length_1 = config["road_data"]["1"]["road_length"]
        road_length_2 = config["road_data"]["2"]["road_length"]
        print(f"Loading data source for {dataset}...")
        data_source = i24_motion_data.I24MotionData(2, time_origin, time_origin+time_length, 0, road_length_1, config_path=config_path)
        print("Creating macro processing object...")
        macro = I24MotionMacro(data_source, 2, "road_2", config_path=config_path)
        print("Creating raw macro data and saving it...")
        macro.createRawMacroData()
        print("Creating processed macro and saving it...")
        macro.createProcessedMacroData()
        print("Loading data source...")
        data_source = i24_motion_data.I24MotionData(1, time_origin, time_origin+time_length, 0, road_length_2, config_path=config_path)
        print("Creating macro processing object...")
        macro = I24MotionMacro(data_source, 1, "road_1", config_path=config_path)
        print("Creating raw macro data and saving it...")
        macro.createRawMacroData()
        print("Creating processed macro and saving it...")
        macro.createProcessedMacroData()