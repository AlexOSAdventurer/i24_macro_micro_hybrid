import sys
sys.path.append("..")
import i24_motion_macro
import simulation
import riemann_exact_solver_triangular
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import os
import math
import copy
from functools import partial

micro_length = 500.0

class SimplifiedSimulationDataVerification:
    def __init__(self, config_folder="verification_config", config_template="config_template.json", name="verification1", density_and_velocity_function=None, config_path="config.json", network_path="network.json", macro_path="macro.parquet", time_step=1.0, cell_length=100.0):
        self.config_folder = config_folder
        with open(config_template, "r") as f:
            self.config = json.load(f)
        self.name = name
        self.simulation_path = os.path.join(config_folder, name)
        self.config_path = os.path.join(config_folder, config_path)
        self.network_path = os.path.join(self.simulation_path, network_path)
        self.macro_final_path = os.path.join(self.simulation_path, macro_path)
        self.density_and_velocity_function = density_and_velocity_function
        os.makedirs(self.simulation_path, exist_ok=True)
        self.config["time_step"] = time_step
        self.config["cell_length"] = cell_length
        self.config["road_data"]["1"]["time_step"] = time_step
        self.config["road_data"]["1"]["cell_length"] = cell_length
        self.config["storage_locations"]["simulation_dataset"] = self.simulation_path
        self.fd = simulation.TriangularFD(v_f=self.config["fd"]["v_f"], w=self.config["fd"]["w"], rho_j=self.config["fd"]["rho_j"])
        with open(self.config_path, "w+") as f:
            json.dump(self.config, f, indent=4)

    def create_network_file(self):
        config = self.config["road_data"]["1"]
        self.network_generator = simulation.SimplifiedOneLaneRoadNetwork(fd=self.fd, lambda_lc=self.config["fd"]["lambda_lc"])
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
                density, velocity = self.density_and_velocity_function(i, j, total_cells, total_time, fd)
                new_index = (i * total_time) + j
                time_list[new_index] = (j * time_step) + time_origin
                timelength_list[new_index] = time_step
                road_id_list[new_index] = road_str
                cell_id_list[new_index] = cell_id_base_reference[i]
                density_list[new_index] = density
                velocity_list[new_index] = velocity

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

def case_m_1(i, j, total_cells, total_time, fd):
    position = float(i) / float(total_cells)
    if position < 0.1:
        return fd.rho_c, fd.velocity_from_density(fd.rho_c)
    else:
        return 0.05 * fd.rho_c, fd.velocity_from_density(0.05 * fd.rho_c)

def case_m_2(i, j, total_cells, total_time, fd):
    return 0.05 * fd.rho_c, fd.velocity_from_density(0.05 * fd.rho_c)

def case_m_3(i, j, total_cells, total_time, fd):
    position = float(i) / float(total_cells)
    if position <= 0.2:
        return 0.05 * fd.rho_c, fd.velocity_from_density(0.05 * fd.rho_c)
    else:
        return fd.rho_j, fd.velocity_from_density(fd.rho_j)

def case_m_4(i, j, total_cells, total_time, fd):
    position = float(i) / float(total_cells)
    if position <= 0.2:
        return fd.rho_j, fd.velocity_from_density(fd.rho_j)
    else:
        return 0.05 * fd.rho_c, fd.velocity_from_density(0.05 * fd.rho_c)

macro_cases = {
    "case_m_1": case_m_1,
    "case_m_2": case_m_2,
    "case_m_3": case_m_3,
    "case_m_4": case_m_4
}

def vanilla_lwr(sim_data: SimplifiedSimulationDataVerification) -> list[riemann_exact_solver_triangular.Component]:
    return []

def vanilla_mass_check(sim_data: SimplifiedSimulationDataVerification, solver: riemann_exact_solver_triangular.Solver, previous_state: list[riemann_exact_solver_triangular.Component], dt: float, previous_mass: float, eps: float = 1e-8):
    """
    print("Mass check")
    print("Previous state: ")
    for i, c in enumerate(previous_state):
        if (isinstance(c, (riemann_exact_solver_triangular.SilentBoundary, riemann_exact_solver_triangular.MovingBoundary))):
            print("Boundary ", i, c.s, c.velocity)
        elif (isinstance(c, (riemann_exact_solver_triangular.ConstantRegion))):
            print("Constant region ", i, c.s, c.region_length, c.density)
        elif (isinstance(c, (riemann_exact_solver_triangular.WaveFront))):
            print("Wave ", i, c.s, c.velocity)
        elif (isinstance(c, riemann_exact_solver_triangular.Mask)):
            print("Mask ", i, c.rear, c.front, c.s, c.region_length, c.velocity)
    print("New state: ")
    solver.print_info()
    """
    rearmost_density_region_or_surface = None
    frontmost_density_region_or_surface = None
    for (i,c) in enumerate(previous_state):
        if (isinstance(c, (riemann_exact_solver_triangular.ConstantRegion, riemann_exact_solver_triangular.MovingBoundary))):
            rearmost_density_region_or_surface = c
            break
    for (i,c) in enumerate(previous_state[::-1]):
        if (isinstance(c, (riemann_exact_solver_triangular.ConstantRegion, riemann_exact_solver_triangular.MovingBoundary))):
            frontmost_density_region_or_surface = c
            break
    assert((rearmost_density_region_or_surface is not None) and (frontmost_density_region_or_surface is not None))
    print(rearmost_density_region_or_surface.density, frontmost_density_region_or_surface.density)
    expected_mass_change = (solver.fd._flow(rearmost_density_region_or_surface.density) - solver.fd._flow(frontmost_density_region_or_surface.density)) * dt
    new_mass = solver.get_constant_region_mass()
    actual_mass_change = new_mass - previous_mass
    assert(abs(expected_mass_change - actual_mass_change) < eps), f"Expected mass change of {expected_mass_change} but instead received {actual_mass_change}, for a difference of {expected_mass_change - actual_mass_change}!"

def boundary_gmax(fd: simulation.TriangularFD, boundary_velocity: float):
    return (fd.v_f - boundary_velocity) * fd.rho_c

def boundary_demand(fd: simulation.TriangularFD, boundary_velocity: float, density: float):
    if (density > fd.rho_c):
        return boundary_gmax(fd, boundary_velocity)
    return (fd.v_f - boundary_velocity) * density

def boundary_supply(fd: simulation.TriangularFD, boundary_velocity: float, density: float):
    if (density <= fd.rho_c):
        return boundary_gmax(fd, boundary_velocity)
    return (fd.w * fd.rho_j) - ((fd.w + boundary_velocity) * density)

def micro_mass_check(sim_data: SimplifiedSimulationDataVerification, solver: riemann_exact_solver_triangular.Solver, previous_state: list[riemann_exact_solver_triangular.Component], dt: float, previous_mass: float, eps: float = 1e-8):
    print("Mass check")
    print("Previous state: ")
    for i, c in enumerate(previous_state):
        if (isinstance(c, riemann_exact_solver_triangular.SilentBoundary)):
            print("Silent Boundary ", i, c.s, c.velocity)
        elif (isinstance(c, riemann_exact_solver_triangular.MovingBoundary)):
            print("Moving Boundary ", i, c.s, c.velocity, c.density)
        elif (isinstance(c, (riemann_exact_solver_triangular.ConstantRegion))):
            print("Constant region ", i, c.s, c.region_length, c.density)
        elif (isinstance(c, (riemann_exact_solver_triangular.WaveFront))):
            print("Wave ", i, c.s, c.velocity)
        elif (isinstance(c, riemann_exact_solver_triangular.Mask)):
            print("Mask ", i, c.rear, c.front, c.s, c.region_length, c.velocity)
    print("New state: ")
    solver.print_info()
    # First, check the region to the left of the micro mask
    moving_boundary_list = [b for b in previous_state if isinstance(b, riemann_exact_solver_triangular.MovingBoundary)]
    assert(len(moving_boundary_list) <= 1), "Expected at most one moving boundary!"
    left_micro_boundary = moving_boundary_list[0] if len(moving_boundary_list) > 0 else None
    right_micro_boundary = None
    left_of_boundary_components = [c for c in previous_state if (left_micro_boundary is None) or ((left_micro_boundary.s >= (c.s - eps)) and (left_micro_boundary != c))]
    right_of_boundary_components = []

    # Default to vanilla if the left side of the micro has already left
    if (left_micro_boundary is None):
        return vanilla_mass_check(sim_data, solver, previous_state, dt, previous_mass, eps)
    else:
        silent_boundaries = [b for b in previous_state if (b.s >= moving_boundary_list[0].s) and isinstance(b, riemann_exact_solver_triangular.SilentBoundary)]
        silent_boundaries = sorted(silent_boundaries, key=lambda k: k.s)
        right_micro_boundary = silent_boundaries[0]
        right_of_boundary_components = [c for c in previous_state if ((right_micro_boundary.s <= (c.s + eps)) and (right_micro_boundary != c))]

    # Otherwise, calculate the expected net flux in the region behind the micro, and ahead.
    expected_mass_change = 0.0
    # Region behind micro
    left_constant_regions = [c for c in left_of_boundary_components if isinstance(c, riemann_exact_solver_triangular.ConstantRegion)]
    rear_left_region = left_constant_regions[0]
    rear_right_region = left_constant_regions[-1]
    expected_mass_change += (solver.fd._flow(rear_left_region.density) * dt)
    rear_micro_demand = boundary_demand(solver.fd, left_micro_boundary.velocity, rear_right_region.density)
    rear_micro_supply = boundary_supply(solver.fd, left_micro_boundary.velocity, left_micro_boundary.density)
    expected_mass_change -= (min(rear_micro_demand, rear_micro_supply) * dt)

    # Region ahead of micro
    right_constant_regions = [c for c in right_of_boundary_components if isinstance(c, riemann_exact_solver_triangular.ConstantRegion)]
    if (len(right_constant_regions) > 0):
        front_left_region = right_constant_regions[0]
        front_right_region = right_constant_regions[-1]
        expected_mass_change -= (solver.fd._flow(front_right_region.density) * dt)
        front_micro_demand = boundary_demand(solver.fd, right_micro_boundary.velocity, front_left_region.density)
        front_micro_supply = boundary_supply(solver.fd, right_micro_boundary.velocity, front_left_region.density)
        expected_mass_change += (min(front_micro_demand, front_micro_supply) * dt)
    new_mass = solver.get_constant_region_mass()
    actual_mass_change = new_mass - previous_mass
    assert(abs(expected_mass_change - actual_mass_change) < eps), f"Expected mass change of {expected_mass_change} but instead received {actual_mass_change}, for a difference of {expected_mass_change - actual_mass_change}!"

def micro_case_b_1(sim_data: SimplifiedSimulationDataVerification) -> list[riemann_exact_solver_triangular.Component]:
    road_length = sim_data.config["road_data"]["1"]["road_length"]
    boundary_start_s = (road_length * 0.2) - (micro_length / 2.0)
    boundary_end_s = boundary_start_s + micro_length
    fd = sim_data.fd
    def boundary_init_function(boundary: riemann_exact_solver_triangular.MovingBoundary, solver: riemann_exact_solver_triangular.Solver):
        boundary.velocity = 0.0
        boundary.density = fd.rho_j
    rear_boundary = riemann_exact_solver_triangular.MovingBoundary(s=boundary_start_s, boundary_init_function=boundary_init_function, delete_on_boundary_collision=True)
    front_boundary = riemann_exact_solver_triangular.SilentBoundary(s=boundary_end_s, velocity=0.0, delete_on_boundary_collision=True)
    mask = riemann_exact_solver_triangular.Mask(rear=rear_boundary, front=front_boundary)
    return [rear_boundary, front_boundary, mask]

def micro_case_b_2(sim_data: SimplifiedSimulationDataVerification) -> list[riemann_exact_solver_triangular.Component]:
    road_length = sim_data.config["road_data"]["1"]["road_length"]
    boundary_start_s = (road_length * 0.2) - (micro_length / 2.0)
    boundary_end_s = boundary_start_s + micro_length
    fd = sim_data.fd
    def boundary_init_function(boundary: riemann_exact_solver_triangular.MovingBoundary, solver: riemann_exact_solver_triangular.Solver):
        boundary.velocity = fd.v_f
        boundary.density = 0.05 * fd.rho_c
    rear_boundary = riemann_exact_solver_triangular.MovingBoundary(s=boundary_start_s, boundary_init_function=boundary_init_function, delete_on_boundary_collision=True)
    front_boundary = riemann_exact_solver_triangular.SilentBoundary(s=boundary_end_s, velocity=fd.v_f, delete_on_boundary_collision=True)
    mask = riemann_exact_solver_triangular.Mask(rear=rear_boundary, front=front_boundary)
    return [rear_boundary, front_boundary, mask]

def micro_case_b_3(sim_data: SimplifiedSimulationDataVerification) -> list[riemann_exact_solver_triangular.Component]:
    road_length = sim_data.config["road_data"]["1"]["road_length"]
    boundary_start_s = (road_length * 0.2) - (micro_length / 2.0)
    boundary_end_s = boundary_start_s + micro_length
    fd = sim_data.fd
    def boundary_init_function(boundary: riemann_exact_solver_triangular.MovingBoundary, solver: riemann_exact_solver_triangular.Solver):
        boundary.velocity = fd.velocity_from_density(2.0 * fd.rho_c)
        boundary.density = 2.0 * fd.rho_c
    rear_boundary = riemann_exact_solver_triangular.MovingBoundary(s=boundary_start_s, boundary_init_function=boundary_init_function, delete_on_boundary_collision=True)
    front_boundary = riemann_exact_solver_triangular.SilentBoundary(s=boundary_end_s, velocity=fd.velocity_from_density(2.0 * fd.rho_c), delete_on_boundary_collision=True)
    mask = riemann_exact_solver_triangular.Mask(rear=rear_boundary, front=front_boundary)
    return [rear_boundary, front_boundary, mask]

def micro_case_b_4(sim_data: SimplifiedSimulationDataVerification) -> list[riemann_exact_solver_triangular.Component]:
    road_length = sim_data.config["road_data"]["1"]["road_length"]
    boundary_start_s = (road_length * 0.2) - (micro_length / 2.0)
    boundary_end_s = boundary_start_s + micro_length
    fd = sim_data.fd
    def boundary_init_function(boundary: riemann_exact_solver_triangular.MovingBoundary, solver: riemann_exact_solver_triangular.Solver):
        boundary.velocity = fd.velocity_from_density(2.0 * fd.rho_c)
        boundary.density = 0.05 * fd.rho_c
    rear_boundary = riemann_exact_solver_triangular.MovingBoundary(s=boundary_start_s, boundary_init_function=boundary_init_function, delete_on_boundary_collision=True)
    front_boundary = riemann_exact_solver_triangular.SilentBoundary(s=boundary_end_s, velocity=fd.velocity_from_density(2.0 * fd.rho_c), delete_on_boundary_collision=True)
    mask = riemann_exact_solver_triangular.Mask(rear=rear_boundary, front=front_boundary)
    return [rear_boundary, front_boundary, mask]

def micro_case_b_5(sim_data: SimplifiedSimulationDataVerification) -> list[riemann_exact_solver_triangular.Component]:
    road_length = sim_data.config["road_data"]["1"]["road_length"]
    boundary_start_s = (road_length * 0.2) - (micro_length / 2.0)
    boundary_end_s = boundary_start_s + micro_length
    fd = sim_data.fd
    front_boundary = riemann_exact_solver_triangular.SilentBoundary(s=boundary_end_s, velocity=None, delete_on_boundary_collision=True)
    def boundary_init_function(rear_boundary: riemann_exact_solver_triangular.MovingBoundary, solver: riemann_exact_solver_triangular.Solver):
        rear_boundary.velocity = 0.0
        rear_boundary.density = fd.rho_j
        front_boundary.velocity = 0.0
        def breakpoint(solver):
            rear_boundary.velocity = fd.v_f
            front_boundary.velocity = fd.v_f
        solver.register_breakpoint(breakpoint, 10.0)
    rear_boundary = riemann_exact_solver_triangular.MovingBoundary(s=boundary_start_s, boundary_init_function=boundary_init_function, delete_on_boundary_collision=True)
    mask = riemann_exact_solver_triangular.Mask(rear=rear_boundary, front=front_boundary)
    return [rear_boundary, front_boundary, mask]

def micro_case_b_6(sim_data: SimplifiedSimulationDataVerification) -> list[riemann_exact_solver_triangular.Component]:
    road_length = sim_data.config["road_data"]["1"]["road_length"]
    boundary_start_s = (road_length * 0.2) - (micro_length / 2.0)
    boundary_end_s = boundary_start_s + micro_length
    fd = sim_data.fd
    front_boundary = riemann_exact_solver_triangular.SilentBoundary(s=boundary_end_s, velocity=None, delete_on_boundary_collision=True)
    def boundary_init_function(rear_boundary: riemann_exact_solver_triangular.MovingBoundary, solver: riemann_exact_solver_triangular.Solver):
        rear_boundary.velocity = fd.v_f
        rear_boundary.density = fd.rho_j
        front_boundary.velocity = fd.v_f
        def breakpoint(solver):
            rear_boundary.density = 0.05 * fd.rho_c
        solver.register_breakpoint(breakpoint, 10.0)
    rear_boundary = riemann_exact_solver_triangular.MovingBoundary(s=boundary_start_s, boundary_init_function=boundary_init_function, delete_on_boundary_collision=True)
    mask = riemann_exact_solver_triangular.Mask(rear=rear_boundary, front=front_boundary)
    return [rear_boundary, front_boundary, mask]

def micro_case_b_7(sim_data: SimplifiedSimulationDataVerification) -> list[riemann_exact_solver_triangular.Component]:
    road_length = sim_data.config["road_data"]["1"]["road_length"]
    boundary_start_s = (road_length * 0.2) - (micro_length / 2.0)
    boundary_end_s = boundary_start_s + micro_length
    fd = sim_data.fd
    front_boundary = riemann_exact_solver_triangular.SilentBoundary(s=boundary_end_s, velocity=None, delete_on_boundary_collision=True)
    def boundary_init_function(rear_boundary: riemann_exact_solver_triangular.MovingBoundary, solver: riemann_exact_solver_triangular.Solver):
        rear_boundary.velocity = fd.v_f
        rear_boundary.density = 0.05 * fd.rho_c
        front_boundary.velocity = fd.v_f
        def breakpoint(solver):
            rear_boundary.velocity = 0.0
            front_boundary.velocity = 0.0
        solver.register_breakpoint(breakpoint, 10.0)
    rear_boundary = riemann_exact_solver_triangular.MovingBoundary(s=boundary_start_s, boundary_init_function=boundary_init_function, delete_on_boundary_collision=True)
    mask = riemann_exact_solver_triangular.Mask(rear=rear_boundary, front=front_boundary)
    return [rear_boundary, front_boundary, mask]

def micro_case_b_8(sim_data: SimplifiedSimulationDataVerification) -> list[riemann_exact_solver_triangular.Component]:
    road_length = sim_data.config["road_data"]["1"]["road_length"]
    boundary_start_s = (road_length * 0.2) - (micro_length / 2.0)
    boundary_end_s = boundary_start_s + micro_length
    fd = sim_data.fd
    front_boundary = riemann_exact_solver_triangular.SilentBoundary(s=boundary_end_s, velocity=None, delete_on_boundary_collision=True)
    def boundary_init_function(rear_boundary: riemann_exact_solver_triangular.MovingBoundary, solver: riemann_exact_solver_triangular.Solver):
        rear_boundary.velocity = 0.0
        rear_boundary.density = 0.05 * fd.rho_c
        front_boundary.velocity = 0.0
        def breakpoint(solver):
            rear_boundary.density = fd.rho_j
        solver.register_breakpoint(breakpoint, 10.0)
    rear_boundary = riemann_exact_solver_triangular.MovingBoundary(s=boundary_start_s, boundary_init_function=boundary_init_function, delete_on_boundary_collision=True)
    mask = riemann_exact_solver_triangular.Mask(rear=rear_boundary, front=front_boundary)
    return [rear_boundary, front_boundary, mask]

micro_cases = {
    "vanilla_lwr": {
        "function": vanilla_lwr,
        "mass_check": vanilla_mass_check,
        "macro_cases": [
            "case_m_1",
            "case_m_2",
            "case_m_3",
            "case_m_4"
        ]
    },
    "micro_case_b_1": {
        "function": micro_case_b_1,
        "mass_check": micro_mass_check,
        "macro_cases": [
            "case_m_2",
            "case_m_3",
        ]
    },
    "micro_case_b_2": {
        "function": micro_case_b_2,
        "mass_check": micro_mass_check,
        "macro_cases": [
            "case_m_2",
            "case_m_4",
        ]
    },
    "micro_case_b_3": {
        "function": micro_case_b_3,
        "mass_check": micro_mass_check,
        "macro_cases": [
            "case_m_2",
        ]
    },
    "micro_case_b_4": {
        "function": micro_case_b_4,
        "mass_check": micro_mass_check,
        "macro_cases": [
            "case_m_1",
        ]
    },
    "micro_case_b_5": {
        "function": micro_case_b_5,
        "mass_check": micro_mass_check,
        "macro_cases": [
            "case_m_1",
        ]
    },
    "micro_case_b_6": {
        "function": micro_case_b_6,
        "mass_check": micro_mass_check,
        "macro_cases": [
            "case_m_4",
        ]
    },
    "micro_case_b_7": {
        "function": micro_case_b_7,
        "mass_check": micro_mass_check,
        "macro_cases": [
            "case_m_3",
        ]
    },
    "micro_case_b_8": {
        "function": micro_case_b_8,
        "mass_check": micro_mass_check,
        "macro_cases": [
            "case_m_3",
        ]
    }
}

if __name__ == "__main__":
    dx_sweep = [128.0 / (2.0 ** n) for n in range(5)]
    dt_sweep = [1.0 / (2.0 ** n) for n in range(5)]
    for macro_case in macro_cases:
        func = macro_cases[macro_case]
        for x, t in zip(dx_sweep, dt_sweep):
            name = f"{macro_case}_dx_{x}_dt_{t}"
            sim_data = SimplifiedSimulationDataVerification(
                name=name,
                density_and_velocity_function=func,
                config_path=f"{name}.json",
                time_step=t,
                cell_length=x
            )
            sim_data.create_network_file()
            sim_data.generate_macro_data()
            print(f"{name} initial data done!")
            print(f"Generating {name} simulation data!")
            network_path = sim_data.network_path
            macro_path = sim_data.macro_final_path
            time_origin = sim_data.config["time_origin"]
            time_length = sim_data.config["time_length"]
            road_id = "1"
            lane_id = -1
            min_s = 0.0
            max_s = sim_data.config["road_data"]["1"]["road_length"]
            for micro_case in micro_cases:
                if macro_case not in micro_cases[micro_case]["macro_cases"]:
                    continue
                final_data = pd.DataFrame()
                data_path = os.path.join(sim_data.simulation_path, f"{micro_case}.csv")
                lwr_solver = riemann_exact_solver_triangular.Solver.generate_solver_from_macro_data(network_path=network_path, macro_data_path=macro_path, fd = sim_data.fd, t=time_origin, road_id=road_id, lane_id=lane_id, min_s=min_s, max_s=max_s, additional_components=micro_cases[micro_case]["function"](sim_data))
                while ((lwr_solver.t + time_origin) <= time_length):
                    #previous_mass = lwr_solver.get_constant_region_mass()
                    #previous_state = copy.deepcopy(lwr_solver.current_state)
                    """
                    for c in lwr_solver.current_state:
                        if (isinstance(c, (riemann_exact_solver_triangular.SilentBoundary, riemann_exact_solver_triangular.MovingBoundary))):
                            print("Boundary ", c.s, c.velocity)
                        elif (isinstance(c, (riemann_exact_solver_triangular.ConstantRegion))):
                            print("Constant region ", c.s, c.region_length, c.density)
                        elif (isinstance(c, (riemann_exact_solver_triangular.WaveFront))):
                            print("Wave ", c.s, c.velocity)
                    wave_fronts = [f for f in lwr_solver.current_state if isinstance(f, riemann_exact_solver_triangular.WaveFront)]
                    """
                    current_data = lwr_solver.generate_density_field()
                    final_data = pd.concat([final_data, current_data])
                    lwr_solver.advance(t, advance_hook=partial(micro_cases[micro_case]["mass_check"], sim_data))
                    #micro_cases[micro_case]["mass_check"](sim_data, lwr_solver, previous_state, t, previous_mass)
                    print(lwr_solver.t)
                final_data.to_csv(data_path, index=False)
                print(f"{micro_case} done!")