from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
import copy
import json

import jax.numpy as jnp
import numpy as np
import pandas as pd


# =========================
# Core data model
# =========================

Connection = Tuple[str, str]  # (road_id, cell_id)


@dataclass
class Cell:
    road_id: str
    cell_id: str
    lane: int
    start_s: float
    end_s: float
    density: float
    inflow_connections: List[Connection] = field(default_factory=list)
    outflow_connections: List[Connection] = field(default_factory=list)

    @property
    def length(self) -> float:
        return float(self.end_s - self.start_s)

    def validate(self) -> None:
        if self.end_s <= self.start_s:
            raise ValueError(
                f"Cell {self.road_id}/{self.cell_id} has non-positive length: "
                f"start_s={self.start_s}, end_s={self.end_s}"
            )
        if self.density < 0.0:
            raise ValueError(
                f"Cell {self.road_id}/{self.cell_id} has negative density: {self.density}"
            )


@dataclass
class Road:
    road_id: str
    polyline: List[Tuple[float, float]]
    cells: Dict[str, Cell] = field(default_factory=dict)

    def validate(self) -> None:
        if len(self.polyline) < 2:
            raise ValueError(f"Road {self.road_id} must have at least 2 polyline points.")
        for cell in self.cells.values():
            if cell.road_id != self.road_id:
                raise ValueError(
                    f"Cell {cell.cell_id} road_id mismatch: {cell.road_id} != {self.road_id}"
                )
            cell.validate()


@dataclass
class Network:
    network_id: str
    roads: Dict[str, Road] = field(default_factory=dict)

    def validate(self) -> None:
        for road_id, road in self.roads.items():
            if road.road_id != road_id:
                raise ValueError(f"Road dict key mismatch: {road_id} != {road.road_id}")
            road.validate()

        # Check all connections exist
        for road in self.roads.values():
            for cell in road.cells.values():
                for nbr_road_id, nbr_cell_id in cell.inflow_connections:
                    self.get_cell(nbr_road_id, nbr_cell_id)
                for nbr_road_id, nbr_cell_id in cell.outflow_connections:
                    self.get_cell(nbr_road_id, nbr_cell_id)

    def get_cell(self, road_id: str, cell_id: str) -> Cell:
        if road_id not in self.roads:
            raise KeyError(f"Road '{road_id}' not found.")
        if cell_id not in self.roads[road_id].cells:
            raise KeyError(f"Cell '{cell_id}' not found in road '{road_id}'.")
        return self.roads[road_id].cells[cell_id]

    def clone(self) -> "Network":
        return copy.deepcopy(self)

    def all_cell_keys(self) -> List[Connection]:
        keys: List[Connection] = []
        for road_id, road in self.roads.items():
            for cell_id in road.cells:
                keys.append((road_id, cell_id))
        return keys

    def to_dict(self) -> Dict[str, Any]:
        return {
            "network_id": self.network_id,
            "roads": {
                road_id: {
                    "road_id": road.road_id,
                    "polyline": [[float(x), float(y)] for x, y in road.polyline],
                    "cells": {
                        cell_id: {
                            "road_id": cell.road_id,
                            "cell_id": cell.cell_id,
                            "lane": int(cell.lane),
                            "start_s": float(cell.start_s),
                            "end_s": float(cell.end_s),
                            "density": float(cell.density),
                            "inflow_connections": [
                                [r, c] for (r, c) in cell.inflow_connections
                            ],
                            "outflow_connections": [
                                [r, c] for (r, c) in cell.outflow_connections
                            ],
                        }
                        for cell_id, cell in road.cells.items()
                    },
                }
                for road_id, road in self.roads.items()
            },
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Network":
        roads: Dict[str, Road] = {}

        for road_id, road_data in data["roads"].items():
            cells: Dict[str, Cell] = {}
            for cell_id, cell_data in road_data["cells"].items():
                cells[cell_id] = Cell(
                    road_id=cell_data["road_id"],
                    cell_id=cell_data["cell_id"],
                    lane=int(cell_data["lane"]),
                    start_s=float(cell_data["start_s"]),
                    end_s=float(cell_data["end_s"]),
                    density=float(cell_data["density"]),
                    inflow_connections=[
                        (str(x[0]), str(x[1]))
                        for x in cell_data.get("inflow_connections", [])
                    ],
                    outflow_connections=[
                        (str(x[0]), str(x[1]))
                        for x in cell_data.get("outflow_connections", [])
                    ],
                )

            roads[road_id] = Road(
                road_id=road_data["road_id"],
                polyline=[(float(x), float(y)) for x, y in road_data["polyline"]],
                cells=cells,
            )

        network = Network(network_id=data["network_id"], roads=roads)
        network.validate()
        return network

    @staticmethod
    def from_json(json_path: str) -> "Network":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Network.from_dict(data)

    def to_json(self, json_path: str) -> None:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# =========================
# Ground-truth data model
# =========================

REQUIRED_GT_COLUMNS = {
    "time",
    "time_length",
    "road_id",
    "cell_id",
    "density",
    "velocity",
    "inflow",
    "outflow",
}


class GroundTruthStore:
    """
    Thin wrapper over a parquet table of time-indexed cell statistics.
    """

    def __init__(self, df: pd.DataFrame):
        missing = REQUIRED_GT_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Ground-truth parquet missing columns: {sorted(missing)}")

        self.df = df.copy()
        self.df["road_id"] = self.df["road_id"].astype(str)
        self.df["cell_id"] = self.df["cell_id"].astype(str)
        self.df["time"] = self.df["time"].astype(float)
        self.df["time_length"] = self.df["time_length"].astype(float)
        self.df["density"] = self.df["density"].astype(float)
        self.df["velocity"] = self.df["velocity"].astype(float)
        self.df["inflow"] = self.df["inflow"].astype(float)
        self.df["outflow"] = self.df["outflow"].astype(float)

        self.df.sort_values(["time", "road_id", "cell_id"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    @staticmethod
    def from_parquet(parquet_path: str) -> "GroundTruthStore":
        df = pd.read_parquet(parquet_path)
        return GroundTruthStore(df)

    def snapshot_at_time(
        self, time_value: float, tolerance: float = 1e-6
    ) -> pd.DataFrame:
        """
        Return rows for the nearest matching timestamp within tolerance.
        """
        unique_times = self.df["time"].unique()
        idx = np.argmin(np.abs(unique_times - time_value))
        chosen_time = float(unique_times[idx])

        if abs(chosen_time - time_value) > tolerance:
            raise KeyError(
                f"No ground-truth snapshot near time={time_value}. "
                f"Closest available is {chosen_time}."
            )

        out = self.df[self.df["time"] == chosen_time].copy()
        out.reset_index(drop=True, inplace=True)
        return out

    def apply_density_snapshot_to_network(
        self, network: Network, time_value: float, tolerance: float = 1e-6
    ) -> None:
        snapshot = self.snapshot_at_time(time_value, tolerance=tolerance)
        for _, row in snapshot.iterrows():
            road_id = str(row["road_id"])
            cell_id = str(row["cell_id"])
            density = float(row["density"])
            network.get_cell(road_id, cell_id).density = density


# =========================
# CTM / LWR Simulation
# =========================

@dataclass
class RolloutStep:
    sim_time: float
    network: Network


class Simulation:
    """
    Simplified LWR-CTM simulator.

    Notes:
    - Uses a triangular fundamental diagram.
    - Uses equal split at diverges.
    - Uses equal-priority supply allocation at merges.
    - Keeps densities in Cell objects, but uses jax.numpy internally each step.
    """

    def __init__(
        self,
        network: Network,
        time_resolution: float,
        origin_time: float,
        free_flow_speed: float,
        congestion_wave_speed: float,
        jam_density: float,
        inflow_boundary_map: Optional[Dict[Connection, float]] = None,
        outflow_boundary_map: Optional[Dict[Connection, float]] = None,
    ):
        """
        Args:
            network: Network object.
            time_resolution: dt in seconds.
            origin_time: UNIX start time.
            free_flow_speed: v_f in m/s.
            congestion_wave_speed: w in m/s (positive scalar).
            jam_density: rho_j in veh/m.
            inflow_boundary_map:
                Optional map for source cells with no predecessors:
                max inflow into that cell in veh/s.
            outflow_boundary_map:
                Optional map for sink cells with no successors:
                max outflow from that cell in veh/s.
        """
        self.network = network
        self.time_resolution = float(time_resolution)
        self.origin_time = float(origin_time)
        self.current_time = float(origin_time)
        self.rollout_results: List[RolloutStep] = []

        self.free_flow_speed = float(free_flow_speed)
        self.congestion_wave_speed = float(congestion_wave_speed)
        self.jam_density = float(jam_density)

        if self.free_flow_speed <= 0.0:
            raise ValueError("free_flow_speed must be > 0.")
        if self.congestion_wave_speed <= 0.0:
            raise ValueError("congestion_wave_speed must be > 0.")
        if self.jam_density <= 0.0:
            raise ValueError("jam_density must be > 0.")

        # Capacity for triangular FD
        self.capacity = (
            self.free_flow_speed
            * self.congestion_wave_speed
            * self.jam_density
            / (self.free_flow_speed + self.congestion_wave_speed)
        )

        self.inflow_boundary_map = inflow_boundary_map or {}
        self.outflow_boundary_map = outflow_boundary_map or {}

        self.network.validate()

    # -------------------------
    # Network loaders
    # -------------------------

    @staticmethod
    def from_json(
        json_path: str,
        time_resolution: float,
        origin_time: float,
        free_flow_speed: float,
        congestion_wave_speed: float,
        jam_density: float,
        inflow_boundary_map: Optional[Dict[Connection, float]] = None,
        outflow_boundary_map: Optional[Dict[Connection, float]] = None,
    ) -> "Simulation":
        network = Network.from_json(json_path)
        return Simulation(
            network=network,
            time_resolution=time_resolution,
            origin_time=origin_time,
            free_flow_speed=free_flow_speed,
            congestion_wave_speed=congestion_wave_speed,
            jam_density=jam_density,
            inflow_boundary_map=inflow_boundary_map,
            outflow_boundary_map=outflow_boundary_map,
        )

    # -------------------------
    # Fundamental diagram
    # -------------------------

    def demand(self, rho: jnp.ndarray) -> jnp.ndarray:
        """
        Sending flow S(rho) in veh/s.
        """
        return jnp.minimum(self.free_flow_speed * rho, self.capacity)

    def supply(self, rho: jnp.ndarray) -> jnp.ndarray:
        """
        Receiving flow R(rho) in veh/s.
        """
        return jnp.minimum(
            self.capacity,
            self.congestion_wave_speed * jnp.maximum(self.jam_density - rho, 0.0),
        )

    def velocity_from_density(self, rho: jnp.ndarray) -> jnp.ndarray:
        """
        Simple velocity estimate consistent with q = rho * v:
        v = min(v_f, q(rho)/rho), with v=0 if rho=0.
        """
        q = jnp.minimum(
            self.free_flow_speed * rho,
            self.congestion_wave_speed * jnp.maximum(self.jam_density - rho, 0.0),
        )
        return jnp.where(rho > 1e-12, q / rho, self.free_flow_speed)

    # -------------------------
    # State helpers
    # -------------------------

    def _cell_key_order(self) -> List[Connection]:
        return self.network.all_cell_keys()

    def _densities_to_jax(self, cell_order: List[Connection]) -> jnp.ndarray:
        return jnp.array(
            [self.network.get_cell(r, c).density for (r, c) in cell_order],
            dtype=jnp.float32,
        )

    def _lengths_to_jax(self, cell_order: List[Connection]) -> jnp.ndarray:
        return jnp.array(
            [self.network.get_cell(r, c).length for (r, c) in cell_order],
            dtype=jnp.float32,
        )

    def _write_densities_back(
        self, cell_order: List[Connection], densities: jnp.ndarray
    ) -> None:
        densities_np = np.asarray(densities, dtype=np.float64)
        for i, (r, c) in enumerate(cell_order):
            self.network.get_cell(r, c).density = float(densities_np[i])

    def _snapshot(self) -> None:
        self.rollout_results.append(
            RolloutStep(sim_time=self.current_time, network=self.network.clone())
        )

    # -------------------------
    # Flow allocation
    # -------------------------

    def _compute_edge_flows(
        self,
        rho_map: Dict[Connection, float],
    ) -> Dict[Tuple[Connection, Connection], float]:
        """
        Compute flow on each directed edge predecessor -> successor.

        Simplification:
        - Diverge: predecessor demand split equally among successors.
        - Merge: successor supply apportioned equally among predecessors.
        - Edge flow = min(allocated predecessor demand, allocated successor supply)
        - Then per-successor rescaling enforces total incoming flow <= successor supply
        - Then per-predecessor rescaling enforces total outgoing flow <= predecessor demand

        This is a reasonable simple baseline for a one-way highway.
        """
        cell_keys = list(rho_map.keys())

        demand_map: Dict[Connection, float] = {}
        supply_map: Dict[Connection, float] = {}

        for key in cell_keys:
            rho = float(rho_map[key])
            d = min(self.free_flow_speed * rho, self.capacity)
            s = min(self.capacity, self.congestion_wave_speed * max(self.jam_density - rho, 0.0))
            demand_map[key] = d
            supply_map[key] = s

        # Raw per-edge proposals
        edge_flow: Dict[Tuple[Connection, Connection], float] = {}

        # First pass: equal split across outgoing and incoming degree
        for u in cell_keys:
            u_cell = self.network.get_cell(*u)
            successors = u_cell.outflow_connections

            if len(successors) == 0:
                continue

            per_out_demand = demand_map[u] / float(len(successors))

            for v in successors:
                v_cell = self.network.get_cell(*v)
                predecessors_of_v = v_cell.inflow_connections
                if len(predecessors_of_v) == 0:
                    per_in_supply = supply_map[v]
                else:
                    per_in_supply = supply_map[v] / float(len(predecessors_of_v))

                edge_flow[(u, v)] = min(per_out_demand, per_in_supply)

        # Enforce successor supply
        for v in cell_keys:
            incoming_edges = [e for e in edge_flow if e[1] == v]
            if not incoming_edges:
                continue
            total_in = sum(edge_flow[e] for e in incoming_edges)
            cap_in = supply_map[v]
            if total_in > cap_in and total_in > 1e-12:
                scale = cap_in / total_in
                for e in incoming_edges:
                    edge_flow[e] *= scale

        # Enforce predecessor demand
        for u in cell_keys:
            outgoing_edges = [e for e in edge_flow if e[0] == u]
            if not outgoing_edges:
                continue
            total_out = sum(edge_flow[e] for e in outgoing_edges)
            cap_out = demand_map[u]
            if total_out > cap_out and total_out > 1e-12:
                scale = cap_out / total_out
                for e in outgoing_edges:
                    edge_flow[e] *= scale

        return edge_flow

    def _compute_boundary_flows(
        self,
        rho_map: Dict[Connection, float],
    ) -> Tuple[Dict[Connection, float], Dict[Connection, float]]:
        """
        Boundary source/sink flows for cells with no predecessors / no successors.

        Returns:
            external_inflow[cell]  in veh/s
            external_outflow[cell] in veh/s
        """
        external_inflow: Dict[Connection, float] = {}
        external_outflow: Dict[Connection, float] = {}

        for key, rho in rho_map.items():
            cell = self.network.get_cell(*key)
            rho_val = float(rho)
            demand_val = min(self.free_flow_speed * rho_val, self.capacity)
            supply_val = min(
                self.capacity,
                self.congestion_wave_speed * max(self.jam_density - rho_val, 0.0),
            )

            if len(cell.inflow_connections) == 0:
                upstream_max = float(self.inflow_boundary_map.get(key, self.capacity))
                external_inflow[key] = min(upstream_max, supply_val)

            if len(cell.outflow_connections) == 0:
                downstream_max = float(self.outflow_boundary_map.get(key, self.capacity))
                external_outflow[key] = min(demand_val, downstream_max)

        return external_inflow, external_outflow

    # -------------------------
    # Simulation step
    # -------------------------

    def step(self) -> None:
        """
        Advance by one timestep using CTM:
            rho^{n+1} = rho^n + dt/L * (sum inflows - sum outflows)
        """
        self._snapshot()

        cell_order = self._cell_key_order()
        rho = self._densities_to_jax(cell_order)
        lengths = self._lengths_to_jax(cell_order)

        rho_np = np.asarray(rho, dtype=np.float64)
        lengths_np = np.asarray(lengths, dtype=np.float64)

        rho_map: Dict[Connection, float] = {
            key: float(rho_np[i]) for i, key in enumerate(cell_order)
        }

        edge_flow = self._compute_edge_flows(rho_map)
        external_inflow, external_outflow = self._compute_boundary_flows(rho_map)

        net_flow = np.zeros(len(cell_order), dtype=np.float64)

        index_of = {key: i for i, key in enumerate(cell_order)}

        # Internal edge flows
        for (u, v), q in edge_flow.items():
            ui = index_of[u]
            vi = index_of[v]
            net_flow[ui] -= q
            net_flow[vi] += q

        # Boundary flows
        for cell_key, q in external_inflow.items():
            idx = index_of[cell_key]
            net_flow[idx] += q

        for cell_key, q in external_outflow.items():
            idx = index_of[cell_key]
            net_flow[idx] -= q

        dt = self.time_resolution
        new_rho = rho_np + dt * net_flow / lengths_np
        new_rho = np.clip(new_rho, 0.0, self.jam_density)

        self._write_densities_back(cell_order, jnp.array(new_rho, dtype=jnp.float32))
        self.current_time += self.time_resolution

    def run(self, duration: float) -> None:
        """
        Roll out the simulation for a desired duration in seconds.
        """
        if duration < 0.0:
            raise ValueError("duration must be non-negative.")
        num_steps = int(np.round(duration / self.time_resolution))
        for _ in range(num_steps):
            self.step()

    # -------------------------
    # Optional utilities
    # -------------------------

    def initialize_from_ground_truth(
        self,
        gt_store: GroundTruthStore,
        time_value: Optional[float] = None,
        tolerance: float = 1e-6,
    ) -> None:
        """
        Set cell densities from a ground-truth snapshot.
        """
        t = self.current_time if time_value is None else float(time_value)
        gt_store.apply_density_snapshot_to_network(self.network, t, tolerance=tolerance)

    def current_state_dataframe(self) -> pd.DataFrame:
        """
        Return current cell states as a DataFrame.
        """
        rows = []
        for road_id, road in self.network.roads.items():
            for cell_id, cell in road.cells.items():
                rho = float(cell.density)
                q = min(self.free_flow_speed * rho, self.capacity)
                v = float(q / rho) if rho > 1e-12 else self.free_flow_speed
                rows.append(
                    {
                        "time": self.current_time,
                        "road_id": road_id,
                        "cell_id": cell_id,
                        "lane": cell.lane,
                        "start_s": cell.start_s,
                        "end_s": cell.end_s,
                        "length": cell.length,
                        "density": rho,
                        "flow": q,
                        "velocity": v,
                    }
                )
        return pd.DataFrame(rows)

    def rollout_dataframe(self) -> pd.DataFrame:
        """
        Flatten all stored rollout snapshots into a DataFrame.
        """
        rows = []
        for step in self.rollout_results:
            net = step.network
            for road_id, road in net.roads.items():
                for cell_id, cell in road.cells.items():
                    rho = float(cell.density)
                    q = min(self.free_flow_speed * rho, self.capacity)
                    v = float(q / rho) if rho > 1e-12 else self.free_flow_speed
                    rows.append(
                        {
                            "time": step.sim_time,
                            "road_id": road_id,
                            "cell_id": cell_id,
                            "lane": cell.lane,
                            "start_s": cell.start_s,
                            "end_s": cell.end_s,
                            "length": cell.length,
                            "density": rho,
                            "flow": q,
                            "velocity": v,
                        }
                    )
        return pd.DataFrame(rows)

    def save_rollout_parquet(self, path: str) -> None:
        self.rollout_dataframe().to_parquet(path, index=False)


# =========================
# Example usage
# =========================

if __name__ == "__main__":
    # Example:
    #
    # sim = Simulation.from_json(
    #     json_path="network.json",
    #     time_resolution=1.0,
    #     origin_time=1669819550.0,
    #     free_flow_speed=30.0,         # m/s
    #     congestion_wave_speed=5.0,    # m/s
    #     jam_density=0.16,             # veh/m
    # )
    #
    # gt = GroundTruthStore.from_parquet("ground_truth.parquet")
    # sim.initialize_from_ground_truth(gt, time_value=1669819550.0)
    # sim.run(duration=300.0)
    # sim.save_rollout_parquet("ctm_rollout.parquet")
    #
    # print(sim.current_state_dataframe().head())
    pass