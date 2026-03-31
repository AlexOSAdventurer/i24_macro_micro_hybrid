from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
import copy
import json
import math

import plotly.graph_objects as go
import numpy as np
import pandas as pd

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
    fd: Optional[FundamentalDiagram] = None

    @property
    def length(self) -> float:
        return float(self.end_s - self.start_s)

    @property
    def mass(self) -> float:
        return float(self.density * self.length)

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
    left_polyline: List[Tuple[float, float]]
    right_polyline: List[Tuple[float, float]]
    lane_data: Dict[int, Dict[str, float]] = field(default_factory=dict)
    cells: Dict[str, Cell] = field(default_factory=dict)

    def validate(self) -> None:
        if len(self.left_polyline) < 2 or len(self.right_polyline) < 2:
            raise ValueError(
                f"Road {self.road_id} must have valid left/right polylines."
            )
        if len(self.left_polyline) != len(self.right_polyline):
            raise ValueError(
                f"Road {self.road_id} left/right polylines must have same length."
            )

        for lane_id, lane_info in self.lane_data.items():
            if "lateral_position" not in lane_info or "width" not in lane_info:
                raise ValueError(
                    f"Lane {lane_id} in road {self.road_id} missing required fields."
                )
            if float(lane_info["width"]) <= 0.0:
                raise ValueError(
                    f"Lane {lane_id} in road {self.road_id} has non-positive width."
                )

        for cell in self.cells.values():
            if cell.road_id != self.road_id:
                raise ValueError(
                    f"Cell {cell.cell_id} road_id mismatch: {cell.road_id} != {self.road_id}"
                )
            if cell.lane not in self.lane_data:
                raise ValueError(
                    f"Cell {cell.cell_id} references lane {cell.lane} not in lane_data."
                )
            cell.validate()

    def cells_for_lane(self, lane: int) -> List[Cell]:
        out = [c for c in self.cells.values() if c.lane == lane]
        out.sort(key=lambda c: (c.start_s, c.end_s, c.cell_id))
        return out


@dataclass
class Network:
    network_id: str
    roads: Dict[str, Road] = field(default_factory=dict)

    def validate(self) -> None:
        for road_id, road in self.roads.items():
            if road.road_id != road_id:
                raise ValueError(f"Road dict key mismatch: {road_id} != {road.road_id}")
            road.validate()

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
                    "left_polyline": [[float(x), float(y)] for x, y in road.left_polyline],
                    "right_polyline": [[float(x), float(y)] for x, y in road.right_polyline],
                    "lane_data": {
                        str(lane_id): {
                            "lateral_position": float(lane_info["lateral_position"]),
                            "width": float(lane_info["width"]),
                        }
                        for lane_id, lane_info in road.lane_data.items()
                    },
                    "cells": {
                        cell_id: {
                            "road_id": cell.road_id,
                            "cell_id": cell.cell_id,
                            "lane": int(cell.lane),
                            "start_s": float(cell.start_s),
                            "end_s": float(cell.end_s),
                            "density": float(cell.density),
                            "inflow_connections": [[r, c] for (r, c) in cell.inflow_connections],
                            "outflow_connections": [[r, c] for (r, c) in cell.outflow_connections],
                        }
                        for cell_id, cell in road.cells.items()
                    },
                }
                for road_id, road in self.roads.items()
            },
        }

    @staticmethod
    def merge_networks(network1: Network, network2: Network, new_id: str):
        new_dict = {}
        for key in network1.roads:
            new_dict[key] = network1.roads[key]

        for key in network2.roads:
            if key in new_dict:
                raise Exception("Networks have overlapping keys!")
            new_dict[key] = network2.roads[key]

        return Network(new_id, new_dict)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Network":
        roads: Dict[str, Road] = {}

        for road_id, road_data in data["roads"].items():
            cells: Dict[str, Cell] = {}
            for cell_id, cell_data in road_data["cells"].items():
                cells[cell_id] = Cell(
                    road_id=str(cell_data["road_id"]),
                    cell_id=str(cell_data["cell_id"]),
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
                road_id=str(road_data["road_id"]),
                left_polyline=[(float(x), float(y)) for x, y in road_data["left_polyline"]],
                right_polyline=[(float(x), float(y)) for x, y in road_data["right_polyline"]],
                lane_data={
                    int(lane_id): {
                        "lateral_position": float(lane_info["lateral_position"]),
                        "width": float(lane_info["width"]),
                    }
                    for lane_id, lane_info in road_data["lane_data"].items()
                },
                cells=cells,
            )

        network = Network(network_id=str(data["network_id"]), roads=roads)
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

    def plot_network(self) -> None:
        fig = go.Figure()

        for road in self.roads.values():
            left_poly = np.array(road.left_polyline, dtype=float)
            right_poly = np.array(road.right_polyline, dtype=float)

            fig.add_trace(
                go.Scatter(
                    x=left_poly[:, 0],
                    y=left_poly[:, 1],
                    mode="lines",
                    line=dict(width=3, color="black"),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=right_poly[:, 0],
                    y=right_poly[:, 1],
                    mode="lines",
                    line=dict(width=3, color="black"),
                    showlegend=False,
                )
            )

            seg_lengths = np.linalg.norm(np.diff(left_poly, axis=0), axis=1)
            cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])

            def interpolate_lane_edges(s: float, lane: int) -> Tuple[np.ndarray, np.ndarray]:
                lane_info = road.lane_data[lane]
                lat = float(lane_info["lateral_position"])
                width = float(lane_info["width"])

                idx = int(np.clip(np.searchsorted(cumulative, s) - 1, 0, len(seg_lengths) - 1))
                ds = s - cumulative[idx]
                t = ds / max(seg_lengths[idx], 1e-8)

                l = left_poly[idx] + t * (left_poly[idx + 1] - left_poly[idx])
                r = right_poly[idx] + t * (right_poly[idx + 1] - right_poly[idx])
                lr = r - l
                w = np.linalg.norm(lr)
                d = lr / max(w, 1e-8)

                lane_left = l + (-lat) * d
                lane_right = l + (-(lat - width)) * d
                return lane_left, lane_right

            for lane_id in sorted(road.lane_data):
                pts_left = []
                pts_right = []
                for i in range(len(left_poly)):
                    l = left_poly[i]
                    r = right_poly[i]
                    d = r - l
                    w = np.linalg.norm(d)
                    d = d / max(w, 1e-8)
                    lat = float(road.lane_data[lane_id]["lateral_position"])
                    width = float(road.lane_data[lane_id]["width"])
                    pts_left.append(l + (-lat) * d)
                    pts_right.append(l + (-(lat - width)) * d)

                pts_left = np.array(pts_left)
                pts_right = np.array(pts_right)
                fig.add_trace(go.Scatter(x=pts_left[:, 0], y=pts_left[:, 1], mode="lines",
                                         line=dict(width=1, dash="dot"), showlegend=False))
                fig.add_trace(go.Scatter(x=pts_right[:, 0], y=pts_right[:, 1], mode="lines",
                                         line=dict(width=1, dash="dot"), showlegend=False))

            for cell in road.cells.values():
                p1, p2 = interpolate_lane_edges(cell.start_s, cell.lane)
                p3, p4 = interpolate_lane_edges(cell.end_s, cell.lane)
                x = [p1[0], p2[0], p4[0], p3[0], p1[0]]
                y = [p1[1], p2[1], p4[1], p3[1], p1[1]]
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        fill="toself",
                        mode="lines",
                        line=dict(color="blue"),
                        fillcolor="rgba(173,216,230,0.3)",
                        showlegend=False,
                    )
                )

        fig.update_layout(
            title="Road Network Geometry",
            xaxis=dict(scaleanchor="y"),
            yaxis=dict(),
            template="plotly_white",
        )
        fig.show()

class NetworkGenerator(ABC):
    def __init__(self):
        self.network = None

    @abstractmethod
    def create_network(self):
        raise NotImplementedError
    
    def save_network(self, path):
        self.network.validate()
        network_dict = self.network.to_dict()
        with open(path, "w+") as f:
            json.dump(network_dict, f, indent=4)    

class I24WestBoundNetwork(NetworkGenerator):
    def __init__(self):
        super().__init__()
        self.network = None

    def create_network(self):
        network_id = "i24_westbound"
        road_id = "2"
        road_length = 1600.0
        longitudinal_step = 50.0
        lane_width = 3.6576
        lane_count = 4
        offset_from_medium = lane_width
        road_width = lane_width * lane_count
        starting_x = 200.0 + lane_width
        starting_y = 0.0
        ending_x = 0.0 + lane_width

        road_left_polyline = [
            (starting_x, starting_y),
            (ending_x, ((road_length ** 2) - (starting_x ** 2)) ** 0.5)
        ]

        direction_vector = ((road_left_polyline[1][0] - road_left_polyline[0][0]) / road_length, (road_left_polyline[1][1] - road_left_polyline[0][1]) / road_length)
        norm_vector = (direction_vector[1], -direction_vector[0]) # Goes to the right side of the road

        road_right_polyline = [
            (road_left_polyline[0][0] + (road_width * norm_vector[0]), road_left_polyline[0][1] + (road_width * norm_vector[1])),
            (road_left_polyline[1][0] + (road_width * norm_vector[0]), road_left_polyline[1][1] + (road_width * norm_vector[1]))
        ]

        road_lane_data = {
            -1: {
                "width": lane_width,
                "lateral_position": 0.0
            },
            -2: {
                "width": lane_width,
                "lateral_position": -lane_width
            },
            -3: {
                "width": lane_width,
                "lateral_position": -(lane_width*2.0)
            },
            -4: {
                "width": lane_width,
                "lateral_position": -(lane_width*3.0)
            }
        }

        longitudinal_steps = np.arange(0.0, road_length, longitudinal_step).tolist()
        cells = {}
        for i, step in enumerate(longitudinal_steps):
            for lane in road_lane_data:
                cell_id = f"road_{road_id}_cell_{lane}_step_{i}"
                start_s = step
                end_s = step + longitudinal_step
                density = 0
                inflow_connections = []
                outflow_connections = []
                if (i > 0):
                    inflow_connections.append((road_id, f"road_{road_id}_cell_{lane}_step_{i - 1}"))
                if (i < (len(longitudinal_steps) - 1)):
                    outflow_connections.append((road_id, f"road_{road_id}_cell_{lane}_step_{i + 1}"))
                cell = Cell(road_id=road_id, cell_id=cell_id, lane=lane, start_s=start_s, end_s=end_s, density=density, inflow_connections=inflow_connections, outflow_connections=outflow_connections)
                cells[cell_id] = cell

        road = Road(road_id=road_id, left_polyline=road_left_polyline, right_polyline=road_right_polyline, lane_data=road_lane_data, cells=cells)
        self.network = Network(network_id=network_id, roads={road_id: road})

class I24EastBoundNetwork(NetworkGenerator):
    def __init__(self):
        super().__init__()
        self.network = None

    def create_network(self):
        network_id = "i24_eastbound"
        road_id = "1"
        road_length = 1600.0
        longitudinal_step = 50.0
        lane_width = 3.6576
        lane_count = 4
        offset_from_medium = lane_width
        road_width = lane_width * lane_count
        starting_x = 0.0 - offset_from_medium
        ending_y = 0.0
        ending_x = 200.0 - offset_from_medium

        road_left_polyline = [
            (starting_x, ((road_length ** 2) - (ending_x ** 2)) ** 0.5),
            (ending_x, ending_y)
        ]

        direction_vector = ((road_left_polyline[1][0] - road_left_polyline[0][0]) / road_length, (road_left_polyline[1][1] - road_left_polyline[0][1]) / road_length)
        norm_vector = (direction_vector[1], -direction_vector[0]) # Goes to the right side of the road

        road_right_polyline = [
            (road_left_polyline[0][0] + (road_width * norm_vector[0]), road_left_polyline[0][1] + (road_width * norm_vector[1])),
            (road_left_polyline[1][0] + (road_width * norm_vector[0]), road_left_polyline[1][1] + (road_width * norm_vector[1]))
        ]

        road_lane_data = {
            -1: {
                "width": lane_width,
                "lateral_position": 0.0
            },
            -2: {
                "width": lane_width,
                "lateral_position": -lane_width
            },
            -3: {
                "width": lane_width,
                "lateral_position": -(lane_width*2.0)
            },
            -4: {
                "width": lane_width,
                "lateral_position": -(lane_width*3.0)
            }
        }

        longitudinal_steps = np.arange(0.0, road_length, longitudinal_step).tolist()
        cells = {}
        for i, step in enumerate(longitudinal_steps):
            for lane in road_lane_data:
                cell_id = f"road_{road_id}_cell_{lane}_step_{i}"
                start_s = step
                end_s = step + longitudinal_step
                density = 0
                inflow_connections = []
                outflow_connections = []
                if (i > 0):
                    inflow_connections.append((road_id, f"road_{road_id}_cell_{lane}_step_{i - 1}"))
                if (i < (len(longitudinal_steps) - 1)):
                    outflow_connections.append((road_id, f"road_{road_id}_cell_{lane}_step_{i + 1}"))
                cell = Cell(road_id=road_id, cell_id=cell_id, lane=lane, start_s=start_s, end_s=end_s, density=density, inflow_connections=inflow_connections, outflow_connections=outflow_connections)
                cells[cell_id] = cell

        road = Road(road_id=road_id, left_polyline=road_left_polyline, right_polyline=road_right_polyline, lane_data=road_lane_data, cells=cells)
        self.network = Network(network_id=network_id, roads={road_id: road})

class I24WestAndEastNetwork(NetworkGenerator):
    def __init__(self):
        super().__init__()
        self.network = None

    def create_network(self):
        network_id = "i24_west_and_east_network"
        westbound_network = I24WestBoundNetwork()
        eastbound_network = I24EastBoundNetwork()
        westbound_network.create_network()
        eastbound_network.create_network()
        self.network = Network.merge_networks(westbound_network.network, eastbound_network.network, network_id)

# =========================
# Ground-truth data
# =========================

REQUIRED_GT_COLUMNS = {
    "macro": {
        "time",
        "time_length",
        "road_id",
        "cell_id",
        "density",
        "velocity"
    },
    "micro": {
        "id",
        "class",
        "time",
        "road_id",
        "s",
        "t",
        "length",
        "width",
        "height"
    }
}

class GroundTruthStore:
    def __init__(self, micro_df: pd.DataFrame, macro_df: pd.DataFrame):
        missing_micro = REQUIRED_GT_COLUMNS["micro"] - set(micro_df.columns)
        if missing_micro:
            raise ValueError(f"Ground-truth micro parquet missing columns: {sorted(missing_micro)}")

        self.micro_df = micro_df.copy()
        self.micro_df["id"] = self.micro_df["id"].astype(int)
        self.micro_df["class"] = self.micro_df["class"].astype(int)
        self.micro_df["time"] = self.micro_df["time"].astype(float)
        self.micro_df["road_id"] = self.micro_df["road_id"].astype(str)
        self.micro_df["s"] = self.micro_df["s"].astype(float)
        self.micro_df["t"] = self.micro_df["t"].astype(float)
        self.micro_df["length"] = self.micro_df["length"].astype(float)
        self.micro_df["width"] = self.micro_df["width"].astype(float)
        self.micro_df["height"] = self.micro_df["height"].astype(float)

        missing_macro = REQUIRED_GT_COLUMNS["macro"] - set(macro_df.columns)
        if missing_macro:
            raise ValueError(f"Ground-truth macro parquet missing columns: {sorted(missing_macro)}")

        self.macro_df = macro_df.copy()
        self.macro_df["time"] = self.macro_df["time"].astype(float)
        self.macro_df["time_length"] = self.macro_df["time_length"].astype(float)
        self.macro_df["road_id"] = self.macro_df["road_id"].astype(str)
        self.macro_df["cell_id"] = self.macro_df["cell_id"].astype(str)
        self.macro_df["density"] = self.macro_df["density"].astype(float)
        self.macro_df["velocity"] = self.macro_df["velocity"].astype(float)

    @staticmethod
    def from_parquet(micro_parquet_path: str, macro_parquet_path: str) -> "GroundTruthStore":
        return GroundTruthStore(pd.read_parquet(micro_parquet_path), pd.read_parquet(macro_parquet_path))

    def macro_snapshot_at_time(self, time_value: float, tolerance: float = 1e-2) -> pd.DataFrame:
        unique_times = self.macro_df["time"].unique()
        idx = int(np.argmin(np.abs(unique_times - time_value)))
        chosen_time = float(unique_times[idx])

        if abs(chosen_time - time_value) > tolerance:
            raise KeyError(
                f"No ground-truth snapshot near time={time_value}. Closest available is {chosen_time}."
            )

        out = self.macro_df[self.macro_df["time"] == chosen_time].copy()
        out.reset_index(drop=True, inplace=True)
        return out
    
    def micro_snapshot_at_time(self, time_value: float, tolerance: float = 1e-2) -> pd.DataFrame:
        unique_times = self.micro_df["time"].unique()
        idx = int(np.argmin(np.abs(unique_times - time_value)))
        chosen_time = float(unique_times[idx])

        if abs(chosen_time - time_value) > tolerance:
            raise KeyError(
                f"No ground-truth snapshot near time={time_value}. Closest available is {chosen_time}."
            )

        out = self.micro_df[self.micro_df["time"] == chosen_time].copy()
        out.reset_index(drop=True, inplace=True)
        return out

    def apply_density_snapshot_to_network(
        self, network: Network, time_value: float, tolerance: float = 1e-2
    ) -> None:
        snapshot = self.macro_snapshot_at_time(time_value, tolerance=tolerance)
        for _, row in snapshot.iterrows():
            network.get_cell(str(row["road_id"]), str(row["cell_id"])).density = float(row["density"])

    def apply_density_snapshot_to_network_boundaries(
        self, network: Network, time_value: float, tolerance: float = 1e-2
    ) -> None:
        snapshot = self.macro_snapshot_at_time(time_value, tolerance=tolerance)
        for _, row in snapshot.iterrows():
            network_cell = network.get_cell(str(row["road_id"]), str(row["cell_id"]))
            if (len(network_cell.inflow_connections) == 0) or (len(network_cell.outflow_connections) == 0):
                network_cell.density = float(row["density"])


# =========================
# Mask overlays / active mesh
# =========================

@dataclass
class MaskedSegmentRef:
    road_id: str
    lane: int
    start_s: float
    end_s: float

    def validate(self) -> None:
        if self.end_s <= self.start_s:
            raise ValueError(
                f"Invalid masked segment on {self.road_id}, lane {self.lane}: "
                f"[{self.start_s}, {self.end_s}]"
            )


class ArbitraryMaskingCell(ABC):
    """
    Dynamic overlay that replaces native CTM logic over one or more
    connected road/lane intervals.
    """

    def __init__(self, mask_id: str, network: Network, segments: List[MaskedSegmentRef]):
        self.mask_id = str(mask_id)
        self.network = network
        self.segments = segments
        self.validate()

    def validate(self) -> None:
        if len(self.segments) == 0:
            raise ValueError(f"Mask {self.mask_id} must have at least one segment.")
        for seg in self.segments:
            seg.validate()
            if seg.road_id not in self.network.roads:
                raise KeyError(f"Mask {self.mask_id}: road '{seg.road_id}' not found.")
            if seg.lane not in self.network.roads[seg.road_id].lane_data:
                raise KeyError(
                    f"Mask {self.mask_id}: lane {seg.lane} not in road {seg.road_id}."
                )

    @abstractmethod
    def demand(self, sim_time: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def supply(self, sim_time: float) -> float:
        raise NotImplementedError

    def update(self, sim_time: float, dt: float) -> None:
        """
        Optional hook for moving masks.
        Override in subclasses.
        """
        return


@dataclass
class ActiveCell:
    active_cell_id: str
    road_id: str
    lane: int
    start_s: float
    end_s: float
    kind: str  # "normal" or "mask"
    density: float
    base_segments: List[Tuple[Connection, float, float]] = field(default_factory=list)
    mask_id: Optional[str] = None
    fd: Optional[FundamentalDiagram] = None
    inflow_neighbors: List[str] = field(default_factory=list)
    outflow_neighbors: List[str] = field(default_factory=list)

    @property
    def length(self) -> float:
        return float(self.end_s - self.start_s)

    @property
    def mass(self) -> float:
        return float(self.density * self.length)


@dataclass
class ActiveNetwork:
    active_cells: Dict[str, ActiveCell] = field(default_factory=dict)

    def ordered_ids(self) -> List[str]:
        keys = list(self.active_cells.keys())
        keys.sort(key=lambda k: (
            self.active_cells[k].road_id,
            self.active_cells[k].lane,
            self.active_cells[k].start_s,
            self.active_cells[k].end_s,
            self.active_cells[k].active_cell_id,
        ))
        return keys


class ActiveMeshBuilder:
    """
    Builds a timestep-specific active CTM mesh from:
    - static base network
    - dynamic mask overlays
    """

    def __init__(self, network: Network, masks: Dict[str, ArbitraryMaskingCell], min_cell_length: float):
        self.network = network
        self.masks = masks
        self.min_cell_length = float(min_cell_length)

    def build(self) -> ActiveNetwork:
        active = ActiveNetwork()

        for road_id, road in self.network.roads.items():
            for lane in sorted(road.lane_data):
                lane_cells = road.cells_for_lane(lane)
                if not lane_cells:
                    continue

                cut_points: Set[float] = set()
                for c in lane_cells:
                    cut_points.add(float(c.start_s))
                    cut_points.add(float(c.end_s))

                lane_masks: List[Tuple[str, float, float]] = []
                for mask_id, mask in self.masks.items():
                    for seg in mask.segments:
                        if seg.road_id == road_id and seg.lane == lane:
                            cut_points.add(float(seg.start_s))
                            cut_points.add(float(seg.end_s))
                            lane_masks.append((mask_id, float(seg.start_s), float(seg.end_s)))

                points = sorted(cut_points)
                raw_intervals: List[Dict[str, Any]] = []

                for i in range(len(points) - 1):
                    a = points[i]
                    b = points[i + 1]
                    if b <= a:
                        continue

                    owners = []
                    for mask_id, ms, me in lane_masks:
                        if a >= ms - 1e-9 and b <= me + 1e-9:
                            owners.append(mask_id)

                    if len(owners) > 1:
                        raise ValueError(
                            f"Overlapping masks detected on {road_id}, lane {lane}, interval [{a}, {b}]"
                        )

                    mask_id = owners[0] if owners else None

                    overlaps: List[Tuple[Connection, float, float]] = []
                    for c in lane_cells:
                        s0 = max(a, c.start_s)
                        s1 = min(b, c.end_s)
                        if s1 > s0:
                            overlaps.append(((c.road_id, c.cell_id), float(s0), float(s1)))

                    if len(overlaps) == 0:
                        continue

                    raw_intervals.append(
                        {
                            "road_id": road_id,
                            "lane": lane,
                            "start_s": a,
                            "end_s": b,
                            "kind": "mask" if mask_id is not None else "normal",
                            "mask_id": mask_id,
                            "base_segments": overlaps,
                        }
                    )

                merged = self._merge_small_intervals(raw_intervals)

                for idx, item in enumerate(merged):
                    active_cell_id = f"{road_id}|lane{lane}|{idx}"
                    cell_fd = None
                    if item["kind"] == "normal" and item["base_segments"]:
                        cell_fd = self.network.get_cell(*item["base_segments"][0][0]).fd
                    active.active_cells[active_cell_id] = ActiveCell(
                        active_cell_id=active_cell_id,
                        road_id=road_id,
                        lane=lane,
                        start_s=float(item["start_s"]),
                        end_s=float(item["end_s"]),
                        kind=str(item["kind"]),
                        density=0.0,
                        base_segments=list(item["base_segments"]),
                        mask_id=item["mask_id"],
                        fd=cell_fd,
                    )

        self._build_neighbors(active)
        return active

    def _merge_small_intervals(self, intervals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not intervals:
            return []

        intervals = sorted(intervals, key=lambda x: (x["start_s"], x["end_s"]))
        changed = True

        while changed:
            changed = False
            out: List[Dict[str, Any]] = []
            i = 0
            while i < len(intervals):
                cur = copy.deepcopy(intervals[i])
                cur_len = float(cur["end_s"] - cur["start_s"])

                if cur_len >= self.min_cell_length or len(intervals) == 1:
                    out.append(cur)
                    i += 1
                    continue

                left_ok = len(out) > 0
                right_ok = (i + 1) < len(intervals)

                chosen = None
                if left_ok:
                    left = out[-1]
                    if left["kind"] == cur["kind"] and left.get("mask_id") == cur.get("mask_id"):
                        chosen = "left"
                if chosen is None and right_ok:
                    right = intervals[i + 1]
                    if right["kind"] == cur["kind"] and right.get("mask_id") == cur.get("mask_id"):
                        chosen = "right"
                if chosen is None and left_ok:
                    chosen = "left"
                if chosen is None and right_ok:
                    chosen = "right"

                if chosen == "left":
                    out[-1]["end_s"] = cur["end_s"]
                    out[-1]["base_segments"].extend(cur["base_segments"])
                    changed = True
                    i += 1
                elif chosen == "right":
                    right = copy.deepcopy(intervals[i + 1])
                    right["start_s"] = cur["start_s"]
                    right["base_segments"] = list(cur["base_segments"]) + list(right["base_segments"])
                    out.append(right)
                    changed = True
                    i += 2
                else:
                    out.append(cur)
                    i += 1

            intervals = out

        return intervals

    def _build_neighbors(self, active: ActiveNetwork) -> None:
        by_lane: Dict[Tuple[str, int], List[ActiveCell]] = {}
        for ac in active.active_cells.values():
            by_lane.setdefault((ac.road_id, ac.lane), []).append(ac)

        for _, lane_cells in by_lane.items():
            lane_cells.sort(key=lambda c: (c.start_s, c.end_s))
            for i, cell in enumerate(lane_cells):
                if i > 0:
                    cell.inflow_neighbors.append(lane_cells[i - 1].active_cell_id)
                if i + 1 < len(lane_cells):
                    cell.outflow_neighbors.append(lane_cells[i + 1].active_cell_id)


class ConservativeRemapper:
    @staticmethod
    def base_to_active(network: Network, active: ActiveNetwork) -> None:
        for ac in active.active_cells.values():
            total_mass = 0.0
            for (base_key, s0, s1) in ac.base_segments:
                base_cell = network.get_cell(*base_key)
                overlap_len = s1 - s0
                total_mass += float(base_cell.density * overlap_len)

            ac.density = 0.0 if ac.length <= 1e-12 else total_mass / ac.length

    @staticmethod
    def active_to_base(network: Network, active: ActiveNetwork) -> None:
        base_mass_updates: Dict[Connection, float] = {}
        base_lengths: Dict[Connection, float] = {}

        for road in network.roads.values():
            for cell in road.cells.values():
                base_mass_updates[(cell.road_id, cell.cell_id)] = 0.0
                base_lengths[(cell.road_id, cell.cell_id)] = float(cell.length)

        for ac in active.active_cells.values():
            for (base_key, s0, s1) in ac.base_segments:
                overlap_len = s1 - s0
                base_mass_updates[base_key] += float(ac.density * overlap_len)

        for base_key, mass in base_mass_updates.items():
            cell = network.get_cell(*base_key)
            L = base_lengths[base_key]
            cell.density = 0.0 if L <= 1e-12 else mass / L


# =========================
# Fundamental diagrams
# =========================

class FundamentalDiagram(ABC):
    @abstractmethod
    def demand(self, rho: float) -> float:
        """Sending flow from an upstream cell."""
        raise NotImplementedError

    @abstractmethod
    def supply(self, rho: float) -> float:
        """Receiving flow into a downstream cell."""
        raise NotImplementedError

    @abstractmethod
    def velocity_from_density(self, rho: float) -> float:
        raise NotImplementedError


@dataclass
class TriangularFD(FundamentalDiagram):
    """
    Piecewise-linear (triangular) fundamental diagram.
      Free-flow:  q = v_f * rho           (rho <= rho_c)
      Congested:  q = w * (rho_j - rho)   (rho >  rho_c)
    """
    v_f: float    # free-flow speed (m/s)
    w: float      # backward wave speed magnitude (m/s)
    rho_j: float  # jam density (veh/m)

    @property
    def rho_c(self) -> float:
        return (self.w / (self.v_f + self.w)) * self.rho_j

    @property
    def capacity(self) -> float:
        return self.v_f * self.rho_c

    def demand(self, rho: float) -> float:
        return min(self.v_f * rho, self.capacity)

    def supply(self, rho: float) -> float:
        return min(self.capacity, self.w * max(self.rho_j - rho, 0.0))

    def velocity_from_density(self, rho: float) -> float:
        q = self.demand(rho)
        return q / rho if rho > 1e-12 else self.v_f


@dataclass
class GreenshieldsFD(FundamentalDiagram):
    """
    Greenshields (quadratic) fundamental diagram.
      q(rho) = v_f * rho * (1 - rho / rho_j)

    Demand and supply are computed by evaluating the flow function at
    the appropriate side of the critical density (rho_j / 2).
    """
    v_f: float    # free-flow speed (m/s)
    rho_j: float  # jam density (veh/m)

    @property
    def rho_c(self) -> float:
        return self.rho_j / 2.0

    @property
    def capacity(self) -> float:
        return self._flow(self.rho_c)

    def _flow(self, rho: float) -> float:
        return self.v_f * rho * (1.0 - rho / self.rho_j)

    def demand(self, rho: float) -> float:
        rho = max(0.0, min(rho, self.rho_j))
        rho = min(rho, self.rho_c)
        return self._flow(rho)

    def supply(self, rho: float) -> float:
        rho = max(0.0, min(rho, self.rho_j))
        rho = max(rho, self.rho_c)
        return self._flow(rho)

    def velocity_from_density(self, rho: float) -> float:
        q = self._flow(max(0.0, min(rho, self.rho_j)))
        return q / rho if rho > 1e-12 else self.v_f


# =========================
# Lane-change models
# =========================

class LaneChangeModel(ABC):
    @abstractmethod
    def lateral_delta_density(self, active: ActiveNetwork, dt: float) -> Dict[str, float]:
        """Return per active-cell-id density deltas from lateral exchange."""
        raise NotImplementedError


class NoLaneChange(LaneChangeModel):
    def lateral_delta_density(self, active: ActiveNetwork, dt: float) -> Dict[str, float]:
        return {}


@dataclass
class SpeedIncentiveLaneChange(LaneChangeModel):
    """
    Speed-differential lane-change model ported from the notebook's CTMSolver.

    For each pair of adjacent lanes, vehicles migrate toward the faster lane
    proportional to the speed gap and gap-acceptance probability.  Mass is
    conserved exactly: vehicles transferred = s_lat * overlap_length * dt,
    then divided by each cell's own length to give the density delta.
    """
    lambda_lc: float

    def lateral_delta_density(self, active: ActiveNetwork, dt: float) -> Dict[str, float]:
        # Group non-mask cells by (road_id, lane), sorted by start_s
        by_road_lane: Dict[Tuple[str, int], List[ActiveCell]] = {}
        for ac in active.active_cells.values():
            if ac.kind == "mask":
                continue
            by_road_lane.setdefault((ac.road_id, ac.lane), []).append(ac)
        for key in by_road_lane:
            by_road_lane[key].sort(key=lambda c: c.start_s)

        # Sorted lane indices per road
        lanes_per_road: Dict[str, List[int]] = {}
        for road_id, lane in by_road_lane:
            lanes_per_road.setdefault(road_id, []).append(lane)
        for road_id in lanes_per_road:
            lanes_per_road[road_id].sort()

        delta: Dict[str, float] = {}

        for road_id, sorted_lanes in lanes_per_road.items():
            for idx in range(len(sorted_lanes) - 1):
                lane_a = sorted_lanes[idx]
                lane_b = sorted_lanes[idx + 1]
                cells_a = by_road_lane[(road_id, lane_a)]
                cells_b = by_road_lane[(road_id, lane_b)]

                for ca in cells_a:
                    for cb in cells_b:
                        overlap = min(ca.end_s, cb.end_s) - max(ca.start_s, cb.start_s)
                        if overlap <= 0.0:
                            continue

                        rho_a = ca.density
                        rho_b = cb.density
                        v_a = ca.fd.velocity_from_density(rho_a)
                        v_b = cb.fd.velocity_from_density(rho_b)
                        p_gap_a = max(0.0, 1.0 - rho_a / ca.fd.rho_j)
                        p_gap_b = max(0.0, 1.0 - rho_b / cb.fd.rho_j)

                        # Net density flux rate from lane_a -> lane_b (veh/m/s)
                        dv_ab = max(v_b - v_a, 0.0) / max(ca.fd.v_f, 1e-9)
                        dv_ba = max(v_a - v_b, 0.0) / max(cb.fd.v_f, 1e-9)
                        s_lat = (self.lambda_lc * dv_ab * rho_a * p_gap_b
                               - self.lambda_lc * dv_ba * rho_b * p_gap_a)

                        # Vehicles transferred over the shared boundary in this timestep
                        vehicles = s_lat * overlap * dt
                        delta[ca.active_cell_id] = delta.get(ca.active_cell_id, 0.0) - vehicles / ca.length
                        delta[cb.active_cell_id] = delta.get(cb.active_cell_id, 0.0) + vehicles / cb.length

        return delta


# =========================
# CTM / simulation
# =========================

@dataclass
class RolloutStep:
    sim_time: float
    network: Network


class Simulation:
    """
    Base state lives on the static Network.
    Each timestep:
    - update masks
    - build active mesh
    - remap base -> active
    - run CTM on active mesh
    - remap active -> base
    """

    def __init__(
        self,
        network: Network,
        time_resolution: float,
        origin_time: float,
        inflow_boundary_map: Optional[Dict[Connection, float]] = None,
        outflow_boundary_map: Optional[Dict[Connection, float]] = None,
        min_cell_length: Optional[float] = None,
        lane_change_model: Optional[LaneChangeModel] = None,
    ):
        self.network = network
        self.time_resolution = float(time_resolution)
        self.origin_time = float(origin_time)
        self.current_time = float(origin_time)
        self.rollout_results: List[RolloutStep] = []

        self.inflow_boundary_map = inflow_boundary_map or {}
        self.outflow_boundary_map = outflow_boundary_map or {}

        self.min_cell_length = float(min_cell_length) if min_cell_length is not None else 1.0

        self.lane_change_model: LaneChangeModel = lane_change_model if lane_change_model is not None else NoLaneChange()

        self.masking_cells: Dict[str, ArbitraryMaskingCell] = {}

        self.network.validate()

        self.gt_store = None

    @staticmethod
    def from_json(
        json_path: str,
        time_resolution: float,
        origin_time: float,
        inflow_boundary_map: Optional[Dict[Connection, float]] = None,
        outflow_boundary_map: Optional[Dict[Connection, float]] = None,
        min_cell_length: Optional[float] = None,
    ) -> "Simulation":
        network = Network.from_json(json_path)
        return Simulation(
            network=network,
            time_resolution=time_resolution,
            origin_time=origin_time,
            inflow_boundary_map=inflow_boundary_map,
            outflow_boundary_map=outflow_boundary_map,
            min_cell_length=min_cell_length,
        )

    def add_masking_cell(self, mask: ArbitraryMaskingCell) -> None:
        if mask.mask_id in self.masking_cells:
            raise KeyError(f"Mask '{mask.mask_id}' already exists.")
        self.masking_cells[mask.mask_id] = mask

    def remove_masking_cell(self, mask_id: str) -> None:
        if mask_id in self.masking_cells:
            del self.masking_cells[mask_id]

    def _snapshot(self) -> None:
        self.rollout_results.append(
            RolloutStep(sim_time=self.current_time, network=self.network.clone())
        )
    def initialize_from_ground_truth(
        self,
        gt_store: GroundTruthStore,
        time_value: Optional[float] = None,
        tolerance: float = 1e-6,
    ) -> None:
        t = self.current_time if time_value is None else float(time_value)
        gt_store.apply_density_snapshot_to_network(self.network, t, tolerance=tolerance)
        self.gt_store = gt_store

    def _update_masks(self) -> None:
        for mask in self.masking_cells.values():
            mask.update(self.current_time, self.time_resolution)
            mask.validate()

    def _build_active_network(self) -> ActiveNetwork:
        builder = ActiveMeshBuilder(
            network=self.network,
            masks=self.masking_cells,
            min_cell_length=self.min_cell_length,
        )
        active = builder.build()
        ConservativeRemapper.base_to_active(self.network, active)
        return active

    def _compute_active_demand_supply(
        self, active: ActiveNetwork
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        demand_map: Dict[str, float] = {}
        supply_map: Dict[str, float] = {}

        for aid, ac in active.active_cells.items():
            if ac.kind == "normal":
                rho = float(ac.density)
                if ac.fd is not None:
                    demand_map[aid] = ac.fd.demand(rho)
                    supply_map[aid] = ac.fd.supply(rho)
                else:
                    raise Exception ("No FD available!")
            elif ac.kind == "mask":
                mask = self.masking_cells[ac.mask_id]
                demand_map[aid] = float(mask.demand(self.current_time))
                supply_map[aid] = float(mask.supply(self.current_time))
            else:
                raise ValueError(f"Unknown active cell kind: {ac.kind}")

        return demand_map, supply_map

    def _compute_active_edge_flows(
        self, active: ActiveNetwork
    ) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float], Dict[str, float]]:
        demand_map, supply_map = self._compute_active_demand_supply(active)

        edge_flow: Dict[Tuple[str, str], float] = {}

        active_ids = active.ordered_ids()
        for u in active_ids:
            u_cell = active.active_cells[u]
            succ = list(u_cell.outflow_neighbors)
            if len(succ) == 0:
                continue

            per_out_demand = demand_map[u] / float(len(succ))
            for v in succ:
                v_cell = active.active_cells[v]
                preds = list(v_cell.inflow_neighbors)
                per_in_supply = supply_map[v] if len(preds) == 0 else supply_map[v] / float(len(preds))
                edge_flow[(u, v)] = min(per_out_demand, per_in_supply)

        for v in active_ids:
            incoming_edges = [e for e in edge_flow if e[1] == v]
            if not incoming_edges:
                continue
            total_in = sum(edge_flow[e] for e in incoming_edges)
            cap_in = supply_map[v]
            if total_in > cap_in and total_in > 1e-12:
                scale = cap_in / total_in
                for e in incoming_edges:
                    edge_flow[e] *= scale

        for u in active_ids:
            outgoing_edges = [e for e in edge_flow if e[0] == u]
            if not outgoing_edges:
                continue
            total_out = sum(edge_flow[e] for e in outgoing_edges)
            cap_out = demand_map[u]
            if total_out > cap_out and total_out > 1e-12:
                scale = cap_out / total_out
                for e in outgoing_edges:
                    edge_flow[e] *= scale

        external_inflow: Dict[str, float] = {}
        external_outflow: Dict[str, float] = {}
        for aid in active_ids:
            ac = active.active_cells[aid]
            cell_capacity = ac.fd.capacity if ac.fd is not None else None
            if len(ac.inflow_neighbors) == 0:
                base_key = ac.base_segments[0][0]
                external_inflow[aid] = min(
                    float(self.inflow_boundary_map.get(base_key, cell_capacity)),
                    supply_map[aid],
                ) if cell_capacity is not None else 0.0
            if len(ac.outflow_neighbors) == 0:
                base_key = ac.base_segments[-1][0]
                external_outflow[aid] = min(
                    demand_map[aid],
                    float(self.outflow_boundary_map.get(base_key, cell_capacity)),
                ) if cell_capacity is not None else demand_map[aid]

        return edge_flow, external_inflow, external_outflow

    def _step_active_network(self, active: ActiveNetwork) -> None:
        active_ids = active.ordered_ids()
        index_of = {aid: i for i, aid in enumerate(active_ids)}

        densities = np.array([active.active_cells[aid].density for aid in active_ids], dtype=np.float64)
        lengths = np.array([active.active_cells[aid].length for aid in active_ids], dtype=np.float64)

        dt = self.time_resolution

        edge_flow, external_inflow, external_outflow = self._compute_active_edge_flows(active)
        lateral_deltas = self.lane_change_model.lateral_delta_density(active, dt)

        net_flow = np.zeros(len(active_ids), dtype=np.float64)

        for (u, v), q in edge_flow.items():
            net_flow[index_of[u]] -= q
            net_flow[index_of[v]] += q

        for aid, q in external_inflow.items():
            net_flow[index_of[aid]] += q

        for aid, q in external_outflow.items():
            net_flow[index_of[aid]] -= q

        new_densities = densities + dt * net_flow / np.maximum(lengths, 1e-12)

        for i, aid in enumerate(active_ids):
            ac = active.active_cells[aid]
            if ac.kind != "mask":
                if ac.fd is None:
                    raise ValueError(f"Active cell {aid} has no FundamentalDiagram assigned.")
                new_rho = float(new_densities[i]) + lateral_deltas.get(aid, 0.0)
                ac.density = float(np.clip(new_rho, 0.0, ac.fd.rho_j))

    def step(self) -> None:
        self._snapshot()
        self._update_masks()
        active = self._build_active_network()
        self._step_active_network(active)
        ConservativeRemapper.active_to_base(self.network, active)
        self.current_time += self.time_resolution
        self.gt_store.apply_density_snapshot_to_network_boundaries(self.network, self.current_time)
        print(self.current_time)

    def run(self, duration: float) -> None:
        if duration < 0.0:
            raise ValueError("duration must be non-negative.")
        num_steps = int(np.round(duration / self.time_resolution))
        for _ in range(num_steps):
            self.step()

    def current_state_dataframe(self) -> pd.DataFrame:
        rows = []
        for road_id, road in self.network.roads.items():
            for cell_id, cell in road.cells.items():
                rho = float(cell.density)
                if cell.fd is None:
                    raise ValueError(f"Cell {road_id}/{cell_id} has no FundamentalDiagram assigned.")
                q = cell.fd.demand(rho)
                v = cell.fd.velocity_from_density(rho)
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
        rows = []
        for step in self.rollout_results:
            net = step.network
            for road_id, road in net.roads.items():
                for cell_id, cell in road.cells.items():
                    rho = float(cell.density)
                    if cell.fd is None:
                        raise ValueError(f"Cell {road_id}/{cell_id} has no FundamentalDiagram assigned.")
                    q = cell.fd.demand(rho)
                    v = cell.fd.velocity_from_density(rho)
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


class FixedScalarMask(ArbitraryMaskingCell):
    """
    Simple example masking cell with fixed scalar demand/supply.
    """

    def __init__(
        self,
        mask_id: str,
        network: Network,
        segments: List[MaskedSegmentRef],
        demand_value: float,
        supply_value: float,
    ):
        super().__init__(mask_id=mask_id, network=network, segments=segments)
        self.demand_value = float(demand_value)
        self.supply_value = float(supply_value)

    def demand(self, sim_time: float) -> float:
        return self.demand_value

    def supply(self, sim_time: float) -> float:
        return self.supply_value
    
class I24MicroMask(ArbitraryMaskingCell):

    def __init__(
        self,
        mask_id: str,
        network: Network,

        road_id: str,
        middle_s: float,
        margin_s: float
    ):
        super().__init__(mask_id=mask_id, network=network, 
                         segments=[
                             MaskedSegmentRef(road_id, -1, middle_s - margin_s, middle_s + margin_s),
                             MaskedSegmentRef(road_id, -2, middle_s - margin_s, middle_s + margin_s),
                             MaskedSegmentRef(road_id, -3, middle_s - margin_s, middle_s + margin_s),
                             MaskedSegmentRef(road_id, -4, middle_s - margin_s, middle_s + margin_s)
                         ])
        self.road_id = road_id,
        self.middle_s = middle_s
        self.margin_s = margin_s
    
    """
    Demand and Supply calculations here form our core contributions.
    We don't have the equations placed here just yet.
    """
    def demand(self, sim_time: float) -> float:
        return 0
    
    def supply(self, sim_time: float) -> float:
        return 0
    
    #def get
    
    def update(self, sim_time: float, dt: float, new_middle_s: float) -> I24MicroMask:
        return I24MicroMask(self.mask_id, self.network, self.road_id, new_middle_s, self.margin_s)

# =========================
# Example usage
# =========================

if __name__ == "__main__":
    # sim = Simulation.from_json(
    #     json_path="network.json",
    #     time_resolution=1.0,
    #     origin_time=1669819550.0,
    #     free_flow_speed=30.0,
    #     congestion_wave_speed=5.0,
    #     jam_density=0.16,
    # )
    #
    # gt = GroundTruthStore.from_parquet("ground_truth.parquet")
    # sim.initialize_from_ground_truth(gt, time_value=1669819550.0)
    #
    # mask = FixedScalarMask(
    #     mask_id="mask_1",
    #     network=sim.network,
    #     segments=[MaskedSegmentRef("road_a", -1, 100.0, 160.0)],
    #     demand_value=0.4,
    #     supply_value=0.2,
    # )
    # sim.add_masking_cell(mask)
    #
    # sim.run(duration=60.0)
    # print(sim.current_state_dataframe().head())
    pass
