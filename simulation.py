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
import plotly.io as pio                                                                                                                                                                                                      

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
    lane_change_model: Optional[LaneChangeModel] = None

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
        """Shallow-copy structure, deep-copy only cell densities."""
        new_roads: Dict[str, Road] = {}
        for road_id, road in self.roads.items():
            new_cells: Dict[str, Cell] = {
                cell_id: Cell(
                    road_id=cell.road_id,
                    cell_id=cell.cell_id,
                    lane=cell.lane,
                    start_s=cell.start_s,
                    end_s=cell.end_s,
                    density=cell.density,
                    inflow_connections=cell.inflow_connections,
                    outflow_connections=cell.outflow_connections,
                    fd=cell.fd,
                    lane_change_model=cell.lane_change_model,
                )
                for cell_id, cell in road.cells.items()
            }
            new_roads[road_id] = Road(
                road_id=road.road_id,
                left_polyline=road.left_polyline,
                right_polyline=road.right_polyline,
                lane_data=road.lane_data,
                cells=new_cells,
            )
        return Network(network_id=self.network_id, roads=new_roads)

    def all_cell_keys(self) -> List[Connection]:
        keys: List[Connection] = []
        for road_id, road in self.roads.items():
            for cell_id in road.cells:
                keys.append((road_id, cell_id))
        return keys

    @staticmethod
    def _fd_to_dict(fd: Optional["FundamentalDiagram"]) -> Optional[Dict[str, Any]]:
        if fd is None:
            return None
        if isinstance(fd, GreenshieldsFD):
            return {"type": "greenshields", "v_f": fd.v_f, "rho_j": fd.rho_j}
        if isinstance(fd, TriangularFD):
            return {"type": "triangular", "v_f": fd.v_f, "w": fd.w, "rho_j": fd.rho_j}
        raise ValueError(f"Unknown FundamentalDiagram type: {type(fd)}")

    @staticmethod
    def _fd_from_dict(data: Optional[Dict[str, Any]]) -> Optional["FundamentalDiagram"]:
        if data is None:
            return None
        t = data["type"]
        if t == "greenshields":
            return GreenshieldsFD(v_f=data["v_f"], rho_j=data["rho_j"])
        if t == "triangular":
            return TriangularFD(v_f=data["v_f"], w=data["w"], rho_j=data["rho_j"])
        raise ValueError(f"Unknown FundamentalDiagram type in JSON: {t}")

    @staticmethod
    def _lcm_to_dict(lcm: Optional["LaneChangeModel"]) -> Optional[Dict[str, Any]]:
        if lcm is None:
            return None
        if isinstance(lcm, NoLaneChange):
            return {"type": "no_lane_change"}
        if isinstance(lcm, SpeedIncentiveLaneChange):
            return {"type": "speed_incentive", "lambda_lc": lcm.lambda_lc}
        raise ValueError(f"Unknown LaneChangeModel type: {type(lcm)}")

    @staticmethod
    def _lcm_from_dict(data: Optional[Dict[str, Any]]) -> Optional["LaneChangeModel"]:
        if data is None:
            return None
        t = data["type"]
        if t == "no_lane_change":
            return NoLaneChange()
        if t == "speed_incentive":
            return SpeedIncentiveLaneChange(lambda_lc=data["lambda_lc"])
        raise ValueError(f"Unknown LaneChangeModel type in JSON: {t}")

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
                            "fd": Network._fd_to_dict(cell.fd),
                            "lane_change_model": Network._lcm_to_dict(cell.lane_change_model),
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
                    fd=Network._fd_from_dict(cell_data.get("fd")),
                    lane_change_model=Network._lcm_from_dict(cell_data.get("lane_change_model")),
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

    @staticmethod
    def _cell_polygon(road: "Road", start_s: float, end_s: float, lane: int) -> Tuple[List[float], List[float]]:
        """Return (x, y) coordinate lists for a cell polygon in road-world coordinates."""
        left_poly = np.array(road.left_polyline, dtype=float)
        right_poly = np.array(road.right_polyline, dtype=float)
        seg_lengths = np.linalg.norm(np.diff(left_poly, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        lane_info = road.lane_data[lane]
        lat = float(lane_info["lateral_position"])
        width = float(lane_info["width"])

        def interp(s: float) -> Tuple[np.ndarray, np.ndarray]:
            idx = int(np.clip(np.searchsorted(cumulative, s) - 1, 0, len(seg_lengths) - 1))
            t = (s - cumulative[idx]) / max(seg_lengths[idx], 1e-8)
            l = left_poly[idx] + t * (left_poly[idx + 1] - left_poly[idx])
            r = right_poly[idx] + t * (right_poly[idx + 1] - right_poly[idx])
            d = (r - l) / max(float(np.linalg.norm(r - l)), 1e-8)
            return l + (-lat) * d, l + (-(lat - width)) * d

        p1, p2 = interp(start_s)
        p3, p4 = interp(end_s)
        return [p1[0], p2[0], p4[0], p3[0], p1[0]], [p1[1], p2[1], p4[1], p3[1], p1[1]]

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
                x, y = Network._cell_polygon(road, cell.start_s, cell.end_s, cell.lane)
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
            width=2400,
            height=6400,
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
    def __init__(self, v_f, rho_j, lambda_lc):
        super().__init__()
        self.network = None
        self.fd = GreenshieldsFD(v_f=v_f, rho_j=rho_j)
        self.lane_change_model = SpeedIncentiveLaneChange(lambda_lc=lambda_lc)

    def create_network(self, road_length=1600.0, longitudinal_step=50.0, lane_count=4, lane_width=3.6576):
        network_id = "i24_westbound"
        road_id = "2"
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
                cell = Cell(road_id=road_id, cell_id=cell_id, lane=lane, start_s=start_s, end_s=end_s, density=density, inflow_connections=inflow_connections, outflow_connections=outflow_connections, fd=self.fd, lane_change_model=self.lane_change_model)
                cells[cell_id] = cell

        road = Road(road_id=road_id, left_polyline=road_left_polyline, right_polyline=road_right_polyline, lane_data=road_lane_data, cells=cells)
        self.network = Network(network_id=network_id, roads={road_id: road})

class I24EastBoundNetwork(NetworkGenerator):
    def __init__(self, v_f, rho_j, lambda_lc):
        super().__init__()
        self.network = None
        self.fd = GreenshieldsFD(v_f=v_f, rho_j=rho_j)
        self.lane_change_model = SpeedIncentiveLaneChange(lambda_lc=lambda_lc)

    def create_network(self, road_length=1600.0, longitudinal_step=50.0, lane_count=4, lane_width=3.6576):
        network_id = "i24_eastbound"
        road_id = "1"
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
                cell = Cell(road_id=road_id, cell_id=cell_id, lane=lane, start_s=start_s, end_s=end_s, density=density, inflow_connections=inflow_connections, outflow_connections=outflow_connections, fd=self.fd, lane_change_model=self.lane_change_model)
                cells[cell_id] = cell

        road = Road(road_id=road_id, left_polyline=road_left_polyline, right_polyline=road_right_polyline, lane_data=road_lane_data, cells=cells)
        self.network = Network(network_id=network_id, roads={road_id: road})

class I24WestAndEastNetwork(NetworkGenerator):
    def __init__(self, v_f=21.11708033086837, rho_j=0.0659339614507989, lambda_lc=0.3747470640406389):
        super().__init__()
        self.network = None
        self.v_f = v_f
        self.rho_j = rho_j
        self.lambda_lc = lambda_lc

    def create_network(self, road_length=1600.0, longitudinal_step=50.0, lane_count=4, lane_width=3.6576):
        network_id = "i24_west_and_east_network"
        westbound_network = I24WestBoundNetwork(self.v_f, self.rho_j, self.lambda_lc)
        eastbound_network = I24EastBoundNetwork(self.v_f, self.rho_j, self.lambda_lc)
        westbound_network.create_network(road_length, longitudinal_step, lane_count, lane_width)
        eastbound_network.create_network(road_length, longitudinal_step, lane_count, lane_width)
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
        "lane_id",
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

        # Fast lookup: sorted time array + {time: {(road_id, cell_id): density}}
        self._macro_density_lookup: Dict[float, Dict[Tuple[str, str], float]] = {
            float(t): {
                (str(r), str(c)): float(d)
                for r, c, d in zip(grp["road_id"], grp["cell_id"], grp["density"])
            }
            for t, grp in self.macro_df.groupby("time")
        }
        self._macro_times: np.ndarray = np.sort(np.array(list(self._macro_density_lookup.keys())))

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

    def _nearest_time(self, time_value: float, tolerance: float) -> float:
        idx = int(np.searchsorted(self._macro_times, time_value))
        candidates = [np.clip(idx, 0, len(self._macro_times) - 1),
                      np.clip(idx - 1, 0, len(self._macro_times) - 1)]
        chosen = float(self._macro_times[min(candidates, key=lambda i: abs(self._macro_times[i] - time_value))])
        if abs(chosen - time_value) > tolerance:
            raise KeyError(f"No ground-truth snapshot near time={time_value}. Closest: {chosen}.")
        return chosen

    def apply_density_snapshot_to_network(
        self, network: Network, time_value: float, tolerance: float = 1e-1
    ) -> None:
        density_map = self._macro_density_lookup[self._nearest_time(time_value, tolerance)]
        for (road_id, cell_id), density in density_map.items():
            network.get_cell(road_id, cell_id).density = density

    def apply_density_snapshot_to_network_boundaries(
        self, network: Network, time_value: float, tolerance: float = 1e-1
    ) -> None:
        density_map = self._macro_density_lookup[self._nearest_time(time_value, tolerance)]
        for (road_id, cell_id), density in density_map.items():
            cell = network.get_cell(road_id, cell_id)
            if len(cell.inflow_connections) == 0 or len(cell.outflow_connections) == 0:
                cell.density = density


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
        self.rear_flow = 0.0
        self.front_flow = 0.0
        self.validate()

    def validate(self) -> None:
        if len(self.segments) == 0:
            raise ValueError(f"Mask {self.mask_id} must have at least one segment.")
        for seg in self.segments:
            seg.validate()
            if seg.road_id not in self.network.roads:
                raise KeyError(f"Mask {self.mask_id}: road '{seg.road_id}' not found in {self.network.roads.keys()}.")
            if seg.lane not in self.network.roads[seg.road_id].lane_data:
                raise KeyError(
                    f"Mask {self.mask_id}: lane {seg.lane} not in road {seg.road_id}."
                )

    @abstractmethod
    def rear_boundary_flux(self, rho_exterior: float, fd_exterior: "FundamentalDiagram", sim_time: float, dt: float) -> float:
        """Net flux entering the mask through its rear (upstream) face.
        May be negative (backward flow out of rear face).
        """
        raise NotImplementedError

    @abstractmethod
    def front_boundary_flux(self, rho_exterior: float, fd_exterior: "FundamentalDiagram", sim_time: float, dt: float) -> float:
        """Net flux leaving the mask through its front (downstream) face.
        May be negative (backward flow into front face).
        """
        raise NotImplementedError

    def update(self, sim_time: float, dt: float) -> None:
        """
        Optional hook for moving masks.
        Override in subclasses.
        """
        return

    def render(self, rotate_fn) -> list:
        """Return a list of plotly traces for this mask's contents.

        Override in subclasses that have visual contents (e.g. vehicles).
        Default returns an empty list.
        """
        return []


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
    lane_change_model: Optional[LaneChangeModel] = None
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

    def snapshot(self) -> "ActiveNetwork":
        """Lightweight copy — shares structure, copies only densities."""
        snap = ActiveNetwork()
        for aid, ac in self.active_cells.items():
            snap.active_cells[aid] = ActiveCell(
                active_cell_id=ac.active_cell_id,
                road_id=ac.road_id,
                lane=ac.lane,
                start_s=ac.start_s,
                end_s=ac.end_s,
                kind=ac.kind,
                density=ac.density,
                base_segments=ac.base_segments,
                mask_id=ac.mask_id,
                fd=ac.fd,
                lane_change_model=ac.lane_change_model,
                inflow_neighbors=ac.inflow_neighbors,
                outflow_neighbors=ac.outflow_neighbors,
            )
        return snap

    def lateral_delta_density(self, dt: float) -> Dict[str, float]:
        """Compute per-cell density deltas from lateral lane exchange.

        Mask cells are excluded (treated as NoLaneChange). For each adjacent lane
        pair in each road, the lane change model is taken from the first cell in
        the lower-indexed lane.
        """
        by_road_lane: Dict[Tuple[str, int], List[ActiveCell]] = {}
        for ac in self.active_cells.values():
            if ac.kind == "mask":
                continue
            by_road_lane.setdefault((ac.road_id, ac.lane), []).append(ac)
        for key in by_road_lane:
            by_road_lane[key].sort(key=lambda c: c.start_s)

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
                model = cells_a[0].lane_change_model if cells_a else None
                if model is None:
                    continue
                for aid, d in model.lateral_delta_density_for_pair(cells_a, cells_b, dt).items():
                    delta[aid] = delta.get(aid, 0.0) + d
        return delta


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

                merged = self._merge_same_mask_intervals(raw_intervals)
                merged = self._merge_small_intervals(merged)

                for idx, item in enumerate(merged):
                    active_cell_id = f"{road_id}|lane{lane}|{idx}"
                    cell_fd = None
                    cell_lane_change_model = None
                    if item["kind"] == "normal" and item["base_segments"]:
                        cell_fd = self.network.get_cell(*item["base_segments"][0][0]).fd
                        cell_lane_change_model = self.network.get_cell(*item["base_segments"][0][0]).lane_change_model
                    elif item["kind"] == "normal":
                        print(f"Substituting {item}'s behavior model!")
                        print("---------------------------")
                        cell_fd = GreenshieldsFD(v_f=26.9, rho_j=0.065)
                        cell_lane_change_model = SpeedIncentiveLaneChange(lambda_lc=0.195)
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
                        lane_change_model=cell_lane_change_model
                    )

        self._build_neighbors(active)
        return active

    def _merge_same_mask_intervals(self, intervals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collapse consecutive intervals that share the same mask_id into one cell."""
        if not intervals:
            return []
        out = [dict(intervals[0])]
        for cur in intervals[1:]:
            prev = out[-1]
            if cur["mask_id"] is not None and cur["mask_id"] == prev["mask_id"]:
                prev["end_s"] = cur["end_s"]
                prev["base_segments"].extend(cur["base_segments"])
            else:
                out.append(dict(cur))
        return out

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
        #print(intervals)
        return intervals

    def _build_neighbors(self, active: ActiveNetwork) -> None:
        by_lane: Dict[Tuple[str, int], List[ActiveCell]] = {}
        for ac in active.active_cells.values():
            by_lane.setdefault((ac.road_id, ac.lane), []).append(ac)

        for _, lane_cells in by_lane.items():
            lane_cells.sort(key=lambda c: (c.start_s, c.end_s))
            for i, cell in enumerate(lane_cells):
                if i > 0:
                    prev = lane_cells[i - 1]
                    if not (prev.kind == "mask" and cell.kind == "mask"):
                        cell.inflow_neighbors.append(prev.active_cell_id)
                if i + 1 < len(lane_cells):
                    nxt = lane_cells[i + 1]
                    if not (cell.kind == "mask" and nxt.kind == "mask"):
                        cell.outflow_neighbors.append(nxt.active_cell_id)


class ConservativeRemapper:
    @staticmethod
    def base_to_active(network: Network, active: ActiveNetwork) -> None:
        for ac in active.active_cells.values():
            if ac.kind == "mask":
                continue  # mask interior is owned by the micro simulation
            total_mass = 0.0
            for (base_key, s0, s1) in ac.base_segments:
                base_cell = network.get_cell(*base_key)
                overlap_len = s1 - s0
                total_mass += float(base_cell.density * overlap_len)

            ac.density = 0.0 if ac.length <= 1e-12 else total_mass / ac.length

    @staticmethod
    def active_to_base(network: Network, active: ActiveNetwork) -> None:
        # Only accumulate mass from non-mask active cells.
        # Base cells exclusively covered by a mask are left unchanged — their
        # density is the micro simulation's responsibility (managed by the bridge).
        #
        # IMPORTANT: divide by *covered* length, not full base cell length.
        # A mask boundary can fall within a base cell, so the normal active cell
        # covers only a fraction of it.  Dividing by the full length would dilute
        # the density each step by that fraction, causing runaway decay at mask
        # boundaries.  Dividing by covered_length preserves the correct density
        # for the normal portion and leaves the masked portion untouched.
        base_mass_updates: Dict[Connection, float] = {}
        base_covered_lengths: Dict[Connection, float] = {}

        for ac in active.active_cells.values():
            if ac.kind == "mask":
                continue
            for (base_key, s0, s1) in ac.base_segments:
                overlap_len = s1 - s0
                if base_key not in base_mass_updates:
                    base_mass_updates[base_key] = 0.0
                    base_covered_lengths[base_key] = 0.0
                base_mass_updates[base_key] += float(ac.density * overlap_len)
                base_covered_lengths[base_key] += overlap_len

        for base_key, mass in base_mass_updates.items():
            cell = network.get_cell(*base_key)
            covered_L = base_covered_lengths[base_key]
            cell.density = 0.0 if covered_L <= 1e-12 else mass / covered_L


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
    def lateral_delta_density_for_pair(
        self,
        cells_a: List[ActiveCell],
        cells_b: List[ActiveCell],
        dt: float,
    ) -> Dict[str, float]:
        """Return per active-cell-id density deltas from lateral exchange between two adjacent lanes.

        cells_a and cells_b are sorted by start_s and represent one adjacent lane pair.
        """
        raise NotImplementedError


class NoLaneChange(LaneChangeModel):
    def lateral_delta_density_for_pair(
        self,
        cells_a: List[ActiveCell],
        cells_b: List[ActiveCell],
        dt: float,
    ) -> Dict[str, float]:
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

    def lateral_delta_density_for_pair(
        self,
        cells_a: List[ActiveCell],
        cells_b: List[ActiveCell],
        dt: float,
    ) -> Dict[str, float]:
        delta: Dict[str, float] = {}

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
    active_network: Optional[ActiveNetwork] = None
    mask_snapshots: Optional[Dict[str, Any]] = None


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
    ):
        self.network = network
        self.time_resolution = float(time_resolution)
        self.origin_time = float(origin_time)
        self.current_time = float(origin_time)
        self.rollout_results: List[RolloutStep] = []

        self.inflow_boundary_map = inflow_boundary_map or {}
        self.outflow_boundary_map = outflow_boundary_map or {}

        self.min_cell_length = float(min_cell_length) if min_cell_length is not None else 1.0

        self.masking_cells: Dict[str, ArbitraryMaskingCell] = {}
        self._step_callbacks: Dict[str, Any] = {}

        self.network.validate()

        self.gt_store = None
        self._cached_active_network: Optional[ActiveNetwork] = None

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

    def register_step_callback(self, fn, key) -> None:
        """Register a callable invoked at the start of each step as fn(sim_time, dt)."""
        self._step_callbacks[key] = fn

    def unregister_step_callback(self, key) -> None:
        if key in self._step_callbacks:
            del self._step_callbacks[key]

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
        if not self.masking_cells and self._cached_active_network is not None:
            active = self._cached_active_network
        else:
            builder = ActiveMeshBuilder(
                network=self.network,
                masks=self.masking_cells,
                min_cell_length=self.min_cell_length,
            )
            active = builder.build()
            if not self.masking_cells:
                self._cached_active_network = active
        ConservativeRemapper.base_to_active(self.network, active)
        return active

    def _compute_active_demand_supply(
        self, active: ActiveNetwork
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Returns demand and supply maps for normal cells only.
        Mask boundary fluxes are computed directly in _compute_active_edge_flows
        via each mask's rear_boundary_flux / front_boundary_flux methods.
        """
        demand_map: Dict[str, float] = {}
        supply_map: Dict[str, float] = {}

        for aid, ac in active.active_cells.items():
            if ac.kind == "normal":
                rho = float(ac.density)
                if ac.fd is not None:
                    demand_map[aid] = ac.fd.demand(rho)
                    supply_map[aid] = ac.fd.supply(rho)
                else:
                    raise Exception("No FD available!")
            elif ac.kind != "mask":
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
            if not succ:
                continue

            # Pre-split demand for normal upstream cells
            if u_cell.kind == "normal":
                per_out_demand = demand_map[u] / float(len(succ))

            for v in succ:
                v_cell = active.active_cells[v]
                if v_cell.kind == "mask":
                    # Normal → mask: rear boundary flux determined by mask's Riemann solver
                    mask = self.masking_cells[v_cell.mask_id]
                    
                    edge_flow[(u, v)] = mask.rear_boundary_flux(
                        u_cell.density, u_cell.fd, self.current_time, self.time_resolution
                    )
                elif u_cell.kind == "mask":
                    # Mask → normal: front boundary flux determined by mask's Riemann solver
                    mask = self.masking_cells[u_cell.mask_id]
                    edge_flow[(u, v)] = mask.front_boundary_flux(
                        v_cell.density, v_cell.fd, self.current_time, self.time_resolution
                    )
                else:
                    # Normal → normal: standard Godunov supply/demand
                    preds = list(v_cell.inflow_neighbors)
                    per_in_supply = supply_map[v] if not preds else supply_map[v] / float(len(preds))
                    edge_flow[(u, v)] = min(per_out_demand, per_in_supply)

        # Cap inflow to normal cells.
        # Mask boundary fluxes are fixed (Riemann solver output) and are not scaled —
        # only normal-normal edges are subject to capping. The fixed mask flux is
        # subtracted from remaining supply before scaling normal inflow.
        for v in active_ids:
            v_cell = active.active_cells[v]
            if v_cell.kind == "mask":
                continue
            all_incoming = [e for e in edge_flow if e[1] == v]
            normal_incoming = [e for e in all_incoming if active.active_cells[e[0]].kind == "normal"]
            if not normal_incoming:
                continue
            mask_inflow = sum(edge_flow[e] for e in all_incoming if e not in normal_incoming)
            normal_total_in = sum(edge_flow[e] for e in normal_incoming)
            remaining_supply = supply_map[v] - mask_inflow
            if normal_total_in > remaining_supply and normal_total_in > 1e-12:
                scale = remaining_supply / normal_total_in
                for e in normal_incoming:
                    edge_flow[e] *= scale

        # Cap outflow from normal cells, same logic.
        for u in active_ids:
            u_cell = active.active_cells[u]
            if u_cell.kind == "mask":
                continue
            all_outgoing = [e for e in edge_flow if e[0] == u]
            normal_outgoing = [e for e in all_outgoing if active.active_cells[e[1]].kind == "normal"]
            if not normal_outgoing:
                continue
            mask_outflow = sum(edge_flow[e] for e in all_outgoing if e not in normal_outgoing)
            normal_total_out = sum(edge_flow[e] for e in normal_outgoing)
            remaining_demand = demand_map[u] - mask_outflow
            if normal_total_out > remaining_demand and normal_total_out > 1e-12:
                scale = remaining_demand / normal_total_out
                for e in normal_outgoing:
                    edge_flow[e] *= scale

        # External road boundaries — mask cells have no external boundaries in the macro sense
        external_inflow: Dict[str, float] = {}
        external_outflow: Dict[str, float] = {}
        for aid in active_ids:
            ac = active.active_cells[aid]
            if ac.kind == "mask":
                continue
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
        lateral_deltas = active.lateral_delta_density(dt)

        #net_flow = np.zeros(len(active_ids), dtype=np.float64)
        rear_flow = np.zeros(len(active_ids), dtype=np.float64)
        front_flow = np.zeros(len(active_ids), dtype=np.float64)
        for (u, v), q in edge_flow.items():
            front_flow[index_of[u]] -= q
            rear_flow[index_of[v]] += q

        for aid, q in external_inflow.items():
            rear_flow[index_of[aid]] += q

        for aid, q in external_outflow.items():
            front_flow[index_of[aid]] -= q

        net_flow = rear_flow + front_flow
        new_densities = densities + (dt * (net_flow / np.maximum(lengths, 1e-12)))

        """
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
        """
        for i, aid in enumerate(active_ids):
            ac = active.active_cells[aid]
            if ac.kind != "mask":
                if ac.fd is None:
                    raise ValueError(f"Active cell {aid} has no FundamentalDiagram assigned.")
                new_rho = float(new_densities[i]) + lateral_deltas.get(aid, 0.0)
                ac.density = float(np.clip(new_rho, 0.0, ac.fd.rho_j))
            else:
                # We need to impart the new flow into the masked cells as needed.
                # We separately compute the flow in the rear of the cell and the front of the cell.
                rear_flow_ac = float(rear_flow[i]) * dt
                front_flow_ac = float(front_flow[i]) * dt
                self.masking_cells[ac.mask_id].rear_flow += rear_flow_ac
                self.masking_cells[ac.mask_id].front_flow += front_flow_ac

    def step(self) -> None:
        self._snapshot()
        callbacks = [cb for cb in self._step_callbacks]
        for cb in callbacks:
            self._step_callbacks[cb](self.current_time, self.time_resolution)
        self._update_masks()
        active = self._build_active_network()
        self.rollout_results[-1].active_network = active.snapshot()
        if self.masking_cells:
            self.rollout_results[-1].mask_snapshots = dict(self.masking_cells)
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

    def plot_rollout(self, rotation_deg: float = 0.0) -> "go.Figure":
        """Return an animated Plotly figure stepping through simulation rollouts.

        Parameters
        ----------
        rotation_deg : float
            Rotate all road geometry by this many degrees counter-clockwise.
            Use 90 to display a vertical road in landscape orientation.

        Interactive controls:
        - Slider to scrub through time steps
        - Play / Pause buttons
        - Six combination buttons: (Base | Active) × (Density | Velocity | Flow)
        """
        import matplotlib.cm as cm
        pio.json.config.default_engine = 'orjson'

        _angle = float(rotation_deg) * np.pi / 180.0
        _cos, _sin = float(np.cos(_angle)), float(np.sin(_angle))

        def _rotate(px: List[float], py: List[float]) -> Tuple[List[float], List[float]]:
            rx = [x * _cos - y * _sin for x, y in zip(px, py)]
            ry = [x * _sin + y * _cos for x, y in zip(px, py)]
            return rx, ry

        QUANTITIES = ["density", "velocity", "flow"]
        MASK_FILL = "#dc5050"   # rgba(220,80,80,0.85)
        MASK_LINE = "#b42828"
        NORMAL_LINE = "#505050"  # rgba(80,80,80,0.4)
        CMAP = cm.get_cmap("viridis")

        if not self.rollout_results:
            raise ValueError("No rollout results. Run the simulation first.")

        def _q_value(density: float, fd: Optional[FundamentalDiagram], q: str) -> float:
            if fd is None:
                return 0.0
            if q == "velocity":
                return fd.velocity_from_density(density)
            if q == "flow":
                return fd.demand(density)
            return density

        def _rgba(v: float, vmin: float, vmax: float) -> str:
            t = float(np.clip((v - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0))
            r, g, b, _ = CMAP(t)
            return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

        # Per-quantity global value ranges
        ranges: Dict[str, Tuple[float, float]] = {}
        for q in QUANTITIES:
            vals: List[float] = []
            for rs in self.rollout_results:
                for road in rs.network.roads.values():
                    for cell in road.cells.values():
                        vals.append(_q_value(cell.density, cell.fd, q))
                if rs.active_network:
                    for ac in rs.active_network.active_cells.values():
                        if ac.kind == "normal" and ac.fd is not None:
                            vals.append(_q_value(ac.density, ac.fd, q))
            ranges[q] = (float(min(vals)) if vals else 0.0, float(max(vals)) if vals else 1.0)

        # Base cell polygons — geometry fixed, only densities vary
        ref_network = self.rollout_results[0].network
        base_cells: List[Tuple[str, str, List[float], List[float], Optional[FundamentalDiagram]]] = []
        for road in ref_network.roads.values():
            for cell in road.cells.values():
                px, py = _rotate(*Network._cell_polygon(road, cell.start_s, cell.end_s, cell.lane))
                base_cells.append((road.road_id, cell.cell_id, px, py, cell.fd))

        active_per_step: List[List[ActiveCell]] = [
            list(rs.active_network.active_cells.values()) if rs.active_network else []
            for rs in self.rollout_results
        ]
        max_active = max((len(a) for a in active_per_step), default=0)
        N_base = len(base_cells)

        # Trace layout (per quantity q_idx in 0,1,2):
        #   base traces:   q_idx * N_base  ..  (q_idx+1) * N_base - 1
        #   active traces: 3*N_base + q_idx * max_active  ..  3*N_base + (q_idx+1) * max_active - 1
        #   colorbar:      3*N_base + 3*max_active + q_idx
        def base_start(q_idx: int) -> int:
            return q_idx * N_base
        def active_start(q_idx: int) -> int:
            return 3 * N_base + q_idx * max_active
        def colorbar_idx(q_idx: int) -> int:
            return 3 * N_base + 3 * max_active + q_idx

        N_total = 3 * N_base + 3 * max_active + 3

        def _visibility(show_base: bool, q_idx: int) -> List[bool]:
            vis = [False] * N_total
            group_start = base_start(q_idx) if show_base else active_start(q_idx)
            group_size = N_base if show_base else max_active
            for i in range(group_size):
                vis[group_start + i] = True
            vis[colorbar_idx(q_idx)] = True
            return vis

        # Build initial traces for all 3 quantities × 2 network views
        first_rs = self.rollout_results[0]
        first_active = active_per_step[0]
        traces: List[Any] = []

        for q_idx, q in enumerate(QUANTITIES):
            vmin, vmax = ranges[q]
            visible_initially = (q_idx == 0)  # show density / base on load

            # Base traces for this quantity
            for road_id, cell_id, px, py, fd in base_cells:
                cell = first_rs.network.get_cell(road_id, cell_id)
                traces.append(go.Scatter(
                    x=px, y=py,
                    fill="toself",
                    fillcolor=_rgba(_q_value(cell.density, fd, q), vmin, vmax),
                    mode="lines",
                    line=dict(color=NORMAL_LINE, width=0.5),
                    showlegend=False,
                    visible=visible_initially,
                    hoverinfo="skip",
                ))

        for q_idx, q in enumerate(QUANTITIES):
            vmin, vmax = ranges[q]

            # Active traces for this quantity (padded to max_active)
            for i in range(max_active):
                if i < len(first_active):
                    ac = first_active[i]
                    road = self.network.roads[ac.road_id]
                    px, py = _rotate(*Network._cell_polygon(road, ac.start_s, ac.end_s, ac.lane))
                    fill = MASK_FILL if ac.kind == "mask" else _rgba(_q_value(ac.density, ac.fd, q), vmin, vmax)
                    lcolor = MASK_LINE if ac.kind == "mask" else NORMAL_LINE
                else:
                    px, py = [], []
                    fill = "#000000"
                    lcolor = "#000000"
                traces.append(go.Scatter(
                    x=px, y=py,
                    fill="toself",
                    fillcolor=fill,
                    mode="lines",
                    line=dict(color=lcolor, width=0.5),
                    showlegend=False,
                    visible=False,
                    hoverinfo="skip",
                ))

        # Colorbar dummy traces (one per quantity)
        colorbar_x = [1.02, 1.10, 1.18]
        for q_idx, q in enumerate(QUANTITIES):
            vmin, vmax = ranges[q]
            traces.append(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(
                    colorscale="Viridis",
                    cmin=vmin, cmax=vmax,
                    color=[vmin],
                    showscale=True,
                    colorbar=dict(title=q, x=colorbar_x[q_idx], thickness=15),
                ),
                showlegend=False,
                visible=(q_idx == 0),
                hoverinfo="skip",
            ))
        print("Basic cells added!")

        # ── Precompute all colors and active geometry ───────────────────────
        N_frames = len(self.rollout_results)

        def _color_matrix(vals: np.ndarray, vmin: float, vmax: float) -> List[List[str]]:
            """(N_frames, N_cells) values → (N_frames, N_cells) hex color strings via vectorized colormap."""
            t = np.clip((vals - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
            rgb = (CMAP(t)[..., :3] * 255).astype(np.uint8)
            # Pack R,G,B into a single uint32 for fast hex formatting
            packed = (rgb[..., 0].astype(np.uint32) << 16
                      | rgb[..., 1].astype(np.uint32) << 8
                      | rgb[..., 2].astype(np.uint32))
            nf, nc = vals.shape
            return [
                [f"#{packed[i,j]:06x}" for j in range(nc)]
                for i in range(nf)
            ]

        # Base: density matrix (N_frames, N_base), then derive other quantities per cell
        base_rho = np.array([
            [rs.network.get_cell(rid, cid).density for rid, cid, _px, _py, _fd in base_cells]
            for rs in self.rollout_results
        ])  # (N_frames, N_base)

        base_frame_colors: Dict[str, List[List[str]]] = {}
        for q in QUANTITIES:
            vmin, vmax = ranges[q]
            if q == "density":
                vals = base_rho
            else:
                vals = np.zeros_like(base_rho)
                for j, (_rid, _cid, _px, _py, fd) in enumerate(base_cells):
                    vals[:, j] = [_q_value(rho, fd, q) for rho in base_rho[:, j]]
            base_frame_colors[q] = _color_matrix(vals, vmin, vmax)

        print("Rho computed!")

        # Active: precompute geometry and colors per step
        active_geo: List[List[Tuple[List, List]]] = []
        active_lcolor: List[List[str]] = []
        active_frame_colors: Dict[str, List[List[str]]] = {q: [] for q in QUANTITIES}

        for step_idx, step_active in enumerate(active_per_step):
            step_geo: List[Tuple[List, List]] = []
            step_lc: List[str] = []
            step_density: List[float] = []
            step_fd: List[Optional[FundamentalDiagram]] = []
            step_is_mask: List[bool] = []

            for i in range(max_active):
                if i < len(step_active):
                    ac = step_active[i]
                    road = self.network.roads[ac.road_id]
                    step_geo.append(_rotate(*Network._cell_polygon(road, ac.start_s, ac.end_s, ac.lane)))
                    step_lc.append(MASK_LINE if ac.kind == "mask" else NORMAL_LINE)
                    step_density.append(ac.density)
                    step_fd.append(ac.fd)
                    step_is_mask.append(ac.kind == "mask")
                else:
                    step_geo.append(([], []))
                    step_lc.append(NORMAL_LINE)
                    step_density.append(0.0)
                    step_fd.append(None)
                    step_is_mask.append(False)

            active_geo.append(step_geo)
            active_lcolor.append(step_lc)

            for q in QUANTITIES:
                vmin, vmax = ranges[q]
                step_colors = []
                for i in range(max_active):
                    if step_is_mask[i]:
                        step_colors.append(MASK_FILL)
                    elif step_fd[i] is not None:
                        t = float(np.clip((_q_value(step_density[i], step_fd[i], q) - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0))
                        r, g, b, _ = CMAP(t)
                        step_colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
                    else:
                        step_colors.append("#000000")
                active_frame_colors[q].append(step_colors)

        print("Active done!")

        # Precompute fixed frame_traces list (same for every frame)
        frame_traces_template: List[int] = []
        for q_idx in range(len(QUANTITIES)):
            for i in range(N_base):
                frame_traces_template.append(base_start(q_idx) + i)
        for q_idx in range(len(QUANTITIES)):
            for i in range(max_active):
                frame_traces_template.append(active_start(q_idx) + i)

        # ── Build frames using precomputed data and plain dicts ─────────────
        frames: List[go.Frame] = []
        for step_idx, rs in enumerate(self.rollout_results):
            frame_data: List[Any] = []

            for q in QUANTITIES:
                colors = base_frame_colors[q][step_idx]
                for color in colors:
                    frame_data.append({"fillcolor": color})

            for q in QUANTITIES:
                colors = active_frame_colors[q][step_idx]
                geo = active_geo[step_idx]
                lcs = active_lcolor[step_idx]
                for i in range(max_active):
                    px, py = geo[i]
                    frame_data.append({"x": px, "y": py, "fillcolor": colors[i],
                                       "line": {"color": lcs[i], "width": 0.5}})

            frames.append({"data": frame_data, "traces": frame_traces_template,
                           "name": str(int(rs.sim_time))})
            print(step_idx)

        # Slider
        slider_steps = [
            dict(
                args=[[str(int(rs.sim_time))], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                label=str(int(rs.sim_time)),
                method="animate",
            )
            for rs in self.rollout_results
        ]
        sliders = [dict(
            active=0,
            steps=slider_steps,
            x=0.05, xanchor="left",
            y=0.0, yanchor="top",
            len=0.9,
            pad={"t": 50},
            currentvalue=dict(prefix="t = ", visible=True, xanchor="center"),
            transition=dict(duration=0),
        )]

        # Playback buttons + 6 combination view buttons (Base|Active × Density|Velocity|Flow)
        view_buttons = []
        for show_base, net_label in [(True, "Base"), (False, "Active")]:
            for q_idx, q in enumerate(QUANTITIES):
                view_buttons.append(dict(
                    label=f"{net_label} · {q.capitalize()}",
                    method="update",
                    args=[{"visible": _visibility(show_base, q_idx)}],
                ))

        updatemenus = [
            dict(
                type="buttons", direction="left",
                x=0.05, y=2.0, xanchor="left", showactive=True,
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True, "mode": "immediate"}]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
                ],
            ),
            dict(
                type="buttons", direction="left",
                x=0.05, y=1.06, xanchor="left", showactive=True,
                buttons=view_buttons,
            ),
        ]

        fig = go.Figure(
            data=traces,
            layout=go.Layout(
                title="Simulation rollout",
                xaxis=dict(scaleanchor="y", showgrid=False),
                yaxis=dict(showgrid=False),
                template="plotly_white",
                sliders=sliders,
                updatemenus=updatemenus,
                margin=dict(t=120),
            ),
        )
        print("Final frames....")

        # Bypass Plotly's frame validator (which converts every frame dict into a
        # go.Frame object and validates each trace inside it — O(frames × traces)).
        # The serialization protocol only requires to_plotly_json(), so a thin
        # wrapper around the already-correct raw dicts is sufficient.
        class _RawFrame:
            def __init__(self, d: dict) -> None:
                self._props = d
            def to_plotly_json(self) -> dict:
                return self._props

        fig._frame_objs = [_RawFrame(f) for f in frames]
        return fig

    def build_rollout_renderer(self, rotation_deg: float = 0.0) -> "RolloutRenderer":
        """Precompute geometry and colours for all rollout steps.

        Returns a RolloutRenderer whose get_figure() can be called cheaply per
        frame, e.g. from a Dash slider callback.
        """
        return RolloutRenderer(self, rotation_deg)

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

    def load_rollout_parquet(self, path: str) -> pd.DataFrame:
        return pd.read_parquet(path)


class RolloutRenderer:
    """Precomputed rendering state for a Simulation rollout.

    Create once via sim.build_rollout_renderer(); then call get_figure()
    cheaply for any step (e.g. from a Dash callback).
    """

    QUANTITIES = ["density", "velocity", "flow"]
    MASK_FILL = "#dc5050"
    MASK_LINE = "#b42828"
    NORMAL_LINE = "#505050"

    def __init__(self, sim: "Simulation", rotation_deg: float = 0.0) -> None:
        import matplotlib.cm as cm
        CMAP = cm.get_cmap("viridis")
        self._cmap = CMAP

        if not sim.rollout_results:
            raise ValueError("No rollout results. Run the simulation first.")

        _angle = float(rotation_deg) * np.pi / 180.0
        _cos, _sin = float(np.cos(_angle)), float(np.sin(_angle))

        def _rotate(px: List[float], py: List[float]) -> Tuple[List[float], List[float]]:
            return (
                [x * _cos - y * _sin for x, y in zip(px, py)],
                [x * _sin + y * _cos for x, y in zip(px, py)],
            )

        self._rotate = _rotate

        def _q_value(density: float, fd: Optional[FundamentalDiagram], q: str) -> float:
            if fd is None:
                return 0.0
            if q == "velocity":
                return fd.velocity_from_density(density)
            if q == "flow":
                return fd.velocity_from_density(density) * density
            return density

        # Global per-quantity value ranges (used for consistent colorbar across all frames)
        self.ranges: Dict[str, Tuple[float, float]] = {}
        for q in self.QUANTITIES:
            vals: List[float] = []
            for rs in sim.rollout_results:
                for road in rs.network.roads.values():
                    for cell in road.cells.values():
                        vals.append(_q_value(cell.density, cell.fd, q))
                if rs.active_network:
                    for ac in rs.active_network.active_cells.values():
                        if ac.kind == "normal" and ac.fd is not None:
                            vals.append(_q_value(ac.density, ac.fd, q))
            self.ranges[q] = (float(min(vals)) if vals else 0.0, float(max(vals)) if vals else 1.0)

        # Base cell geometry — fixed for all frames
        ref_network = sim.rollout_results[0].network
        self.base_cells: List[Tuple[str, str, List[float], List[float], Optional[FundamentalDiagram]]] = []
        for road in ref_network.roads.values():
            for cell in road.cells.values():
                px, py = _rotate(*Network._cell_polygon(road, cell.start_s, cell.end_s, cell.lane))
                self.base_cells.append((road.road_id, cell.cell_id, px, py, cell.fd))

        active_per_step: List[List[ActiveCell]] = [
            list(rs.active_network.active_cells.values()) if rs.active_network else []
            for rs in sim.rollout_results
        ]
        self.max_active: int = max((len(a) for a in active_per_step), default=0)
        self.N_frames: int = len(sim.rollout_results)
        min_sim_time = min([rs.sim_time for rs in sim.rollout_results])
        self.sim_times: List[float] = [rs.sim_time - min_sim_time for rs in sim.rollout_results]
        self.mask_snapshots_per_step: List[Dict[str, Any]] = [
            rs.mask_snapshots or {} for rs in sim.rollout_results
        ]

        def _color_matrix(vals: np.ndarray, vmin: float, vmax: float) -> List[List[str]]:
            t = np.clip((vals - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
            rgb = (CMAP(t)[..., :3] * 255).astype(np.uint8)
            packed = (rgb[..., 0].astype(np.uint32) << 16
                      | rgb[..., 1].astype(np.uint32) << 8
                      | rgb[..., 2].astype(np.uint32))
            nf, nc = vals.shape
            return [[f"#{packed[i, j]:06x}" for j in range(nc)] for i in range(nf)]

        # Base colors: (N_frames, N_base) per quantity
        base_rho = np.array([
            [rs.network.get_cell(rid, cid).density for rid, cid, _px, _py, _fd in self.base_cells]
            for rs in sim.rollout_results
        ])
        self.base_frame_colors: Dict[str, List[List[str]]] = {}
        for q in self.QUANTITIES:
            vmin, vmax = self.ranges[q]
            if q == "density":
                vals_arr = base_rho
            else:
                vals_arr = np.zeros_like(base_rho)
                for j, (_rid, _cid, _px, _py, fd) in enumerate(self.base_cells):
                    vals_arr[:, j] = [_q_value(rho, fd, q) for rho in base_rho[:, j]]
            self.base_frame_colors[q] = _color_matrix(vals_arr, vmin, vmax)

        # Active cell geometry and colors per step
        self.active_geo: List[List[Tuple[List, List]]] = []
        self.active_lcolor: List[List[str]] = []
        self.active_frame_colors: Dict[str, List[List[str]]] = {q: [] for q in self.QUANTITIES}

        for step_active in active_per_step:
            step_geo: List[Tuple[List, List]] = []
            step_lc: List[str] = []
            step_density: List[float] = []
            step_fd: List[Optional[FundamentalDiagram]] = []
            step_is_mask: List[bool] = []

            for i in range(self.max_active):
                if i < len(step_active):
                    ac = step_active[i]
                    road = sim.network.roads[ac.road_id]
                    step_geo.append(_rotate(*Network._cell_polygon(road, ac.start_s, ac.end_s, ac.lane)))
                    step_lc.append(self.MASK_LINE if ac.kind == "mask" else self.NORMAL_LINE)
                    step_density.append(ac.density)
                    step_fd.append(ac.fd)
                    step_is_mask.append(ac.kind == "mask")
                else:
                    step_geo.append(([], []))
                    step_lc.append(self.NORMAL_LINE)
                    step_density.append(0.0)
                    step_fd.append(None)
                    step_is_mask.append(False)

            self.active_geo.append(step_geo)
            self.active_lcolor.append(step_lc)

            for q in self.QUANTITIES:
                vmin, vmax = self.ranges[q]
                step_colors: List[str] = []
                for i in range(self.max_active):
                    if step_is_mask[i]:
                        step_colors.append(self.MASK_FILL)
                    elif step_fd[i] is not None:
                        t = float(np.clip((_q_value(step_density[i], step_fd[i], q) - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0))
                        r, g, b, _ = CMAP(t)
                        step_colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
                    else:
                        step_colors.append("#000000")
                self.active_frame_colors[q].append(step_colors)

        # Time-space data: (road_id, lane) → per-quantity (N_frames, N_cells) arrays
        self.ts_data: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
        ref_net = sim.rollout_results[0].network
        macro_density_lookup: Dict[float, Dict[Tuple[str, str], float]] = {
            float(t): {
                (str(r), str(c)): (float(d), float(v))
                for r, c, d, v in zip(grp["road_id"], grp["cell_id"], grp["density"], grp["velocity"])
            }
            for t, grp in sim.gt_store.macro_df.groupby("time")
        }
        macro_times: np.ndarray = np.sort(np.array(list(macro_density_lookup.keys())))
        for road in ref_net.roads.values():
            for lane in sorted(road.lane_data):
                cells = road.cells_for_lane(lane)
                if not cells:
                    continue
                cell_ids = [c.cell_id for c in cells]
                s_mids = [(c.start_s + c.end_s) / 2.0 for c in cells]
                fds = [c.fd for c in cells]
                rho_ts = np.array([
                    [rs.network.get_cell(road.road_id, cid).density for cid in cell_ids]
                    for rs in sim.rollout_results
                ])  # (N_frames, N_cells)
                for version in ["sim", "empirical"]:
                    entry: Dict[str, Any] = {"s_mids": s_mids}
                    mask_extents: List[List[Optional[Tuple[float, float]]]] = [[]]
                    if (version == "sim"):
                        entry["density"] = rho_ts
                        for q in ("flow", "velocity"):
                            vals = np.zeros_like(rho_ts)
                            for j, fd in enumerate(fds):
                                vals[:, j] = [_q_value(rho, fd, q) for rho in rho_ts[:, j]]
                            entry[q] = vals

                        # Mask extents per timestep: (start_s, end_s) or None
                        for snapshots in self.mask_snapshots_per_step:
                            extent: Optional[Tuple[float, float]] = None
                            for mask in snapshots.values():
                                for seg in mask.segments:
                                    if seg.road_id == road.road_id and seg.lane == lane:
                                        lo, hi = float(seg.start_s), float(seg.end_s)
                                        extent = (lo, hi) if extent is None else (min(extent[0], lo), max(extent[1], hi))
                            if (extent == None) and (len(mask_extents[-1]) > 0) and (mask_extents[-1][-1] is not None):
                                mask_extents.append([])
                            elif (extent is not None) and (len(mask_extents[-1]) > 0) and (mask_extents[-1][-1] is None):
                                mask_extents.append([])
                            mask_extents[-1].append(extent)
                    elif (version == "empirical"):
                        vals_density = np.zeros((len(macro_times), len(cell_ids)))
                        vals_velocity = np.zeros_like(vals_density)
                        print(vals_density.shape, len(macro_times), len(cell_ids))
                        for i, t in enumerate(macro_times):
                            for j, cid in enumerate(cell_ids):
                                vals_density[i, j] = macro_density_lookup[t][(road.road_id, cid)][0]
                                vals_velocity[i, j] = macro_density_lookup[t][(road.road_id, cid)][1]
                        vals_flow = vals_density * vals_velocity
                        entry["density"] = vals_density
                        entry["velocity"] = vals_velocity
                        entry["flow"] = vals_flow
                    entry["mask_extents"] = mask_extents
                    self.ts_data[(road.road_id, lane, version)] = entry

    def get_ts_figure(self, road_id: str, lane: int, quantity: str = "density", version: str = "sim", render_masks: bool = True) -> go.Figure:
        """Return a time-space heatmap (viridis pcolormesh style) for one road+lane."""
        key = (road_id, lane, version)
        if key not in self.ts_data:
            return go.Figure()
        entry = self.ts_data[key]
        vmin, vmax = self.ranges[quantity]
        z = entry[quantity].T  # (N_cells, N_frames)
        fig = go.Figure(go.Heatmap(
            x=self.sim_times,
            y=entry["s_mids"],
            z=z,
            zmin=vmin,
            zmax=vmax,
            colorscale="Viridis",
            colorbar=dict(title=quantity),
            zsmooth=False,
        ))

        # Overlay mask trajectory as a filled band
        if render_masks:
            mask_position = 0
            for mask in entry["mask_extents"]:
                pairs = [
                    (t, ext)
                    for t, ext in zip(self.sim_times[mask_position:(mask_position + len(mask))], mask)
                    if ext is not None
                ]
                t_fwd = [t for t, _ in pairs]
                t_rev = t_fwd[::-1]
                s_lo = [ext[0] for _, ext in pairs]
                s_hi = [ext[1] for _, ext in pairs]
                fig.add_trace(go.Scatter(
                    x=t_fwd + t_rev,
                    y=s_lo + s_hi[::-1],
                    fill="toself",
                    fillcolor="rgba(220,80,80,0.25)",
                    line=dict(color="rgba(220,80,80,0.8)", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                mask_position += len(mask)

        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Position (m)",
            title=f"Road {road_id} · Lane {lane} · {quantity} · {version}",
            template="plotly_white",
            margin=dict(t=60),
            uirevision="ts-constant",
        )
        return fig

    def get_figure(self, step_idx: int, show_base: bool = True, quantity: str = "density") -> go.Figure:
        """Return a static go.Figure for a single simulation timestep."""
        q = quantity
        vmin, vmax = self.ranges[q]
        traces: List[Any] = []

        if show_base:
            colors = self.base_frame_colors[q][step_idx]
            for (_rid, _cid, px, py, _fd), color in zip(self.base_cells, colors):
                traces.append(go.Scatter(
                    x=px, y=py,
                    fill="toself",
                    fillcolor=color,
                    mode="lines",
                    line=dict(color=self.NORMAL_LINE, width=0.5),
                    showlegend=False,
                    hoverinfo="skip",
                ))
        else:
            colors = self.active_frame_colors[q][step_idx]
            geo = self.active_geo[step_idx]
            lcs = self.active_lcolor[step_idx]
            for i in range(self.max_active):
                px, py = geo[i]
                traces.append(go.Scatter(
                    x=px, y=py,
                    fill="toself",
                    fillcolor=colors[i],
                    mode="lines",
                    line=dict(color=lcs[i], width=0.5),
                    showlegend=False,
                    hoverinfo="skip",
                ))
            for mask in self.mask_snapshots_per_step[step_idx].values():
                traces.extend(mask.render(self._rotate))

        traces.append(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(
                colorscale="Viridis",
                cmin=vmin, cmax=vmax,
                color=[vmin],
                showscale=True,
                colorbar=dict(title=q, x=1.02, thickness=15),
            ),
            showlegend=False,
            hoverinfo="skip",
        ))

        return go.Figure(
            data=traces,
            layout=go.Layout(
                title=f"t = {int(self.sim_times[step_idx])} s",
                xaxis=dict(scaleanchor="y", showgrid=False),
                yaxis=dict(showgrid=False),
                template="plotly_white",
                margin=dict(t=60),
                uirevision="constant",
            ),
        )


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

    def rear_boundary_flux(self, rho_exterior: float, fd_exterior: "FundamentalDiagram", sim_time: float, dt: float) -> float:
        return min(fd_exterior.demand(rho_exterior), self.supply_value)

    def front_boundary_flux(self, rho_exterior: float, fd_exterior: "FundamentalDiagram", sim_time: float, dt: float) -> float:
        return min(self.demand_value, fd_exterior.supply(rho_exterior))
    
class I24MicroMask(ArbitraryMaskingCell):
    """Single-lane micro-domain mask for one lane of the I-24 moving window.

    Four of these (one per lane) are managed together by MicroSimBridge,
    which replaces them each step with updated position and boundary values.
    """

    def __init__(
        self,
        mask_id: str,
        network: Network,
        road_id: str,
        lane: int,
        middle_s: float,
        margin_s: float,
        anchor_speed: float,
        rear_flux_memory: float,
        front_flux_memory: float
    ):
        super().__init__(
            mask_id=mask_id,
            network=network,
            segments=[MaskedSegmentRef(road_id, lane, middle_s - margin_s, middle_s + margin_s)],
        )
        self.road_id = road_id
        self.lane = lane
        self.middle_s = middle_s
        self.margin_s = margin_s
        self.vehicles: Dict[str, Any] = {}
        self.anchor_speed = anchor_speed
        self.rear_flux_memory = rear_flux_memory
        self.front_flux_memory = front_flux_memory

    def get_rear_vehicle(self):
        vehicle = None
        for new_vehicle_key in self.vehicles:
            if (vehicle is None) or (self.vehicles[new_vehicle_key].s < vehicle.s):
                vehicle = self.vehicles[new_vehicle_key]
        return vehicle
    
    def get_front_vehicle(self):
        vehicle = None
        for new_vehicle_key in self.vehicles:
            if (vehicle is None) or (self.vehicles[new_vehicle_key].s > vehicle.s):
                vehicle = self.vehicles[new_vehicle_key]
        return vehicle

    """
    Boundary flux calculations here form our core contributions — to be
    derived from the constrained Riemann solver once the weak entropy
    solution is in hand.
    """
    """
    def rear_boundary_flux(self, rho_exterior: float, fd_exterior: "FundamentalDiagram", sim_time: float, dt: float) -> float:
        rear_vehicle = self.get_rear_vehicle()
        if rear_vehicle is not None:
            rho_interior = 1.0 / rear_vehicle.s
            rear_vehicle_flux = rho_interior * (rear_vehicle.s_dt - self.anchor_speed) * dt
            rear_vehicle_flux = min(0.0, rear_vehicle_flux)
        else:
            rear_vehicle_flux = 0.0
        rear_macro_flux = rho_exterior * fd_exterior.velocity_from_density(rho_exterior) * dt
        rear_micro_geometric_flux = rho_exterior * fd_exterior.velocity_from_density(self.anchor_speed) * dt
        total_non_clipped_flux = rear_macro_flux - rear_micro_geometric_flux + rear_vehicle_flux
        self.rear_flux_memory += total_non_clipped_flux
        return total_non_clipped_flux

    def front_boundary_flux(self, rho_exterior: float, fd_exterior: "FundamentalDiagram", sim_time: float, dt: float) -> float:
        front_vehicle = self.get_front_vehicle()
        if front_vehicle is not None:
            rho_interior = 1.0 / (self.middle_s + self.margin_s - front_vehicle.s)
            front_vehicle_flux = rho_interior * (front_vehicle.s_dt - self.anchor_speed) * dt
            front_vehicle_flux = max(0.0, front_vehicle_flux)
        else:
            front_vehicle_flux = 0.0
        front_macro_flux = rho_exterior * fd_exterior.velocity_from_density(rho_exterior) * dt
        front_micro_geometric_flux = rho_exterior * fd_exterior.velocity_from_density(self.anchor_speed) * dt
        total_non_clipped_flux = front_micro_geometric_flux - front_macro_flux + front_vehicle_flux
        self.front_flux_memory += total_non_clipped_flux
        return total_non_clipped_flux
    """

    def rear_boundary_flux(self, rho_exterior: float, fd_exterior: "FundamentalDiagram", sim_time: float, dt: float) -> float:
        rear_vehicle = self.get_rear_vehicle()
        if rear_vehicle is not None:
            interior_s = rear_vehicle.s
            vehicle_leaving = (interior_s < 0)
            rho_interior = min(1.0 / interior_s, fd_exterior.rho_j) if not vehicle_leaving else fd_exterior.rho_j
        else:
            rho_interior = 0.0
            vehicle_leaving = False
        available_supply = fd_exterior.supply(rho_interior) *  (0 if vehicle_leaving else 1) * dt
        vehicle_leaving_flux = -1 if vehicle_leaving else 0
        flux_cap = available_supply + vehicle_leaving_flux

        flux_demand_moving = (fd_exterior.demand(rho_exterior) - rho_exterior*self.anchor_speed) * dt
        flux_reconciled = min(flux_cap, flux_demand_moving)
        self.rear_flux_memory += flux_reconciled
        return flux_reconciled

    def front_boundary_flux(self, rho_exterior: float, fd_exterior: "FundamentalDiagram", sim_time: float, dt: float) -> float:
        front_vehicle = self.get_front_vehicle()
        if front_vehicle is not None:
            interior_s = front_vehicle.s
            vehicle_leaving = ((self.middle_s + self.margin_s - interior_s) < 0)
            rho_interior = min(1.0 / (self.middle_s + self.margin_s - interior_s), fd_exterior.rho_j) if not vehicle_leaving else fd_exterior.rho_j
        else:
            rho_interior = 0.0
            vehicle_leaving = False

        available_exterior_supply = fd_exterior.supply(rho_exterior) - rho_exterior*self.anchor_speed * dt
        vehicle_leaving_flux = 1 if vehicle_leaving else 0
        flux_capped_for_external = min(available_exterior_supply, vehicle_leaving_flux)

        vehicle_can_enter = (not vehicle_leaving) and (rho_interior < fd_exterior.rho_j)
        vehicle_entering_flux_allowed = 1 if vehicle_can_enter else 0
        available_interior_supply = -fd_exterior.supply(rho_interior) * vehicle_entering_flux_allowed * dt
        flux_reconciled = max(available_interior_supply, flux_capped_for_external)

        self.front_flux_memory += flux_reconciled
        return flux_reconciled


    def render(self, rotate_fn) -> list:
        """Draw each vehicle in this lane as a filled rectangle."""
        road = self.network.roads[self.road_id]
        s_offset = self.middle_s - self.margin_s
        traces = []
        for vehicle in self.vehicles.values():
            if vehicle.lane != self.lane:
                continue
            s_abs = vehicle.s + s_offset
            s_start = s_abs
            s_end = s_abs + vehicle.length
            px, py = rotate_fn(*Network._cell_polygon(road, s_start, s_end, self.lane))
            traces.append(go.Scatter(
                x=px, y=py,
                fill="toself",
                fillcolor="#1f77b4",
                mode="lines",
                line=dict(color="#0d4f8b", width=0.5),
                showlegend=False,
                hoverinfo="skip",
            ))
        # Render Flux Memories
        # Rear Flux
        s_start = self.middle_s - self.margin_s
        s_end = self.middle_s + self.margin_s
        px, py = rotate_fn(*Network._cell_polygon(road, s_start - 10.0, s_start, self.lane))
        traces.append(go.Scatter(
            x=px, y=py,
            fill="toself",
            fillcolor="#ffffff",
            mode="lines",
            line=dict(color="#0d4f8b", width=0.5),
            showlegend=False,
            text=f"Rear: {self.rear_flux_memory}, Anchor Speed: {self.anchor_speed}"
        ))

        # Front Flux
        px, py = rotate_fn(*Network._cell_polygon(road, s_end, s_end + 10.0, self.lane))
        traces.append(go.Scatter(
            x=px, y=py,
            fill="toself",
            fillcolor="#ffffff",
            mode="lines",
            line=dict(color="#0d4f8b", width=0.5),
            showlegend=False,
            text=f"Front: {self.front_flux_memory}, Anchor Speed: {self.anchor_speed}"
        ))
        return traces

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