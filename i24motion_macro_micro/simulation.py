from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, is_dataclass, replace
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
    mass: float
    mask_mass: float

    # Length over which `mass` was last spread by active_to_base, i.e. the unmasked
    # portion of this cell at write time. base_to_active must divide `mass` by THIS
    # length, not by the unmasked length of the newly rebuilt mesh: for a moving mask
    # the two differ every step, and dividing by the new one compresses the whole
    # cell's mass into the shrinking sliver. None means "never masked, use length".
    macro_length: Optional[float] = None

    # Mean speed (m/s). Only second-order macro models (METANETModel) carry speed as
    # an independent state variable; under the default first-order Godunov/CTM path
    # this stays None and speed is recovered from the FD as v = q(rho)/rho.
    velocity: Optional[float] = None

    inflow_connections: List[Connection] = field(default_factory=list)
    outflow_connections: List[Connection] = field(default_factory=list)
    fd: Optional[FundamentalDiagram] = None
    lane_change_model: Optional[LaneChangeModel] = None

    @property
    def length(self) -> float:
        return float(self.end_s - self.start_s)
    
    @property
    def density(self) -> float:
        return float(self.mass / self.length)
    
    @property
    def density_viz(self) -> float:
        return float((self.mass + self.mask_mass) / self.length)

    def validate(self) -> None:
        if self.end_s <= self.start_s:
            raise ValueError(
                f"Cell {self.road_id}/{self.cell_id} has non-positive length: "
                f"start_s={self.start_s}, end_s={self.end_s}"
            )
        if self.mass < 0.0:
            raise ValueError(
                f"Cell {self.road_id}/{self.cell_id} has negative mass: {self.mass}"
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
        # Base-cell membership and extents are static after construction, so the
        # sorted per-lane list only needs computing once. Callers treat the result
        # as read-only. (If cells are ever mutated, drop `_lane_cells_cache`.)
        cache = getattr(self, "_lane_cells_cache", None)
        if cache is None:
            cache = {}
            self._lane_cells_cache = cache
        out = cache.get(lane)
        if out is None:
            out = [c for c in self.cells.values() if c.lane == lane]
            out.sort(key=lambda c: (c.start_s, c.end_s, c.cell_id))
            cache[lane] = out
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
                    mass=cell.mass,
                    mask_mass=cell.mask_mass,
                    velocity=cell.velocity,
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
                            "mass": float(cell.mass),
                            "mask_mass": float(cell.mask_mass),
                            "velocity": None if cell.velocity is None else float(cell.velocity),
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
                    mass=float(cell_data["mass"]),
                    mask_mass=float(cell_data["mask_mass"]),
                    velocity=(
                        None if cell_data.get("velocity") is None
                        else float(cell_data["velocity"])
                    ),
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
    def __init__(self, fd, lambda_lc):# , v_f, rho_j, lambda_lc):
        super().__init__()
        self.network = None
        #self.fd = GreenshieldsFD(v_f=v_f, rho_j=rho_j)
        self.fd = fd
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
        road_lane_data = {k: v for (k, v) in road_lane_data.items() if k >= (-lane_count)}

        longitudinal_steps = np.arange(0.0, road_length, longitudinal_step).tolist()
        cells = {}
        for i, step in enumerate(longitudinal_steps):
            for lane in road_lane_data:
                cell_id = f"road_{road_id}_cell_{lane}_step_{i}"
                start_s = step
                end_s = step + longitudinal_step
                mass = 0
                mask_mass = 0
                inflow_connections = []
                outflow_connections = []
                if (i > 0):
                    inflow_connections.append((road_id, f"road_{road_id}_cell_{lane}_step_{i - 1}"))
                if (i < (len(longitudinal_steps) - 1)):
                    outflow_connections.append((road_id, f"road_{road_id}_cell_{lane}_step_{i + 1}"))
                cell = Cell(road_id=road_id, cell_id=cell_id, lane=lane, start_s=start_s, end_s=end_s, mass=mass, mask_mass=mask_mass, inflow_connections=inflow_connections, outflow_connections=outflow_connections, fd=self.fd, lane_change_model=self.lane_change_model)
                cells[cell_id] = cell

        road = Road(road_id=road_id, left_polyline=road_left_polyline, right_polyline=road_right_polyline, lane_data=road_lane_data, cells=cells)
        self.network = Network(network_id=network_id, roads={road_id: road})

class I24EastBoundNetwork(NetworkGenerator):
    def __init__(self, fd, lambda_lc):# , v_f, rho_j, lambda_lc):
        super().__init__()
        self.network = None
        #self.fd = GreenshieldsFD(v_f=v_f, rho_j=rho_j)
        self.fd = fd
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
        road_lane_data = {k: v for (k, v) in road_lane_data.items() if k >= (-lane_count)}

        longitudinal_steps = np.arange(0.0, road_length, longitudinal_step).tolist()
        cells = {}
        for i, step in enumerate(longitudinal_steps):
            for lane in road_lane_data:
                cell_id = f"road_{road_id}_cell_{lane}_step_{i}"
                start_s = step
                end_s = step + longitudinal_step
                mass = 0
                mask_mass = 0
                inflow_connections = []
                outflow_connections = []
                if (i > 0):
                    inflow_connections.append((road_id, f"road_{road_id}_cell_{lane}_step_{i - 1}"))
                if (i < (len(longitudinal_steps) - 1)):
                    outflow_connections.append((road_id, f"road_{road_id}_cell_{lane}_step_{i + 1}"))
                cell = Cell(road_id=road_id, cell_id=cell_id, lane=lane, start_s=start_s, end_s=end_s, mass=mass, mask_mass=mask_mass, inflow_connections=inflow_connections, outflow_connections=outflow_connections, fd=self.fd, lane_change_model=self.lane_change_model)
                cells[cell_id] = cell

        road = Road(road_id=road_id, left_polyline=road_left_polyline, right_polyline=road_right_polyline, lane_data=road_lane_data, cells=cells)
        self.network = Network(network_id=network_id, roads={road_id: road})

class I24WestAndEastNetwork(NetworkGenerator):
    def __init__(self, fd, lambda_lc=0.05):
        super().__init__()
        self.network = None
        self.fd = fd
        self.lambda_lc = lambda_lc

    def create_network(self, road_length=1600.0, longitudinal_step=50.0, lane_count=4, lane_width=3.6576):
        network_id = "i24_west_and_east_network"
        westbound_network = I24WestBoundNetwork(self.fd, self.lambda_lc)
        eastbound_network = I24EastBoundNetwork(self.fd, self.lambda_lc)
        westbound_network.create_network(road_length, longitudinal_step, lane_count, lane_width)
        eastbound_network.create_network(road_length, longitudinal_step, lane_count, lane_width)
        self.network = Network.merge_networks(westbound_network.network, eastbound_network.network, network_id)

class I24WestAndEastNetworkCollapsed(NetworkGenerator):
    """`I24WestAndEastNetwork` with each carriageway's lanes collapsed into one cell.

    One cell per longitudinal station per road instead of `lane_count` of them. The
    collapsed cell's `mass` is the total across lanes, so its `density` is the total
    (all-lane) density and the interface flux is total flow. This is the mesh METANET
    is normally written on: pair it with `METANETParams.lanes` set to the lane count
    and keep the paper's PER-LANE `rho_crit`/`kappa` (see `METANETModel`). Lane counts
    per road are recorded on `lanes_per_road` after `create_network`.

    Lateral exchange disappears by construction -- `lateral_delta_density` pairs
    adjacent lanes and there are none left -- so the collapsed cells carry
    `NoLaneChange`. Lane changing is no longer modelled explicitly; it is implicit in
    the aggregated dynamics.

    The collapsed `fd` is the per-lane `fd` with `rho_j` (and hence capacity) scaled
    by the lane count, so a first-order CTM run -- or just the rollout colouring --
    sees an aggregate jam density rather than a single lane's.

    `collapse_lanes` is a plain `Network -> Network` conversion, so it also works on a
    network loaded from JSON (e.g. `i24_westbound_network.json`), keeping the real
    polylines. It is a construction-time helper: it refuses a network carrying live
    remap state (`macro_length`).
    """

    def __init__(self, fd, lambda_lc=0.05):
        super().__init__()
        self.network = None
        self.fd = fd
        self.lambda_lc = lambda_lc
        # Lanes aggregated into each road's cells; feed this to METANETParams.lanes.
        self.lanes_per_road: Dict[str, int] = {}

    def create_network(self, road_length=1600.0, longitudinal_step=50.0, lane_count=4, lane_width=3.6576):
        network_id = "i24_west_and_east_network_collapsed"
        source = I24WestAndEastNetwork(self.fd, self.lambda_lc)
        source.create_network(road_length, longitudinal_step, lane_count, lane_width)
        self.lanes_per_road = {
            road_id: len(road.lane_data) for road_id, road in source.network.roads.items()
        }
        self.network = I24WestAndEastNetworkCollapsed.collapse_lanes(source.network, network_id)

    @staticmethod
    def scale_fd(fd: Optional[FundamentalDiagram], lanes: int) -> Optional[FundamentalDiagram]:
        """Per-lane FD -> lane-aggregate FD: rho_j scales with lanes, speeds do not."""
        if fd is None or lanes == 1:
            return fd
        if not is_dataclass(fd) or not hasattr(fd, "rho_j"):
            raise ValueError(
                f"Cannot scale {type(fd).__name__} to {lanes} lanes: it is not a "
                f"dataclass with a rho_j field. Build the aggregate FD by hand."
            )
        return replace(fd, rho_j=float(fd.rho_j) * lanes)

    @staticmethod
    def collapse_lanes(network: Network, network_id: Optional[str] = None) -> Network:
        """Return a copy of `network` with every road's lanes merged into one chain."""
        # Pass 1: per road, the station list and the source-cell -> collapsed-cell map,
        # so connections can be rewritten in pass 2 (including across roads).
        per_road: Dict[str, Dict[str, Any]] = {}
        id_map: Dict[Connection, Connection] = {}
        for road_id, road in network.roads.items():
            lanes = sorted(road.lane_data)
            lane_count = len(lanes)

            stations: Dict[Tuple[float, float], List[Cell]] = {}
            for cell in road.cells.values():
                if cell.macro_length is not None:
                    raise ValueError(
                        f"Cell {road_id}/{cell.cell_id} carries macro_length: "
                        f"collapse_lanes is a construction-time helper and cannot "
                        f"merge a network mid-run."
                    )
                key = (round(float(cell.start_s), 9), round(float(cell.end_s), 9))
                stations.setdefault(key, []).append(cell)

            ordered = sorted(stations)
            collapsed_lane = max(lanes)
            for i, extent in enumerate(ordered):
                group = stations[extent]
                if len(group) != lane_count:
                    raise ValueError(
                        f"Road {road_id} station {extent} has {len(group)} cells but "
                        f"the road has {lane_count} lanes: lanes are not aligned "
                        f"longitudinally, so they cannot be collapsed."
                    )
                collapsed_id = f"road_{road_id}_cell_collapsed_step_{i}"
                for cell in group:
                    id_map[(road_id, cell.cell_id)] = (road_id, collapsed_id)

            per_road[road_id] = {
                "road": road,
                "lanes": lanes,
                "stations": stations,
                "ordered": ordered,
                "collapsed_lane": collapsed_lane,
            }

        # Pass 2: build the collapsed cells.
        roads: Dict[str, Road] = {}
        for road_id, info in per_road.items():
            road: Road = info["road"]
            lanes: List[int] = info["lanes"]
            lane_count = len(lanes)
            collapsed_lane: int = info["collapsed_lane"]

            # A single lane spanning the whole carriageway, so the cell polygons the
            # plots draw still cover exactly the lanes they replaced.
            top = max(float(road.lane_data[l]["lateral_position"]) for l in lanes)
            bottom = min(
                float(road.lane_data[l]["lateral_position"]) - float(road.lane_data[l]["width"])
                for l in lanes
            )
            lane_data = {collapsed_lane: {"lateral_position": top, "width": top - bottom}}

            cells: Dict[str, Cell] = {}
            for i, extent in enumerate(info["ordered"]):
                group: List[Cell] = info["stations"][extent]
                collapsed_id = id_map[(road_id, group[0].cell_id)][1]

                inflow: List[Connection] = []
                outflow: List[Connection] = []
                for cell in group:
                    for conn in cell.inflow_connections:
                        mapped = id_map[conn]
                        if mapped not in inflow:
                            inflow.append(mapped)
                    for conn in cell.outflow_connections:
                        mapped = id_map[conn]
                        if mapped not in outflow:
                            outflow.append(mapped)

                mass = sum(float(c.mass) for c in group)
                # Mean speed is a per-lane quantity, so it is mass-weighted rather
                # than summed. Only second-order models set it; None stays None.
                velocities = [(float(c.mass), float(c.velocity)) for c in group if c.velocity is not None]
                if not velocities:
                    velocity = None
                elif mass > 1e-12:
                    velocity = sum(m * v for m, v in velocities) / mass
                else:
                    velocity = sum(v for _, v in velocities) / len(velocities)

                cells[collapsed_id] = Cell(
                    road_id=road_id,
                    cell_id=collapsed_id,
                    lane=collapsed_lane,
                    start_s=float(extent[0]),
                    end_s=float(extent[1]),
                    mass=mass,
                    mask_mass=sum(float(c.mask_mass) for c in group),
                    velocity=velocity,
                    inflow_connections=inflow,
                    outflow_connections=outflow,
                    fd=I24WestAndEastNetworkCollapsed.scale_fd(group[0].fd, lane_count),
                    lane_change_model=NoLaneChange(),
                )

            roads[road_id] = Road(
                road_id=road_id,
                left_polyline=road.left_polyline,
                right_polyline=road.right_polyline,
                lane_data=lane_data,
                cells=cells,
            )

        return Network(network_id=network_id or f"{network.network_id}_collapsed", roads=roads)

class SimplifiedOneLaneRoadNetwork(NetworkGenerator):
    def __init__(self, fd, lambda_lc=0.05):
        super().__init__()
        self.network = None
        self.fd = fd
        self.lane_change_model = SpeedIncentiveLaneChange(lambda_lc=lambda_lc)

    def create_network(self, road_length, longitudinal_step=50.0, lane_width=3.6576):
        network_id = "simplified_one_lane_road_network"
        road_id = "1"
        offset_from_medium = lane_width
        road_width = lane_width
        starting_x = 0.0
        starting_y = 0.0
        ending_x = road_length
        ending_y = 0.0

        road_left_polyline = [
            (starting_x, starting_y),
            (ending_x, ending_y)
        ]
        road_right_polyline = [
            (starting_x, starting_y - road_width),
            (ending_x, ending_y - road_width)
        ]

        road_lane_data = {
            -1: {
                "width": lane_width,
                "lateral_position": 0.0
            }
        }

        longitudinal_steps = np.arange(0.0, road_length, longitudinal_step).tolist()
        cells = {}
        for i, step in enumerate(longitudinal_steps):
            for lane in road_lane_data:
                cell_id = f"road_{road_id}_cell_{lane}_step_{i}"
                start_s = step
                end_s = min(step + longitudinal_step, road_length)
                mass = 0
                mask_mass = 0
                inflow_connections = []
                outflow_connections = []
                if (i > 0):
                    inflow_connections.append((road_id, f"road_{road_id}_cell_{lane}_step_{i - 1}"))
                if (i < (len(longitudinal_steps) - 1)):
                    outflow_connections.append((road_id, f"road_{road_id}_cell_{lane}_step_{i + 1}"))
                cell = Cell(road_id=road_id, cell_id=cell_id, lane=lane, start_s=start_s, end_s=end_s, mass=mass, mask_mass=mask_mass, inflow_connections=inflow_connections, outflow_connections=outflow_connections, fd=self.fd, lane_change_model=self.lane_change_model)
                cells[cell_id] = cell

        road = Road(road_id=road_id, left_polyline=road_left_polyline, right_polyline=road_right_polyline, lane_data=road_lane_data, cells=cells)
        self.network = Network(network_id=network_id, roads={road_id: road})

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
    def __init__(self, micro_df: pd.DataFrame = None, macro_df: pd.DataFrame = None):
        if micro_df is not None:
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

        if macro_df is not None:
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
    def from_parquet(micro_parquet_path: str = None, macro_parquet_path: str = None) -> "GroundTruthStore":
        micro_df = pd.read_parquet(micro_parquet_path) if micro_parquet_path is not None else None
        macro_df = pd.read_parquet(macro_parquet_path) if macro_parquet_path is not None else None
        return GroundTruthStore(micro_df, macro_df)

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

    # The empirical macroscopic data is aggregated at 1 Hz, so a simulation whose
    # step is shorter than that (see EnvConfig.macro_dt, which can run the outer
    # loop at 10 Hz) asks for times that fall between snapshots.  The tolerance
    # below spans a whole interval, which holds the nearest snapshot until the next
    # one starts; at a 1 s step it still picks the exact match.  The initial
    # condition is a different matter and stays strict: see
    # apply_density_snapshot_to_network, whose caller passes 1e-6.
    BOUNDARY_TIME_TOLERANCE = 1.0

    def get_empirical_densities_at_time(self, time_value: float, tolerance: float = BOUNDARY_TIME_TOLERANCE):
        density_map = self._macro_density_lookup[self._nearest_time(time_value, tolerance)]
        return density_map

    def apply_density_snapshot_to_network(
        self, network: Network, time_value: float, tolerance: float = 1e-1
    ) -> None:
        density_map = self._macro_density_lookup[self._nearest_time(time_value, tolerance)]
        for (road_id, cell_id), density in density_map.items():
            network.get_cell(road_id, cell_id).mass = density * network.get_cell(road_id, cell_id).length

    def apply_density_snapshot_to_network_boundaries(
        self, network: Network, time_value: float, tolerance: float = BOUNDARY_TIME_TOLERANCE
    ) -> None:
        density_map = self._macro_density_lookup[self._nearest_time(time_value, tolerance)]
        for (road_id, cell_id), density in density_map.items():
            cell = network.get_cell(road_id, cell_id)
            if len(cell.inflow_connections) == 0 or len(cell.outflow_connections) == 0:
                cell.mass = density * cell.length


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
            
    @property
    def mass(self) -> float:
        return 0

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

    def update(self, sim_time: float, dt: float) -> float:
        """
        Optional hook for moving masks. Returns the longitudinal forward
        speed (m/s) at which the mask should be translated this step.
        A return of 0.0 means the mask is stationary.

        Override in subclasses.
        """
        return 0.0

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
    mass: float
    # Mean speed (m/s); see Cell.velocity. None under the first-order path.
    velocity: Optional[float] = None
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
    def density(self) -> float:
        return float(self.mass / self.length)


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
                mass=ac.mass,
                velocity=ac.velocity,
                base_segments=ac.base_segments,
                mask_id=ac.mask_id,
                fd=ac.fd,
                lane_change_model=ac.lane_change_model,
                inflow_neighbors=ac.inflow_neighbors,
                outflow_neighbors=ac.outflow_neighbors,
            )
        return snap
    
    def get_cell_with_mask(self, mask_id):
        for aid, ac in self.active_cells.items():
            if ac.mask_id == mask_id:
                return ac
        return None

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
                ncells = len(lane_cells)
                ci = 0  # monotonic pointer into (sorted) lane_cells

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

                    # Every base-cell boundary is a cut point, so [a, b] lies wholly
                    # within a single base cell (or a gap between cells). Walk the
                    # monotonic pointer to that cell instead of scanning all cells.
                    while ci < ncells and lane_cells[ci].end_s <= a + 1e-9:
                        ci += 1
                    if ci >= ncells:
                        continue
                    base_cell = lane_cells[ci]
                    if base_cell.start_s > a + 1e-9 or base_cell.end_s < b - 1e-9:
                        # interval falls in a gap between base cells
                        continue

                    raw_intervals.append(
                        {
                            "road_id": road_id,
                            "lane": lane,
                            "start_s": a,
                            "end_s": b,
                            "kind": "mask" if mask_id is not None else "normal",
                            "mask_id": mask_id,
                            "base_segments": [((base_cell.road_id, base_cell.cell_id), a, b)],
                        }
                    )

                merged = self._merge_same_mask_intervals(raw_intervals)
                merged = self._merge_small_intervals(merged)

                lane_active_cells: List[ActiveCell] = []
                for idx, item in enumerate(merged):
                    active_cell_id = f"{road_id}|lane{lane}|{idx}"
                    cell_fd = None
                    cell_lane_change_model = None
                    if item["kind"] == "normal" and item["base_segments"]:
                        base_cell = self.network.get_cell(*item["base_segments"][0][0])
                        cell_fd = base_cell.fd
                        cell_lane_change_model = base_cell.lane_change_model
                    elif item["kind"] == "normal":
                        print(f"Substituting {item}'s behavior model!")
                        print("---------------------------")
                        cell_fd = GreenshieldsFD(v_f=26.9, rho_j=0.065)
                        cell_lane_change_model = SpeedIncentiveLaneChange(lambda_lc=0.195)
                    ac = ActiveCell(
                        active_cell_id=active_cell_id,
                        road_id=road_id,
                        lane=lane,
                        start_s=float(item["start_s"]),
                        end_s=float(item["end_s"]),
                        kind=str(item["kind"]),
                        mass=0.0,
                        base_segments=list(item["base_segments"]),
                        mask_id=item["mask_id"],
                        fd=cell_fd,
                        lane_change_model=cell_lane_change_model
                    )
                    active.active_cells[active_cell_id] = ac
                    lane_active_cells.append(ac)

                # Link neighbours inline: cells are already emitted in start_s order
                # within this lane, so no regrouping/sorting is needed.
                for i, cell in enumerate(lane_active_cells):
                    if i > 0:
                        prev = lane_active_cells[i - 1]
                        if not (prev.kind == "mask" and cell.kind == "mask"):
                            cell.inflow_neighbors.append(prev.active_cell_id)
                    if i + 1 < len(lane_active_cells):
                        nxt = lane_active_cells[i + 1]
                        if not (cell.kind == "mask" and nxt.kind == "mask"):
                            cell.outflow_neighbors.append(nxt.active_cell_id)

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
                # Shallow-copy the dict and its base_segments list; the segment
                # tuples inside are immutable, so no deep copy is needed.
                cur = dict(intervals[i])
                cur["base_segments"] = list(cur["base_segments"])
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
                    # base_segments is rebuilt as a fresh list below, so a shallow
                    # dict copy is safe (the original list is never mutated).
                    right = dict(intervals[i + 1])
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


class ConservativeRemapper:
    @staticmethod
    def base_to_active(simulation: Simulation, network: Network, active: ActiveNetwork) -> None:
        # `cell.mass` represents macro mass on the *unmasked* portion of the base
        # cell. To distribute it back to normal active cells, divide by the unmasked
        # length (= sum of overlaps with normal active cells), not the full base
        # length. Otherwise mass on partially-masked base cells decays each cycle
        # because mass intended for the masked portion is silently dropped.
        for ac in active.active_cells.values():
            total_mass = 0.0
            # Velocity is intensive, so it is transported as a mass-weighted average
            # over the overlapping base segments (with a length-weighted fallback for
            # near-empty cells, where the mass weights carry no information).
            vel_mass_num = 0.0
            vel_mass_den = 0.0
            vel_len_num = 0.0
            vel_len_den = 0.0
            if ac.kind == "mask":
                total_mass = float(simulation.masking_cells[ac.mask_id].mass)
            else:
                for (base_key, s0, s1) in ac.base_segments:
                    base_cell = network.get_cell(*base_key)
                    overlap_len = s1 - s0
                    # Divide by the length `mass` was written over, NOT by this mesh's
                    # unmasked length. They differ whenever the mask moved since the
                    # last active_to_base, and using the new one inflates the seam
                    # cells by full_length/unmasked_length every step.
                    macro_len = base_cell.macro_length
                    if macro_len is None:
                        macro_len = base_cell.length
                    macro_density = (base_cell.mass / macro_len) if macro_len > 1e-12 else 0.0
                    seg_mass = macro_density * overlap_len
                    total_mass += seg_mass
                    if base_cell.velocity is not None:
                        vel_mass_num += base_cell.velocity * seg_mass
                        vel_mass_den += seg_mass
                        vel_len_num += base_cell.velocity * overlap_len
                        vel_len_den += overlap_len

            ac.mass = 0.0 if ac.length <= 1e-12 else total_mass
            if vel_mass_den > 1e-12:
                ac.velocity = vel_mass_num / vel_mass_den
            elif vel_len_den > 1e-12:
                ac.velocity = vel_len_num / vel_len_den

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
        base_mask_mass_updates: Dict[Connection, float] = {}
        base_covered_lengths: Dict[Connection, float] = {}
        # Mass-weighted and length-weighted velocity accumulators, mirroring
        # base_to_active. Mask active cells are skipped: speed inside the bubble is
        # the micro simulation's state, not the macro model's.
        base_vel_mass: Dict[Connection, Tuple[float, float]] = {}
        base_vel_len: Dict[Connection, Tuple[float, float]] = {}

        # Reset all masses.
        for ac in active.active_cells.values():
            for (base_key, s0, s1) in ac.base_segments:
                network.get_cell(*base_key).mass = 0.0
                network.get_cell(*base_key).mask_mass = 0.0

        for ac in active.active_cells.values():
            if ac.kind == "mask":
                for (base_key, s0, s1) in ac.base_segments:
                    overlap_len = s1 - s0
                    if base_key not in base_mask_mass_updates:
                        base_mask_mass_updates[base_key] = 0.0
                    base_mask_mass_updates[base_key] += float(ac.density * overlap_len)
            else:
                for (base_key, s0, s1) in ac.base_segments:
                    overlap_len = s1 - s0
                    if base_key not in base_mass_updates:
                        base_mass_updates[base_key] = 0.0
                        base_covered_lengths[base_key] = 0.0
                    seg_mass = float(ac.density * overlap_len)
                    base_mass_updates[base_key] += seg_mass
                    base_covered_lengths[base_key] += overlap_len
                    if ac.velocity is not None:
                        m_num, m_den = base_vel_mass.get(base_key, (0.0, 0.0))
                        base_vel_mass[base_key] = (
                            m_num + ac.velocity * seg_mass, m_den + seg_mass
                        )
                        l_num, l_den = base_vel_len.get(base_key, (0.0, 0.0))
                        base_vel_len[base_key] = (
                            l_num + ac.velocity * overlap_len, l_den + overlap_len
                        )

        for base_key, mass in base_mass_updates.items():
            cell = network.get_cell(*base_key)
            covered_l = base_covered_lengths[base_key]
            cell.mass = 0.0 if covered_l <= 1e-12 else mass
            # Remember the length this mass was spread over so base_to_active can
            # recover the density even after the mask has moved.
            cell.macro_length = covered_l if covered_l > 1e-12 else None

        for base_key, mass in base_mask_mass_updates.items():
            cell = network.get_cell(*base_key)
            cell.mask_mass = mass

        for base_key in base_vel_mass:
            cell = network.get_cell(*base_key)
            m_num, m_den = base_vel_mass[base_key]
            l_num, l_den = base_vel_len[base_key]
            if m_den > 1e-12:
                cell.velocity = m_num / m_den
            elif l_den > 1e-12:
                cell.velocity = l_num / l_den

    @staticmethod
    def move_active_masks(simulation: "Simulation", active: ActiveNetwork) -> None:
        """Translate every mask forward by `mask.update() * dt`.

        For each mask:
          - calls `mask.update(sim_time, dt)` to obtain a longitudinal forward speed,
          - shifts the mask's own `MaskedSegmentRef`s by `dx = speed * dt`,
          - shifts every active cell with `kind == "mask"` and matching `mask_id` by dx,
          - shifts the abutting upstream cell's `end_s` and the abutting downstream
            cell's `start_s` by dx so neighbours stay contiguous,
          - rebuilds `base_segments` on every cell whose extent changed by intersecting
            the new [start_s, end_s] with the underlying base cells in (road_id, lane).
        """
        sim_time = simulation.current_time
        dt = simulation.time_resolution
        network = simulation.network

        def recompute_base_segments(ac: ActiveCell) -> None:
            road = network.roads[ac.road_id]
            new_segs: List[Tuple[Connection, float, float]] = []
            for c in road.cells_for_lane(ac.lane):
                s0 = max(float(ac.start_s), float(c.start_s))
                s1 = min(float(ac.end_s), float(c.end_s))
                if s1 > s0:
                    new_segs.append(((c.road_id, c.cell_id), float(s0), float(s1)))
            ac.base_segments = new_segs

        mask_active_cells: Dict[str, List[ActiveCell]] = {}
        for ac in active.active_cells.values():
            if ac.kind == "mask" and ac.mask_id is not None:
                mask_active_cells.setdefault(ac.mask_id, []).append(ac)

        cells_by_road_lane: Dict[Tuple[str, int], List[ActiveCell]] = {}
        for ac in active.active_cells.values():
            cells_by_road_lane.setdefault((ac.road_id, ac.lane), []).append(ac)

        TOL = 1e-9
        for mask_id, mask in simulation.masking_cells.items():
            speed = float(mask.update(sim_time, dt))
            dx = speed * dt
            if dx == 0.0:
                continue

            shifted: Set[str] = set()
            for mac in mask_active_cells.get(mask_id, []):
                old_start = float(mac.start_s)
                old_end = float(mac.end_s)
                lane_cells = cells_by_road_lane.get((mac.road_id, mac.lane), [])
                for other in lane_cells:
                    if other is mac or other.kind == "mask":
                        continue
                    if abs(other.end_s - old_start) < TOL:
                        other.end_s = float(other.end_s + dx)
                        shifted.add(other.active_cell_id)
                    if abs(other.start_s - old_end) < TOL:
                        other.start_s = float(other.start_s + dx)
                        shifted.add(other.active_cell_id)
                mac.start_s = float(old_start + dx)
                mac.end_s = float(old_end + dx)
                shifted.add(mac.active_cell_id)

            for seg in mask.segments:
                seg.start_s = float(seg.start_s + dx)
                seg.end_s = float(seg.end_s + dx)

            for aid in shifted:
                recompute_base_segments(active.active_cells[aid])


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

    @abstractmethod
    def density_from_velocity(self, v: float) -> float:
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
    
    def _flow(self, rho: float) -> float:
        if (rho > self.rho_c):
            return self.w * (self.rho_j - rho)
        else:
            return self.v_f * rho

    def demand(self, rho: float) -> float:
        return min(self.v_f * rho, self.capacity)

    def supply(self, rho: float) -> float:
        return min(self.capacity, self.w * max(self.rho_j - rho, 0.0))

    def velocity_from_density(self, rho: float) -> float:
        q = self._flow(rho)
        return q / rho if rho > 1e-12 else self.v_f

    def density_from_velocity(self, v: float) -> float:
        if (v >= self.v_f):
            return self.rho_c
        elif (v >= 0.0):
            return (self.w * self.rho_j) / (v + self.w)
        else:
            return self.rho_j
    
    def shock_speed(self, rho_left: float, rho_right: float) -> float:
        both_free = (rho_left <= self.rho_c) and (rho_right <= self.rho_c)
        both_cong = (rho_left >= self.rho_c) and (rho_right >= self.rho_c)

        # Same-branch jumps are contacts at the branch speed, in EITHER direction.
        if both_free:
            return self.v_f
        if both_cong:
            return -self.w

        # Straddling rho_c. Up-jump (free left, congested right) is a single shock.
        if rho_left < rho_right:
            # rho_left free, rho_right congested -> admissible shock.
            return (self._flow(rho_left) - self._flow(rho_right)) / (rho_left - rho_right)


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
    
    def density_from_velocity(self, v: float) -> float:
        if (v >= self.v_f):
            return 0.0
        elif (v >= 0.0):
            return self.rho_j * (1.0 - (v/self.v_f))
        else:
            return self.rho_j
    
    def sonic_point(self, y: float) -> float:
        return ((self.v_f - y) * self.rho_j) / (2 * self.v_f)
    
    def shock_speed(self, rho_left: float, rho_right: float) -> float:
        return self.v_f * (1.0 - ((rho_left + rho_right) / self.rho_j))


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
# Macroscopic models
# =========================


class MacroModel(ABC):
    """Strategy governing how the active mesh advances each step.

    The default path (``Simulation.macro_model is None``) is first-order
    Godunov/CTM: cell state is density alone and the interface flux is
    ``min(demand(rho_u), supply(rho_v))`` read off each cell's `FundamentalDiagram`.

    A `MacroModel` replaces that flux rule and may carry extra per-cell state
    (e.g. METANET's mean speed). Everything else — external boundaries, lateral
    lane exchange, mask coupling, the base<->active remap — is shared.
    """

    @abstractmethod
    def prepare(self, simulation: "Simulation", active: "ActiveNetwork") -> None:
        """Called once at the top of each step, before any state is mutated."""
        raise NotImplementedError

    @abstractmethod
    def sending_flow(self, ac: "ActiveCell") -> float:
        """Flow (veh/s) this cell offers downstream. Replaces ``fd.demand``."""
        raise NotImplementedError

    @abstractmethod
    def receiving_flow(self, ac: "ActiveCell") -> float:
        """Flow (veh/s) this cell will accept. Replaces ``fd.supply``.

        Return ``math.inf`` for models with no supply-side constraint.
        """
        raise NotImplementedError

    def boundary_capacity(self, ac: "ActiveCell") -> Optional[float]:
        """Default cap at an unconnected downstream end; ``None`` means uncapped.

        Only used when `outflow_boundary_map` has no entry for the cell.
        """
        return ac.fd.capacity if ac.fd is not None else None

    def advance(self, simulation: "Simulation", active: "ActiveNetwork") -> None:
        """Called after the density update to advance any extra state."""
        return None


@dataclass
class METANETParams:
    """METANET parameters in SI units (metres, seconds, veh/m).

    Chan, Raghavan & Wu, "Open-Source METANET Calibration for Reproducible Freeway
    Traffic Macroscopic Simulation" (ITSC 2026), arXiv:2605.23042, eqs. (1)-(4).

    The paper states these in veh/km, km/h and hours; use `from_paper_units` to
    convert. `rho_crit` and `kappa` are the paper's PER-LANE values converted to
    veh/m, and `lanes` is the paper's lambda: the number of lanes the cell holding
    this state represents. A cell's `density` is always the total across those lanes,
    so the per-lane density the speed equation needs is `density / lanes`. With the
    default `lanes=1` (one `Cell` == one lane) this reduces to the paper's rho.
    """

    tau: float        # relaxation time (s)
    eta: float        # anticipation constant (m^2/s)
    kappa: float      # smoothing constant (veh/m/lane)
    v_free: float     # free-flow speed v* (m/s)
    rho_crit: float   # critical density rho* (veh/m/lane)
    alpha: float      # model exponent (dimensionless)
    r: float = 0.0    # on-ramp inflow (veh/s), source term in eq. (1)
    beta: float = 0.0  # off-ramp split ratio (dimensionless), eq. (1)
    lanes: float = 1.0  # lambda: lanes aggregated into one cell, eq. (1)

    @classmethod
    def from_paper_units(
        cls,
        tau_h: float,
        eta_km2_per_h: float,
        kappa_veh_per_km_lane: float,
        v_free_kmh: float,
        rho_crit_veh_per_km_lane: float,
        alpha: float,
        r_veh_per_h: float = 0.0,
        beta: float = 0.0,
        lanes: float = 1.0,
    ) -> "METANETParams":
        """Build from the paper's units (hours, km, km/h, veh/km/lane).

        eta carries units of [length]x[speed] so that eta/L is a speed; the paper's
        bounds (5-60) and synthetic value (30) are in km^2/h, matching the standard
        METANET literature.
        """
        return cls(
            tau=float(tau_h) * 3600.0,
            eta=float(eta_km2_per_h) * (1.0e6 / 3600.0),
            kappa=float(kappa_veh_per_km_lane) / 1000.0,
            v_free=float(v_free_kmh) / 3.6,
            rho_crit=float(rho_crit_veh_per_km_lane) / 1000.0,
            alpha=float(alpha),
            r=float(r_veh_per_h) / 3600.0,
            beta=float(beta),
            lanes=float(lanes),
        )

    @classmethod
    def synthetic_bottleneck(cls) -> "METANETParams":
        """Ground-truth parameters for the paper's synthetic scenario (Table II)."""
        return cls.from_paper_units(
            tau_h=18.0 / 3600.0,
            eta_km2_per_h=30.0,
            kappa_veh_per_km_lane=40.0,
            v_free_kmh=120.0,
            rho_crit_veh_per_km_lane=37.45,
            alpha=1.4,
        )

    def per_lane_density(self, rho: float) -> float:
        """Paper's rho/lambda: total cell density -> per-lane density (veh/m/lane)."""
        return float(rho) / self.lanes if self.lanes > 0.0 else float(rho)

    def equilibrium_speed(self, rho: float) -> float:
        """V[rho] from eq. (4): v* exp[-(1/alpha) ((rho/lambda) / rho*)^alpha].

        `rho` is the cell's TOTAL density across its `lanes` lanes; rho* is per lane.
        """
        if rho <= 0.0:
            return self.v_free
        ratio = self.per_lane_density(rho) / self.rho_crit
        return self.v_free * math.exp(-(1.0 / self.alpha) * (ratio ** self.alpha))


class METANETModel(MacroModel):
    """Second-order METANET dynamics on the active mesh (arXiv:2605.23042).

    Density follows eq. (1) with the upwind flux q = rho*v of eq. (2); mean speed is
    an independent state variable advanced by eq. (3) from relaxation toward the
    equilibrium speed V[rho] of eq. (4), upwind convection, and downstream
    anticipation.

    Differences from the reference implementation, all reducing to it exactly on the
    uniform ramp-free mesh the paper uses:

    - The finite differences use centre-to-centre spacing rather than a single global
      L, so the model stays well-posed on the non-uniform mesh the active builder
      produces around a mask. On a uniform mesh centre spacing equals L.
    - Parameters may vary per base cell (the paper's segment-varying calibration)
      via `per_cell_params`, keyed by ``(road_id, cell_id)``.
    - Lane count lambda lives on `METANETParams.lanes` (default 1, i.e. one `Cell`
      per lane). Set it to aggregate a whole carriageway into one cell per
      longitudinal station: the cell's density/mass are then totals across those
      lanes, the flux q = rho*v is total flow, and only the speed equation converts
      back to per-lane density via `per_lane_density`. Lambda may vary along the
      road (lane drops) through `per_cell_params`; the extra lane-drop term of the
      full METANET speed equation is not modelled.

    Boundary conditions follow the paper: upstream demand (via the simulation's
    `inflow_boundary_map`) and downstream density (via `downstream_density_map`).
    Where either is unset the corresponding gradient term is taken as zero.
    """

    def __init__(
        self,
        params: METANETParams,
        per_cell_params: Optional[Dict[Connection, METANETParams]] = None,
        upstream_velocity_map: Optional[Dict[Connection, float]] = None,
        downstream_density_map: Optional[Dict[Connection, float]] = None,
        v_min: float = 0.0,
        clamp_to_v_free: bool = False,
    ):
        self.params = params
        self.per_cell_params = per_cell_params or {}
        self.upstream_velocity_map = upstream_velocity_map or {}
        self.downstream_density_map = downstream_density_map or {}
        # Speeds are floored because a negative mean speed would reverse the upwind
        # flux of eq. (2) and break the scheme. The paper specifies no upper clamp,
        # so v may transiently exceed v_free unless clamp_to_v_free is set.
        self.v_min = float(v_min)
        self.clamp_to_v_free = bool(clamp_to_v_free)
        # State at time t, captured by prepare() before the density update overwrites
        # it. eq. (3) is evaluated entirely at time t.
        self._prev: Dict[str, Tuple[float, float]] = {}

    def params_for(self, ac: "ActiveCell") -> METANETParams:
        if self.per_cell_params and ac.base_segments:
            return self.per_cell_params.get(ac.base_segments[0][0], self.params)
        return self.params

    def prepare(self, simulation: "Simulation", active: "ActiveNetwork") -> None:
        self._prev = {}
        for aid, ac in active.active_cells.items():
            if ac.kind != "normal":
                continue
            if ac.velocity is None:
                # Cold start: begin at equilibrium for the current density.
                ac.velocity = self.params_for(ac).equilibrium_speed(float(ac.density))
            self._prev[aid] = (float(ac.density), float(ac.velocity))

    def sending_flow(self, ac: "ActiveCell") -> float:
        # eq. (2): q = rho * v, pure upwind. No supply-side min().
        v = ac.velocity
        if v is None:
            v = self.params_for(ac).equilibrium_speed(float(ac.density))
        return max(0.0, float(ac.density) * float(v))

    def receiving_flow(self, ac: "ActiveCell") -> float:
        # METANET has no supply function: a cell accepts whatever is sent to it.
        return math.inf

    def boundary_capacity(self, ac: "ActiveCell") -> Optional[float]:
        # Free outflow at the downstream end unless outflow_boundary_map says otherwise.
        return None

    @staticmethod
    def _centre_distance(a: "ActiveCell", b: "ActiveCell") -> float:
        d = abs(((b.start_s + b.end_s) - (a.start_s + a.end_s)) * 0.5)
        return d if d > 1e-9 else max(a.length, 1e-9)

    def advance(self, simulation: "Simulation", active: "ActiveNetwork") -> None:
        dt = float(simulation.time_resolution)
        cells = active.active_cells

        for aid, (rho, v) in self._prev.items():
            ac = cells[aid]
            p = self.params_for(ac)

            # Neighbour state is read from _prev, never off the live cells: by the
            # time advance() runs, the density update has already written rho_{t+1}
            # and earlier iterations of this loop have written v_{t+1}. Mask cells
            # are absent from _prev, which gives the hybrid seams their zero-gradient
            # fallback for free.

            # --- upwind neighbour: supplies v_{t,x-1} for the convection term ---
            v_up, dx_up = v, ac.length
            up = self._neighbour(cells, ac.inflow_neighbors)
            if up is not None:
                # HYBRID SEAM: a mask neighbour has no macro speed state (velocity is
                # None inside the bubble), so we fall back to zero-gradient. Coupling
                # a second-order macro model to the micro region means handing the
                # mask a speed boundary condition here as well as a flux.
                up_prev = self._prev.get(up.active_cell_id)
                if up_prev is not None:
                    v_up = up_prev[1]
                    dx_up = self._centre_distance(up, ac)
            else:
                key = ac.base_segments[0][0] if ac.base_segments else None
                if key in self.upstream_velocity_map:
                    v_up = float(self.upstream_velocity_map[key])

            # --- downstream neighbour: supplies rho_{t,x+1} for anticipation ---
            # The anticipation term of eq. (3) is written in the paper's per-lane
            # density, so both sides of the gradient are converted with their OWN
            # lane count: across a lane drop the per-lane density jumps even when the
            # total is continuous, which is exactly what the term should feel.
            rho_lane = p.per_lane_density(rho)
            rho_dn_lane, dx_dn = rho_lane, ac.length
            dn = self._neighbour(cells, ac.outflow_neighbors)
            if dn is not None:
                # HYBRID SEAM: mask density is micro-derived; using it here would
                # feed micro state into the macro speed update. Deliberately left as
                # zero-gradient until the hybrid coupling is designed.
                dn_prev = self._prev.get(dn.active_cell_id)
                if dn_prev is not None:
                    rho_dn_lane = self.params_for(dn).per_lane_density(dn_prev[0])
                    dx_dn = self._centre_distance(ac, dn)
            else:
                key = ac.base_segments[-1][0] if ac.base_segments else None
                if key in self.downstream_density_map:
                    # Map values follow the cell convention: total across `lanes`.
                    rho_dn_lane = p.per_lane_density(
                        float(self.downstream_density_map[key])
                    )

            # eq. (3), evaluated wholly at time t.
            relaxation = (dt / p.tau) * (p.equilibrium_speed(rho) - v)
            convection = (dt * v / dx_up) * (v_up - v)
            anticipation = (
                (p.eta * dt) / (p.tau * dx_dn)
                * (rho_dn_lane - rho_lane) / (rho_lane + p.kappa)
            )
            v_new = v + relaxation + convection - anticipation

            if self.clamp_to_v_free:
                v_new = min(v_new, p.v_free)
            ac.velocity = max(self.v_min, v_new)

            # Ramp source/sink of eq. (1). The mainline flux carried across the
            # interface is the plain q of eq. (2), so the off-ramp share
            # q*beta/(1-beta) is removed here rather than folded into edge_flow.
            if p.r != 0.0:
                ac.mass += dt * p.r
            if p.beta != 0.0:
                ac.mass -= dt * (rho * v) * (p.beta / (1.0 - p.beta))
            if ac.mass < 0.0:
                ac.mass = 0.0

    @staticmethod
    def _neighbour(cells: Dict[str, "ActiveCell"], ids: List[str]) -> Optional["ActiveCell"]:
        # The active mesh is a 1-D chain within a lane, so there is at most one.
        return cells[ids[0]] if len(ids) == 1 else None

    def stability_report(self, simulation: "Simulation") -> List[str]:
        """Return CFL / relaxation warnings for the current network. Empty if fine.

        METANET's convection term is explicit upwind, so it needs v*dt <= dx, and the
        relaxation term needs dt <= tau to avoid oscillating about V[rho].
        """
        warnings: List[str] = []
        dt = float(simulation.time_resolution)
        for road in simulation.network.roads.values():
            for cell in road.cells.values():
                p = self.per_cell_params.get((cell.road_id, cell.cell_id), self.params)
                if dt > p.tau:
                    warnings.append(
                        f"{cell.road_id}/{cell.cell_id}: dt={dt:g}s exceeds tau={p.tau:g}s "
                        f"— relaxation term will oscillate."
                    )
                cfl = p.v_free * dt / cell.length
                if cfl > 1.0:
                    warnings.append(
                        f"{cell.road_id}/{cell.cell_id}: CFL={cfl:.2f} > 1 "
                        f"(v_free={p.v_free:g} m/s, dt={dt:g}s, dx={cell.length:g}m)."
                    )
        return warnings


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
        macro_model: Optional["MacroModel"] = None,
    ):
        self.network = network
        self.active = None
        # None => first-order Godunov/CTM on the cells' FundamentalDiagrams (the
        # historical behaviour). Set to a MacroModel (e.g. METANETModel) to swap the
        # flux rule and carry extra per-cell state.
        self.macro_model = macro_model
        self.time_resolution = float(time_resolution)
        self.origin_time = float(origin_time)
        self.current_time = float(origin_time)
        self.rollout_results: List[RolloutStep] = []
        # When False, step() skips recording per-step rollout snapshots (base
        # network clone + active/mask snapshots). Set this for headless runs that
        # don't feed the viewer — it removes a full 5000-cell deep-copy per step.
        self.record_rollout = True

        self.inflow_boundary_map = inflow_boundary_map or {}
        self.outflow_boundary_map = outflow_boundary_map or {}

        self.min_cell_length = float(min_cell_length) if min_cell_length is not None else 1.0

        self.masking_cells: Dict[str, ArbitraryMaskingCell] = {}
        self._step_callbacks: Dict[str, Any] = {}
        self._prestep_callbacks: Dict[str, Any] = {}
        self._poststep_callbacks: Dict[str, Any] = {}

        self.network.validate()

        self.gt_store = None
        # Whether `gt_store` also *drives* the open ends, overwriting boundary cell
        # mass at the end of every step (a Dirichlet density BC). That is the
        # first-order path's boundary treatment, so it defaults on.
        #
        # Second-order models supply their own boundary conditions instead — an
        # upstream demand and a downstream density, as fluxes and ghost states
        # through inflow_boundary_map / outflow_boundary_map and the model's own
        # upstream_velocity_map / downstream_density_map. Pinning the end cells'
        # state on top of that fights them: the pinned cell is no longer simulated,
        # and its q = rho*v is injected with no capacity limit. Those runs set this
        # False. It is deliberately separate from `gt_store` itself, which stays
        # attached either way because RolloutRenderer reads it for the empirical
        # time-space heatmap.
        self.gt_drives_boundaries = True
        self._cached_active_network: Optional[ActiveNetwork] = None

    @staticmethod
    def from_json(
        json_path: str,
        time_resolution: float,
        origin_time: float,
        inflow_boundary_map: Optional[Dict[Connection, float]] = None,
        outflow_boundary_map: Optional[Dict[Connection, float]] = None,
        min_cell_length: Optional[float] = None,
        macro_model: Optional["MacroModel"] = None,
    ) -> "Simulation":
        network = Network.from_json(json_path)
        return Simulation(
            network=network,
            time_resolution=time_resolution,
            origin_time=origin_time,
            inflow_boundary_map=inflow_boundary_map,
            outflow_boundary_map=outflow_boundary_map,
            min_cell_length=min_cell_length,
            macro_model=macro_model,
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

    def register_prestep_callback(self, fn, key) -> None:
        """Register a callable invoked at the start of each step as fn(sim_time, dt), before the step callbacks have been run."""
        self._prestep_callbacks[key] = fn

    def unregister_prestep_callback(self, key) -> None:
        if key in self._prestep_callbacks:
            del self._prestep_callbacks[key]

    def register_poststep_callback(self, fn, key) -> None:
        """Register a callable invoked at the start of each step as fn(sim_time, dt), before the step callbacks have been run."""
        self._poststep_callbacks[key] = fn

    def unregister_poststep_callback(self, key) -> None:
        if key in self._poststep_callbacks:
            del self._poststep_callbacks[key]

    def _snapshot(self) -> None:
        self.rollout_results.append(
            RolloutStep(sim_time=self.current_time, network=self.network.clone())
        )
    def initialize_from_ground_truth(
        self,
        gt_store: GroundTruthStore,
        time_value: Optional[float] = None,
        tolerance: float = 1e-6,
        drives_boundaries: bool = True,
    ) -> None:
        """Seed every cell from the ground truth and attach the store.

        `drives_boundaries` controls only what happens afterwards: with it set, the
        store overwrites boundary cell mass every step. Pass False when the run
        supplies its own boundary conditions (see `gt_drives_boundaries`); the store
        stays attached for the initial condition and for the renderer either way.
        """
        t = self.current_time if time_value is None else float(time_value)
        gt_store.apply_density_snapshot_to_network(self.network, t, tolerance=tolerance)
        self.gt_store = gt_store
        self.gt_drives_boundaries = bool(drives_boundaries)

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
        ConservativeRemapper.base_to_active(self, self.network, active)
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
                if self.macro_model is not None:
                    demand_map[aid] = self.macro_model.sending_flow(ac)
                    supply_map[aid] = self.macro_model.receiving_flow(ac)
                elif ac.fd is not None:
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
        mask_flow: Dict[Tuple[str, str], float] = {}

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
                    mask_flow[(u, v)] = -edge_flow[(u, v)]
                elif u_cell.kind == "mask":
                    # Mask → normal: front boundary flux determined by mask's Riemann solver
                    mask = self.masking_cells[u_cell.mask_id]
                    edge_flow[(u, v)] = mask.front_boundary_flux(
                        v_cell.density, v_cell.fd, self.current_time, self.time_resolution
                    )
                    mask_flow[(u, v)] = edge_flow[(u, v)]
                else:
                    # Normal → normal: standard Godunov supply/demand
                    preds = list(v_cell.inflow_neighbors)
                    per_in_supply = supply_map[v] if not preds else supply_map[v] / float(len(preds))
                    edge_flow[(u, v)] = min(per_out_demand, per_in_supply)

        # Index edges by their target and source cell once, so the capping loops
        # below are O(edges) instead of rescanning all of edge_flow per cell.
        incoming_by_v: Dict[str, List[Tuple[str, str]]] = {}
        outgoing_by_u: Dict[str, List[Tuple[str, str]]] = {}
        for e in edge_flow:
            incoming_by_v.setdefault(e[1], []).append(e)
            outgoing_by_u.setdefault(e[0], []).append(e)

        # Cap inflow to normal cells.
        # Mask boundary fluxes are fixed (Riemann solver output) and are not scaled —
        # only normal-normal edges are subject to capping. The fixed mask flux is
        # subtracted from remaining supply before scaling normal inflow.
        for v in active_ids:
            v_cell = active.active_cells[v]
            if v_cell.kind == "mask":
                continue
            all_incoming = incoming_by_v.get(v, ())
            normal_incoming = [e for e in all_incoming if active.active_cells[e[0]].kind == "normal"]
            if not normal_incoming:
                continue
            mask_inflow = sum(edge_flow[e] for e in all_incoming if active.active_cells[e[0]].kind != "normal")
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
            all_outgoing = outgoing_by_u.get(u, ())
            normal_outgoing = [e for e in all_outgoing if active.active_cells[e[1]].kind == "normal"]
            if not normal_outgoing:
                continue
            mask_outflow = sum(edge_flow[e] for e in all_outgoing if active.active_cells[e[1]].kind != "normal")
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
            if self.macro_model is not None:
                cell_capacity = self.macro_model.boundary_capacity(ac)
            else:
                cell_capacity = ac.fd.capacity if ac.fd is not None else None
            if len(ac.inflow_neighbors) == 0:
                base_key = ac.base_segments[0][0]
                external_inflow[aid] = min(
                    float(self.inflow_boundary_map.get(base_key, 0.0)),
                    supply_map[aid],
                )
            if len(ac.outflow_neighbors) == 0:
                base_key = ac.base_segments[-1][0]
                # `cell_capacity` is None only when the model declares free outflow
                # (METANET); the first-order path always has an FD capacity here, so
                # this is equivalent to the previous min-against-capacity form.
                cap = self.outflow_boundary_map.get(base_key, cell_capacity)
                external_outflow[aid] = (
                    demand_map[aid] if cap is None
                    else min(demand_map[aid], float(cap))
                )

        return edge_flow, external_inflow, external_outflow, mask_flow

    def _step_active_network(self, active: ActiveNetwork) -> None:
        # Must run before the flux computation: it lazily seeds any missing extra
        # state and captures the time-t state the model's own update is evaluated at.
        if self.macro_model is not None:
            self.macro_model.prepare(self, active)

        active_ids = active.ordered_ids()
        index_of = {aid: i for i, aid in enumerate(active_ids)}

        masses = np.array([active.active_cells[aid].mass for aid in active_ids], dtype=np.float64)
        lengths = np.array([active.active_cells[aid].length for aid in active_ids], dtype=np.float64)

        dt = self.time_resolution

        edge_flow, external_inflow, external_outflow, mask_flow = self._compute_active_edge_flows(active)
        lateral_deltas = active.lateral_delta_density(dt)

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
        # Mass-delta update. Mask faces return moving-frame flux, fixed faces return
        # lab-frame flux; both feed `net_flow` directly. New density is recovered as
        # `ac.mass / ac.length` *after* `move_active_masks` shifts the moving faces.
        new_masses = masses + dt * net_flow

        for i, aid in enumerate(active_ids):
            ac = active.active_cells[aid]            
            #if ac.fd is None:
            #    raise ValueError(f"Active cell {aid} has no FundamentalDiagram assigned.")
            if ac.kind == "mask":
                # Mask active-cell mass is owned by the mask itself (set by base_to_active);
                # we just hand each mask the per-face flux it saw this step.
                ac.mass = float(new_masses[i])
                rear_flow_ac = float(rear_flow[i]) * dt
                front_flow_ac = float(front_flow[i]) * dt
                self.masking_cells[ac.mask_id].rear_flow += rear_flow_ac
                self.masking_cells[ac.mask_id].front_flow += front_flow_ac
            else:
                lateral_mass = lateral_deltas.get(aid, 0.0) * lengths[i]
                ac.mass = float(new_masses[i] + lateral_mass)

        if self.macro_model is not None:
            self.macro_model.advance(self, active)

    def step(self) -> None:
        if self.record_rollout:
            self._snapshot()
        callbacks = [cb for cb in self._prestep_callbacks]
        for cb in callbacks:
            if cb in self._prestep_callbacks:
                self._prestep_callbacks[cb](self.current_time, self.time_resolution)
        callbacks = [cb for cb in self._step_callbacks]
        for cb in callbacks:
            if cb in self._step_callbacks:
                self._step_callbacks[cb](self.current_time, self.time_resolution)
        self.active = self._build_active_network()
        if self.record_rollout:
            self.rollout_results[-1].active_network = self.active.snapshot()
            if self.masking_cells:
                self.rollout_results[-1].mask_snapshots = dict(self.masking_cells)
        self._step_active_network(self.active)
        self._update_masks()
        ConservativeRemapper.move_active_masks(self, self.active)
        """
        for ac in active.active_cells.values():
            if ac.kind == "mask" or ac.fd is None or ac.length <= 1e-12:
                continue
            rho = ac.mass / ac.length
            rho = max(0.0, min(rho, ac.fd.rho_j * 2.0))
            ac.mass = float(rho * ac.length)
        """
        ConservativeRemapper.active_to_base(self.network, self.active)
        self.current_time += self.time_resolution
        # Optional: a run with no ground-truth store, or one that drives its own open
        # ends (METANET), takes its boundary conditions from the inflow/outflow maps
        # instead of having the end cells' mass overwritten here.
        if self.gt_store is not None and self.gt_drives_boundaries:
            self.gt_store.apply_density_snapshot_to_network_boundaries(self.network, self.current_time)
        callbacks = [cb for cb in self._poststep_callbacks]
        for cb in callbacks:
            if cb in self._poststep_callbacks:
                self._poststep_callbacks[cb](self.current_time, self.time_resolution)
        #print(self.current_time)

    def initialize_callbacks(self):
        self.active = self._build_active_network()
        """
        callbacks = [cb for cb in self._step_callbacks]
        for cb in callbacks:
            self._step_callbacks[cb](self.current_time, self.time_resolution)
        self.active = self._build_active_network()
        """

    def run(self, duration: float, initialize=True) -> None:
        if duration < 0.0:
            raise ValueError("duration must be non-negative.")
        num_steps = int(np.round(duration / self.time_resolution))
        if initialize:
            self.initialize_callbacks()
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

        def _q_value(
            density: float,
            fd: Optional[FundamentalDiagram],
            q: str,
            velocity: Optional[float] = None,
        ) -> float:
            if q == "density":
                return density
            # A second-order model (METANET) solves for mean speed, so use it rather
            # than inverting the FD. `velocity is None` is the first-order path, and
            # also a mask cell in a hybrid run: both fall back to the FD.
            if velocity is not None:
                return float(velocity) if q == "velocity" else float(velocity) * density
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
                        vals.append(_q_value(cell.density_viz, cell.fd, q, cell.velocity))
                if rs.active_network:
                    for ac in rs.active_network.active_cells.values():
                        if ac.kind == "normal" and ac.fd is not None:
                            # ActiveCell has no density_viz: mask mass lives in `mass`
                            # on the active mesh, so density already includes it.
                            vals.append(_q_value(ac.density, ac.fd, q, ac.velocity))
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
        n_base = len(base_cells)

        # Trace layout (per quantity q_idx in 0,1,2):
        #   base traces:   q_idx * n_base  ..  (q_idx+1) * n_base - 1
        #   active traces: 3*n_base + q_idx * max_active  ..  3*n_base + (q_idx+1) * max_active - 1
        #   colorbar:      3*n_base + 3*max_active + q_idx
        def base_start(q_idx: int) -> int:
            return q_idx * n_base
        def active_start(q_idx: int) -> int:
            return 3 * n_base + q_idx * max_active
        def colorbar_idx(q_idx: int) -> int:
            return 3 * n_base + 3 * max_active + q_idx

        n_total = 3 * n_base + 3 * max_active + 3

        def _visibility(show_base: bool, q_idx: int) -> List[bool]:
            vis = [False] * n_total
            group_start = base_start(q_idx) if show_base else active_start(q_idx)
            group_size = n_base if show_base else max_active
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
                    fillcolor=_rgba(_q_value(cell.density_viz, fd, q, cell.velocity), vmin, vmax),
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
                    fill = MASK_FILL if ac.kind == "mask" else _rgba(_q_value(ac.density, ac.fd, q, ac.velocity), vmin, vmax)
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
        n_frames = len(self.rollout_results)

        def _color_matrix(vals: np.ndarray, vmin: float, vmax: float) -> List[List[str]]:
            """(n_frames, N_cells) values → (n_frames, N_cells) hex color strings via vectorized colormap."""
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

        # Base: density matrix (n_frames, n_base), then derive other quantities per cell
        base_rho = np.array([
            [rs.network.get_cell(rid, cid).density_viz for rid, cid, _px, _py, _fd in base_cells]
            for rs in self.rollout_results
        ])  # (n_frames, n_base)
        # Model mean speed per frame, None where the model does not carry one.
        base_vel: List[List[Optional[float]]] = [
            [rs.network.get_cell(rid, cid).velocity for rid, cid, _px, _py, _fd in base_cells]
            for rs in self.rollout_results
        ]

        base_frame_colors: Dict[str, List[List[str]]] = {}
        for q in QUANTITIES:
            vmin, vmax = ranges[q]
            if q == "density":
                vals = base_rho
            else:
                vals = np.zeros_like(base_rho)
                for j, (_rid, _cid, _px, _py, fd) in enumerate(base_cells):
                    vals[:, j] = [
                        _q_value(rho, fd, q, base_vel[f][j]) for f, rho in enumerate(base_rho[:, j])
                    ]
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
            step_velocity: List[Optional[float]] = []
            step_fd: List[Optional[FundamentalDiagram]] = []
            step_is_mask: List[bool] = []

            for i in range(max_active):
                if i < len(step_active):
                    ac = step_active[i]
                    road = self.network.roads[ac.road_id]
                    step_geo.append(_rotate(*Network._cell_polygon(road, ac.start_s, ac.end_s, ac.lane)))
                    step_lc.append(MASK_LINE if ac.kind == "mask" else NORMAL_LINE)
                    step_density.append(ac.density)
                    step_velocity.append(ac.velocity)
                    step_fd.append(ac.fd)
                    step_is_mask.append(ac.kind == "mask")
                else:
                    step_geo.append(([], []))
                    step_lc.append(NORMAL_LINE)
                    step_density.append(0.0)
                    step_velocity.append(None)
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
                        t = float(np.clip((_q_value(step_density[i], step_fd[i], q, step_velocity[i]) - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0))
                        r, g, b, _ = CMAP(t)
                        step_colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
                    else:
                        step_colors.append("#000000")
                active_frame_colors[q].append(step_colors)

        print("Active done!")

        # Precompute fixed frame_traces list (same for every frame)
        frame_traces_template: List[int] = []
        for q_idx in range(len(QUANTITIES)):
            for i in range(n_base):
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

    def rollout_dataframe(self) -> pd.DataFrame:
        rows = []
        for step in self.rollout_results:
            net = step.network
            for road_id, road in net.roads.items():
                for cell_id, cell in road.cells.items():
                    rho = float(cell.density)
                    if cell.velocity is not None:
                        # Second-order model (METANET): mean speed is state, and the
                        # flux it carries is q = rho*v, not the FD's demand.
                        v = float(cell.velocity)
                        q = v * rho
                    elif cell.fd is None:
                        raise ValueError(f"Cell {road_id}/{cell_id} has no FundamentalDiagram assigned.")
                    else:
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
                            "mass": cell.mass,
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

        def _q_value(
            density: float,
            fd: Optional[FundamentalDiagram],
            q: str,
            velocity: Optional[float] = None,
        ) -> float:
            if q == "density":
                return density
            # Second-order models (METANET) carry mean speed as state; prefer it over
            # inverting the FD. None means the first-order path, or a mask cell in a
            # hybrid run — both keep the FD behaviour.
            if velocity is not None:
                return float(velocity) if q == "velocity" else float(velocity) * density
            if fd is None:
                return 0.0
            if q == "velocity":
                return fd.velocity_from_density(density)
            if q == "flow":
                return fd.velocity_from_density(density) * density
            return density

        # --- Precomputed per-road arc-length tables (static geometry) ---------
        # Network._cell_polygon otherwise rebuilds the polyline arrays + arc
        # lengths on every one of ~(frames * cells) calls; build them once.
        self._road_geom: Dict[str, Any] = {}
        for road in sim.network.roads.values():
            lp = np.asarray(road.left_polyline, dtype=float)
            rp = np.asarray(road.right_polyline, dtype=float)
            seg = np.linalg.norm(np.diff(lp, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            self._road_geom[road.road_id] = (lp, rp, seg, cum, road.lane_data)

        def _polygon(road_id: str, s0: float, s1: float, lane: int) -> Tuple[List[float], List[float]]:
            lp, rp, seg, cum, lane_data = self._road_geom[road_id]
            info = lane_data[lane]
            lat = float(info["lateral_position"])
            width = float(info["width"])
            nseg = len(seg)

            def _pt(s: float):
                idx = int(np.clip(np.searchsorted(cum, s) - 1, 0, nseg - 1))
                t = (s - cum[idx]) / max(seg[idx], 1e-8)
                l = lp[idx] + t * (lp[idx + 1] - lp[idx])
                r = rp[idx] + t * (rp[idx + 1] - rp[idx])
                d = (r - l) / max(float(np.linalg.norm(r - l)), 1e-8)
                return l + (-lat) * d, l + (-(lat - width)) * d

            p1, p2 = _pt(s0)
            p3, p4 = _pt(s1)
            return _rotate(
                [p1[0], p2[0], p4[0], p3[0], p1[0]],
                [p1[1], p2[1], p4[1], p3[1], p1[1]],
            )
        self._polygon = _polygon

        # --- Vectorised quantity transform (matches scalar _q_value exactly) ---
        def _q_transform(
            rho: np.ndarray,
            fd: Optional[FundamentalDiagram],
            q: str,
            velocity: Optional[np.ndarray] = None,
        ) -> np.ndarray:
            """Vectorised _q_value. `velocity` is the model's own speed, NaN where absent."""
            rho = np.asarray(rho, dtype=float)
            if q == "density":
                return rho
            if fd is None:
                vel = np.zeros_like(rho)
            else:
                guard = np.where(rho > 1e-12, rho, 1.0)
                if isinstance(fd, TriangularFD):
                    flow_raw = np.where(rho > fd.rho_c, fd.w * (fd.rho_j - rho), fd.v_f * rho)
                elif isinstance(fd, GreenshieldsFD):
                    rc = np.clip(rho, 0.0, fd.rho_j)
                    flow_raw = fd.v_f * rc * (1.0 - rc / fd.rho_j)
                else:  # generic fallback (rare)
                    flow_raw = np.vectorize(lambda x: fd._flow(x))(rho)
                vel = np.where(rho > 1e-12, flow_raw / guard, fd.v_f)
            if velocity is not None:
                # Per cell and per frame: the model's speed where it has one, the FD's
                # otherwise. All-NaN (the first-order path) leaves `vel` untouched.
                velocity = np.asarray(velocity, dtype=float)
                vel = np.where(np.isfinite(velocity), velocity, vel)
            if q == "velocity":
                return vel
            return vel * rho  # flow == mean speed * density
        self._q_transform = _q_transform
        self._q_value_scalar = _q_value

        def _q_matrix(
            density: np.ndarray,
            fds: List[Any],
            q: str,
            velocity: Optional[np.ndarray] = None,
        ) -> np.ndarray:
            """Per-FD-group _q_transform. `velocity` must match `density`'s shape."""
            density = np.asarray(density, dtype=float)
            if q == "density" or density.size == 0:
                return density
            out = np.zeros_like(density)
            groups: Dict[int, Tuple[Any, List[int]]] = {}
            for j, fd in enumerate(fds):
                groups.setdefault(id(fd), (fd, []))[1].append(j)
            for _fdid, (fd, cols) in groups.items():
                vel_cols = None if velocity is None else np.asarray(velocity, dtype=float)[:, cols]
                out[:, cols] = _q_transform(density[:, cols], fd, q, vel_cols)
            return out

        def _color_matrix(vals: np.ndarray, vmin: float, vmax: float) -> List[List[str]]:
            t = np.clip((vals - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
            rgb = (CMAP(t)[..., :3] * 255).astype(np.uint8)
            packed = (rgb[..., 0].astype(np.uint32) << 16
                      | rgb[..., 1].astype(np.uint32) << 8
                      | rgb[..., 2].astype(np.uint32))
            nf, nc = vals.shape
            return [[f"#{packed[i, j]:06x}" for j in range(nc)] for i in range(nf)]

        # --- Base cell geometry (fixed for all frames) ------------------------
        ref_network = sim.rollout_results[0].network
        self.base_cells: List[Tuple[str, str, List[float], List[float], Optional[FundamentalDiagram]]] = []
        base_keys: List[Tuple[str, str]] = []
        base_fds: List[Any] = []
        for road in ref_network.roads.values():
            for cell in road.cells.values():
                px, py = self._polygon(road.road_id, cell.start_s, cell.end_s, cell.lane)
                self.base_cells.append((road.road_id, cell.cell_id, px, py, cell.fd))
                base_keys.append((road.road_id, cell.cell_id))
                base_fds.append(cell.fd)

        active_per_step: List[List[ActiveCell]] = [
            list(rs.active_network.active_cells.values()) if rs.active_network else []
            for rs in sim.rollout_results
        ]
        self._active_per_step = active_per_step
        self.max_active: int = max((len(a) for a in active_per_step), default=0)
        self.N_frames: int = len(sim.rollout_results)
        min_sim_time = min([rs.sim_time for rs in sim.rollout_results])
        self.sim_times: List[float] = [rs.sim_time - min_sim_time for rs in sim.rollout_results]
        self.mask_snapshots_per_step: List[Dict[str, Any]] = [
            rs.mask_snapshots or {} for rs in sim.rollout_results
        ]

        # --- Base density matrices -> ranges + colors + time-space (one pass) -
        # Gather raw mass + mask_mass once (dataclass fields, cheaper than the
        # .density/.density_viz properties) and derive both matrices vectorised.
        # The time-space diagram then slices columns out of _base_density_viz
        # instead of re-reading every frame snapshot.
        self._base_col: Dict[Tuple[str, str], int] = {key: j for j, key in enumerate(base_keys)}
        if base_keys:
            base_lengths = np.array([
                ref_network.roads[rid].cells[cid].length for (rid, cid) in base_keys
            ])
            base_lengths = np.where(base_lengths > 0.0, base_lengths, 1.0)
            mass = np.empty((self.N_frames, len(base_keys)))
            mask_mass = np.empty((self.N_frames, len(base_keys)))
            # Model mean speed, NaN where the cell carries none (the first-order path,
            # and mask cells in a hybrid run). Gathered here so every velocity/flow
            # consumer can prefer it over inverting the FD.
            base_velocity = np.full((self.N_frames, len(base_keys)), np.nan)
            for f, rs in enumerate(sim.rollout_results):
                cells_by_road = {rid: road.cells for rid, road in rs.network.roads.items()}
                mass[f] = [cells_by_road[rid][cid].mass for (rid, cid) in base_keys]
                mask_mass[f] = [cells_by_road[rid][cid].mask_mass for (rid, cid) in base_keys]
                base_velocity[f] = [
                    np.nan if cells_by_road[rid][cid].velocity is None
                    else cells_by_road[rid][cid].velocity
                    for (rid, cid) in base_keys
                ]
            base_rho = mass / base_lengths
            self._base_density_viz = (mass + mask_mass) / base_lengths
        else:
            base_rho = np.zeros((self.N_frames, 0))
            base_velocity = np.zeros((self.N_frames, 0))
            self._base_density_viz = np.zeros((self.N_frames, 0))
        self._base_velocity = base_velocity

        # Global per-quantity value ranges for a consistent colorbar. Computed
        # from the base mesh; normal active cells are conservatively remapped
        # from it, so their value range coincides.
        self.ranges: Dict[str, Tuple[float, float]] = {}
        for q in self.QUANTITIES:
            vals = _q_matrix(base_rho, base_fds, q, base_velocity)
            self.ranges[q] = (float(vals.min()) if vals.size else 0.0,
                              float(vals.max()) if vals.size else 1.0)

        # Base fill-colors are built lazily per frame in get_figure (see
        # _base_frame_colors) so launch does not pay 3 * N_frames * N_cells hex
        # conversions up front.
        self._base_rho = base_rho
        self._base_fds = base_fds
        self._q_matrix = _q_matrix
        self._color_matrix = _color_matrix
        self._base_color_cache: Dict[Tuple[int, str], List[str]] = {}

        # Active-view geometry and colors are built lazily per frame in
        # get_figure (see _active_frame) and cached. The active view is off by
        # default, so this per-frame work is usually never triggered at all.
        self._active_geo_cache: Dict[int, Any] = {}
        self._active_color_cache: Dict[Tuple[int, str], List[str]] = {}

        # Time-space data: (road_id, lane, version) → per-quantity arrays.
        # Built lazily per lane in get_ts_figure (only the viewed lane is paid
        # for) — see _ensure_ts_lane / _macro_pivots.
        self.ts_data: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
        self._ref_net = sim.rollout_results[0].network
        self._rollout_results = sim.rollout_results
        self._gt_store = sim.gt_store
        self._macro_pivot_cache: Optional[Tuple[np.ndarray, Any, Any]] = None
        # (road_id, lane) pairs that have cells — for populating UI selectors
        # without forcing the (lazy) time-space data to be built.
        self.ts_road_lanes: List[Tuple[str, int]] = [
            (road.road_id, lane)
            for road in self._ref_net.roads.values()
            for lane in sorted(road.lane_data)
            if road.cells_for_lane(lane)
        ]

    def _has_macro_gt(self) -> bool:
        """True when there is empirical macro data to compare against.

        Pure-macro runs (e.g. METANET validation) often have no ground-truth store at
        all, and a store built from micro data alone has no macro_df attribute.
        """
        return getattr(self._gt_store, "macro_df", None) is not None

    def _macro_pivots(self):
        """Lazily pivot the empirical macro data into (times, density, velocity)
        DataFrames indexed by time with (road_id, cell_id) columns. Cached."""
        if self._macro_pivot_cache is None:
            df = self._gt_store.macro_df.copy()
            df["road_id"] = df["road_id"].astype(str)
            df["cell_id"] = df["cell_id"].astype(str)
            # (time, road_id, cell_id) is a unique complete grid, so plain pivot
            # (no aggregation) is correct and faster than pivot_table.
            piv = df.pivot(index="time", columns=["road_id", "cell_id"], values=["density", "velocity"]).sort_index()
            dp = piv["density"]
            vp = piv["velocity"]
            times = dp.index.to_numpy().astype(float)
            self._macro_pivot_cache = (times, dp, vp)
        return self._macro_pivot_cache

    def _ensure_ts_lane(self, road_id: str, lane: int) -> None:
        """Build (and cache) the sim + empirical time-space entries for one lane."""
        if (road_id, lane, "sim") in self.ts_data:
            return
        if road_id not in self._ref_net.roads:
            return
        road = self._ref_net.roads[road_id]
        if lane not in road.lane_data:
            return
        cells = road.cells_for_lane(lane)
        if not cells:
            return
        cell_ids = [c.cell_id for c in cells]
        s_mids = [(c.start_s + c.end_s) / 2.0 for c in cells]
        fds = [c.fd for c in cells]

        # density_viz for this lane is a column slice of the matrix gathered
        # once at construction (no per-frame snapshot re-reads).
        cols = [self._base_col[(road_id, cid)] for cid in cell_ids]
        rho_ts = self._base_density_viz[:, cols]  # (N_frames, N_cells)
        vel_ts = self._base_velocity[:, cols]     # NaN where there is no model speed
        sim_entry: Dict[str, Any] = {"s_mids": s_mids, "density": rho_ts}
        for q in ("flow", "velocity"):
            sim_entry[q] = self._q_matrix(rho_ts, fds, q, vel_ts)

        # Mask extents per timestep: (start_s, end_s) or None
        mask_extents: List[List[Optional[Tuple[float, float]]]] = [[]]
        for snapshots in self.mask_snapshots_per_step:
            extent: Optional[Tuple[float, float]] = None
            for mask in snapshots.values():
                for seg in mask.segments:
                    if seg.road_id == road_id and seg.lane == lane:
                        lo, hi = float(seg.start_s), float(seg.end_s)
                        extent = (lo, hi) if extent is None else (min(extent[0], lo), max(extent[1], hi))
            if (extent is None) and (len(mask_extents[-1]) > 0) and (mask_extents[-1][-1] is not None):
                mask_extents.append([])
            elif (extent is not None) and (len(mask_extents[-1]) > 0) and (mask_extents[-1][-1] is None):
                mask_extents.append([])
            mask_extents[-1].append(extent)
        sim_entry["mask_extents"] = mask_extents
        self.ts_data[(road_id, lane, "sim")] = sim_entry

        # Empirical (ground-truth macro) heatmap, vectorised via pivot. Skipped when
        # there is no macro ground truth: the sim heatmap must still work for a
        # pure-macro run with no store, and get_ts_figure returns an empty figure for
        # the "empirical" version that is then absent.
        if not self._has_macro_gt():
            return
        _times, dp, vp = self._macro_pivots()
        cols = [(road_id, cid) for cid in cell_ids]
        vals_density = dp.reindex(columns=cols).to_numpy()
        vals_velocity = vp.reindex(columns=cols).to_numpy()
        self.ts_data[(road_id, lane, "empirical")] = {
            "s_mids": s_mids,
            "density": vals_density,
            "velocity": vals_velocity,
            "flow": vals_density * vals_velocity,
            "mask_extents": [[]],
        }

    def get_ts_figure(self, road_id: str, lane: int, quantity: str = "density", version: str = "sim", render_masks: bool = True) -> go.Figure:
        """Return a time-space heatmap (viridis pcolormesh style) for one road+lane."""
        self._ensure_ts_lane(road_id, lane)
        key = (road_id, lane, version)
        if key not in self.ts_data:
            return go.Figure()
        entry = self.ts_data[key]
        #vmin, vmax = self.ranges[quantity]
        if (quantity == "density"):
            vmin, vmax = 0.0, 0.20
        elif (quantity == "velocity"):
            vmin, vmax = 0.0, 60.0
        elif (quantity == "flow"):
            vmin, vmax = 0.0, 12.0
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

    def _white_template(self) -> dict:
        """Resolve the named 'plotly_white' template to a plain dict, once.

        plotly.js does not know Python-side named templates, so a dict figure
        must embed the fully-resolved template dict rather than the string.
        """
        tmpl = getattr(self, "_tmpl_white_cache", None)
        if tmpl is None:
            import plotly.io as pio
            tmpl = pio.templates["plotly_white"].to_plotly_json()
            self._tmpl_white_cache = tmpl
        return tmpl

    def _base_frame_colors(self, step_idx: int, q: str) -> List[str]:
        """Lazily build one frame's base fill-colors for quantity q. Cached."""
        key = (step_idx, q)
        cached = self._base_color_cache.get(key)
        if cached is not None:
            return cached
        row = self._base_rho[step_idx:step_idx + 1]
        vel = self._base_velocity[step_idx:step_idx + 1]
        vmin, vmax = self.ranges[q]
        colors = self._color_matrix(self._q_matrix(row, self._base_fds, q, vel), vmin, vmax)[0]
        self._base_color_cache[key] = colors
        return colors

    def _active_frame_geo(self, step_idx: int):
        """Lazily build (geometry, line-colors, density, velocity, fd, is_mask) for one
        frame's active cells, padded to max_active. Cached per frame."""
        cached = self._active_geo_cache.get(step_idx)
        if cached is not None:
            return cached
        step_active = self._active_per_step[step_idx]
        geo: List[Tuple[List, List]] = []
        lc: List[str] = []
        dens: List[float] = []
        vels: List[Optional[float]] = []
        fds: List[Optional[FundamentalDiagram]] = []
        is_mask: List[bool] = []
        for i in range(self.max_active):
            if i < len(step_active):
                ac = step_active[i]
                geo.append(self._polygon(ac.road_id, ac.start_s, ac.end_s, ac.lane))
                lc.append(self.MASK_LINE if ac.kind == "mask" else self.NORMAL_LINE)
                dens.append(ac.density)
                vels.append(ac.velocity)
                fds.append(ac.fd)
                is_mask.append(ac.kind == "mask")
            else:
                geo.append(([], []))
                lc.append(self.NORMAL_LINE)
                dens.append(0.0)
                vels.append(None)
                fds.append(None)
                is_mask.append(False)
        cached = (geo, lc, dens, vels, fds, is_mask)
        self._active_geo_cache[step_idx] = cached
        return cached

    def _active_frame_colors(self, step_idx: int, q: str) -> List[str]:
        """Lazily build one frame's active fill-colors for quantity q. Cached."""
        key = (step_idx, q)
        cached = self._active_color_cache.get(key)
        if cached is not None:
            return cached
        _geo, _lc, dens, vels, fds, is_mask = self._active_frame_geo(step_idx)
        vmin, vmax = self.ranges[q]
        colors: List[str] = []
        for i in range(self.max_active):
            if is_mask[i]:
                colors.append(self.MASK_FILL)
            elif fds[i] is not None:
                t = float(np.clip((self._q_value_scalar(dens[i], fds[i], q, vels[i]) - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0))
                r, g, b, _ = self._cmap(t)
                colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
            else:
                colors.append("#000000")
        self._active_color_cache[key] = colors
        return colors

    def get_figure(self, step_idx: int, show_base: bool = True, quantity: str = "density") -> dict:
        """Return a static figure (plain dict) for a single simulation timestep.

        Returns a plain dict rather than a go.Figure on purpose: under repeated
        Dash playback callbacks, constructing graph objects (go.Figure/go.Scatter)
        retains ~one figure's worth of BasePlotlyType objects per call and leaks
        memory (see leakrun diagnosis). Dash accepts dict figures directly.
        """
        q = quantity
        vmin, vmax = self.ranges[q]
        traces: List[Any] = []

        if show_base:
            colors = self._base_frame_colors(step_idx, q)
            for (_rid, _cid, px, py, _fd), color in zip(self.base_cells, colors):
                traces.append({
                    "type": "scatter",
                    "x": px, "y": py,
                    "fill": "toself",
                    "fillcolor": color,
                    "mode": "lines",
                    "line": {"color": self.NORMAL_LINE, "width": 0.5},
                    "showlegend": False,
                    "hoverinfo": "skip",
                })
        else:
            geo, lcs, _dens, _vels, _fds, _is_mask = self._active_frame_geo(step_idx)
            colors = self._active_frame_colors(step_idx, q)
            for i in range(self.max_active):
                px, py = geo[i]
                traces.append({
                    "type": "scatter",
                    "x": px, "y": py,
                    "fill": "toself",
                    "fillcolor": colors[i],
                    "mode": "lines",
                    "line": {"color": lcs[i], "width": 0.5},
                    "showlegend": False,
                    "hoverinfo": "skip",
                })
            for mask in self.mask_snapshots_per_step[step_idx].values():
                traces.extend(mask.render(self._rotate))

        traces.append({
            "type": "scatter",
            "x": [None], "y": [None],
            "mode": "markers",
            "marker": {
                "colorscale": "Viridis",
                "cmin": vmin, "cmax": vmax,
                "color": [vmin],
                "showscale": True,
                "colorbar": {"title": q, "x": 1.02, "thickness": 15},
            },
            "showlegend": False,
            "hoverinfo": "skip",
        })

        return {
            "data": traces,
            "layout": {
                "title": f"t = {int(self.sim_times[step_idx])} s",
                "xaxis": {"scaleanchor": "y", "showgrid": False},
                "yaxis": {"showgrid": False},
                "template": self._white_template(),
                "margin": {"t": 60},
                "uirevision": "constant",
            },
        }


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
        front_flux_memory: float,
        rear_flux_total: float = 0.0,
        front_flux_total: float = 0.0
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
        # Pure cumulative integral of the coupling flux. Unlike *_flux_memory, which is a
        # running balance that spawn/despawn also debit and credit, these are only ever
        # written by the boundary flux functions -- so they are the true continuum mass
        # exchanged, and can be reconciled against the discrete vehicle count.
        self.rear_flux_total = rear_flux_total
        self.front_flux_total = front_flux_total
        self.current_rear_flux = None
        self.current_front_flux = None

    @property
    def mass(self):
        return float(len(self.vehicles))

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

    def get_rear_vehicles(self, region):
        vehicles = []
        for new_vehicle_key in self.vehicles:
            if (self.vehicles[new_vehicle_key].s <= region):
                vehicles.append(self.vehicles[new_vehicle_key])
        return vehicles
    
    def get_front_vehicles(self, region):
        vehicles = []
        window_length = (self.margin_s * 2)
        for new_vehicle_key in self.vehicles:
            if (self.vehicles[new_vehicle_key].s >= (window_length - region)):
                vehicles.append(self.vehicles[new_vehicle_key])
        return vehicles

    @staticmethod
    def _boundary_density(vehicle, fd_exterior: "FundamentalDiagram", vehicle_velocity: float):
        """Interior density seen by a boundary Riemann solve.

        A boundary vehicle may carry an explicit `rho` (prescribed boundary
        conditions, as in PrescribedSimBridge). Prefer it: inferring density from
        speed is impossible on the free-flow branch of a triangular FD, where
        density_from_velocity collapses every v in [.., v_f] onto rho_c and the
        boundary therefore always demands capacity.
        """
        prescribed = getattr(vehicle, "rho", None)
        if prescribed is not None:
            return float(prescribed)
        return fd_exterior.density_from_velocity(vehicle_velocity)

    def get_rear_density(self, fd_exterior: "FundamentalDiagram"):
        leaving_region = 1 / fd_exterior.rho_c
        rear_vehicle = self.get_rear_vehicle()
        if rear_vehicle is not None:
            interior_s = rear_vehicle.s
            vehicle_leaving = (interior_s < leaving_region)
            vehicle_velocity = rear_vehicle.s_dt

            if vehicle_leaving:
                rho_interior = self._boundary_density(rear_vehicle, fd_exterior, vehicle_velocity)
            else:
                rho_interior = 0.0
        else:
            interior_s = 2 * self.margin_s
            vehicle_leaving = False
            rho_interior = 0.0
            vehicle_velocity = 0
        return rho_interior 

    def rear_boundary_flux(self, rho_exterior: float, fd_exterior: "FundamentalDiagram", sim_time: float, dt: float) -> float:
        rho_interior = self.get_rear_density(fd_exterior)

        demand = None
        supply = None
        if isinstance(fd_exterior, TriangularFD):
            g_max = (fd_exterior.v_f - self.anchor_speed) * fd_exterior.rho_c
            if rho_exterior > fd_exterior.rho_c:
                demand = g_max
            else:
                demand = (fd_exterior.v_f - self.anchor_speed) * rho_exterior

            if rho_interior > fd_exterior.rho_c:
                supply = (fd_exterior.w * fd_exterior.rho_j) - ((fd_exterior.w + self.anchor_speed) * rho_interior)
            else:
                supply = g_max
            
            """
            p_star = None
            #Rarefaction
            if (rho_exterior > rho_interior):
                if (rho_exterior <= fd_exterior.rho_c):
                    p_star = rho_exterior
                elif (rho_interior > fd_exterior.rho_c):
                    p_star = rho_interior
                else:
                    p_star = fd_exterior.rho_c
            # Shock
            else:
                s = fd_exterior.shock_speed(rho_exterior, rho_interior)
                if (s > self.anchor_speed):
                    p_star = rho_exterior
                else:
                    p_star = rho_interior
            """

        net_flux = min(demand, supply)
        #net_flux = (fd_exterior._flow(p_star) - (self.anchor_speed * p_star))

        self.rear_flux_memory += (net_flux * dt)
        self.rear_flux_total += (net_flux * dt)
        self.current_rear_flux = net_flux
        return net_flux
        #return 0.0

    def get_front_density(self, fd_exterior: "FundamentalDiagram"):
        front_vehicle = self.get_front_vehicle()
        leaving_region = 1 / fd_exterior.rho_c
        window_length = (self.margin_s * 2)
        if front_vehicle is not None:
            interior_s = window_length - front_vehicle.s
            vehicle_leaving = (interior_s < leaving_region)
            vehicle_velocity = front_vehicle.s_dt
            if vehicle_leaving:
                rho_interior = self._boundary_density(front_vehicle, fd_exterior, vehicle_velocity)
            else:
                rho_interior = 1.0 / interior_s
        else:
            interior_s = window_length
            rho_interior = 0.0
            vehicle_velocity = 0.0
            vehicle_leaving = False

        return rho_interior

    def front_boundary_flux(self, rho_exterior: float, fd_exterior: "FundamentalDiagram", sim_time: float, dt: float) -> float:
        rho_interior = self.get_front_density(fd_exterior)
    
        demand = None
        supply = None
        if isinstance(fd_exterior, TriangularFD):
            g_max = (fd_exterior.v_f - self.anchor_speed) * fd_exterior.rho_c
            if rho_interior > fd_exterior.rho_c:
                demand = g_max
            else:
                demand = (fd_exterior.v_f - self.anchor_speed) * rho_interior

            if rho_exterior > fd_exterior.rho_c:
                supply = (fd_exterior.w * fd_exterior.rho_j) - ((fd_exterior.w + self.anchor_speed) * rho_exterior)
            else:
                supply = g_max
            """
            p_star = None
            #Rarefaction
            if (rho_interior > rho_exterior):
                if (rho_interior <= fd_exterior.rho_c):
                    p_star = rho_interior
                elif (rho_exterior > fd_exterior.rho_c):
                    p_star = rho_exterior
                else:
                    p_star = fd_exterior.rho_c
            # Shock
            else:
                s = fd_exterior.shock_speed(rho_interior, rho_exterior)
                if (s > self.anchor_speed):
                    p_star = rho_interior
                else:
                    p_star = rho_exterior
            """

        net_flux = min(demand, supply)
        #net_flux = (fd_exterior._flow(p_star) - (self.anchor_speed * p_star))
        self.front_flux_memory += (net_flux * dt)
        self.front_flux_total += (net_flux * dt)
        self.current_front_flux = net_flux
        return net_flux
        #return 0.0

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

    def update(self, sim_time: float, dt: float) -> float:
        return self.anchor_speed

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