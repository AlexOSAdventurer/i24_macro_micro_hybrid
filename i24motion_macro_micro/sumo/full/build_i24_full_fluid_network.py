"""Build the full-corridor fluid network (simulation.Network JSON) from the SUMO net.

Part of the full-corridor pipeline (``sumo/full``). The 1-mile pipeline and its
``simulation_dataset/`` are untouched; output goes to ``simulation_dataset_full/``.

Inputs: ``i24_full_corridor.net.xml`` + ``.meta.json`` (build_i24_full_network.py).

Layout
------
* One mainline Road per direction, ids as in the 1-mile network: "1" eastbound
  (MM increasing), "2" westbound. Its s runs along the median (left) edge of the
  carriageway, through junction-internal lanes, from the upstream cut.
* One Road per ramp, id ``<dir>_<on|off>_<first SUMO edge>``, s from its far
  end (on-ramp) or from I-24 (off-ramp).
* Lane ids follow the project convention: -1 is the leftmost lane, -k is SUMO
  lane index n-k on an n-lane edge. Lanes are median-aligned in the SUMO net,
  so lanes -1..-4 run the whole mainline and auxiliary lanes appear as -5 only
  where SUMO has them. ``lane_data`` holds the widest cross-section.
* Uniform cells per road (``--cell-length``, default 50 m; the road length is
  divided into a whole number of equal cells). SUMO junctions are snapped to
  the nearest cell boundary, so every lane add/drop and ramp junction falls on
  a boundary shared by all lanes.
* Cell connections are copied from the SUMO lane connections: consecutive cells
  within a lane, lane-to-lane across adds/drops, and road-to-road at ramps.
  Where snapping leaves a SUMO edge with no cells, the connections across it
  are composed.
* Sources and sinks are structural, as in simulation.py: a cell with no inflow
  connections takes external inflow, a cell with no outflow connections lets
  mass leave. The generator asserts that such cells are exactly the mainline
  ends and the far ends of the ramps, and that every cell lies on a
  source -> sink path.
* Every cell, ramps included, carries the canonical triangular FD and
  lane-change rate from ``fd_params.py``, the same calibration the 1-mile
  network files, couplers and RL use. Calibrating ramps separately is a
  macro-model question.

A sidecar ``network_meta.json`` records, per road, its SUMO edges with their s
ranges (the s <-> (edge, pos) map a SUMO coupler needs), the junction cell
boundaries, ramp attachments, sources/sinks, and per cell the SUMO edge/lane
and the distance to the end of its lane.

Usage::

    python3 build_i24_full_fluid_network.py [--cell-length 50]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, PROJECT)
from fd_params import CANONICAL_FD_PARAMS, CANONICAL_LAMBDA_LC  # noqa: E402

DEFAULT_NET = os.path.join(HERE, "i24_full_corridor.net.xml")
DEFAULT_META = os.path.join(HERE, "i24_full_corridor.meta.json")
DEFAULT_OUT_DIR = os.path.join(PROJECT, "simulation_dataset_full")
MAINLINE_ROAD_ID = {"eb": "1", "wb": "2"}

Pt = Tuple[float, float]


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def parse_shape(s: str) -> List[Pt]:
    return [tuple(float(v) for v in p.split(",")[:2]) for p in s.split()]


def offset_polyline(pts: List[Pt], d: float) -> List[Pt]:
    """Offset by d metres to the LEFT of travel (negative = right), using the
    averaged normal of the adjacent segments at each vertex."""
    out = []
    for i, p in enumerate(pts):
        nx = ny = 0.0
        for j in (i - 1, i):
            if 0 <= j < len(pts) - 1:
                a, b = pts[j], pts[j + 1]
                L = math.dist(a, b)
                if L > 0:
                    nx += -(b[1] - a[1]) / L
                    ny += (b[0] - a[0]) / L
        nl = math.hypot(nx, ny)
        if nl == 0:
            out.append(p)
        else:
            out.append((p[0] + d * nx / nl, p[1] + d * ny / nl))
    return out


def arclen(pts: List[Pt]) -> float:
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


def dedupe(pts: List[Pt], tol: float = 1e-3) -> List[Pt]:
    out = [pts[0]]
    for p in pts[1:]:
        if math.dist(p, out[-1]) > tol:
            out.append(p)
    return out


def clean_polyline(pts: List[Pt], min_seg: float = 0.5) -> List[Pt]:
    """Drop points closer than min_seg to the last kept point and points that
    double back (a lane shape and a junction-internal lane overlap by a few cm
    at every junction). A doubled-back vertex has near-opposite neighbouring
    normals, so any offset of it points in an arbitrary direction."""
    out = [pts[0]]
    for p in pts[1:]:
        if math.dist(p, out[-1]) < min_seg:
            continue
        if len(out) >= 2:
            ax, ay = out[-1][0] - out[-2][0], out[-1][1] - out[-2][1]
            bx, by = p[0] - out[-1][0], p[1] - out[-1][1]
            if ax * bx + ay * by <= 0:
                continue
        out.append(p)
    if math.dist(out[-1], pts[-1]) > 1e-6 and math.dist(out[-1], pts[-1]) < min_seg and len(out) > 1:
        out[-1] = pts[-1]
    return out


def project_arclen(pts: List[Pt], q: Pt) -> float:
    best, acc = None, 0.0
    for a, b in zip(pts, pts[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / L2))
        d = math.dist(q, (a[0] + t * dx, a[1] + t * dy))
        if best is None or d < best[0]:
            best = (d, acc + t * math.sqrt(L2))
        acc += math.sqrt(L2)
    return best[1]


# --------------------------------------------------------------------------- #
# SUMO net
# --------------------------------------------------------------------------- #
class SumoNet:
    def __init__(self, path: str):
        root = ET.parse(path).getroot()
        self.edges: Dict[str, Dict] = {}
        self.internal_lanes: Dict[str, Dict] = {}
        for e in root.iter("edge"):
            lanes = e.findall("lane")
            info = [{"id": ln.get("id"), "index": int(ln.get("index")), "length": float(ln.get("length")),
                     "width": float(ln.get("width", 3.2)), "shape": parse_shape(ln.get("shape"))} for ln in lanes]
            if e.get("function") == "internal":
                for ln in info:
                    self.internal_lanes[ln["id"]] = ln
                continue
            info.sort(key=lambda x: x["index"])
            self.edges[e.get("id")] = {"id": e.get("id"), "from": e.get("from"), "to": e.get("to"),
                                       "lanes": info, "n": len(info), "length": info[0]["length"],
                                       "speed": float(lanes[0].get("speed"))}
        # (from_edge, to_edge) -> [(fromLane, toLane, via)]
        self.conn: Dict[Tuple[str, str], List[Tuple[int, int, Optional[str]]]] = defaultdict(list)
        for c in root.iter("connection"):
            f, t = c.get("from"), c.get("to")
            if f in self.edges and t in self.edges:
                self.conn[(f, t)].append((int(c.get("fromLane")), int(c.get("toLane")), c.get("via")))
        self.succ: Dict[str, Set[str]] = defaultdict(set)
        self.pred: Dict[str, Set[str]] = defaultdict(set)
        for f, t in self.conn:
            self.succ[f].add(t)
            self.pred[t].add(f)
        self.out_of: Dict[str, List[str]] = defaultdict(list)
        self.in_of: Dict[str, List[str]] = defaultdict(list)
        for eid, e in self.edges.items():
            self.out_of[e["from"]].append(eid)
            self.in_of[e["to"]].append(eid)

    def lane_rel(self, path: List[str]) -> Set[Tuple[int, int]]:
        """SUMO lane-index relation along a path of edges, composed through the
        intermediate edges: {(lane on path[0], lane on path[-1])}."""
        rel = {(i, i) for i in range(self.edges[path[0]]["n"])}
        for a, b in zip(path, path[1:]):
            step = {(f, t) for f, t, _ in self.conn.get((a, b), [])}
            rel = {(i, k) for i, j in rel for j2, k in step if j == j2}
        return rel


# --------------------------------------------------------------------------- #
# Roads
# --------------------------------------------------------------------------- #
class RoadBuild:
    """One road: an ordered chain of SUMO edges discretised into uniform cells."""

    def __init__(self, road_id: str, kind: str, direction: str, edges: List[str], net: SumoNet,
                 cell_length: float):
        self.road_id, self.kind, self.direction, self.edge_ids = road_id, kind, direction, edges
        self.net = net
        # --- median (left) border: the leftmost lane's centre line through the
        # leftmost-to-leftmost junction-internal lanes (never another lane's,
        # which would step the border sideways), cleaned, then offset half a
        # lane to the left. s is arc length along this border, as in
        # simulation.Network._cell_polygon.
        width = max(ln["width"] for e in edges for ln in net.edges[e]["lanes"])
        self.lane_width = width
        centre: List[Pt] = []
        ends = []  # (edge, first point, last point) of its leftmost lane
        for k, eid in enumerate(edges):
            e = net.edges[eid]
            left = e["lanes"][-1]
            if k > 0:
                prev = net.edges[edges[k - 1]]
                vias = [v for f, t, v in net.conn.get((edges[k - 1], eid), [])
                        if f == prev["n"] - 1 and t == e["n"] - 1 and v in net.internal_lanes]
                if vias:
                    centre.extend(net.internal_lanes[vias[0]]["shape"])
            centre.extend(left["shape"])
            ends.append((eid, left["shape"][0], left["shape"][-1]))
        centre = clean_polyline(centre)
        pts = clean_polyline(offset_polyline(centre, width / 2))
        self.left_polyline = pts
        self.length = arclen(pts)
        self.n_max = max(net.edges[e]["n"] for e in edges)
        self.right_polyline = offset_polyline(pts, -self.n_max * width)
        # Edge extents along the border: project each edge's leftmost-lane end
        # points onto it (they lie half a lane to its right).
        spans = []
        for eid, a, b in ends:
            spans.append((eid, project_arclen(pts, a), project_arclen(pts, b)))
        spans[0] = (spans[0][0], 0.0, spans[0][2])
        spans[-1] = (spans[-1][0], spans[-1][1], self.length)
        self.n_cells = max(1, int(round(self.length / cell_length)))
        self.dx = self.length / self.n_cells
        # --- junction positions (middle of each via region) snapped to the grid
        js = [0] + [int(round(((spans[k - 1][2] + spans[k][1]) / 2) / self.dx)) for k in range(1, len(spans))] + [self.n_cells]
        for k in range(1, len(js)):
            js[k] = min(max(js[k], js[k - 1]), self.n_cells)
        self.edge_steps = {eid: (js[k], js[k + 1]) for k, eid in enumerate(edges)}
        self.edge_s = {eid: (s0, s1) for eid, s0, s1 in spans}
        self.junction_step = {net.edges[edges[k]]["from"]: js[k] for k in range(1, len(edges))}
        self.junction_step[net.edges[edges[0]]["from"]] = 0
        self.junction_step[net.edges[edges[-1]]["to"]] = self.n_cells

    def nonempty(self) -> List[str]:
        return [e for e in self.edge_ids if self.edge_steps[e][1] > self.edge_steps[e][0]]

    def lane_id(self, eid: str, idx: int) -> int:
        return idx - self.net.edges[eid]["n"]

    def cell_id(self, lane: int, step: int) -> str:
        return f"road_{self.road_id}_cell_{lane}_step_{step}"


def walk_ramp(net: SumoNet, start_edge: str, forward: bool, mainline_junctions: Set[str]) -> List[str]:
    chain = [start_edge]
    while True:
        cur = net.edges[chain[-1]]
        end_j = cur["to"] if forward else cur["from"]
        if end_j in mainline_junctions:
            break
        nxt = net.succ[chain[-1]] if forward else net.pred[chain[-1]]
        if len(nxt) != 1:
            raise SystemExit(f"ramp walk from {start_edge} hit {sorted(nxt)} after {chain[-1]}: not a simple chain")
        chain.append(next(iter(nxt)))
    return chain if forward else list(reversed(chain))


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--net", default=DEFAULT_NET)
    p.add_argument("--meta", default=DEFAULT_META)
    p.add_argument("--cell-length", type=float, default=50.0, help="target cell length (m)")
    p.add_argument("--output", default=os.path.join(DEFAULT_OUT_DIR, "network.json"))
    p.add_argument("--meta-output", default=os.path.join(DEFAULT_OUT_DIR, "network_meta.json"))
    p.add_argument("--network-id", default="i24_full_corridor_network")
    args = p.parse_args()

    net = SumoNet(args.net)
    with open(args.meta) as h:
        meta = json.load(h)
    canonical = {"fd": {"type": "triangular", **CANONICAL_FD_PARAMS},
                 "lane_change_model": {"type": "speed_incentive", "lambda_lc": CANONICAL_LAMBDA_LC}}
    params = {d: canonical for d in MAINLINE_ROAD_ID}

    # ---- mainline chains ---- #
    def base(eid):
        return eid.split("#", 1)[0].lstrip("-")

    roads: Dict[str, RoadBuild] = {}
    mainline_edges: Dict[str, str] = {}
    for d, cw in meta["carriageways"].items():
        ways = set(cw["osm_way_order"])
        chain, cur = [], cw["cut_upstream"]["node_id"]
        while cur != cw["cut_downstream"]["node_id"]:
            nxt = [x for x in net.out_of[cur] if base(x) in ways]
            if len(nxt) != 1:
                raise SystemExit(f"{d}: mainline chain broken at junction {cur}: {nxt}")
            chain.append(nxt[0])
            cur = net.edges[nxt[0]]["to"]
        rb = RoadBuild(MAINLINE_ROAD_ID[d], "mainline", d, chain, net, args.cell_length)
        roads[rb.road_id] = rb
        for e in chain:
            mainline_edges[e] = rb.road_id
    mainline_junctions = {net.edges[e]["from"] for e in mainline_edges} | {net.edges[e]["to"] for e in mainline_edges}

    # ---- ramp chains ---- #
    ramp_info = {}
    for r in meta["ramp_ends"]:
        j = r["sumo_junction"]
        if r["role"] == "source":
            chain = walk_ramp(net, net.out_of[j][0], True, mainline_junctions)
            attach_j = net.edges[chain[-1]]["to"]
            kind = "on_ramp"
        else:
            chain = walk_ramp(net, net.in_of[j][0], False, mainline_junctions)
            attach_j = net.edges[chain[0]]["from"]
            kind = "off_ramp"
        main_here = [e for e in net.in_of[attach_j] + net.out_of[attach_j] if e in mainline_edges]
        main_road = mainline_edges[main_here[0]]
        d = roads[main_road].direction
        rid = f"{d}_{'on' if kind == 'on_ramp' else 'off'}_{chain[0] if kind == 'on_ramp' else chain[0]}"
        rb = RoadBuild(rid, kind, d, chain, net, args.cell_length)
        roads[rid] = rb
        ramp_info[rid] = {"ramp_end": r, "attach_junction": attach_j, "mainline_road": main_road}

    # ---- cells ---- #
    cells: Dict[Tuple[str, str], Dict] = {}
    cell_extra: Dict[Tuple[str, str], Dict] = {}
    for rb in roads.values():
        for eid in rb.nonempty():
            e = net.edges[eid]
            k0, k1 = rb.edge_steps[eid]
            s_e0, s_e1 = rb.edge_s[eid]
            for idx in range(e["n"]):
                lane = rb.lane_id(eid, idx)
                for k in range(k0, k1):
                    cid = rb.cell_id(lane, k)
                    cells[(rb.road_id, cid)] = {
                        "road_id": rb.road_id, "cell_id": cid, "lane": lane,
                        "start_s": k * rb.dx, "end_s": (k + 1) * rb.dx if k + 1 < rb.n_cells else rb.length,
                        "mass": 0.0, "mask_mass": 0.0, "velocity": None,
                        "inflow_connections": [], "outflow_connections": [],
                        "fd": params[rb.direction]["fd"], "lane_change_model": params[rb.direction]["lane_change_model"],
                    }
                    mid = (k + 0.5) * rb.dx
                    cell_extra[(rb.road_id, cid)] = {
                        "sumo_edge": eid, "sumo_lane": f"{eid}_{idx}", "lanes_here": e["n"],
                        "sumo_pos_mid": round(min(max(mid - s_e0, 0.0), e["length"]), 2),
                    }

    links: Set[Tuple[Tuple[str, str], Tuple[str, str]]] = set()
    cross_log: List[str] = []

    def link(a, b, why=None):
        if a not in cells or b not in cells:
            raise SystemExit(f"connection {a} -> {b} references a missing cell ({why})")
        links.add((a, b))
        if why:
            cross_log.append(f"{a[1]} -> {b[1]}  [{why}]")

    for rb in roads.values():
        ne = rb.nonempty()
        # within each non-empty edge
        for eid in ne:
            k0, k1 = rb.edge_steps[eid]
            for idx in range(net.edges[eid]["n"]):
                lane = rb.lane_id(eid, idx)
                for k in range(k0, k1 - 1):
                    link((rb.road_id, rb.cell_id(lane, k)), (rb.road_id, rb.cell_id(lane, k + 1)))
        # across junctions between consecutive non-empty edges
        for a, b in zip(ne, ne[1:]):
            ia, ib = rb.edge_ids.index(a), rb.edge_ids.index(b)
            path = rb.edge_ids[ia:ib + 1]
            rel = net.lane_rel(path)
            if not rel:
                raise SystemExit(f"road {rb.road_id}: no lane connection along {path}")
            ka, kb = rb.edge_steps[a][1] - 1, rb.edge_steps[b][0]
            for i, j in sorted(rel):
                la, lb = rb.lane_id(a, i), rb.lane_id(b, j)
                link((rb.road_id, rb.cell_id(la, ka)), (rb.road_id, rb.cell_id(lb, kb)),
                     None if la == lb else f"lane {'add' if net.edges[b]['n'] > net.edges[a]['n'] else 'drop'} "
                                           f"{a}->{b}")

    # ramps <-> mainline
    # A ramp whose junction snaps onto the mainline's own end has nothing to
    # connect to there: an off-ramp diverging at the upstream cut becomes a
    # boundary source, an on-ramp merging at the downstream cut a boundary sink.
    boundary_ramps: Dict[str, str] = {}
    for rid, info in ramp_info.items():
        rb = roads[rid]
        mb = roads[info["mainline_road"]]
        j = info["attach_junction"]
        chain = mb.edge_ids
        ne = mb.nonempty()
        info["mainline_step"] = mb.junction_step.get(j)
        info["mainline_s"] = round(mb.junction_step.get(j) * mb.dx, 2)
        if rb.kind == "off_ramp" and not any(e in ne for e in chain[:next(i for i, e in enumerate(chain) if net.edges[e]["to"] == j) + 1]):
            boundary_ramps[rid] = "source"
            info["boundary"] = "source: diverges at the mainline's upstream cut, fed as a boundary"
            continue
        if rb.kind == "on_ramp" and not any(e in ne for e in chain[next(i for i, e in enumerate(chain) if net.edges[e]["from"] == j):]):
            boundary_ramps[rid] = "sink"
            info["boundary"] = "sink: merges at the mainline's downstream cut, drains as a boundary"
            continue
        if rb.kind == "on_ramp":
            r_edge = rb.nonempty()[-1]
            tail = rb.edge_ids[rb.edge_ids.index(r_edge):]
            k = next(i for i, e in enumerate(chain) if net.edges[e]["from"] == j)
            b = next(e for e in chain[k:] if e in ne)
            path = tail + chain[k:chain.index(b) + 1]
            rel = net.lane_rel(path)
            for i, jj in sorted(rel):
                link((rid, rb.cell_id(rb.lane_id(r_edge, i), rb.n_cells - 1)),
                     (mb.road_id, mb.cell_id(mb.lane_id(b, jj), mb.edge_steps[b][0])), f"on-ramp {rid} merge")
        else:
            r_edge = rb.nonempty()[0]
            head = rb.edge_ids[:rb.edge_ids.index(r_edge) + 1]
            k = next(i for i, e in enumerate(chain) if net.edges[e]["to"] == j)
            a = next(e for e in reversed(chain[:k + 1]) if e in ne)
            path = chain[chain.index(a):k + 1] + head
            rel = net.lane_rel(path)
            for i, jj in sorted(rel):
                link((mb.road_id, mb.cell_id(mb.lane_id(a, i), mb.edge_steps[a][1] - 1)),
                     (rid, rb.cell_id(rb.lane_id(r_edge, jj), 0)), f"off-ramp {rid} diverge")
        if not rel:
            raise SystemExit(f"ramp {rid}: no lane connection to the mainline along {path}")

    for a, b in sorted(links):
        cells[a]["outflow_connections"].append([b[0], b[1]])
        cells[b]["inflow_connections"].append([a[0], a[1]])

    # ---- structural sources / sinks: must be exactly the declared ends ---- #
    declared_src, declared_snk = set(), set()
    for rb in roads.values():
        first, last = rb.nonempty()[0], rb.nonempty()[-1]
        if rb.kind in ("mainline", "on_ramp") or boundary_ramps.get(rb.road_id) == "source":
            for idx in range(net.edges[first]["n"]):
                declared_src.add((rb.road_id, rb.cell_id(rb.lane_id(first, idx), 0)))
        if rb.kind in ("mainline", "off_ramp") or boundary_ramps.get(rb.road_id) == "sink":
            for idx in range(net.edges[last]["n"]):
                declared_snk.add((rb.road_id, rb.cell_id(rb.lane_id(last, idx), rb.n_cells - 1)))
    src = {k for k, c in cells.items() if not c["inflow_connections"]}
    snk = {k for k, c in cells.items() if not c["outflow_connections"]}
    problems = []
    for k in sorted(src - declared_src):
        problems.append(f"cell {k[1]} has no inflow but is not a declared source (would inject boundary flow)")
    for k in sorted(declared_src - src):
        problems.append(f"declared source {k[1]} has inflow connections")
    for k in sorted(snk - declared_snk):
        problems.append(f"cell {k[1]} has no outflow but is not a declared sink (would leak mass)")
    for k in sorted(declared_snk - snk):
        problems.append(f"declared sink {k[1]} has outflow connections")

    # every cell on a source -> sink path
    out_adj, in_adj = defaultdict(list), defaultdict(list)
    for a, b in links:
        out_adj[a].append(b)
        in_adj[b].append(a)

    def bfs(seeds, adj):
        seen, q = set(seeds), deque(seeds)
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return seen

    fwd, bwd = bfs(src, out_adj), bfs(snk, in_adj)
    for k in cells:
        if k not in fwd or k not in bwd:
            problems.append(f"cell {k[1]} is not on a source -> sink path")
    if problems:
        for msg in problems:
            print("FAIL:", msg)
        raise SystemExit(f"{len(problems)} structural problem(s); network not written")

    # distance to the end of each cell's lane (along same-lane successors)
    for k in sorted(cells, key=lambda k: (k[0], cells[k]["lane"], -cells[k]["start_s"])):
        c = cells[k]
        same = [tuple(x) for x in c["outflow_connections"]
                if x[0] == c["road_id"] and cells[tuple(x)]["lane"] == c["lane"]]
        nxt = cell_extra[same[0]]["dist_to_lane_end_m"] if same else 0.0
        cell_extra[k]["dist_to_lane_end_m"] = round(nxt + (c["end_s"] - c["start_s"]) if same else 0.0, 2)
        cell_extra[k]["lane_ends_in"] = (cell_extra[same[0]]["lane_ends_in"] if same else
                                         ("sink" if k in snk else
                                          ",".join(sorted({f"{x[0]}:{cells[tuple(x)]['lane']}" for x in c["outflow_connections"]}))))

    # ---- write ---- #
    network = {"network_id": args.network_id, "roads": {}}
    for rid, rb in roads.items():
        lanes = range(1, rb.n_max + 1)
        network["roads"][rid] = {
            "road_id": rid,
            "left_polyline": [[x, y] for x, y in rb.left_polyline],
            "right_polyline": [[x, y] for x, y in rb.right_polyline],
            "lane_data": {str(-k): {"lateral_position": -(k - 1) * rb.lane_width, "width": rb.lane_width} for k in lanes},
            "cells": {k[1]: c for k, c in sorted(cells.items(), key=lambda kv: (kv[1]["lane"], kv[1]["start_s"])) if k[0] == rid},
        }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as h:
        json.dump(network, h, indent=2)

    sidecar = {
        "description": "Sidecar for network.json, written by sumo/full/build_i24_full_fluid_network.py",
        "sumo_net": os.path.relpath(args.net, os.path.dirname(os.path.abspath(args.meta_output))),
        "fd": {"source": "fd_params.py", **CANONICAL_FD_PARAMS, "lambda_lc": CANONICAL_LAMBDA_LC},
        "cell_length_target_m": args.cell_length,
        "roads": {
            rid: {
                "kind": rb.kind, "direction": rb.direction, "length_m": round(rb.length, 2),
                "cell_length_m": round(rb.dx, 3), "n_steps": rb.n_cells, "max_lanes": rb.n_max,
                "sumo_edges": [{"edge": e, "s_start": round(rb.edge_s[e][0], 2), "s_end": round(rb.edge_s[e][1], 2),
                                "steps": list(rb.edge_steps[e]), "lanes": net.edges[e]["n"]} for e in rb.edge_ids],
                **({k: v for k, v in ramp_info[rid].items()} if rid in ramp_info else {}),
            } for rid, rb in roads.items()
        },
        "sources": sorted([list(k) for k in src]),
        "sinks": sorted([list(k) for k in snk]),
        "cross_connections": cross_log,
        "cells": {f"{k[0]}/{k[1]}": v for k, v in sorted(cell_extra.items())},
    }
    with open(args.meta_output, "w") as h:
        json.dump(sidecar, h, indent=2)

    # ---- report ---- #
    print(f"wrote {args.output}: {len(roads)} roads, {len(cells)} cells, {len(links)} connections")
    for rid, rb in roads.items():
        prof = []
        for e in rb.nonempty():
            n = net.edges[e]["n"]
            if not prof or prof[-1][1] != n:
                prof.append([rb.edge_steps[e][0], n])
        print(f"  {rid:<22} {rb.kind:<9} {rb.length:7.1f} m  {rb.n_cells:>3} x {rb.dx:5.1f} m  "
              f"lanes by step: {' '.join(f'{k}:{n}' for k, n in prof)}"
              + (f"  -> joins road {ramp_info[rid]['mainline_road']} at step {ramp_info[rid]['mainline_step']}"
                 if rid in ramp_info else ""))
    print(f"sources: {len(src)} cells, sinks: {len(snk)} cells; cross-lane/road connections: {len(cross_log)}")
    for line in cross_log:
        print("   ", line)
    print(f"wrote {args.meta_output}")


if __name__ == "__main__":
    main()
