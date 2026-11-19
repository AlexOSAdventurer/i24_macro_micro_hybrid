"""Build a SUMO network of the full I-24 MOTION corridor, ramps included, from OSM.

This is a separate pipeline from ``sumo/build_i24_network.py``, which converts
the 1-mile CARLA map; that script and its outputs are left untouched.

What gets built
---------------
* Both I-24 carriageways, each trimmed to the extent of the trajectory data
  (``--eb-mm-range`` default MM 59.40 -> 62.78, ``--wb-mm-range`` 62.78 ->
  58.67; see evaluate_i24_full_network.py). Each carriageway is cut where the
  boundary mile-marker point projects onto it, and stays ONE continuous chain:
  interior data outages never split it. Eastbound runs toward increasing MM,
  westbound toward decreasing MM.
* Mainline lane counts from OSM ``lanes`` tags, corrected by
  ``lane_overrides.json`` (MM spans fitted to the trajectory data, with the
  evidence recorded per entry). Ways are split at span ends; ends near a ramp
  attachment snap to it. Lanes are ``--lane-width`` (3.66 m = 12 ft) wide.
* Ramps meet the OUTSIDE of the carriageway. OSM joins ramps to the
  carriageway centreline, so a direct import draws them into the middle of
  the road. netconvert therefore runs twice: OSM -> plain XML, then
  ``place_ramps_on_outside`` pushes each ramp's first 150 m out beside the
  outermost lane, ends it on that lane, and keeps ramp<->mainline
  connections on the outermost lane(s) only; then plain XML -> .net.xml.
* Every ramp (``highway=motorway_link``) attached to the trimmed mainline,
  kept out to ``--ramp-max-length`` (default 175 m) of travel along the link
  graph from I-24, its first contact with a non-I-24 road, or its first fork
  or join with another ramp branch, whichever comes first. That end is the ramp's external SOURCE (on-ramp) or SINK
  (off-ramp). Local roads are dropped, so each ramp end is a dead end in SUMO:
  a place to insert or remove vehicles. The I-24 MOTION cameras only see the
  first stretch of each ramp, so nothing farther out has data anyway.

Coordinates
-----------
OSM is imported with netconvert's default UTM projection and normalized offset,
so SUMO x/y are metres in a local frame. The net's ``<location>`` element keeps
the projection, so lat/lon can always be recovered. The CARLA/OpenDRIVE frame of
the 1-mile network does NOT apply here.

Outputs (next to this script unless overridden)
------------------------------------------------
* ``i24_full_corridor.net.xml``: the SUMO network.
* ``i24_full_corridor.meta.json``: provenance and topology: OSM timestamp,
  MM range, cut points, the ordered OSM way chain for each carriageway, and
  every ramp terminal (source/sink, local road name, location). The fluid-network
  generator and the verifier read this file.
* ``osm/i24_full_corridor.filtered.osm``: the trimmed OSM handed to netconvert.
* ``osm/plain/i24_full_corridor.*.xml``: the plain-XML intermediate after the
  ramp-geometry fix (the input of the second netconvert pass).

Usage::

    ./download_osm_corridor.sh            # once; writes osm/i24_corridor.osm
    python3 build_i24_full_network.py
    python3 verify_i24_full_network.py
    python3 sample_i24_trajectories.py    # once; eval/traj_sample.npz
    python3 evaluate_i24_full_network.py  # map vs trajectory data
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))

DEFAULT_OSM = os.path.join(HERE, "osm", "i24_corridor.osm")
DEFAULT_FILTERED = os.path.join(HERE, "osm", "i24_full_corridor.filtered.osm")
DEFAULT_NET = os.path.join(HERE, "i24_full_corridor.net.xml")
DEFAULT_META = os.path.join(HERE, "i24_full_corridor.meta.json")
DEFAULT_MILE_MARKERS = os.path.join(PROJECT, "auxiliary_information", "mile_marker_layer.csv")
DEFAULT_LANE_OVERRIDES = os.path.join(HERE, "lane_overrides.json")

I24_REF_RE = re.compile(r"\bI[ -]?24\b")
LINK = "motorway_link"
# Synthetic node/way ids for cut points and split pieces. Well above any real
# OSM id, so they cannot collide.
SYNTHETIC_ID_BASE = 10**13
# A boundary mile marker lies on the median, so each carriageway is roughly half
# a cross-section away. Anything farther means the projection picked the wrong
# stretch of road.
MAX_CUT_OFFSET_M = 60.0

EARTH_R = 6371008.8
# How far a split ramp-end node is pulled back from the shared local-road node.
SPLIT_PULLBACK_M = 1.0


def dist_latlon(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    kx = math.radians(1.0) * EARTH_R * math.cos(math.radians((a[0] + b[0]) / 2))
    ky = math.radians(1.0) * EARTH_R
    return math.hypot((b[1] - a[1]) * kx, (b[0] - a[0]) * ky)


# --------------------------------------------------------------------------- #
# OSM loading and geometry helpers
# --------------------------------------------------------------------------- #
@dataclass
class Way:
    way_id: str
    tags: Dict[str, str]
    nodes: List[str]


@dataclass
class OsmData:
    osm_base: str
    nodes: Dict[str, Tuple[float, float]]  # id -> (lat, lon)
    ways: Dict[str, Way]


def load_osm(path: str) -> OsmData:
    root = ET.parse(path).getroot()
    meta = root.find("meta")
    osm_base = meta.get("osm_base", "unknown") if meta is not None else "unknown"
    nodes = {n.get("id"): (float(n.get("lat")), float(n.get("lon"))) for n in root.iter("node")}
    ways = {}
    for w in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in w.iter("tag")}
        nds = [nd.get("ref") for nd in w.iter("nd")]
        ways[w.get("id")] = Way(w.get("id"), tags, nds)
    return OsmData(osm_base, nodes, ways)


class LocalFrame:
    """Equirectangular metres about a reference latitude: accurate to well under
    0.1% across a ~7 km corridor, which is all the cutting step needs."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0, self.lon0 = lat0, lon0
        self.kx = math.radians(1.0) * EARTH_R * math.cos(math.radians(lat0))
        self.ky = math.radians(1.0) * EARTH_R

    def xy(self, lat: float, lon: float) -> Tuple[float, float]:
        return (lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky

    def latlon(self, x: float, y: float) -> Tuple[float, float]:
        return self.lat0 + y / self.ky, self.lon0 + x / self.kx


def load_mile_markers(path: str) -> List[Tuple[float, float, float]]:
    """[(mm, lat, lon)] sorted by MM (the source CSV is not fully sorted)."""
    with open(path, newline="") as handle:
        rows = [(float(r["MM"]), float(r["Y_WGS84"]), float(r["X_WGS84"])) for r in csv.DictReader(handle)]
    return sorted(rows)


def mm_to_latlon(mm: float, markers: List[Tuple[float, float, float]]) -> Tuple[float, float]:
    """Piecewise-linear in MM between survey posts, extrapolating off the ends
    from the nearest pair (MM 58.7 lies 0.1 before the first post)."""
    if mm <= markers[0][0]:
        a, b = markers[0], markers[1]
    elif mm >= markers[-1][0]:
        a, b = markers[-2], markers[-1]
    else:
        i = next(k for k in range(len(markers) - 1) if markers[k][0] <= mm <= markers[k + 1][0])
        a, b = markers[i], markers[i + 1]
    f = (mm - a[0]) / (b[0] - a[0])
    return a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])


def project_point_on_segment(p, a, b) -> Tuple[float, float]:
    """(fraction along a->b clamped to [0,1], distance from p to that point)."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    f = 0.0 if denom == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / denom))
    qx, qy = ax + f * dx, ay + f * dy
    return f, math.hypot(p[0] - qx, p[1] - qy)


# --------------------------------------------------------------------------- #
# Mainline: chain the I-24 ways per carriageway and trim to the MM range
# --------------------------------------------------------------------------- #
def is_i24_mainline(way: Way) -> bool:
    return way.tags.get("highway") == "motorway" and bool(I24_REF_RE.search(way.tags.get("ref", "")))


def way_direction(way: Way, osm: OsmData) -> str:
    """Eastbound I-24 travels south-east here (MM increasing, lon increasing)."""
    lon_a = osm.nodes[way.nodes[0]][1]
    lon_b = osm.nodes[way.nodes[-1]][1]
    return "eb" if lon_b > lon_a else "wb"


def build_chain(ways: List[Way]) -> List[Way]:
    """Order a carriageway's ways head-to-tail. Fails loudly on a branch or gap:
    a mainline that is not one simple chain needs a human to look at it."""
    by_start: Dict[str, List[Way]] = defaultdict(list)
    ends = set()
    for w in ways:
        by_start[w.nodes[0]].append(w)
        ends.add(w.nodes[-1])
    heads = [w for w in ways if w.nodes[0] not in ends]
    if len(heads) != 1:
        raise SystemExit(
            f"mainline does not form one chain: {len(heads)} head ways "
            f"({[w.way_id for w in heads]}); inspect the OSM extract"
        )
    chain = [heads[0]]
    seen = {heads[0].way_id}
    while True:
        nxt = by_start.get(chain[-1].nodes[-1], [])
        nxt = [w for w in nxt if w.way_id not in seen]
        if not nxt:
            break
        if len(nxt) > 1:
            raise SystemExit(
                f"mainline branches after way {chain[-1].way_id}: {[w.way_id for w in nxt]}"
            )
        chain.append(nxt[0])
        seen.add(nxt[0].way_id)
    if len(chain) != len(ways):
        missing = sorted({w.way_id for w in ways} - seen)
        raise SystemExit(f"mainline chain skipped ways {missing}: gap or loop in OSM extract")
    return chain


@dataclass
class ChainPos:
    seg: int          # index of the node pair in the flattened chain
    frac: float       # fraction along that pair
    arc_m: float      # distance from the chain head
    offset_m: float   # perpendicular distance of the projected point


@dataclass
class TrimmedCarriageway:
    direction: str
    ways: List[Way]                       # trimmed pieces, upstream -> downstream
    node_ids: List[str]                   # trimmed node sequence
    cut_upstream: Dict[str, float]
    cut_downstream: Dict[str, float]
    length_m: float
    source_way_order: List[str] = field(default_factory=list)
    piece_origin: Dict[str, str] = field(default_factory=dict)  # piece way id -> source OSM way id
    lane_overrides_applied: List[Dict] = field(default_factory=list)


def trim_carriageway(direction, chain, osm, frame, mm_points, new_nodes, next_id) -> TrimmedCarriageway:
    # Flatten to (node_a, node_b, way) pairs.
    pairs = [(w.nodes[i], w.nodes[i + 1], w) for w in chain for i in range(len(w.nodes) - 1)]
    xy = {n: frame.xy(*osm.nodes[n]) for w in chain for n in w.nodes}
    seg_len = [math.dist(xy[a], xy[b]) for a, b, _ in pairs]

    def locate(latlon) -> ChainPos:
        p = frame.xy(*latlon)
        best = None
        arc = 0.0
        for k, (a, b, _) in enumerate(pairs):
            f, d = project_point_on_segment(p, xy[a], xy[b])
            if best is None or d < best.offset_m:
                best = ChainPos(k, f, arc + f * seg_len[k], d)
            arc += seg_len[k]
        return best

    positions = {mm: locate(ll) for mm, ll in mm_points.items()}
    for mm, pos in positions.items():
        if pos.offset_m > MAX_CUT_OFFSET_M:
            raise SystemExit(
                f"{direction}: MM {mm} projects {pos.offset_m:.1f} m off the carriageway "
                f"(limit {MAX_CUT_OFFSET_M} m); the OSM extract may not cover it"
            )
    (mm_up, up), (mm_dn, dn) = sorted(positions.items(), key=lambda kv: kv[1].arc_m)
    expected_up = min(mm_points) if direction == "eb" else max(mm_points)
    if mm_up != expected_up:
        raise SystemExit(f"{direction}: MM {mm_up} is upstream in chain order, expected {expected_up}")

    def make_cut_node(pos: ChainPos) -> str:
        a, b, _ = pairs[pos.seg]
        x = xy[a][0] + pos.frac * (xy[b][0] - xy[a][0])
        y = xy[a][1] + pos.frac * (xy[b][1] - xy[a][1])
        nid = str(next_id())
        new_nodes[nid] = frame.latlon(x, y)
        return nid

    cut_a, cut_b = make_cut_node(up), make_cut_node(dn)

    # Rebuild node pairs strictly between the cuts, then regroup by source way.
    kept: List[Tuple[str, str, Way]] = []
    for k, (a, b, w) in enumerate(pairs):
        if k < up.seg or k > dn.seg:
            continue
        start = cut_a if k == up.seg else a
        end = cut_b if k == dn.seg else b
        kept.append((start, end, w))

    pieces: List[Way] = []
    for a, b, w in kept:
        if pieces and pieces[-1].way_id == w.way_id:
            pieces[-1].nodes.append(b)
        else:
            pieces.append(Way(w.way_id, dict(w.tags), [a, b]))
    node_ids = [pieces[0].nodes[0]] + [n for p in pieces for n in p.nodes[1:]]

    def describe(mm, pos, nid):
        lat, lon = (new_nodes[nid])
        return {"mm": mm, "lat": lat, "lon": lon, "offset_from_marker_m": round(pos.offset_m, 2),
                "osm_way": pairs[pos.seg][2].way_id, "node_id": nid}

    return TrimmedCarriageway(
        direction=direction,
        ways=pieces,
        node_ids=node_ids,
        cut_upstream=describe(mm_up, up, cut_a),
        cut_downstream=describe(mm_dn, dn, cut_b),
        length_m=dn.arc_m - up.arc_m,
        source_way_order=[w.way_id for w in chain],
        piece_origin={w.way_id: w.way_id for w in pieces},
    )


# --------------------------------------------------------------------------- #
# Mainline lane-count overrides (lane_overrides.json)
# --------------------------------------------------------------------------- #
# A span end this close (along the carriageway) to a ramp attachment node snaps
# to it, so a lane add/drop coincides with the ramp junction. OSM ramp nodes sit
# 40-90 m from the surveyed gore MMs in lane_overrides.json, and no two
# attachments on one carriageway are closer than ~150 m, hence 100 m.
OVERRIDE_SNAP_M = 100.0
# Per-lane tags that would contradict an overridden lane count.
PER_LANE_TAGS = ("turn:lanes", "destination:lanes", "destination:ref:lanes", "change:lanes",
                 "width:lanes", "placement", "lanes:forward", "lanes:backward")


def apply_lane_overrides(cw: TrimmedCarriageway, overrides: List[Dict], osm: OsmData, frame: LocalFrame,
                         markers, attach_nodes: Set[str], new_nodes, next_id) -> None:
    """Set the `lanes` tag of the mainline between two MMs, splitting OSM ways
    where a span starts or ends. The carriageway stays one chain."""

    def latlon(n):
        return new_nodes[n] if n in new_nodes else osm.nodes[n]

    def seq():
        return [cw.ways[0].nodes[0]] + [n for w in cw.ways for n in w.nodes[1:]]

    def arcs(nodes):
        pts = [frame.xy(*latlon(n)) for n in nodes]
        out = [0.0]
        for a, b in zip(pts, pts[1:]):
            out.append(out[-1] + math.dist(a, b))
        return pts, out

    def node_at(mm: float, snap: bool = True) -> Tuple[str, str]:
        nodes = seq()
        pts, arc = arcs(nodes)
        p = frame.xy(*mm_to_latlon(mm, markers))
        best = None
        for k in range(len(nodes) - 1):
            f, d = project_point_on_segment(p, pts[k], pts[k + 1])
            if best is None or d < best[0]:
                best = (d, k, f)
        d, k, f = best
        if d > MAX_CUT_OFFSET_M:
            raise SystemExit(f"{cw.direction}: lane override MM {mm} projects {d:.1f} m off the carriageway")
        a = arc[k] + f * (arc[k + 1] - arc[k])
        snaps = [(abs(arc[i] - a), i) for i, n in enumerate(nodes)
                 if snap and n in attach_nodes and abs(arc[i] - a) <= OVERRIDE_SNAP_M]
        if snaps:
            gap, i = min(snaps)
            return nodes[i], f"snapped {gap:.0f} m to ramp attachment {nodes[i]}"
        for i in (k, k + 1):
            if abs(arc[i] - a) < 1.0:
                return nodes[i], f"at existing node {nodes[i]}"
        # Insert a new node into the piece that holds this node pair.
        x = pts[k][0] + f * (pts[k + 1][0] - pts[k][0])
        y = pts[k][1] + f * (pts[k + 1][1] - pts[k][1])
        nid = str(next_id())
        new_nodes[nid] = frame.latlon(x, y)
        for w in cw.ways:
            for j in range(len(w.nodes) - 1):
                if w.nodes[j] == nodes[k] and w.nodes[j + 1] == nodes[k + 1]:
                    w.nodes.insert(j + 1, nid)
                    return nid, "new node"
        raise AssertionError("node pair not found in carriageway pieces")

    def split_at(nid: str) -> None:
        for idx, w in enumerate(cw.ways):
            if nid in w.nodes[1:-1]:
                j = w.nodes.index(nid)
                tail_id = str(next_id())
                tail = Way(tail_id, dict(w.tags), w.nodes[j:])
                w.nodes = w.nodes[: j + 1]
                cw.ways.insert(idx + 1, tail)
                cw.piece_origin[tail_id] = cw.piece_origin[w.way_id]
                return

    for ov in overrides:
        if ov["direction"] != cw.direction:
            continue
        (na, how_a) = node_at(ov["from_mm"], ov.get("from_snap", True))
        (nb, how_b) = node_at(ov["to_mm"], ov.get("to_snap", True))
        split_at(na)
        split_at(nb)
        nodes = seq()
        ia, ib = sorted((nodes.index(na), nodes.index(nb)))
        if ia == ib:
            raise SystemExit(f"{cw.direction}: lane override {ov} collapses to a single node")
        pos = 0
        changed = []
        for w in cw.ways:
            start, end = pos, pos + len(w.nodes) - 1
            if start >= ia and end <= ib:
                old = w.tags.get("lanes")
                w.tags["lanes"] = str(ov["lanes"])
                for t in PER_LANE_TAGS:
                    w.tags.pop(t, None)
                changed.append({"piece": w.way_id, "osm_way": cw.piece_origin[w.way_id], "lanes_was": old})
            pos = end
        cw.lane_overrides_applied.append({**ov, "from_node": how_a, "to_node": how_b, "pieces": changed})
        print(f"{cw.direction}: lanes={ov['lanes']} MM {ov['from_mm']} ({how_a}) -> MM {ov['to_mm']} ({how_b}): "
              + ", ".join(f"{c['piece']} ({c['lanes_was']}->{ov['lanes']})" for c in changed))
    cw.node_ids = seq()


# --------------------------------------------------------------------------- #
# Ramps: walk the motorway_link graph out from the trimmed mainline
# --------------------------------------------------------------------------- #
def link_travels_forward(way: Way) -> bool:
    """True if traffic moves in node order. motorway_link is one-way by default."""
    return way.tags.get("oneway", "yes") != "-1"


def collect_ramps(osm: OsmData, mainline_in: Set[str], mainline_all: Set[str], new_nodes, next_id,
                  max_len_m: float):
    """Return (link pieces, piece origins, ramp ends, warnings).

    Each ramp is kept from its I-24 attachment out to whichever comes FIRST:
      * ``max_len_m`` of travel along the link graph (shortest path from any
        trimmed-mainline attachment node), cutting mid-segment if need be;
      * the first node it shares with any non-I-24 road (a local-road terminal); or
      * the first point where it forks into, or is fed by, more than one link
        (a branch point). The branches beyond are dropped.
    The far end becomes a dead end: a SOURCE if traffic on the kept piece flows
    away from it (on-ramp), a SINK if traffic flows into it (off-ramp).

    Distances come from a Dijkstra over link node pairs. A link joining two
    attachments is kept whole only if every point on it lies within
    ``max_len_m`` of one attachment or the other. Otherwise its middle is
    dropped, leaving a sink piece and a source piece.

    Each link touching a local-road terminal gets its own copy of that node, so
    an off-ramp ending where an on-ramp begins cannot become a through path
    that skips the local road.
    """
    local_roads_at: Dict[str, Set[str]] = defaultdict(set)
    link_pairs_at: Dict[str, List[Tuple[str, str, Way]]] = defaultdict(list)
    for w in osm.ways.values():
        hw = w.tags.get("highway", "")
        if hw == LINK:
            for i in range(len(w.nodes) - 1):
                pair = (w.nodes[i], w.nodes[i + 1], w)
                link_pairs_at[w.nodes[i]].append(pair)
                link_pairs_at[w.nodes[i + 1]].append(pair)
        elif hw != "motorway":
            label = w.tags.get("name") or w.tags.get("ref") or f"{hw} way {w.way_id}"
            for n in w.nodes:
                local_roads_at[n].add(f"{hw}:{label}")

    def plen(a: str, b: str) -> float:
        return dist_latlon(osm.nodes[a], osm.nodes[b])

    # A link node with more than two link neighbours is a fork (off-ramp
    # splitting into branches) or a join (branches combining into one on-ramp).
    # The ramp ends there; the branches beyond it are not modelled.
    link_neighbours: Dict[str, Set[str]] = defaultdict(set)
    for n, pairs in link_pairs_at.items():
        for a, b, _ in pairs:
            link_neighbours[n].add(b if n == a else a)

    def is_branch(n: str) -> bool:
        return n not in mainline_all and len(link_neighbours[n]) > 2

    # --- shortest link-path distance from the trimmed mainline -------------- #
    terminals: Dict[str, Dict] = {}
    warnings: List[str] = []
    dist: Dict[str, float] = {}
    expandable: Set[str] = set()  # nodes the ramp may continue past
    heap = [(0.0, n) for n in mainline_in if n in link_pairs_at]
    for _, n in heap:
        dist[n] = 0.0
    heapq.heapify(heap)
    while heap:
        d, n = heapq.heappop(heap)
        if d > dist.get(n, math.inf):
            continue
        if n in local_roads_at:
            terminals.setdefault(n, {"node_id": n, "roads": sorted(local_roads_at[n]), "reason": "local_road"})
            continue
        if is_branch(n):
            terminals.setdefault(n, {"node_id": n, "roads": ["ramp branch point"], "reason": "branch"})
            continue
        if n in mainline_all and n not in mainline_in:
            if n not in terminals:
                warnings.append(f"a ramp reaches I-24 OUTSIDE the trim range at node {n}; it ends there")
            terminals.setdefault(n, {"node_id": n, "roads": ["I-24 outside trim range"], "reason": "outside_range"})
            continue
        if d >= max_len_m:
            continue
        expandable.add(n)
        for a, b, _ in link_pairs_at[n]:
            other = b if n == a else a
            nd = d + plen(a, b)
            if nd < dist.get(other, math.inf):
                dist[other] = nd
                heapq.heappush(heap, (nd, other))

    def reach(n: str) -> float:
        """How far the ramp may run on past node n."""
        return max_len_m - dist[n] if n in expandable else 0.0

    # --- cut every link way to the kept stretches --------------------------- #
    pieces: List[Way] = []
    piece_origin: Dict[str, str] = {}
    cut_nodes: Dict[str, Dict] = {}

    def cut_point(a: str, b: str, frac: float) -> str:
        (la, lo), (lb, lob) = osm.nodes[a], osm.nodes[b]
        nid = str(next_id())
        new_nodes[nid] = (la + frac * (lb - la), lo + frac * (lob - lo))
        cut_nodes[nid] = {"node_id": nid, "roads": [f"{max_len_m:g} m ramp cutoff"], "reason": "max_length"}
        return nid

    for w in sorted((w for w in osm.ways.values() if w.tags.get("highway") == LINK), key=lambda w: w.way_id):
        runs: List[List[str]] = []
        run: List[str] = []
        for i in range(len(w.nodes) - 1):
            a, b = w.nodes[i], w.nodes[i + 1]
            L = plen(a, b)
            ka, kb = min(reach(a), L), min(reach(b), L)
            if (ka > 0 or kb > 0) and ka + kb >= L - 1e-6:
                if not run:
                    run = [a]
                run.append(b)
                continue
            # Part (or none) of this segment survives, from one or both ends.
            if ka > 0:
                if not run:
                    run = [a]
                run.append(cut_point(a, b, ka / L))
            if len(run) > 1:
                runs.append(run)
            run = []
            if kb > 0:
                run = [cut_point(a, b, 1.0 - kb / L), b]
        if len(run) > 1:
            runs.append(run)
        for k, nodes in enumerate(runs):
            pid = w.way_id if k == 0 else str(next_id())
            pieces.append(Way(pid, dict(w.tags), nodes))
            piece_origin[pid] = w.way_id

    # Find dead ends that touch no local road (OSM gaps); they still act as ramp
    # ends, but get flagged.
    degree: Dict[str, int] = defaultdict(int)
    for p in pieces:
        for i in range(len(p.nodes) - 1):
            degree[p.nodes[i]] += 1
            degree[p.nodes[i + 1]] += 1
    for p in pieces:
        for node in (p.nodes[0], p.nodes[-1]):
            if (node not in terminals and node not in cut_nodes
                    and degree[node] == 1 and node not in mainline_in):
                warnings.append(
                    f"link {p.way_id} ({piece_origin[p.way_id]}) dead-ends at node {node} without "
                    f"touching a local road or reaching {max_len_m:g} m; treated as an unlabelled ramp end"
                )
                terminals[node] = {"node_id": node, "roads": ["DANGLING"], "reason": "dangling"}

    # One ramp end per (link piece, end node). Cut nodes are already unique to
    # their piece; shared terminal nodes get a per-piece copy.
    ramp_ends: List[Dict] = []
    for p in pieces:
        forward = link_travels_forward(p)
        for idx, role_if_first in ((0, "source"), (-1, "sink")):
            node = p.nodes[idx]
            role = role_if_first if forward else ("sink" if role_if_first == "source" else "source")
            if node in cut_nodes:
                lat, lon = new_nodes[node]
                ramp_ends.append({
                    "sumo_junction": node,
                    "role": role,
                    "reason": "max_length",
                    "path_from_i24_m": round(max_len_m, 1),
                    "link_way": p.way_id,
                    "osm_way": piece_origin[p.way_id],
                    "osm_terminal_node": None,
                    "roads": cut_nodes[node]["roads"],
                    "lat": lat,
                    "lon": lon,
                })
                continue
            if node not in terminals:
                continue
            # netconvert's OSM importer merges nodes at identical coordinates,
            # which would re-join the copies. Pull each copy back along its own
            # link by SPLIT_PULLBACK_M (at most a quarter of the first segment).
            lat, lon = osm.nodes[node]
            nb = p.nodes[1] if idx == 0 else p.nodes[-2]
            nlat, nlon = new_nodes.get(nb, osm.nodes.get(nb))
            seg = dist_latlon((lat, lon), (nlat, nlon))
            f = min(SPLIT_PULLBACK_M / seg, 0.25) if seg > 0 else 0.0
            split_id = str(next_id())
            new_nodes[split_id] = (lat + f * (nlat - lat), lon + f * (nlon - lon))
            p.nodes[idx] = split_id
            ramp_ends.append({
                "sumo_junction": split_id,
                "role": role,
                "reason": terminals[node]["reason"],
                "path_from_i24_m": round(dist[node], 1) if node in dist else None,
                "link_way": p.way_id,
                "osm_way": piece_origin[p.way_id],
                "osm_terminal_node": node,
                "roads": terminals[node]["roads"],
                "lat": lat,
                "lon": lon,
            })
    unused = [t for t in terminals if not any(e["osm_terminal_node"] == t for e in ramp_ends)]
    for t in unused:
        warnings.append(f"terminal {t} ({terminals[t]['roads']}) is not the end of any kept link; check topology")

    ramp_ends.sort(key=lambda e: (e["lon"], e["lat"], e["role"]))
    return pieces, piece_origin, ramp_ends, warnings


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_filtered_osm(path, osm, new_nodes, ways: List[Way]) -> None:
    """Minimal OSM XML: nodes carry no tags, so OSM traffic signals and other
    node-level features at ramp terminals cannot leak into the SUMO junctions."""
    used = []
    seen = set()
    for w in ways:
        for n in w.nodes:
            if n not in seen:
                seen.add(n)
                used.append(n)
    root = ET.Element("osm", version="0.6", generator="build_i24_full_network.py")
    ET.SubElement(root, "meta", osm_base=osm.osm_base)
    for n in used:
        lat, lon = new_nodes[n] if n in new_nodes else osm.nodes[n]
        ET.SubElement(root, "node", id=n, lat=f"{lat:.8f}", lon=f"{lon:.8f}", version="1", visible="true")
    for w in ways:
        el = ET.SubElement(root, "way", id=w.way_id, version="1", visible="true")
        for n in w.nodes:
            ET.SubElement(el, "nd", ref=n)
        for k, v in sorted(w.tags.items()):
            ET.SubElement(el, "tag", k=k, v=v)
    ET.indent(root)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _netconvert(args: List[str]) -> None:
    netconvert = shutil.which("netconvert")
    if netconvert is None:
        raise SystemExit("netconvert not on PATH")
    env = dict(os.environ)
    if "SUMO_HOME" not in env:
        # Without it netconvert falls back to built-in type maps and fetches
        # schemas from the web. Infer it from the binary (/usr/bin -> /usr/share/sumo).
        guess = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(netconvert))), "share", "sumo")
        if os.path.isdir(guess):
            env["SUMO_HOME"] = guess
    command = [netconvert] + args
    print("running:", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    if result.returncode != 0:
        raise SystemExit(f"netconvert failed with exit code {result.returncode}")


# Options shared by both netconvert passes.
NETCONVERT_COMMON = [
    "--remove-edges.isolated", "true",
    "--tls.guess", "false",
    "--no-turnarounds", "true",
    "--output.street-names", "true",
    "--output.original-names", "true",
    "--geometry.remove", "false",
]
# How far out from its I-24 junction a ramp's shape is moved to the outside of
# the carriageway, the resampling step, and the taper over which its end
# point eases from "beside the outermost lane" to "on the outermost lane".
RAMP_ALONGSIDE_M = 150.0
RAMP_RESAMPLE_M = 10.0
RAMP_TAPER_M = 40.0


def _shape(s: str) -> List[Tuple[float, float]]:
    return [tuple(float(v) for v in p.split(",")[:2]) for p in s.split()]


def _resample_from_start(pts, upto_m, step_m):
    """Points every step_m along pts for the first upto_m, then the original
    points beyond; each returned with its distance from pts[0]."""
    out = [(pts[0], 0.0)]
    acc = 0.0
    for a, b in zip(pts, pts[1:]):
        L = math.dist(a, b)
        if L == 0:
            continue
        t = step_m - (acc % step_m) if acc < upto_m else None
        while t is not None and t < L and acc + t <= upto_m:
            f = t / L
            out.append(((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])), acc + t))
            t += step_m
        acc += L
        out.append((b, acc))
    return out


def place_ramps_on_outside(prefix: str, mainline_by_dir: Dict[str, Set[str]], ramp_ids: Set[str]) -> List[str]:
    """Edit netconvert's plain-XML export in place so ramps meet the OUTSIDE
    of the carriageway.

    OSM draws each carriageway as one centreline and joins ramps to it, and
    SUMO spreads lanes symmetrically about that line, so every ramp is drawn
    running into the middle of the road. Here, over the first RAMP_ALONGSIDE_M
    of each ramp, every shape point is pushed out (never in) to clear the
    carriageway edge on the ramp's side, and the end at the I-24 junction is
    put on the outermost lane(s). Lane connections between a ramp and the
    mainline are limited to the outermost mainline lanes (one per ramp lane).

    Mainline edges wider than the carriageway's base lane count are shifted
    right by half a lane per extra lane. Otherwise SUMO centres every edge on
    the OSM line and the whole cross-section jumps half a lane at each lane
    add or drop; real lanes are added on the right with the median edge
    continuous. Junction node positions are untouched. Returns log lines.
    """
    nod_path, edg_path, con_path = (f"{prefix}.{k}.xml" for k in ("nod", "edg", "con"))
    nodes = {n.get("id"): (float(n.get("x")), float(n.get("y"))) for n in ET.parse(nod_path).getroot().iter("node")}
    edg_tree = ET.parse(edg_path)
    edges = {e.get("id"): e for e in edg_tree.getroot().iter("edge")}

    def base(eid):
        return eid.split("#", 1)[0].lstrip("-")

    def full_shape(e):
        pts = _shape(e.get("shape")) if e.get("shape") else []
        a, b = nodes[e.get("from")], nodes[e.get("to")]
        if not pts or math.dist(pts[0], a) > 0.01:
            pts = [a] + pts
        if math.dist(pts[-1], b) > 0.01:
            pts = pts + [b]
        return pts

    def width(e):
        return float(e.get("width", 3.2))

    dir_of = {}
    for d, bases in mainline_by_dir.items():
        for eid in edges:
            if base(eid) in bases:
                dir_of[eid] = d
    segs = {d: [] for d in mainline_by_dir}
    for eid, d in dir_of.items():
        pts = full_shape(edges[eid])
        segs[d].extend(zip(pts, pts[1:]))

    def project(d, p):
        """(foot point, unit right-of-travel normal, signed lateral, + = right)."""
        best = None
        for a, b in segs[d]:
            f, dist = project_point_on_segment(p, a, b)
            if best is None or dist < best[0]:
                best = (dist, a, b, f)
        _, a, b, f = best
        L = math.dist(a, b)
        ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        foot = (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
        right = (uy, -ux)
        lat = (p[0] - foot[0]) * right[0] + (p[1] - foot[1]) * right[1]
        return foot, right, lat

    at_node = defaultdict(list)
    for eid, d in dir_of.items():
        at_node[edges[eid].get("from")].append(eid)
        at_node[edges[eid].get("to")].append(eid)

    # Cross-section frame. OSM's line is the centre of the BASE carriageway
    # (its minimum lane count, n_base). The median (left) edge is held at
    # -n_base*w/2 everywhere and extra lanes are added on the right, so an
    # n-lane edge is centred (n - n_base)*w/2 right of the OSM line. Lateral
    # values below are metres right of the OSM line.
    n_base = {d: min(int(edges[e].get("numLanes", 1)) for e in dir_of if dir_of[e] == d) for d in mainline_by_dir}

    log = []
    ramp_side = {}  # ramp edge id -> (+1 right / -1 left, mainline lanes at its junction)
    for eid, e in edges.items():
        if base(eid) not in ramp_ids:
            continue
        on = e.get("to") in at_node
        off = e.get("from") in at_node
        if not (on or off):
            continue  # inner piece of a multi-edge ramp
        junction = e.get("to") if on else e.get("from")
        main_here = at_node[junction]
        d = dir_of[main_here[0]]
        n_max = max(int(edges[m].get("numLanes", 1)) for m in main_here)
        w = max(width(edges[m]) for m in main_here)
        k = int(e.get("numLanes", 1))
        wr = width(e)
        pts = full_shape(e)
        outward = list(reversed(pts)) if on else pts
        samples = _resample_from_start(outward, RAMP_ALONGSIDE_M, RAMP_RESAMPLE_M)
        # Which side the ramp leaves on: where it is at the end of the window.
        probe = next((p for p, s in samples if s >= RAMP_ALONGSIDE_M), samples[-1][0])
        side = 1 if project(d, probe)[2] >= 0 else -1
        # The mainline edge the ramp runs alongside: the one entering the
        # junction for an on-ramp, the one leaving it for an off-ramp.
        alongside = [m for m in main_here if (edges[m].get("to") if on else edges[m].get("from")) == junction]
        n_side = int(edges[alongside[0]].get("numLanes", 1)) if alongside else n_max
        half_base = n_base[d] * w / 2
        if side > 0:
            # Right edge of an n-lane edge sits at -half_base + n*w.
            on_lane = -half_base + n_max * w - k * wr / 2   # ramp centre over the outermost k lanes
            beside = -half_base + n_side * w + k * wr / 2   # ramp just outside the edge it parallels
        else:
            on_lane = half_base - k * wr / 2                # the median edge never moves
            beside = half_base + k * wr / 2
        new = []
        for p, s in samples:
            if s > RAMP_ALONGSIDE_M:
                new.append(p)
                continue
            want = beside if s >= RAMP_TAPER_M else on_lane + (beside - on_lane) * s / RAMP_TAPER_M
            foot, right, lat = project(d, p)
            if side * lat < want:
                p = (foot[0] + side * want * right[0], foot[1] + side * want * right[1])
            new.append(p)
        if on:
            new.reverse()
        e.set("shape", " ".join(f"{x:.2f},{y:.2f}" for x, y in new))
        ramp_side[eid] = (side, n_max)
        log.append(f"ramp {eid} ({'on' if on else 'off'}, {k} lane(s), {'right' if side > 0 else 'left'} side): "
                   f"end {on_lane:.2f} m right of the OSM line ({n_max} lanes at the junction), "
                   f"alongside at {beside:.2f} m ({n_side}-lane edge)")

    # Shift wider mainline edges right so the median edge is continuous.
    for eid, d in dir_of.items():
        e = edges[eid]
        shift = (int(e.get("numLanes", 1)) - n_base[d]) * width(e) / 2
        if abs(shift) < 1e-6:
            continue
        pts = full_shape(e)
        moved = []
        for i, p in enumerate(pts):
            normals = []
            segs_here = [(pts[j], pts[j + 1]) for j in (i - 1, i) if 0 <= j < len(pts) - 1]
            for a, b in segs_here:
                L = math.dist(a, b)
                if L > 0:
                    normals.append(((b[1] - a[1]) / L, -(b[0] - a[0]) / L))
            nx = sum(n[0] for n in normals) / len(normals)
            ny = sum(n[1] for n in normals) / len(normals)
            nl = math.hypot(nx, ny)
            moved.append((p[0] + shift * nx / nl, p[1] + shift * ny / nl))
        e.set("shape", " ".join(f"{x:.2f},{y:.2f}" for x, y in moved))
        log.append(f"mainline {eid}: {e.get('numLanes')} lanes, shifted {shift:.2f} m right to keep the median edge continuous")
    edg_tree.write(edg_path, encoding="utf-8", xml_declaration=True)

    # Connections: ramp lanes <-> only the outermost mainline lanes.
    con_tree = ET.parse(con_path)
    root = con_tree.getroot()
    for c in list(root.iter("connection")):
        f, t = c.get("from"), c.get("to")
        if f in ramp_side and t in dir_of:
            ramp, main, main_lane, ramp_lane = f, t, int(c.get("toLane")), int(c.get("fromLane"))
        elif t in ramp_side and f in dir_of:
            ramp, main, main_lane, ramp_lane = t, f, int(c.get("fromLane")), int(c.get("toLane"))
        else:
            continue
        side, _ = ramp_side[ramp]
        n_main = int(edges[main].get("numLanes", 1))
        k = int(edges[ramp].get("numLanes", 1))
        allowed = range(0, k) if side > 0 else range(n_main - k, n_main)
        if main_lane not in allowed:
            root.remove(c)
            log.append(f"dropped connection {f}_{c.get('fromLane')} -> {t}_{c.get('toLane')} "
                       f"(not an outermost lane of {main})")
        elif (main_lane - allowed[0]) != ramp_lane:
            log.append(f"note: connection {f}_{c.get('fromLane')} -> {t}_{c.get('toLane')} crosses lanes")

    # Mainline lane adds/drops with no ramp at the junction: netconvert may
    # add the new lane on the left. Rewrite them median-aligned: through lanes
    # keep their position counted from the median; an added lane on the right
    # is fed from the outermost lane, a dropped one merges into it.
    ramp_nodes = {edges[r].get("to") for r in ramp_side} | {edges[r].get("from") for r in ramp_side}
    pairs = {(c.get("from"), c.get("to")) for c in root.iter("connection")
             if c.get("from") in dir_of and c.get("to") in dir_of}
    for f, t in sorted(pairs):
        nf, nt = int(edges[f].get("numLanes", 1)), int(edges[t].get("numLanes", 1))
        if nf == nt or edges[f].get("to") in ramp_nodes:
            continue
        for c in [c for c in root.iter("connection") if c.get("from") == f and c.get("to") == t]:
            root.remove(c)
        if nt > nf:
            add = nt - nf
            lanes = [(i, i + add) for i in range(nf)] + [(0, j) for j in range(add)]
        else:
            drop = nf - nt
            lanes = [(i, i - drop) for i in range(drop, nf)] + [(i, 0) for i in range(drop)]
        for a, b in sorted(lanes):
            ET.SubElement(root, "connection", {"from": f, "to": t, "fromLane": str(a), "toLane": str(b)})
        log.append(f"lane {'add' if nt > nf else 'drop'} {f} ({nf}) -> {t} ({nt}): median-aligned connections "
                   + ", ".join(f"{a}->{b}" for a, b in sorted(lanes)))
    con_tree.write(con_path, encoding="utf-8", xml_declaration=True)
    return log


def run_netconvert(osm_path: str, net_path: str, lane_width: float, plain_dir: str,
                   mainline_by_dir: Dict[str, Set[str]], ramp_ids: Set[str]) -> List[str]:
    """Pass 1: OSM -> plain XML. Fix ramp geometry/connections in the plain
    XML (place_ramps_on_outside). Pass 2: plain XML -> the final .net.xml."""
    os.makedirs(plain_dir, exist_ok=True)
    prefix = os.path.join(plain_dir, "i24_full_corridor")
    _netconvert([
        "--osm-files", osm_path,
        "--plain-output-prefix", prefix,
        # Lane counts come from OSM `lanes` tags; do not invent ramp lanes.
        "--ramps.guess", "false",
        "--keep-edges.by-vclass", "passenger",
        "--default.lanewidth", f"{lane_width:g}",
        "-o", os.devnull,
    ] + NETCONVERT_COMMON)
    log = place_ramps_on_outside(prefix, mainline_by_dir, ramp_ids)
    for line in log:
        print("  " + line)
    _netconvert([
        "--node-files", f"{prefix}.nod.xml",
        "--edge-files", f"{prefix}.edg.xml",
        "--connection-files", f"{prefix}.con.xml",
        "--type-files", f"{prefix}.typ.xml",
        "--tllogic-files", f"{prefix}.tll.xml",
        "--ramps.guess", "false",
        # The plain nodes are already in the pass-1 frame; keep it.
        "--offset.disable-normalization", "true",
        "-o", net_path,
    ] + NETCONVERT_COMMON)
    return log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--osm", default=DEFAULT_OSM, help="raw Overpass extract (download_osm_corridor.sh)")
    parser.add_argument("--mile-markers", default=DEFAULT_MILE_MARKERS)
    # Per-direction cuts at the extent of the trajectory data (found with
    # evaluate_i24_full_network.py: EB data spans MM ~59.38-62.78, WB
    # ~62.80-58.66). Interior outages never move a cut: each carriageway stays
    # one continuous road.
    parser.add_argument("--eb-mm-range", type=float, nargs=2, default=[59.40, 62.78], metavar=("START", "END"),
                        help="eastbound upstream and downstream cut MM")
    parser.add_argument("--wb-mm-range", type=float, nargs=2, default=[62.78, 58.67], metavar=("START", "END"),
                        help="westbound upstream and downstream cut MM")
    parser.add_argument("--lane-overrides", default=DEFAULT_LANE_OVERRIDES,
                        help="JSON of mainline lane-count spans (pass '' to use OSM lanes only)")
    parser.add_argument("--lane-width", type=float, default=3.66,
                        help="SUMO lane width (m); 3.66 = the 12 ft lanes in the MOTION data")
    parser.add_argument("--ramp-max-length", type=float, default=175.0,
                        help="metres of ramp kept from I-24, unless a local road comes first")
    parser.add_argument("--filtered-osm", default=DEFAULT_FILTERED)
    parser.add_argument("--output", default=DEFAULT_NET, help="output .net.xml")
    parser.add_argument("--meta", default=DEFAULT_META, help="output topology/provenance json")
    args = parser.parse_args()

    if not os.path.isfile(args.osm):
        raise SystemExit(f"{args.osm} not found; run download_osm_corridor.sh first")
    osm = load_osm(args.osm)
    markers = load_mile_markers(args.mile_markers)
    ranges = {"eb": args.eb_mm_range, "wb": args.wb_mm_range}
    for d, (a, b) in ranges.items():
        if (b > a) != (d == "eb"):
            raise SystemExit(f"--{d}-mm-range {a} {b}: EB runs toward increasing MM, WB toward decreasing")
    mm_points_by_dir = {d: {mm: mm_to_latlon(mm, markers) for mm in r} for d, r in ranges.items()}
    all_ll = [ll for pts in mm_points_by_dir.values() for ll in pts.values()]
    frame = LocalFrame(sum(ll[0] for ll in all_ll) / len(all_ll), sum(ll[1] for ll in all_ll) / len(all_ll))

    counter = [SYNTHETIC_ID_BASE]

    def next_id() -> int:
        counter[0] += 1
        return counter[0]

    new_nodes: Dict[str, Tuple[float, float]] = {}
    mainline_ways = [w for w in osm.ways.values() if is_i24_mainline(w)]
    link_nodes = {n for w in osm.ways.values() if w.tags.get("highway") == LINK for n in w.nodes}
    overrides = []
    if args.lane_overrides:
        with open(args.lane_overrides) as handle:
            overrides = json.load(handle)["overrides"]
    carriageways: Dict[str, TrimmedCarriageway] = {}
    for direction in ("eb", "wb"):
        ways = [w for w in mainline_ways if way_direction(w, osm) == direction]
        chain = build_chain(ways)
        carriageways[direction] = trim_carriageway(direction, chain, osm, frame, mm_points_by_dir[direction],
                                                   new_nodes, next_id)
        cw = carriageways[direction]
        apply_lane_overrides(cw, overrides, osm, frame, markers, link_nodes & set(cw.node_ids), new_nodes, next_id)
        print(f"{direction}: {len(chain)} OSM ways -> {len(cw.ways)} trimmed pieces, "
              f"{cw.length_m:.1f} m between MM {cw.cut_upstream['mm']} and MM {cw.cut_downstream['mm']}")

    mainline_in = {n for cw in carriageways.values() for n in cw.node_ids}
    mainline_all = {n for w in mainline_ways for n in w.nodes}
    link_pieces, piece_origin, ramp_ends, warnings = collect_ramps(
        osm, mainline_in, mainline_all, new_nodes, next_id, args.ramp_max_length
    )
    print(f"ramps: {len(link_pieces)} link pieces, {len(ramp_ends)} ramp ends "
          f"({sum(e['role'] == 'source' for e in ramp_ends)} sources, "
          f"{sum(e['role'] == 'sink' for e in ramp_ends)} sinks); "
          f"{sum(e['reason'] == 'local_road' for e in ramp_ends)} at a local road, "
          f"{sum(e['reason'] == 'branch' for e in ramp_ends)} at a branch point, "
          f"{sum(e['reason'] == 'max_length' for e in ramp_ends)} at the {args.ramp_max_length:g} m cutoff")
    no_lanes = sorted(p.way_id for p in link_pieces if "lanes" not in p.tags)
    if no_lanes:
        warnings.append(f"{len(no_lanes)} link pieces have no `lanes` tag; netconvert's type default (1) applies: {no_lanes}")
    for msg in warnings:
        print("WARNING:", msg)

    all_ways = [w for cw in carriageways.values() for w in cw.ways] + link_pieces
    write_filtered_osm(args.filtered_osm, osm, new_nodes, all_ways)
    print(f"wrote {args.filtered_osm}")
    ramp_log = run_netconvert(
        args.filtered_osm, args.output, args.lane_width,
        os.path.join(os.path.dirname(os.path.abspath(args.filtered_osm)), "plain"),
        {d: {w.way_id for w in cw.ways} for d, cw in carriageways.items()},
        {p.way_id for p in link_pieces},
    )
    print(f"wrote {args.output}")

    meta = {
        "description": "Full I-24 MOTION corridor, built from OSM by build_i24_full_network.py",
        "osm_source": os.path.relpath(args.osm, HERE),
        "osm_base": osm.osm_base,
        "net_file": os.path.relpath(args.output, HERE),
        "mm_range": {"eb": list(args.eb_mm_range), "wb": list(args.wb_mm_range)},
        "lane_width_m": args.lane_width,
        "ramp_max_length_m": args.ramp_max_length,
        "carriageways": {
            d: {
                "direction": "eastbound (MM increasing)" if d == "eb" else "westbound (MM decreasing)",
                "osm_way_order": [w.way_id for w in cw.ways],
                "piece_origin": {w.way_id: cw.piece_origin[w.way_id] for w in cw.ways},
                "lanes_by_piece": {w.way_id: w.tags.get("lanes") for w in cw.ways},
                "lane_overrides_applied": cw.lane_overrides_applied,
                "cut_upstream": cw.cut_upstream,
                "cut_downstream": cw.cut_downstream,
                "length_m_local_frame": round(cw.length_m, 2),
            }
            for d, cw in carriageways.items()
        },
        "ramp_link_ways": {p.way_id: {"osm_way": piece_origin[p.way_id],
                                      "lanes": p.tags.get("lanes"),
                                      "destination": p.tags.get("destination")}
                           for p in link_pieces},
        "ramp_ends": ramp_ends,
        "ramp_geometry_log": ramp_log,
        "warnings": warnings,
    }
    with open(args.meta, "w") as handle:
        json.dump(meta, handle, indent=2)
    print(f"wrote {args.meta}")


if __name__ == "__main__":
    main()
