"""Check the full-corridor SUMO network built by build_i24_full_network.py.

Reports, and fails on, the properties the coupled simulation relies on:

1. Each carriageway is one unbroken edge chain from its upstream cut to its
   downstream cut. Prints lanes, length, speed and cumulative s per edge.
2. Mainline ends and every ramp end in the meta file are SUMO junctions with
   exactly one edge, pointing the right way (source = outgoing, sink = incoming).
3. Every ramp attaches to the mainline near a surveyed I-24 MOTION landmark
   (auxiliary_information/ramp_and_landmark_layer.csv). Distances are reported;
   anything farther than --landmark-tol is flagged for a human look.
4. Flow connectivity over SUMO connections: every non-mainline edge lies on
   some path from an entry (mainline start or ramp source) to an exit (mainline
   end or ramp sink).

Standard library only; no sumolib needed.

Usage:  python3 verify_i24_full_network.py [--net ...] [--meta ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, deque

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_NET = os.path.join(HERE, "i24_full_corridor.net.xml")
DEFAULT_META = os.path.join(HERE, "i24_full_corridor.meta.json")
DEFAULT_FILTERED = os.path.join(HERE, "osm", "i24_full_corridor.filtered.osm")
DEFAULT_LANDMARKS = os.path.join(PROJECT, "auxiliary_information", "ramp_and_landmark_layer.csv")

failures = []


def fail(msg):
    failures.append(msg)
    print("  FAIL:", msg)


def base_way(edge_id: str) -> str:
    return edge_id.split("#", 1)[0].lstrip("-")


def dist_m(a, b):
    (la1, lo1), (la2, lo2) = a, b
    kx = 111320 * math.cos(math.radians((la1 + la2) / 2))
    return math.hypot((lo2 - lo1) * kx, (la2 - la1) * 110540)


def load_net(path):
    root = ET.parse(path).getroot()
    edges = {}
    internal_lane_len = {}
    for e in root.iter("edge"):
        if e.get("function") == "internal":
            for ln in e.findall("lane"):
                internal_lane_len[ln.get("id")] = float(ln.get("length"))
            continue
        lanes = e.findall("lane")
        edges[e.get("id")] = {
            "from": e.get("from"),
            "to": e.get("to"),
            "lanes": len(lanes),
            "length": float(lanes[0].get("length")),
            "speed": float(lanes[0].get("speed")),
            "name": e.get("name", ""),
        }
    junctions = {j.get("id"): j.get("type") for j in root.iter("junction") if j.get("type") != "internal"}
    succ = defaultdict(set)
    # Length of the junction-internal lane a vehicle drives between two edges.
    # Edge lengths stop at the junction boundary, so a path's true length is
    # the edge lengths plus these. The longest lane-to-lane via is kept, which
    # is conservative for an s-coordinate check.
    via_len = {}
    for c in root.iter("connection"):
        f, t = c.get("from"), c.get("to")
        if f in edges and t in edges:
            succ[f].add(t)
            v = internal_lane_len.get(c.get("via"), 0.0)
            via_len[(f, t)] = max(via_len.get((f, t), 0.0), v)
    return edges, junctions, succ, via_len


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--net", default=DEFAULT_NET)
    p.add_argument("--meta", default=DEFAULT_META)
    p.add_argument("--filtered-osm", default=DEFAULT_FILTERED)
    p.add_argument("--landmarks", default=DEFAULT_LANDMARKS)
    p.add_argument("--landmark-tol", type=float, default=150.0, help="metres")
    args = p.parse_args()

    edges, junctions, succ, via_len = load_net(args.net)
    with open(args.meta) as h:
        meta = json.load(h)
    osm_root = ET.parse(args.filtered_osm).getroot()
    latlon = {n.get("id"): (float(n.get("lat")), float(n.get("lon"))) for n in osm_root.iter("node")}
    with open(args.landmarks, newline="") as h:
        landmarks = [(r["Direction"], r["Landmark name"], float(r["MM"]), (float(r["Y_WGS84"]), float(r["X_WGS84"])))
                     for r in csv.DictReader(h)]

    out_edges = defaultdict(list)
    in_edges = defaultdict(list)
    for eid, e in edges.items():
        out_edges[e["from"]].append(eid)
        in_edges[e["to"]].append(eid)

    # ---------------- 1. mainline chains ---------------- #
    mainline_edges = {}
    entries, exits = set(), set()
    for d, cw in meta["carriageways"].items():
        print(f"\n== {d}: {cw['direction']} ==")
        ways = set(cw["osm_way_order"])
        start, end = cw["cut_upstream"]["node_id"], cw["cut_downstream"]["node_id"]
        cur, s, chain = start, 0.0, []
        print(f"  {'edge':<16}{'lanes':>6}{'length':>9}{'s_end':>9}{'speed':>7}  attachments at downstream junction")
        while cur != end:
            nxt = [eid for eid in out_edges[cur] if base_way(eid) in ways]
            if len(nxt) != 1:
                fail(f"{d}: at junction {cur} expected one mainline edge, found {nxt}")
                break
            eid = nxt[0]
            e = edges[eid]
            if chain:
                s += via_len.get((chain[-1], eid), 0.0)
            s += e["length"]
            chain.append(eid)
            mainline_edges[eid] = d
            j = e["to"]
            side_in = [x for x in in_edges[j] if base_way(x) not in ways]
            side_out = [x for x in out_edges[j] if base_way(x) not in ways]
            note = ""
            if side_in:
                note += f"MERGE from {side_in} "
            if side_out:
                note += f"DIVERGE to {side_out}"
            print(f"  {eid:<16}{e['lanes']:>6}{e['length']:>9.1f}{s:>9.1f}{e['speed'] * 2.23694:>6.0f}mph  {note}")
            cur = j
        print(f"  total {s:.1f} m over {len(chain)} edges (meta local-frame estimate {cw['length_m_local_frame']} m)")
        if chain:
            entries.add(chain[0])
            exits.add(chain[-1])
            for j, which in ((start, "upstream"), (end, "downstream")):
                n = len(in_edges[j]) + len(out_edges[j])
                if n != 1:
                    fail(f"{d}: {which} mainline end {j} has {n} edges, expected 1")

    # ---------------- 2. ramp ends ---------------- #
    # Shortest ramp-path distance (undirected, SUMO edge lengths plus junction
    # via lanes) from any mainline junction, to check the ramp length cap.
    max_len = meta.get("ramp_max_length_m")
    mainline_junctions = {edges[e]["from"] for e in mainline_edges} | {edges[e]["to"] for e in mainline_edges}
    adj = defaultdict(list)
    for eid, e in edges.items():
        if eid in mainline_edges:
            continue
        adj[e["from"]].append((e["to"], e["length"]))
        adj[e["to"]].append((e["from"], e["length"]))
    import heapq
    jdist = {j: 0.0 for j in mainline_junctions}
    heap = [(0.0, j) for j in mainline_junctions]
    heapq.heapify(heap)
    while heap:
        d, j = heapq.heappop(heap)
        if d > jdist.get(j, math.inf):
            continue
        for k, L in adj[j]:
            if d + L < jdist.get(k, math.inf):
                jdist[k] = d + L
                heapq.heappush(heap, (d + L, k))

    print(f"\n== ramp ends (cap {max_len} m from I-24) ==")
    print(f"  {'role':<6} {'junction':<16} {'edge':<16} {'reason':<11} {'path_m':>7}  status  roads")
    for r in meta["ramp_ends"]:
        j = r["sumo_junction"]
        ins, outs = in_edges.get(j, []), out_edges.get(j, [])
        ok_dir = (r["role"] == "source" and outs and not ins) or (r["role"] == "sink" and ins and not outs)
        status = "ok" if (len(ins) + len(outs) == 1 and ok_dir) else "BAD"
        edge = (outs or ins or ["<none>"])[0]
        path = jdist.get(j, math.inf)
        # SUMO edges stop short of junctions (via lanes are excluded), so the
        # edge-only path can only under-run the OSM path; 5 m covers rounding.
        if max_len is not None and path > max_len + 5.0:
            fail(f"ramp end {j} is {path:.1f} m of ramp from I-24, over the {max_len} m cap")
        print(f"  {r['role']:<6} {j:<16} {edge:<16} {r.get('reason', '?'):<11} {path:>7.1f}  {status:<6}  "
              f"{'; '.join(r['roads'])}")
        if status != "ok":
            fail(f"ramp end {j} ({r['role']}, link {r['link_way']}): in={ins} out={outs}, "
                 f"junction {'missing' if j not in junctions else junctions[j]}")
        if r["role"] == "source" and outs:
            entries.add(outs[0])
        if r["role"] == "sink" and ins:
            exits.add(ins[0])

    # ---------------- 3. attachments vs landmarks ---------------- #
    print(f"\n== ramp attachments vs surveyed landmarks (tolerance {args.landmark_tol:.0f} m) ==")
    for eid, d in mainline_edges.items():
        j = edges[eid]["to"]
        kinds = []
        if any(x not in mainline_edges for x in in_edges[j]):
            kinds.append("on")
        if any(x not in mainline_edges for x in out_edges[j]):
            kinds.append("off")
        for kind in kinds:
            if j not in latlon:
                print(f"  {d} {kind:<3} junction {j}: no OSM position (netconvert renamed it?)")
                continue
            cands = [(dist_m(latlon[j], ll), name, mm) for (dd, name, mm, ll) in landmarks
                     if dd == d and f"_{kind}_" in name]
            dm, name, mm = min(cands)
            flag = "" if dm <= args.landmark_tol else "   <-- check"
            print(f"  {d} {kind:<3} junction {j:<14} nearest {name:<20} (MM {mm:5.2f}) {dm:7.1f} m{flag}")

    # ---------------- 4. connectivity ---------------- #
    print("\n== connectivity ==")
    pred = defaultdict(set)
    for f, ts in succ.items():
        for t in ts:
            pred[t].add(f)

    def bfs(seeds, nbrs):
        seen = set(seeds)
        q = deque(seeds)
        while q:
            u = q.popleft()
            for v in nbrs[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return seen

    from_entry = bfs(entries, succ)
    to_exit = bfs(exits, pred)
    orphans = sorted(e for e in edges if e not in from_entry or e not in to_exit)
    print(f"  {len(edges)} edges, {len(entries)} entries, {len(exits)} exits")
    for e in orphans:
        why = []
        if e not in from_entry:
            why.append("unreachable from any entry")
        if e not in to_exit:
            why.append("cannot reach any exit")
        fail(f"edge {e} ({edges[e]['lanes']} lanes, {edges[e]['length']:.0f} m): {', '.join(why)}")
    if not orphans:
        print("  every edge lies on an entry->exit path")

    print(f"\n{len(failures)} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
