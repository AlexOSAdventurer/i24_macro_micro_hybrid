"""Score the full-corridor SUMO network against sampled I-24 MOTION trajectories.

Part of the full-corridor pipeline (``sumo/full``); the 1-mile pipeline is
untouched. Input is a trajectory sample from ``sample_i24_trajectories.py``.

How points are placed
---------------------
* Longitudinal: x/5280 is a mile marker. It is converted to a position on the
  median with the surveyed mile posts (piecewise linear), which agree with the
  MOTION landmark sheet to ~10 m; plain arc length is off by up to 150 m.
* Lateral: offset by y from that median point (EB y<0 is to the right of
  travel). The data's own lane comes from 12 ft bands of |y|:
  [12,24) -> lane -1 (leftmost) ... [60,72) -> -5, [72,84) -> -6. The inner
  ``--aux-guard-ft`` of the aux bands is ignored (lane -4 spill where the
  cross-section shifts outward).
* Counts are distinct vehicles per bin and lane, not points, so a vehicle
  stopped on the shoulder does not look like a lane. Aux evidence is smoothed
  over ``--aux-smooth-m``.
* Each point is projected onto its carriageway's SUMO mainline (edge centre
  shapes, upstream cut -> downstream cut) to get s, and the SUMO lane count at
  s is read from the edge there (inside a junction: the larger neighbour).
* Ramp points: every point is also matched to the nearest SUMO edge centreline
  (mainline or ramp) within half its width + 2 m, giving per-ramp-edge counts.

Lane convention matches the project: lane -1 is leftmost, auxiliary lanes are
added on the right as -5, -6, and SUMO has lane -k wherever it has >= k lanes.

What is reported (per carriageway)
----------------------------------
* A lane x distance diagram, one character per ``--bin`` metres:
    '#' >= 40% of the lane's median count, '+' 15-40%, '.' 1-15%, 'X' none
    '!' data says a lane exists here (>= --aux-ratio of the main-lane mean)
        but SUMO has none, ' ' no lane in SUMO and no lane-like data.
* Metrics at ``--metric-bin`` resolution:
    data_extent_mm     model-free MM span, first to last bin with >= 50% of the
                       median main-lane data
    dead_modelled_m    modelled mainline with no main-lane data, touching an end
                       (a cut that overshoots the data)
    interior_outages   dead stretches between live ones: expected data gaps,
                       reported only, never a map defect
    unmodelled_data_m  data-covered mainline outside the model's cuts
    aux_missing_m      length where data shows lane -5/-6 but SUMO lacks it
    aux_empty_m        length where SUMO has lane -5/-6 but data shows none
* Ramp edges with no matched points.

``--json`` appends the metrics to a file so iterations of the generator can be
compared.

Usage::

    python3 sample_i24_trajectories.py      # once
    python3 build_i24_full_network.py
    python3 evaluate_i24_full_network.py [--label NAME] [--json eval/history.jsonl]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_NET = os.path.join(HERE, "i24_full_corridor.net.xml")
DEFAULT_META = os.path.join(HERE, "i24_full_corridor.meta.json")
DEFAULT_SAMPLE = os.path.join(HERE, "eval", "traj_sample.npz")
DEFAULT_POSTS = os.path.join(PROJECT, "auxiliary_information", "mile_marker_layer.csv")
DEFAULT_LANDMARKS = os.path.join(PROJECT, "auxiliary_information", "ramp_and_landmark_layer.csv")
FT = 0.3048
MI = 1609.344
LANE_FT = 12.0
DEFAULT_SUMO_LANE_WIDTH = 3.2


# --------------------------------------------------------------------------- #
# Projection: WGS84 -> UTM (Snyder), then the net's offset
# --------------------------------------------------------------------------- #
def utm_forward(lat, lon, zone):
    a, f, k0 = 6378137.0, 1 / 298.257223563, 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    phi = np.radians(np.asarray(lat, float))
    lam = np.radians(np.asarray(lon, float))
    lam0 = math.radians((zone - 1) * 6 - 180 + 3)
    sin, cos, tan = np.sin(phi), np.cos(phi), np.tan(phi)
    N = a / np.sqrt(1 - e2 * sin ** 2)
    T = tan ** 2
    C = ep2 * cos ** 2
    A = cos * (lam - lam0)
    M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * phi
             - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * np.sin(2 * phi)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * np.sin(4 * phi)
             - (35 * e2 ** 3 / 3072) * np.sin(6 * phi))
    x = k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A ** 5 / 120) + 500000.0
    y = k0 * (M + N * tan * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
                             + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A ** 6 / 720))
    return x, y


class NetFrame:
    def __init__(self, net_root):
        loc = net_root.find("location")
        self.off = [float(v) for v in loc.get("netOffset").split(",")]
        self.zone = int(re.search(r"\+zone=(\d+)", loc.get("projParameter")).group(1))

    def xy(self, lat, lon):
        x, y = utm_forward(lat, lon, self.zone)
        return np.stack([x + self.off[0], y + self.off[1]], -1)


def parse_shape(s):
    return np.array([[float(v) for v in p.split(",")[:2]] for p in s.split()])


# --------------------------------------------------------------------------- #
# Polyline projection
# --------------------------------------------------------------------------- #
class Polyline:
    def __init__(self, pts):
        self.P = np.asarray(pts, float)
        self.A, self.B = self.P[:-1], self.P[1:]
        self.seg = self.B - self.A
        self.L = np.maximum(np.hypot(*self.seg.T), 1e-9)
        self.arc0 = np.concatenate([[0], np.cumsum(self.L)])[:-1]
        self.total = float(self.L.sum())

    def project(self, q, chunk=20000):
        """(s, signed lateral offset (+ = right of travel), distance)."""
        q = np.asarray(q, float)
        s_out = np.empty(len(q))
        lat_out = np.empty(len(q))
        d_out = np.empty(len(q))
        for k in range(0, len(q), chunk):
            qq = q[k:k + chunk]
            rel = qq[:, None, :] - self.A[None]
            t = np.clip((rel * self.seg[None]).sum(2) / (self.L ** 2)[None], 0, 1)
            proj = self.A[None] + t[..., None] * self.seg[None]
            diff = qq[:, None, :] - proj
            dist = np.hypot(diff[..., 0], diff[..., 1])
            j = dist.argmin(1)
            r = np.arange(len(qq))
            s_out[k:k + chunk] = self.arc0[j] + t[r, j] * self.L[j]
            d_out[k:k + chunk] = dist[r, j]
            u = self.seg[j] / self.L[j][:, None]
            cross = u[:, 0] * diff[r, j, 1] - u[:, 1] * diff[r, j, 0]
            lat_out[k:k + chunk] = -cross  # left of travel is +cross
        return s_out, lat_out, d_out


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
def load_net(path):
    root = ET.parse(path).getroot()
    edges = {}
    for e in root.iter("edge"):
        if e.get("function") == "internal":
            continue
        lanes = e.findall("lane")
        # Edge centre from the lane shapes, which (unlike the edge's own
        # `shape` attribute) are trimmed at the junctions: average of the
        # outermost lanes when they line up, otherwise the middle lane.
        s0, s1 = parse_shape(lanes[0].get("shape")), parse_shape(lanes[-1].get("shape"))
        shape = (s0 + s1) / 2 if s0.shape == s1.shape else parse_shape(lanes[len(lanes) // 2].get("shape"))
        width = sum(float(ln.get("width", DEFAULT_SUMO_LANE_WIDTH)) for ln in lanes)
        edges[e.get("id")] = {"from": e.get("from"), "to": e.get("to"), "lanes": len(lanes),
                              "length": float(lanes[0].get("length")), "shape": shape, "width": width,
                              "lane_shapes": [parse_shape(ln.get("shape")) for ln in lanes]}
    return root, edges


def mainline_chain(edges, start, end, ways):
    out = defaultdict(list)
    for eid, e in edges.items():
        out[e["from"]].append(eid)
    chain, cur = [], start
    while cur != end:
        nxt = [x for x in out[cur] if x.split("#")[0].lstrip("-") in ways]
        if len(nxt) != 1:
            raise SystemExit(f"mainline chain broken at junction {cur}: {nxt}")
        chain.append(nxt[0])
        cur = edges[nxt[0]]["to"]
    return chain


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--net", default=DEFAULT_NET)
    p.add_argument("--meta", default=DEFAULT_META)
    p.add_argument("--sample", default=DEFAULT_SAMPLE)
    p.add_argument("--posts", default=DEFAULT_POSTS)
    p.add_argument("--landmarks", default=DEFAULT_LANDMARKS)
    p.add_argument("--bin", type=float, default=100.0, help="diagram resolution (m)")
    p.add_argument("--metric-bin", type=float, default=25.0, help="metric resolution (m)")
    p.add_argument("--aux-ratio", type=float, default=0.1,
                   help="lane -5/-6 count / main-lane mean count above which data shows a lane "
                        "(plain 4-lane road reads ~0.01; lightly used aux lanes 0.1-0.2)")
    p.add_argument("--aux-smooth-m", type=float, default=75.0, help="smoothing window for aux evidence (m)")
    p.add_argument("--aux-guard-ft", type=float, default=2.0,
                   help="ignore the innermost feet of each aux band (lane -4 spill)")
    p.add_argument("--label", default="")
    p.add_argument("--json", default=None, help="append metrics as one JSON line to this file")
    p.add_argument("--quiet", action="store_true", help="metrics only, no diagram")
    args = p.parse_args()

    z = np.load(args.sample)
    X, Y, DAY, TRAJ = z["x"], z["y"], z["day"], z["traj"]
    ndays = len(z["days"])
    with open(args.meta) as h:
        meta = json.load(h)
    root, edges = load_net(args.net)
    frame = NetFrame(root)

    # Mile posts on the median, in net coordinates.
    with open(args.posts, newline="") as h:
        posts = sorted((float(r["MM"]), float(r["Y_WGS84"]), float(r["X_WGS84"])) for r in csv.DictReader(h))
    pmm = np.array([r[0] for r in posts])
    pxy = frame.xy([r[1] for r in posts], [r[2] for r in posts])

    def place(mm, y_ft):
        i = np.clip(np.searchsorted(pmm, mm) - 1, 0, len(pmm) - 2)
        f = (mm - pmm[i]) / (pmm[i + 1] - pmm[i])
        d = pxy[i + 1] - pxy[i]
        u = d / np.linalg.norm(d, axis=1)[:, None]
        right = np.stack([u[:, 1], -u[:, 0]], 1)  # right of increasing MM
        return pxy[i] + f[:, None] * d + (-y_ft * FT)[:, None] * right

    with open(args.landmarks, newline="") as h:
        landmarks = [(r["Direction"], r["Landmark name"], float(r["MM"])) for r in csv.DictReader(h)]

    ay = np.abs(Y)
    lane_band = np.where((ay >= 12) & (ay < 84), -(np.floor(ay / LANE_FT).astype(int)), 0)
    # Guard band on the inner edge of the auxiliary bands: where the cross-
    # section shifts outward by a few feet, lane -4 spills into [60, 62) ft
    # and would otherwise read as a fifth lane.
    lane_band[(lane_band <= -5) & (ay % LANE_FT < args.aux_guard_ft)] = 0
    Q = place(X / 5280.0, Y)
    metrics = {"label": args.label, "net": os.path.relpath(args.net, HERE), "n_points": int(len(X)),
               "ramp_max_length_m": meta.get("ramp_max_length_m"), "carriageways": {}}
    chains = {d: mainline_chain(edges, cw["cut_upstream"]["node_id"], cw["cut_downstream"]["node_id"],
                                set(w.lstrip("-") for w in cw["osm_way_order"]))
              for d, cw in meta["carriageways"].items()}
    mainline_ids = {e for c in chains.values() for e in c}

    # Nearest SUMO edge per point (centreline within half its width + 2 m).
    cand = list(edges)
    best_d = np.full(len(Q), np.inf)
    best_e = np.full(len(Q), -1)
    for i, eid in enumerate(cand):
        e = edges[eid]
        lo, hi = e["shape"].min(0) - 30, e["shape"].max(0) + 30
        m = np.all((Q >= lo) & (Q <= hi), 1)
        if not m.any():
            continue
        _, _, dist = Polyline(e["shape"]).project(Q[m])
        idx = np.where(m)[0]
        better = (dist <= e["width"] / 2 + 2.0) & (dist < best_d[idx])
        best_d[idx[better]] = dist[better]
        best_e[idx[better]] = i
    on_ramp_edge = np.isin(best_e, [i for i, e in enumerate(cand) if e not in mainline_ids])

    for d, cw in meta["carriageways"].items():
        chain = chains[d]
        pts, spans = [], []
        for eid in chain:
            sh = edges[eid]["shape"]
            if pts and np.hypot(*(sh[0] - pts[-1])) < 1e-6:
                sh = sh[1:]
            start_idx = len(pts)
            pts.extend(sh)
            spans.append((eid, start_idx, len(pts) - 1))
        line = Polyline(pts)
        arc = np.concatenate([[0], np.cumsum(line.L)])
        # SUMO lane count per metre along s; junction gaps take the larger
        # neighbour so they do not read as missing lanes.
        res = 1.0
        n_m = int(math.ceil(line.total / res))
        lanes_m = np.zeros(n_m, int)
        prev_end, prev_n = 0.0, None
        for eid, i0, i1 in spans:
            a, b, n = arc[i0], arc[i1], edges[eid]["lanes"]
            if prev_n is not None:
                lanes_m[int(prev_end / res):int(a / res) + 1] = max(prev_n, n)
            lanes_m[int(a / res):int(math.ceil(b / res))] = n
            prev_end, prev_n = b, n
        lanes_m[lanes_m == 0] = prev_n

        sel = (Y < 0) if d == "eb" else (Y > 0)
        s, lat, dist = line.project(Q[sel])
        band = lane_band[sel]
        day = DAY[sel]
        traj = TRAJ[sel]
        # Points that sit on a SUMO ramp edge are already modelled; they do not
        # count as evidence of a mainline auxiliary lane.
        on = (band != 0) & (dist < 40) & (s > 1) & (s < line.total - 1) & ~((band <= -5) & on_ramp_edge[sel])
        mm_cut = (cw["cut_upstream"]["mm"], cw["cut_downstream"]["mm"])

        def lane_counts(binw):
            # A remainder shorter than half a bin joins the last bin, so a
            # sliver at the end cannot read as an outage.
            nb = max(1, int(round(line.total / binw)))
            b = np.minimum((s[on] // binw).astype(int), nb - 1)
            # Distinct vehicles per (bin, lane), not points: a vehicle stopped
            # on the shoulder would otherwise look like a busy lane.
            key = np.unique(np.stack([traj[on], b, band[on]], 1), axis=0)
            counts = {k: np.bincount(key[key[:, 2] == k, 1], minlength=nb) for k in range(-6, 0)}
            centres = ((np.arange(nb) + 0.5) * binw / res).astype(int).clip(0, n_m - 1)
            return nb, b, counts, lanes_m[centres]

        def classify(nb, counts, lanes_at):
            main = np.mean([counts[k] for k in (-1, -2, -3, -4)], 0)
            med = np.median(main[main > 0]) if (main > 0).any() else 0.0
            dead = main < 0.15 * med
            # Aux evidence on counts smoothed over ~aux_smooth_m, so a single
            # bin of lane changers or ramp traffic does not read as a lane.
            w = max(1, int(round(args.aux_smooth_m / (line.total / nb))))
            kern = np.ones(w) / w
            main_s = np.convolve(main, kern, mode="same")
            aux_data = {k: (~dead) & (np.convolve(counts[k], kern, mode="same")
                                      >= args.aux_ratio * np.maximum(main_s, 1e-9)) for k in (-5, -6)}
            return main, med, dead, aux_data

        # ---------------- metrics ---------------- #
        mb = args.metric_bin
        nb, _, counts, lanes_at = lane_counts(mb)
        main, med, dead, aux_data = classify(nb, counts, lanes_at)
        aux_missing = sum(int(((~(lanes_at >= -k)) & aux_data[k]).sum()) for k in (-5, -6)) * mb
        aux_empty = sum(int(((lanes_at >= -k) & ~aux_data[k] & ~dead).sum()) for k in (-5, -6)) * mb
        mm_cut0 = (cw["cut_upstream"]["mm"], cw["cut_downstream"]["mm"])

        def spans_of(mask):
            out = []
            for i in np.where(mask)[0]:
                if out and out[-1][1] == i * mb:
                    out[-1][1] = (i + 1) * mb
                else:
                    out.append([i * mb, (i + 1) * mb])
            to_mm = lambda v: round(mm_cut0[0] + (mm_cut0[1] - mm_cut0[0]) * v / line.total, 3)
            return [[float(a), float(b), to_mm(a), to_mm(b)] for a, b in out]

        aux_spans = {f"missing_lane{k}": spans_of((~(lanes_at >= -k)) & aux_data[k]) for k in (-5, -6)}
        aux_spans.update({f"empty_lane{k}": spans_of((lanes_at >= -k) & ~aux_data[k] & ~dead) for k in (-5, -6)})

        # Model-free extent of main-lane data, in MM (posts frame), at ~metric-bin.
        dsel = sel & (lane_band >= -4) & (lane_band <= -1)
        mm_pts = X[dsel] / 5280.0
        step = mb / MI
        mm_bins = np.arange(min(pmm[0], *mm_cut) - 0.3, max(pmm[-1], *mm_cut) + step, step)
        hist, _ = np.histogram(mm_pts, mm_bins)
        good = np.where(hist >= 0.5 * np.median(hist[hist > 0]))[0]
        # First to last well-covered bin: interior outages are expected and
        # do not shorten the extent.
        ext = (round(float(mm_bins[good[0]]), 3), round(float(mm_bins[good[-1] + 1]), 3))
        lo_cut, hi_cut = min(mm_cut), max(mm_cut)
        unmodelled = (max(0.0, lo_cut - ext[0]) + max(0.0, ext[1] - hi_cut)) * MI
        # Dead bins touching either end are a cut that overshoots the data;
        # dead bins in between are interior outages (informational only).
        live = np.where(~dead)[0]
        end_dead = dead.copy()
        if len(live):
            end_dead[live[0]:live[-1] + 1] = False
        dead_m = float(end_dead.sum() * mb)
        interior = dead & ~end_dead
        interior_spans = []
        for i in np.where(interior)[0]:
            if interior_spans and interior_spans[-1][1] == i * mb:
                interior_spans[-1][1] = (i + 1) * mb
            else:
                interior_spans.append([i * mb, (i + 1) * mb])

        days_dead_bins = 0
        for di in range(ndays):
            m = on & (day == di) & (band >= -4)
            cnt = np.bincount(np.minimum((s[m] // mb).astype(int), nb - 1), minlength=nb)
            days_dead_bins += int(((cnt == 0) & ~dead).sum())

        cm = {
            "cut_mm": list(mm_cut),
            "length_m": round(line.total, 1),
            "points_on_mainline": int(on.sum()),
            "data_extent_mm": list(ext),
            "dead_modelled_m": dead_m,
            "interior_outages_s_m": interior_spans,
            "unmodelled_data_m": round(unmodelled, 1),
            "aux_missing_m": aux_missing,
            "aux_empty_m": aux_empty,
            "aux_spans_s_s_mm_mm": aux_spans,
            "single_day_dead_bin_days": days_dead_bins,
            "lanes_profile": [[round(float(arc[i0]), 1), edges[eid]["lanes"]] for eid, i0, _ in spans],
        }
        metrics["carriageways"][d] = cm

        # ---------------- diagram ---------------- #
        if not args.quiet:
            B = args.bin
            nbB, _, cB, laB = lane_counts(B)
            mainB, medB, deadB, auxB = classify(nbB, cB, laB)
            print(f"\n=== {cw['direction']}: MM {mm_cut[0]} -> {mm_cut[1]}, {line.total:.0f} m, "
                  f"{B:.0f} m per char, {int(on.sum()):,} points ===")
            ruler = [" "] * nbB
            for k in range(0, nbB, 10):
                label = f"{mm_cut[0] + (mm_cut[1] - mm_cut[0]) * (k * B / line.total):.2f}"
                for i, ch in enumerate(label):
                    if k + i < nbB:
                        ruler[k + i] = ch
            print(f"{'MM (approx)':<12}|{''.join(ruler)}|")
            marks = [" "] * nbB
            for dd, name, mm in landmarks:
                if dd != d or not any(t in name for t in ("_on_", "_off_")):
                    continue
                k = int((mm - mm_cut[0]) / (mm_cut[1] - mm_cut[0]) * line.total // B)
                if 0 <= k < nbB:
                    marks[k] = "v" if "_off_" in name else "^"
            print(f"{'landmarks':<12}|{''.join(marks)}|  v off-ramp gore, ^ on-ramp gore (sheet MM)")
            print(f"{'SUMO lanes':<12}|{''.join(str(v) for v in laB)}|")
            for k in (-1, -2, -3, -4, -5, -6):
                c = cB[k]
                exists = laB >= -k
                lmed = np.median(c[exists & ~deadB]) if (exists & ~deadB).any() else 0
                row = []
                for i in range(nbB):
                    if not exists[i]:
                        row.append("!" if k <= -5 and auxB[k][i] else " ")
                    elif c[i] == 0:
                        row.append("X")
                    elif c[i] < 0.15 * lmed:
                        row.append(".")
                    elif c[i] < 0.40 * lmed:
                        row.append("+")
                    else:
                        row.append("#")
                print(f"{'lane ' + str(k):<12}|{''.join(row)}|")
            dd_row = np.zeros(nbB, int)
            for di in range(ndays):
                m = on & (day == di) & (band >= -4)
                cnt = np.bincount(np.minimum((s[m] // B).astype(int), nbB - 1), minlength=nbB)
                dd_row += cnt == 0
            print(f"{'days dead':<12}|{''.join(str(v) if v < 10 else 'A' for v in dd_row)}|")
        print(f"  {d}: data extent MM {ext[0]:.3f}-{ext[1]:.3f} | dead at ends {dead_m:.0f} m | "
              f"unmodelled data {unmodelled:.0f} m | aux missing {aux_missing:.0f} m | aux empty {aux_empty:.0f} m"
              f" | interior outages (s, m) {[[float(a), float(b)] for a, b in interior_spans]}")
        for key, spans in aux_spans.items():
            if spans:
                print(f"      {key}: " + ", ".join(f"s {a:.0f}-{b:.0f} (~MM {ma:.2f}-{mb_:.2f})" for a, b, ma, mb_ in spans))

    # ---------------- ramps ---------------- #
    ramp_counts = {}
    for i, eid in enumerate(cand):
        if eid in mainline_ids:
            continue
        ramp_counts[eid] = int((best_e == i).sum())
    empty = sorted(e for e, c in ramp_counts.items() if c == 0)
    metrics["ramp_edges"] = len(ramp_counts)
    metrics["ramp_edges_empty"] = empty
    metrics["ramp_counts"] = ramp_counts
    print(f"\nramp edges: {len(ramp_counts)}, with no matched points: {len(empty)} {empty}")
    if not args.quiet:
        for eid in sorted(ramp_counts, key=lambda e: -ramp_counts[e]):
            print(f"  {eid:<18} {edges[eid]['lanes']} lanes {edges[eid]['length']:7.1f} m  {ramp_counts[eid]:>7,} pts")

    tot = {k: sum(c[k] for c in metrics["carriageways"].values() if not isinstance(c[k], list))
           for k in ("dead_modelled_m", "unmodelled_data_m", "aux_missing_m", "aux_empty_m")}
    metrics["totals"] = tot
    print("totals:", {k: round(v) for k, v in tot.items()}, "| empty ramp edges:", len(empty))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "a") as h:
            h.write(json.dumps(metrics) + "\n")


if __name__ == "__main__":
    main()
