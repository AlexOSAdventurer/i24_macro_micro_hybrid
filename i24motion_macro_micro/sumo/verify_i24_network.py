"""Verify the converted SUMO corridor against the source OpenDRIVE map.

Checks the invariants the macro/micro coupling depends on:

  * each OpenDRIVE road became exactly one SUMO edge, with 4 driving lanes
    (i.e. the zero-width center lane did not leak in as a drivable lane);
  * edge lengths still match the OpenDRIVE road lengths;
  * the speed limit survived the import (SUMO's importer drops road-level
    <type><speed>, so this catches a regression in the propagation patch);
  * the (road_id, lane_id) -> SUMO lane mapping is the expected
    ``sumo_index = n_lanes + lane_id``.

Run inside the container:

    python3.10 /workspaces/i24motion_macro_micro/sumo/verify_i24_network.py
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import sumolib

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_XODR = (
    "/workspaces/carla/Unreal/CarlaUE4/Content/Carla/Maps/"
    "FinalMapI24Mini/OpenDrive/FinalMapI24Mini.xodr"
)
DEFAULT_NET = os.path.join(HERE, "i24_corridor.net.xml")

MAINLINE_ROADS = ("1", "2")
EXPECTED_LANES = 4
LENGTH_TOLERANCE = 0.5  # meters; curve-resolution sampling loses a little
SPEED_TOLERANCE = 0.05  # m/s

ROAD_RE = re.compile(r'<road id="(?P<id>[^"]+)"[^>]*\blength="(?P<length>[^"]+)"')
ROAD_SPEED_RE = re.compile(
    r'<road id="(?P<id>[^"]+)".*?<type\b[^>]*>\s*<speed\b[^>]*\bmax="(?P<max>[^"]+)"'
    r'(?:[^>]*\bunit="(?P<unit>[^"]+)")?[^>]*/>',
    re.S,
)

UNIT_TO_MPS = {
    "mph": 1609.344 / 3600.0,
    "km/h": 1000.0 / 3600.0,
    "m/s": 1.0,
}


def read_opendrive(xodr_path: str) -> tuple[dict[str, float], dict[str, float]]:
    with open(xodr_path, "r", encoding="utf-8") as handle:
        text = handle.read()

    lengths = {m.group("id"): float(m.group("length")) for m in ROAD_RE.finditer(text)}

    speeds: dict[str, float] = {}
    for match in ROAD_SPEED_RE.finditer(text):
        unit = match.group("unit") or "m/s"
        factor = UNIT_TO_MPS.get(unit)
        if factor is None:
            continue
        speeds[match.group("id")] = float(match.group("max")) * factor
    return lengths, speeds


def check(net_path: str, xodr_path: str) -> int:
    net = sumolib.net.readNet(net_path)
    lengths, speeds = read_opendrive(xodr_path)

    failures: list[str] = []

    print(f"OpenDRIVE : {xodr_path}")
    print(f"SUMO net  : {net_path}")
    print()

    edge_ids = sorted(e.getID() for e in net.getEdges())
    print(f"edges: {edge_ids}")
    print()

    header = f"{'road':>5} {'edge':>6} {'lanes':>6} {'xodr len':>10} {'sumo len':>10} {'d len':>8} {'speed m/s':>10} {'mph':>7}"
    print(header)
    print("-" * len(header))

    for road_id in sorted(lengths, key=int):
        edge_id = "-" + road_id  # right-hand lanes become the negative edge
        try:
            edge = net.getEdge(edge_id)
        except KeyError:
            failures.append(f"road {road_id}: expected edge {edge_id!r}, not found")
            continue

        lane_count = len(edge.getLanes())
        xodr_length = lengths[road_id]
        sumo_length = edge.getLength()
        delta = sumo_length - xodr_length
        speed = edge.getSpeed()

        print(
            f"{road_id:>5} {edge_id:>6} {lane_count:>6} {xodr_length:>10.3f} "
            f"{sumo_length:>10.3f} {delta:>8.3f} {speed:>10.4f} {speed * 2.2369362920544:>7.2f}"
        )

        if lane_count != EXPECTED_LANES:
            failures.append(
                f"road {road_id}: expected {EXPECTED_LANES} drivable lanes, got {lane_count}"
                " (a non-drivable center lane may have been imported)"
            )
        if abs(delta) > LENGTH_TOLERANCE:
            failures.append(
                f"road {road_id}: length drift {delta:+.3f} m exceeds {LENGTH_TOLERANCE} m"
            )
        expected_speed = speeds.get(road_id)
        if expected_speed is not None and abs(speed - expected_speed) > SPEED_TOLERANCE:
            failures.append(
                f"road {road_id}: speed {speed:.4f} m/s != OpenDRIVE {expected_speed:.4f} m/s"
                " (road-level <type><speed> was probably dropped on import)"
            )

    print()
    print("=== (road_id, lane_id) -> SUMO lane, mainlines ===")
    for road_id in MAINLINE_ROADS:
        edge = net.getEdge("-" + road_id)
        lanes = edge.getLanes()
        for lane in lanes:
            orig = lane.getParam("origId")
            index = lane.getIndex()
            expected_orig = f"{road_id}_{index - len(lanes)}"
            flag = "" if orig == expected_orig else f"  <-- expected origId {expected_orig}"
            print(
                f"  road {road_id} lane {index - len(lanes):>3}  ->  {lane.getID():<6} "
                f"(index {index}, origId {orig}){flag}"
            )
            if orig != expected_orig:
                failures.append(
                    f"road {road_id} lane index {index}: origId {orig!r} breaks the"
                    f" sumo_index = n_lanes + lane_id mapping"
                )
    print()
    print("  mapping rule: sumo_index = n_lanes + lane_id   (lane -1 -> index 3, lane -4 -> index 0)")

    print()
    if failures:
        print(f"FAILED ({len(failures)} problem(s)):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("All checks passed.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", default=DEFAULT_NET)
    parser.add_argument("--xodr", default=DEFAULT_XODR)
    args = parser.parse_args()
    sys.exit(check(args.net, args.xodr))


if __name__ == "__main__":
    main()
