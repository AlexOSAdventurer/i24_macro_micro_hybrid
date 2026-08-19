"""Build a SUMO network for the 1-mile I-24 corridor from the CARLA OpenDRIVE map.

The map CARLA serves (FinalMapI24Mini.xodr) is the authoritative geometry: the
CARLA coupler addresses vehicles as (road_id, lane_id, s) OpenDRIVE triples, so
converting that same file keeps SUMO and CARLA on one coordinate system.

Two things have to happen before netconvert will accept it:

1. Each road's center lane is written as ``<lane id="0">`` with no ``type``
   attribute.  The center lane is a zero-width boundary carrying only road
   markings -- never drivable -- and OpenDRIVE spells that ``type="none"``.
   CARLA tolerates the omission; SUMO's parser rejects it outright.  We patch a
   copy so the map CARLA loads is never touched.

2. ``--offset.disable-normalization`` keeps SUMO's x/y identical to the
   OpenDRIVE (and therefore CARLA) frame, so positions can be compared across
   the two simulators without an offset fixup.

Run inside the container (SUMO lives at ~/.local/bin there):

    python3.10 /workspaces/i24motion_macro_micro/sumo/build_i24_network.py
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_XODR = (
    "/workspaces/carla/Unreal/CarlaUE4/Content/Carla/Maps/"
    "FinalMapI24Mini/OpenDrive/FinalMapI24Mini.xodr"
)
DEFAULT_OUTPUT = os.path.join(HERE, "i24_corridor.net.xml")

# The corridor mainlines. Road 1 is eastbound, road 2 westbound; 3-6 are 25 m
# end stubs that exist only to give the mainlines somewhere to connect.
MAINLINE_ROADS = ("1", "2")

# <lane id="0"> and the self-closing <lane id="0" />, neither carrying a type.
CENTER_LANE_RE = re.compile(r'<lane id="0"\s*(/?)>')

# One whole <road>...</road> block.
ROAD_RE = re.compile(r'(<road id="(?P<id>[^"]+)"[^>]*>)(?P<body>.*?)(</road>)', re.S)

# The road-level speed record, e.g. <speed max="70" unit="mph"/> inside <type>.
ROAD_TYPE_SPEED_RE = re.compile(
    r'<type\b[^>]*>\s*<speed\b[^>]*\bmax="(?P<max>[^"]+)"'
    r'(?:[^>]*\bunit="(?P<unit>[^"]+)")?[^>]*/>',
    re.S,
)

# A driving lane element, opening tag through its closing </lane>.
DRIVING_LANE_RE = re.compile(
    r'(?P<open><lane id="-\d+" type="driving"[^>]*>)(?P<body>.*?)(?P<close></lane>)', re.S
)


def patch_center_lanes(xodr_text: str) -> tuple[str, int]:
    """Give every center lane an explicit ``type="none"``.

    The center lane is a zero-width boundary carrying only road markings, never
    drivable, so ``none`` is the spec-conformant type.  Returns the patched text
    and the number of substitutions made.
    """

    def repl(match: re.Match) -> str:
        self_closing = match.group(1)
        return f'<lane id="0" type="none" level="false"{" /" if self_closing else ""}>'

    patched, count = CENTER_LANE_RE.subn(repl, xodr_text)
    return patched, count


def patch_lane_speeds(xodr_text: str) -> tuple[str, dict[str, str]]:
    """Copy each road's ``<type><speed>`` down onto its driving lanes.

    SUMO's OpenDRIVE importer only honours a ``<speed>`` record whose parent
    element is ``<lane>`` (NIImporter_OpenDrive.cpp, OPENDRIVE_TAG_SPEED: it
    checks ``myElementStack.back() == OPENDRIVE_TAG_LANE``).  A speed declared
    under ``<road><type>`` -- which is where this map puts it -- is silently
    dropped, leaving every edge at SUMO's 13.89 m/s default.  Mirroring the
    value onto each driving lane keeps the limit sourced from the map instead
    of a hardcoded netconvert flag.

    Returns the patched text and a {road_id: "max unit"} report.
    """
    report: dict[str, str] = {}

    def patch_road(road_match: re.Match) -> str:
        road_id = road_match.group("id")
        body = road_match.group("body")

        speed_match = ROAD_TYPE_SPEED_RE.search(body)
        if speed_match is None:
            return road_match.group(0)

        speed_max = speed_match.group("max")
        unit = speed_match.group("unit")
        unit_attr = f' unit="{unit}"' if unit else ""
        report[road_id] = f"{speed_max} {unit or '(m/s)'}"

        record = f'<speed sOffset="0.0" max="{speed_max}"{unit_attr}/>'

        def patch_lane(lane_match: re.Match) -> str:
            if "<speed" in lane_match.group("body"):
                return lane_match.group(0)  # already lane-level; leave alone
            return (
                lane_match.group("open")
                + lane_match.group("body")
                + record
                + lane_match.group("close")
            )

        return (
            road_match.group(1)
            + DRIVING_LANE_RE.sub(patch_lane, body)
            + road_match.group(4)
        )

    return ROAD_RE.sub(patch_road, xodr_text), report


def build(xodr_path: str, output_path: str, curve_resolution: float) -> None:
    if not os.path.isfile(xodr_path):
        sys.exit(f"OpenDRIVE file not found: {xodr_path}")

    netconvert = shutil.which("netconvert")
    if netconvert is None:
        sys.exit(
            "netconvert not on PATH. Inside the container it lives at "
            "~/.local/bin/netconvert (pip install eclipse-sumo)."
        )

    with open(xodr_path, "r", encoding="utf-8") as handle:
        original = handle.read()

    patched, patched_count = patch_center_lanes(original)
    print(f"patched {patched_count} center lane(s) with type=\"none\"")
    if patched_count == 0:
        print("  (nothing to patch -- map may already be spec-conformant)")

    patched, speed_report = patch_lane_speeds(patched)
    if speed_report:
        print("propagated road <type><speed> onto driving lanes:")
        for road_id in sorted(speed_report, key=int):
            print(f"  road {road_id}: {speed_report[road_id]}")
    else:
        print("no road-level <type><speed> records found to propagate")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # The patched map is a 19 MB intermediate; keep it out of the repo.
    tmp_dir = tempfile.mkdtemp(prefix="i24_sumo_")
    tmp_xodr = os.path.join(tmp_dir, "FinalMapI24Mini.patched.xodr")
    try:
        with open(tmp_xodr, "w", encoding="utf-8") as handle:
            handle.write(patched)

        command = [
            netconvert,
            "--opendrive-files", tmp_xodr,
            "-o", output_path,
            # Keep SUMO's coordinates aligned with OpenDRIVE/CARLA.
            "--offset.disable-normalization", "true",
            # Record the originating OpenDRIVE ids as lane params.
            "--output.original-names", "true",
            # Driving lanes only -- no shoulders or sidewalks.
            "--opendrive.import-all-lanes", "false",
            "--opendrive.curve-resolution", str(curve_resolution),
            # Do not let netconvert merge or reshape the mainlines; road
            # identity has to survive so (road_id, lane_id, s) stays meaningful.
            "--geometry.remove", "false",
            "--geometry.min-radius.fix", "false",
            "--junctions.corner-detail", "0",
            "--no-turnarounds", "true",
        ]
        print("running:", " ".join(command))
        result = subprocess.run(command, capture_output=True, text=True)
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        if result.returncode != 0:
            sys.exit(f"netconvert failed with exit code {result.returncode}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xodr", default=DEFAULT_XODR, help="source OpenDRIVE map")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="output .net.xml")
    parser.add_argument(
        "--curve-resolution",
        type=float,
        default=1.0,
        help="netconvert --opendrive.curve-resolution, in meters",
    )
    args = parser.parse_args()
    build(args.xodr, args.output, args.curve_resolution)


if __name__ == "__main__":
    main()
