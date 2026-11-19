"""Draw a uniform random sample of I-24 MOTION trajectories for map evaluation.

Part of the full-corridor pipeline (``sumo/full``). The existing 1-mile
pipeline is untouched.

Each day file is one JSON array of trajectory records (5-26 GB). Records are
located by grepping the byte offset of every top-level ``"_id"`` key, then
``--per-day`` of them are drawn at random (fixed seed) and parsed. Every
``--stride``-th point is kept (25 Hz -> 5 Hz at the default stride of 5).

Output: one compressed .npz with flat arrays over all sampled points
  x    x_position (ft, MOTION frame; /5280 = mile marker)
  y    y_position (ft lateral; EB negative, WB positive)
  t    timestamp (s, epoch)
  day  index into ``days``
  traj running trajectory index (so points can be regrouped per vehicle)
  days the day names

Usage::

    python3 sample_i24_trajectories.py            # writes eval/traj_sample.npz
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_DATA = os.path.join(PROJECT, "i24_motion_source_data")
DEFAULT_OUT = os.path.join(HERE, "eval", "traj_sample.npz")
DAYS = ["2022-11-21", "2022-11-22", "2022-11-23", "2022-11-24", "2022-11-25",
        "2022-11-28", "2022-11-29", "2022-11-30", "2022-12-01", "2022-12-02"]


def record_offsets(path: str):
    res = subprocess.run(["grep", "-b", "-o", '^  "_id"', path], capture_output=True, text=True, check=True)
    return [int(line.split(":", 1)[0]) for line in res.stdout.splitlines()]


def read_record(fh, start: int, end: int) -> dict:
    fh.seek(start)
    raw = fh.read(end - start).decode()
    # The slice runs from this record's "_id" to the next one's; trim the
    # trailing "}, {" so it parses as one object.
    return json.loads("{" + raw[: raw.rstrip().rstrip("{").rstrip().rstrip(",").rstrip().rfind("}") + 1])


def sample_day(job):
    path, per_day, stride, seed = job
    rng = random.Random(seed)
    offs = record_offsets(path)
    size = os.path.getsize(path)
    picks = sorted(rng.sample(range(len(offs)), min(per_day, len(offs))))
    xs, ys, ts, tr = [], [], [], []
    with open(path, "rb") as fh:
        for k, i in enumerate(picks):
            rec = read_record(fh, offs[i], offs[i + 1] if i + 1 < len(offs) else size)
            x = np.asarray(rec["x_position"][::stride], float)
            xs.append(x)
            ys.append(np.asarray(rec["y_position"][::stride], float))
            ts.append(np.asarray(rec["timestamp"][::stride], float))
            tr.append(np.full(len(x), k, np.int32))
    print(f"sampled {os.path.basename(path)}: {len(picks)} of {len(offs)} records", flush=True)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ts), np.concatenate(tr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DEFAULT_DATA)
    p.add_argument("--per-day", type=int, default=2000)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--output", default=DEFAULT_OUT)
    args = p.parse_args()

    jobs = [(os.path.join(args.data_dir, f"{d}.json"), args.per_day, args.stride, args.seed + i)
            for i, d in enumerate(DAYS)]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(sample_day, jobs))
    x, y, t, traj, day = [], [], [], [], []
    offset = 0
    for i, (xs, ys, ts, tr) in enumerate(results):
        x.append(xs), y.append(ys), t.append(ts)
        traj.append(tr + offset)
        offset += int(tr.max()) + 1 if len(tr) else 0
        day.append(np.full(len(xs), i, np.int8))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, x=np.concatenate(x), y=np.concatenate(y), t=np.concatenate(t),
                        traj=np.concatenate(traj), day=np.concatenate(day), days=np.array(DAYS))
    print(f"wrote {args.output}: {sum(len(a) for a in x):,} points from {offset:,} trajectories")


if __name__ == "__main__":
    main()
