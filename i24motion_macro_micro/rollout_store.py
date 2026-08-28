"""Key-value store for `RolloutRenderer` time-space arrays.

A rollout's time-space data lives in `RolloutRenderer.ts_data`, keyed by
`(road_id, lane, version)` and holding one `(N_frames, N_cells)` array per
quantity. This module persists those arrays across many runs (optuna trials,
different dates, different models) in a single SQLite file, addressed by

    (run_id, road_id, lane, version, quantity)

Lookup is two B-tree probes plus a decompress, so a `get` is sub-millisecond
for the array sizes this project produces.

    store = RolloutStore("rollouts.db")
    run_id = store.put_renderer(renderer, metadata=trial.params)
    vel = store["trial-17", "2", -1, "sim", "velocity"]

Storage notes
-------------
* Arrays are cast to `dtype` (default float32; pass `dtype=None` to keep the
  source dtype for a lossless archive), byte-shuffled, then zstd-compressed.
* Blobs are content-addressed, so arrays that repeat across runs -- notably the
  `empirical` ground truth, which does not depend on calibrated parameters --
  are stored exactly once per database.
* Run metadata is JSON, queryable via `find()` / `runs_frame()`.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
import zlib
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # pyarrow ships a zstd codec; ~10x faster than zlib at a better ratio.
    import pyarrow as _pa
except ImportError:  # pragma: no cover - fallback path
    _pa = None


Key = Tuple[str, str, int, str, str]  # (run_id, road_id, lane, version, quantity)

DEFAULT_VERSIONS = ("sim", "empirical")
DEFAULT_QUANTITIES = ("density", "velocity")

_SCHEMA_VERSION = 1
_MISSING = object()  # `get(default=...)` sentinel; None is a legitimate default

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    metadata   TEXT NOT NULL DEFAULT '{}'
);

-- Content-addressed payloads. Identical arrays across runs collapse to one row.
CREATE TABLE IF NOT EXISTS blobs (
    digest      TEXT PRIMARY KEY,
    dtype       TEXT    NOT NULL,
    shape       TEXT    NOT NULL,
    codec       TEXT    NOT NULL,
    shuffle     INTEGER NOT NULL,
    raw_nbytes  INTEGER NOT NULL,
    data        BLOB    NOT NULL
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS arrays (
    run_id   TEXT    NOT NULL,
    road_id  TEXT    NOT NULL,
    lane     INTEGER NOT NULL,
    version  TEXT    NOT NULL,
    quantity TEXT    NOT NULL,
    digest   TEXT    NOT NULL,
    PRIMARY KEY (run_id, road_id, lane, version, quantity)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS arrays_by_digest ON arrays (digest);

-- Per (run, road, lane) coordinates shared by every quantity/version.
CREATE TABLE IF NOT EXISTS axes (
    run_id       TEXT    NOT NULL,
    road_id      TEXT    NOT NULL,
    lane         INTEGER NOT NULL,
    s_mids       TEXT    NOT NULL,
    sim_times    TEXT    NOT NULL,
    mask_extents TEXT    NOT NULL,
    PRIMARY KEY (run_id, road_id, lane)
) WITHOUT ROWID;
"""


# --------------------------------------------------------------------------
# codec
# --------------------------------------------------------------------------

def _shuffle(buf: bytes, itemsize: int) -> bytes:
    """Group the i-th byte of every element together.

    Smooth fields (density, velocity) share exponent and high mantissa bytes
    across neighbours, so this lifts the zstd ratio ~10-15% and is *faster*
    than compressing the interleaved bytes.
    """
    if itemsize <= 1:
        return buf
    return np.frombuffer(buf, dtype=np.uint8).reshape(-1, itemsize).T.copy().tobytes()


def _unshuffle(buf: bytes, itemsize: int) -> bytes:
    if itemsize <= 1:
        return buf
    return np.frombuffer(buf, dtype=np.uint8).reshape(itemsize, -1).T.copy().tobytes()


def _encode(array: np.ndarray) -> Tuple[bytes, str, str, str, int, int, bytes]:
    """-> (digest_bytes_source, dtype, shape_json, codec, shuffle, raw_nbytes, payload)."""
    arr = np.ascontiguousarray(array)
    raw = arr.tobytes()
    itemsize = arr.dtype.itemsize
    staged = _shuffle(raw, itemsize)
    if _pa is not None:
        payload = _pa.compress(staged, codec="zstd", asbytes=True)
        codec = "zstd"
    else:
        payload = zlib.compress(staged, 1)
        codec = "zlib"
    if len(payload) >= len(raw):  # incompressible -- skip the decode work later
        payload, codec, shuffle = raw, "raw", 0
    else:
        shuffle = itemsize
    return (
        raw,
        arr.dtype.str,
        json.dumps(list(arr.shape)),
        codec,
        shuffle,
        len(raw),
        payload,
    )


def _decode(data: bytes, dtype: str, shape: str, codec: str, shuffle: int, raw_nbytes: int) -> np.ndarray:
    if codec == "zstd":
        staged = _pa.decompress(data, decompressed_size=raw_nbytes, codec="zstd", asbytes=True)
    elif codec == "zlib":
        staged = zlib.decompress(data)
    else:
        staged = data
    raw = _unshuffle(staged, shuffle) if shuffle else staged
    out = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(json.loads(shape))
    out.flags.writeable = False  # decoded blobs may be handed out from cache
    return out


def _digest(raw: bytes) -> str:
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------

class RolloutStore:
    """SQLite-backed key-value store for rollout time-space arrays."""

    def __init__(
        self,
        path: str,
        *,
        dtype: Optional[Any] = np.float32,
        readonly: bool = False,
        cache_size: int = 64,
        timeout: float = 30.0,
    ) -> None:
        """
        Parameters
        ----------
        path : str
            SQLite file. Created (with parent directories) if absent.
        dtype : numpy dtype or None
            Arrays are cast to this on write. ``np.float32`` halves the file at
            ~1e-7 relative error, which is far below the noise in the fields
            themselves. Pass ``None`` for a lossless archive.
        readonly : bool
            Open in SQLite read-only mode. Safe alongside a writing process.
        cache_size : int
            Decoded arrays held in an in-process LRU, keyed by content digest.
            Set to 0 to disable.
        """
        self.path = str(path)
        self.dtype = None if dtype is None else np.dtype(dtype)
        self.readonly = bool(readonly)

        parent = os.path.dirname(os.path.abspath(self.path))
        if parent and not readonly:
            os.makedirs(parent, exist_ok=True)

        if readonly:
            uri = f"file:{os.path.abspath(self.path)}?mode=ro"
            self._db = sqlite3.connect(uri, uri=True, timeout=timeout)
        else:
            self._db = sqlite3.connect(self.path, timeout=timeout)
        self._db.row_factory = sqlite3.Row

        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
        self._db.execute("PRAGMA mmap_size=1073741824")  # 1 GiB of mmapped reads
        if not readonly:
            self._db.executescript(_SCHEMA)
            self._db.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(_SCHEMA_VERSION),),
            )
            self._db.commit()

        self._cache_size = int(cache_size)
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "RolloutStore":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # -- writing -----------------------------------------------------------

    def put_renderer(
        self,
        renderer: Any,
        run_id: Optional[str] = None,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        road_lanes: Optional[Sequence[Tuple[str, int]]] = None,
        versions: Sequence[str] = DEFAULT_VERSIONS,
        quantities: Sequence[str] = DEFAULT_QUANTITIES,
        overwrite: bool = False,
    ) -> str:
        """Store one rollout's time-space arrays. Returns the run_id.

        Lanes default to ``renderer.ts_road_lanes``; each is materialised via
        ``_ensure_ts_lane`` before being read, so a freshly built renderer works
        without touching the Dash figures. Versions absent from the renderer
        (e.g. ``empirical`` on a pure-macro run with no ground-truth store) are
        skipped silently.
        """
        self._require_writable()
        run_id = run_id or uuid.uuid4().hex[:16]
        if not overwrite and self.has_run(run_id):
            raise KeyError(f"run_id {run_id!r} already exists (pass overwrite=True to replace)")

        pairs = list(road_lanes) if road_lanes is not None else list(renderer.ts_road_lanes)

        with self._db:  # one transaction: a failed rollout writes nothing
            if overwrite:
                self._delete_run_rows(run_id)
            self._db.execute(
                "INSERT OR REPLACE INTO runs (run_id, created_at, metadata) VALUES (?, ?, ?)",
                (run_id, time.time(), json.dumps(_jsonable(metadata or {}))),
            )

            sim_times = [float(t) for t in getattr(renderer, "sim_times", [])]
            for road_id, lane in pairs:
                renderer._ensure_ts_lane(road_id, int(lane))
                stored_axes = False
                for version in versions:
                    entry = renderer.ts_data.get((road_id, int(lane), version))
                    if entry is None:
                        continue
                    if not stored_axes:
                        self._db.execute(
                            "INSERT OR REPLACE INTO axes "
                            "(run_id, road_id, lane, s_mids, sim_times, mask_extents) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                run_id,
                                str(road_id),
                                int(lane),
                                json.dumps([float(s) for s in entry.get("s_mids", [])]),
                                json.dumps(sim_times),
                                json.dumps(_jsonable(entry.get("mask_extents", []))),
                            ),
                        )
                        stored_axes = True
                    for quantity in quantities:
                        values = entry.get(quantity)
                        if values is None:
                            continue
                        self._put_array_row(
                            run_id, str(road_id), int(lane), str(version), str(quantity), values
                        )
        return run_id

    def put_array(
        self,
        run_id: str,
        road_id: str,
        lane: int,
        version: str,
        quantity: str,
        array: np.ndarray,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a single array under an explicit key, creating the run if needed."""
        self._require_writable()
        with self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO runs (run_id, created_at, metadata) VALUES (?, ?, ?)",
                (run_id, time.time(), json.dumps(_jsonable(metadata or {}))),
            )
            if metadata:
                self._db.execute(
                    "UPDATE runs SET metadata = ? WHERE run_id = ?",
                    (json.dumps(_jsonable(metadata)), run_id),
                )
            self._put_array_row(run_id, str(road_id), int(lane), str(version), str(quantity), array)

    def _put_array_row(
        self, run_id: str, road_id: str, lane: int, version: str, quantity: str, array: np.ndarray
    ) -> None:
        arr = np.asarray(array)
        if self.dtype is not None and arr.dtype != self.dtype:
            arr = arr.astype(self.dtype, copy=False)
        raw, dtype, shape, codec, shuffle, nbytes, payload = _encode(arr)
        digest = _digest(raw)
        self._db.execute(
            "INSERT OR IGNORE INTO blobs "
            "(digest, dtype, shape, codec, shuffle, raw_nbytes, data) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (digest, dtype, shape, codec, shuffle, nbytes, payload),
        )
        self._db.execute(
            "INSERT OR REPLACE INTO arrays "
            "(run_id, road_id, lane, version, quantity, digest) VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, road_id, lane, version, quantity, digest),
        )

    def set_metadata(self, run_id: str, metadata: Dict[str, Any], *, merge: bool = True) -> None:
        """Attach metadata to an existing run (e.g. the objective value, once known)."""
        self._require_writable()
        current = self.metadata(run_id) if merge else {}
        current.update(_jsonable(metadata))
        with self._db:
            self._db.execute(
                "UPDATE runs SET metadata = ? WHERE run_id = ?", (json.dumps(current), run_id)
            )

    # -- reading -----------------------------------------------------------

    def get(
        self,
        run_id: str,
        road_id: str,
        lane: int,
        version: str = "sim",
        quantity: str = "density",
        default: Any = _MISSING,
    ) -> np.ndarray:
        """Return the array for one key. Read-only; copy it before mutating."""
        # Resolve the digest first: this probe is index-only on the WITHOUT ROWID
        # primary key, so a cache hit never touches the (megabyte) BLOB page.
        row = self._db.execute(
            "SELECT digest FROM arrays WHERE run_id = ? AND road_id = ? AND lane = ? "
            "AND version = ? AND quantity = ?",
            (str(run_id), str(road_id), int(lane), str(version), str(quantity)),
        ).fetchone()
        if row is None:
            if default is not _MISSING:
                return default
            raise KeyError((run_id, road_id, lane, version, quantity))

        digest = row[0]
        cached = self._cache.get(digest)
        if cached is not None:
            self._cache.move_to_end(digest)
            return cached

        blob = self._db.execute(
            "SELECT dtype, shape, codec, shuffle, raw_nbytes, data FROM blobs WHERE digest = ?",
            (digest,),
        ).fetchone()
        if blob is None:  # only reachable if a vacuum raced an in-flight read
            raise KeyError((run_id, road_id, lane, version, quantity))
        out = _decode(
            blob["data"], blob["dtype"], blob["shape"], blob["codec"],
            blob["shuffle"], blob["raw_nbytes"],
        )
        if self._cache_size:
            self._cache[digest] = out
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return out

    def __getitem__(self, key: Key) -> np.ndarray:
        return self.get(*key)

    def __contains__(self, key: Key) -> bool:
        run_id, road_id, lane, version, quantity = key
        row = self._db.execute(
            "SELECT 1 FROM arrays WHERE run_id = ? AND road_id = ? AND lane = ? "
            "AND version = ? AND quantity = ?",
            (str(run_id), str(road_id), int(lane), str(version), str(quantity)),
        ).fetchone()
        return row is not None

    def axes(self, run_id: str, road_id: str, lane: int) -> Dict[str, Any]:
        """`s_mids`, `sim_times` and `mask_extents` for one (run, road, lane)."""
        row = self._db.execute(
            "SELECT s_mids, sim_times, mask_extents FROM axes "
            "WHERE run_id = ? AND road_id = ? AND lane = ?",
            (str(run_id), str(road_id), int(lane)),
        ).fetchone()
        if row is None:
            raise KeyError((run_id, road_id, lane))
        return {
            "s_mids": np.asarray(json.loads(row["s_mids"]), dtype=float),
            "sim_times": np.asarray(json.loads(row["sim_times"]), dtype=float),
            "mask_extents": json.loads(row["mask_extents"]),
        }

    def metadata(self, run_id: str) -> Dict[str, Any]:
        row = self._db.execute("SELECT metadata FROM runs WHERE run_id = ?", (str(run_id),)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return json.loads(row["metadata"])

    def has_run(self, run_id: str) -> bool:
        return self._db.execute(
            "SELECT 1 FROM runs WHERE run_id = ?", (str(run_id),)
        ).fetchone() is not None

    def runs(self) -> List[str]:
        return [r[0] for r in self._db.execute("SELECT run_id FROM runs ORDER BY created_at")]

    def keys(self, run_id: Optional[str] = None) -> List[Key]:
        sql = "SELECT run_id, road_id, lane, version, quantity FROM arrays"
        args: Tuple[Any, ...] = ()
        if run_id is not None:
            sql += " WHERE run_id = ?"
            args = (str(run_id),)
        return [tuple(r) for r in self._db.execute(sql + " ORDER BY run_id, road_id, lane", args)]

    def find(self, **equals: Any) -> List[str]:
        """Run ids whose metadata matches every ``field=value`` pair.

        Nested fields use dotted paths: ``store.find(**{"params.v_f": 38.0})``.
        """
        if not equals:
            return self.runs()
        clauses, args = [], []
        for field, value in equals.items():
            clauses.append(f"json_extract(metadata, '$.{field}') = ?")
            args.append(value)
        sql = f"SELECT run_id FROM runs WHERE {' AND '.join(clauses)} ORDER BY created_at"
        return [r[0] for r in self._db.execute(sql, args)]

    def runs_frame(self):
        """All runs as a pandas DataFrame with metadata flattened into columns."""
        import pandas as pd

        rows = []
        for r in self._db.execute("SELECT run_id, created_at, metadata FROM runs ORDER BY created_at"):
            row = {"run_id": r["run_id"], "created_at": r["created_at"]}
            row.update(_flatten(json.loads(r["metadata"])))
            rows.append(row)
        return pd.DataFrame(rows)

    # -- maintenance -------------------------------------------------------

    def delete_run(self, run_id: str) -> None:
        """Drop a run's keys. Payloads survive until `vacuum()` (they may be shared)."""
        self._require_writable()
        with self._db:
            self._delete_run_rows(run_id)

    def _delete_run_rows(self, run_id: str) -> None:
        self._db.execute("DELETE FROM arrays WHERE run_id = ?", (str(run_id),))
        self._db.execute("DELETE FROM axes WHERE run_id = ?", (str(run_id),))
        self._db.execute("DELETE FROM runs WHERE run_id = ?", (str(run_id),))

    def vacuum(self) -> int:
        """Delete unreferenced blobs and compact the file. Returns blobs removed."""
        self._require_writable()
        with self._db:
            cur = self._db.execute(
                "DELETE FROM blobs WHERE digest NOT IN (SELECT digest FROM arrays)"
            )
            removed = cur.rowcount
        self._cache.clear()
        self._db.execute("VACUUM")
        return removed

    def stats(self) -> Dict[str, Any]:
        n_runs = self._db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        n_keys = self._db.execute("SELECT COUNT(*) FROM arrays").fetchone()[0]
        n_blobs, stored, logical = self._db.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(data)), 0), COALESCE(SUM(raw_nbytes), 0) FROM blobs"
        ).fetchone()
        referenced = self._db.execute(
            "SELECT COALESCE(SUM(b.raw_nbytes), 0) FROM arrays a JOIN blobs b ON b.digest = a.digest"
        ).fetchone()[0]
        return {
            "runs": n_runs,
            "keys": n_keys,
            "blobs": n_blobs,
            "file_bytes": os.path.getsize(self.path) if os.path.exists(self.path) else 0,
            "stored_bytes": stored,
            "unique_bytes": logical,
            "logical_bytes": referenced,
            "dedup_ratio": (referenced / logical) if logical else 1.0,
            "compression_ratio": (logical / stored) if stored else 1.0,
        }

    def _require_writable(self) -> None:
        if self.readonly:
            raise RuntimeError("store opened readonly")

    def __repr__(self) -> str:  # pragma: no cover
        return f"RolloutStore({self.path!r}, runs={len(self.runs())})"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    """Coerce numpy scalars/arrays so metadata round-trips through JSON."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def _flatten(obj: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in obj.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, f"{name}."))
        else:
            out[name] = value
    return out
