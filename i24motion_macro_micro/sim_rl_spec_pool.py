"""A fixed pool of episode specs with their counterfactuals precomputed.

``I24SumoHeroEnv.reset`` spends about 77% of its time in ``_baseline_pass``,
which builds a whole second episode and replays it with the hero handed back to
SUMO.  That is the counterfactual every differenced reward term is scored
against.  Because ``SubprocVecEnv`` is synchronous, that cost does not just slow
one worker down -- the other twenty-three block on it, which is why the box sits
near idle while training crawls.

The counterfactual is worth precomputing because of two properties:

* **It is deterministic given the spec.**  ``_build_episode`` reseeds the global
  RNG the spawn stream draws from with ``spec.sumo_seed``, so a stored
  counterfactual is bit-identical to one computed live.  This is the same
  property the acceptance test rests on -- it is why ``sumo`` scores exactly
  0.000000 on every term.
* **It depends on ``EnvConfig`` alone, never on ``RewardConfig``.**  It is only
  "what the road did with nobody driving", so one pool survives every reward
  weighting you might want to compare.

The cost is that training then draws from a fixed set of scenarios rather than a
fresh one each episode.  That is a real change, but the status quo is the
pathological end of the same axis: with roughly 1,200 distinct start times and
17,000 episodes in a run, the policy currently sees almost every situation
exactly once, which leaves a recurrent policy nothing to generalise *from*.  The
pool size sets where you sit -- 500 specs over a 1M-step run is ~35 repeats.

A pool also makes a held-out evaluation set possible for the first time:
``holdout`` reserves a deterministic slice that training never samples.

A pool can also be SPECS ONLY (``counterfactuals=None``, the CLI default).  The
stored counterfactual is just densities and rear fluxes -- enough for differenced
reward terms, useless for comparing controllers -- so an evaluation that runs its
baselines live (sim_rl_sumo_demo.py replays ``sumo``/``idm`` on the same specs)
needs nothing but the spec list.  Such a pool skips the fingerprint check, and
``counterfactual()`` returns None, so any reward that does difference computes its
baseline live exactly as it would for a spec outside the pool.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_POOL_PATH = os.path.join(HERE, "run_data", "rl", "spec_data", "data_full_no_counterfactual.pkl.gz")
DEFAULT_CONFIG_FOLDER = os.path.join(HERE, "config")

# EnvConfig fields that cannot change what a do-nothing rollout does.  Everything
# else is fingerprinted, so the default on an unrecognised new field is to
# invalidate the pool rather than to silently reuse a stale counterfactual.
COUNTERFACTUAL_IRRELEVANT = frozenset({
    "gui",
    "verbose",
    # The pool settings themselves, or the fingerprint is circular: pointing a
    # config at a pool would change the hash and the pool could never load.
    "spec_pool",
    "spec_pool_holdout",
    "spec_pool_use_holdout",
    # Observation shape only -- the counterfactual has no policy reading it.
    "macro_lookahead_cells",
    "macro_lookbehind_cells",
    "platoon_window",
    # Action limits only -- the counterfactual issues no actions.
    "max_acceleration",
    "max_deceleration",
    # Observation SCALING only -- same reasoning as the shape fields above.  These
    # select which ring arm's normalisers the agent sees (10.0 for published AVC,
    # v_f for the corridor-matched arm) and cannot alter a do-nothing rollout.
    "obs_max_speed",
    "obs_max_dist",
})

POOL_FORMAT_VERSION = 2


# ---------------------------------------------------------------------------
# Compact storage
#
# A counterfactual as _baseline_pass returns it is a list of ~40 dicts keyed by
# (road_id, cell_id) over ~128 cells, plus a list of ~40 dicts keyed by lane.
# That is about 5,000 dict entries per spec and measured at 1,055 KB resident --
# 38x its own compressed size, and every one of the 24 workers holds a copy, so
# a 2,000-spec pool would want ~49 GB across the fleet.
#
# The keys, though, are the same at every step and for every spec.  Hoisting
# them into one index per pool and keeping the values as float64 arrays takes a
# spec from ~1,055 KB to ~41 KB, which is what makes a pool of thousands rather
# than hundreds affordable.  The two view classes below give back exactly the
# dict interface the reward code uses -- `.get((road, cell))` on densities, and
# iteration plus `[lane]` on fluxes -- so nothing downstream changes.
# ---------------------------------------------------------------------------


class _DensitySnapshot:
    """One step's base-cell densities, backed by a row of a float64 array."""

    __slots__ = ("_index", "_row")

    def __init__(self, index: Dict[Tuple[str, str], int], row) -> None:
        self._index = index
        self._row = row

    def get(self, key, default=None):
        column = self._index.get(key)
        if (column is None):
            return default
        value = self._row[column]
        # NaN marks a cell this spec never had, which .get must report as absent
        # rather than as a density of nan -- _upstream_profiles skips on None.
        return default if (value != value) else float(value)

    def __contains__(self, key) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        return len(self._index)


class _FluxSnapshot:
    """One step's per-lane rear flux, backed by a row of a float64 array."""

    __slots__ = ("_lanes", "_row")

    def __init__(self, lanes: Dict[int, int], row) -> None:
        self._lanes = lanes
        self._row = row

    def __iter__(self):
        return iter(self._lanes)

    def __getitem__(self, lane: int) -> float:
        return float(self._row[self._lanes[lane]])

    def __len__(self) -> int:
        return len(self._lanes)


class _SnapshotSequence(Sequence):
    """Lazy list of per-step views over a (steps, keys) array."""

    __slots__ = ("_index", "_array", "_factory")

    def __init__(self, index, array, factory) -> None:
        self._index = index
        self._array = array
        self._factory = factory

    def __len__(self) -> int:
        return int(self._array.shape[0])

    def __getitem__(self, position):
        if isinstance(position, slice):
            raise TypeError("counterfactual snapshots are indexed by step, not sliced")
        return self._factory(self._index, self._array[position])


def _compact(
    counterfactual: Tuple[List[Dict], List[Dict]],
    cell_index: Dict[Tuple[str, str], int],
    lane_index: Dict[int, int],
):
    """Dict-of-dicts counterfactual -> (densities, fluxes) float64 arrays."""
    maps, fluxes = counterfactual
    density = np.full((len(maps), len(cell_index)), np.nan, dtype=np.float64)
    for step, snapshot in enumerate(maps):
        for key, value in snapshot.items():
            density[step, cell_index[key]] = value
    flux = np.full((len(fluxes), len(lane_index)), np.nan, dtype=np.float64)
    for step, snapshot in enumerate(fluxes):
        for lane, value in snapshot.items():
            flux[step, lane_index[lane]] = value
    return density, flux


def counterfactual_fingerprint(env_config) -> str:
    """Hash of every EnvConfig field that can alter a do-nothing rollout.

    Deliberately conservative: fields are included unless they are in
    ``COUNTERFACTUAL_IRRELEVANT``.  A pool built under different bubble geometry
    or a different vehicle type would produce counterfactuals for a road that is
    not the one being scored, and every differenced reward term would be quietly
    wrong -- so a mismatch is a hard error, never a warning.
    """
    fields = {
        key: value for key, value in asdict(env_config).items()
        if key not in COUNTERFACTUAL_IRRELEVANT
    }
    blob = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _spec_key(spec) -> Tuple[str, str, float, int]:
    """Identity of a spec, for looking its counterfactual up."""
    return (str(spec.dataset), str(spec.road), float(spec.start_time), int(spec.sumo_seed))


class SpecPoolMismatch(RuntimeError):
    """The pool on disk was built for a different environment."""


class SpecPool:
    """Specs plus their counterfactuals, keyed by spec identity."""

    def __init__(
        self,
        fingerprint: str,
        specs: Sequence[Any],
        counterfactuals: Optional[Sequence[Tuple[List[Dict], List[Dict]]]],
        env_fields: Optional[Dict[str, Any]] = None,
        cell_index: Optional[Dict[Tuple[str, str], int]] = None,
        lane_index: Optional[Dict[int, int]] = None,
        provenance: Optional[Sequence[str]] = None,
    ) -> None:
        # None means a specs-only pool: see the module docstring.
        self.has_counterfactuals = counterfactuals is not None
        if (self.has_counterfactuals and (len(specs) != len(counterfactuals))):
            raise ValueError("specs and counterfactuals must be the same length")
        self.fingerprint = str(fingerprint)
        self.specs = list(specs)
        self.env_fields = dict(env_fields or {})
        # How this pool was derived from its parent, one line per filter applied.  A
        # filtered pool is otherwise indistinguishable from a sampled one on disk.
        self.provenance = list(provenance or [])

        counterfactuals = list(counterfactuals) if self.has_counterfactuals else [None] * len(self.specs)
        already_compact = (
            self.has_counterfactuals and bool(counterfactuals) and isinstance(counterfactuals[0][0], np.ndarray)
        )
        if (not self.has_counterfactuals):
            self.cell_index, self.lane_index = {}, {}
            arrays = counterfactuals
        elif already_compact:
            if (cell_index is None) or (lane_index is None):
                raise ValueError("compact counterfactuals need their key indices")
            self.cell_index, self.lane_index = dict(cell_index), dict(lane_index)
            arrays = counterfactuals
        else:
            # One key index for the whole pool: the cells and lanes are the same
            # at every step and for every spec, so storing them per entry is what
            # made the dict form 38x its compressed size.
            cells: Dict[Tuple[str, str], int] = {}
            lanes: Dict[int, int] = {}
            for maps, fluxes in counterfactuals:
                for snapshot in maps:
                    for key in snapshot:
                        cells.setdefault(key, len(cells))
                for snapshot in fluxes:
                    for lane in snapshot:
                        lanes.setdefault(lane, len(lanes))
            self.cell_index, self.lane_index = cells, lanes
            arrays = [_compact(c, cells, lanes) for c in counterfactuals]

        self._by_key = {
            _spec_key(spec): entry for spec, entry in zip(self.specs, arrays)
        }
        # Specs grouped by the ground-truth band they need.  Sampling uniformly
        # at random across the whole pool looks harmless and is not: consecutive
        # episodes would then almost never share a band, and each reset would pay
        # a ~3 s parquet read.  Measured, that gave back the entire saving --
        # reset stayed at 4.4 s instead of dropping to 1.3 s.
        self._groups: Dict[Tuple[str, str, float, float], List[Any]] = {}
        for spec in self.specs:
            key = (str(spec.dataset), str(spec.road), float(spec.band_start), float(spec.band_end))
            self._groups.setdefault(key, []).append(spec)
        self._group_keys = sorted(self._groups)
        self._current_group: Optional[Tuple[str, str, float, float]] = None
        self._group_uses = 0

    def __len__(self) -> int:
        return len(self.specs)

    @property
    def band_count(self) -> int:
        return len(self._group_keys)

    def counterfactual(self, spec) -> Optional[Tuple[Sequence, Sequence]]:
        """Stored counterfactual for ``spec``, or None if it is not in the pool.

        Returns per-step views rather than dicts, but they answer the same calls
        the reward makes (``.get((road, cell))``, iteration and ``[lane]``), so
        callers cannot tell the difference.

        Returning None rather than raising matters: the demo and the analysis
        harnesses pin arbitrary specs, and those must still fall back to
        computing the counterfactual live.
        """
        entry = self._by_key.get(_spec_key(spec))
        if (entry is None):             # not in the pool, or a specs-only pool
            return None
        densities, fluxes = entry
        return (
            _SnapshotSequence(self.cell_index, densities, _DensitySnapshot),
            _SnapshotSequence(self.lane_index, fluxes, _FluxSnapshot),
        )

    def sample(self, rng, episodes_per_band: int = 4) -> Any:
        """Draw a spec, staying within one ground-truth band while it lasts.

        Mirrors ``_sample_spec``'s band reuse, for the same reason: the band is
        a ~3 s parquet read and the one-entry cache only helps if consecutive
        episodes want the same one.  Groups with a single spec simply move on
        after one use.
        """
        exhausted = (
            (self._current_group is None)
            or (self._group_uses >= min(episodes_per_band, len(self._groups[self._current_group])))
        )
        if exhausted:
            self._current_group = self._group_keys[int(rng.integers(0, len(self._group_keys)))]
            self._group_uses = 0
        members = self._groups[self._current_group]
        self._group_uses += 1
        return members[int(rng.integers(0, len(members)))]

    def split(self, holdout: float) -> Tuple["SpecPool", "SpecPool"]:
        """Deterministic (train, held-out) split.

        Split on a hash of the spec identity rather than on position, so the same
        spec lands on the same side whatever order the pool was built in.
        """
        if (not (0.0 <= holdout < 1.0)):
            raise ValueError("holdout must be in [0, 1)")
        if (holdout == 0.0):
            return self, SpecPool(self.fingerprint, [], [] if self.has_counterfactuals else None, self.env_fields)
        ordered = sorted(
            self.specs,
            key=lambda s: hashlib.sha256(repr(_spec_key(s)).encode("utf-8")).hexdigest(),
        )
        cut = int(round(len(ordered) * (1.0 - holdout)))
        train, test = ordered[:cut], ordered[cut:]
        return (
            SpecPool(self.fingerprint, train, self._counterfactuals_for(train),
                     self.env_fields, self.cell_index, self.lane_index),
            SpecPool(self.fingerprint, test, self._counterfactuals_for(test),
                     self.env_fields, self.cell_index, self.lane_index),
        )

    def _counterfactuals_for(self, specs: Sequence[Any]) -> Optional[List[Any]]:
        """Stored arrays for ``specs`` in order, or None for a specs-only pool."""
        if (not self.has_counterfactuals):
            return None
        return [self._by_key[_spec_key(spec)] for spec in specs]

    def filter(self, predicate: "SpecFilter", description: str = "") -> "SpecPool":
        """Sub-pool of the specs ``predicate`` accepts, in their original order.

        ``predicate(spec, densities, fluxes, pool)`` gets the stored counterfactual as
        its raw arrays -- densities (steps, cells) indexed by ``pool.cell_index``, fluxes
        (steps, lanes) indexed by ``pool.lane_index`` -- and returns True to keep the
        spec.  On a specs-only pool both arrays are None, so only filters that look at
        the spec itself apply.  Nothing is recomputed: the kept specs carry their
        counterfactuals and the parent's fingerprint across, so the sub-pool is exactly
        as valid as the parent.
        """
        kept = [
            spec for spec in self.specs
            if predicate(spec, *(self._by_key[_spec_key(spec)] or (None, None)), self)
        ]
        note = f"filter {description or getattr(predicate, '__name__', 'predicate')}: kept {len(kept)}/{len(self.specs)}"
        return SpecPool(
            self.fingerprint, kept, self._counterfactuals_for(kept),
            self.env_fields, self.cell_index, self.lane_index, self.provenance + [note],
        )

    # -- persistence ----------------------------------------------------

    def save(self, path: str) -> None:
        payload = {
            "version": POOL_FORMAT_VERSION,
            "fingerprint": self.fingerprint,
            "env_fields": self.env_fields,
            "specs": [asdict(spec) for spec in self.specs],
            "counterfactuals": self._counterfactuals_for(self.specs),
            "cell_index": self.cell_index,
            "lane_index": self.lane_index,
            "provenance": self.provenance,
            "created_at": time.time(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with gzip.open(path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str, env_config, spec_class, require_counterfactuals: bool = True) -> "SpecPool":
        """Load a pool, enforcing the counterfactual fingerprint unless told not to.

        The fingerprint exists to protect *differenced* reward terms: a stored
        counterfactual from a different environment describes a road this run is not
        simulating, and every differenced term would be quietly wrong.  When no reward
        term is differenced -- RewardConfig.uses_counterfactual() is False, which is the
        case for AVC's raw mean-speed reward -- the counterfactuals are never read and
        the pool is serving only as a spec list.  Enforcing the hash then blocks a
        perfectly valid run for data nobody looks at, and rebuilding is not cheap:
        build_pool replays a do-nothing episode per spec, which for 1,499 specs at
        macro_dt=0.1 is the 34-hour pass this reward was restructured to avoid.

        Pass require_counterfactuals=False to downgrade the mismatch to a warning.  The
        specs themselves are always valid: a spec is (dataset, road, start_time, band,
        seed), none of which depends on macro_dt, the vehicle type or the reward.
        """
        with gzip.open(path, "rb") as handle:
            payload = pickle.load(handle)
        if (payload.get("version") != POOL_FORMAT_VERSION):
            raise SpecPoolMismatch(
                f"{path} is format version {payload.get('version')}, expected {POOL_FORMAT_VERSION}"
            )
        expected = counterfactual_fingerprint(env_config)
        if (payload["counterfactuals"] is None):
            # Specs only: there is nothing stale to protect against, and every
            # counterfactual() lookup falls back to a live pass.
            pass
        elif ((payload["fingerprint"] != expected) and (not require_counterfactuals)):
            print(
                f"WARNING: {path} was built for a different environment "
                f"(pool {payload['fingerprint']}, this config {expected}); "
                f"differing fields: {_describe_difference(payload.get('env_fields', {}), env_config)}. "
                "Using it for its SPECS ONLY -- the stored counterfactuals are stale and "
                "no reward term reads them. Rebuild the pool before re-enabling any "
                "differenced reward term."
            )
        elif (payload["fingerprint"] != expected):
            differing = _describe_difference(payload.get("env_fields", {}), env_config)
            raise SpecPoolMismatch(
                f"{path} was built for a different environment "
                f"(pool {payload['fingerprint']}, this config {expected}).\n"
                f"  differing fields: {differing}\n"
                f"  rebuild with: python3.10 sim_rl_spec_pool.py --out {path} "
                f"--count <n> --from-run <run dir>\n"
                f"  (without that, the stored counterfactuals describe a road "
                f"this run is not simulating)"
            )
        specs = [spec_class(**entry) for entry in payload["specs"]]
        return SpecPool(
            payload["fingerprint"], specs, payload["counterfactuals"],
            payload.get("env_fields"), payload.get("cell_index"), payload.get("lane_index"),
            payload.get("provenance"),
        )

    @staticmethod
    def read(path: str, spec_class=None) -> "SpecPool":
        """Load a pool for offline tooling (filtering, inspection), with no fingerprint check.

        ``load`` compares the pool against a live EnvConfig because a run is about to
        use it.  Filtering does not simulate anything, and the fingerprint is carried
        across unchanged, so the check belongs to whoever later loads the sub-pool.
        """
        if (spec_class is None):
            from sim_rl_sumo_training import EpisodeSpec as spec_class
        with gzip.open(path, "rb") as handle:
            payload = pickle.load(handle)
        if (payload.get("version") != POOL_FORMAT_VERSION):
            raise SpecPoolMismatch(
                f"{path} is format version {payload.get('version')}, expected {POOL_FORMAT_VERSION}"
            )
        return SpecPool(
            payload["fingerprint"], [spec_class(**entry) for entry in payload["specs"]],
            payload["counterfactuals"], payload.get("env_fields"), payload.get("cell_index"),
            payload.get("lane_index"), payload.get("provenance"),
        )


# ---------------------------------------------------------------------------
# Filters
#
# A filter is a predicate over (spec, densities, fluxes, pool), built by a named
# factory so it can be chosen from the command line:
#
#   python3 sim_rl_spec_pool.py --filter-from run_data/rl/spec_data/data_full_no_counterfactual.pkl.gz \
#       --out run_data/rl/spec_data/data_full_inband_050_070.pkl.gz \
#       --filter density_band --filter-kwargs '{"low": 0.5, "high": 0.7}'
#
# Add a factory to SPEC_FILTERS to make a new selection available.
# ---------------------------------------------------------------------------

SpecFilter = Callable[[Any, np.ndarray, np.ndarray, "SpecPool"], bool]


def accept_all() -> SpecFilter:
    """Keep every spec (a filtered copy identical to the parent)."""
    def predicate(spec, densities, fluxes, pool) -> bool:
        return True
    predicate.__name__ = "accept_all"
    return predicate


def density_band(
    low: float,
    high: float,
    cells: Sequence[int] = (2, 3, 4),
    rho_j: Optional[float] = None,
    config_folder: Optional[str] = None,
) -> SpecFilter:
    """Keep specs whose INITIAL density over ``cells`` sits in [low, high] of jam.

    Reads the ground-truth macro snapshot the episode is seeded from
    (``initialize_from_ground_truth`` at ``spec.start_time``) straight out of the
    day's ``macro.parquet``, so it needs no simulation and works on specs-only pools.
    The density is averaged over every lane of the spec's road and over ``cells``
    (the ``_step_<x>`` index, 100 m base cells), then divided by ``rho_j``.

    The default cells 2-4 are 200-500 m, where the bubble sits at reset for
    initial_middle_s 350 and margin_s 150.  ``rho_j`` and the dataset location come
    from the pool's env_fields unless given.  Each (day, road) is read once, only the
    wanted cells, and reduced to one value per snapshot time.
    """
    wanted = {int(cell) for cell in cells}
    initial_by_day: Dict[Tuple[str, str], Dict[float, float]] = {}

    def initial_density(pool, dataset: str, road: str) -> Dict[float, float]:
        key = (dataset, road)
        if (key not in initial_by_day):
            import pandas as pd

            folder = config_folder or pool.env_fields.get("config_folder") or DEFAULT_CONFIG_FOLDER
            if (not os.path.isdir(folder)):
                folder = DEFAULT_CONFIG_FOLDER
            with open(os.path.join(folder, dataset), "r") as handle:
                dataset_dir = json.load(handle)["storage_locations"]["simulation_dataset"]
            if (not os.path.isabs(dataset_dir)):
                dataset_dir = os.path.join(HERE, dataset_dir)
            cell_ids = [f"road_{road}_cell_{lane}_step_{x}" for lane in (-1, -2, -3, -4) for x in sorted(wanted)]
            frame = pd.read_parquet(
                os.path.join(dataset_dir, "macro.parquet"),
                columns=["time", "density"],
                filters=[("road_id", "==", road), ("cell_id", "in", cell_ids)],
            )
            means = frame.groupby("time")["density"].mean()
            initial_by_day[key] = {round(float(t), 3): float(v) for t, v in means.items()}
        return initial_by_day[key]

    def predicate(spec, densities, fluxes, pool) -> bool:
        jam = rho_j if (rho_j is not None) else float(pool.env_fields["fd_params"]["rho_j"])
        value = initial_density(pool, str(spec.dataset), str(spec.road)).get(round(float(spec.start_time), 3))
        if (value is None):
            return False            # start time not on the data's snapshot grid
        return bool(low <= (value / jam) <= high)

    predicate.__name__ = f"density_band[{low},{high}] initial cells={sorted(wanted)}"
    return predicate


def counterfactual_density_band(
    low: float,
    high: float,
    cells: Sequence[int] = (2, 3, 4),
    min_time_fraction: float = 0.5,
    rho_j: Optional[float] = None,
) -> SpecFilter:
    """Keep specs whose do-nothing density over ``cells`` sits in [low, high] of jam.

    At each counterfactual snapshot the density is averaged over every lane of the
    spec's road and over ``cells`` (the ``_step_<x>`` index, one base cell per
    ``cell_length``), then divided by ``rho_j``.  The spec is kept when that value is
    in band for at least ``min_time_fraction`` of the snapshots.

    The default cells 2-4 are 200-500 m, where the bubble sits at reset for
    initial_middle_s 350 and margin_s 150 on 100 m cells; it measures what the hero
    starts in, not where it goes.  ``rho_j`` defaults to the pool's own fd_params.
    """
    wanted = {int(cell) for cell in cells}
    columns_by_road: Dict[str, List[int]] = {}

    def predicate(spec, densities, fluxes, pool) -> bool:
        if (densities is None):
            raise ValueError(
                "counterfactual_density_band reads the stored counterfactual densities, and this "
                "pool is specs-only; use density_band (initial ground-truth density) instead"
            )
        road = str(spec.road)
        if (road not in columns_by_road):
            columns_by_road[road] = [
                column for (key_road, key), column in pool.cell_index.items()
                if (str(key_road) == road) and (int(key.rsplit("_step_", 1)[1]) in wanted)
            ]
        columns = columns_by_road[road]
        if (not columns):
            return False
        jam = rho_j if (rho_j is not None) else float(pool.env_fields["fd_params"]["rho_j"])
        window = densities[:, columns]
        valid = ~np.all(np.isnan(window), axis=1)
        if (not valid.any()):
            return False
        fraction_of_jam = np.nanmean(window[valid], axis=1) / jam
        in_band = (fraction_of_jam >= low) & (fraction_of_jam <= high)
        return bool(in_band.mean() >= min_time_fraction)

    predicate.__name__ = (
        f"counterfactual_density_band[{low},{high}] cells={sorted(wanted)} min_time_fraction={min_time_fraction}"
    )
    return predicate


SPEC_FILTERS: Dict[str, Callable[..., SpecFilter]] = {
    "all": accept_all,
    "density_band": density_band,
    "counterfactual_density_band": counterfactual_density_band,
}


def _describe_difference(stored: Dict[str, Any], env_config) -> str:
    current = {
        key: value for key, value in asdict(env_config).items()
        if key not in COUNTERFACTUAL_IRRELEVANT
    }
    names = [
        key for key in sorted(set(stored) | set(current))
        if (json.dumps(stored.get(key), sort_keys=True, default=str)
            != json.dumps(current.get(key), sort_keys=True, default=str))
    ]
    return ", ".join(names) if names else "(none visible; fingerprint differs anyway)"


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


_WORKER_ENV = None


def _worker_init(env_kwargs: Dict[str, Any]) -> None:
    """One environment per worker process, reused across specs.

    Building an environment is itself ~1.3 s, so it is created once here rather
    than per spec.
    """
    global _WORKER_ENV
    from sim_rl_sumo_training import I24SumoHeroEnv

    _WORKER_ENV = I24SumoHeroEnv(**env_kwargs)


def _worker_counterfactual(spec):
    global _WORKER_ENV
    try:
        return spec, _WORKER_ENV._baseline_pass(spec), None
    except Exception as exc:                                    # noqa: BLE001
        return spec, None, f"{type(exc).__name__}: {exc}"


def tile_specs(env_config, seed: int = 0) -> List[Any]:
    """Every non-overlapping episode of every day and road, in time order.

    Random sampling (``_sample_spec``) draws start times independently, so a pool
    much larger than (usable window / episode length) replays the same minutes from
    starts a few seconds apart -- 750 specs per day of 30 s episodes over a ~3 h
    window is one start every ~14 s, each episode overlapping about four others.
    That inflates the episode count a paired test sees. Tiling instead lays episodes
    end to end, so each spec is a distinct stretch of data.

    The window matches ``_sample_band``: from ``time_origin + warmup_s`` to the last
    start that leaves room for ``max_steps`` outer steps plus 60 s. Starts sit on the
    data's snapshot grid, ``max_steps * macro_step`` apart. ``band_start/band_end`` is
    the ``band_span_s`` band the start falls in, as ``_sample_band`` would have drawn
    it, so the ground-truth cache and the pool's band grouping behave as for a
    sampled spec. Seeds come from ``seed`` and are fixed per spec.
    """
    from sim_rl_sumo_training import EpisodeSpec

    rng = np.random.default_rng(seed)
    specs: List[Any] = []
    for dataset in env_config.datasets:
        _, config = env_config.dataset_paths(dataset)
        origin = float(config["time_origin"])
        grid = float(config["time_step"])
        first = origin + env_config.warmup_s
        budget = (env_config.max_steps * env_config.macro_step(config)) + 60.0
        last = origin + float(config["time_length"]) - budget
        spacing = env_config.max_steps * env_config.macro_step(config)
        span = env_config.band_span_s
        for road in env_config.roads:
            start = first
            while (start <= last + 1e-9):
                quantized = origin + (round((start - origin) / grid) * grid)
                band_start = first + (math.floor((quantized - first) / span) * span)
                specs.append(EpisodeSpec(
                    dataset=dataset,
                    road=str(road),
                    start_time=quantized,
                    band_start=band_start,
                    band_end=min(band_start + span, last),
                    sumo_seed=int(rng.integers(0, 2 ** 31 - 1)),
                ))
                start += spacing
    return specs


def build_pool(
    env_config,
    reward_config,
    count: int,
    workers: int = 12,
    seed: int = 0,
    verbose: bool = True,
    with_counterfactuals: bool = True,
    tile: bool = False,
):
    """Build a pool of specs and, unless ``with_counterfactuals`` is False, compute each one's counterfactual once.

    ``tile`` lays non-overlapping episodes end to end over every day (``tile_specs``,
    ``count`` ignored); otherwise ``count`` specs are sampled at random the way
    training draws them.

    A specs-only pool runs no simulation, so it also never learns that a spec's
    minute of data is unusable; the demo skips such episodes with a warning.
    """
    from sim_rl_sumo_training import I24SumoHeroEnv

    if (tile):
        specs = tile_specs(env_config, seed)
        if verbose:
            print(f"tiled {len(specs)} non-overlapping specs over {len(env_config.datasets)} day(s)")
    else:
        sampler = I24SumoHeroEnv(
            env_config=env_config, reward_config=reward_config, seed=seed,
            label_prefix="poolgen", record_rollout=False,
        )
        try:
            specs = [sampler._sample_spec() for _ in range(count)]
        finally:
            sampler.close()

    if (not with_counterfactuals):
        unique = list({_spec_key(spec): spec for spec in specs}.values())
        if verbose:
            print(f"{len(unique)} specs ({len(specs) - len(unique)} duplicates dropped), no counterfactuals")
        return SpecPool(
            counterfactual_fingerprint(env_config),
            unique,
            None,
            {k: v for k, v in asdict(env_config).items() if k not in COUNTERFACTUAL_IRRELEVANT},
        )

    env_kwargs = {
        "env_config": env_config,
        "reward_config": reward_config,
        "seed": seed,
        "label_prefix": "poolworker",
        "record_rollout": False,
    }

    started = time.time()
    good_specs, good_counterfactuals, failures = [], [], 0
    context = mp.get_context("spawn")
    with context.Pool(max(1, workers), initializer=_worker_init, initargs=(env_kwargs,)) as pool:
        for index, (spec, counterfactual, error) in enumerate(
            pool.imap_unordered(_worker_counterfactual, specs, chunksize=1)
        ):
            if (error is not None):
                failures += 1
                if verbose:
                    print(f"  spec failed ({error})")
                continue
            good_specs.append(spec)
            good_counterfactuals.append(counterfactual)
            if (verbose and ((index + 1) % 25 == 0)):
                rate = (index + 1) / (time.time() - started)
                print(f"  {index + 1}/{count} at {rate:.1f} spec/s, "
                      f"{(count - index - 1) / max(rate, 1e-9) / 60:.1f} min left")

    if verbose:
        print(f"built {len(good_specs)} counterfactuals in "
              f"{(time.time() - started) / 60.0:.1f} min ({failures} failed)")
    return SpecPool(
        counterfactual_fingerprint(env_config),
        good_specs,
        good_counterfactuals,
        {k: v for k, v in asdict(env_config).items() if k not in COUNTERFACTUAL_IRRELEVANT},
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    from sim_rl_sumo_training import (
        EnvConfig, RewardConfig, env_config_from_dict, load_run_config,
    )

    # Every day in config/, held-out days included: the default pool is for evaluation with live
    # baselines, where there is no training set to protect.
    all_days = sorted(name for name in os.listdir(DEFAULT_CONFIG_FOLDER) if name.endswith(".json"))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_POOL_PATH, help="where to write the pool (.pkl.gz)")
    parser.add_argument("--count", type=int, default=500,
                        help="specs to sample at random (ignored when tiling)")
    parser.add_argument(
        "--random-specs", action="store_true",
        help="sample --count specs at random, as training draws them, instead of tiling every day into "
             "non-overlapping episodes (tiling is the default for specs-only pools)",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--from-run", default=None,
        help="take EnvConfig from this run directory rather than the dataclass defaults",
    )
    parser.add_argument(
        "--with-counterfactuals", action="store_true",
        help="also replay a do-nothing episode per spec and store its densities and fluxes "
             "(needed only to cache differenced reward terms; slow)",
    )
    parser.add_argument("--datasets", nargs="+", default=all_days)
    parser.add_argument("--roads", nargs="+", default=None)
    parser.add_argument(
        "--filter-from", default=None,
        help="instead of building, filter this existing pool into --out (no simulation)",
    )
    parser.add_argument("--filter", default="all", choices=sorted(SPEC_FILTERS),
                        help="named filter for --filter-from")
    parser.add_argument("--filter-kwargs", default="{}",
                        help='JSON keyword arguments for the filter factory, e.g. \'{"low": 0.5, "high": 0.7}\'')
    args = parser.parse_args(argv)

    if (args.filter_from is not None):
        parent = SpecPool.read(args.filter_from)
        predicate = SPEC_FILTERS[args.filter](**json.loads(args.filter_kwargs))
        pool = parent.filter(predicate)
        pool.save(args.out)
        print(f"{pool.provenance[-1]}")
        print(f"wrote {len(pool)} specs ({pool.band_count} bands) to {args.out}, "
              f"fingerprint {pool.fingerprint} carried from {args.filter_from}")
        return

    if (args.from_run is not None):
        env_config, reward_config, _ = load_run_config(args.from_run)
    else:
        env_config, reward_config = EnvConfig(), RewardConfig()
    # Never build a pool out of a pool.  ``--from-run`` carries the run's own
    # ``spec_pool`` across, and the sampler below draws from it when it is set --
    # so the "new" pool would be a re-run of the old pool's specs, on the old
    # pool's days, however ``--datasets`` was pointed.  Building a held-out set
    # is exactly the case where that silently returns the training set.
    overrides: Dict[str, Any] = {"spec_pool": None}
    if (args.datasets):
        overrides["datasets"] = tuple(args.datasets)
    if (args.roads):
        overrides["roads"] = tuple(args.roads)
    env_config = env_config_from_dict(asdict(env_config), **overrides)

    # Tile by default for specs-only (evaluation) pools; counterfactual pools feed training and
    # keep the random draws training itself makes.
    tile = (not args.with_counterfactuals) and (not args.random_specs)
    if (args.with_counterfactuals):
        print(f"building {'tiled' if tile else args.count} counterfactuals on {args.workers} workers")
    else:
        print(f"{'tiling' if tile else f'sampling {args.count}'} specs over "
              f"{len(env_config.datasets)} day(s), no counterfactuals")
    print(f"fingerprint {counterfactual_fingerprint(env_config)}")
    pool = build_pool(env_config, reward_config, args.count, args.workers, args.seed,
                      with_counterfactuals=args.with_counterfactuals, tile=tile)
    pool.save(args.out)
    size = os.path.getsize(args.out) / 1e6
    print(f"wrote {len(pool)} specs ({pool.band_count} bands) to {args.out} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
