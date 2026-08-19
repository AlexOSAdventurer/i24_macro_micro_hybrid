from __future__ import annotations

"""Vanilla (pure-macro) METANET calibration against the empirical macro data.

The open-loop counterpart of `sim_calibration_open_loop.py`, with the micro bridge
removed: METANET has no mask coupling yet (the hybrid seams in `METANETModel` are
zero-gradient), so this is macro-only from end to end.

Two structural differences from the first-order scripts:

- The mesh is `I24WestAndEastNetworkCollapsed`: one cell per longitudinal station per
  carriageway, all lanes aggregated. Cell density is the total across lanes, so the
  shared `METANETParams` carries the lane count in `lanes` and keeps the paper's
  per-lane `rho_crit`/`kappa`. Every cell shares one parameter set (no
  `per_cell_params`).
- The ground truth is collapsed the same way, and the metric is computed from model
  state recorded in the poststep callback rather than from `RolloutRenderer`. The
  renderer now reports second-order speed correctly (it prefers `Cell.velocity` over
  inverting the FD, and the two were verified to agree exactly), but scoring here
  avoids paying for a per-step network clone plus a renderer build on every trial.

Boundary conditions are supplied as fluxes and ghost states rather than by pinning
the end cells, so all 16 cells per carriageway are genuinely simulated -- see
`install_boundary_conditions`. `initialize_from_ground_truth` is called with
drives_boundaries=False to stop the store overwriting boundary cell mass on top.
"""

from simulation import (
    GroundTruthStore,
    I24WestAndEastNetworkCollapsed,
    METANETModel,
    METANETParams,
    Simulation,
    TriangularFD,
)
import numpy as np
import optuna
import pandas as pd
import json
import os

# Road the metric is scored on (westbound), matching the existing calibrations.
ROAD_ID = "2"
# Velocity below which a cell counts as jammed, in m/s. Same value as the
# first-order calibrations, so the objective stays comparable.
#JAM_THRESHOLD = 15.0

with open("i24_motion_to_dataset.json", "r") as f:
    config = json.load(f)

# Macro only: nothing here is microscopic, so the ~550 MB micro parquet is not read.
MACRO_PATH = os.path.join(config["storage_locations"]["simulation_dataset"], "macro.parquet")
MACRO_METANET_PATH = os.path.join(config["storage_locations"]["simulation_dataset"], "macro_metanet.parquet")


def collapse_macro_ground_truth(gt: GroundTruthStore) -> GroundTruthStore:
    """Aggregate per-lane macro ground truth onto the collapsed one-cell-per-station mesh.

    Fallback for when `macro_metanet.parquet` has not been generated yet; it produces
    the same numbers as `I24SimulationData.generate_macro_data_metanet`, which does the
    aggregation from the source arrays and caches it.

    Density sums over lanes (it is a total, like the collapsed cell's own density);
    mean speed is flow-weighted, Sum(rho*v) / Sum(rho), since speed is intensive. Cells
    with no vehicles in any lane fall back to the unweighted mean.
    """
    df = gt.macro_df.copy()
    # cell ids are road_{road}_cell_{lane}_step_{i}; only the station index survives
    df["step"] = df["cell_id"].str.rsplit("_step_", n=1).str[-1].astype(int)
    df["flow"] = df["density"] * df["velocity"]
    grouped = df.groupby(["time", "time_length", "road_id", "step"], as_index=False).agg(
        density=("density", "sum"),
        flow=("flow", "sum"),
        velocity_mean=("velocity", "mean"),
    )
    dens = grouped["density"].to_numpy()
    safe = np.where(dens > 1e-12, dens, 1.0)
    grouped["velocity"] = np.where(
        dens > 1e-12, grouped["flow"].to_numpy() / safe, grouped["velocity_mean"].to_numpy()
    )
    grouped["cell_id"] = (
        "road_" + grouped["road_id"].astype(str) + "_cell_collapsed_step_" + grouped["step"].astype(str)
    )
    return GroundTruthStore(
        macro_df=grouped[["time", "time_length", "road_id", "cell_id", "density", "velocity"]]
    )


class MacroSeries:
    """Per-time (road_id, cell_id) -> density/velocity lookup with nearest-time snapping.

    `GroundTruthStore` keeps this for density only, and its snapping helper is private;
    velocity needs the same treatment here for the speed boundary condition and for
    scoring, so both are built once up front.
    """

    def __init__(self, gt: GroundTruthStore):
        self.density: dict[float, dict[tuple[str, str], float]] = {}
        self.velocity: dict[float, dict[tuple[str, str], float]] = {}
        for t, grp in gt.macro_df.groupby("time"):
            key = float(t)
            keys = list(zip(grp["road_id"].astype(str), grp["cell_id"].astype(str)))
            self.density[key] = dict(zip(keys, grp["density"].astype(float)))
            self.velocity[key] = dict(zip(keys, grp["velocity"].astype(float)))
        self.times = np.sort(np.array(list(self.density.keys()), dtype=float))

    def nearest(self, time_value: float, tolerance: float = 1e-1) -> float:
        idx = int(np.searchsorted(self.times, time_value))
        candidates = [
            int(np.clip(idx, 0, len(self.times) - 1)),
            int(np.clip(idx - 1, 0, len(self.times) - 1)),
        ]
        chosen = float(self.times[min(candidates, key=lambda i: abs(self.times[i] - time_value))])
        if abs(chosen - time_value) > tolerance:
            raise KeyError(f"No ground-truth snapshot near time={time_value}. Closest: {chosen}.")
        return chosen


if os.path.exists(MACRO_METANET_PATH):
    gt_collapsed = GroundTruthStore.from_parquet(macro_parquet_path=MACRO_METANET_PATH)
else:
    print(f"{MACRO_METANET_PATH} not found; collapsing {MACRO_PATH} in memory. "
          f"Run I24SimulationData().generate_macro_data_metanet() to cache it.")
    gt_collapsed = collapse_macro_ground_truth(
        GroundTruthStore.from_parquet(macro_parquet_path=MACRO_PATH)
    )
gt_series = MacroSeries(gt_collapsed)


def suggest_params(trial, lanes: int) -> METANETParams:
    """Sample one parameter set shared by every cell.

    Ranges are the METANET calibration bounds of arXiv:2605.23042 where the paper
    states them (eta 5-60 km^2/h, kappa and rho* per lane), widened around its
    synthetic ground truth (Table II: tau=18 s, eta=30, kappa=40, v*=120 km/h,
    rho*=37.45 veh/km/lane, alpha=1.4) for the free-flow speed of this corridor.

    v_free is the one bound that had to move. Least-squares fitting V[rho] to the
    collapsed ground truth puts the optimum at 80.8 km/h (rho* 23.6 veh/km/lane,
    alpha 2.06, residual 2.9 m/s against a data std of 5.3), so the old 100-200 km/h
    range excluded it outright. That range was set when the empirical speeds were
    still being computed from the raw feet-valued x/y and so came out 3.28x too fast;
    this corridor is a congested morning peak whose measured speeds top out at
    27.4 m/s.
    """
    return METANETParams.from_paper_units(
        tau_h=trial.suggest_float("tau_s", 1.0, 60.0) / 3600.0,
        eta_km2_per_h=trial.suggest_float("eta_km2_per_h", 1.0, 60.0),
        kappa_veh_per_km_lane=trial.suggest_float("kappa_veh_per_km_lane", 1.0, 100.0),
        v_free_kmh=trial.suggest_float("v_free_kmh", 40.0, 140.0),
        rho_crit_veh_per_km_lane=trial.suggest_float("rho_crit_veh_per_km_lane", 15.0, 100.0),
        alpha=trial.suggest_float("alpha", 0.5, 5.0),
        lanes=lanes,
    ), 15.0 #, trial.suggest_float("jam_threshold", 5.0, 20.0)


def install_boundary_conditions(sim: Simulation, model: METANETModel) -> None:
    """Drive the open ends the way METANET expects, rather than pinning their state.

    The paper's boundary conditions are a prescribed upstream demand and a prescribed
    downstream density, with the end cells still simulated. `simulation.py` already
    carries every hook for that:

        Simulation.inflow_boundary_map      -> external_inflow = min(q_in, supply)
        Simulation.outflow_boundary_map     -> the discharge at the far end
        METANETModel.upstream_velocity_map  -> v_{x-1} for the convection term
        METANETModel.downstream_density_map -> rho_{x+1} for the anticipation term

    Two deliberate departures, both because this corridor is measured at both ends:

    - The discharge is prescribed too. The paper lets the last cell empty itself at
      its own q = rho*v, which is right when nothing downstream is measured, but here
      it lets the road drift far off the data (down to 93 vehicles against 211 over an
      hour) without the model being wrong about anything the jam/free metric can see.
    - There is no measured cell beyond the domain, so the downstream ghost density is
      the last cell's *own* ground truth. That makes the anticipation term a soft pull
      toward the data rather than a true exterior state. The alternative is to leave
      the map unset, which zeroes the term (the model's built-in fallback); neither is
      clean on a corridor with nothing past the end of it.

    Written as a *prestep* callback so the maps are current when `_step_active_network`
    computes the fluxes -- prestep callbacks run before the step callbacks and before
    the active mesh is built.
    """
    inlets, outlets = [], []
    for road_id, road in sim.network.roads.items():
        for cell in road.cells.values():
            # The same test apply_density_snapshot_to_network_boundaries uses to pick
            # its targets, so the two treatments select exactly the same cells and no
            # cell ends up both pinned and flux-driven.
            if not cell.inflow_connections:
                inlets.append((road_id, cell.cell_id))
            if not cell.outflow_connections:
                outlets.append((road_id, cell.cell_id))

    def update(current_time, resolution):
        t = gt_series.nearest(current_time)
        rho_map, v_map = gt_series.density[t], gt_series.velocity[t]
        for key in inlets:
            rho, v = rho_map.get(key), v_map.get(key)
            if rho is None or v is None:
                continue
            sim.inflow_boundary_map[key] = float(rho) * float(v)
            model.upstream_velocity_map[key] = float(v)
        for key in outlets:
            rho, v = rho_map.get(key), v_map.get(key)
            if rho is None:
                continue
            # Both maps follow the cell convention: totals across `lanes`. The model
            # converts to the paper's per-lane density itself.
            model.downstream_density_map[key] = float(rho)
            if v is not None:
                sim.outflow_boundary_map[key] = float(rho) * float(v)

    sim.register_prestep_callback(update, "metanet_boundary_conditions")


def build_simulation(params_for_lanes) -> tuple[Simulation, METANETModel]:
    """Collapsed network + METANET model, initialised from the ground truth at t0.

    `params_for_lanes` is called with the lane count once the network is built, so the
    caller can sample parameters that depend on it.
    """
    road_config = config["road_data"][ROAD_ID]
    # The FD is decorative under METANET (the model supplies demand and supply
    # itself); it is kept only so the cells carry one for plotting. The collapsed
    # generator scales its rho_j by the lane count. NOTE: these values are stale --
    # they were calibrated against the pre-fix macro data, whose speeds were 3.28x
    # too fast. Nothing here depends on them, but re-running the first-order
    # calibration will replace them.
    generator = I24WestAndEastNetworkCollapsed(
        fd=TriangularFD(v_f=49.73562026160161, w=5.697835695812354, rho_j=0.1304577157114563),
        lambda_lc=0.0,  # unused: one lane per road leaves no pair to exchange across
    )
    generator.create_network(
        road_config["road_length"],
        road_config["cell_length"],
        road_config["lanes"],
        lane_width=road_config["lane_width"],
    )

    lane_counts = set(generator.lanes_per_road.values())
    if len(lane_counts) != 1:
        raise ValueError(
            f"Roads have differing lane counts {generator.lanes_per_road}; a single "
            f"shared METANETParams cannot describe them. Use per_cell_params."
        )
    params, JAM_THRESHOLD = params_for_lanes(lane_counts.pop())

    model = METANETModel(params)
    # No masks, so the active mesh is just the base cells: leave min_cell_length at
    # its default rather than passing the cell length and risking a merge.
    sim = Simulation(
        network=generator.network,
        time_resolution=config["time_step"],
        origin_time=config["time_origin"],
        macro_model=model,
    )
    sim.record_rollout = False  # the metric records what it needs itself

    t0 = float(config["time_origin"])
    # drives_boundaries=False: the store still supplies the initial condition and the
    # renderer's empirical heatmap, but it no longer overwrites boundary cell mass
    # every step. install_boundary_conditions supplies the open ends instead.
    sim.initialize_from_ground_truth(gt_collapsed, time_value=t0, drives_boundaries=False)
    # Density comes from the store; seed the model's speed state from the data too,
    # so the first steps are not relaxing away from an FD-equilibrium guess.
    v0 = gt_series.velocity[gt_series.nearest(t0)]
    for road_id, road in sim.network.roads.items():
        for cell in road.cells.values():
            if (road_id, cell.cell_id) in v0:
                cell.velocity = float(v0[(road_id, cell.cell_id)])
    install_boundary_conditions(sim, model)
    return sim, model, JAM_THRESHOLD


def run_calibration(trial):
    sim, model, JAM_THRESHOLD = build_simulation(lambda lanes: suggest_params(trial, lanes))

    warnings = model.stability_report(sim)
    if warnings:
        # dt > tau (oscillating relaxation) or CFL > 1: the scheme is not solving the
        # model, so the trial's score would be meaningless.
        print("Pruning:", warnings[0])
        raise optuna.TrialPruned()

    cells = sorted(sim.network.roads[ROAD_ID].cells.values(), key=lambda c: c.start_s)
    scored_keys = [(ROAD_ID, c.cell_id) for c in cells]
    # Every cell is simulated now that the open ends are driven by flux rather than by
    # overwriting their state, so this is a diagnostic on the cells furthest from the
    # boundary conditions rather than a correction for cells that are really data.
    interior = np.array([bool(c.inflow_connections) and bool(c.outflow_connections) for c in cells])

    sim_velocity: list[list[float]] = []
    sim_density: list[list[float]] = []
    gt_velocity: list[list[float]] = []
    gt_density: list[list[float]] = []

    def record(current_time, resolution):
        t = gt_series.nearest(current_time)
        v_map, rho_map = gt_series.velocity[t], gt_series.density[t]
        sim_velocity.append([float(c.velocity) if c.velocity is not None else 0.0 for c in cells])
        sim_density.append([float(c.density) for c in cells])
        gt_velocity.append([float(v_map[k]) for k in scored_keys])
        gt_density.append([float(rho_map[k]) for k in scored_keys])

    sim.register_poststep_callback(record, "metanet_record")

    steps = int(round(config["road_data"][ROAD_ID]["time_length"] / config["time_step"])) - 1
    sim.run(steps * config["time_step"])

    sim_v = np.asarray(sim_velocity)
    sim_rho = np.asarray(sim_density)
    gt_v = np.asarray(gt_velocity)
    gt_rho = np.asarray(gt_density)

    if not (np.all(np.isfinite(sim_v)) and np.all(np.isfinite(sim_rho))):
        # METANET has no supply constraint, so a bad parameter set can run away.
        print("Pruning: simulation produced non-finite state")
        raise optuna.TrialPruned()

    def jam_free_metric(sim_vel, emp_vel):
        sim_in_jam = (sim_vel < JAM_THRESHOLD).astype(int).reshape(-1)
        empirical_in_jam = (emp_vel < JAM_THRESHOLD).astype(int).reshape(-1)
        sim_in_free = (sim_vel >= JAM_THRESHOLD).astype(int).reshape(-1)
        empirical_in_free = (emp_vel >= JAM_THRESHOLD).astype(int).reshape(-1)
        in_jam = -(sim_in_jam * empirical_in_jam).sum() / (empirical_in_jam.sum() + 1e-12)
        in_free = -(sim_in_free * empirical_in_free).sum() / (empirical_in_free.sum() + 1e-12)
        return in_jam, in_free

    def l1_error(sim, emp):
        return np.abs(sim - emp).sum() / ((sim.reshape(-1).shape[0]) * emp.reshape(-1).max())


    in_jam, in_free = jam_free_metric(sim_v, gt_v)
    l1_rho, l1_vel = l1_error(sim_rho, gt_rho, ), l1_error(sim_v, gt_v)
    metric = in_jam + in_free# + l1_rho + l1_vel
    #metric = l1_rho + l1_vel

    # Diagnostics: the objective only sees jam/free agreement, so keep the magnitudes
    # around for judging whether a good score came from a sensible field.
    interior_jam, interior_free = jam_free_metric(sim_v[:, interior], gt_v[:, interior])
    trial.set_user_attr("jam_recall", float(-in_jam))
    trial.set_user_attr("free_recall", float(-in_free))
    trial.set_user_attr("metric_interior", float(interior_jam + interior_free))
    trial.set_user_attr("velocity_rmse", float(np.sqrt(np.mean((sim_v - gt_v) ** 2))))
    trial.set_user_attr("density_rmse", float(np.sqrt(np.mean((sim_rho - gt_rho) ** 2))))

    print(in_jam, in_free)
    print(l1_rho, l1_vel)
    print(metric)
    return metric


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(run_calibration, n_trials=500, n_jobs=1)
    print(study.best_params)
    print(study.best_trial.user_attrs)
