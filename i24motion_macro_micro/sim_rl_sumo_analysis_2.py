"""Hero, lane and corridor metrics for the ring controllers, from the per-vehicle traces.

Compares three arms on the same episode specs -- by default SUMO IDM (the ``sumo`` hero),
the published AVC ring controller and the corridor-matched one -- using what
sim_rl_sumo_demo.py writes per episode:

  <demo>/traces/vehicles/<controller>_<episode>.parquet   every bubble vehicle, every step
  <demo>/traces/<controller>_<episode>.csv                 per-step record (upstream/downstream flow)
  <demo>/episodes.csv                                      spec identity (dataset, start_time)

Positions in the hero's lane are ranked by ``s`` on each side of the hero at every step and
followed as series, whoever occupies them.  The chain is set by LEADERS and FOLLOWERS, currently

  L1  H  F1 F2 F3        (leader ahead, hero, followers behind)

SUMMARY TABLE (one value per episode over the window step * dt >= --skip-s, default 30 s,
then median [min, max] across episodes per arm):

  hero_speed           mean hero speed (m/s)
  hero_gap             median space gap to L1, L1.s - (H.s + H.length) (m)
  <first>_var          speed variance of the first position in the chain (m^2/s^2), reported raw
  <position>_link      for every later position: var(position) / var(vehicle directly ahead of
                       it), over the steps where both are present; below 1 attenuates
  cumulative_<last>_<first>  var(last) / var(first), over the steps where both are present
  cutins_ahead         vehicles entering the hero's lane directly ahead of it (count)
  upstream_flow        flow rho*v in the macro cell behind the bubble, lane-averaged (veh/h/lane)
  downstream_flow      same, in the cell ahead of the bubble

A variance needs at least --min-samples-s of shared samples, and a ratio a leader variance
above 1e-6; otherwise the episode is left out of that row, and n says how many remain.

SPEED OSCILLATION TIME SERIES (every step, whole episode): rolling speed variance per position
over a centred --oscillation-window-s window, and the rolling link ratio against the vehicle
ahead.  Figures show the median across episodes with the interquartile range, one colour per
arm, the excluded start of the episode shaded.

BINNED VARIANCE RATIOS (summary window only): each pair in BINNED_RATIOS (H/L1, H/F1, F1/F2, F2/F3,
F3/L1) as var(numerator)/var(denominator) per --bin-s bin (default 5 s), over the steps of the bin
where both are present (at least half the bin).  Figures show the median per bin with IQR whiskers.

Paired effects against the reference arm (per metric, per spec) are kept alongside, with a
bootstrap 95% interval over 15-minute clock bins: adjacent tiles in the same bin share a jam.

Outputs under --out:
  tables/episode_metrics.csv          every summary metric, every arm and episode
  tables/summary_statistics.csv       median, min, max and n per metric and arm
  tables/summary_statistics.tex       the paper table, median [min, max]
  tables/paired_vs_reference.csv      paired differences, bin-bootstrap interval, Wilcoxon p
  tables/oscillation_timeseries.csv   per-step median, quartiles and n per arm and position
  tables/binned_variance_ratio.csv    per-bin median, quartiles and n per arm and ratio
  figures/timeseries/                 variance and link ratio by position, binned variance ratios (.png/.pdf)
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# The chain is set by these two lists alone; every row, caption and figure is derived from them.
# L3..F5 left L3 and F5 unoccupied in many episodes, and unequally across arms (a long AVC gap
# pushes L3 out of the bubble, a fast IDM hero leaves F5 behind it), so the chain is L1..F3.
# LEADERS = ["L3", "L2", "L1"]
# FOLLOWERS = ["F1", "F2", "F3", "F4", "F5"]
LEADERS = ["L1"]
FOLLOWERS = ["F1", "F2", "F3"]
POSITIONS = LEADERS + ["H"] + FOLLOWERS          # each entry's leader is the one before it
LINKS = POSITIONS[1:]                            # positions with a leader in the chain
FIRST, LAST = POSITIONS[0], POSITIONS[-1]        # chain ends: FIRST reported raw, cumulative LAST / FIRST
# Binned speed-variance ratios over the summary window, (numerator, denominator) in panel order.
# Note the direction differs between pairs: H/L1 and F3/L1 are below 1 when the rear vehicle is
# quieter, while H/F1, F1/F2 and F2/F3 are above 1 when the rear vehicle is quieter.
BINNED_RATIOS = [("H", "L1"), ("H", "F1"), ("F1", "F2"), ("F2", "F3"), ("F3", "L1")]
BIN_S = 900.0
MIN_LEADER_VARIANCE = 1e-6

DEFAULT_ARMS = [
    "SUMO IDM=run_data/rl/analysis/avc_original:sumo",
    "Arm-1=run_data/rl/analysis/avc_original:trpo",     # published AVC ring controller
    "Arm-2=run_data/rl/analysis/avc_corridor:trpo",     # corridor-matched AVC controller
]
# The default palette's first three categorical slots, assigned in this fixed order.
ARM_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]

# (metric, LaTeX label, decimals) in table order.
TABLE_ROWS: List[Tuple[str, str, int]] = (
    [("hero_speed", "Speed (m/s)", 1), ("hero_gap", "Gap to leader (m)", 1),
     (f"{FIRST}_var", f"{FIRST} $\\sigma_v^2$ (m$^2$/s$^2$)", 1)]
    + [(f"{p}_link", f"{p} / {POSITIONS[POSITIONS.index(p) - 1]}", 1) for p in LINKS]
    + [(f"cumulative_{LAST}_{FIRST}", f"Cumulative {LAST} / {FIRST}", 1),
       ("cutins_ahead", "Cut-ins ahead of hero", 1),
       ("upstream_flow", "Upstream flow (veh/h/lane)", 0),
       ("downstream_flow", "Downstream flow (veh/h/lane)", 0)]
)
TABLE_SECTIONS = {"hero_speed": "Hero", f"{FIRST}_var": "Speed oscillation along the hero's lane",
                  "cutins_ahead": "Lane and corridor"}


# ---------------------------------------------------------------------------
# Per-episode reconstruction
# ---------------------------------------------------------------------------


def positional_wide(vehicles: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], float]:
    """Per-step tables (index step, columns POSITIONS) of speed, s, length and prev_lane, plus dt."""
    dt = float(np.median(np.diff(np.sort(vehicles["time"].unique()))))
    v = vehicles.sort_values(["vehicle_id", "step"]).reset_index(drop=True)
    same_vehicle = v["vehicle_id"].eq(v["vehicle_id"].shift()) & v["step"].eq(v["step"].shift() + 1)
    v["prev_lane"] = v["lane_id"].shift().where(same_vehicle)
    hero = v[v["is_hero"]].set_index("step")[["s", "lane_id"]].rename(columns={"s": "hero_s", "lane_id": "hero_lane"})
    v = v.join(hero, on="step")
    lane = v[(~v["is_hero"]) & (v["lane_id"] == v["hero_lane"])]
    ahead = lane[lane["s"] > lane["hero_s"]].copy()
    behind = lane[lane["s"] < lane["hero_s"]].copy()
    ahead["rank"] = ahead.groupby("step")["s"].rank(method="first", ascending=True)
    behind["rank"] = behind.groupby("step")["s"].rank(method="first", ascending=False)
    ahead = ahead[ahead["rank"] <= len(LEADERS)].assign(position=lambda f: "L" + f["rank"].astype(int).astype(str))
    behind = behind[behind["rank"] <= len(FOLLOWERS)].assign(position=lambda f: "F" + f["rank"].astype(int).astype(str))
    hero_rows = v[v["is_hero"]].assign(position="H")
    long = pd.concat([ahead, hero_rows, behind], ignore_index=True)
    steps = np.arange(int(vehicles["step"].min()), int(vehicles["step"].max()) + 1)
    wide = {
        field: long.pivot_table(index="step", columns="position", values=field, aggfunc="first")
        .reindex(index=steps, columns=POSITIONS)
        for field in ("speed", "s", "length", "prev_lane", "hero_lane")
    }
    wide["hero_lane"] = hero["hero_lane"].reindex(steps)
    return wide, dt


def _variance(values: pd.Series, min_samples: int) -> float:
    values = values.dropna()
    return float(values.var(ddof=0)) if (len(values) >= min_samples) else float("nan")


def _pair_ratio(numerator: pd.Series, denominator: pd.Series, min_samples: int) -> float:
    both = numerator.notna() & denominator.notna()
    if (both.sum() < min_samples):
        return float("nan")
    top, bottom = _variance(numerator[both], min_samples), _variance(denominator[both], min_samples)
    return top / bottom if (np.isfinite(top) and np.isfinite(bottom) and (bottom > MIN_LEADER_VARIANCE)) else float("nan")


def episode_metrics(
    vehicles: pd.DataFrame, trace: Optional[pd.DataFrame], args
) -> Tuple[Dict[str, float], pd.DataFrame, List[Dict[str, float]]]:
    """Summary metrics over the window, the per-step oscillation series over the whole episode, and
    the BINNED_RATIOS per --bin-s bin of the window."""
    wide, dt = positional_wide(vehicles)
    speed = wide["speed"]
    in_window = (speed.index.to_numpy() * dt) >= (args.skip_s - 1e-9)
    min_samples = max(2, int(round(args.min_samples_s / dt)))
    w = {field: table[in_window] for field, table in wide.items() if isinstance(table, pd.DataFrame)}
    hero_lane = wide["hero_lane"][in_window]

    out: Dict[str, float] = {}
    out["hero_speed"] = float(w["speed"]["H"].mean())
    gap = w["s"]["L1"] - (w["s"]["H"] + w["length"]["H"])
    out["hero_gap"] = float(gap.median()) if gap.notna().any() else float("nan")
    out[f"{FIRST}_var"] = _variance(w["speed"][FIRST], min_samples)
    for p in LINKS:
        leader = POSITIONS[POSITIONS.index(p) - 1]
        out[f"{p}_link"] = _pair_ratio(w["speed"][p], w["speed"][leader], min_samples)
    out[f"cumulative_{LAST}_{FIRST}"] = _pair_ratio(w["speed"][LAST], w["speed"][FIRST], min_samples)
    entered = w["prev_lane"]["L1"].notna() & (w["prev_lane"]["L1"] != hero_lane)
    out["cutins_ahead"] = float(entered.sum())
    for side in ("upstream", "downstream"):
        column = f"{side}_flow"
        if (trace is not None) and (column in trace.columns):
            late = trace[(trace["step"] * dt) >= (args.skip_s - 1e-9)][column]
            out[column] = float(late.mean()) if late.notna().any() else float("nan")
        else:
            out[column] = float("nan")

    window = max(2, int(round(args.oscillation_window_s / dt)))
    rolling = speed.rolling(window, center=True, min_periods=max(2, window // 2)).var(ddof=0)
    series = rolling.copy()
    series.columns = [f"{p}_var" for p in POSITIONS]
    for p in LINKS:
        leader = POSITIONS[POSITIONS.index(p) - 1]
        denominator = rolling[leader].where(rolling[leader] > MIN_LEADER_VARIANCE)
        series[f"{p}_link"] = rolling[p] / denominator
    series.insert(0, "time_s", series.index.to_numpy() * dt)

    # Variance ratios per bin of the summary window, each over the steps of that bin where both
    # vehicles are present; a bin needs at least half of its samples shared.
    binned: List[Dict[str, float]] = []
    elapsed = speed.index.to_numpy() * dt
    bin_samples = max(2, int(round(0.5 * args.bin_s / dt)))
    end_s = float(elapsed.max())
    start = args.skip_s
    while (start < end_s - 1e-9):
        stop = start + args.bin_s
        in_bin = (elapsed >= start - 1e-9) & (elapsed < stop - 1e-9)
        for top, bottom in BINNED_RATIOS:
            binned.append({
                "ratio": f"{top}/{bottom}", "bin_start_s": start, "bin_end_s": stop,
                "value": _pair_ratio(speed[top][in_bin], speed[bottom][in_bin], bin_samples),
            })
        start = stop
    return out, series, binned


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def parse_arm(text: str) -> Tuple[str, str, str]:
    label, _, target = text.partition("=")
    demo, _, controller = target.rpartition(":")
    if (not label) or (not demo) or (not controller):
        raise ValueError(f"--arm {text!r} is not LABEL=DEMO_DIR:CONTROLLER")
    return label.strip(), demo if os.path.isabs(demo) else os.path.join(HERE, demo), controller


def load_arm(label: str, demo: str, controller: str, args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    episodes = pd.read_csv(os.path.join(demo, "episodes.csv"))
    rows, series, binned = [], [], []
    for path in sorted(glob.glob(os.path.join(demo, "traces", "vehicles", f"{controller}_*.parquet"))):
        index = os.path.splitext(os.path.basename(path))[0].rpartition("_")[2]
        if (not index.isdigit()):
            continue
        trace_path = os.path.join(demo, "traces", f"{controller}_{index}.csv")
        trace = pd.read_csv(trace_path) if os.path.isfile(trace_path) else None
        metrics, per_step, per_bin = episode_metrics(pd.read_parquet(path), trace, args)
        match = episodes[(episodes["controller"] == controller) & (episodes["episode"] == int(index))]
        if (not match.empty):
            metrics["dataset"] = str(match.iloc[0]["dataset"])
            metrics["start_time"] = float(match.iloc[0]["start_time"])
        metrics.update({"arm": label, "episode": int(index)})
        rows.append(metrics)
        series.append(per_step.assign(arm=label, episode=int(index)))
        binned.extend({**row, "arm": label, "episode": int(index)} for row in per_bin)
    if (not rows):
        raise FileNotFoundError(f"no traces/vehicles/{controller}_*.parquet under {demo}")
    return (pd.DataFrame(rows), pd.concat(series, ignore_index=False).rename_axis("step").reset_index(),
            pd.DataFrame(binned))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def summary_statistics(frame: pd.DataFrame, arms: Sequence[str]) -> pd.DataFrame:
    rows = []
    for metric, _, _ in TABLE_ROWS:
        row: Dict[str, object] = {"metric": metric}
        for arm in arms:
            values = pd.to_numeric(frame.loc[frame["arm"] == arm, metric], errors="coerce").dropna()
            row[f"{arm} median"] = float(values.median()) if len(values) else float("nan")
            row[f"{arm} min"] = float(values.min()) if len(values) else float("nan")
            row[f"{arm} max"] = float(values.max()) if len(values) else float("nan")
            row[f"{arm} n"] = int(len(values))
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(value: float, decimals: int) -> str:
    return "--" if (not np.isfinite(value)) else f"{value:.{decimals}f}"


def latex_table(summary: pd.DataFrame, arms: Sequence[str], args) -> str:
    counts = sorted({int(summary[f"{arm} n"].max()) for arm in arms})
    lines = [
        "% Preamble: \\usepackage{booktabs}",
        # Median on the first line, [min, max] in a smaller font below it, to keep the columns narrow.
        # "\\providecommand{\\mmm}[3]{#1\\,[#2,\\,#3]}",
        "\\providecommand{\\mmm}[3]{\\begin{tabular}[t]{@{}c@{}}#1\\\\[-2pt]{\\scriptsize[#2,\\,#3]}\\end{tabular}}",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Hero, lane, and corridor metrics on the in-band I-24 episodes "
        f"({'/'.join(str(c) for c in counts)} episodes, {args.skip_s:g}--end\\,s of each episode). "
        "Each cell is the median over episodes, with [min, max] below it. Link rows are the speed variance of a "
        f"vehicle in the hero's lane divided by that of the vehicle directly ahead of it ({FIRST} is "
        "reported raw); values below 1 attenuate. "
        f"Cumulative is $\\sigma_v^2(\\mathrm{{{LAST}}})/\\sigma_v^2(\\mathrm{{{FIRST}}})$. "
        "Flow is $\\rho v$ in the macroscopic cell behind or ahead of the bubble, averaged over the four lanes.}",
        "\\label{tab:controller_metrics}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{l" + "c" * len(arms) + "}",
        "\\toprule",
        " & " + " & ".join(arms) + " \\\\",
    ]
    indexed = summary.set_index("metric")
    for metric, label, decimals in TABLE_ROWS:
        if (metric in TABLE_SECTIONS):
            lines.append("\\midrule")
            lines.append(f"\\multicolumn{{{len(arms) + 1}}}{{l}}{{\\textit{{{TABLE_SECTIONS[metric]}}}}} \\\\")
        row = indexed.loc[metric]
        cells = [
            f"\\mmm{{{_fmt(row[f'{arm} median'], decimals)}}}{{{_fmt(row[f'{arm} min'], decimals)}}}"
            f"{{{_fmt(row[f'{arm} max'], decimals)}}}"
            for arm in arms
        ]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def bootstrap_ci(differences: np.ndarray, clusters: np.ndarray, reps: int, rng) -> Tuple[float, float]:
    labels = np.unique(clusters)
    if (len(labels) < 2):
        return float("nan"), float("nan")
    groups = [differences[clusters == label] for label in labels]
    means = np.empty(reps)
    for r in range(reps):
        pick = rng.integers(0, len(groups), size=len(groups))
        means[r] = np.concatenate([groups[i] for i in pick]).mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_vs_reference(frame: pd.DataFrame, arms: Sequence[str], reps: int, seed: int) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    rng = np.random.default_rng(seed)
    reference = arms[0]
    ref = frame[frame["arm"] == reference].set_index("episode")
    rows = []
    for arm in arms[1:]:
        test = frame[frame["arm"] == arm].set_index("episode")
        shared = test.index.intersection(ref.index)
        if ("start_time" in ref.columns) and ("start_time" in test.columns):
            mismatched = (test.loc[shared, "start_time"] != ref.loc[shared, "start_time"]).sum()
            if (mismatched):
                print(f"WARNING: {arm} and {reference} disagree on the spec of {mismatched} episode(s); pairing by index anyway")
        clusters = (ref.loc[shared, "dataset"].astype(str) + "_"
                    + (ref.loc[shared, "start_time"] // BIN_S).astype(int).astype(str))
        for metric, _, _ in TABLE_ROWS:
            a = pd.to_numeric(test.loc[shared, metric], errors="coerce")
            b = pd.to_numeric(ref.loc[shared, metric], errors="coerce")
            keep = a.notna() & b.notna()
            if (keep.sum() == 0):
                continue
            d = (a[keep] - b[keep]).to_numpy()
            low, high = bootstrap_ci(d, clusters[keep].to_numpy(), reps, rng)
            try:
                p_value = float(wilcoxon(d).pvalue) if ((len(d) >= 5) and (not np.allclose(d, 0.0))) else float("nan")
            except ValueError:
                p_value = float("nan")
            rows.append({
                "arm": arm, "reference": reference, "metric": metric, "n": int(keep.sum()),
                "clusters": int(clusters[keep].nunique()), "median_difference": float(np.median(d)),
                "mean_difference": float(d.mean()), "ci_low": low, "ci_high": high,
                "ci_excludes_zero": bool(np.isfinite(low) and ((low > 0.0) or (high < 0.0))), "p_value": p_value,
            })
    return pd.DataFrame(rows)


def aggregate_series(series: pd.DataFrame, arms: Sequence[str]) -> pd.DataFrame:
    """Per arm, step and quantity: median, quartiles and the number of episodes contributing."""
    quantities = [f"{p}_var" for p in POSITIONS] + [f"{p}_link" for p in LINKS]
    long = series.melt(id_vars=["arm", "episode", "step", "time_s"], value_vars=quantities, var_name="quantity")
    grouped = long.dropna(subset=["value"]).groupby(["arm", "quantity", "step"])["value"]
    table = pd.DataFrame({
        "time_s": long.groupby(["arm", "quantity", "step"])["time_s"].first(),
        "median": grouped.median(), "q25": grouped.quantile(0.25), "q75": grouped.quantile(0.75), "n": grouped.size(),
    }).reset_index()
    table["n"] = table["n"].fillna(0).astype(int)
    table["arm"] = pd.Categorical(table["arm"], categories=list(arms), ordered=True)
    return table.sort_values(["arm", "quantity", "step"]).reset_index(drop=True)


def aggregate_binned(binned: pd.DataFrame, arms: Sequence[str]) -> pd.DataFrame:
    """Per arm, ratio and bin: median, quartiles and the number of episodes with a value."""
    valid = binned.dropna(subset=["value"])
    grouped = valid.groupby(["arm", "ratio", "bin_start_s", "bin_end_s"])["value"]
    table = pd.DataFrame({
        "median": grouped.median(), "q25": grouped.quantile(0.25), "q75": grouped.quantile(0.75), "n": grouped.size(),
    }).reset_index()
    table["arm"] = pd.Categorical(table["arm"], categories=list(arms), ordered=True)
    order = {f"{top}/{bottom}": i for i, (top, bottom) in enumerate(BINNED_RATIOS)}
    table["ratio_order"] = table["ratio"].map(order)
    return table.sort_values(["ratio_order", "arm", "bin_start_s"]).drop(columns="ratio_order").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                         "grid.color": "#cccccc", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888888", "legend.frameon": False, "lines.linewidth": 2.0})
    return plt


def _save(figure, directory: str, name: str) -> None:
    os.makedirs(directory, exist_ok=True)
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(os.path.join(directory, f"{name}.{extension}"))
    print(f"wrote {os.path.join(directory, name)}.png")


def _draw(axis, aggregated: pd.DataFrame, quantity: str, arms: Sequence[str], skip_s: float, min_episodes: int) -> None:
    for colour, arm in zip(ARM_COLOURS, arms):
        rows = aggregated[(aggregated["arm"] == arm) & (aggregated["quantity"] == quantity) & (aggregated["n"] >= min_episodes)]
        if (rows.empty):
            continue
        axis.fill_between(rows["time_s"], rows["q25"], rows["q75"], color=colour, alpha=0.15, linewidth=0)
        axis.plot(rows["time_s"], rows["median"], color=colour, label=arm)
    axis.axvspan(0.0, skip_s, color="#888888", alpha=0.08, linewidth=0)
    if quantity.endswith("_link"):
        # Ratios are multiplicative (0.5 and 2 are equally far from 1), and a quiet leader makes
        # them spike, so a log axis keeps both halving and doubling readable.
        axis.set_yscale("log")
        axis.axhline(1.0, color="#888888", linewidth=0.8)


def plot_timeseries(aggregated: pd.DataFrame, arms: Sequence[str], directory: str, args) -> None:
    plt = _plt()
    groups = [
        ("oscillation_variance", [f"{p}_var" for p in POSITIONS], "speed variance (m$^2$/s$^2$)", lambda q: q[:-4]),
        ("oscillation_link_ratio", [f"{p}_link" for p in LINKS], "variance / vehicle ahead",
         lambda q: f"{q[:-5]} / {POSITIONS[POSITIONS.index(q[:-5]) - 1]}"),
    ]
    subtitle = f"rolling {args.oscillation_window_s:g} s variance; median and IQR across episodes; shaded: before {args.skip_s:g} s"
    for name, quantities, ylabel, title in groups:
        columns = min(4, len(quantities))
        rows = int(np.ceil(len(quantities) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(3.3 * columns, 2.5 * rows), squeeze=False, sharex=True)
        for index, quantity in enumerate(quantities):
            axis = axes[index // columns][index % columns]
            _draw(axis, aggregated, quantity, arms, args.skip_s, args.min_episodes)
            axis.set_title(title(quantity), fontsize=9)
            if (index % columns == 0):
                axis.set_ylabel(ylabel)
            if (index // columns == rows - 1):
                axis.set_xlabel("episode time (s)")
        for index in range(len(quantities), rows * columns):
            axes[index // columns][index % columns].axis("off")
        axes[0][0].legend(fontsize=8, loc="upper left")
        figure.suptitle(subtitle, fontsize=8, color="#555555")
        _save(figure, directory, name)
        plt.close(figure)
        for quantity in quantities:
            single, axis = plt.subplots(figsize=(4.8, 3.0))
            _draw(axis, aggregated, quantity, arms, args.skip_s, args.min_episodes)
            axis.set_title(title(quantity), fontsize=10)
            axis.set_ylabel(ylabel)
            axis.set_xlabel("episode time (s)")
            axis.legend(fontsize=8)
            _save(single, directory, f"{name}_{quantity}")
            plt.close(single)


def _draw_binned(axis, aggregated: pd.DataFrame, ratio: str, arms: Sequence[str], args) -> None:
    """Median per bin (markers joined by lines) with IQR whiskers, arms side by side within each bin."""
    offsets = np.linspace(-0.2, 0.2, len(arms)) * args.bin_s if (len(arms) > 1) else [0.0]
    for colour, arm, offset in zip(ARM_COLOURS, arms, offsets):
        rows = aggregated[(aggregated["arm"] == arm) & (aggregated["ratio"] == ratio) & (aggregated["n"] >= args.min_episodes)]
        if (rows.empty):
            continue
        centre = (rows["bin_start_s"] + rows["bin_end_s"]) / 2.0 + offset
        axis.errorbar(centre, rows["median"], yerr=[rows["median"] - rows["q25"], rows["q75"] - rows["median"]],
                      color=colour, marker="o", markersize=5, linewidth=1.5, elinewidth=1.0, capsize=2, label=arm)
    axis.set_yscale("log")
    # Panels whose range stays inside one decade would otherwise show a single "10^0" label.
    from matplotlib.ticker import LogLocator, FormatStrFormatter
    axis.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    axis.yaxis.set_minor_formatter(FormatStrFormatter(""))
    axis.axhline(1.0, color="#888888", linewidth=0.8)
    edges = np.arange(args.skip_s, aggregated["bin_end_s"].max() + 1e-9, args.bin_s)
    axis.set_xticks(edges)
    axis.set_xlim(edges[0], edges[-1])


def plot_binned(aggregated: pd.DataFrame, arms: Sequence[str], directory: str, args) -> None:
    if (aggregated.empty):
        return
    plt = _plt()
    ratios = [f"{top}/{bottom}" for top, bottom in BINNED_RATIOS]
    title = lambda ratio: ratio.replace("/", " / ")
    figure, axes = plt.subplots(1, len(ratios), figsize=(3.0 * len(ratios), 3.0), squeeze=False, sharex=True)
    for index, ratio in enumerate(ratios):
        axis = axes[0][index]
        _draw_binned(axis, aggregated, ratio, arms, args)
        axis.set_title(title(ratio), fontsize=9)
        axis.set_xlabel("episode time (s)")
        if (index == 0):
            axis.set_ylabel("speed variance ratio")
    axes[0][0].legend(fontsize=8, loc="best")
    figure.suptitle(f"{args.bin_s:g} s bins over {args.skip_s:g} s onward; median and IQR across episodes",
                    fontsize=8, color="#555555")
    _save(figure, directory, "binned_variance_ratio")
    plt.close(figure)
    for ratio in ratios:
        single, axis = plt.subplots(figsize=(4.8, 3.0))
        _draw_binned(axis, aggregated, ratio, arms, args)
        axis.set_title(title(ratio), fontsize=10)
        axis.set_ylabel("speed variance ratio")
        axis.set_xlabel("episode time (s)")
        axis.legend(fontsize=8)
        _save(single, directory, f"binned_variance_ratio_{ratio.replace('/', '_over_')}")
        plt.close(single)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def print_summary(summary: pd.DataFrame, arms: Sequence[str]) -> None:
    width = 26
    print("\n" + f"{'metric':<22}" + "".join(f"{arm:>{width}}" for arm in arms))
    print("-" * (22 + width * len(arms)))
    for metric, _, decimals in TABLE_ROWS:
        row = summary.set_index("metric").loc[metric]
        cells = [f"{_fmt(row[f'{a} median'], decimals)} [{_fmt(row[f'{a} min'], decimals)}, "
                 f"{_fmt(row[f'{a} max'], decimals)}] n={int(row[f'{a} n'])}" for a in arms]
        print(f"{metric:<22}" + "".join(f"{c:>{width}}" for c in cells))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", default=None, metavar="LABEL=DEMO_DIR:CONTROLLER",
                        help="an arm to compare; repeat for each, reference first (default: SUMO IDM, AVC original, AVC corridor)")
    parser.add_argument("--out", default=os.path.join(HERE, "run_data", "rl", "analysis", "stability"))
    parser.add_argument("--skip-s", type=float, default=30.0, help="summary metrics use steps with step * dt >= this")
    parser.add_argument("--min-samples-s", type=float, default=5.0, help="shared samples a variance or ratio needs")
    parser.add_argument("--oscillation-window-s", type=float, default=10.0, help="centred window of the rolling variance")
    parser.add_argument("--bin-s", type=float, default=5.0, help="bin width of the binned variance ratios over the summary window")
    parser.add_argument("--min-episodes", type=int, default=5, help="time-series steps drawn only where this many episodes contribute")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-figures", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    specs = [parse_arm(text) for text in (args.arm or DEFAULT_ARMS)]
    arms = [label for label, _, _ in specs]
    frames, series, binned = [], [], []
    for label, demo, controller in specs:
        print(f"reading {label}: {controller} in {demo}")
        metrics, per_step, per_bin = load_arm(label, demo, controller, args)
        frames.append(metrics)
        series.append(per_step)
        binned.append(per_bin)
    frame = pd.concat(frames, ignore_index=True)
    table_dir = os.path.join(args.out, "tables")
    os.makedirs(table_dir, exist_ok=True)

    frame.to_csv(os.path.join(table_dir, "episode_metrics.csv"), index=False)
    summary = summary_statistics(frame, arms)
    summary.to_csv(os.path.join(table_dir, "summary_statistics.csv"), index=False)
    with open(os.path.join(table_dir, "summary_statistics.tex"), "w") as handle:
        handle.write(latex_table(summary, arms, args))
    paired_vs_reference(frame, arms, args.bootstrap, args.seed).to_csv(
        os.path.join(table_dir, "paired_vs_reference.csv"), index=False
    )
    aggregated = aggregate_series(pd.concat(series, ignore_index=True), arms)
    aggregated.to_csv(os.path.join(table_dir, "oscillation_timeseries.csv"), index=False)
    binned_table = aggregate_binned(pd.concat(binned, ignore_index=True), arms)
    binned_table.to_csv(os.path.join(table_dir, "binned_variance_ratio.csv"), index=False)
    print(f"wrote episode_metrics, summary_statistics (.csv/.tex), paired_vs_reference, oscillation_timeseries "
          f"and binned_variance_ratio to {table_dir}")
    print_summary(summary, arms)
    if (not args.no_figures):
        plot_timeseries(aggregated, arms, os.path.join(args.out, "figures", "timeseries"), args)
        plot_binned(binned_table, arms, os.path.join(args.out, "figures", "timeseries"), args)


if __name__ == "__main__":
    main()
