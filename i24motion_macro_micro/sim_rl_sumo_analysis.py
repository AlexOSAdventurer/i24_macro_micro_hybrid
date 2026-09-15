"""Analysis of the RL hero-controller runs produced by sim_rl_sumo_training/demo.

Three things get analysed, from three sources on disk:

  training progress   ``<run>/progress.csv`` (SB3's logger) and
                      ``<run>/monitor_*.monitor.csv`` (one row per episode) --
                      learning curves for the return and for the traffic
                      quantities the reward is built out of.
  controller results  ``<demo>/episodes.csv`` -- every controller replayed on the
                      same episode specs, so the comparison is paired and a
                      signed-rank test on the per-episode differences is the
                      right test rather than an unpaired one across noisy days.
  single episodes     ``<demo>/traces/<controller>_<index>.csv`` -- the hero's
                      speed, acceleration and the macroscopic velocity ahead of
                      the bubble, which is where you can see whether a
                      controller is actually reacting to the downstream state or
                      just to its leader.

This module imports neither SUMO, gymnasium nor stable-baselines3 -- it reads
only the CSVs the other two leave behind -- but it still runs in the container
like everything else in this project::

    docker exec <container> bash -lc \\
        'cd /workspaces/i24motion_macro_micro && python3.10 sim_rl_sumo_analysis.py \\
             --run run_data/rl/ppo_smoothing --demo run_data/rl/ppo_smoothing/demo --latex'

It is algorithm-agnostic: RecurrentPPO, PPO and TRPO runs log the same rollout/
and episode/ series, and figure titles name which one a run used so the
recurrent/memoryless ablation and the TRPO runs can be told apart.

Figures land in ``<out>/figures`` and LaTeX tables in ``<out>/tables``, matching
how ``sim_analysis.py`` reports the macroscopic model comparison.
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_RUN_ROOT = os.path.join(HERE, "run_data", "rl")


# ---------------------------------------------------------------------------
# Metric vocabulary
# ---------------------------------------------------------------------------

# Every metric with the direction that counts as better, its units, and how it
# is written in a figure axis and in a LaTeX table.  "higher" for the things a
# smoothing controller is supposed to increase, "lower" for the costs.
METRICS: Dict[str, Dict[str, Any]] = {
    "return": {
        "better": "higher", "unit": "", "label": "Episode return", "latex": "Return",
    },
    "hero_mean_speed": {
        "better": "higher", "unit": "m/s", "label": "Hero speed", "latex": "Hero $\\bar{v}$",
    },
    "platoon_mean_speed": {
        "better": "higher", "unit": "m/s", "label": "Platoon speed", "latex": "Platoon $\\bar{v}$",
    },
    "platoon_speed_std": {
        "better": "lower", "unit": "m/s", "label": "Platoon speed spread", "latex": "Platoon $\\sigma_v$",
    },
    "acceleration_rms": {
        "better": "lower", "unit": "m/s$^2$", "label": "Acceleration RMS", "latex": "$a_{\\mathrm{rms}}$",
    },
    "jerk_rms": {
        "better": "lower", "unit": "m/s$^3$", "label": "Jerk RMS", "latex": "$j_{\\mathrm{rms}}$",
    },
    "min_headway": {
        "better": "higher", "unit": "s", "label": "Minimum time headway", "latex": "$h_{\\min}$",
    },
    "energy_per_metre": {
        "better": "lower", "unit": "J/m", "label": "Tractive energy", "latex": "Energy",
    },
}

DEFAULT_TABLE_METRICS = (
    "return", "platoon_mean_speed", "platoon_speed_std", "acceleration_rms",
    "jerk_rms", "min_headway", "energy_per_metre",
)

CONTROLLER_DISPLAY_NAMES = {
    "ppo": "PPO",
    "trpo": "TRPO",
    "sumo": "SUMO (uncontrolled)",
    "idm": "IDM",
    "follower_stopper": "FollowerStopper",
    "macro_lookahead": "Macro lookahead",
}

# Training-log series worth a learning curve, keyed by their column in SB3's
# progress.csv.
LEARNING_CURVE_SERIES = (
    ("rollout/ep_rew_mean", "Episode return"),
    ("rollout/ep_len_mean", "Episode length (steps)"),
    ("episode/platoon_mean_speed", "Platoon speed (m/s)"),
    ("episode/platoon_speed_std", "Platoon speed spread (m/s)"),
    ("episode/jerk_rms", "Jerk RMS (m/s$^3$)"),
    ("episode/energy_per_metre", "Tractive energy (J/m)"),
    ("episode/min_headway", "Minimum time headway (s)"),
    ("episode/fraction_completed", "Episodes reaching the corridor end"),
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def run_label(run_dir: str) -> str:
    """``<name> (<algorithm>)`` for figure titles.

    Naming the algorithm matters once recurrent and memoryless runs of the same
    configuration sit side by side: the two are otherwise indistinguishable from
    their logs, and comparing them is the point of the ``--algorithm ppo``
    ablation.
    """
    name = os.path.basename(os.path.normpath(run_dir))
    config_path = os.path.join(run_dir, "run_config.json")
    if (os.path.isfile(config_path)):
        try:
            import json

            with open(config_path, "r") as handle:
                algorithm = json.load(handle).get("train", {}).get("algorithm")
            if (algorithm):
                return f"{name} ({algorithm})"
        except Exception:
            pass
    return name


def load_progress(run_dir: str) -> pd.DataFrame:
    """SB3's per-update log. Empty frame when a run has not logged one yet."""
    path = os.path.join(run_dir, "progress.csv")
    if (not os.path.isfile(path)):
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if ("time/total_timesteps" in frame.columns):
        frame = frame.sort_values("time/total_timesteps")
    return frame


def load_monitor_episodes(run_dir: str) -> pd.DataFrame:
    """Every training episode, concatenated across the vectorised workers.

    Monitor writes a json header line before its own header row, hence
    ``skiprows=1``; ``t`` is seconds since that worker started, so the frames are
    ordered by it within a worker and pooled across workers afterwards.
    """
    frames = []
    for path in sorted(glob.glob(os.path.join(run_dir, "monitor_*.monitor.csv"))):
        try:
            frame = pd.read_csv(path, skiprows=1)
        except Exception as exc:
            print(f"WARNING: could not read {path}: {exc}")
            continue
        frame["worker"] = os.path.basename(path).split(".")[0]
        frames.append(frame)
    if (len(frames) == 0):
        return pd.DataFrame()
    episodes = pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)
    # A pooled step axis: episodes finish interleaved across workers, so the
    # cumulative length in completion order is the closest thing to "how much
    # experience had been collected when this episode ended".
    episodes["cumulative_steps"] = episodes["l"].cumsum()
    return episodes


def load_demo_episodes(demo_dir: str) -> pd.DataFrame:
    path = demo_dir if (os.path.isfile(demo_dir)) else os.path.join(demo_dir, "episodes.csv")
    if (not os.path.isfile(path)):
        raise FileNotFoundError(f"no episodes.csv at {path}; run sim_rl_sumo_demo.py first")
    return pd.read_csv(path)


def load_traces(demo_dir: str) -> Dict[Tuple[str, int], pd.DataFrame]:
    """Per-step traces keyed by (controller, episode index)."""
    traces: Dict[Tuple[str, int], pd.DataFrame] = {}
    for path in sorted(glob.glob(os.path.join(demo_dir, "traces", "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        controller, _, index = name.rpartition("_")
        if (not index.isdigit()):
            continue
        traces[(controller, int(index))] = pd.read_csv(path)
    return traces


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def summarize_controllers(
    episodes: pd.DataFrame, metrics: Sequence[str] = DEFAULT_TABLE_METRICS
) -> pd.DataFrame:
    """Mean, standard deviation and episode count per controller."""
    rows = []
    for controller, group in episodes.groupby("controller", sort=False):
        row: Dict[str, Any] = {"controller": controller, "episodes": len(group)}
        for metric in metrics:
            if (metric not in group.columns):
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if (len(values) > 0) else float("nan")
            row[f"{metric}_std"] = float(values.std(ddof=0)) if (len(values) > 0) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(
    episodes: pd.DataFrame,
    controller: str,
    reference: str,
    metric: str,
) -> Optional[np.ndarray]:
    """Per-episode ``controller - reference`` differences on the shared specs.

    Controllers are replayed on identical episode specs by the demo, so pairing
    on the episode index removes the day-to-day variance that otherwise swamps
    any controller effect -- across the corridor, the difference between a busy
    minute and a quiet one is far larger than the difference between two
    controllers on the same minute.
    """
    if ((metric not in episodes.columns) or ("episode" not in episodes.columns)):
        return None
    left = episodes[episodes["controller"] == controller].set_index("episode")[metric]
    right = episodes[episodes["controller"] == reference].set_index("episode")[metric]
    shared = left.index.intersection(right.index)
    if (len(shared) == 0):
        return None
    return (
        pd.to_numeric(left.loc[shared], errors="coerce")
        - pd.to_numeric(right.loc[shared], errors="coerce")
    ).dropna().to_numpy()


def paired_test(differences: np.ndarray) -> Dict[str, float]:
    """Wilcoxon signed-rank on the paired differences, with the mean effect.

    Signed-rank rather than a t-test because a handful of episodes is not enough
    to lean on normality, and the metric distributions here are visibly skewed
    (a single jam dominates the energy and headway columns).
    """
    result = {
        "n": float(len(differences)),
        "mean": float(np.mean(differences)) if (len(differences) > 0) else float("nan"),
        "median": float(np.median(differences)) if (len(differences) > 0) else float("nan"),
        "p_value": float("nan"),
    }
    if (len(differences) < 5) or np.allclose(differences, 0.0):
        # Below about five pairs the signed-rank test cannot reach any useful
        # p-value, so report the effect and leave the column empty.
        return result
    try:
        from scipy.stats import wilcoxon

        result["p_value"] = float(wilcoxon(differences).pvalue)
    except Exception as exc:
        print(f"WARNING: could not run the signed-rank test: {exc}")
    return result


def compare_to_reference(
    episodes: pd.DataFrame,
    reference: str = "sumo",
    metrics: Sequence[str] = DEFAULT_TABLE_METRICS,
) -> pd.DataFrame:
    """Paired effect of every controller against ``reference``, per metric."""
    controllers = [c for c in episodes["controller"].unique() if (c != reference)]
    rows = []
    for controller in controllers:
        for metric in metrics:
            differences = paired_differences(episodes, controller, reference, metric)
            if (differences is None) or (len(differences) == 0):
                continue
            statistics = paired_test(differences)
            reference_values = pd.to_numeric(
                episodes[episodes["controller"] == reference][metric], errors="coerce"
            ).dropna()
            baseline = float(reference_values.mean()) if (len(reference_values) > 0) else float("nan")
            rows.append({
                "controller": controller,
                "metric": metric,
                "n": statistics["n"],
                "mean_difference": statistics["mean"],
                "median_difference": statistics["median"],
                "percent_change": (
                    100.0 * statistics["mean"] / baseline if (abs(baseline) > 1e-12) else float("nan")
                ),
                "p_value": statistics["p_value"],
                "improved": _is_improvement(metric, statistics["mean"]),
            })
    return pd.DataFrame(rows)


def _is_improvement(metric: str, difference: float) -> bool:
    if (not np.isfinite(difference)):
        return False
    better = METRICS.get(metric, {}).get("better", "higher")
    return (difference > 0.0) if (better == "higher") else (difference < 0.0)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _figure_style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })
    return plt


def _save(figure, output_dir: str, name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    figure.tight_layout()
    figure.savefig(path)
    figure.savefig(os.path.join(output_dir, f"{name}.pdf"))
    print(f"wrote {path}")
    return path


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if ((window <= 1) or (len(values) < window)):
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_learning_curves(run_dir: str, output_dir: str, name: str = "learning_curves") -> Optional[str]:
    """Return and traffic quantities against environment steps."""
    progress = load_progress(run_dir)
    if (progress.empty) or ("time/total_timesteps" not in progress.columns):
        print(f"WARNING: no usable progress.csv under {run_dir}")
        return None
    plt = _figure_style()
    series = [(column, label) for column, label in LEARNING_CURVE_SERIES if (column in progress.columns)]
    if (len(series) == 0):
        return None
    columns = 2
    rows = int(np.ceil(len(series) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(9.0, 2.1 * rows), squeeze=False)
    steps = progress["time/total_timesteps"].to_numpy(dtype=float)
    for index, (column, label) in enumerate(series):
        axis = axes[index // columns][index % columns]
        values = pd.to_numeric(progress[column], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(values)
        axis.plot(steps[mask], values[mask], linewidth=1.4, color="#1f5fa8")
        axis.set_ylabel(label)
        axis.set_xlabel("environment steps")
    for index in range(len(series), rows * columns):
        axes[index // columns][index % columns].axis("off")
    figure.suptitle(f"Training progress: {run_label(run_dir)}", y=1.0)
    return _save(figure, output_dir, name)


def plot_training_episodes(
    run_dir: str, output_dir: str, window: int = 20, name: str = "training_episodes"
) -> Optional[str]:
    """Per-episode training metrics, raw and smoothed.

    The learning curves above are SB3's per-update means; this is the underlying
    episode scatter, which is where the spread across days and minutes shows up.
    """
    episodes = load_monitor_episodes(run_dir)
    if (episodes.empty):
        print(f"WARNING: no monitor_*.monitor.csv under {run_dir}")
        return None
    plt = _figure_style()
    panels = [
        ("r", "Episode return"),
        ("platoon_mean_speed", "Platoon speed (m/s)"),
        ("platoon_speed_std", "Platoon speed spread (m/s)"),
        ("jerk_rms", "Jerk RMS (m/s$^3$)"),
    ]
    panels = [(column, label) for column, label in panels if (column in episodes.columns)]
    figure, axes = plt.subplots(len(panels), 1, figsize=(8.0, 2.0 * len(panels)), squeeze=False, sharex=True)
    steps = episodes["cumulative_steps"].to_numpy(dtype=float)
    for index, (column, label) in enumerate(panels):
        axis = axes[index][0]
        values = pd.to_numeric(episodes[column], errors="coerce").to_numpy(dtype=float)
        axis.plot(steps, values, ".", markersize=2.5, alpha=0.35, color="#888888")
        smoothed = _smooth(values, window)
        if (len(smoothed) > 0):
            axis.plot(steps[len(steps) - len(smoothed):], smoothed, linewidth=1.6, color="#c1440e")
        axis.set_ylabel(label)
        axis.set_ylim(bottom=np.percentile(values, 10))
        axis.set_ylim(top=np.percentile(values, 90))
    axes[-1][0].set_xlabel("cumulative environment steps")
    figure.suptitle(f"Training episodes: {run_label(run_dir)}", y=1.0)
    return _save(figure, output_dir, name)


def plot_controller_comparison(
    episodes: pd.DataFrame,
    output_dir: str,
    metrics: Sequence[str] = DEFAULT_TABLE_METRICS,
    name: str = "controller_comparison",
) -> Optional[str]:
    """One panel per metric: the controller means with the per-episode points.

    The points are drawn because with a handful of paired episodes the mean on
    its own hides whether a controller wins everywhere or wins once by a lot.
    """
    if (episodes.empty):
        return None
    plt = _figure_style()
    controllers = list(dict.fromkeys(episodes["controller"]))
    metrics = [m for m in metrics if (m in episodes.columns)]
    columns = 2
    rows = int(np.ceil(len(metrics) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(9.0, 2.4 * rows), squeeze=False)
    positions = np.arange(len(controllers))
    for index, metric in enumerate(metrics):
        axis = axes[index // columns][index % columns]
        means, spreads = [], []
        for offset, controller in enumerate(controllers):
            values = pd.to_numeric(
                episodes[episodes["controller"] == controller][metric], errors="coerce"
            ).dropna().to_numpy()
            means.append(float(np.mean(values)) if (len(values) > 0) else np.nan)
            spreads.append(float(np.std(values)) if (len(values) > 0) else np.nan)
            jitter = (np.random.default_rng(offset).uniform(-0.12, 0.12, size=len(values)))
            axis.plot(np.full(len(values), offset) + jitter, values, ".", markersize=4, alpha=0.5, color="#444444")
        axis.bar(positions, means, yerr=spreads, capsize=3, alpha=0.45, color="#1f5fa8")
        info = METRICS.get(metric, {})
        unit = info.get("unit", "")
        axis.set_ylabel(f"{info.get('label', metric)}" + (f" ({unit})" if (unit) else ""))
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [CONTROLLER_DISPLAY_NAMES.get(c, c) for c in controllers], rotation=20, ha="right"
        )
        axis.set_title("higher is better" if (info.get("better") == "higher") else "lower is better", fontsize=7.5)
    for index in range(len(metrics), rows * columns):
        axes[index // columns][index % columns].axis("off")
    return _save(figure, output_dir, name)


def plot_episode_traces(
    traces: Dict[Tuple[str, int], pd.DataFrame],
    output_dir: str,
    episode: int = 0,
    name: Optional[str] = None,
) -> Optional[str]:
    """One episode, every controller: speed, acceleration and the macro lookahead.

    The bottom panel is the point of the coupled simulator -- the macroscopic
    velocity in the cell ahead of the bubble is the signal a controller can act
    on before its own leader has reacted, so a policy that is using it should
    show its speed turning over before the leader's does.
    """
    selected = {controller: frame for (controller, index), frame in traces.items() if (index == episode)}
    if (len(selected) == 0):
        print(f"WARNING: no traces for episode {episode}")
        return None
    plt = _figure_style()
    figure, axes = plt.subplots(3, 1, figsize=(8.5, 6.4), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for offset, (controller, frame) in enumerate(sorted(selected.items())):
        color = colors[offset % len(colors)]
        label = CONTROLLER_DISPLAY_NAMES.get(controller, controller)
        steps = frame["step"].to_numpy(dtype=float)
        axes[0].plot(steps, frame["hero_speed"], linewidth=1.4, color=color, label=label)
        axes[1].plot(steps, frame["realised_acceleration"], linewidth=1.2, color=color, label=label)
        if ("macro_velocity_ahead" in frame.columns):
            axes[2].plot(steps, frame["macro_velocity_ahead"], linewidth=1.2, color=color, label=label)
    axes[0].set_ylabel("hero speed (m/s)")
    axes[1].set_ylabel("realised $a$ (m/s$^2$)")
    axes[2].set_ylabel("macro $v$ ahead (m/s)")
    axes[2].set_xlabel("macroscopic step (1 s)")
    axes[0].legend(ncol=2, fontsize=8)
    figure.suptitle(f"Episode {episode}", y=1.0)
    return _save(figure, output_dir, name or f"episode_{episode}_traces")


# ---------------------------------------------------------------------------
# LaTeX reporting
# ---------------------------------------------------------------------------

# Preamble requirements for the generated tables: \usepackage{booktabs}.

_latex_escapes = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(text) -> str:
    return "".join(_latex_escapes.get(character, character) for character in str(text))


def _format(value: float, precision: int = 2) -> str:
    if ((value is None) or (not np.isfinite(value))):
        return "--"
    return "%.*f" % (precision, value)


def _best_controller(summary: pd.DataFrame, metric: str) -> Optional[str]:
    column = f"{metric}_mean"
    if (column not in summary.columns):
        return None
    values = summary[column]
    if (values.dropna().empty):
        return None
    better = METRICS.get(metric, {}).get("better", "higher")
    index = values.idxmax() if (better == "higher") else values.idxmin()
    return str(summary.loc[index, "controller"])


def generate_latex_for_controllers(
    episodes: pd.DataFrame,
    metrics: Sequence[str] = DEFAULT_TABLE_METRICS,
    precision: int = 2,
    caption: str = (
        "Hero-vehicle controllers evaluated on identical episodes of the coupled "
        "macroscopic/microscopic simulation. Each cell is the mean over episodes "
        "with the standard deviation across them in parentheses; the best "
        "controller in each column is bold."
    ),
    label: str = "tab:rl_controllers",
    font_size: str = r"\footnotesize",
    column_separation: str = "4pt",
    full_width: bool = False,
    output_path: Optional[str] = None,
) -> str:
    """Controller comparison table, in the reporting style of sim_analysis.py.

    Rows are controllers, columns are metrics; the arrow in each header says
    which direction is better, so the table can be read without the caption.
    """
    summary = summarize_controllers(episodes, metrics)
    metrics = [m for m in metrics if (f"{m}_mean" in summary.columns)]
    best = {metric: _best_controller(summary, metric) for metric in metrics}

    lines = []
    lines.append(r"\begin{table%s}[%s]" % ("*" if (full_width) else "", "!t" if (full_width) else "htbp"))
    lines.append(r"\centering")
    lines.append(r"\caption{%s}" % latex_escape(caption))
    lines.append(r"\label{%s}" % label)
    if (font_size):
        lines.append(font_size)
    if (column_separation):
        lines.append(r"\setlength{\tabcolsep}{%s}" % column_separation)
    lines.append(r"\begin{tabular}{%s}" % ("lr" + ("r" * len(metrics))))
    lines.append(r"\toprule")

    header = ["Controller", "Episodes"]
    for metric in metrics:
        info = METRICS.get(metric, {})
        arrow = r"$\uparrow$" if (info.get("better") == "higher") else r"$\downarrow$"
        unit = info.get("unit", "")
        # The units are already math-mode fragments in METRICS, so they go in raw.
        header.append(
            "%s %s%s" % (info.get("latex", latex_escape(metric)), arrow, (" (%s)" % unit) if (unit) else "")
        )
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for _, row in summary.iterrows():
        controller = str(row["controller"])
        cells = [latex_escape(CONTROLLER_DISPLAY_NAMES.get(controller, controller)), "%d" % int(row["episodes"])]
        for metric in metrics:
            cell = "%s (%s)" % (
                _format(row.get(f"{metric}_mean"), precision),
                _format(row.get(f"{metric}_std"), precision),
            )
            if (best.get(metric) == controller):
                cell = r"\textbf{%s}" % cell
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table%s}" % ("*" if (full_width) else ""))

    table = "\n".join(lines) + "\n"
    if (output_path is not None):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as handle:
            handle.write(table)
        print(f"wrote {output_path}")
    return table


def generate_latex_for_paired_comparison(
    episodes: pd.DataFrame,
    reference: str = "sumo",
    metrics: Sequence[str] = DEFAULT_TABLE_METRICS,
    precision: int = 2,
    caption: Optional[str] = None,
    label: str = "tab:rl_paired",
    font_size: str = r"\footnotesize",
    output_path: Optional[str] = None,
) -> str:
    """Paired differences against the reference controller, with signed-rank p."""
    comparison = compare_to_reference(episodes, reference, metrics)
    if (comparison.empty):
        return ""
    if (caption is None):
        caption = (
            "Per-episode differences against the %s controller on the same episodes "
            "(controller minus reference), with the Wilcoxon signed-rank p-value. "
            "Positive or negative is stated as an improvement or a regression "
            "according to each metric's direction." % CONTROLLER_DISPLAY_NAMES.get(reference, reference)
        )
    controllers = list(dict.fromkeys(comparison["controller"]))
    metrics = [m for m in metrics if (m in set(comparison["metric"]))]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{%s}" % latex_escape(caption))
    lines.append(r"\label{%s}" % label)
    if (font_size):
        lines.append(font_size)
    lines.append(r"\begin{tabular}{llrrr}")
    lines.append(r"\toprule")
    lines.append(r"Controller & Metric & $\Delta$ mean & Change (\%) & $p$ \\")
    lines.append(r"\midrule")
    for index, controller in enumerate(controllers):
        if (index > 0):
            lines.append(r"\midrule")
        for metric in metrics:
            rows = comparison[
                (comparison["controller"] == controller) & (comparison["metric"] == metric)
            ]
            if (rows.empty):
                continue
            row = rows.iloc[0]
            difference = _format(row["mean_difference"], precision)
            if (bool(row["improved"])):
                difference = r"\textbf{%s}" % difference
            p_value = row["p_value"]
            lines.append(
                " & ".join([
                    latex_escape(CONTROLLER_DISPLAY_NAMES.get(controller, controller)) if (metric == metrics[0]) else "",
                    METRICS.get(metric, {}).get("latex", latex_escape(metric)),
                    difference,
                    _format(row["percent_change"], 1),
                    "--" if (not np.isfinite(p_value)) else ("%.3f" % p_value),
                ]) + r" \\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table = "\n".join(lines) + "\n"
    if (output_path is not None):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as handle:
            handle.write(table)
        print(f"wrote {output_path}")
    return table


# ---------------------------------------------------------------------------
# Terminal reporting
# ---------------------------------------------------------------------------


def print_summary(episodes: pd.DataFrame, reference: str, metrics: Sequence[str]) -> None:
    summary = summarize_controllers(episodes, metrics)
    metrics = [m for m in metrics if (f"{m}_mean" in summary.columns)]
    header = f"{'controller':<20}{'n':>4}" + "".join(f"{m[:12]:>14}" for m in metrics)
    print()
    print(header)
    print("-" * len(header))
    for _, row in summary.iterrows():
        line = f"{str(row['controller']):<20}{int(row['episodes']):>4}"
        for metric in metrics:
            line += f"{row.get(f'{metric}_mean', float('nan')):>14.3f}"
        print(line)

    comparison = compare_to_reference(episodes, reference, metrics)
    if (comparison.empty):
        return
    print()
    print(f"paired differences against '{reference}' (bold = improvement, p from Wilcoxon signed-rank)")
    header = f"{'controller':<20}{'metric':<22}{'delta':>10}{'change %':>10}{'p':>8}{'':>4}"
    print(header)
    print("-" * len(header))
    for _, row in comparison.iterrows():
        p_value = row["p_value"]
        print(
            f"{row['controller']:<20}{row['metric']:<22}{row['mean_difference']:>10.3f}"
            f"{row['percent_change']:>10.1f}"
            f"{'    --' if (not np.isfinite(p_value)) else f'{p_value:>8.3f}'}"
            f"{'  +' if (row['improved']) else '  -':>4}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", default=None, help="training run directory (for learning curves)")
    parser.add_argument("--demo", default=None, help="demo output directory (for the controller comparison)")
    parser.add_argument("--out", default=None, help="where figures/ and tables/ go (default: the demo or run directory)")
    parser.add_argument("--reference", default="sumo", help="controller the paired differences are taken against")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_TABLE_METRICS))
    parser.add_argument("--episode", type=int, default=0, help="which episode's traces to plot")
    parser.add_argument("--smooth-window", type=int, default=20)
    parser.add_argument("--latex", action="store_true", help="also write the LaTeX tables")
    parser.add_argument("--no-figures", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if ((args.run is None) and (args.demo is None)):
        raise SystemExit("pass --run (a training run), --demo (demo results), or both")

    output_root = args.out or args.demo or args.run
    figure_dir = os.path.join(output_root, "figures")
    table_dir = os.path.join(output_root, "tables")

    if (args.run is not None) and (not args.no_figures):
        plot_learning_curves(args.run, figure_dir)
        plot_training_episodes(args.run, figure_dir, window=args.smooth_window)

    if (args.demo is not None):
        episodes = load_demo_episodes(args.demo)
        metrics = [m for m in args.metrics if (m in episodes.columns)]
        print_summary(episodes, args.reference, metrics)
        if (not args.no_figures):
            plot_controller_comparison(episodes, figure_dir, metrics)
            plot_episode_traces(load_traces(args.demo), figure_dir, episode=args.episode)
        if (args.latex):
            generate_latex_for_controllers(
                episodes, metrics, output_path=os.path.join(table_dir, "rl_controllers.tex")
            )
            generate_latex_for_paired_comparison(
                episodes, args.reference, metrics,
                output_path=os.path.join(table_dir, "rl_paired.tex"),
            )


if __name__ == "__main__":
    main()
