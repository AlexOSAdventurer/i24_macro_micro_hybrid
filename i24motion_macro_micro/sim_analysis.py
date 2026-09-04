import os
import math
import numpy as np
import scipy
import rollout_store

config_folder = "config/"
database_file = "run_data/results_final.db"

def load_data():
    return rollout_store.RolloutStore(database_file, readonly=True)

def get_masks(store, run_id, road_id, lane_id):
    mask_sections = store.axes(run_id, road_id, lane_id)["mask_extents"]
    extents = []
    for section in mask_sections:
        extents = extents + section
    return extents

def extract_paranthesis_str(str):
    return str.split("(")[1].split(")")[0]

def extract_paranthesis_float(str):
    return float(extract_paranthesis_str(str))

def extract_run_data(store, run_id, baseline=False):
    if (baseline):
        # We want to use the last _ breakpoint and treat the first str (no matter how many underscores)
        # as completely separate. Thus, some string karate.
        base_info = run_id[::-1].split("_", 1)
        for i, str in enumerate(base_info):
            base_info[i] = str[::-1]
        base_info = base_info[::-1]
        sim_type = base_info[0]
        road_id = ["1", "2"]
        lanes = [-1, -2, -3, -4]
        time_origin_sim = store.metadata(run_id)["config_folder"]["time_origin"]
        time_origin_empirical = store.metadata(run_id)["config_folder"]["time_origin"]
        time_step = store.metadata(run_id)["config_folder"]["time_step"]
        episode_length = store.metadata(run_id)["config_folder"]["time_length"] - 1.0
        dataset = base_info[1]
    else:
        base_info = run_id.split("_")
        sim_type = base_info[0]
        road_id = [extract_paranthesis_str(base_info[1])]
        lanes = [-1, -2, -3, -4]
        time_origin_sim = extract_paranthesis_float(base_info[2])
        time_origin_empirical = store.metadata(run_id)["config_folder"]["time_origin"]
        time_step = store.metadata(run_id)["config_folder"]["time_step"]
        episode_length = extract_paranthesis_float(base_info[3])
        dataset = base_info[4]
    return sim_type, road_id, lanes, time_origin_sim, time_origin_empirical, episode_length, time_step, dataset

# This assumes a cell s origin of 0 and a cell length of 100.
def _get_highest_upstream_cell_id(s: float, s_origin: float = 0.0, cell_length: float = 100.0):
    return math.floor((s - s_origin) / cell_length) - 1

def get_sim_and_empirical_densities(store, run_id, road_id, lane_id, baseline=False):
    sim_data = store[run_id, road_id, lane_id, "sim", "density"]
    unaligned_empirical_data = store[run_id, road_id, lane_id, "empirical", "density"]
    _, _, _, time_origin_sim, time_origin_empirical, episode_length, time_step, _ = extract_run_data(store, run_id, baseline)
    empirical_alignment_start_index = int((time_origin_sim - time_origin_empirical) / time_step)
    empirical_alignment_end_index = empirical_alignment_start_index + int(episode_length / time_step)
    aligned_empirical_data = unaligned_empirical_data[empirical_alignment_start_index:empirical_alignment_end_index, :]
    return sim_data, aligned_empirical_data

def pair_and_filter_for_upstream_data(store, run_id, road_id, lane_id, sim_data, empirical_data):
    masks = get_masks(store, run_id, road_id, lane_id)
    pairs = []
    for i, mask in enumerate(masks):
        start_cell = 0
        end_cell = sim_data.shape[1] - 1
        if (mask is not None):
            end_cell = _get_highest_upstream_cell_id(mask[0])
        pair_sim = sim_data[i][start_cell:(end_cell + 1)]
        pair_empirical = empirical_data[i][start_cell:(end_cell + 1)]
        pairs.append([pair_sim, pair_empirical])
    return pairs

def calculate_metrics_from_pairs(pairs):
    metrics = {
        "l1": [],
        "wasserstein": []
    }
    for (pair_sim, pair_empirical) in pairs:
        x = np.arange(0, 1600.0, 100.0)[:pair_sim.shape[0]]
        l1_error = float(np.sum(np.abs(pair_sim - pair_empirical)) / pair_sim.shape[0])
        wasserstein_error = float(scipy.stats.wasserstein_distance(x, x, pair_sim, pair_empirical))
        metrics["l1"].append(l1_error)
        metrics["wasserstein"].append(wasserstein_error)
    return metrics

def join_metrics(metric1, metric2):
    metrics = {
        "l1": metric1["l1"] + metric2["l1"],
        "wasserstein": metric1["wasserstein"] + metric2["wasserstein"]
    }
    return metrics

# Basic Rule saying that if lwr_triangular is in the string we go with baseline.
# Works for now but will need to be changed
def str_is_baseline(run):
    return ("lwr_triangular" in run)

def collate_metrics_for_sim_and_dataset(store, sim, dataset):
    metrics = {
        "l1": [],
        "wasserstein": []
    }
    baseline = str_is_baseline(sim)
    runs = [run for run in store.runs() if (sim == extract_run_data(store, run, str_is_baseline(run))[0]) and (dataset == extract_run_data(store, run, str_is_baseline(run))[-1])]
    for run in runs:
        _, road_id, lanes, time_origin_sim, time_origin_empirical, episode_length, time_step, _ = extract_run_data(store, run, baseline)
        for current_road_id in road_id:
            for current_lane_id in lanes:
                sim_data, empirical_data = get_sim_and_empirical_densities(store, run, current_road_id, current_lane_id, baseline)
                pairs = pair_and_filter_for_upstream_data(store, run, current_road_id, current_lane_id, sim_data, empirical_data)
                new_metrics = calculate_metrics_from_pairs(pairs)
                metrics = join_metrics(metrics, new_metrics)
    return metrics

def generate_sim_metrics(store, sim):
    datasets = os.listdir(config_folder)
    summary_metrics = {
        "l1": [],
        "wasserstein": []
    }
    dataset_metrics = {}
    for dataset in datasets:
        dataset_metrics[dataset] = collate_metrics_for_sim_and_dataset(store, sim, dataset)
        summary_metrics = join_metrics(summary_metrics, dataset_metrics[dataset])

    return summary_metrics, dataset_metrics

def generate_full_metrics(store):
    sims = ["lwr_triangular", "sumo", "carla"]
    results = {}
    for sim in sims:
        results[sim] = {}
        results[sim]["summary_metrics"], results[sim]["dataset_metrics"] = generate_sim_metrics(store, sim)
    return results

# ---------------------------------------------------------------------------
# LaTeX reporting
# ---------------------------------------------------------------------------

# Preamble requirements for the generated table: \usepackage{booktabs}.

sim_display_names = {
    "lwr_triangular": "LWR",
    "sumo": "SUMO",
    "carla": "CARLA"
}

metric_display_names = {
    "l1": r"$L^1$ error",
    "wasserstein": "Wasserstein distance"
}

# Narrow variants, used when compact=True.  A \multicolumn header wider than the
# stat columns it spans stretches the whole table, so the long names are the
# single most expensive thing in the header.
metric_short_names = {
    "l1": r"$L^1$",
    "wasserstein": r"$W_1$"
}

stat_short_names = {
    "min": "Min",
    "mean": "Mean",
    "median": "Med.",
    "std": "SD",
    "max": "Max"
}

stat_display_names = {
    #"min": "Min",
    "mean": "Mean",
    "median": "Median",
    "std": "Std",
    "max": "Max"
}

stat_order = ["mean", "median", "std", "max"]

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
    "^": r"\textasciicircum{}"
}

def latex_escape(text):
    return "".join(_latex_escapes.get(character, character) for character in str(text))

def format_dataset_name(dataset, compact=False):
    name = os.path.splitext(str(dataset))[0]
    if (compact):
        # "2022-11-21" -> "11-21"; every dataset shares the same year.
        parts = name.split("-")
        if (len(parts) == 3) and (len(parts[0]) == 4):
            name = "-".join(parts[1:])
    return latex_escape(name)

def summarize_metric_values(values):
    if (values is None) or (len(values) == 0):
        return None
    values = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "max": float(np.max(values))
    }

def format_metric_value(value, precision=4, scientific=False, scale=1.0):
    if (value is None) or (not np.isfinite(value)):
        return "--"
    value = value * scale
    if (scientific):
        mantissa, exponent = ("%.*e" % (precision, value)).split("e")
        return r"$%s \times 10^{%d}$" % (mantissa, int(exponent))
    return "%.*f" % (precision, value)

def _bold_cell(cell):
    # \textbf does not carry into math mode, so scientific cells need \boldmath.
    if (cell.startswith("$")):
        return r"{\boldmath%s}" % cell
    return r"\textbf{%s}" % cell

def _build_metric_row_cells(metrics, metric_keys, stats, precision, scientific, scale):
    cells = []
    for metric_key in metric_keys:
        summary = summarize_metric_values(None if (metrics is None) else metrics.get(metric_key))
        for stat in stats:
            cells.append(format_metric_value(None if (summary is None) else summary[stat], precision, scientific, scale))
    return cells

def _build_sim_block_lines(results, sim, datasets, metric_keys, stats, precision, scientific, scale, compact, total_columns):
    lines = []
    sim_label = sim_display_names.get(sim, latex_escape(sim))
    for i, dataset in enumerate(datasets):
        row = [sim_label if (i == 0) else "", format_dataset_name(dataset, compact)]
        row.extend(_build_metric_row_cells(results[sim]["dataset_metrics"].get(dataset), metric_keys, stats, precision, scientific, scale))
        lines.append(" & ".join(row) + r" \\")
    summary_row = ["", r"\textbf{%s}" % ("All" if (compact) else "All datasets")]
    summary_row.extend(_bold_cell(cell) for cell in _build_metric_row_cells(results[sim]["summary_metrics"], metric_keys, stats, precision, scientific, scale))
    lines.append(r"\cmidrule(lr){2-%d}" % total_columns)
    lines.append(" & ".join(summary_row) + r" \\")
    return lines

def generate_latex_for_metrics(store=None,
                               results=None,
                               metric_keys=("l1", "wasserstein"),
                               stats=None,
                               precision=2,
                               scientific=False,
                               scale=1000.0,
                               scale_label=r"$(\times 10^{-3})$",
                               metric_layout="columns",
                               compact=True,
                               font_size=r"\footnotesize",
                               column_separation="4pt",
                               full_width=False,
                               caption="Upstream density error of each simulation technique against the empirical I-24 MOTION data, per dataset and aggregated over all datasets.",
                               label="tab:full_metrics",
                               output_path=None):
    """Build a LaTeX table summarising generate_full_metrics' results dictionary.

    Rows are grouped per simulation technique (LWR / SUMO / CARLA); within a
    group there is one row per dataset plus a final "All datasets" row built
    from that technique's summary metrics.

    Width control, in decreasing order of how much space it buys:

    scale / scale_label
        Factor the common magnitude out of every cell and state it once in the
        header, so cells read "2.94" instead of "$2.9380 \\times 10^{-3}$".
        With densities in veh/m, scale=1000.0 makes the numbers veh/km, so
        scale_label="veh/km" is an equally valid header.
    metric_layout
        "columns" puts the metrics side by side (2 + len(stats) * len(metrics)
        columns).  "rows" stacks them as separate row blocks under one set of
        stat columns (2 + len(stats) columns), which roughly halves the width
        at the cost of twice the rows.
    stats
        Which of min / mean / median / std / max to emit; defaults to the
        module-level stat_order.  Every stat dropped is a whole column.
    compact
        Short headers ($L^1$, $W_1$, Med., SD) and short dataset labels
        ("11-21"), so no header is wider than the numbers beneath it.
    precision, font_size, column_separation, full_width
        Digits after the point, the size command applied inside the float,
        \\tabcolsep, and table vs. table* (two-column span).

    Returns the table as a string and, when output_path is given, also writes
    it to that file.
    """
    if (results is None):
        if (store is None):
            store = load_data()
        results = generate_full_metrics(store)

    metric_keys = list(metric_keys)
    stats = list(stat_order if (stats is None) else stats)
    if (metric_layout not in ("columns", "rows")):
        raise ValueError("metric_layout must be 'columns' or 'rows', got %r" % (metric_layout,))

    datasets = []
    for sim in results:
        for dataset in results[sim]["dataset_metrics"]:
            if (dataset not in datasets):
                datasets.append(dataset)
    datasets = sorted(datasets)
    sims = [sim for sim in ["lwr_triangular", "sumo", "carla"] if (sim in results)]

    metric_names = metric_short_names if (compact) else metric_display_names
    stat_names = stat_short_names if (compact) else stat_display_names
    metrics_per_row = len(metric_keys) if (metric_layout == "columns") else 1
    total_columns = 2 + (len(stats) * metrics_per_row)

    lines = []
    lines.append(r"\begin{table%s}[%s]" % ("*" if (full_width) else "", "!t" if (full_width) else "htbp"))
    lines.append(r"\centering")
    lines.append(r"\caption{%s%s}" % (latex_escape(caption), "" if (scale == 1.0) else (" All values are scaled by %g." % scale)))
    lines.append(r"\label{%s}" % label)
    if (font_size):
        lines.append(font_size)
    if (column_separation):
        lines.append(r"\setlength{\tabcolsep}{%s}" % column_separation)
    lines.append(r"\begin{tabular}{%s}" % ("ll" + ("r" * (total_columns - 2))))
    lines.append(r"\toprule")

    def metric_header(metric_key):
        name = metric_names.get(metric_key, latex_escape(metric_key))
        return name if (not scale_label) else ("%s %s" % (name, scale_label))

    if (metric_layout == "columns"):
        header_groups = ["", ""]
        header_rules = []
        for i, metric_key in enumerate(metric_keys):
            header_groups.append(r"\multicolumn{%d}{c}{%s}" % (len(stats), metric_header(metric_key)))
            first_column = 3 + (i * len(stats))
            header_rules.append(r"\cmidrule(lr){%d-%d}" % (first_column, first_column + len(stats) - 1))
        lines.append(" & ".join(header_groups) + r" \\")
        lines.append(" ".join(header_rules))

    header_stats = ["Model", "Dataset"]
    for _ in range(metrics_per_row):
        header_stats.extend(stat_names.get(stat, stat.capitalize()) for stat in stats)
    lines.append(" & ".join(header_stats) + r" \\")
    lines.append(r"\midrule")

    if (metric_layout == "columns"):
        for i, sim in enumerate(sims):
            if (i > 0):
                lines.append(r"\midrule")
            lines.extend(_build_sim_block_lines(results, sim, datasets, metric_keys, stats, precision, scientific, scale, compact, total_columns))
    else:
        for i, metric_key in enumerate(metric_keys):
            if (i > 0):
                lines.append(r"\midrule")
            lines.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (total_columns, metric_header(metric_key)))
            lines.append(r"\addlinespace[2pt]")
            for j, sim in enumerate(sims):
                if (j > 0):
                    lines.append(r"\addlinespace[2pt]")
                lines.extend(_build_sim_block_lines(results, sim, datasets, [metric_key], stats, precision, scientific, scale, compact, total_columns))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table%s}" % ("*" if (full_width) else ""))

    table = "\n".join(lines) + "\n"
    if (output_path is not None):
        with open(output_path, "w") as output_file:
            output_file.write(table)
    return table

if __name__ == "__main__":
    store = load_data()
    generate_latex_for_metrics(store, precision=2, scientific=True, output_path="results.tex", full_width=True, scale=1, metric_layout="columns", scale_label="")