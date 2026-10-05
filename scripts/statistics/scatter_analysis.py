"""
n_SLH vs Delta Metric Scatter Analysis
=======================================

Generates scatter plots showing the relationship between SLH episode count
and counterfactual metric changes for a selected zone-expansion offset.

Uses pre-computed results from:
- Episode detection (n_SLH counts)
- Sensitivity analysis (delta metrics)

Outputs for each metric:
- Scatter plot with regression line
- Spearman correlation with 95% CI
- Points colored by beneficial/detrimental direction
- Statistical annotations

Artwork specification
---------------------
No title is drawn inside the artwork.

Glucose-valued deltas (mean, median, SD) follow the process-wide display
unit, so ``CGM_GLUCOSE_UNIT=mmol/L`` produces figures with no mg/dL left in
them; percentage metrics are unit-independent.

Authors: Guillaume Deside

Usage
-----
    python3 -m scripts.statistics.scatter_analysis
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats

# Import configuration
from src.config import (
    COHORT_SENSITIVITY_CSV,
    EPISODES_SUMMARY_CSV,
    SCATTER_ANALYSIS_PATH,
    SCATTER_FIGURES_PATH,
    SCATTER_STATS_CSV,
)
from src.config.figure_config import LANCET_MIN_FONT_PT, lancet_figsize
from src.counterfactual.counterfactual_config import (
    APPLY_KERNEL,
    KERNEL_MIN,
    SELECTED_OFFSET_MIN,
)

# Import data loading
from src.episodes.load_episodes import load_episode_summary
from src.counterfactual.load_counterfactual import load_cohort_sensitivity_results as load_sensitivity_results

# Import visualization
from src.visualization.style import (
    apply_lancet_style,
    enforce_lancet_minimums,
    save_figure,
    style_axes,
)
from src.visualization.colors import (
    COLOR_BENEFICIAL,
    COLOR_DETRIMENTAL,
    COLOR_NEUTRAL,
    COLOR_REGRESSION,
    COLOR_ZERO_LINE,
)
from src.visualization.glucose_axis import apply_glucose_axis
from src.utils.unit_conversion import glucose_unit_label, is_glucose_valued_column


# =============================================================================
# CONFIGURATION
# =============================================================================

SELECTED_OFFSET = SELECTED_OFFSET_MIN

# ── Output paths ─────────────────────────────────────────────────────────────
OUTPUT_DIR  = str(SCATTER_ANALYSIS_PATH)
FIGURES_DIR = str(SCATTER_FIGURES_PATH)
STATS_CSV   = str(SCATTER_STATS_CSV)

# ── Metrics to analyze ───────────────────────────────────────────────────────
METRICS = [
    "TIR70-180",
    "TAR>180",
    "TAR>250",
    "TBR<70",
    "mean",
    "median",
    "sd",
    "cv",
]

# Metric display properties.  The unit is kept separate from the name so that
# glucose-valued metrics can follow the display unit; see metric_label().
METRIC_LABELS = {
    "TIR70-180": "ΔTIR 70–180",
    "TAR>180":   "ΔTAR >180",
    "TAR>250":   "ΔTAR >250",
    "TBR<70":    "ΔTBR <70",
    "mean":      "ΔMean glucose",
    "median":    "ΔMedian glucose",
    "sd":        "ΔSD",
    "cv":        "ΔCV",
}

METRIC_UNITS = {
    "TIR70-180": "(%)",
    "TAR>180":   "(%)",
    "TAR>250":   "(%)",
    "TBR<70":    "(%)",
    "mean":      "(mg/dL)",
    "median":    "(mg/dL)",
    "sd":        "(mg/dL)",
    "cv":        "(%)",
}

#: Axis label for the x variable, used identically everywhere.
N_SLH_LABEL = "Number of SLH episodes"


def metric_unit(metric: str) -> str:
    """METRIC_UNITS entry, with mg/dL replaced by the current display unit."""
    unit = METRIC_UNITS[metric]

    return f"({glucose_unit_label()})" if unit == "(mg/dL)" else unit


def metric_label(metric: str) -> str:
    """Axis label for a metric: name plus unit in the current display unit."""
    return f"{METRIC_LABELS[metric]} {metric_unit(metric)}"


def format_p_value(p: float) -> str:
    """
    Format a p value in house style: two significant figures, or ``p<0.0001``.

    Replaces the ``*`` / ``**`` significance stars, which say less than the
    value itself.
    """
    if p < 0.0001:
        return "p<0.0001"

    return f"p={float(f'{p:.2g}'):g}"

# Beneficial direction (+1 = higher is better, -1 = lower is better)
METRIC_DIRECTION = {
    "TIR70-180": +1,
    "TAR>180":   -1,
    "TAR>250":   -1,
    "TBR<70":    -1,
    "mean":      -1,
    "median":    -1,
    "sd":        -1,
    "cv":        -1,
}

# ── Daytime vs all-day delta column prefix ───────────────────────────────────
# "delta_"         -> all-day deltas (default)
# "daytime_delta_" -> deltas restricted to daytime (06:00-22:00) episodes only
DELTA_PREFIX_ALL_DAY = "delta_"
DELTA_PREFIX_DAYTIME = "daytime_delta_"

# ── Representative patients highlighted on scatter plots (optional) ──────────
PATIENT_IDS = []   # e.g. ["A174", "A224"] to highlight specific patients
PATIENT_COLORS = {
    "A174": "#E69F00",   # Orange
    "A224": "#CC79A7",   # Purple
}
PATIENT_MARKERS = {
    "A174": "D",         # Diamond
    "A224": "s",         # Square
}

# ── Figure parameters (column widths) ─────────────────────────────────
# A single scatter is drawn two columns wide (115 mm) and the grid at full page
# width (175 mm); see src.config.figure_config.
FIGURE_COLUMNS_SINGLE = "double"   # 115 mm — one scatter
FIGURE_COLUMNS_GRID   = "full"     # 175 mm — every metric
FIGURE_HEIGHT         = 3.5        # Single scatter height

# Grid layout: two columns rather than three.  At 175 mm, three columns leave
# each panel under 50 mm wide, which is too small for a 130-point scatter.
GRID_NCOLS               = 2
GRID_PANEL_HEIGHT_INCHES = 2.05
GRID_LEGEND_HEIGHT_INCHES = 0.35

# ── Scatter styling ──────────────────────────────────────────────────────────
POINT_SIZE       = 25
POINT_ALPHA      = 0.65
POINT_EDGE_WIDTH = 0

# ── Reference lines ──────────────────────────────────────────────────────────
ZERO_LINE_STYLE = ":"
ZERO_LINE_WIDTH = 0.7

# ── Statistical parameters ───────────────────────────────────────────────────
CI_LEVEL            = 0.95    # Confidence level for Fisher z-transform
MIN_POINTS_SPEARMAN = 8       # Minimum points required for Spearman
N_BOOTSTRAP         = 10000   # Bootstrap iterations for CI


# =============================================================================
# DATA LOADING
# =============================================================================

def merge_data(episodes_df: pd.DataFrame,
               sensitivity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge episode counts with sensitivity results.

    Parameters
    ----------
    episodes_df : pd.DataFrame
        Episode summary with n_SLH
    sensitivity_df : pd.DataFrame
        Sensitivity results with delta metrics

    Returns
    -------
    pd.DataFrame
        Merged data
    """
    merged = sensitivity_df.merge(
        episodes_df[["patient_id", "n_SLH"]],
        on="patient_id",
        how="inner"
    )

    print(f"  Merged data: {len(merged)} patients")

    return merged


# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================
def fisher_ci(r: float, n: int, level: float = CI_LEVEL) -> tuple:
    """
    Compute 95% CI for Spearman r via Fisher z-transformation.

    Parameters
    ----------
    r : float
        Correlation coefficient
    n : int
        Sample size
    level : float
        Confidence level (default 0.95)

    Returns
    -------
    (ci_low, ci_high)
    """
    if n < 4 or abs(r) >= 1.0:
        return (np.nan, np.nan)

    z  = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    zc = stats.norm.ppf(1 - (1 - level) / 2)

    ci_low  = np.tanh(z - zc * se)
    ci_high = np.tanh(z + zc * se)

    return (round(ci_low, 4), round(ci_high, 4))


def compute_statistics(data: pd.DataFrame, metric: str,
                       delta_prefix: str = DELTA_PREFIX_ALL_DAY) -> dict:
    """
    Compute statistics for one metric.

    Parameters
    ----------
    delta_prefix : str
        Column prefix selecting all-day ("delta_") or daytime-only
        ("daytime_delta_") metric deltas.

    Returns dict with:
    - spearman_r, spearman_p
    - ci_low, ci_high
    - r_squared
    - ols_slope, ols_intercept, ols_r, ols_p
    - n_points
    """
    delta_col = f"{delta_prefix}{metric}"

    valid_data = data[["n_SLH", delta_col]].dropna()

    if len(valid_data) < MIN_POINTS_SPEARMAN:
        return None

    x = valid_data["n_SLH"].values
    y = valid_data[delta_col].values

    spearman_r, spearman_p = stats.spearmanr(x, y)
    ci_low, ci_high = fisher_ci(spearman_r, len(x))

    ols_slope, ols_intercept, ols_r, ols_p, ols_stderr = stats.linregress(x, y)

    return {
        "metric":        metric,
        "n_points":      len(x),
        "spearman_r":    round(spearman_r, 4),
        "spearman_p":    round(spearman_p, 5),
        "ci_low":        ci_low,
        "ci_high":       ci_high,
        "r_squared":     round(spearman_r ** 2, 4),
        "ols_slope":     round(ols_slope, 6),
        "ols_intercept": round(ols_intercept, 4),
        "ols_r":         round(ols_r, 4),
        "ols_p":         round(ols_p, 5),
    }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
PANEL_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def assign_point_colors(delta_values: np.ndarray, metric: str) -> list:
    """
    Assign colors to points based on beneficial/detrimental direction.

    Parameters
    ----------
    delta_values : np.ndarray
        Delta metric values
    metric : str
        Metric name

    Returns
    -------
    list of colors
    """
    direction = METRIC_DIRECTION.get(metric, 1)

    colors = []
    for delta in delta_values:
        if delta * direction > 0:
            colors.append(COLOR_BENEFICIAL)
        elif delta * direction < 0:
            colors.append(COLOR_DETRIMENTAL)
        else:
            colors.append(COLOR_NEUTRAL)

    return colors


def direction_legend_handles(colors: list) -> list:
    """
    Legend keys for the point colours actually used.

    A metric whose every participant moved in the beneficial direction should
    not carry a "detrimental" key, and the grey "no change" points need one
    when they are there.
    """
    present = set(colors)

    keys = [
        (COLOR_BENEFICIAL, "Beneficial change"),
        (COLOR_DETRIMENTAL, "Detrimental change"),
        (COLOR_NEUTRAL, "No change"),
    ]

    return [
        mpatches.Patch(facecolor=color, alpha=POINT_ALPHA, label=label)
        for color, label in keys
        if color in present
    ]


def create_scatter_plot(data: pd.DataFrame, metric: str,
                        stats_dict: dict, offset: int,
                        delta_prefix: str = DELTA_PREFIX_ALL_DAY,
                        patient_ids: list = None) -> plt.Figure:
    """
    Create scatter plot for one metric.

    Parameters
    ----------
    data : pd.DataFrame
        Analysis data
    metric : str
        Metric name
    stats_dict : dict
        Statistics dictionary
    offset : int
        Zone-expansion offset (minutes)
    delta_prefix : str
        Column prefix selecting all-day or daytime-only deltas.
    patient_ids : list of str, optional
        Patient IDs to highlight with a distinct marker/color.

    Returns
    -------
    plt.Figure
    """
    apply_lancet_style()

    delta_col = f"{delta_prefix}{metric}"
    patient_ids = patient_ids or []

    plot_data = data[["patient_id", "n_SLH", delta_col]].dropna()

    if len(plot_data) < MIN_POINTS_SPEARMAN:
        return None

    x = plot_data["n_SLH"].values
    y = plot_data[delta_col].values

    colors = assign_point_colors(y, metric)

    # No title is drawn inside the artwork.
    fig, ax = plt.subplots(
        figsize=lancet_figsize(FIGURE_HEIGHT, columns=FIGURE_COLUMNS_SINGLE),
        constrained_layout=True,
    )

    ax.scatter(x, y, c=colors, s=POINT_SIZE, alpha=POINT_ALPHA,
               edgecolors='none', linewidths=POINT_EDGE_WIDTH, zorder=4)

    for pid in patient_ids:
        row = plot_data[plot_data["patient_id"] == pid]
        if row.empty:
            continue
        ax.scatter(row["n_SLH"], row[delta_col],
                   s=POINT_SIZE * 2.2,
                   c=PATIENT_COLORS.get(pid, "#999999"),
                   alpha=0.95, edgecolors="black", linewidths=1.0,
                   marker=PATIENT_MARKERS.get(pid, "o"),
                   zorder=10, label=pid)

    if stats_dict and len(x) > 2:
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = stats_dict["ols_slope"] * x_line + stats_dict["ols_intercept"]
        ax.plot(x_line, y_line, color=COLOR_REGRESSION, lw=1.5,
                ls="--", alpha=0.8, zorder=3, label="OLS regression")

    ax.axhline(0, color=COLOR_ZERO_LINE, linestyle=ZERO_LINE_STYLE,
               linewidth=ZERO_LINE_WIDTH, alpha=0.7, zorder=1)

    ax.set_xlabel(N_SLH_LABEL, fontsize=LANCET_MIN_FONT_PT, labelpad=4)
    ax.set_ylabel(metric_label(metric), fontsize=LANCET_MIN_FONT_PT, labelpad=4)

    # Glucose-valued deltas stay in mg/dL; only the tick labels follow the
    # display unit.  Called once the limits are final, as the helper requires.
    if is_glucose_valued_column(metric):
        apply_glucose_axis(ax, set_label=False)

    # The statistic goes on the panel, not into a title: there is no title.
    if stats_dict:
        ci = ""
        if np.isfinite(stats_dict["ci_low"]):
            ci = (f"\n95% CI {stats_dict['ci_low']:.2f} to "
                  f"{stats_dict['ci_high']:.2f}")
        ax.set_title(
            f"Spearman ρ={stats_dict['spearman_r']:.2f}, "
            f"{format_p_value(stats_dict['spearman_p'])}, "
            f"n={stats_dict['n_points']}{ci}",
            fontsize=LANCET_MIN_FONT_PT, loc="left", pad=4,
        )

    ax.grid(True, alpha=0.3, linewidth=0.5, zorder=0)
    style_axes(ax)
    ax.tick_params(length=3, width=0.8, labelsize=LANCET_MIN_FONT_PT)

    handles = direction_legend_handles(colors) + [
        Line2D([0], [0], color=COLOR_REGRESSION, ls="--", lw=1.5,
               label="OLS fit"),
    ]
    for pid in patient_ids:
        handles.append(
            Line2D([0], [0], marker=PATIENT_MARKERS.get(pid, "o"), color="w",
                   markerfacecolor=PATIENT_COLORS.get(pid, "#999999"),
                   markeredgecolor="black", markersize=5, label=pid)
        )
    ax.legend(handles=handles, fontsize=LANCET_MIN_FONT_PT, loc="best",
             frameon=True, framealpha=1.0, edgecolor="#cccccc",
             borderpad=0.35, handletextpad=0.5)

    return fig


def create_grid_figure(data: pd.DataFrame, metrics: list,
                       stats_list: list, offset: int,
                       delta_prefix: str = DELTA_PREFIX_ALL_DAY) -> plt.Figure:
    """
    Create grid layout with all metrics in one figure.

    Parameters
    ----------
    data : pd.DataFrame
        Analysis data
    metrics : list
        List of metrics to plot
    stats_list : list
        List of statistics dictionaries
    offset : int
        Zone-expansion offset (minutes)
    delta_prefix : str
        Column prefix selecting all-day or daytime-only deltas.

    Returns
    -------
    plt.Figure
    """
    apply_lancet_style()

    n_metrics = len(metrics)
    ncols = GRID_NCOLS
    nrows = int(np.ceil(n_metrics / ncols))

    # No title is drawn inside the artwork.  Height is set
    # so the panels plus the legend stay inside one page; lancet_figsize
    # clamps it if a longer metric list is passed.
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=lancet_figsize(
            GRID_PANEL_HEIGHT_INCHES * nrows + GRID_LEGEND_HEIGHT_INCHES,
            columns=FIGURE_COLUMNS_GRID,
        ),
        constrained_layout=True,
    )

    axes_flat = np.array(axes).reshape(-1) if n_metrics > 1 else [axes]

    stats_dict_map = {s["metric"]: s for s in stats_list if s is not None}
    colors_used = set()

    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]

        delta_col = f"{delta_prefix}{metric}"
        plot_data = data[["n_SLH", delta_col]].dropna()

        if len(plot_data) < MIN_POINTS_SPEARMAN:
            ax.set_visible(False)
            continue

        x = plot_data["n_SLH"].values
        y = plot_data[delta_col].values

        colors = assign_point_colors(y, metric)
        colors_used.update(colors)

        ax.scatter(x, y, c=colors, s=POINT_SIZE * 0.8, alpha=POINT_ALPHA,
                   edgecolors='none', linewidths=0, zorder=4)

        stats_dict = stats_dict_map.get(metric)
        if stats_dict and len(x) > 2:
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = stats_dict["ols_slope"] * x_line + stats_dict["ols_intercept"]
            ax.plot(x_line, y_line, color=COLOR_REGRESSION, lw=1.2,
                    ls="--", alpha=0.7, zorder=3)

        ax.axhline(0, color=COLOR_ZERO_LINE, linestyle=ZERO_LINE_STYLE,
                   linewidth=ZERO_LINE_WIDTH, alpha=0.7, zorder=1)

        ax.set_xlabel(N_SLH_LABEL, fontsize=LANCET_MIN_FONT_PT, labelpad=2)
        ax.set_ylabel(metric_label(metric), fontsize=LANCET_MIN_FONT_PT,
                      labelpad=2)

        # Glucose-valued deltas stay in mg/dL; only the ticks are relabelled.
        if is_glucose_valued_column(metric):
            apply_glucose_axis(ax, set_label=False)

        # Panel letter, then the statistic the panel is there to show.
        heading = PANEL_LETTERS[idx]
        if stats_dict:
            heading += (f"   Spearman ρ={stats_dict['spearman_r']:.2f}, "
                       f"{format_p_value(stats_dict['spearman_p'])}")
        ax.set_title(heading, fontsize=LANCET_MIN_FONT_PT, fontweight="bold",
                     loc="left", pad=3)

        ax.grid(True, alpha=0.3, linewidth=0.5, zorder=0)
        style_axes(ax)
        ax.tick_params(length=3, width=0.8, labelsize=LANCET_MIN_FONT_PT)

    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    handles = direction_legend_handles(sorted(colors_used)) + [
        Line2D([0], [0], color=COLOR_REGRESSION, ls="--", lw=1.2,
               label="OLS fit"),
    ]

    # "outside lower center" makes constrained_layout reserve room for the
    # legend inside the canvas, so the figure can be saved uncropped at
    # exactly the page width.
    fig.legend(handles=handles, loc="outside lower center", ncol=3,
               fontsize=LANCET_MIN_FONT_PT, frameon=False)

    return fig


# =============================================================================
# SAVING
# =============================================================================

def save_lancet_figure(fig: plt.Figure, path: str) -> None:
    """
    Save a figure at exactly its column width.

    ``bbox_inches=None`` keeps the page width the figure was drawn at — the
    default "tight" crop would change it — so the layout must already leave
    room for everything it draws.
    """
    fixed = enforce_lancet_minimums(fig)
    if any(fixed.values()):
        print(f"      Minimums enforced on {fixed['text']} text "
              f"object(s) and {fixed['lines']} line(s)")

    save_figure(fig, path, bbox_inches=None)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="n_SLH vs Delta Metric Scatter Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--offset", type=int, default=SELECTED_OFFSET,
                        help="Zone-expansion offset (minutes)")
    parser.add_argument("--daytime", action="store_true",
                        help="Use daytime-only (06:00-22:00) deltas instead of all-day deltas")
    parser.add_argument("--highlight_patients", nargs="*", default=PATIENT_IDS,
                        help="Patient IDs to highlight on scatter plots (e.g. A174 A224)")
    return parser.parse_args()


def main():
    """Main execution pipeline."""
    args = parse_arguments()
    offset = args.offset
    delta_prefix = DELTA_PREFIX_DAYTIME if args.daytime else DELTA_PREFIX_ALL_DAY
    suffix = "_daytime" if args.daytime else ""

    print("=" * 70)
    print("  n_SLH vs Delta Metric Scatter Analysis")
    print("=" * 70)
    print(f"  Selected offset:  {offset} min")
    print(f"  Delta window:     {'daytime (06:00-22:00)' if args.daytime else 'all-day'}")
    print(f"  Display unit:     {glucose_unit_label()}")
    print(f"  Episodes:    {EPISODES_SUMMARY_CSV}")
    print(f"  Sensitivity: {COHORT_SENSITIVITY_CSV}")
    print(f"  Output:      {OUTPUT_DIR}")
    print("=" * 70)

    episodes_df    = load_episode_summary()
    sensitivity_df = load_sensitivity_results(offset)

    merged_df = merge_data(episodes_df, sensitivity_df)

    print(f"\n  n_SLH range: {merged_df['n_SLH'].min():.0f} - {merged_df['n_SLH'].max():.0f}")
    print(f"  n_SLH median: {merged_df['n_SLH'].median():.0f}")

    stats_list = []

    for metric in METRICS:
        stats_dict = compute_statistics(merged_df, metric, delta_prefix=delta_prefix)
        if stats_dict:
            stats_list.append(stats_dict)
            print(f"    {metric:12s}: rho = {stats_dict['spearman_r']:+.3f}  "
                  f"{format_p_value(stats_dict['spearman_p'])}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if stats_list:
        stats_df = pd.DataFrame(stats_list)
        stats_path = STATS_CSV.replace(".csv", f"{suffix}.csv") if suffix else STATS_CSV
        stats_df.to_csv(stats_path, index=False)
        print(f"\n  Statistics saved: {stats_path}")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    for metric in METRICS:
        stats_dict = next((s for s in stats_list if s["metric"] == metric), None)

        fig = create_scatter_plot(merged_df, metric, stats_dict, offset,
                                  delta_prefix=delta_prefix,
                                  patient_ids=args.highlight_patients)

        if fig:
            path = os.path.join(FIGURES_DIR, f"{metric}_scatter_offset{offset}{suffix}")
            save_lancet_figure(fig, path)
            print(f"      {metric}")

    fig_grid  = create_grid_figure(merged_df, METRICS, stats_list, offset,
                                   delta_prefix=delta_prefix)
    path_grid = os.path.join(FIGURES_DIR, f"all_metrics_grid_offset{offset}{suffix}")
    save_lancet_figure(fig_grid, path_grid)
    print("      Grid layout - done")

    print("\n" + "=" * 70)
    print(f"  Figures:    {FIGURES_DIR}/")
    print(f"  Statistics: {STATS_CSV}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
