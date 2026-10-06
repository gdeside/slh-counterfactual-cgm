"""
scripts/counterfactual/visualize_boxplots_sensitivity.py
========================================================
Cohort Sensitivity Boxplot Visualization
=========================================

Generates boxplots showing distribution of metric changes (Δ) across the cohort
for different zone-expansion offset values (δ).

Three types of figures per metric:

1. Full-time metrics only
   - Single panel showing delta distribution across offsets
   - File: {metric}_fulltime_boxplot.pdf/.png

2. Daytime metrics only
   - Single panel showing delta distribution across offsets
   - File: {metric}_daytime_boxplot.pdf/.png

3. Side-by-side comparison
   - Two panels: full-time (left) and daytime (right)
   - File: {metric}_comparison_boxplot.pdf/.png

Plus one combined supplementary figure with every metric as its own subplot
(whole-day and daytime boxplots grouped side by side per offset within each
subplot), labelled A, B, C, ... -- see generate_combined_boxplot_figure().
File: SupplementaryFigure_Sensitivity_Boxplots.pdf/.svg

Whole day vs daytime
--------------------
The two boxes of a pair share the offset colour and are told apart by their
fill: the **whole-day** box is open (white fill, coloured edge) and the
**daytime** box is solid. Hatching was used before and was invisible at
print size, which made the two windows indistinguishable.

Artwork specification
---------------------
No title is drawn inside the artwork.

Input:  Results/Sensitivity_LinInterp_Modular/Data/cohort_sensitivity.csv
Output: Results/Sensitivity_LinInterp_Modular/Figures/Cohort/boxplots/
        Figures/Supplementary/SupplementaryFigure_Sensitivity_Boxplots.{pdf,svg}

Usage:
    python3 -m scripts.counterfactual.visualize_boxplots_sensitivity
    python3 -m scripts.counterfactual.visualize_boxplots_sensitivity --metric TIR70-180

Authors: Guillaume Deside
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import configuration
from src.config import BASE_PATH
from src.config.figure_config import LANCET_MIN_FONT_PT, lancet_figsize
from src.counterfactual.counterfactual_config import OFFSET_VALUES, KERNEL_MIN, APPLY_KERNEL

# Import visualization
from src.visualization.style import (
    apply_lancet_style,
    enforce_lancet_minimums,
    save_figure,
    style_axes,
)

from src.visualization.colors import OFFSET_COLORS
from src.visualization.glucose_axis import apply_glucose_axis
from src.utils.unit_conversion import glucose_unit_label, is_glucose_valued_column

# =============================================================================
# CONFIGURATION
# =============================================================================

# ── Paths ────────────────────────────────────────────────────────────────────
SENSITIVITY_RESULTS_DIR = os.path.join(BASE_PATH, "Results", "Sensitivity_Analysis")
DATA_DIR = os.path.join(SENSITIVITY_RESULTS_DIR, "Data")
FIGURES_OUTPUT_DIR = os.path.join(SENSITIVITY_RESULTS_DIR, "Figures", "Cohort", "boxplots")

COHORT_CSV = os.path.join(DATA_DIR, "cohort_sensitivity.csv")

# ── Metrics to plot ──────────────────────────────────────────────────────────
# List of metrics to generate boxplots for
METRICS = [
    "TIR70-180",
    "TAR>180",
    "TAR>250",
    "mean",
    "median",
    "sd",
    "cv",
    "TBR<70",
]

# Metric display properties
METRIC_LABELS = {
    "TIR70-180": "ΔTIR 70–180",
    "TAR>180": "ΔTAR >180",
    "TAR>250": "ΔTAR >250",
    "mean": "ΔMean glucose",
    "median": "ΔMedian glucose",
    "sd": "ΔSD",
    "cv": "ΔCV",
    "TBR<70": "ΔTBR <70",
}

METRIC_UNITS = {
    "TIR70-180": "(%)",
    "TAR>180": "(%)",
    "TAR>250": "(%)",
    "mean": "(mg/dL)",
    "median": "(mg/dL)",
    "sd": "(mg/dL)",
    "cv": "(%)",
    "TBR<70": "(%)",
}


def metric_unit(metric: str) -> str:
    """METRIC_UNITS entry, with mg/dL replaced by the current display unit."""
    unit = METRIC_UNITS[metric]
    return f"({glucose_unit_label()})" if unit == "(mg/dL)" else unit


# Direction of improvement for each metric
METRIC_DIRECTION = {
    "TIR70-180": +1,   # Higher is better (increase TIR)
    "TAR>180": -1,     # Lower is better (decrease TAR)
    "TAR>250": -1,     # Lower is better (decrease TAR)
    "mean": -1,        # Lower is better (decrease mean)
    "median": -1,      # Lower is better (decrease median)
    "sd": -1,          # Lower is better (decrease variability)
    "cv": -1,          # Lower is better (decrease variability)
    "TBR<70": -1,      # Lower is better (decrease hypoglycemia)
}

# ── Figure parameters (column widths) ─────────────────────────────────
# A single panel is drawn two columns wide (115 mm) and a side-by-side pair at
# full page width (175 mm); see src.config.figure_config.
FIGURE_COLUMNS_SINGLE = "double"   # 115 mm — one panel
FIGURE_COLUMNS_DOUBLE = "full"     # 175 mm — two panels side by side
FIGURE_HEIGHT = 3.5                # Height for boxplot figures

# ── Boxplot styling ──────────────────────────────────────────────────────────
BOX_WIDTH = 0.6
FLIER_SIZE = 3              # Outlier marker size
FLIER_ALPHA = 0.5           # Outlier transparency
MEDIAN_LW = 1.5             # Median line width
BOX_LW = 1.0                # Box edge line width

# ── Colors ───────────────────────────────────────────────────────────────────
OFFSET_FACE_ALPHA = 0.6     # Box fill transparency
OFFSET_EDGE_ALPHA = 0.9     # Box edge transparency

# Reference line for zero change
ZERO_LINE_COLOR = "#555555"
ZERO_LINE_STYLE = "--"
ZERO_LINE_WIDTH = 1.0
ZERO_LINE_ALPHA = 0.6

# ── Combined supplementary figure (all metrics, one subplot each) ────────────
SUPPLEMENTARY_OUTPUT_DIR = os.path.join(BASE_PATH, "Figures", "Supplementary")
COMBINED_FIGURE_NAME = "SupplementaryFigure_Sensitivity_Boxplots"

PANEL_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ── Analysis window naming and encoding ──────────────────────────────────────
# One name per window, used identically in axis labels, panel labels, legends
# and labels, so the two are never referred to in two different ways.
LABEL_WHOLE_DAY = "Whole day (00:00–24:00)"
LABEL_DAYTIME   = "Daytime (06:00–22:00)"

# Within a pair the two windows share the offset colour and differ by fill:
# the whole-day box is open, the daytime box solid.  Hatching was tried first
# and is illegible at print size, which is what made the two indistinguishable.
WHOLE_DAY_FACECOLOR = "#FFFFFF"   # Open box — whole day
WHOLE_DAY_EDGE_LW = 1.1           # Slightly heavier edge, since there is no fill

PAIR_GAP = 1.0               # Spacing between consecutive offset pairs
GROUP_GAP = 0.42             # Offset between the whole-day and daytime box in a pair
PAIR_BOX_WIDTH = 0.34

# ── Combined figure layout ───────────────────────────────────────────────────
COMBINED_PANEL_HEIGHT_INCHES = 2.05   # Per-row height; 4 rows + legend fit one page
COMBINED_LEGEND_HEIGHT_INCHES = 0.35


# =============================================================================
# DATA LOADING
# =============================================================================

def load_cohort_data() -> pd.DataFrame:
    """Load cohort sensitivity results."""
    if not os.path.exists(COHORT_CSV):
        raise FileNotFoundError(f"Cohort CSV not found: {COHORT_CSV}")
    
    df = pd.read_csv(COHORT_CSV)
    return df


def extract_metric_data(df: pd.DataFrame, metric: str, 
                       daytime_only: bool = False) -> dict:
    """
    Extract delta values for a metric across all offsets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cohort data
    metric : str
        Metric name (e.g., "TIR70-180")
    daytime_only : bool
        If True, extract daytime metrics instead of full-time
        
    Returns
    -------
    dict
        {offset: array of delta values}
    """
    prefix = "daytime_delta_" if daytime_only else "delta_"
    col_name = f"{prefix}{metric}"
    
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in cohort data")
    
    # Extract data for each offset
    data_by_offset = {}
    
    for offset in sorted(df["offset"].unique()):
        mask = df["offset"] == offset
        values = df.loc[mask, col_name].dropna().values
        data_by_offset[offset] = values
    
    return data_by_offset


# =============================================================================
# BOXPLOT GENERATION
# =============================================================================

def create_boxplot_data(data_by_offset: dict) -> tuple:
    """
    Prepare data for boxplot visualization.
    
    Returns
    -------
    (positions, data_list, labels, colors)
    """
    offsets = sorted(data_by_offset.keys())
    
    positions = list(range(len(offsets)))
    data_list = [data_by_offset[offset] for offset in offsets]
    # Bare values: the x-axis label already says "δ (min)", and repeating it on
    # every tick makes the labels collide at column width.
    labels = [f"{offset:g}" for offset in offsets]
    colors = [OFFSET_COLORS.get(offset, "#999999") for offset in offsets]
    
    return positions, data_list, labels, colors, offsets


def style_boxplot(bp: dict, colors: list, filled: bool = True):
    """
    Apply consistent styling to boxplot elements.

    Parameters
    ----------
    bp : dict
        Boxplot dictionary returned by plt.boxplot()
    colors : list
        List of colors for each box
    filled : bool, default True
        Fill each box with its colour.  Pass False for an *open* box — white
        fill, coloured edge — which is how the whole-day boxes are told apart
        from the daytime ones in the paired panels.  The distinction survives
        greyscale reproduction, which a colour or hatch difference does not.
    """
    for patch, color in zip(bp['boxes'], colors):
        patch.set_edgecolor(color)

        if filled:
            patch.set_facecolor(color)
            patch.set_linewidth(BOX_LW)
            patch.set_alpha(OFFSET_FACE_ALPHA)
        else:
            patch.set_facecolor(WHOLE_DAY_FACECOLOR)
            patch.set_linewidth(WHOLE_DAY_EDGE_LW)
            patch.set_alpha(1.0)
    
    # Median lines
    for median in bp['medians']:
        median.set_color('#000000')
        median.set_linewidth(MEDIAN_LW)
    
    # Whiskers and caps
    for whisker in bp['whiskers']:
        whisker.set_linewidth(BOX_LW)
        whisker.set_linestyle('-')
        whisker.set_alpha(0.7)
    
    for cap in bp['caps']:
        cap.set_linewidth(BOX_LW)
        cap.set_alpha(0.7)
    
    # Fliers (outliers)
    for flier in bp['fliers']:
        flier.set_marker('o')
        flier.set_markersize(FLIER_SIZE)
        flier.set_alpha(FLIER_ALPHA)
        flier.set_markerfacecolor('#555555')
        flier.set_markeredgecolor('none')


# def add_beneficial_zone(ax: plt.Axes, metric: str):
#     """Add shading for beneficial direction of change."""
#     direction = METRIC_DIRECTION.get(metric, 0)
    
#     if direction == 0:
#         return
    
#     ylim = ax.get_ylim()
    
#     if direction > 0:  # Positive is good (e.g., TIR increase)
#         ax.axhspan(0, ylim[1], alpha=BENEFICIAL_ALPHA, 
#                   color=BENEFICIAL_COLOR, zorder=0)
#     else:  # Negative is good (e.g., TAR decrease)
#         ax.axhspan(ylim[0], 0, alpha=BENEFICIAL_ALPHA, 
#                   color=BENEFICIAL_COLOR, zorder=0)


def format_boxplot_axis(ax: plt.Axes, metric: str, labels: list,
                       n_patients: int, daytime_only: bool = False,
                       show_window: bool = True):
    """
    Apply standard formatting to a boxplot axis.

    Parameters
    ----------
    show_window : bool, default True
        Name the analysis window in the y-axis label.  Pass False where the
        panel title already names it, so it is not stated twice.

    Notes
    -----
    Call this once the y limits are final: for glucose metrics it fixes the
    tick positions, which a later ``set_ylim`` would not refresh.
    """
    # Y-axis label
    window = LABEL_DAYTIME if daytime_only else LABEL_WHOLE_DAY
    ylabel = f"{METRIC_LABELS[metric]} {metric_unit(metric)}"
    if show_window:
        ylabel = f"{ylabel}, {window}"
    ax.set_ylabel(ylabel, fontsize=LANCET_MIN_FONT_PT, labelpad=3)

    # Glucose metrics: Δ values stay in mg/dL, only the ticks are relabelled
    if is_glucose_valued_column(metric):
        apply_glucose_axis(ax, set_label=False)
    
    # X-axis
    ax.set_xlabel("Contextual window width δ (min)",
                  fontsize=LANCET_MIN_FONT_PT, labelpad=3)
    ax.set_xticklabels(labels, fontsize=LANCET_MIN_FONT_PT)
    
    # Zero reference line
    ax.axhline(0, color=ZERO_LINE_COLOR, linestyle=ZERO_LINE_STYLE, 
              linewidth=ZERO_LINE_WIDTH, alpha=ZERO_LINE_ALPHA, zorder=1)
    
    # Add beneficial zone shading
    # add_beneficial_zone(ax, metric)
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5, zorder=0)
    
    style_axes(ax)
    ax.tick_params(length=3, width=0.8, labelsize=LANCET_MIN_FONT_PT)


# =============================================================================
# FIGURE TYPE 1: FULL-TIME ONLY
# =============================================================================

def generate_fulltime_boxplot(df: pd.DataFrame, metric: str) -> plt.Figure:
    """
    Generate boxplot for full-time (24h) metrics only.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cohort data
    metric : str
        Metric name
        
    Returns
    -------
    plt.Figure
    """
    apply_lancet_style()

    # Extract data
    data_by_offset = extract_metric_data(df, metric, daytime_only=False)
    positions, data_list, labels, colors, offsets = create_boxplot_data(data_by_offset)

    # Count patients
    n_patients = df["patient_id"].nunique()

    # Create figure (no title is drawn inside the artwork)
    fig, ax = plt.subplots(
        figsize=lancet_figsize(FIGURE_HEIGHT, columns=FIGURE_COLUMNS_SINGLE),
        constrained_layout=True,
    )

    # Create boxplot
    bp = ax.boxplot(
        data_list,
        positions=positions,
        widths=BOX_WIDTH,
        patch_artist=True,
        showfliers=True,
        notch=False,
    )
    
    # Style boxplot — open boxes, the whole-day encoding used throughout
    style_boxplot(bp, colors, filled=False)

    # Format axis
    format_boxplot_axis(ax, metric, labels, n_patients, daytime_only=False)

    return fig


# =============================================================================
# FIGURE TYPE 2: DAYTIME ONLY
# =============================================================================

def generate_daytime_boxplot(df: pd.DataFrame, metric: str) -> plt.Figure:
    """
    Generate boxplot for daytime (06:00-22:00) metrics only.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cohort data
    metric : str
        Metric name
        
    Returns
    -------
    plt.Figure
    """
    apply_lancet_style()

    # Extract data
    data_by_offset = extract_metric_data(df, metric, daytime_only=True)
    positions, data_list, labels, colors, offsets = create_boxplot_data(data_by_offset)

    # Count patients
    n_patients = df["patient_id"].nunique()

    # Create figure (no title is drawn inside the artwork)
    fig, ax = plt.subplots(
        figsize=lancet_figsize(FIGURE_HEIGHT, columns=FIGURE_COLUMNS_SINGLE),
        constrained_layout=True,
    )

    # Create boxplot
    bp = ax.boxplot(
        data_list,
        positions=positions,
        widths=BOX_WIDTH,
        patch_artist=True,
        showfliers=True,
        notch=False,
    )
    
    # Style boxplot — solid boxes, the daytime encoding used throughout
    style_boxplot(bp, colors, filled=True)

    # Format axis
    format_boxplot_axis(ax, metric, labels, n_patients, daytime_only=True)

    return fig


# =============================================================================
# FIGURE TYPE 3: SIDE-BY-SIDE COMPARISON
# =============================================================================

def _add_panel_label(ax: plt.Axes, letter: str, window: str) -> None:
    """
    Label a panel with its bold letter and the analysis window it covers.

    Placed above the axes rather than inside it, so it never sits on the data
    and needs no opaque background box — panel letters are set as bold
    capitals outside the plotting area.
    """
    ax.set_title(f"({letter}) {window}",
                 fontsize=LANCET_MIN_FONT_PT, fontweight="bold",
                 loc="left", pad=4)


def generate_comparison_boxplot(df: pd.DataFrame, metric: str) -> plt.Figure:
    """
    Generate side-by-side comparison: full-time vs daytime metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cohort data
    metric : str
        Metric name
        
    Returns
    -------
    plt.Figure
    """
    apply_lancet_style()

    # Extract data for both
    data_fulltime = extract_metric_data(df, metric, daytime_only=False)
    data_daytime = extract_metric_data(df, metric, daytime_only=True)
    
    # Prepare data
    pos_ft, data_ft, labels_ft, colors_ft, offsets = create_boxplot_data(data_fulltime)
    pos_dt, data_dt, labels_dt, colors_dt, _ = create_boxplot_data(data_daytime)
    
    # Count patients
    n_patients = df["patient_id"].nunique()
    
    # Create figure with two panels
    # (no title is drawn inside the artwork)
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=lancet_figsize(FIGURE_HEIGHT, columns=FIGURE_COLUMNS_DOUBLE),
        constrained_layout=True,
    )

    # ── Panel A: whole day ───────────────────────────────────────────────────
    bp1 = ax1.boxplot(
        data_ft,
        positions=pos_ft,
        widths=BOX_WIDTH,
        patch_artist=True,
        showfliers=True,
        notch=False,
    )
    style_boxplot(bp1, colors_ft, filled=False)

    # ── Panel B: daytime ─────────────────────────────────────────────────────
    bp2 = ax2.boxplot(
        data_dt,
        positions=pos_dt,
        widths=BOX_WIDTH,
        patch_artist=True,
        showfliers=True,
        notch=False,
    )
    style_boxplot(bp2, colors_dt, filled=True)

    # Synchronise the y-axes *before* formatting: for glucose metrics the
    # formatter fixes tick positions from the current limits, so a later
    # set_ylim would leave one panel's ticks stopping short of its axis.
    y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    # The panel titles name the window, so the y labels do not repeat it.
    format_boxplot_axis(ax1, metric, labels_ft, n_patients,
                        daytime_only=False, show_window=False)
    format_boxplot_axis(ax2, metric, labels_dt, n_patients,
                        daytime_only=True, show_window=False)
    _add_panel_label(ax1, "A", LABEL_WHOLE_DAY)
    _add_panel_label(ax2, "B", LABEL_DAYTIME)

    return fig


# =============================================================================
# COMBINED SUPPLEMENTARY FIGURE: ALL METRICS AS SUBPLOTS (A, B, C, ...)
# =============================================================================

def _grouped_positions(n_offsets: int) -> tuple:
    """x positions for (full-time, daytime) box pairs, one pair per offset."""
    base = np.arange(n_offsets) * PAIR_GAP
    return base, base + GROUP_GAP


def window_legend_handles() -> list:
    """
    Legend keys for the two analysis windows, in a neutral grey.

    The same open-vs-solid distinction the panels use, so the legend explains
    the encoding rather than repeating the offset colours.
    """
    return [
        mpatches.Patch(facecolor=WHOLE_DAY_FACECOLOR, edgecolor="#555555",
                      linewidth=WHOLE_DAY_EDGE_LW, label=LABEL_WHOLE_DAY),
        mpatches.Patch(facecolor="#999999", alpha=OFFSET_FACE_ALPHA,
                      edgecolor="#555555", linewidth=BOX_LW, label=LABEL_DAYTIME),
    ]


def plot_metric_panel_combined(ax: plt.Axes, df: pd.DataFrame, metric: str) -> None:
    """
    Draw one metric's sensitivity-to-δ boxplots on a single Axes.

    For every δ, the whole-day and daytime distributions are drawn as a pair:
    both carry the δ colour (matching the single-metric figures), and the
    whole-day box is left open while the daytime box is solid.  Fill, unlike
    the hatching used previously, stays legible at print size and in
    greyscale — which is what makes the two windows tellable apart at all.
    """
    data_ft = extract_metric_data(df, metric, daytime_only=False)
    data_dt = extract_metric_data(df, metric, daytime_only=True)
    offsets = sorted(data_ft.keys())
    colors = [OFFSET_COLORS.get(o, "#999999") for o in offsets]

    pos_ft, pos_dt = _grouped_positions(len(offsets))

    bp_ft = ax.boxplot([data_ft[o] for o in offsets], positions=pos_ft,
                       widths=PAIR_BOX_WIDTH, patch_artist=True, showfliers=True)
    bp_dt = ax.boxplot([data_dt[o] for o in offsets], positions=pos_dt,
                       widths=PAIR_BOX_WIDTH, patch_artist=True, showfliers=True)

    style_boxplot(bp_ft, colors, filled=False)
    style_boxplot(bp_dt, colors, filled=True)

    tick_pos = (pos_ft + pos_dt) / 2
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"{o:g}" for o in offsets], fontsize=LANCET_MIN_FONT_PT)
    ax.set_xlim(pos_ft[0] - 0.45, pos_dt[-1] + 0.45)
    ax.set_xlabel("Contextual window width δ (min)",
                  fontsize=LANCET_MIN_FONT_PT, labelpad=2)

    ax.axhline(0, color=ZERO_LINE_COLOR, linestyle=ZERO_LINE_STYLE,
              linewidth=ZERO_LINE_WIDTH, alpha=ZERO_LINE_ALPHA, zorder=1)
    ax.set_ylabel(f"{METRIC_LABELS[metric]} {metric_unit(metric)}",
                  fontsize=LANCET_MIN_FONT_PT, labelpad=2)
    if is_glucose_valued_column(metric):
        apply_glucose_axis(ax, set_label=False)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5, zorder=0)
    style_axes(ax)
    ax.tick_params(length=3, width=0.8, labelsize=LANCET_MIN_FONT_PT)


def generate_combined_boxplot_figure(df: pd.DataFrame, metrics: list = METRICS) -> plt.Figure:
    """
    Supplementary figure: every metric's sensitivity-to-offset boxplot as its
    own subplot (full 24h + daytime grouped per offset within the subplot),
    labelled A, B, C, ... in a grid.
    """
    apply_lancet_style()

    n_metrics = len(metrics)
    n_cols = 2
    n_rows = int(np.ceil(n_metrics / n_cols))

    # Height is set so that the panels plus the legend stay inside one
    # page; lancet_figsize clamps it if a longer metric list is passed.
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=lancet_figsize(
            COMBINED_PANEL_HEIGHT_INCHES * n_rows + COMBINED_LEGEND_HEIGHT_INCHES,
            columns="full",
        ),
        constrained_layout=True,
    )
    axes_flat = np.array(axes).reshape(-1)

    for i, metric in enumerate(metrics):
        ax = axes_flat[i]
        plot_metric_panel_combined(ax, df, metric)
        ax.set_title(PANEL_LETTERS[i], fontsize=LANCET_MIN_FONT_PT + 1,
                    fontweight="bold", loc="left", pad=3)

    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    # One figure-level legend for the open/solid encoding, below the panels:
    # inside panel A it sat on the data. "outside lower center" makes
    # constrained_layout reserve the space, so nothing needs cropping and the
    # figure saves at exactly the page width.
    fig.legend(handles=window_legend_handles(), loc="outside lower center",
              ncol=2, fontsize=LANCET_MIN_FONT_PT, frameon=False)

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
        print(f"    Minimums enforced on {fixed['text']} text object(s) "
              f"and {fixed['lines']} line(s)")

    save_figure(fig, path, bbox_inches=None)


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_all_metrics(df: pd.DataFrame, metric_filter: str = None):
    """
    Generate all boxplot figures for all metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cohort data
    metric_filter : str, optional
        If provided, only process this metric
    """
    # Determine which metrics to process
    if metric_filter:
        if metric_filter not in METRICS:
            print(f"  [!] Unknown metric: {metric_filter}")
            print(f"      Available: {METRICS}")
            return
        metrics_to_process = [metric_filter]
    else:
        metrics_to_process = METRICS
    
    # Create output directory
    os.makedirs(FIGURES_OUTPUT_DIR, exist_ok=True)
    
    print(f"\n  Processing {len(metrics_to_process)} metric(s)...")

    offsets = sorted(df["offset"].unique())
    n_patients = df["patient_id"].nunique()

    for metric in metrics_to_process:
        print(f"\n  {metric}:")

        try:
            # Generate whole-day boxplot
            fig_fulltime = generate_fulltime_boxplot(df, metric)
            path_fulltime = os.path.join(FIGURES_OUTPUT_DIR, f"{metric}_fulltime_boxplot")
            save_lancet_figure(fig_fulltime, path_fulltime)
            print(f"    ✓ Whole-day boxplot")

            # Generate daytime boxplot
            fig_daytime = generate_daytime_boxplot(df, metric)
            path_daytime = os.path.join(FIGURES_OUTPUT_DIR, f"{metric}_daytime_boxplot")
            save_lancet_figure(fig_daytime, path_daytime)
            print(f"    ✓ Daytime boxplot")

            # Generate comparison boxplot
            fig_comparison = generate_comparison_boxplot(df, metric)
            path_comparison = os.path.join(FIGURES_OUTPUT_DIR, f"{metric}_comparison_boxplot")
            save_lancet_figure(fig_comparison, path_comparison)
            print(f"    ✓ Comparison boxplot")

        except Exception as e:
            print(f"    [!] Error: {e}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate cohort sensitivity boxplots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Process specific metric only (default: all metrics)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  Cohort Sensitivity Boxplot Visualization")
    print("=" * 70)
    print(f"  Input:  {COHORT_CSV}")
    print(f"  Output: {FIGURES_OUTPUT_DIR}")
    if args.metric:
        print(f"  Metric: {args.metric} (single)")
    else:
        print(f"  Metrics: All ({len(METRICS)} total)")
    print("=" * 70)
    
    # Load data
    try:
        df = load_cohort_data()
        n_patients = df["patient_id"].nunique()
        n_offsets = df["offset"].nunique()
        offsets = sorted(int(o) for o in df["offset"].unique())
        print(f"\n  Loaded cohort data:")
        print(f"    Patients: {n_patients}")
        print(f"    Offsets:  {offsets} min")
        print(f"    Rows:     {len(df)}")
        print(f"    Display unit: {glucose_unit_label()}")

        # Every δ in the analysis config must be present in the cohort CSV,
        # or the figures would silently drop one (δ=75 and δ=90 in particular).
        missing = [o for o in OFFSET_VALUES if o not in offsets]
        if missing:
            print(f"\n  [WARNING] δ values configured but absent from the cohort "
                  f"CSV: {missing}")
            print(f"            Re-run the sensitivity analysis to include them:")
            print(f"              python3 -m scripts.counterfactual.sensitivity_analysis "
                  f"--patient_id ALL")
    except FileNotFoundError as e:
        print(f"\n  [ERROR] {e}")
        print(f"\n  Run sensitivity analysis first:")
        print(f"    python3 sensitivity_lininterp_modular.py --patient_id ALL")
        return
    
    # Process metrics
    process_all_metrics(df, args.metric)

    # Combined supplementary figure (all metrics, one subplot each) -- only
    # meaningful in all-metrics mode.
    if not args.metric:
        print("\n  Generating combined supplementary figure (all metrics)...")
        fig_combined = generate_combined_boxplot_figure(df)
        os.makedirs(SUPPLEMENTARY_OUTPUT_DIR, exist_ok=True)
        combined_path = os.path.join(SUPPLEMENTARY_OUTPUT_DIR, COMBINED_FIGURE_NAME)
        save_lancet_figure(fig_combined, combined_path)

    print("\n" + "=" * 70)
    print(f"  Output: {FIGURES_OUTPUT_DIR}")
    if not args.metric:
        print(f"  Supplementary figure: {SUPPLEMENTARY_OUTPUT_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
