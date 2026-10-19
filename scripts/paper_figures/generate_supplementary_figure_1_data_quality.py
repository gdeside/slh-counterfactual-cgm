"""
scripts/paper_figures/generate_supplementary_figure_1_data_quality.py
=====================================================================

Supplementary Figure 1: Data Quality and Cohort Selection
---------------------------------------------------------
Five-panel figure showing preprocessing and filtering steps:

A) Timestep distribution (raw data)
B) Coverage distribution histogram
C) Individual coverage strip plot
D) Median glucose histogram with selection
E) Median glucose vs TIR scatter plot

Author: Guillaume Deside
Date: 2026-04-21

Usage
-----
    python3 -m scripts.paper_figures.generate_supplementary_figure_1_data_quality [--help for options]
    (or: make suppfig1)

Output
------
    <CGM_DATA_ROOT>/Figures/Supplementary/supplementary_figure_1.{pdf,svg}
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator, MaxNLocator
from tqdm import tqdm

# =============================================================================
# PROJECT IMPORTS
# =============================================================================
from src.config import (
    FIG_DIR,
    WINDOW_SUMMARY_CSV,
    MIN_COVERAGE_PCT,
    REQUIRED_SAMPLING_MIN,
    RANDOM_SEED,
    COL_COVERAGE,
    COL_SAMPLING,
    PATIENT_ID_COL,
)

from src.preprocessing.load_data import resolve_patients
from src.preprocessing.timestep import (
    snap_to_canonical,
    dominant_step_of_series,
)
from src.preprocessing.load_data import load_raw_time_column

from src.metrics.load_metrics import load_patient_metrics
from src.filtering.filtering_config import MEDIAN_GLUCOSE_MIN, MEDIAN_GLUCOSE_MAX
from src.filtering.filtering import apply_median_glucose_filter

from src.visualization.colors import (
    C_5MIN, C_15MIN,
    COLOR_PRIMARY, COLOR_THRESHOLD, COLOR_ANNOTATION, COLOR_BOX_BG,
    COLOR_INCLUDED, COLOR_EXCLUDED, COLOR_ZONE, COLORMAP_CONTINUOUS,
)
from src.visualization.style import apply_publication_style, style_axes, save_figure_with_panels
from src.visualization.glucose_axis import apply_glucose_axis
from src.utils.unit_conversion import format_glucose, glucose_label

# =============================================================================
# CONSTANTS
# =============================================================================
COL_MEDIAN_GLUCOSE = "Median_glycemia"
COL_TIR = "TIR70180"
COL_CV = "CV"

# The median-glucose label is built at draw time by glucose_label() so it
# follows the display unit (CGM_GLUCOSE_UNIT / set_display_unit).
LABEL_TIR = "TIR 70–180 (%)"

# Visual parameters
HISTOGRAM_BIN_WIDTH = 2.0
HISTOGRAM_START = 70.0
HISTOGRAM_END = 101.5
STRIP_PLOT_JITTER = 0.14
STRIP_PLOT_POINT_SIZE = 8
STRIP_PLOT_BOX_WIDTH = 0.22

ALPHA_HISTOGRAM = 0.65
ALPHA_IQR_BAND = 0.10
ALPHA_STRIP_POINT = 0.45
ALPHA_IQR_BOX = 0.25
ALPHA_HISTOGRAM_BAR = 0.90
ALPHA_HISTOGRAM_SELECTED = 0.85
ALPHA_ZONE_SHADING = 0.06

SCATTER_SIZE_EXCLUDED = 18
SCATTER_SIZE_SELECTED = 32
SCATTER_ALPHA_EXCLUDED = 0.60
SCATTER_ALPHA_SELECTED = 0.85
SCATTER_EDGE_WIDTH = 0.35

HISTOGRAM_N_BINS = 30

# =============================================================================
# PANEL A: TIMESTEP DISTRIBUTION
# =============================================================================
def draw_timestep_panel(ax, raw_n):
    """Draw timestep distribution bar chart."""
    step_labels = {5: "5 min", 15: "15 min"}
    step_colors = {5: C_5MIN, 15: C_15MIN}
    
    active_steps = [s for s in (5, 15) if raw_n.get(s, 0) > 0]
    if raw_n.get("other", 0) > 0:
        active_steps.append("other")
    
    labels = [step_labels.get(s, "Other") for s in active_steps]
    colors = [step_colors.get(s, "#999999") for s in active_steps]
    values = [raw_n.get(s, 0) for s in active_steps]
    
    x = np.arange(len(active_steps))
    
    bars = ax.bar(
        x, values,
        width=0.55,
        color=colors,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    
    # Add count labels
    y_top = max(values) * 1.15 if values else 1
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_top * 0.02,
                f"{val}",
                ha="center", va="bottom",
                fontsize=7, fontweight="bold",
                color="#333333",
            )
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.set_ylim(0, y_top)
    
    style_axes(ax, xlabel="Timestep", ylabel="Number of participants")

# =============================================================================
# PANEL B: COVERAGE HISTOGRAM
# =============================================================================
def draw_coverage_histogram(ax, coverage_data, stats_dict):
    """Draw coverage distribution histogram."""
    bins = np.arange(HISTOGRAM_START, HISTOGRAM_END, HISTOGRAM_BIN_WIDTH)
    
    ax.hist(
        coverage_data,
        bins=bins,
        color=COLOR_PRIMARY,
        alpha=ALPHA_HISTOGRAM,
        label=f"n = {stats_dict['n']}",
        zorder=3
    )
    
    ax.axvline(
        stats_dict["median"],
        color=COLOR_PRIMARY,
        lw=1.2, ls="--",
        zorder=4
    )
    
    # ax.axvspan(
    #     stats_dict["q25"],
    #     stats_dict["q75"],
    #     color=COLOR_PRIMARY,
    #     alpha=ALPHA_IQR_BAND,
    #     zorder=2
    # )
    
    ax.axvline(
        MIN_COVERAGE_PCT,
        color=COLOR_THRESHOLD,
        lw=0.9, ls=":",
        zorder=5,
        label=f"{MIN_COVERAGE_PCT:.0f} % threshold"
    )
    
    ax.set_xlim(MIN_COVERAGE_PCT - 1, 101)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    
    style_axes(ax, xlabel="Coverage (%)", ylabel="Number of participants")
    
    ax.legend(fontsize=6.5, frameon=False, loc="upper center",
              handlelength=1.2, handleheight=0.85)
    
    y_max = ax.get_ylim()[1]
    ax.text(
        stats_dict["median"] - 5,
        y_max * 1.02,
        f"median {stats_dict['median']:.1f} %",
        color=COLOR_PRIMARY,
        fontsize=6,
        va="top",
        style="italic"
    )

# =============================================================================
# PANEL C: COVERAGE STRIP PLOT
# =============================================================================
def draw_coverage_strip(ax, coverage_data, stats_dict):
    """Draw individual coverage strip plot."""
    np.random.seed(RANDOM_SEED)
    
    x_position = 0
    jitter = np.random.uniform(
        -STRIP_PLOT_JITTER,
        STRIP_PLOT_JITTER,
        size=len(coverage_data)
    )
    
    ax.scatter(
        np.full(len(coverage_data), x_position) + jitter,
        coverage_data,
        color=COLOR_PRIMARY,
        alpha=ALPHA_STRIP_POINT,
        s=STRIP_PLOT_POINT_SIZE,
        lw=0,
        zorder=3,
    )
    
    # IQR box
    box_rect = plt.Rectangle(
        (x_position - STRIP_PLOT_BOX_WIDTH / 2, stats_dict["q25"]),
        STRIP_PLOT_BOX_WIDTH,
        stats_dict["q75"] - stats_dict["q25"],
        facecolor=COLOR_PRIMARY,
        alpha=ALPHA_IQR_BOX,
        edgecolor=COLOR_PRIMARY,
        linewidth=0.8,
        zorder=4,
    )
    ax.add_patch(box_rect)
    
    # Median line
    ax.plot(
        [x_position - STRIP_PLOT_BOX_WIDTH / 2,
         x_position + STRIP_PLOT_BOX_WIDTH / 2],
        [stats_dict["median"], stats_dict["median"]],
        color=COLOR_PRIMARY,
        lw=1.4,
        zorder=5
    )
    
    # Whiskers
    ax.plot(
        [x_position, x_position],
        [stats_dict["min"], stats_dict["q25"]],
        color=COLOR_PRIMARY,
        lw=0.7, ls="--",
        zorder=4
    )
    ax.plot(
        [x_position, x_position],
        [stats_dict["q75"], stats_dict["max"]],
        color=COLOR_PRIMARY,
        lw=0.7, ls="--",
        zorder=4
    )
    
    # Mean marker
    ax.scatter(
        [x_position],
        [stats_dict["mean"]],
        marker="D",
        s=18,
        color=COLOR_BOX_BG,
        edgecolors=COLOR_PRIMARY,
        lw=0.9,
        zorder=6
    )
    
    ax.axhline(
        MIN_COVERAGE_PCT,
        color=COLOR_THRESHOLD,
        lw=0.9, ls=":",
        zorder=2
    )
    
    ax.set_xticks([x_position])
    ax.set_xticklabels([f"{REQUIRED_SAMPLING_MIN:.0f} min"], fontsize=7.5)
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(MIN_COVERAGE_PCT - 2, 102)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    
    style_axes(ax, ylabel="Coverage (%)")
    
    median_line = mlines.Line2D(
        [], [],
        color=COLOR_ANNOTATION,
        lw=1.2,
        label="Median"
    )
    mean_marker = mlines.Line2D(
        [], [],
        marker="D",
        color=COLOR_BOX_BG,
        markeredgecolor=COLOR_ANNOTATION,
        markersize=5,
        lw=0,
        label="Mean"
    )
    ax.legend(
        handles=[median_line, mean_marker],
        fontsize=6,
        frameon=False,
        loc="center right",
        handlelength=1.0
    )

# =============================================================================
# PANEL D: MEDIAN GLUCOSE HISTOGRAM
# =============================================================================
def compute_adaptive_bins(data_values, lower_bound, upper_bound, n_bins):
    """Compute histogram bins aligned with threshold boundaries."""
    data_min = float(np.nanmin(data_values))
    data_max = float(np.nanmax(data_values))
    
    threshold_range = upper_bound - lower_bound
    bin_width = threshold_range / (n_bins * 0.6)
    
    left_edges = np.arange(lower_bound, data_min - bin_width, -bin_width)[::-1]
    center_edges = np.arange(lower_bound, upper_bound + bin_width/2, bin_width)
    right_edges = np.arange(upper_bound + bin_width, data_max + bin_width, bin_width)
    
    all_edges = np.unique(np.concatenate([left_edges, center_edges, right_edges]))
    return all_edges

def draw_median_glucose_histogram(ax, df_all, df_selected):
    """Draw median glucose distribution histogram."""
    n_selected = len(df_selected)
    n_total = len(df_all)
    n_excluded = n_total - n_selected
    
    values_all = df_all[COL_MEDIAN_GLUCOSE].dropna().values
    values_selected = df_selected[COL_MEDIAN_GLUCOSE].dropna().values
    
    excluded_ids = set(df_all[PATIENT_ID_COL]) - set(df_selected[PATIENT_ID_COL])
    df_excluded = df_all[df_all[PATIENT_ID_COL].isin(excluded_ids)]
    values_excluded = df_excluded[COL_MEDIAN_GLUCOSE].dropna().values
    
    bins = compute_adaptive_bins(
        values_all,
        MEDIAN_GLUCOSE_MIN,
        MEDIAN_GLUCOSE_MAX,
        HISTOGRAM_N_BINS
    )
    
    ax.hist(
        values_excluded,
        bins=bins,
        color=COLOR_EXCLUDED,
        alpha=ALPHA_HISTOGRAM_BAR,
        edgecolor="white",
        linewidth=0.4,
        label=f"Excluded (n = {n_excluded})",
        zorder=2
    )
    ax.hist(
        values_selected,
        bins=bins,
        color=COLOR_INCLUDED,
        alpha=ALPHA_HISTOGRAM_SELECTED,
        edgecolor="white",
        linewidth=0.4,
        label=f"Selected (n = {n_selected})",
        zorder=3
    )
    
    ax.axvline(
        MEDIAN_GLUCOSE_MIN,
        color=COLOR_THRESHOLD,
        lw=1.2, ls="--",
        zorder=5,
        label=f"Lower bound ({format_glucose(MEDIAN_GLUCOSE_MIN)})"
    )
    ax.axvline(
        MEDIAN_GLUCOSE_MAX,
        color=COLOR_THRESHOLD,
        lw=1.2, ls="--",
        zorder=5,
        label=f"Upper bound ({format_glucose(MEDIAN_GLUCOSE_MAX)})"
    )
    
    ax.axvspan(
        MEDIAN_GLUCOSE_MIN,
        MEDIAN_GLUCOSE_MAX,
        color=COLOR_ZONE,
        alpha=ALPHA_ZONE_SHADING,
        zorder=1
    )
    
    style_axes(ax, xlabel=glucose_label("Median glucose"),
               ylabel="Number of participants")
    apply_glucose_axis(ax, axis="x", set_label=False)
    
    pct = n_selected / n_total * 100
    ax.text(
        0.97, 0.96,
        f"Selected: {n_selected}/{n_total} ({pct:.1f}%)",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=7,
        color=COLOR_INCLUDED,
        fontweight="bold"
    )
    
    ax.legend(fontsize=6.5, loc="center right", framealpha=0.88)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

# =============================================================================
# PANEL E: MEDIAN GLUCOSE VS TIR SCATTER
# =============================================================================
def draw_median_tir_scatter(ax, df_all, df_selected):
    """Draw median glucose vs TIR scatter plot."""
    from src.filtering.filtering_config import MEDIAN_GLUCOSE_MIN, MEDIAN_GLUCOSE_MAX
    
    n_selected = len(df_selected)
    n_total = len(df_all)
    n_excluded = n_total - n_selected
    
    excluded_ids = set(df_all[PATIENT_ID_COL]) - set(df_selected[PATIENT_ID_COL])
    df_excluded = df_all[df_all[PATIENT_ID_COL].isin(excluded_ids)]
    
    # Excluded patients
    ax.scatter(
        df_excluded[COL_MEDIAN_GLUCOSE],
        df_excluded[COL_TIR],
        s=SCATTER_SIZE_EXCLUDED,
        color=COLOR_EXCLUDED,
        alpha=SCATTER_ALPHA_EXCLUDED,
        linewidths=0,
        zorder=1,
        label=f"Excluded (n = {n_excluded})"
    )
    
    # Selected patients (colored by CV) - EXACT PATTERN FROM WORKING CODE
    if COL_CV in df_selected.columns:
        hue_values = df_selected[COL_CV].values
        
        scatter = ax.scatter(
            df_selected[COL_MEDIAN_GLUCOSE],
            df_selected[COL_TIR],
            c=hue_values,
            cmap=COLORMAP_CONTINUOUS,
            s=SCATTER_SIZE_SELECTED,
            alpha=SCATTER_ALPHA_SELECTED,
            linewidths=SCATTER_EDGE_WIDTH,
            edgecolors="white",
            zorder=3,
            label=f"Selected (n = {n_selected})"
        )
        
        cb = plt.colorbar(scatter, ax=ax,pad=0.02, shrink=0.75,
                          aspect=18, fraction=0.035)
        cb.set_label("Coefficient of variation", fontsize=7, labelpad=4)
        cb.ax.tick_params(labelsize=6.5, length=2.5, width=0.7)
        cb.outline.set_linewidth(0.6)
    
    # Threshold lines
    ax.axvline(MEDIAN_GLUCOSE_MIN, color=COLOR_THRESHOLD, lw=0.9, ls="--", alpha=0.80, zorder=4)
    ax.axvline(MEDIAN_GLUCOSE_MAX, color=COLOR_THRESHOLD, lw=0.9, ls="--", alpha=0.80, zorder=4)
    
    style_axes(ax, xlabel=glucose_label("Median glucose"), ylabel=LABEL_TIR)
    apply_glucose_axis(ax, axis="x", set_label=False)
    
    pct = n_selected / n_total * 100
    ax.text(
        0.97, 0.3,
        f"Selected: {n_selected}/{n_total} ({pct:.1f}%)",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=7,
        color=COLOR_INCLUDED,
        fontweight="bold"
    )

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def compute_timestep_distribution():
    """Compute raw timestep distribution."""
    patient_ids = resolve_patients()
    
    raw_steps = {}
    print("  Computing timestep distribution...")
    for pid in tqdm(patient_ids, desc="Patients", ncols=80):
        df_raw = load_raw_time_column(pid)
        if df_raw is None:
            raw_steps[pid] = None
            continue
        
        raw_med = dominant_step_of_series(df_raw["time"])
        raw_steps[pid] = (
            snap_to_canonical(raw_med)
            if not np.isnan(raw_med)
            else None
        )
    
    valid_steps = [s for s in raw_steps.values() if s is not None]
    
    return {
        5: valid_steps.count(5),
        10: valid_steps.count(10),
        15: valid_steps.count(15),
        "other": len(valid_steps) - valid_steps.count(5) - valid_steps.count(10) - valid_steps.count(15),
    }

def compute_coverage_statistics(data):
    """Compute coverage summary statistics."""
    if len(data) == 0:
        return {k: np.nan for k in ["n", "mean", "median", "sd", "q25", "q75", "min", "max"]}
    
    return {
        "n": len(data),
        "mean": np.mean(data),
        "median": np.median(data),
        "sd": np.std(data, ddof=1),
        "q25": np.percentile(data, 25),
        "q75": np.percentile(data, 75),
        "min": np.min(data),
        "max": np.max(data),
    }

def add_panel_label(ax, label, x=-0.15, y=1.05):
    """Add panel label (A, B, C, etc.) to axis."""
    ax.text(
        x, y,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="right"
    )

# =============================================================================
# MAIN FIGURE GENERATION
# =============================================================================
def generate_supplementary_figure_1_data_quality():
    """Generate complete supplementary figure 1."""
    
    print("\n" + "=" * 70)
    print("  GENERATING SUPPLEMENTARY FIGURE 1")
    print("=" * 70)
    
    apply_publication_style()
    
    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("\n  Loading data...")
    
    # Timestep data
    raw_n = compute_timestep_distribution()
    
    # Coverage data
    df_windows = pd.read_csv(WINDOW_SUMMARY_CSV)
    df_coverage = df_windows[df_windows[COL_SAMPLING] == REQUIRED_SAMPLING_MIN].copy()
    coverage_data = df_coverage[COL_COVERAGE].values
    coverage_stats = compute_coverage_statistics(coverage_data)
    
    # Filtering data
    from src.config import DATA_PATH
    metrics_csv = DATA_PATH / "Metrics" / "metrics_all_patients.csv"
    df_all = load_patient_metrics(metrics_csv)
    df_selected = apply_median_glucose_filter(df_all)
    
    print(f"  Timestep: {raw_n[5]} patients (5 min), {raw_n[15]} patients (15 min)")
    print(f"  Coverage: {len(df_coverage)} patients")
    print(f"  Filtering: {len(df_selected)}/{len(df_all)} selected")
    
    # -------------------------------------------------------------------------
    # Create figure
    # -------------------------------------------------------------------------
    print("\n  Creating figure...")
    
    fig = plt.figure(figsize=(7.24, 7.5))
    
    # Create grid: 2 rows x 4 columns
    # Row 1: [timestep (1/4), coverage hist (2/4), coverage strip (1/4)]
    # Row 2: [median hist (2/4), scatter (2/4)] - equal 50/50 split
    
    gs = fig.add_gridspec(
        2, 4,
        height_ratios=[1, 1.1],
        width_ratios=[1.0, 1.0, 1.0, 1.0],  # Equal columns
        hspace=0.35,
        wspace=0.45,
        left=0.08,
        right=0.97,
        top=0.96,
        bottom=0.06
    )
    
    # Top row - 1/4, 2/4, 1/4 split
    ax_timestep = fig.add_subplot(gs[0, 0])      # 1 column (1/4)
    ax_cov_hist = fig.add_subplot(gs[0, 1:3])    # 2 columns (2/4)
    ax_cov_strip = fig.add_subplot(gs[0, 3])     # 1 column (1/4)
    
    # Bottom row - equal 50/50 split
    ax_med_hist = fig.add_subplot(gs[1, 0:2])   # Spans columns 0-1 (50%)
    ax_scatter = fig.add_subplot(gs[1, 2:4])     # Spans columns 2-3 (50%)
    
    # -------------------------------------------------------------------------
    # Draw panels
    # -------------------------------------------------------------------------   
    draw_timestep_panel(ax_timestep, raw_n)
    add_panel_label(ax_timestep, "A", x=-0.18)
    
    draw_coverage_histogram(ax_cov_hist, coverage_data, coverage_stats)
    add_panel_label(ax_cov_hist, "B", x=-0.12)
    
    draw_coverage_strip(ax_cov_strip, coverage_data, coverage_stats)
    add_panel_label(ax_cov_strip, "C", x=-0.25)
    
    draw_median_glucose_histogram(ax_med_hist, df_all, df_selected)
    add_panel_label(ax_med_hist, "D", x=-0.06, y=1.03)
    
    draw_median_tir_scatter(ax_scatter, df_all, df_selected)
    add_panel_label(ax_scatter, "E", x=-0.1, y=1.03)
    
    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    output_dir = FIG_DIR / "Supplementary"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = output_dir / "supplementary_figure_1"
    save_figure_with_panels(fig, output_path, "supplementary_figure_1",
                            {"A": ax_timestep, "B": ax_cov_hist, "C": ax_cov_strip,
                             "D": ax_med_hist, "E": ax_scatter})
    
    print("\n" + "=" * 70)
    print("  FIGURE SAVED")
    print("=" * 70)
    print(f"  Location: {output_path}")
    

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    generate_supplementary_figure_1_data_quality()