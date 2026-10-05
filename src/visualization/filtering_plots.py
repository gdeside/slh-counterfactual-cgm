"""
src/visualization/filtering_plots.py
======================================

Publication-ready figures for the patient-filtering (cohort-selection) step.

Functions
---------
compute_adaptive_bins
    Build histogram bin edges aligned with the inclusion-criterion boundaries
    so that all selected patients fall within complete bins.
generate_histogram_figure
    Stacked histogram of median glucose for excluded vs selected patients,
    with threshold lines and a shaded inclusion zone.
generate_scatter_figure
    Scatter plot of two CGM metrics (e.g. CV vs TIR), with excluded patients
    shown in grey and selected patients coloured by a third metric.

Color constants are imported from ``src.visualization.colors``; filtering
thresholds from ``src.filtering.filtering_config``; patient ID column and
global config from ``src.config``.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

from src.config import PATIENT_ID_COL, FIGURE_WIDTH_INCHES

from src.filtering.filtering_config import (
    MEDIAN_GLUCOSE_MIN,
    MEDIAN_GLUCOSE_MAX,
    COL_MEDIAN_GLUCOSE,
    COL_CV,
    COL_TIR,
    LABEL_CV,
    LABEL_TIR,
)

from src.visualization.colors import COLOR_INCLUDED, COLOR_EXCLUDED, COLOR_THRESHOLD, COLOR_ZONE, COLORMAP_CONTINUOUS

from src.visualization.style import apply_publication_style, style_axes, save_figure
from src.visualization.glucose_axis import apply_glucose_axis, apply_glucose_colorbar
from src.utils.unit_conversion import format_glucose, glucose_label

# Transparency levels
ALPHA_HISTOGRAM_BAR = 0.90
ALPHA_HISTOGRAM_SELECTED = 0.85
ALPHA_ZONE_SHADING = 0.06

# Histogram parameters
HISTOGRAM_N_BINS = 30
HISTOGRAM_BIN_ALIGNMENT = True

# Scatter plot parameters
SCATTER_SIZE_EXCLUDED = 18
SCATTER_SIZE_SELECTED = 32
SCATTER_ALPHA_EXCLUDED = 0.60
SCATTER_ALPHA_SELECTED = 0.85
SCATTER_EDGE_WIDTH = 0.35


def compute_adaptive_bins(data_values, lower_bound, upper_bound, n_bins):
    """
    Compute histogram bins aligned with threshold boundaries.
    
    Ensures that bin edges align with the lower and upper bounds of the
    criterion, so all selected patients fall within complete bins.
    
    Parameters
    ----------
    data_values : np.ndarray
        All data values
    lower_bound : float
        Lower threshold
    upper_bound : float
        Upper threshold
    n_bins : int
        Approximate number of bins
        
    Returns
    -------
    np.ndarray
        Bin edges
    """
    if not HISTOGRAM_BIN_ALIGNMENT:
        # Simple quantile-based bins
        q_lo = float(np.nanpercentile(data_values, 1))
        q_hi = float(np.nanpercentile(data_values, 99))
        return np.linspace(q_lo, q_hi, n_bins + 1)
    
    # Adaptive binning aligned with thresholds
    # Strategy: create bins such that [lower_bound, upper_bound] are bin edges
    
    # Find data range
    data_min = float(np.nanmin(data_values))
    data_max = float(np.nanmax(data_values))
    
    # Compute bin width based on the threshold range
    threshold_range = upper_bound - lower_bound
    bin_width = threshold_range / (n_bins * 0.6)  # ~60% of bins in threshold range
    
    # Build bin edges
    # Left side: from data_min to lower_bound
    left_edges = np.arange(lower_bound, data_min - bin_width, -bin_width)[::-1]
    
    # Center: from lower_bound to upper_bound
    center_edges = np.arange(lower_bound, upper_bound + bin_width/2, bin_width)
    
    # Right side: from upper_bound to data_max
    right_edges = np.arange(upper_bound + bin_width, data_max + bin_width, bin_width)
    
    # Combine all edges (remove duplicates)
    all_edges = np.unique(np.concatenate([left_edges, center_edges, right_edges]))
    
    return all_edges

# ─────────────────────────────────────────────────────────────────────────────
# HISTOGRAM FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def generate_histogram_figure(df_all, df_selected, output_dir):
    """
    Generate histogram showing distribution of median glucose.
    
    Shows excluded patients in grey and selected patients in blue,
    with threshold lines and shaded inclusion zone.
    
    Parameters
    ----------
    df_all : pd.DataFrame
        All patients
    df_selected : pd.DataFrame
        Selected patients
    output_dir : str
        Output directory for figures
    """
    apply_publication_style()
    
    n_selected = len(df_selected)
    n_total = len(df_all)
    n_excluded = n_total - n_selected
    
    # Extract values
    values_all = df_all[COL_MEDIAN_GLUCOSE].dropna().values
    values_selected = df_selected[COL_MEDIAN_GLUCOSE].dropna().values
    
    # Get excluded values
    excluded_ids = set(df_all[PATIENT_ID_COL]) - set(df_selected[PATIENT_ID_COL])
    df_excluded = df_all[df_all[PATIENT_ID_COL].isin(excluded_ids)]
    values_excluded = df_excluded[COL_MEDIAN_GLUCOSE].dropna().values
    
    # Compute adaptive bins
    bins = compute_adaptive_bins(
        values_all, 
        MEDIAN_GLUCOSE_MIN, 
        MEDIAN_GLUCOSE_MAX, 
        HISTOGRAM_N_BINS
    )
    
    # Create figure
    fig = plt.figure(figsize=(FIGURE_WIDTH_INCHES * 0.58, FIGURE_WIDTH_INCHES * 0.44))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.15)
    ax = fig.add_subplot(111)
    
    # Plot histograms (stacked: excluded then selected)
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
    
    # Threshold lines
    ax.axvline(
        MEDIAN_GLUCOSE_MIN, 
        color=COLOR_THRESHOLD, 
        lw=1.2, 
        ls="--", 
        zorder=5,
        label=f"Lower bound ({format_glucose(MEDIAN_GLUCOSE_MIN)})"
    )
    ax.axvline(
        MEDIAN_GLUCOSE_MAX, 
        color=COLOR_THRESHOLD, 
        lw=1.2, 
        ls="--", 
        zorder=5,
        label=f"Upper bound ({format_glucose(MEDIAN_GLUCOSE_MAX)})"
    )
    
    # Shaded inclusion zone
    ax.axvspan(
        MEDIAN_GLUCOSE_MIN, 
        MEDIAN_GLUCOSE_MAX, 
        color=COLOR_ZONE, 
        alpha=ALPHA_ZONE_SHADING, 
        zorder=1
    )
    
    # Labels and styling
    apply_glucose_axis(ax, axis="x", quantity="Median glucose",
                       fontsize=8, labelpad=3)
    ax.set_ylabel("Number of participants", fontsize=8, labelpad=3)
    ax.set_title(
        f"Median glucose distribution — Selection criterion "
        f"[{format_glucose(MEDIAN_GLUCOSE_MIN, with_unit=False)}, "
        f"{format_glucose(MEDIAN_GLUCOSE_MAX)}]",
        fontsize=8, 
        fontweight="bold", 
        pad=6,
    )
    
    # Percentage annotation
    pct = n_selected / n_total * 100
    ax.text(
        0.97, 0.96,
        f"Selected: {n_selected}/{n_total} ({pct:.1f}%)",
        transform=ax.transAxes, 
        ha="right", 
        va="top",
        fontsize=7, 
        color=COLOR_INCLUDED, 
        fontweight="bold"
    )
    
    # Legend and formatting
    ax.legend(fontsize=6.5, loc="center right", framealpha=0.88)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
    style_axes(ax)
    
    # Save
    output_path = os.path.join(output_dir, "median_glucose_histogram")
    save_figure(fig, output_path)



def generate_scatter_figure(df_all, df_selected, x_metric, y_metric, 
                            hue_metric, output_dir):
    """
    Generate scatter plot showing two metrics.
    
    Parameters
    ----------
    df_all : pd.DataFrame
        All patients
    df_selected : pd.DataFrame
        Selected patients
    x_metric : str
        Column name for x-axis metric
    y_metric : str
        Column name for y-axis metric
    hue_metric : str
        Column name for color-coding metric
    output_dir : str
        Output directory for figures
    """
    apply_publication_style()
    
    n_selected = len(df_selected)
    n_total = len(df_all)
    n_excluded = n_total - n_selected
    
    # Get metric labels
    metric_labels = {
        COL_MEDIAN_GLUCOSE: glucose_label("Median glucose"),
        COL_CV: LABEL_CV,
        COL_TIR: LABEL_TIR,
    }
    
    x_label = metric_labels.get(x_metric, x_metric)
    y_label = metric_labels.get(y_metric, y_metric)
    hue_label = metric_labels.get(hue_metric, hue_metric)
    
    # Get excluded patients
    excluded_ids = set(df_all[PATIENT_ID_COL]) - set(df_selected[PATIENT_ID_COL])
    df_excluded = df_all[df_all[PATIENT_ID_COL].isin(excluded_ids)]
    
    # Create figure
    fig = plt.figure(figsize=(FIGURE_WIDTH_INCHES * 0.62, FIGURE_WIDTH_INCHES * 0.58))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.13)
    ax = fig.add_subplot(111)
    
    # Plot excluded patients (background, grey)
    ax.scatter(
        df_excluded[x_metric], 
        df_excluded[y_metric],
        s=SCATTER_SIZE_EXCLUDED, 
        color=COLOR_EXCLUDED, 
        alpha=SCATTER_ALPHA_EXCLUDED,
        linewidths=0, 
        zorder=1,
        label=f"Excluded (n = {n_excluded})"
    )
    
    # Plot selected patients (colored by third metric)
    if hue_metric in df_selected.columns:
        hue_values = df_selected[hue_metric].values
        scatter = ax.scatter(
            df_selected[x_metric], 
            df_selected[y_metric],
            c=hue_values, 
            cmap=COLORMAP_CONTINUOUS,
            s=SCATTER_SIZE_SELECTED, 
            alpha=SCATTER_ALPHA_SELECTED,
            linewidths=SCATTER_EDGE_WIDTH, 
            edgecolors="white",
            zorder=3, 
            label=f"Selected (n = {n_selected})"
        )
        
        # Add colorbar
        cb = fig.colorbar(scatter, ax=ax, pad=0.02, shrink=0.75,
                          aspect=18, fraction=0.035)
        if hue_metric == COL_MEDIAN_GLUCOSE:
            apply_glucose_colorbar(cb, quantity="Median glucose",
                                   fontsize=7, labelpad=4)
        else:
            cb.set_label(hue_label, fontsize=7, labelpad=4)
        cb.ax.tick_params(labelsize=6.5, length=2.5, width=0.7)
        cb.outline.set_linewidth(0.6)
    else:
        ax.scatter(
            df_selected[x_metric], 
            df_selected[y_metric],
            s=SCATTER_SIZE_SELECTED, 
            color=COLOR_INCLUDED, 
            alpha=SCATTER_ALPHA_SELECTED,
            linewidths=SCATTER_EDGE_WIDTH, 
            edgecolors="white",
            zorder=3, 
            label=f"Selected (n = {n_selected})"
        )
    
    # Add threshold lines ONLY for median glucose
    if x_metric == COL_MEDIAN_GLUCOSE:
        ax.axvline(
            MEDIAN_GLUCOSE_MIN, 
            color=COLOR_THRESHOLD, 
            lw=0.9, 
            ls="--", 
            alpha=0.80, 
            zorder=4
        )
        ax.axvline(
            MEDIAN_GLUCOSE_MAX, 
            color=COLOR_THRESHOLD, 
            lw=0.9, 
            ls="--", 
            alpha=0.80, 
            zorder=4
        )
    
    if y_metric == COL_MEDIAN_GLUCOSE:
        ax.axhline(
            MEDIAN_GLUCOSE_MIN, 
            color=COLOR_THRESHOLD, 
            lw=0.9, 
            ls="--", 
            alpha=0.80, 
            zorder=4
        )
        ax.axhline(
            MEDIAN_GLUCOSE_MAX, 
            color=COLOR_THRESHOLD, 
            lw=0.9, 
            ls="--", 
            alpha=0.80, 
            zorder=4
        )
    
    # Labels and title
    ax.set_xlabel(x_label, fontsize=8, labelpad=3)
    ax.set_ylabel(y_label, fontsize=8, labelpad=3)

    # Labels above already carry the display unit; only the ticks still need it
    if x_metric == COL_MEDIAN_GLUCOSE:
        apply_glucose_axis(ax, axis="x", set_label=False)
    if y_metric == COL_MEDIAN_GLUCOSE:
        apply_glucose_axis(ax, axis="y", set_label=False)
    
    title = f"{x_label} vs {y_label}"
    if hue_metric:
        title += f" (colored by {hue_label})"
    ax.set_title(title, fontsize=8, fontweight="bold", pad=6)
    
    # Percentage annotation
    pct = n_selected / n_total * 100
    ax.text(
        0.97, 0.03,
        f"Selected: {n_selected}/{n_total} ({pct:.1f}%)",
        transform=ax.transAxes, 
        ha="right", 
        va="bottom",
        fontsize=7, 
        color=COLOR_INCLUDED, 
        fontweight="bold"
    )
    
    # Legend
    if not (hue_metric and hue_metric in df_selected.columns):
        ax.legend(fontsize=6.5, loc="upper right", framealpha=0.88)
    else:
        # Custom legend when colorbar is present
        handles = [
            mpatches.Patch(
                facecolor=COLOR_EXCLUDED, 
                alpha=SCATTER_ALPHA_EXCLUDED,
                label=f"Excluded (n = {n_excluded})"
            ),
            mpatches.Patch(
                facecolor=COLOR_INCLUDED, 
                alpha=SCATTER_ALPHA_SELECTED,
                label=f"Selected (n = {n_selected})"
            ),
        ]
        
        # Add threshold line to legend if applicable
        if x_metric == COL_MEDIAN_GLUCOSE or y_metric == COL_MEDIAN_GLUCOSE:
            handles.append(
                Line2D([0], [0], color=COLOR_THRESHOLD, lw=0.9, ls="--",
                       label="Median glucose bounds")
            )
        
        ax.legend(handles=handles, fontsize=6.5, loc="upper right",
                  framealpha=0.88)
    
    style_axes(ax)
    
    # Save
    x_name = x_metric.replace(COL_MEDIAN_GLUCOSE, "median").replace("_", "").lower()
    y_name = y_metric.replace(COL_MEDIAN_GLUCOSE, "median").replace("_", "").lower()
    output_path = os.path.join(output_dir, f"{x_name}_vs_{y_name}_scatter")
    save_figure(fig, output_path)

