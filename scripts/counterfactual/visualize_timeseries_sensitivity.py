"""
scripts/counterfactual/visualize_timeseries_sensitivity.py
==========================================================
Time Series Visualization: Original vs Counterfactual Signals
==============================================================

Generates three types of figures showing N days of CGM data:

1. Individual delta figures (one file per delta value)
   - Single panel comparing original vs one specific counterfactual
   - Files: {patient_id}_timeseries_delta_{X}.pdf/.png

2. Subplots comparison (all deltas in separate panels)
   - Multiple panels arranged in grid
   - Each panel shows original vs one counterfactual
   - File: {patient_id}_timeseries_subplots.pdf/.png

3. Overlay comparison (all deltas on same plot)
   - Single panel with original + all counterfactual curves
   - File: {patient_id}_timeseries_overlay.pdf/.png

Input:  Results/Sensitivity_LinInterp_Modular/Signals/{patient_id}_signals.csv
Output: Results/Sensitivity_LinInterp_Modular/Figures/{patient_id}/timeseries/

Usage:
    python3 visualize_timeseries_sensitivity.py --patient_id A168
    python3 visualize_timeseries_sensitivity.py --patient_id A168 --n_days 7 --start_day 0

Authors: Guillaume Deside

Usage
-----
    python3 -m scripts.counterfactual.visualize_timeseries_sensitivity [--help for options]
    (or: make timeseries_sensitivity)
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    TIME_COL,
    NIGHT_START_H,
    NIGHT_END_H,
    SENSITIVITY_PATH,
    FIGURE_WIDTH_INCHES,
    FIGURE_HEIGHT_INCHES,
    THRESHOLD_HYPER,
    THRESHOLD_HYPO,
)

from src.counterfactual.load_counterfactual import load_patient_signals, extract_delta_values

from src.visualization.style import apply_publication_style, save_figure, style_axes
from src.visualization.visualization_helper import (
    shade_nights_on_axis,
    setup_relative_time_axis,
)

from src.visualization.colors import (
    OFFSET_COLORS,
    COLOR_AGP_TARGET_ZONE,
    COLOR_ORIGINAL,
    COLOR_HYPO,
    COLOR_THRESHOLD,
)
from src.utils.unit_conversion import format_glucose

# =============================================================================
# CONFIGURATION
# =============================================================================
SIGNALS_DIR = os.path.join(SENSITIVITY_PATH, "Signals")
FIGURES_OUTPUT_DIR = os.path.join(SENSITIVITY_PATH, "Figures")

# ── Default parameters ───────────────────────────────────────────────────────
DEFAULT_N_DAYS = 9          # Number of days to plot
DEFAULT_START_DAY = 10       # Starting day offset (0 = from beginning)

# Subplot layout
SUBPLOT_NCOLS = 2               # Number of columns in subplot grid

# ── Color scheme ─────────────────────────────────────────────────────────────
ALPHA_ORIGINAL = 0.85
ALPHA_COUNTERFACTUAL = 0.80

# Line widths
LW_ORIGINAL = 1.0
LW_COUNTERFACTUAL = 1.2
LW_THRESHOLD = 0.8

# Y-axis limits
Y_MIN = 50
Y_MAX = 400


def get_time_window(signals_df: pd.DataFrame, n_days: int, 
                    start_day: int = 0) -> pd.DataFrame:
    """
    Extract a time window of N days from signals dataframe.
    
    Parameters
    ----------
    signals_df : pd.DataFrame
        Full signal data
    n_days : int
        Number of days to extract
    start_day : int
        Offset in days from start (0 = from beginning)
        
    Returns
    -------
    pd.DataFrame
        Windowed data
    """
    # Calculate day offset for each measurement
    t_start = signals_df[TIME_COL].min()
    signals_df = signals_df.copy()
    signals_df['day_offset'] = (signals_df[TIME_COL] - t_start).dt.total_seconds() / 86400
    
    # Extract window
    window_start = start_day
    window_end = start_day + n_days
    
    mask = (signals_df['day_offset'] >= window_start) & (signals_df['day_offset'] < window_end)
    windowed_df = signals_df[mask].copy()
    
    if len(windowed_df) == 0:
        return None
    
    return windowed_df


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def add_reference_lines(ax: plt.Axes):
    """Add hyperglycemia and hypoglycemia threshold lines."""
    ax.axhline(THRESHOLD_HYPER, color=COLOR_THRESHOLD, ls="--", lw=LW_THRESHOLD, 
              alpha=0.75, zorder=2,
              label=f"Hyperglycemia ({format_glucose(THRESHOLD_HYPER)})")
    ax.axhline(THRESHOLD_HYPO, color=COLOR_HYPO, ls="--", lw=LW_THRESHOLD,
              alpha=0.65, zorder=2,
              label=f"Hypoglycemia ({format_glucose(THRESHOLD_HYPO)})")
    # Target range shading
    ax.axhspan(THRESHOLD_HYPO, THRESHOLD_HYPER, alpha=0.03, color=COLOR_AGP_TARGET_ZONE, zorder=0)


# =============================================================================
# FIGURE 1: INDIVIDUAL DELTA FIGURES
# =============================================================================
def generate_individual_delta_figure(window_df: pd.DataFrame, delta: int,
                                    patient_id: str, n_days: int) -> plt.Figure:
    """
    Generate single-panel figure comparing original vs one counterfactual.
    
    Parameters
    ----------
    window_df : pd.DataFrame
        Windowed signal data
    delta : int
        Delta value to plot
    patient_id : str
        Patient identifier
    n_days : int
        Number of days shown
        
    Returns
    -------
    plt.Figure
    """
    apply_publication_style()
    
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.10)
    
    # Title
    t_start = window_df[TIME_COL].min()
    t_end = window_df[TIME_COL].max()
    fig.suptitle(
        f"Participant {patient_id}  ·  {n_days}-day time series  ·  δ = {delta} min\n"
        # f"{t_start.strftime('%-d %b %Y')} to {t_end.strftime('%-d %b %Y')}"
        ,fontsize=8, fontweight="bold"
    )
    
    # Add night shading
    shade_nights_on_axis(ax, t_start, t_end, [])
    
    # Add reference lines
    add_reference_lines(ax)
    
    # Plot original signal
    ax.plot(
        window_df[TIME_COL], window_df["glucose_original"],
        color=COLOR_ORIGINAL, lw=LW_ORIGINAL, alpha=ALPHA_ORIGINAL,
        label="Original", zorder=4
    )
    
    # Plot counterfactual
    col_name = f"glucose_ctf_delta_{delta}"
    if col_name in window_df.columns:
        color = OFFSET_COLORS.get(delta, "#999999")
        ax.plot(
            window_df[TIME_COL], window_df[col_name],
            color=color, lw=LW_COUNTERFACTUAL, alpha=ALPHA_COUNTERFACTUAL,
            label=f"Counterfactual (δ={delta} min)", zorder=5
        )
    
    setup_relative_time_axis(ax, t_start, t_end, n_days)
    style_axes(ax)
    
    # Legend
    ax.legend(fontsize=7, loc="upper right", framealpha=0.75)
    
    # X-axis label
    ax.set_xlabel(f"Time", fontsize=8, labelpad=3)
    
    return fig


# =============================================================================
# FIGURE 2: SUBPLOTS COMPARISON
# =============================================================================
def generate_subplots_figure(window_df: pd.DataFrame, delta_values: list,
                            patient_id: str, n_days: int) -> plt.Figure:
    """
    Generate multi-panel figure with one subplot per delta value.
    
    Parameters
    ----------
    window_df : pd.DataFrame
        Windowed signal data
    delta_values : list
        List of delta values to plot
    patient_id : str
        Patient identifier
    n_days : int
        Number of days shown
        
    Returns
    -------
    plt.Figure
    """
    apply_publication_style()
    
    n_deltas = len(delta_values)
    nrows = int(np.ceil(n_deltas / SUBPLOT_NCOLS))
    
    fig, axes = plt.subplots(
        nrows, SUBPLOT_NCOLS,
        figsize=(FIGURE_WIDTH_INCHES, 2*FIGURE_HEIGHT_INCHES)
    )
    fig.subplots_adjust(left=0.09, right=0.97, top=0.94, bottom=0.08, 
                       hspace=0.40, wspace=0.28)
    
    # Title
    t_start = window_df[TIME_COL].min()
    t_end = window_df[TIME_COL].max()
    fig.suptitle(
        f"Participant {patient_id}  ·  {n_days}-day comparison by delta\n"
        # f"{t_start.strftime('%-d %b %Y')} to {t_end.strftime('%-d %b %Y')}"
        ,fontsize=8, fontweight="bold"
    )
    
    # Flatten axes for easier iteration
    axes_flat = np.array(axes).reshape(-1) if n_deltas > 1 else [axes]
    
    for idx, delta in enumerate(delta_values):
        ax = axes_flat[idx]
        
        # Add night shading
        shade_nights_on_axis(ax, t_start, t_end, [])
        
        # Add reference lines
        add_reference_lines(ax)
        
        # Plot original
        ax.plot(
            window_df[TIME_COL], window_df["glucose_original"],
            color=COLOR_ORIGINAL, lw=LW_ORIGINAL - 0.2, alpha=ALPHA_ORIGINAL - 0.1,
            label="Original", zorder=4
        )
        
        # Plot counterfactual
        col_name = f"glucose_ctf_delta_{delta}"
        if col_name in window_df.columns:
            color = OFFSET_COLORS.get(delta, "#999999")
            ax.plot(
                window_df[TIME_COL], window_df[col_name],
                color=color, lw=LW_COUNTERFACTUAL, alpha=ALPHA_COUNTERFACTUAL,
                label=f"Counterfactual", zorder=5
            )
        
        # Format axes
        #format_time_axis(ax, window_df, n_days)
        setup_relative_time_axis(ax, t_start, t_end, n_days)
        style_axes(ax)
        
        # Title for each panel
        ax.set_title(f"δ = {delta} min", fontsize=8, fontweight="bold", 
                    loc="left", pad=4)
        
        # Legend
        ax.legend(fontsize=6, loc="upper right", framealpha=0.90)
        
        # X-axis label (only bottom row)
        if idx >= (nrows - 1) * SUBPLOT_NCOLS:
            ax.set_xlabel(f"Time", fontsize=7, labelpad=2)
    
    # Hide unused subplots
    for idx in range(n_deltas, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    return fig


# =============================================================================
# FIGURE 3: OVERLAY COMPARISON
# =============================================================================
def generate_overlay_figure(window_df: pd.DataFrame, delta_values: list,
                           patient_id: str, n_days: int) -> plt.Figure:
    """
    Generate single-panel figure with all counterfactuals overlaid.
    
    Parameters
    ----------
    window_df : pd.DataFrame
        Windowed signal data
    delta_values : list
        List of delta values to plot
    patient_id : str
        Patient identifier
    n_days : int
        Number of days shown
        
    Returns
    -------
    plt.Figure
    """
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.10)
    
    # Title
    t_start = window_df[TIME_COL].min()
    t_end = window_df[TIME_COL].max()
    fig.suptitle(
        f"Participant {patient_id}  ·  {n_days}-day time series overlay\n"
        # f"{t_start.strftime('%-d %b %Y')} to {t_end.strftime('%-d %b %Y')}  ·  "
        # f"{len(delta_values)} counterfactual variants",
        ,fontsize=8, fontweight="bold"
    )
    
    # Add night shading
    shade_nights_on_axis(ax, t_start, t_end, [])
    
    # Add reference lines
    ax.axhline(THRESHOLD_HYPER, color=COLOR_THRESHOLD, ls="--", lw=LW_THRESHOLD, 
              alpha=0.75, zorder=2)
    ax.axhline(THRESHOLD_HYPO, color=COLOR_HYPO, ls="--", lw=LW_THRESHOLD,
              alpha=0.65, zorder=2)
    ax.axhspan(THRESHOLD_HYPO, THRESHOLD_HYPER, alpha=0.03, color=COLOR_AGP_TARGET_ZONE, zorder=0)
    
    # Plot original signal (thicker, on top)
    ax.plot(
        window_df[TIME_COL], window_df["glucose_original"],
        color=COLOR_ORIGINAL, lw=LW_ORIGINAL + 0.3, alpha=ALPHA_ORIGINAL,
        label="Original", zorder=10
    )
    
    # Plot all counterfactuals
    for delta in delta_values:
        col_name = f"glucose_ctf_delta_{delta}"
        if col_name in window_df.columns:
            color = OFFSET_COLORS.get(delta, "#999999")
            ax.plot(
                window_df[TIME_COL], window_df[col_name],
                color=color, lw=LW_COUNTERFACTUAL - 0.1, alpha=ALPHA_COUNTERFACTUAL - 0.1,
                label=f"δ={delta} min", zorder=8
            )
    
    setup_relative_time_axis(ax, t_start, t_end, n_days)
    style_axes(ax)

    # Legend
    ax.legend(fontsize=7, loc="upper right", framealpha=0.95, ncol=2)
    
    # X-axis label
    ax.set_xlabel(f"Time ({n_days} days)", fontsize=8, labelpad=3)
    
    return fig


# =============================================================================
# MAIN PROCESSING
# =============================================================================
def process_patient(patient_id: str, n_days: int, start_day: int):
    """
    Generate all time series figures for a patient.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier
    n_days : int
        Number of days to plot
    start_day : int
        Starting day offset
    """
    print(f"\n  Processing {patient_id}...")
    print(f"    Window: {n_days} days starting from day {start_day}")
    
    # Load signals
    signals_df = load_patient_signals(patient_id)
    if signals_df is None:
        print(f"    [!] No signal data found")
        return False
    
    # Extract delta values
    delta_values = extract_delta_values(signals_df)
    if not delta_values:
        print(f"    [!] No counterfactual signals found")
        return False
    
    print(f"    Delta values: {delta_values}")
    
    # Get time window
    window_df = get_time_window(signals_df, n_days, start_day)
    if window_df is None or len(window_df) == 0:
        print(f"    [!] No data in requested window (day {start_day} to {start_day + n_days})")
        total_days = (signals_df[TIME_COL].max() - signals_df[TIME_COL].min()).total_seconds() / 86400
        print(f"    Available: {total_days:.1f} days total")
        return False
    
    print(f"    Data points in window: {len(window_df)}")
    
    # Create output directory
    output_dir = os.path.join(FIGURES_OUTPUT_DIR, patient_id, "timeseries")
    os.makedirs(output_dir, exist_ok=True)
    
    
    # Figure 1: Individual delta figures
    for delta in delta_values:
        fig = generate_individual_delta_figure(window_df, delta, patient_id, n_days)
        output_path = os.path.join(output_dir, f"{patient_id}_timeseries_delta_{delta}")
        save_figure(fig, output_path)
    
    # Figure 2: Subplots comparison
    fig_subplots = generate_subplots_figure(window_df, delta_values, patient_id, n_days)
    output_path_subplots = os.path.join(output_dir, f"{patient_id}_timeseries_subplots")
    save_figure(fig_subplots, output_path_subplots)
    
    # Figure 3: Overlay comparison
    fig_overlay = generate_overlay_figure(window_df, delta_values, patient_id, n_days)
    output_path_overlay = os.path.join(output_dir, f"{patient_id}_timeseries_overlay")
    save_figure(fig_overlay, output_path_overlay)
    
    print(f"Saved {len(delta_values) + 2} figures to {output_dir}/")
    return True


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution pipeline."""
    parser = argparse.ArgumentParser(
        description="Time series visualization for sensitivity analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--patient_id", required=True, help="Patient ID (e.g., A168)")
    parser.add_argument("--n_days", type=int, default=DEFAULT_N_DAYS,
                       help="Number of days to plot")
    parser.add_argument("--start_day", type=int, default=DEFAULT_START_DAY,
                       help="Starting day offset (0 = from beginning)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  Time Series Visualization for Sensitivity Analysis")
    print("=" * 70)
    print(f"  Patient:    {args.patient_id}")
    print(f"  Duration:   {args.n_days} days")
    print(f"  Start day:  {args.start_day}")
    print(f"  Signals:    {SIGNALS_DIR}")
    print(f"  Output:     {FIGURES_OUTPUT_DIR}")
    print("=" * 70)
    
    success = process_patient(args.patient_id, args.n_days, args.start_day)
    
    if success:
        print("\n" + "=" * 70)
        print("  Done!")
        print("=" * 70 + "\n")
    else:
        print("\n" + "=" * 70)
        print("  Processing failed")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
