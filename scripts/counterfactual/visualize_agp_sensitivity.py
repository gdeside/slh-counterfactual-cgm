"""
scripts/counterfactual/visualize_agp_sensitivity.py
===================================================
AGP Visualization for Sensitivity Analysis Results
===================================================
Input:  Results/Sensitivity_LinInterp_Modular/Data/{patient_id}_sensitivity.csv
        Data/Preprocessed/{patient_id}/{patient_id}_win1_*.xlsx
        Data/Episodes/{patient_id}_episodes.csv

Output: Results/Sensitivity_LinInterp_Modular/Figures/{patient_id}/
            {patient_id}_agp_medians_overlay.pdf/.png
            {patient_id}_agp_subplots.pdf/.png

Generates two figure types:

1. Medians Overlay Figure
   - Single panel showing median glucose curves for:
     * Original signal (dark grey, solid)
     * Counterfactual at δ=0 min (if present)
     * Counterfactual at δ=15 min
     * Counterfactual at δ=30 min
     * Counterfactual at δ=60 min

2. Subplots Figure
   - Multiple panels (one per delta value)
   - Each panel shows:
     * Original signal (median + IQR in grey)
     * Counterfactual signal (median + IQR in color)

Authors: Guillaume Deside

Usage
-----
    python3 -m scripts.counterfactual.visualize_agp_sensitivity [--help for options]
    (or: make agp_sensitivity)
"""

import os
import sys
import argparse
from datetime import datetime, time as _dtime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import configuration
from src.config.constants import H_HYPER, H_HYPO, TIME_COL, GLUCOSE_COL
from src.config.paths import (
    SENSITIVITY_LININTERP_DATA,
    SENSITIVITY_LININTERP_FIGURES,
    SENSITIVITY_LININTERP_SIGNALS,
)
from src.config.figure_config import AGP_Y_MIN, AGP_Y_MAX, FIGURE_WIDTH_INCHES

# Import data loading
from src.counterfactual.load_counterfactual import (
    load_patient_signals,
    load_sensitivity_results,
)
from src.counterfactual.counterfactual_config import APPLY_KERNEL, KERNEL_MIN
from src.preprocessing.timestep import detect_dominant_sampling_interval

# Import AGP computation
from src.agp.compute_agp import compute_agp_profile

# Import visualization
from src.visualization.style import apply_publication_style, save_figure
from src.visualization.colors import (
    OFFSET_COLORS,
    COLOR_THRESHOLD,
    COLOR_HYPO,
    COLOR_AGP_TARGET_ZONE,
    COLOR_ORIGINAL,
)
from src.visualization.glucose_axis import apply_glucose_axis

# =============================================================================
# CONFIGURATION
# =============================================================================

# ── Paths ────────────────────────────────────────────────────────────────────
SENSITIVITY_DATA_DIR = str(SENSITIVITY_LININTERP_DATA)
FIGURES_OUTPUT_DIR = str(SENSITIVITY_LININTERP_FIGURES)


# ── AGP parameters ───────────────────────────────────────────────────────────
AGP_X_TICK_INTERVAL_HOURS = 3

# ── Figure parameters ────────────────────────────────────────────────────────
FIGURE_HEIGHT_OVERLAY = 4.0  # Height for overlay figure
FIGURE_HEIGHT_PER_SUBPLOT = 2.2  # Height per subplot in multi-panel figure

# ── Color scheme ─────────────────────────────────────────────────────────────
COLOR_ORIGINAL_ALPHA = 0.15       # IQR fill alpha

DELTA_ALPHA = 0.20  # IQR fill alpha for counterfactual

# Line widths
LW_MEDIAN = 1.6
LW_THRESHOLD = 0.9
LW_MEDIAN_OVERLAY = 1.3

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_agp_x_axis(agp: pd.DataFrame) -> list:
    """Convert AGP time strings to datetime objects for plotting."""
    today = datetime.today().date()
    return [
        datetime.combine(today, datetime.strptime(t, "%H:%M").time())
        for t in agp["time_str"]
    ]


def format_agp_axis(ax: plt.Axes, title: str = None):
    """Apply standard AGP axis formatting."""
    today = datetime.today().date()
    
    # X-axis limits and formatting
    ax.set_xlim(
        datetime.combine(today, _dtime(0, 0)),
        datetime.combine(today, _dtime(23, 59))
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=AGP_X_TICK_INTERVAL_HOURS))
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=7, rotation=0)
    
    # Y-axis
    ax.set_ylim(AGP_Y_MIN, AGP_Y_MAX)
    apply_glucose_axis(ax, fontsize=8, labelpad=3)
    
    # Reference lines
    ax.axhline(H_HYPER, color=COLOR_THRESHOLD, ls="--", lw=LW_THRESHOLD, 
              alpha=0.75, zorder=2)
    ax.axhline(H_HYPO, color=COLOR_HYPO, ls="--", lw=LW_THRESHOLD,
              alpha=0.65, zorder=2)
    
    # Target range shading
    ax.axhspan(H_HYPO, H_HYPER, alpha=0.03, color=COLOR_AGP_TARGET_ZONE, zorder=0)
    
    # Title if provided
    if title:
        ax.set_title(title, fontsize=8, fontweight="bold", loc="left", pad=4)
    
    # Axis styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(length=3, width=0.8, labelsize=7)


# =============================================================================
# DATA LOADING
# =============================================================================
def load_patient_data(patient_id: str) -> tuple:
    """Load signals and detect timestep. Returns (signals_df, timestep) or (None, None)."""
    signals_df = load_patient_signals(patient_id)
    if signals_df is None:
        return None, None
    return signals_df, detect_dominant_sampling_interval(signals_df)


# =============================================================================
# FIGURE 1: MEDIANS OVERLAY
# =============================================================================
def generate_medians_overlay_figure(patient_id: str, agp_original: pd.DataFrame,
                                    agp_counterfactuals: dict) -> plt.Figure:
    """
    Generate overlay figure with all median curves.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier
    agp_original : pd.DataFrame
        AGP profile for original signal
    agp_counterfactuals : dict
        AGP profiles keyed by delta value
        
    Returns
    -------
    plt.Figure
        Generated figure
    """
    apply_publication_style()
    
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_OVERLAY))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.12)
    
    kernel_str = f"k={KERNEL_MIN} min" if APPLY_KERNEL else "no kernel"
    n_deltas = len(agp_counterfactuals)
    
    fig.suptitle(
        f"Participant {patient_id}  ·  AGP median comparison  ·  {kernel_str}  ·  "
        f"{n_deltas} delta values",
        fontsize=8, fontweight="bold"
    )
    
    # Format axis
    format_agp_axis(ax)
    
    # Get x-axis values
    x_orig = get_agp_x_axis(agp_original)
    
    # Plot original signal (thicker, darker)
    ax.plot(
        x_orig, agp_original["median"],
        color=COLOR_ORIGINAL, lw=LW_MEDIAN + 0.4, 
        label="Original", zorder=10
    )
    
    # Plot counterfactual medians (colored by delta)
    delta_values = sorted(agp_counterfactuals.keys())
    
    for delta in delta_values:
        agp_ctf = agp_counterfactuals[delta]
        x_ctf = get_agp_x_axis(agp_ctf)
        color = OFFSET_COLORS.get(delta, "#999999")
        
        ax.plot(
            x_ctf, agp_ctf["median"],
            color=color, lw=LW_MEDIAN_OVERLAY,
            label=f"δ = {delta} min", zorder=8
        )
    
    # Legend
    ax.legend(fontsize=7, loc="upper right", framealpha=0.95, ncol=2)
    
    # X-axis label
    ax.set_xlabel("Time of day", fontsize=8, labelpad=3)
    
    return fig


# =============================================================================
# FIGURE 2: SUBPLOTS (ONE PER DELTA)
# =============================================================================
def generate_subplots_figure(patient_id: str, agp_original: pd.DataFrame,
                            agp_counterfactuals: dict) -> plt.Figure:
    """
    Generate multi-panel figure with one subplot per delta value.
    
    Each panel shows:
    - Original signal: median + IQR (grey)
    - Counterfactual signal: median + IQR (color)
    
    Parameters
    ----------
    patient_id : str
        Patient identifier
    agp_original : pd.DataFrame
        AGP profile for original signal
    agp_counterfactuals : dict
        AGP profiles keyed by delta value
        
    Returns
    -------
    plt.Figure
        Generated figure
    """
    apply_publication_style()
    
    delta_values = sorted(agp_counterfactuals.keys())
    n_deltas = len(delta_values)
    
    # Determine layout
    ncols = min(2, n_deltas)
    nrows = int(np.ceil(n_deltas / ncols))
    
    fig_height = FIGURE_HEIGHT_PER_SUBPLOT * nrows
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(FIGURE_WIDTH_INCHES, fig_height)
    )
    fig.subplots_adjust(left=0.09, right=0.97, top=0.94, bottom=0.08, 
                       hspace=0.35, wspace=0.25)
    
    kernel_str = f"k={KERNEL_MIN} min" if APPLY_KERNEL else "no kernel"
    fig.suptitle(
        f"Participant {patient_id}  ·  AGP comparison by delta  ·  {kernel_str}",
        fontsize=8, fontweight="bold"
    )
    
    # Flatten axes array for easier iteration
    axes_flat = np.array(axes).reshape(-1) if n_deltas > 1 else [axes]
    
    # Get x-axis values
    x_orig = get_agp_x_axis(agp_original)
    
    # Plot each delta in its own subplot
    for idx, delta in enumerate(delta_values):
        ax = axes_flat[idx]
        agp_ctf = agp_counterfactuals[delta]
        x_ctf = get_agp_x_axis(agp_ctf)
        color = OFFSET_COLORS.get(delta, "#999999")
        
        # Format axis
        format_agp_axis(ax, title=f"δ = {delta} min")
        
        # Original signal: IQR + median
        ax.fill_between(
            x_orig, agp_original["q25"], agp_original["q75"],
            alpha=COLOR_ORIGINAL_ALPHA, color=COLOR_ORIGINAL,
            zorder=1, label="Original IQR"
        )
        ax.plot(
            x_orig, agp_original["median"],
            color=COLOR_ORIGINAL, lw=LW_MEDIAN, ls="--",
            zorder=4, label="Original median"
        )
        
        # Counterfactual signal: IQR + median
        ax.fill_between(
            x_ctf, agp_ctf["q25"], agp_ctf["q75"],
            alpha=DELTA_ALPHA, color=color,
            zorder=2, label=f"Counterfactual IQR"
        )
        ax.plot(
            x_ctf, agp_ctf["median"],
            color=color, lw=LW_MEDIAN,
            zorder=5, label=f"Counterfactual median"
        )
        
        # Legend
        ax.legend(fontsize=6, loc="upper right", framealpha=0.95)
        
        # X-axis label (only bottom row)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Time of day", fontsize=8, labelpad=3)
    
    # Hide unused subplots
    for idx in range(n_deltas, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def process_patient(patient_id: str):
    """Generate AGP figures for a single patient."""
    print(f"\n  Processing {patient_id}...")
    
    # Check if sensitivity results exist
    sens_df = load_sensitivity_results(patient_id)
    if sens_df is None:
        print(f"No sensitivity metrics CSV found")
        return False
    
    # Load saved signals
    signals_df, timestep = load_patient_data(patient_id)
    if signals_df is None:
        print(f"No signal data found")
        print(f"        Expected: {os.path.join(str(SENSITIVITY_LININTERP_SIGNALS), f'{patient_id}_signals.csv')}")
        return False
    
    print(f"    Timestep: {timestep:.0f} min  |  Rows: {len(signals_df)}")
    
    delta_values = sorted(sens_df["offset"].unique())
    print(f"    Delta values: {delta_values}")
    
    # Compute AGP for original signal
    df_original = signals_df[[TIME_COL, "glucose_original"]].copy()
    df_original = df_original.rename(columns={"glucose_original": GLUCOSE_COL})
    agp_original = compute_agp_profile(df_original, timestep)
    
    # Compute AGP for each counterfactual from saved signals
    agp_counterfactuals = {}
    
    for delta in delta_values:
        col_name = f"glucose_ctf_delta_{int(delta)}"
        
        if col_name not in signals_df.columns:
            print(f"    [!] Warning: {col_name} not found in signals file")
            continue
        
        df_ctf = signals_df[[TIME_COL, col_name]].copy()
        df_ctf = df_ctf.rename(columns={col_name: GLUCOSE_COL})
        agp_counterfactuals[delta] = compute_agp_profile(df_ctf, timestep)
    
    if not agp_counterfactuals:
        print(f"    [!] No counterfactual signals found")
        return False
    
    # Create output directory
    output_dir = os.path.join(SENSITIVITY_LININTERP_FIGURES, patient_id)
    os.makedirs(output_dir, exist_ok=True)
        
    # Figure 1: Medians overlay
    fig_overlay = generate_medians_overlay_figure(
        patient_id, agp_original, agp_counterfactuals
    )
    output_path_overlay = os.path.join(output_dir, f"{patient_id}_agp_medians_overlay")
    save_figure(fig_overlay, output_path_overlay)
    
    # Figure 2: Subplots
    fig_subplots = generate_subplots_figure(
        patient_id, agp_original, agp_counterfactuals
    )
    output_path_subplots = os.path.join(output_dir, f"{patient_id}_agp_subplots")
    save_figure(fig_subplots, output_path_subplots)
    
    return True


def main():
    """Main execution pipeline."""
    parser = argparse.ArgumentParser(
        description="AGP visualization for sensitivity analysis results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--patient_id", required=True, help="Patient ID (e.g., A168)")
    
    args = parser.parse_args()
    
    kernel_str = f"k={KERNEL_MIN} min" if APPLY_KERNEL else "no kernel"
    
    print("=" * 70)
    print("  AGP Visualization for Sensitivity Analysis")
    print("=" * 70)
    print(f"  Patient:  {args.patient_id}")
    print(f"  Kernel:   {kernel_str}")
    print(f"  Input:    {SENSITIVITY_LININTERP_DATA}")
    print(f"  Output:   {SENSITIVITY_LININTERP_FIGURES}")
    print("=" * 70)
    
    success = process_patient(args.patient_id)
    
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