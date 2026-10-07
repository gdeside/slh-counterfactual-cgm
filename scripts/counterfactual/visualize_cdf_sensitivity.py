"""
scripts/counterfactual/visualize_cdf_sensitivity.py
===================================================
Cohort Sensitivity Cumulative Distribution Visualization
=========================================================

Generates cumulative distribution function (CDF) plots showing the distribution 
of metric changes (Δ) across the cohort for different zone-expansion offset values (δ).

Features:
- Individual patient markers (2 representative cases)
- Quartile boundary lines
- Separate figures per metric (full-time and daytime)
- Side-by-side comparison figures

Input:  Results/Sensitivity_Analysis/Data/cohort_sensitivity.csv
        Results/Quartile_Analysis/quartile_assignments.csv
Output: Results/Sensitivity_Analysis/Figures/Cohort/cdf/

Usage:
    python3 visualize_cdf_sensitivity.py
    python3 -m scripts.counterfactual.visualize_cdf_sensitivity --metric TIR70-180 --patients ID1 ID2

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

# Import configuration
from src.utils.patient_selection import resolve_patient_pair
from src.config import BASE_PATH

# Import visualization
from src.visualization.style import apply_publication_style, save_figure, style_axes
from src.visualization.colors import OFFSET_COLORS
from src.visualization.glucose_axis import apply_glucose_axis
from src.utils.unit_conversion import glucose_unit_label, is_glucose_valued_column

# =============================================================================
# CONFIGURATION
# =============================================================================

# ── Selected offset ──────────────────────────────────────────────────────────
SELECTED_OFFSET = 30  # Zone-expansion offset to plot (minutes)

# ── Paths ────────────────────────────────────────────────────────────────────
SENSITIVITY_RESULTS_DIR = os.path.join(BASE_PATH, "Results", "Sensitivity_Analysis")
DATA_DIR = os.path.join(SENSITIVITY_RESULTS_DIR, "Data")
FIGURES_OUTPUT_DIR = os.path.join(SENSITIVITY_RESULTS_DIR, "Figures", "Cohort", "cdf")

COHORT_CSV = os.path.join(DATA_DIR, "cohort_sensitivity.csv")
QUARTILE_CSV = os.path.join(BASE_PATH, "Results", "Quartile_Analysis", "quartile_assignments.csv")

# ── Default representative patients ──────────────────────────────────────────
DEFAULT_PATIENTS = None  # None = automatic: [high-SLH, low-SLH] patients (CGM_PATIENT_HIGH / CGM_PATIENT_LOW)

# ── Metrics to plot ──────────────────────────────────────────────────────────
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
    "TIR70-180": "ΔTIR 70-180",
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


# ── Figure parameters ────────────────────────────────────────────────────────
FIGURE_WIDTH_SINGLE = 5.0    # Width for single panel
FIGURE_WIDTH_DOUBLE = 7.24   # Width for double panel (side-by-side)
FIGURE_HEIGHT = 3.8          # Height for CDF figures

# ── CDF styling ──────────────────────────────────────────────────────────────
CDF_LINE_WIDTH = 1.8
CDF_LINE_ALPHA = 0.85

# ── Patient markers ──────────────────────────────────────────────────────────
PATIENT_MARKER_SIZE = 80
PATIENT_MARKER_ALPHA = 0.90
PATIENT_MARKER_EDGE_WIDTH = 1.5
PATIENT_MARKER_EDGE_COLOR = "#000000"
PATIENT_COLOR_CYCLE = ["#E69F00", "#CC79A7", "#009E73", "#56B4E9"]  # orange, purple, …
PATIENT_COLORS: dict = {}   # filled from the selected patients in main()

# ── Quartile lines ───────────────────────────────────────────────────────────
QUARTILE_LINE_COLOR = "#666666"
QUARTILE_LINE_STYLE = "--"
QUARTILE_LINE_WIDTH = 0.9
QUARTILE_LINE_ALPHA = 0.5
QUARTILE_LABEL_SIZE = 6.5

# ── Reference lines ──────────────────────────────────────────────────────────
ZERO_LINE_COLOR = "#888888"
ZERO_LINE_STYLE = ":"
ZERO_LINE_WIDTH = 1.0
ZERO_LINE_ALPHA = 0.6

# Horizontal reference at 50th percentile
MEDIAN_LINE_COLOR = "#888888"
MEDIAN_LINE_STYLE = "--"
MEDIAN_LINE_WIDTH = 0.8
MEDIAN_LINE_ALPHA = 0.5


# =============================================================================
# DATA LOADING
# =============================================================================

def load_cohort_data() -> pd.DataFrame:
    """Load cohort sensitivity results."""
    if not os.path.exists(COHORT_CSV):
        raise FileNotFoundError(f"Cohort CSV not found: {COHORT_CSV}")
    
    df = pd.read_csv(COHORT_CSV)
    return df


def load_quartile_data() -> pd.DataFrame:
    """Load quartile assignments."""
    if not os.path.exists(QUARTILE_CSV):
        print(f"  [!] Quartile CSV not found: {QUARTILE_CSV}")
        print(f"      Quartile lines will not be shown.")
        return None
    
    df = pd.read_csv(QUARTILE_CSV)
    return df


def merge_quartile_data(cohort_df: pd.DataFrame, 
                        quartile_df: pd.DataFrame) -> pd.DataFrame:
    """Merge cohort data with quartile assignments."""
    if quartile_df is None:
        return cohort_df
    
    merged = cohort_df.merge(
        quartile_df[["patient_id", "quartile", "n_SLH"]],
        on="patient_id",
        how="left"
    )
    
    return merged


# =============================================================================
# CDF COMPUTATION
# =============================================================================

def compute_cdf(values: np.ndarray) -> tuple:
    """
    Compute cumulative distribution function.
    
    Parameters
    ----------
    values : np.ndarray
        Data values
        
    Returns
    -------
    (sorted_values, cumulative_prob)
    """
    sorted_values = np.sort(values)
    cumulative_prob = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    
    return sorted_values, cumulative_prob


def get_patient_position(patient_id: str, df: pd.DataFrame, 
                        metric: str, offset: int,
                        daytime_only: bool = False) -> tuple:
    """
    Get patient's delta value and CDF position.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier
    df : pd.DataFrame
        Cohort data
    metric : str
        Metric name
    offset : int
        Zone-expansion offset
    daytime_only : bool
        Use daytime metrics
        
    Returns
    -------
    (delta_value, cdf_position) or (None, None) if patient not found
    """
    prefix = "daytime_delta_" if daytime_only else "delta_"
    col_name = f"{prefix}{metric}"
    
    # Filter for patient and offset
    mask = (df["patient_id"] == patient_id) & (df["offset"] == offset)
    patient_data = df.loc[mask, col_name]
    
    if len(patient_data) == 0:
        return None, None
    
    delta_value = patient_data.iloc[0]
    
    if pd.isna(delta_value):
        return None, None
    
    # Get all values for this offset to compute CDF position
    all_values = df.loc[df["offset"] == offset, col_name].dropna().values
    cdf_position = (all_values < delta_value).sum() / len(all_values)
    
    return delta_value, cdf_position


def get_quartile_boundaries(df: pd.DataFrame, metric: str, offset: int,
                           daytime_only: bool = False) -> dict:
    """
    Get delta values at quartile boundaries (25th, 50th, 75th percentiles).
    
    Parameters
    ----------
    df : pd.DataFrame
        Cohort data with quartile assignments
    metric : str
        Metric name
    offset : int
        Zone-expansion offset
    daytime_only : bool
        Use daytime metrics
        
    Returns
    -------
    dict with keys "Q1", "Q2", "Q3" and values (delta_value, cdf_position)
    """
    if "quartile" not in df.columns:
        return {}
    
    prefix = "daytime_delta_" if daytime_only else "delta_"
    col_name = f"{prefix}{metric}"
    
    # Filter for offset
    df_offset = df[df["offset"] == offset].copy()
    all_values = df_offset[col_name].dropna().values
    
    if len(all_values) == 0:
        return {}
    
    # Compute percentiles
    q25 = np.percentile(all_values, 25)
    q50 = np.percentile(all_values, 50)
    q75 = np.percentile(all_values, 75)
    
    return {
        "Q1": (q25, 0.25),
        "Q2": (q50, 0.50),
        "Q3": (q75, 0.75),
    }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_cdf_single(ax: plt.Axes, df: pd.DataFrame, metric: str,
                   patient_ids: list, offset: int, daytime_only: bool = False):
    """
    Plot CDF for selected offset on a single axis.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis
    df : pd.DataFrame
        Cohort data with quartile info
    metric : str
        Metric name
    patient_ids : list
        List of patient IDs to mark
    offset : int
        Zone-expansion offset (minutes)
    daytime_only : bool
        Use daytime metrics
    """
    prefix = "daytime_delta_" if daytime_only else "delta_"
    col_name = f"{prefix}{metric}"
    
    # Get data for selected offset only
    df_offset = df[df["offset"] == offset]
    values = df_offset[col_name].dropna().values
    
    if len(values) < 2:
        return
    
    # Compute CDF
    sorted_vals, cdf = compute_cdf(values)
    
    # Plot CDF line
    color = OFFSET_COLORS.get(offset, "#0072B2")
    ax.plot(sorted_vals, cdf, color=color, lw=CDF_LINE_WIDTH,
           alpha=CDF_LINE_ALPHA, label=f"δ = {offset} min", zorder=3)
    
    # Add quartile boundaries
    quartile_bounds = get_quartile_boundaries(df, metric, offset, daytime_only)
    
    for q_label, (q_val, q_pos) in quartile_bounds.items():
        ax.axvline(q_val, color=QUARTILE_LINE_COLOR, 
                  linestyle=QUARTILE_LINE_STYLE,
                  linewidth=QUARTILE_LINE_WIDTH, 
                  alpha=QUARTILE_LINE_ALPHA, zorder=1)
        
        # Add label
        y_label = 0.95 if q_label == "Q2" else 0.88
        ax.text(q_val, y_label, q_label, 
               fontsize=QUARTILE_LABEL_SIZE, color=QUARTILE_LINE_COLOR,
               ha='center', va='top', transform=ax.get_xaxis_transform(),
               bbox=dict(boxstyle="round,pad=0.2", 
                       facecolor="white", edgecolor=QUARTILE_LINE_COLOR,
                       linewidth=0.5, alpha=0.8))
    
    # Add patient markers
    if patient_ids:
        for patient_id in patient_ids:
            delta_val, cdf_pos = get_patient_position(
                patient_id, df, metric, offset, daytime_only
            )
            
            if delta_val is not None:
                color_patient = PATIENT_COLORS.get(patient_id, "#999999")
                ax.scatter([delta_val], [cdf_pos], s=PATIENT_MARKER_SIZE,
                          c=color_patient, alpha=PATIENT_MARKER_ALPHA,
                          edgecolors=PATIENT_MARKER_EDGE_COLOR,
                          linewidths=PATIENT_MARKER_EDGE_WIDTH,
                          marker='D', zorder=10, label=patient_id)
    
    # Zero reference line
    ax.axvline(0, color=ZERO_LINE_COLOR, linestyle=ZERO_LINE_STYLE,
              linewidth=ZERO_LINE_WIDTH, alpha=ZERO_LINE_ALPHA, zorder=2)
    
    # Median horizontal line
    ax.axhline(0.5, color=MEDIAN_LINE_COLOR, linestyle=MEDIAN_LINE_STYLE,
              linewidth=MEDIAN_LINE_WIDTH, alpha=MEDIAN_LINE_ALPHA, zorder=1)


def format_cdf_axis(ax: plt.Axes, metric: str, n_patients: int,
                   daytime_only: bool = False, show_legend: bool = True):
    """Apply standard formatting to CDF axis."""
    # X-axis label
    xlabel = f"{METRIC_LABELS[metric]} {metric_unit(metric)}"
    ax.set_xlabel(xlabel, fontsize=8, labelpad=3)

    # Glucose metrics: Δ values stay in mg/dL, only the ticks are relabelled
    if is_glucose_valued_column(metric):
        apply_glucose_axis(ax, axis="x", set_label=False)
    
    # Y-axis label
    ax.set_ylabel("Cumulative probability", fontsize=8, labelpad=3)
    
    # Y-axis limits
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(['0', '0.25', '0.50', '0.75', '1.0'])
    
    # Grid
    ax.grid(True, alpha=0.25, linewidth=0.5, zorder=0)
    
    # Style
    style_axes(ax)
    ax.tick_params(length=3, width=0.8, labelsize=7)
    
    # Legend
    if show_legend:
        ax.legend(fontsize=6.5, loc='best', framealpha=0.95, ncol=1)


# =============================================================================
# FIGURE TYPE 1: FULL-TIME ONLY
# =============================================================================

def generate_fulltime_cdf(df: pd.DataFrame, metric: str,
                         patient_ids: list, offset: int) -> plt.Figure:
    """
    Generate CDF for full-time (24h) metrics only.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cohort data
    metric : str
        Metric name
    patient_ids : list
        Patient IDs to mark
    offset : int
        Zone-expansion offset (minutes)
        
    Returns
    -------
    plt.Figure
    """
    apply_publication_style()
    
    # Count patients
    n_patients = df["patient_id"].nunique()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_SINGLE, FIGURE_HEIGHT))
    fig.subplots_adjust(left=0.13, right=0.96, top=0.88, bottom=0.14)
    
    # Title
    time_label = "Full 24-hour metrics"
    fig.suptitle(
        f"Cohort {METRIC_LABELS[metric]} cumulative distribution  (n = {n_patients})\n"
        f"{time_label}  ·  δ = {offset} min",
        fontsize=8, fontweight="bold"
    )
    
    # Plot CDF
    plot_cdf_single(ax, df, metric, patient_ids, offset, daytime_only=False)
    
    # Format axis
    format_cdf_axis(ax, metric, n_patients, daytime_only=False)
    
    return fig


# =============================================================================
# FIGURE TYPE 2: DAYTIME ONLY
# =============================================================================

def generate_daytime_cdf(df: pd.DataFrame, metric: str,
                        patient_ids: list, offset: int) -> plt.Figure:
    """
    Generate CDF for daytime (06:00-22:00) metrics only.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cohort data
    metric : str
        Metric name
    patient_ids : list
        Patient IDs to mark
    offset : int
        Zone-expansion offset (minutes)
        
    Returns
    -------
    plt.Figure
    """
    apply_publication_style()
    
    # Count patients
    n_patients = df["patient_id"].nunique()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_SINGLE, FIGURE_HEIGHT))
    fig.subplots_adjust(left=0.13, right=0.96, top=0.88, bottom=0.14)
    
    # Title
    time_label = "Daytime metrics (06:00-22:00)"
    fig.suptitle(
        f"Cohort {METRIC_LABELS[metric]} cumulative distribution  (n = {n_patients})\n"
        f"{time_label}  ·  δ = {offset} min",
        fontsize=8, fontweight="bold"
    )
    
    # Plot CDF
    plot_cdf_single(ax, df, metric, patient_ids, offset, daytime_only=True)
    
    # Format axis
    format_cdf_axis(ax, metric, n_patients, daytime_only=True)
    
    return fig


# =============================================================================
# FIGURE TYPE 3: SIDE-BY-SIDE COMPARISON
# =============================================================================

def generate_comparison_cdf(df: pd.DataFrame, metric: str,
                           patient_ids: list, offset: int) -> plt.Figure:
    """
    Generate side-by-side comparison: full-time vs daytime metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cohort data
    metric : str
        Metric name
    patient_ids : list
        Patient IDs to mark
    offset : int
        Zone-expansion offset (minutes)
        
    Returns
    -------
    plt.Figure
    """
    apply_publication_style()
    
    # Count patients
    n_patients = df["patient_id"].nunique()
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_DOUBLE, FIGURE_HEIGHT))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.14, wspace=0.28)
    
    # Title
    fig.suptitle(
        f"Cohort {METRIC_LABELS[metric]} cumulative distribution  (n = {n_patients})  ·  δ = {offset} min",
        fontsize=8, fontweight="bold"
    )
    
    # ── Panel A: Full-time ───────────────────────────────────────────────────
    plot_cdf_single(ax1, df, metric, patient_ids, offset, daytime_only=False)
    format_cdf_axis(ax1, metric, n_patients, daytime_only=False, show_legend=True)
    
    # Panel label
    ax1.text(
        0.02, 0.98, "(A) Full 24-hour",
        transform=ax1.transAxes, fontsize=7, fontweight="bold",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                 edgecolor="#666666", linewidth=0.5, alpha=0.9)
    )
    
    # ── Panel B: Daytime ─────────────────────────────────────────────────────
    plot_cdf_single(ax2, df, metric, patient_ids, offset, daytime_only=True)
    format_cdf_axis(ax2, metric, n_patients, daytime_only=True, show_legend=False)
    
    # Panel label
    ax2.text(
        0.02, 0.98, "(B) Daytime (06:00-22:00)",
        transform=ax2.transAxes, fontsize=7, fontweight="bold",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                 edgecolor="#666666", linewidth=0.5, alpha=0.9)
    )
    
    return fig


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_all_metrics(df: pd.DataFrame, patient_ids: list,
                       offset: int, metric_filter: str = None):
    """
    Generate all CDF figures for all metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cohort data
    patient_ids : list
        Patient IDs to mark
    offset : int
        Zone-expansion offset (minutes)
    metric_filter : str, optional
        If provided, only process this metric
    """
    # Filter data for selected offset only
    df_offset = df[df["offset"] == offset].copy()
    
    if len(df_offset) == 0:
        print(f"\n  [ERROR] No data found for offset δ = {offset} min")
        return
    
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
    
    print(f"\n  Processing {len(metrics_to_process)} metric(s) for δ = {offset} min...")
    if patient_ids:
        print(f"  Marking patients: {', '.join(patient_ids)}")
    
    for metric in metrics_to_process:
        print(f"\n  {metric}:")
        
        try:
            # Generate full-time CDF
            fig_fulltime = generate_fulltime_cdf(df_offset, metric, patient_ids, offset)
            path_fulltime = os.path.join(FIGURES_OUTPUT_DIR, 
                                        f"{metric}_fulltime_cdf_offset{offset}")
            save_figure(fig_fulltime, path_fulltime)
            print(f"    ✓ Full-time CDF")
            
            # Generate daytime CDF
            fig_daytime = generate_daytime_cdf(df_offset, metric, patient_ids, offset)
            path_daytime = os.path.join(FIGURES_OUTPUT_DIR, 
                                       f"{metric}_daytime_cdf_offset{offset}")
            save_figure(fig_daytime, path_daytime)
            print(f"    ✓ Daytime CDF")
            
            # Generate comparison CDF
            fig_comparison = generate_comparison_cdf(df_offset, metric, patient_ids, offset)
            path_comparison = os.path.join(FIGURES_OUTPUT_DIR, 
                                          f"{metric}_comparison_cdf_offset{offset}")
            save_figure(fig_comparison, path_comparison)
            print(f"    ✓ Comparison CDF")
            
        except Exception as e:
            print(f"    [!] Error: {e}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate cohort sensitivity cumulative distribution plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Process specific metric only (default: all metrics)"
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=SELECTED_OFFSET,
        help="zone-expansion offset δ (minutes) shown on the CDFs (default: %(default)g; must exist in the sensitivity data)"
    )
    parser.add_argument(
        "--patients",
        nargs='+',
        default=DEFAULT_PATIENTS,
        help="Patient IDs to mark on CDF (default: high- and low-SLH patients, chosen automatically)"
    )
    
    args = parser.parse_args()
    selected_offset = int(args.offset) if float(args.offset).is_integer() else args.offset
    if not args.patients:
        args.patients = list(resolve_patient_pair())
    PATIENT_COLORS.update(zip(args.patients, PATIENT_COLOR_CYCLE))
    
    print("=" * 70)
    print("  Cohort Sensitivity CDF Visualization")
    print("=" * 70)
    print(f"  Input:     {COHORT_CSV}")
    print(f"  Quartiles: {QUARTILE_CSV}")
    print(f"  Output:    {FIGURES_OUTPUT_DIR}")
    print(f"  Selected δ: {selected_offset} min")
    if args.metric:
        print(f"  Metric:    {args.metric} (single)")
    else:
        print(f"  Metrics:   All ({len(METRICS)} total)")
    print(f"  Patients:  {args.patients}")
    print("=" * 70)
    
    # Load data
    try:
        df = load_cohort_data()
        n_patients = df["patient_id"].nunique()
        n_offsets = df["offset"].nunique()
        print(f"\n  Loaded cohort data:")
        print(f"    Patients: {n_patients}")
        print(f"    Offsets:  {sorted(df['offset'].unique())}")
        print(f"    Rows:     {len(df)}")
        
        # Check if selected offset exists
        if selected_offset not in df["offset"].values:
            print(f"\n  [ERROR] Selected offset δ = {selected_offset} min not found in data")
            print(f"  Available offsets: {sorted(df['offset'].unique())}")
            return
            
    except FileNotFoundError as e:
        print(f"\n  [ERROR] {e}")
        print(f"\n  Run sensitivity analysis first:")
        print(f"    python3 sensitivity_lininterp_modular.py --patient_id ALL")
        return
    
    # Load quartile data
    quartile_df = load_quartile_data()
    if quartile_df is not None:
        df = merge_quartile_data(df, quartile_df)
        print(f"\n  Quartile data merged successfully")
    
    # Validate patient IDs
    valid_patients = []
    for pid in args.patients:
        if pid in df["patient_id"].values:
            valid_patients.append(pid)
        else:
            print(f"  [!] Patient {pid} not found in data - will be skipped")
    
    if not valid_patients:
        print(f"\n  [!] No valid patient IDs provided - continuing without markers")
    
    # Process metrics
    process_all_metrics(df, valid_patients, selected_offset, args.metric)
    
    print("\n" + "=" * 70)
    print(f"  Output: {FIGURES_OUTPUT_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()