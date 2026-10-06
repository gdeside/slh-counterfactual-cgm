"""
scripts/counterfactual/visualize_sensitivity_lines.py
=====================================================
Sensitivity Line Plots: Delta Metric vs Zone-Expansion Offset
=============================================================

Generates multi-panel line plots showing how metric changes (Δ) vary with
zone-expansion offset (δ). Each subplot shows one metric with both:
- Full 24-hour metrics (circles)
- Daytime metrics 06:00-22:00 (triangles)

Input:  Results/Sensitivity_LinInterp_Modular/Signals/{patient_id}_signals.csv
        Results/Sensitivity_LinInterp_Modular/Data/{patient_id}_sensitivity.csv
Output: Results/Sensitivity_LinInterp_Modular/Figures/{patient_id}/
            {patient_id}_sensitivity_lines.pdf/.png

Usage:
    python3 visualize_sensitivity_lines.py --patient_id ID

Authors: Guillaume Deside

Usage
-----
    python3 -m scripts.counterfactual.visualize_sensitivity_lines [--help for options]
    (or: make lines_sensitivity)
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.utils.patient_selection import resolve_patient_id
from src.config import (
    TIME_COL,
    THRESHOLD_HYPO,
    THRESHOLD_HYPER,
    SENSITIVITY_LININTERP_SIGNALS,
    SENSITIVITY_LININTERP_FIGURES,
    FIGURE_WIDTH_INCHES,
)
from src.counterfactual.load_counterfactual import (
    load_patient_signals,
    load_sensitivity_results,
    extract_delta_values,
)
from src.utils.time_utils import is_daytime_mask
from src.visualization.style import apply_publication_style, save_figure, style_axes
from src.visualization.colors import OFFSET_COLORS, COLOR_ORIGINAL, COLOR_ZERO_LINE
from src.visualization.glucose_axis import apply_glucose_axis
from src.utils.unit_conversion import glucose_unit_label, is_glucose_valued_column

# =============================================================================
# CONFIGURATION
# =============================================================================

# ── Patient ID (can be changed here or via --patient_id) ────────────────────
DEFAULT_PATIENT_ID = None   # None = automatic: patient with the most SLH episodes

# ── Paths ────────────────────────────────────────────────────────────────────
SIGNALS_DIR = str(SENSITIVITY_LININTERP_SIGNALS)
FIGURES_OUTPUT_DIR = str(SENSITIVITY_LININTERP_FIGURES)


# ── Metrics to plot ──────────────────────────────────────────────────────────
METRICS = [
    "TIR70-180",
    "TAR>180", 
    "TAR>250",
    "mean",
    "median",
    "sd",
    "cv",
]

# Metric display properties
METRIC_LABELS = {
    "TIR70-180": "TIR 70-180 (%)",
    "TAR>180": "TAR >180 (%)",
    "TAR>250": "TAR >250 (%)",
    "mean": "Mean (mg/dL)",
    "median": "Median (mg/dL)",
    "sd": "SD (mg/dL)",
    "cv": "CV (%)",
}


def metric_label(metric: str) -> str:
    """METRIC_LABELS entry, with the glucose unit taken from the display unit."""
    return METRIC_LABELS[metric].replace("(mg/dL)", f"({glucose_unit_label()})")


# Beneficial direction (+1 = increase good, -1 = decrease good)
METRIC_DIRECTION = {
    "TIR70-180": +1,
    "TAR>180": -1,
    "TAR>250": -1,
    "mean": -1,
    "median": -1,
    "sd": -1,
    "cv": -1,
}

# ── Figure layout ────────────────────────────────────────────────────────────
SUBPLOT_NCOLS = 4            # Number of columns
SUBPLOT_HEIGHT_PER_ROW = 2.1 # Height per row

# ── Line and marker styling ──────────────────────────────────────────────────
LINE_COLOR = COLOR_ORIGINAL  # Dark grey for connecting line
LINE_WIDTH = 0.9
LINE_ALPHA = 0.6

# Marker styles
MARKER_SIZE_FULLTIME = 35    # Circle size
MARKER_SIZE_DAYTIME = 40     # Triangle size
MARKER_ALPHA = 0.85

# Full-time markers: circles
MARKER_FULLTIME = "o"
MARKER_EDGE_WIDTH_FULLTIME = 0.8

# Daytime markers: triangles
MARKER_DAYTIME = "^"
MARKER_EDGE_WIDTH_DAYTIME = 0.8


# ── Reference lines ──────────────────────────────────────────────────────────
ZERO_LINE_COLOR = COLOR_ZERO_LINE
ZERO_LINE_STYLE = "--"
ZERO_LINE_WIDTH = 0.7


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_metrics(glucose: np.ndarray, time_series: pd.Series = None,
                   daytime_only: bool = False) -> dict:
    """Compute CGM metrics."""
    # Apply daytime filter if requested
    if daytime_only and time_series is not None:
        daytime_mask = is_daytime_mask(time_series)
        glucose = glucose[daytime_mask]
    
    # Filter out NaN
    g = glucose[~np.isnan(glucose)]
    if len(g) < 10:
        return {k: np.nan for k in METRICS}
    
    n = len(g)
    mean_val = float(np.mean(g))
    
    return {
        "mean": round(mean_val, 2),
        "median": round(float(np.median(g)), 2),
        "sd": round(float(np.std(g, ddof=1)), 2),
        "cv": round(100 * float(np.std(g, ddof=1) / mean_val) if mean_val > 0 else np.nan, 4),
        "TIR70-180": round(100 * ((g >= THRESHOLD_HYPO) & (g <= THRESHOLD_HYPER)).sum() / n, 4),
        "TAR>180": round(100 * (g > THRESHOLD_HYPER).sum() / n, 4),
        "TAR>250": round(100 * (g > 250).sum() / n, 4),
    }


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def compute_all_metrics(signals_df: pd.DataFrame, delta_values: list) -> dict:
    """
    Compute original and counterfactual metrics for all deltas.
    
    Returns
    -------
    dict with keys:
        - 'orig_fulltime': original metrics (24h)
        - 'orig_daytime': original metrics (06:00-22:00)
        - 'ctf_fulltime': dict {delta: metrics} (24h)
        - 'ctf_daytime': dict {delta: metrics} (06:00-22:00)
    """
    time_series = signals_df[TIME_COL]
    
    # Original signal metrics
    orig_glucose = signals_df["glucose_original"].values
    orig_fulltime = compute_metrics(orig_glucose, time_series, daytime_only=False)
    orig_daytime = compute_metrics(orig_glucose, time_series, daytime_only=True)
    
    # Counterfactual metrics for each delta
    ctf_fulltime = {}
    ctf_daytime = {}
    
    for delta in delta_values:
        col_name = f"glucose_ctf_delta_{delta}"
        
        if col_name not in signals_df.columns:
            continue
        
        ctf_glucose = signals_df[col_name].values
        ctf_fulltime[delta] = compute_metrics(ctf_glucose, time_series, daytime_only=False)
        ctf_daytime[delta] = compute_metrics(ctf_glucose, time_series, daytime_only=True)
    
    return {
        'orig_fulltime': orig_fulltime,
        'orig_daytime': orig_daytime,
        'ctf_fulltime': ctf_fulltime,
        'ctf_daytime': ctf_daytime,
    }


def compute_delta_metrics(metrics_dict: dict, delta_values: list) -> dict:
    """
    Compute delta (change) for each metric and offset.
    
    Returns
    -------
    dict with keys:
        - 'fulltime': dict {metric: [deltas]}
        - 'daytime': dict {metric: [deltas]}
    """
    orig_ft = metrics_dict['orig_fulltime']
    orig_dt = metrics_dict['orig_daytime']
    ctf_ft = metrics_dict['ctf_fulltime']
    ctf_dt = metrics_dict['ctf_daytime']
    
    delta_fulltime = {m: [] for m in METRICS}
    delta_daytime = {m: [] for m in METRICS}
    
    for delta in delta_values:
        for metric in METRICS:
            # Full-time delta
            orig_val_ft = orig_ft.get(metric, np.nan)
            ctf_val_ft = ctf_ft.get(delta, {}).get(metric, np.nan)
            
            if np.isfinite(orig_val_ft) and np.isfinite(ctf_val_ft):
                delta_fulltime[metric].append(ctf_val_ft - orig_val_ft)
            else:
                delta_fulltime[metric].append(np.nan)
            
            # Daytime delta
            orig_val_dt = orig_dt.get(metric, np.nan)
            ctf_val_dt = ctf_dt.get(delta, {}).get(metric, np.nan)
            
            if np.isfinite(orig_val_dt) and np.isfinite(ctf_val_dt):
                delta_daytime[metric].append(ctf_val_dt - orig_val_dt)
            else:
                delta_daytime[metric].append(np.nan)
    
    return {
        'fulltime': delta_fulltime,
        'daytime': delta_daytime,
    }


# =============================================================================
# PLOT HELPERS
# =============================================================================

def format_subplot_axis(ax: plt.Axes, metric: str, delta_values: list):
    """Apply standard formatting to subplot axis."""
    # Zero reference line
    ax.axhline(0, color=ZERO_LINE_COLOR, linestyle=ZERO_LINE_STYLE,
              linewidth=ZERO_LINE_WIDTH, zorder=1)

    # Title
    ax.set_title(metric_label(metric), fontsize=7, fontweight="bold",
                loc="left", pad=3)
    
    # Axes labels
    ax.set_xlabel("δ (min)", fontsize=7, labelpad=2)
    ax.set_ylabel("Δ", fontsize=7, labelpad=2)
    
    # Glucose metrics: Δ values are mg/dL, so relabel the y ticks
    if is_glucose_valued_column(metric):
        apply_glucose_axis(ax, set_label=False)

    # X-axis ticks
    ax.set_xticks(delta_values)
    ax.set_xticklabels([str(d) for d in delta_values], fontsize=6.5)
    
    # Styling
    style_axes(ax)
    ax.tick_params(length=3, width=0.8, labelsize=6.5)
    ax.grid(True, alpha=0.3, linewidth=0.5, zorder=0)


# =============================================================================
# FIGURE GENERATION
# =============================================================================
def generate_sensitivity_lines_figure(patient_id: str, delta_values: list,
                                     delta_metrics: dict) -> plt.Figure:
    """
    Generate multi-panel sensitivity line plot.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier
    delta_values : list
        List of offset values
    delta_metrics : dict
        Delta metrics from compute_delta_metrics()
        
    Returns
    -------
    plt.Figure
    """
    apply_publication_style()
    
    n_metrics = len(METRICS)
    nrows = int(np.ceil(n_metrics / SUBPLOT_NCOLS))
    
    fig_height = nrows * SUBPLOT_HEIGHT_PER_ROW
    fig, axes = plt.subplots(
        nrows, SUBPLOT_NCOLS,
        figsize=(FIGURE_WIDTH_INCHES, fig_height)
    )
    fig.subplots_adjust(left=0.09, right=0.97, top=0.94, bottom=0.08,
                       hspace=0.40, wspace=0.35)
    
    # Title
    # fig.suptitle(
    #     f"Participant {patient_id}",
    #     fontsize=8, fontweight="bold"
    # )
    
    # Flatten axes
    axes_flat = np.array(axes).reshape(-1) if n_metrics > 1 else [axes]
    
    # Plot each metric
    for idx, metric in enumerate(METRICS):
        ax = axes_flat[idx]
        
        # Get delta values for this metric
        delta_ft = np.array(delta_metrics['fulltime'][metric])
        delta_dt = np.array(delta_metrics['daytime'][metric])
        
        # Plot connecting line (full-time)
        ax.plot(
            delta_values, delta_ft,
            color=LINE_COLOR, lw=LINE_WIDTH, alpha=LINE_ALPHA,
            zorder=2
        )

        ax.plot(
            delta_values, delta_dt,
            color=LINE_COLOR, lw=LINE_WIDTH, alpha=LINE_ALPHA,
            zorder=2
        )
        
        # Plot markers for each offset
        for i, delta in enumerate(delta_values):
            color = OFFSET_COLORS.get(delta, "#999999")
            
            # Full-time (circle)
            if np.isfinite(delta_ft[i]):
                ax.scatter(
                    [delta], [delta_ft[i]],
                    s=MARKER_SIZE_FULLTIME,
                    marker=MARKER_FULLTIME,
                    c=color,
                    edgecolors=color,
                    linewidths=MARKER_EDGE_WIDTH_FULLTIME,
                    alpha=MARKER_ALPHA,
                    zorder=5,
                    label="Full 24h" if i == 0 else None
                )
            
            # Daytime (triangle)
            if np.isfinite(delta_dt[i]):
                ax.scatter(
                    [delta], [delta_dt[i]],
                    s=MARKER_SIZE_DAYTIME,
                    marker=MARKER_DAYTIME,
                    c=color,
                    edgecolors=color,
                    linewidths=MARKER_EDGE_WIDTH_DAYTIME,
                    alpha=MARKER_ALPHA,
                    zorder=6,
                    label="Daytime" if i == 0 else None
                )
        
        # Format axis
        format_subplot_axis(ax, metric, delta_values)
        
        # Legend (only on first subplot)
        if idx == 0:
            legend_elements = [
                Line2D([0], [0], marker=MARKER_FULLTIME, color='w',
                      markerfacecolor='#666666', markersize=6,
                      markeredgewidth=1, label='Full 24h'),
                Line2D([0], [0], marker=MARKER_DAYTIME, color='w',
                      markerfacecolor='#666666', markersize=7,
                      markeredgewidth=1, label='Daytime (06:00-22:00)'),
            ]
            
            ax.legend(handles=legend_elements, fontsize=6, loc='best',
                     framealpha=0.95)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def process_patient(patient_id: str):
    """Generate sensitivity lines figure for a patient."""
    print(f"\n  Processing {patient_id}...")
    
    # Load signals
    signals_df = load_patient_signals(patient_id)
    if signals_df is None:
        print(f"    [!] No signal data found")
        return False
    
    # Load sensitivity results to verify data exists
    sens_df = load_sensitivity_results(patient_id)
    if sens_df is None:
        print(f"    [!] No sensitivity results found")
        return False
    
    # Extract delta values
    delta_values = extract_delta_values(signals_df)
    if not delta_values:
        print(f"    [!] No counterfactual signals found")
        return False
    
    print(f"    Delta values: {delta_values}")
    
    metrics_dict = compute_all_metrics(signals_df, delta_values)
    delta_metrics = compute_delta_metrics(metrics_dict, delta_values)
    
    fig = generate_sensitivity_lines_figure(patient_id, delta_values, delta_metrics)
    
    # Save
    output_dir = os.path.join(FIGURES_OUTPUT_DIR, patient_id)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{patient_id}_sensitivity_lines")
    save_figure(fig, output_path)
    
    return True


def main():
    """Main execution pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate sensitivity line plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--patient_id",
        default=DEFAULT_PATIENT_ID,
        help="Patient ID"
    )
    
    args = parser.parse_args()
    
    args.patient_id = resolve_patient_id(args.patient_id)
    
    print("=" * 70)
    print("  Sensitivity Line Plots: Δ Metric vs Zone-Expansion δ")
    print("=" * 70)
    print(f"  Patient: {args.patient_id}")
    print(f"  Signals: {SIGNALS_DIR}")
    print(f"  Output:  {FIGURES_OUTPUT_DIR}")
    print("=" * 70)
    
    success = process_patient(args.patient_id)
    
    if success:
        print("\n" + "=" * 70)
        print("  Done!")
        print("=" * 70 + "\n")
    else:
        print("\n" + "=" * 70)
        print("  Processing failed - see messages above")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
