"""
scripts/statistics/quartile_analysis.py
=======================================
Quartile Stratification Analysis
=================================

Stratifies the cohort into quartiles based on n_SLH count and generates
boxplot comparisons across quartiles for each metric and offset.

Uses pre-computed results from:
- Episode detection (n_SLH counts per patient)
- Sensitivity analysis (delta metrics across offsets)

Outputs:
- Quartile assignment CSV
- Boxplot figures (one per metric, subplots per offset)
- Statistics CSV for the selected offset

Authors: Guillaume Deside

Usage
-----
    python3 -m scripts.statistics.quartile_analysis
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import configuration
from src.config import (
    FILTER_STATUS_CSV,
    COHORT_SENSITIVITY_CSV,
    QUARTILE_ANALYSIS_PATH,
    QUARTILE_FIGURES_PATH,
    QUARTILE_STATS_PATH,
    QUARTILE_ASSIGNMENTS_CSV,
)
from src.counterfactual.counterfactual_config import SELECTED_OFFSET_MIN

# Import data loading
from src.episodes.load_episodes import load_episode_summary as _load_episode_raw

# Import filtering
from src.filtering.load_filtering import load_filtered_patient_list

# Import visualization
from src.visualization.style import apply_publication_style, save_figure
from src.visualization.colors import QUARTILE_COLORS, COLOR_ZERO_LINE


# =============================================================================
# CONFIGURATION
# =============================================================================
PATIENT_FILTER_CSV = str(FILTER_STATUS_CSV)

OUTPUT_DIR = str(QUARTILE_ANALYSIS_PATH)
QUARTILES_CSV = str(QUARTILE_ASSIGNMENTS_CSV)
FIGURES_DIR = str(QUARTILE_FIGURES_PATH)
STATS_DIR = str(QUARTILE_STATS_PATH)

# ── Analysis parameters ──────────────────────────────────────────────────────
SELECTED_OFFSET = SELECTED_OFFSET_MIN

# Metrics to analyze
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
    "TIR70-180": "Delta TIR 70-180 (%)",
    "TAR>180":   "Delta TAR >180 (%)",
    "TAR>250":   "Delta TAR >250 (%)",
    "mean":      "Delta Mean (mg/dL)",
    "median":    "Delta Median (mg/dL)",
    "sd":        "Delta SD (mg/dL)",
    "cv":        "Delta CV (%)",
    "TBR<70":    "Delta TBR <70 (%)",
}

# ── Figure parameters ────────────────────────────────────────────────────────
FIGURE_WIDTH = 7.24           # Double-column width
SUBPLOT_HEIGHT = 2.8          # Height per offset row

# ── Boxplot styling ──────────────────────────────────────────────────────────
BOX_WIDTH = 0.6
FLIER_SIZE = 3
FLIER_ALPHA = 0.5
MEDIAN_LW = 1.5
BOX_LW = 1.0

# ── Reference lines ──────────────────────────────────────────────────────────
ZERO_LINE_COLOR = COLOR_ZERO_LINE
ZERO_LINE_STYLE = "--"
ZERO_LINE_WIDTH = 0.8


# =============================================================================
# DATA LOADING
# =============================================================================
def load_episode_summary() -> pd.DataFrame:
    """Load episode summary with n_SLH per patient (filtered patients only)."""
    df = _load_episode_raw()

    filtered_pids = load_filtered_patient_list(PATIENT_FILTER_CSV)

    if filtered_pids is not None:
        df = df[df["patient_id"].isin(filtered_pids)]
        print(f"  Loaded {len(df)} filtered patients from episodes summary")
    else:
        print(f"  Loaded {len(df)} patients from episodes summary (no filter applied)")

    return df


def load_sensitivity_results() -> pd.DataFrame:
    """Load cohort sensitivity results (filtered patients only)."""
    path = str(COHORT_SENSITIVITY_CSV)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Sensitivity results not found: {path}\n"
            "Run sensitivity_analysis.py --patient_id ALL first."
        )

    df = pd.read_csv(path)

    filtered_pids = load_filtered_patient_list(PATIENT_FILTER_CSV)

    if filtered_pids is not None:
        df = df[df["patient_id"].isin(filtered_pids)]
        print(f"  Loaded {len(df)} rows from sensitivity results (filtered)")
    else:
        print(f"  Loaded {len(df)} rows from sensitivity results (no filter)")

    return df


# =============================================================================
# QUARTILE COMPUTATION
# =============================================================================

def compute_quartile_assignments(episodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign patients to quartiles based on n_SLH.

    Parameters
    ----------
    episodes_df : pd.DataFrame
        Episode summary with patient_id and n_SLH columns

    Returns
    -------
    pd.DataFrame
        With columns: patient_id, n_SLH, quartile, quartile_label, n_SLH_range
    """
    # Sort by n_SLH
    df = episodes_df.sort_values("n_SLH").reset_index(drop=True)

    # Compute quartile boundaries
    quartile_bounds = [
        np.percentile(df["n_SLH"], 25),
        np.percentile(df["n_SLH"], 50),
        np.percentile(df["n_SLH"], 75),
    ]

    # Assign quartiles
    df["quartile"] = pd.cut(
        df["n_SLH"],
        bins=[-np.inf] + quartile_bounds + [np.inf],
        labels=["Q1", "Q2", "Q3", "Q4"]
    )

    # Create quartile labels with ranges
    quartile_info = []

    for q_label in ["Q1", "Q2", "Q3", "Q4"]:
        df_q = df[df["quartile"] == q_label]
        n_min = int(df_q["n_SLH"].min())
        n_max = int(df_q["n_SLH"].max())

        quartile_info.append({
            "quartile": q_label,
            "n_SLH_min": n_min,
            "n_SLH_max": n_max,
            "n_SLH_range": f"{n_min}-{n_max}",
            "n_patients": len(df_q),
        })

    # Add range labels to main dataframe
    range_map = {row["quartile"]: row["n_SLH_range"]
                 for row in quartile_info}
    df["n_SLH_range"] = df["quartile"].map(range_map)

    # Add readable label
    df["quartile_label"] = df.apply(
        lambda row: f"{row['quartile']}\n(n_SLH={row['n_SLH_range']})",
        axis=1
    )

    print("\n  Quartile distribution:")
    for info in quartile_info:
        print(f"    {info['quartile']}: n_SLH={info['n_SLH_range']}, "
              f"n={info['n_patients']} patients")

    return df[["patient_id", "n_SLH", "n_LLH", "quartile", "quartile_label", "n_SLH_range"]]


def save_quartile_assignments(quartile_df: pd.DataFrame, output_path: str):
    """Save quartile assignments to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    quartile_df.to_csv(output_path, index=False)
    print(f"\n  Quartile assignments saved: {output_path}")


# =============================================================================
# DATA MERGING
# =============================================================================
def merge_data(sensitivity_df: pd.DataFrame,
               quartile_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge sensitivity results with quartile assignments.

    Parameters
    ----------
    sensitivity_df : pd.DataFrame
        Cohort sensitivity results
    quartile_df : pd.DataFrame
        Quartile assignments

    Returns
    -------
    pd.DataFrame
        Merged data
    """
    merged = sensitivity_df.merge(
        quartile_df,
        on="patient_id",
        how="left"
    )

    # Remove patients without quartile assignment (shouldn't happen)
    merged = merged.dropna(subset=["quartile"])

    print(f"\n  Merged data: {len(merged)} rows")

    return merged


# =============================================================================
# PLOT HELPERS
# =============================================================================
def format_boxplot_axis(ax: plt.Axes, metric: str, offset: int,
                        time_label: str = ""):
    """Apply formatting to boxplot axis."""
    # Zero reference line
    ax.axhline(0, color=ZERO_LINE_COLOR, linestyle=ZERO_LINE_STYLE,
              linewidth=ZERO_LINE_WIDTH, zorder=1)

    # Labels
    ylabel = METRIC_LABELS[metric]
    if time_label:
        ylabel += f"\n{time_label}"

    ax.set_ylabel(ylabel, fontsize=7, labelpad=3)
    ax.set_xlabel("Quartile (by n_SLH)", fontsize=7, labelpad=3)

    # Title
    ax.set_title(f"d = {offset} min", fontsize=7, fontweight="bold",
                loc="left", pad=3)

    # Grid and spines
    ax.grid(True, alpha=0.3, linewidth=0.5, axis='y', zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(length=3, width=0.8, labelsize=6.5)


# =============================================================================
# FIGURE GENERATION
# =============================================================================
def create_quartile_boxplots(data: pd.DataFrame, metric: str,
                            daytime_only: bool = False) -> plt.Figure:
    """
    Create figure with boxplots for one metric across all offsets.

    Parameters
    ----------
    data : pd.DataFrame
        Merged sensitivity + quartile data
    metric : str
        Metric name
    daytime_only : bool
        Use daytime metrics

    Returns
    -------
    plt.Figure
    """
    apply_publication_style()

    # Column prefix
    prefix = "daytime_delta_" if daytime_only else "delta_"
    col_name = f"{prefix}{metric}"

    # Get unique offsets
    offsets = sorted(data["offset"].unique())
    n_offsets = len(offsets)

    # Create figure with subplots (one per offset)
    fig_height = n_offsets * SUBPLOT_HEIGHT
    fig, axes = plt.subplots(n_offsets, 1, figsize=(FIGURE_WIDTH, fig_height))

    # Handle single offset case
    if n_offsets == 1:
        axes = [axes]

    fig.subplots_adjust(left=0.12, right=0.96, top=0.95, bottom=0.08,
                       hspace=0.40)

    # Title
    time_label = "daytime (06:00-22:00)" if daytime_only else "full 24h"
    fig.suptitle(
        f"{METRIC_LABELS[metric]} by quartile ({time_label})",
        fontsize=9, fontweight="bold"
    )

    # Create subplot for each offset
    for idx, offset in enumerate(offsets):
        ax = axes[idx]

        # Filter data for this offset
        data_offset = data[data["offset"] == offset]

        # Prepare data by quartile
        quartile_data = []
        quartile_labels = []
        quartile_colors = []

        for q_label in ["Q1", "Q2", "Q3", "Q4"]:
            df_q = data_offset[data_offset["quartile"] == q_label]

            if len(df_q) > 0:
                delta_vals = df_q[col_name].dropna().values
                quartile_data.append(delta_vals)

                # Get label with n_SLH range
                label = df_q["quartile_label"].iloc[0]
                quartile_labels.append(label)
                quartile_colors.append(QUARTILE_COLORS[q_label])

        # Create boxplot
        if quartile_data:
            bp = ax.boxplot(
                quartile_data,
                tick_labels=quartile_labels,
                widths=BOX_WIDTH,
                patch_artist=True,
                showfliers=True,
                medianprops=dict(color='black', linewidth=MEDIAN_LW),
                boxprops=dict(linewidth=BOX_LW),
                whiskerprops=dict(linewidth=BOX_LW),
                capprops=dict(linewidth=BOX_LW),
                flierprops=dict(marker='o', markersize=FLIER_SIZE,
                               alpha=FLIER_ALPHA, markerfacecolor='#666666',
                               markeredgecolor='none')
            )

            # Color boxes
            for patch, color in zip(bp['boxes'], quartile_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        # Format axis
        time_str = "(daytime)" if daytime_only else ""
        format_boxplot_axis(ax, metric, offset, time_str)

    return fig


def generate_all_metric_figures(data: pd.DataFrame, output_dir: str):
    """
    Generate boxplot figures for all metrics (full day + daytime).

    Parameters
    ----------
    data : pd.DataFrame
        Merged data
    output_dir : str
        Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n  Generating figures (all offsets)...")

    for metric in METRICS:
        # Full day
        print(f"    {metric} (full 24h)...", end="")
        fig_full = create_quartile_boxplots(data, metric, daytime_only=False)
        path_full = os.path.join(output_dir, f"{metric}_fulltime_quartiles")
        save_figure(fig_full, path_full)

        # Daytime
        print(f"    {metric} (daytime)...", end="")
        fig_day = create_quartile_boxplots(data, metric, daytime_only=True)
        path_day = os.path.join(output_dir, f"{metric}_daytime_quartiles")
        save_figure(fig_day, path_day)


def create_single_offset_boxplot(data: pd.DataFrame, metric: str, offset: int,
                                 daytime_only: bool = False) -> plt.Figure:
    """
    Create single-offset boxplot figure (no subplots).

    Parameters
    ----------
    data : pd.DataFrame
        Merged sensitivity + quartile data
    metric : str
        Metric name
    offset : int
        Zone-expansion offset (minutes)
    daytime_only : bool
        Use daytime metrics

    Returns
    -------
    plt.Figure
    """
    apply_publication_style()

    # Column prefix
    prefix = "daytime_delta_" if daytime_only else "delta_"
    col_name = f"{prefix}{metric}"

    # Create single-panel figure
    fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH * 0.6, 3.5))
    fig.subplots_adjust(left=0.14, right=0.96, top=0.90, bottom=0.14)

    # Title
    time_label = "daytime (06:00-22:00)" if daytime_only else "full 24h"
    fig.suptitle(
        f"{METRIC_LABELS[metric]} by quartile - d = {offset} min - {time_label}",
        fontsize=8, fontweight="bold"
    )

    # Filter data for this offset
    data_offset = data[data["offset"] == offset]

    # Prepare data by quartile
    quartile_data = []
    quartile_labels = []
    quartile_colors = []

    for q_label in ["Q1", "Q2", "Q3", "Q4"]:
        df_q = data_offset[data_offset["quartile"] == q_label]

        if len(df_q) > 0:
            delta_vals = df_q[col_name].dropna().values
            quartile_data.append(delta_vals)

            # Get label with n_SLH range
            label = df_q["quartile_label"].iloc[0]
            quartile_labels.append(label)
            quartile_colors.append(QUARTILE_COLORS[q_label])

    # Create boxplot
    if quartile_data:
        bp = ax.boxplot(
            quartile_data,
            labels=quartile_labels,
            widths=BOX_WIDTH,
            patch_artist=True,
            showfliers=True,
            medianprops=dict(color='black', linewidth=MEDIAN_LW),
            boxprops=dict(linewidth=BOX_LW),
            whiskerprops=dict(linewidth=BOX_LW),
            capprops=dict(linewidth=BOX_LW),
            flierprops=dict(marker='o', markersize=FLIER_SIZE,
                           alpha=FLIER_ALPHA, markerfacecolor='#666666',
                           markeredgecolor='none')
        )

        # Color boxes
        for patch, color in zip(bp['boxes'], quartile_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

    # Zero reference line
    ax.axhline(0, color=ZERO_LINE_COLOR, linestyle=ZERO_LINE_STYLE,
              linewidth=ZERO_LINE_WIDTH, zorder=1)

    # Labels
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=8, labelpad=3)
    ax.set_xlabel("Quartile (by n_SLH)", fontsize=8, labelpad=3)

    # Grid and spines
    ax.grid(True, alpha=0.3, linewidth=0.5, axis='y', zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(length=3, width=0.8, labelsize=7)

    return fig


def generate_single_offset_figures(data: pd.DataFrame, offset: int, output_dir: str):
    """
    Generate boxplot figures for selected offset only.

    Parameters
    ----------
    data : pd.DataFrame
        Merged data
    offset : int
        Selected offset (minutes)
    output_dir : str
        Output directory
    """
    offset_dir = os.path.join(output_dir, f"Offset_{offset}min")
    os.makedirs(offset_dir, exist_ok=True)

    print(f"\n  Generating figures (d={offset} min only)...")

    for metric in METRICS:
        # Full day
        print(f"    {metric} (full 24h)...", end="")
        fig_full = create_single_offset_boxplot(data, metric, offset, daytime_only=False)
        path_full = os.path.join(offset_dir, f"{metric}_fulltime_offset{offset}")
        save_figure(fig_full, path_full)
        print(" done")

        # Daytime
        print(f"    {metric} (daytime)...", end="")
        fig_day = create_single_offset_boxplot(data, metric, offset, daytime_only=True)
        path_day = os.path.join(offset_dir, f"{metric}_daytime_offset{offset}")
        save_figure(fig_day, path_day)
        print(" done")


# =============================================================================
# STATISTICS EXPORT
# =============================================================================
def compute_table_statistics(data: pd.DataFrame, metric: str, offset: int,
                             daytime_only: bool = False) -> pd.DataFrame:
    """
    Compute statistics in Table 2 format: median (IQR) for all three columns.

    Returns DataFrame with columns:
    - group: "Overall" or "Quartile X (n_SLH=...)"
    - n: sample size
    - orig_median, orig_q25, orig_q75: Original values
    - ctf_median, ctf_q25, ctf_q75: Counterfactual values
    - delta_median, delta_q25, delta_q75: Delta values (IQR)
    """
    prefix = "daytime" if daytime_only else ""

    orig_col = f"{prefix}_orig_{metric}" if prefix else f"orig_{metric}"
    ctf_col = f"{prefix}_ctf_{metric}" if prefix else f"ctf_{metric}"
    delta_col = f"{prefix}_delta_{metric}" if prefix else f"delta_{metric}"

    # Filter for this offset
    data_offset = data[data["offset"] == offset]

    stats_rows = []

    # Overall cohort
    orig_vals = data_offset[orig_col].dropna().values
    ctf_vals = data_offset[ctf_col].dropna().values
    delta_vals = data_offset[delta_col].dropna().values

    if len(delta_vals) > 0:
        stats_rows.append({
            "group": f"Overall (n = {len(data_offset['patient_id'].unique())})",
            "n": len(data_offset['patient_id'].unique()),
            "orig_median": np.median(orig_vals),
            "orig_q25": np.percentile(orig_vals, 25),
            "orig_q75": np.percentile(orig_vals, 75),
            "ctf_median": np.median(ctf_vals),
            "ctf_q25": np.percentile(ctf_vals, 25),
            "ctf_q75": np.percentile(ctf_vals, 75),
            "delta_median": np.median(delta_vals),
            "delta_q25": np.percentile(delta_vals, 25),
            "delta_q75": np.percentile(delta_vals, 75),
        })

    # Per quartile
    for q_idx, q_label in enumerate(["Q1", "Q2", "Q3", "Q4"], start=1):
        df_q = data_offset[data_offset["quartile"] == q_label]

        if len(df_q) == 0:
            continue

        orig_vals_q = df_q[orig_col].dropna().values
        ctf_vals_q = df_q[ctf_col].dropna().values
        delta_vals_q = df_q[delta_col].dropna().values

        if len(delta_vals_q) == 0:
            continue

        # Get n_SLH range for this quartile
        n_slh_range = df_q["n_SLH_range"].iloc[0]
        n_patients = len(df_q)

        if q_idx == 1:
            group_label = f"Quartile 1 (lowest n_SLH = {n_slh_range}, n = {n_patients})"
        elif q_idx == 4:
            group_label = f"Quartile 4 (highest n_SLH = {n_slh_range}, n = {n_patients})"
        else:
            group_label = f"Quartile {q_idx} (n_SLH = {n_slh_range}, n = {n_patients})"

        stats_rows.append({
            "group": group_label,
            "n": n_patients,
            "orig_median": np.median(orig_vals_q),
            "orig_q25": np.percentile(orig_vals_q, 25),
            "orig_q75": np.percentile(orig_vals_q, 75),
            "ctf_median": np.median(ctf_vals_q),
            "ctf_q25": np.percentile(ctf_vals_q, 25),
            "ctf_q75": np.percentile(ctf_vals_q, 75),
            "delta_median": np.median(delta_vals_q),
            "delta_q25": np.percentile(delta_vals_q, 25),
            "delta_q75": np.percentile(delta_vals_q, 75),
        })

    df_stats = pd.DataFrame(stats_rows)

    # Format as text columns (matching Table 2)
    df_stats["Original"] = df_stats.apply(
        lambda row: f"{row['orig_median']:.1f} ({row['orig_q25']:.1f}-{row['orig_q75']:.1f})",
        axis=1
    )

    df_stats["Counterfactual"] = df_stats.apply(
        lambda row: f"{row['ctf_median']:.1f} ({row['ctf_q25']:.1f}-{row['ctf_q75']:.1f})",
        axis=1
    )

    df_stats["Delta"] = df_stats.apply(
        lambda row: f"{row['delta_median']:+.2f} ({row['delta_q25']:+.2f} to {row['delta_q75']:+.2f})",
        axis=1
    )

    # Final clean columns
    return df_stats[["group", "n", "Original", "Counterfactual", "Delta"]]


def export_table2_statistics(data: pd.DataFrame, offset: int, output_dir: str):
    """
    Export statistics in Table 2 format (separate CSV per metric).

    Creates detailed CSV files matching the manuscript table structure.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  Exporting Table 2 format statistics (d={offset} min)...")

    for metric in METRICS:
        # Full day statistics
        stats_full = compute_table_statistics(
            data, metric, offset, daytime_only=False
        )

        # Add header row with metric info
        stats_full.insert(0, "Metric", metric)

        path_full = os.path.join(
            output_dir,
            f"Table2_{metric}_fulltime_offset{offset}.csv"
        )
        stats_full.to_csv(path_full, index=False)
        print(f"    {metric} (full 24h)... done")

        # Daytime statistics
        stats_day = compute_table_statistics(
            data, metric, offset, daytime_only=True
        )

        stats_day.insert(0, "Metric", metric)

        path_day = os.path.join(
            output_dir,
            f"Table2_{metric}_daytime_offset{offset}.csv"
        )
        stats_day.to_csv(path_day, index=False)
        print(f"    {metric} (daytime)... done")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main execution pipeline."""
    print("=" * 70)
    print("  Quartile Stratification Analysis")
    print("=" * 70)
    print(f"  Sensitivity: {COHORT_SENSITIVITY_CSV}")
    print(f"  Filter:      {PATIENT_FILTER_CSV}")
    print(f"  Output:      {OUTPUT_DIR}")
    print(f"  Selected d:  {SELECTED_OFFSET} min")
    print("=" * 70)

    # Load data
    episodes_df = load_episode_summary()
    sensitivity_df = load_sensitivity_results()

    # Compute quartiles
    quartile_df = compute_quartile_assignments(episodes_df)

    # Save quartile assignments
    save_quartile_assignments(quartile_df, QUARTILES_CSV)

    # Merge data
    merged_df = merge_data(sensitivity_df, quartile_df)

    # Generate figures (all offsets)
    generate_all_metric_figures(merged_df, FIGURES_DIR)

    # Generate figures (selected offset only)
    generate_single_offset_figures(merged_df, SELECTED_OFFSET, FIGURES_DIR)

    # Export Table 2 format statistics
    export_table2_statistics(merged_df, SELECTED_OFFSET, STATS_DIR)

    print("\n" + "=" * 70)
    print(f"  Quartile assignments: {QUARTILES_CSV}")
    print(f"  Figures (all offsets): {FIGURES_DIR}/")
    print(f"  Figures (d={SELECTED_OFFSET}): {FIGURES_DIR}/Offset_{SELECTED_OFFSET}min/")
    print(f"  Statistics (Table 2): {STATS_DIR}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
