"""
scripts/patient_comparison.py
=============================
Patient Comparison Visualization (Enhanced)
============================================
Input       : Data/Preprocessed/{PID}/{PID}_win1_*.xlsx
              Data/Episodes/{PID}_episodes.csv
Output      : Figures/Comparisons/{PID_A}_vs_{PID_B}/
                  comparison_{PID_A}_vs_{PID_B}_duration.{pdf,svg}
                  comparison_{PID_A}_vs_{PID_B}_classification.{pdf,svg}
                  metrics_{PID_A}_vs_{PID_B}.csv

Generates comprehensive side-by-side comparison figures for two patients.

ENHANCEMENTS:
- Legends only shown on second subplot (right column) for each row
- Threshold annotations only shown on second subplot (right column) for each row
- Reduces visual clutter while maintaining all information


Usage:
    python3 -m scripts.patient_comparison
"""
import os
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

from src.utils.patient_selection import resolve_patient_id, resolve_patient_pair
from src.config.constants import (
    TIME_COL, 
    GLUCOSE_COL,
    COL_EPISODE_CLASSIFICATION,
    CLASSIFICATION_SLH,
    CLASSIFICATION_LLH,
)

from src.config.figure_config import (
    FIG_DIR
)

from src.preprocessing.load_data import load_patient_window, prepare_valid_data
from src.preprocessing.timestep import detect_dominant_sampling_interval

from src.episodes.load_episodes import load_patient_episodes

from src.agp.compute_agp import compute_agp_profile, compute_recurrence_by_duration, compute_recurrence_by_classification

from src.visualization.style import apply_publication_style, save_figure

from src.visualization.agp_plot import plot_agp_panel, plot_recurrence_duration_panel, plot_recurrence_classification_panel, plot_metrics_table, plot_7day_trace, add_panel_label


# =============================================================================
# PARAMETERS 
# =============================================================================
# High- / low-SLH-frequency patients: CGM_PATIENT_HIGH / CGM_PATIENT_LOW, else automatic (most / fewest SLH episodes)
PATIENT_A, PATIENT_B = resolve_patient_pair()


FIGURES_OUTPUT_PATH = os.path.join(FIG_DIR, "Comparisons")


# ── Duration bins for recurrence (minutes) ────────────────────────────────────
DURATION_BINS = [0, 15, 30, 45, 60, 75, 90, np.inf]
DURATION_LABELS = ["0–15", "15–30", "30–45", "45–60", "60–75", "75–90", "90+"]


CLASSIFICATION_LABELS = [CLASSIFICATION_SLH, CLASSIFICATION_LLH]

# ── AGP parameters ────────────────────────────────────────────────────────────
AGP_TIME_SLOT_TOLERANCE = 0.4  # Fraction of resolution for time matching
AGP_Y_MIN = 0                   # Y-axis minimum (mg/dL)
AGP_Y_MAX = 350                 # Y-axis maximum (mg/dL)
AGP_X_TICK_INTERVAL_HOURS = 3   # X-axis tick interval

# ── Figure parameters ─────────────────────────────────────────────────────────
FIGURE_WIDTH = 7.24            # Figure width (inches)
FIGURE_HEIGHT = 9.6            # Figure height (inches, increased for better visibility)



# ─────────────────────────────────────────────────────────────────────────────
# METRICS COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_cgm_metrics(df: pd.DataFrame) -> dict:
    """Compute standard CGM metrics."""
    if len(df) < 10:
        return {}
    
    glucose_values = df[GLUCOSE_COL]
    n_total = len(df)
    
    mean_glucose = glucose_values.mean()
    
    return {
        "Mean": round(mean_glucose, 1),
        "Median": round(glucose_values.median(), 1),
        "SD": round(glucose_values.std(), 1),
        "CV": round(glucose_values.std() / mean_glucose * 100 if mean_glucose > 0 else 0, 1),
        "TIR70-180": round(((glucose_values >= 70) & (glucose_values <= 180)).sum() / n_total * 100, 1),
        "TAR>180": round((glucose_values > 180).sum() / n_total * 100, 1),
        "TAR>250": round((glucose_values > 250).sum() / n_total * 100, 1),
        "TBR<70": round((glucose_values < 70).sum() / n_total * 100, 1),
    }


def count_episodes_by_classification(episodes: pd.DataFrame) -> dict:
    """Count episodes by SLH/LLH classification."""
    if episodes is None or len(episodes) == 0:
        return {CLASSIFICATION_SLH: 0, CLASSIFICATION_LLH: 0}
    
    counts = episodes[COL_EPISODE_CLASSIFICATION].value_counts().to_dict()
    
    return {
        CLASSIFICATION_SLH: counts.get(CLASSIFICATION_SLH, 0),
        CLASSIFICATION_LLH: counts.get(CLASSIFICATION_LLH, 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE GENERATION (ENHANCED)
# ─────────────────────────────────────────────────────────────────────────────
def generate_duration_comparison_figure(patient_a: str, patient_b: str,
                                       data_a: dict, data_b: dict) -> plt.Figure:
    """
    Generate duration-based comparison figure.
    ENHANCED: Legends and threshold annotations only on second subplot (column B, D, F).
    """
    apply_publication_style()
    
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    gs = GridSpec(
        4, 2,
        figure=fig,
        height_ratios=[1, 1, 0.8, 1.2],
        hspace=0.38,
        wspace=0.25,
        left=0.08,
        right=0.96,
        top=0.95,
        bottom=0.05
    )
    
    # Row 1: AGP (A: no legend/annotations, B: with legend/annotations)
    ax_agp_a = fig.add_subplot(gs[0, 0])
    ax_agp_b = fig.add_subplot(gs[0, 1])
    
    plot_agp_panel(ax_agp_a, data_a["agp"], patient_a, 
                  show_legend=False, show_threshold_annotations=False)
    plot_agp_panel(ax_agp_b, data_b["agp"], patient_b,
                  show_legend=True, show_threshold_annotations=True)
    add_panel_label(ax_agp_a, "A")
    add_panel_label(ax_agp_b, "B")
    
    # Row 2: Recurrence by duration (C: no legend/annotations, D: with legend/annotations)
    ax_rec_a = fig.add_subplot(gs[1, 0])
    ax_rec_b = fig.add_subplot(gs[1, 1])
    
    plot_recurrence_duration_panel(
        ax_rec_a, data_a["recurrence_duration"], patient_a, data_a["resolution"],
        show_legend=False, show_threshold_annotation=False
    )
    plot_recurrence_duration_panel(
        ax_rec_b, data_b["recurrence_duration"], patient_b, data_b["resolution"],
        show_legend=True, show_threshold_annotation=True
    )
    add_panel_label(ax_rec_a, "C")
    add_panel_label(ax_rec_b, "D")
    
    # Row 3: 7-day glucose traces (E: no annotations, F: with annotations)
    ax_trace_a = fig.add_subplot(gs[2, 0])
    ax_trace_b = fig.add_subplot(gs[2, 1])
    
    plot_7day_trace(ax_trace_a, data_a["valid_data"], data_a["episodes"], patient_a,
                   show_threshold_annotations=False)
    plot_7day_trace(ax_trace_b, data_b["valid_data"], data_b["episodes"], patient_b,
                   show_threshold_annotations=True)
    add_panel_label(ax_trace_a, "E")
    add_panel_label(ax_trace_b, "F")
    
    # Row 4: Metrics table (full width)
    ax_metrics = fig.add_subplot(gs[3, :])
    plot_metrics_table(
        ax_metrics,
        data_a["metrics"], data_b["metrics"],
        data_a["counts"], data_b["counts"],
        patient_a, patient_b,
        include_classification=False
    )
    add_panel_label(ax_metrics, "G")
    
    # Main title
    fig.suptitle(
        f"Patient Comparison — Duration: {patient_a} vs {patient_b}",
        fontsize=14,
        fontweight="bold",
        y=0.985
    )
    
    return fig


def generate_classification_comparison_figure(patient_a: str, patient_b: str,
                                              data_a: dict, data_b: dict) -> plt.Figure:
    """
    Generate classification-based comparison figure.
    ENHANCED: Legends and threshold annotations only on second subplot (column B, D, F).
    """
    apply_publication_style()
    
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    gs = GridSpec(
        4, 2,
        figure=fig,
        height_ratios=[1, 1, 0.8, 1.2],
        hspace=0.38,
        wspace=0.25,
        left=0.08,
        right=0.96,
        top=0.95,
        bottom=0.05
    )
    
    # Row 1: AGP (A: no legend/annotations, B: with legend/annotations)
    ax_agp_a = fig.add_subplot(gs[0, 0])
    ax_agp_b = fig.add_subplot(gs[0, 1])
    
    plot_agp_panel(ax_agp_a, data_a["agp"], patient_a,
                  show_legend=False, show_threshold_annotations=False)
    plot_agp_panel(ax_agp_b, data_b["agp"], patient_b,
                  show_legend=True, show_threshold_annotations=True)
    add_panel_label(ax_agp_a, "A")
    add_panel_label(ax_agp_b, "B")
    
    # Row 2: Recurrence by classification (C: no legend/annotations, D: with legend/annotations)
    ax_rec_a = fig.add_subplot(gs[1, 0])
    ax_rec_b = fig.add_subplot(gs[1, 1])
    
    plot_recurrence_classification_panel(
        ax_rec_a, data_a["recurrence_classification"], patient_a, data_a["resolution"],
        show_legend=False, show_threshold_annotation=False
    )
    plot_recurrence_classification_panel(
        ax_rec_b, data_b["recurrence_classification"], patient_b, data_b["resolution"],
        show_legend=True, show_threshold_annotation=True
    )
    add_panel_label(ax_rec_a, "C")
    add_panel_label(ax_rec_b, "D")
    
    # Row 3: 7-day glucose traces (E: no annotations, F: with annotations)
    ax_trace_a = fig.add_subplot(gs[2, 0])
    ax_trace_b = fig.add_subplot(gs[2, 1])
    
    plot_7day_trace(ax_trace_a, data_a["valid_data"], data_a["episodes"], patient_a,
                   show_threshold_annotations=False)
    plot_7day_trace(ax_trace_b, data_b["valid_data"], data_b["episodes"], patient_b,
                   show_threshold_annotations=True)
    add_panel_label(ax_trace_a, "E")
    add_panel_label(ax_trace_b, "F")
    
    # Row 4: Metrics table (full width)
    ax_metrics = fig.add_subplot(gs[3, :])
    plot_metrics_table(
        ax_metrics,
        data_a["metrics"], data_b["metrics"],
        data_a["counts"], data_b["counts"],
        patient_a, patient_b,
        include_classification=True
    )
    add_panel_label(ax_metrics, "G")
    
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PATIENT DATA PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def process_patient_data(patient_id: str) -> dict | None:
    """Load and process all data for a single patient."""
    # Load CGM window
    window_df = load_patient_window(patient_id)
    if window_df is None:
        return None
    
    valid_data = prepare_valid_data(window_df)
    if len(valid_data) < 10:
        return None
    
    # Load episodes
    episodes = load_patient_episodes(patient_id)
    if episodes is None or len(episodes) == 0:
        return None
    
    # Compute AGP
    resolution = detect_dominant_sampling_interval(valid_data)
    agp = compute_agp_profile(valid_data, resolution)
    
    # Compute metrics
    metrics = compute_cgm_metrics(valid_data)
    counts = count_episodes_by_classification(episodes)
    
    # Compute recurrence
    start_date = valid_data[TIME_COL].dt.date.min()
    end_date = valid_data[TIME_COL].dt.date.max()
    n_days = (end_date - start_date).days + 1
    
    recurrence_duration = compute_recurrence_by_duration(
        episodes, resolution, n_days, start_date
    )
    recurrence_classification = compute_recurrence_by_classification(
        episodes, resolution, n_days, start_date
    )
    
    return {
        "agp": agp,
        "metrics": metrics,
        "counts": counts,
        "resolution": resolution,
        "recurrence_duration": recurrence_duration,
        "recurrence_classification": recurrence_classification,
        "valid_data": valid_data,  # Include for 7-day trace
        "episodes": episodes,       # Include for 7-day trace
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Main comparison pipeline."""
    print("=" * 70)
    print(f"  Patient Comparison: {PATIENT_A} vs {PATIENT_B}")
    print("=" * 70)
    print("  ENHANCEMENTS:")
    print("    - Legends only on right column (B, D)")
    print("    - Threshold annotations only on right column (B, D, F)")
    print("    - Cleaner layout with reduced redundancy")
    print("=" * 70)
    
    # Create pair-specific output subfolder
    pair_dir = os.path.join(FIGURES_OUTPUT_PATH, f"{PATIENT_A}_vs_{PATIENT_B}")
    os.makedirs(pair_dir, exist_ok=True)

    # Load and process data for both patients
    print(f"\n  Loading {PATIENT_A}...", end=" ")
    data_a = process_patient_data(PATIENT_A)
    if data_a is None:
        print("FAILED")
        return
    print(f"OK (SLH={data_a['counts'][CLASSIFICATION_SLH]}, "
          f"LLH={data_a['counts'][CLASSIFICATION_LLH]})")

    print(f"  Loading {PATIENT_B}...", end=" ")
    data_b = process_patient_data(PATIENT_B)
    if data_b is None:
        print("FAILED")
        return
    print(f"OK (SLH={data_b['counts'][CLASSIFICATION_SLH]}, "
          f"LLH={data_b['counts'][CLASSIFICATION_LLH]})")

    # Save metrics CSV
    print(f"\n  Saving metrics CSV...")
    metrics_rows = []
    for pid, data, counts in [
        (PATIENT_A, data_a, data_a["counts"]),
        (PATIENT_B, data_b, data_b["counts"]),
    ]:
        row = {"patient_id": pid}
        row.update(data["metrics"])
        row["n_SLH"] = counts[CLASSIFICATION_SLH]
        row["n_LLH"] = counts[CLASSIFICATION_LLH]
        row["n_total"] = counts[CLASSIFICATION_SLH] + counts[CLASSIFICATION_LLH]
        metrics_rows.append(row)
    metrics_csv_path = os.path.join(
        pair_dir, f"metrics_{PATIENT_A}_vs_{PATIENT_B}.csv"
    )
    pd.DataFrame(metrics_rows).to_csv(metrics_csv_path, index=False)
    print(f"    Saved: {metrics_csv_path}")

    # Generate duration-based comparison
    print(f"\n  Generating duration comparison figure...")
    fig_duration = generate_duration_comparison_figure(
        PATIENT_A, PATIENT_B, data_a, data_b
    )
    output_path_duration = os.path.join(
        pair_dir,
        f"comparison_{PATIENT_A}_vs_{PATIENT_B}_duration"
    )
    save_figure(fig_duration, output_path_duration)
    print(f"    Saved: {output_path_duration}.{{pdf,svg}}")

    # Generate classification-based comparison
    print(f"\n  Generating classification comparison figure...")
    fig_classification = generate_classification_comparison_figure(
        PATIENT_A, PATIENT_B, data_a, data_b
    )
    output_path_classification = os.path.join(
        pair_dir,
        f"comparison_{PATIENT_A}_vs_{PATIENT_B}_classification"
    )
    save_figure(fig_classification, output_path_classification)
    print(f"    Saved: {output_path_classification}.{{pdf,svg}}")

    print("\n" + "=" * 70)
    print(f"  Output directory: {pair_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()