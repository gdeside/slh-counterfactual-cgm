"""
scripts/cgm_agp_recurrence.py
=============================
CGM Episode Visualization
=========================
Input       : Data/Episodes/{PID}_episodes.csv
              Data/Preprocessed/{PID}/{PID}_win1_*.xlsx
Output      : Figures/AGP/{PID}/
                  {PID}_agp_duration.pdf/.png
                  {PID}_agp_classification.pdf/.png

Generates two figure types per patient:

1. AGP with duration-based recurrence
   - Top panel: Ambulatory Glucose Profile (median, IQR, 5-95 percentile)
   - Bottom panel: Recurrence heatmap by episode duration bins

2. AGP with classification-based recurrence
   - Top panel: Ambulatory Glucose Profile (same as above)
   - Bottom panel: Recurrence heatmap by SLH vs LLH classification

Usage
-----
    python3 -m scripts.cgm_agp_recurrence [--help for options]
    (or: make agp_recurrence)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config.paths import (
    WINDOW_SUMMARY_CSV,
)

from src.config.figure_config import (
        FIG_DIR,
        FIGURE_WIDTH_INCHES,
        FIGURE_HEIGHT_RATIO
)


from src.config.constants import (
    TIME_COL, 
    PATIENT_ID_COL,
    CLASSIFICATION_SLH,
    CLASSIFICATION_LLH,
)
from src.preprocessing.load_data import load_patient_window, prepare_valid_data
from src.preprocessing.timestep import detect_dominant_sampling_interval


from src.episodes.load_episodes import load_patient_episodes


from src.agp.compute_agp import compute_agp_profile, compute_recurrence_by_duration, compute_recurrence_by_classification

from src.visualization.style import apply_publication_style, save_figure

from src.visualization.agp_plot import plot_agp_panel, plot_recurrence_classification_panel, plot_recurrence_duration_panel

FIGURES_OUTPUT_PATH = os.path.join(FIG_DIR, "AGP")


# ── Duration bins for recurrence (minutes) ────────────────────────────────────
DURATION_BINS = [0, 15, 30, 45, 60, 75, 90, np.inf]
DURATION_LABELS = ["0–15", "15–30", "30–45", "45–60", "60–75", "75–90", "90+"]

# ── Classification labels ─────────────────────────────────────────────────────
CLASSIFICATION_LABELS = [CLASSIFICATION_SLH, CLASSIFICATION_LLH]


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_agp_duration_figure(agp: pd.DataFrame, 
                                 episodes: pd.DataFrame,
                                 valid_data: pd.DataFrame,
                                 patient_id: str) -> plt.Figure:
    """
    Generate figure with AGP (top) and duration-based recurrence (bottom).
    
    Parameters
    ----------
    agp : pd.DataFrame
        AGP profile data
    episodes : pd.DataFrame
        Episode data
    valid_data : pd.DataFrame
        Valid CGM data
    patient_id : str
        Patient identifier
        
    Returns
    -------
    matplotlib.figure.Figure
        Generated figure
    """
    apply_publication_style()
    
    fig_width = FIGURE_WIDTH_INCHES
    fig_height = fig_width * FIGURE_HEIGHT_RATIO
    
    fig, axes = plt.subplots(
        2, 1,
        figsize=(fig_width, fig_height),
        height_ratios=[1, 1]
    )
    fig.subplots_adjust(left=0.10, right=0.95, top=0.94, bottom=0.08, hspace=0.35)
    
    # Top panel: AGP
    plot_agp_panel(axes[0], agp, patient_id)
    
    # Bottom panel: Recurrence by duration
    resolution_min = detect_dominant_sampling_interval(valid_data)
    start_date = valid_data[TIME_COL].dt.date.min()
    end_date = valid_data[TIME_COL].dt.date.max()
    n_days = (end_date - start_date).days + 1
    
    occurrence = compute_recurrence_by_duration(
        episodes, resolution_min, n_days, start_date
    )
    plot_recurrence_duration_panel(axes[1], occurrence, patient_id, resolution_min)
    
    return fig


def generate_agp_classification_figure(agp: pd.DataFrame,
                                       episodes: pd.DataFrame,
                                       valid_data: pd.DataFrame,
                                       patient_id: str) -> plt.Figure:
    """
    Generate figure with AGP (top) and classification-based recurrence (bottom).
    
    Parameters
    ----------
    agp : pd.DataFrame
        AGP profile data
    episodes : pd.DataFrame
        Episode data
    valid_data : pd.DataFrame
        Valid CGM data
    patient_id : str
        Patient identifier
        
    Returns
    -------
    matplotlib.figure.Figure
        Generated figure
    """
    apply_publication_style()
    
    fig_width = FIGURE_WIDTH_INCHES
    fig_height = fig_width * FIGURE_HEIGHT_RATIO
    
    fig, axes = plt.subplots(
        2, 1,
        figsize=(fig_width, fig_height),
        height_ratios=[1, 1]
    )
    fig.subplots_adjust(left=0.10, right=0.95, top=0.94, bottom=0.08, hspace=0.35)
    
    # Top panel: AGP
    plot_agp_panel(axes[0], agp, patient_id)
    
    # Bottom panel: Recurrence by classification
    resolution_min = detect_dominant_sampling_interval(valid_data)
    start_date = valid_data[TIME_COL].dt.date.min()
    end_date = valid_data[TIME_COL].dt.date.max()
    n_days = (end_date - start_date).days + 1
    
    occurrence = compute_recurrence_by_classification(
        episodes, resolution_min, n_days, start_date
    )
    plot_recurrence_classification_panel(axes[1], occurrence, patient_id, resolution_min)
    
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Main visualization pipeline."""
    print("=" * 70)
    print("  CGM Episode Visualization")
    print("=" * 70)

    # Validate input file exists
    if not os.path.exists(WINDOW_SUMMARY_CSV):
        raise FileNotFoundError(
            f"window_summary.csv not found at:\n  {WINDOW_SUMMARY_CSV}\n"
            "Run cgm_preprocessing.py first."
        )

    # Load patient list
    summary_df = pd.read_csv(WINDOW_SUMMARY_CSV)
    print(f"  {len(summary_df)} patients with valid windows")
    print(f"  Generating AGP and recurrence figures ...\n")

    # Process all patients
    n_success = 0
    skipped_patients = []  # Track skipped patients with reasons
    errors = []

    for _, row in summary_df.iterrows():
        patient_id = row[PATIENT_ID_COL]
        
        # Load CGM window
        window_df = load_patient_window(patient_id)
        if window_df is None:
            skipped_patients.append(f"{patient_id}: No valid window data")
            continue
        
        valid_data = prepare_valid_data(window_df)
        if len(valid_data) < 10:
            skipped_patients.append(f"{patient_id}: Insufficient valid data points ({len(valid_data)} < 10)")
            continue
        
        # Load episodes
        episodes = load_patient_episodes(patient_id)
        if episodes is None or len(episodes) == 0:
            skipped_patients.append(f"{patient_id}: No episodes detected")
            continue
        
        try:
            # Compute AGP
            resolution_min = detect_dominant_sampling_interval(valid_data)
            agp = compute_agp_profile(valid_data, resolution_min)
            
            # Create output directory
            output_dir = os.path.join(FIGURES_OUTPUT_PATH, patient_id)
            os.makedirs(output_dir, exist_ok=True)
            
            # Figure 1: AGP + duration-based recurrence
            fig_duration = generate_agp_duration_figure(
                agp, episodes, valid_data, patient_id
            )
            output_path_duration = os.path.join(output_dir, f"{patient_id}_agp_duration")
            save_figure(fig_duration, output_path_duration)
            
            # Figure 2: AGP + classification-based recurrence
            fig_classification = generate_agp_classification_figure(
                agp, episodes, valid_data, patient_id
            )
            output_path_classification = os.path.join(
                output_dir, f"{patient_id}_agp_classification"
            )
            save_figure(fig_classification, output_path_classification)
            
            n_success += 1
            
        except Exception as e:
            errors.append(f"{patient_id}: {e}")

    # Print skipped patients
    if skipped_patients:
        print("\n  Skipped patients:")
        for skip_msg in skipped_patients:
            print(f"    {skip_msg}")

    # Print warnings if any
    if errors:
        print("\n  Errors:")
        for error in errors:
            print(f"    {error}")

    # Print summary
    print()
    print("=" * 70)
    print(f"  Figures generated: {n_success} patients")
    print(f"  Skipped:           {len(skipped_patients)} patients")
    if errors:
        print(f"  Errors:            {len(errors)} patients")
    print()
    print(f"  Output: {FIGURES_OUTPUT_PATH}")
    print("  Files per patient:")
    print("    {patient_id}_agp_duration.{{pdf,png}}")
    print("    {patient_id}_agp_classification.{{pdf,png}}")
    print("=" * 70)


if __name__ == "__main__":
    main()