"""
Episode Classification Funnel — Single-Patient Visualization
============================================================

Generate a funnel (waterfall) figure illustrating the sequential filtering
of hyperglycemia episodes across criteria C1 → C6 for a single patient.

The pipeline:
    1. Load CGM data
    2. Detect hyperglycemia episodes
    3. Apply sequential exclusion criteria
    4. Visualize retention/exclusion at each stage
"""

import os
import sys
import argparse


# Data & processing
from src.preprocessing.load_data import load_patient_window
from src.preprocessing.timestep import detect_dominant_sampling_interval
from src.episodes.detection import detect_hyperglycemia_episodes
from src.episodes.criteria import classify_episodes_sequential

# Config
from src.config import FIG_DIR

# Episode parameters (used in reporting)
from src.episodes.episode_config import (
    H_CONTEXT,
    D_MIN_MIN, D_MAX_MIN,
    MIN_DATA_COMPLETENESS,
)

# Visualization
from src.visualization.episode_plots import plot_funnel_figure


# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_PATIENT_ID = "A159"
DEFAULT_OUTPUT_DIR = os.path.join(FIG_DIR, "SLH_analysis")


def print_funnel_summary(funnel_stats: dict, patient_id: str):
    """
    Print detailed summary of funnel statistics.
    
    Parameters
    ----------
    funnel_stats : dict
        Statistics from classify_episodes_sequential
    patient_id : str
        Patient identifier
    """
    print("\n" + "=" * 70)
    print(f"  Episode Classification Funnel — {patient_id}")
    print("=" * 70)
    
    print(f"\nRaw episodes detected: {funnel_stats['n_raw']}")
    print(f"\nSequential filtering:")
    print(f"  C1  Duration [{D_MIN_MIN}–{D_MAX_MIN} min]:        {funnel_stats['n_c1']}")
    print(f"  C2  + Data ≥{MIN_DATA_COMPLETENESS*100:.0f}%:                  {funnel_stats['n_c2']}")
    print(f"  C3  + Daytime (06:00–22:00):        {funnel_stats['n_c3']}")
    print(f"  C4  + No hyper before:              {funnel_stats['n_c4']}")
    print(f"  C5  + No hyper after:               {funnel_stats['n_c5']}")
    print(f"  SLH + OR-crossing ≤{H_CONTEXT}:            {funnel_stats['n_slh']}")
    
    print(f"\nExclusions at each stage:")
    for stage, count in funnel_stats['excluded_at_each_stage'].items():
        if count > 0:
            pct = 100 * count / funnel_stats['n_raw']
            print(f"  {stage}: {count:3d} episodes ({pct:5.1f}%)")
    
    print(f"\nSLH crossing composition:")
    print(f"  Before only: {funnel_stats['n_before_only']}")
    print(f"  After only:  {funnel_stats['n_after_only']}")
    print(f"  Bilateral:   {funnel_stats['n_both']}")
    
    if funnel_stats['n_no_cross'] > 0:
        print(f"  No crossing: {funnel_stats['n_no_cross']}  (excluded at C6)")
    
    print("=" * 70)


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def generate_funnel_for_patient(patient_id: str, output_dir: str):
    """
    Generate episode classification funnel for a single patient.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier (e.g., 'A224')
    output_dir : str
        Output directory for figure
        
    Returns
    -------
    dict or None
        Funnel statistics if successful, None if failed
    """
    # Load data
    print(f"\nProcessing patient {patient_id}...")
    df = load_patient_window(patient_id)
    
    if df is None:
        print(f"  Failed to load data for {patient_id}")
        return None
    
    print(f"Loaded {len(df):,} data points")
    
    # Detect timestep
    timestep_min = detect_dominant_sampling_interval(df)
    print(f"Detected sampling interval: {timestep_min:.0f} min")
    
    # Detect raw episodes
    raw_episodes = detect_hyperglycemia_episodes(df)
    print(f"Detected {len(raw_episodes)} raw hyperglycemia episodes")
    
    if not raw_episodes:
        print(f"No episodes detected")
        return None
    
    # Sequential classification
    funnel_stats = classify_episodes_sequential(df, raw_episodes)
    
    # Print summary
    print_funnel_summary(funnel_stats, patient_id)
    
    # Generate figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{patient_id}_funnel")
    
    plot_funnel_figure(funnel_stats, patient_id, output_path)
    
    return funnel_stats


# =============================================================================
# CLI
# =============================================================================
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate episode classification funnel for a patient",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--patient_id", 
        default=DEFAULT_PATIENT_ID,
        help="Patient ID (e.g., A224)"
    )
    parser.add_argument(
        "--output", 
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for figure"
    )
    return parser


def main():
    """Main execution."""
    args = parse_arguments().parse_args()
    
    print("\n" + "=" * 70)
    print("  Episode Classification Funnel Generator")
    print("=" * 70)
    print(f"  Patient: {args.patient_id}")
    print(f"  Output:  {args.output}")
    print("=" * 70)
    
    # Generate funnel
    funnel_stats = generate_funnel_for_patient(args.patient_id, args.output)
    
    if funnel_stats is None:
        print("\nFailed to generate funnel")
        return 1
    
    print("\nDone!")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
