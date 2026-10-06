"""
scripts/preprocessing/cgm_timestep_analysis_raw.py
==================================================

CGM Raw Timestep Distribution (Pre-Preprocessing)
-------------------------------------------------
Generates a figure showing the distribution of dominant CGM sampling
intervals in raw data files, before any preprocessing or window selection.

This figure is intended for the **preprocessing / data quality** section of the manuscript.

Outputs
-------
- Figures/Preprocessing/timestep_distribution_raw.{png,pdf,svg}

Usage
-----
python3 -m scripts.preprocessing.cgm_timestep_analysis_raw

Author
------
Guillaume Deside

Date
----
2026-04-15
"""

# =============================================================================
# LIBRARIES
# =============================================================================
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

# =============================================================================
# PROJECT IMPORTS
# =============================================================================
# Configuration (scientific + figure constants)
from src.config.constants import (
    PATIENT_START,
    PATIENT_END,
)
from src.config.figure_config import (
    FIG_PREPROCESSING_DIR,
)

# Core analysis functions
from src.preprocessing.timestep import (
    snap_to_canonical,
    dominant_step_of_series,
)
from src.preprocessing.load_data import load_raw_time_column

# Visualization
from src.visualization.colors import C_5MIN, C_15MIN
from src.visualization.style import (
    apply_publication_style,
    style_axes,
    save_figure,
)


# =============================================================================
# FIGURE GENERATION
# =============================================================================
def make_raw_timestep_figure(raw_n: dict, output_dir: str | Path) -> None:
    """
    Create a single-panel bar chart showing the distribution of
    raw CGM sampling intervals (publication-ready).

    Parameters
    ----------
    raw_n : dict
        Counts of dominant timesteps (e.g., {5: n, 15: n, "other": n})
    output_dir : str | Path
        Directory where the figure will be saved
    """
    apply_publication_style()

    # Define timestep categories
    step_labels = {5: "5 min", 15: "15 min"}
    step_colors = {5: C_5MIN, 15: C_15MIN}

    active_steps = [s for s in (5, 15) if raw_n.get(s, 0) > 0]
    if raw_n.get("other", 0) > 0:
        active_steps.append("other")

    labels = [step_labels.get(s, "Other") for s in active_steps]
    colors = [step_colors.get(s, "#999999") for s in active_steps]
    values = [raw_n.get(s, 0) for s in active_steps]

    x = np.arange(len(active_steps))

    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.96, bottom=0.14)

    bars = ax.bar(
        x,
        values,
        width=0.55,
        color=colors,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )

    # Add count labels above bars
    y_top = max(values) * 1.15 if values else 1
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_top * 0.02,
                f"{val}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                color="#333333",
            )

    # Styling
    style_axes(ax, xlabel="", ylabel="Number of participants", title="")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.set_ylim(0, y_top)

    # Save
    save_figure(
        fig,
        os.path.join(output_dir, "timestep_distribution_raw")
    )
    plt.close(fig)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main() -> None:
    """
    Main execution:
    - Loads raw CGM timestamps for the full cohort
    - Computes dominant timestep per participant
    - Aggregates distribution
    - Generates and saves the publication-ready figure
    """
    # Ensure output directory exists
    os.makedirs(FIG_PREPROCESSING_DIR, exist_ok=True)

    # Patient cohort
    patient_ids = [
        f"A{n:03d}" for n in range(PATIENT_START, PATIENT_END + 1)
    ]

    print("=" * 70)
    print("  CGM RAW TIMESTEP DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print(f"  Patients analysed : A{PATIENT_START:03d}–A{PATIENT_END:03d} "
          f"({len(patient_ids)} files)")

    # Compute dominant timestep per patient
    raw_steps = {}
    print("  Processing raw CGM files...")

    for pid in tqdm(patient_ids, desc="Patients", unit="patient", ncols=80):
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

    # Aggregate statistics
    valid_steps = [s for s in raw_steps.values() if s is not None]

    raw_n = {
        5: valid_steps.count(5),
        10: valid_steps.count(10),
        15: valid_steps.count(15),
        "other": len(valid_steps)
        - valid_steps.count(5)
        - valid_steps.count(10)
        - valid_steps.count(15),
    }

    n_parseable = len(valid_steps)
    n_missing = len(patient_ids) - n_parseable

    print(f"  Parseable raw files : {n_parseable}")
    print(f"  Files missing / unreadable : {n_missing}")
    print(f"  5-min  : {raw_n[5]:3d}")
    print(f"  15-min : {raw_n[15]:3d}")
    if raw_n["other"] > 0:
        print(f"  Other  : {raw_n['other']:3d}")

    # Generate figure
    make_raw_timestep_figure(raw_n, FIG_PREPROCESSING_DIR)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()