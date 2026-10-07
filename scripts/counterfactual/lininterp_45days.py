"""
scripts/counterfactual/lininterp_45days.py
==========================================
Linear Interpolation Counterfactual — 45-Day Overview Figure
=============================================================

Generates 5-row 45-day overview comparing G(t) and C(t).
Shows original signal, counterfactual signal, and highlights SLH episodes
where counterfactual construction was applied.

No title is drawn inside the artwork.

Usage
-----
    python3 -m scripts.counterfactual.lininterp_45days                       # patient with the most SLH episodes
    python3 -m scripts.counterfactual.lininterp_45days --patient_id ID       # a given patient
    python3 -m scripts.counterfactual.lininterp_45days --offset 45 --out_dir DIR
    (or: make lininterp_45days)

Authors: Guillaume Deside

Output
------
    <CGM_DATA_ROOT>/Figures/Counterfactual_analysis/<patient_id>/overview_45d.{pdf,svg}
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D



# Import existing functions
from src.utils.patient_selection import resolve_patient_id
from src.preprocessing.load_data import load_patient_window, prepare_valid_data
from src.preprocessing.timestep import detect_dominant_sampling_interval
from src.episodes.detection import detect_hyperglycemia_episodes
from src.episodes.criteria import classify_episodes  
from src.episodes.episode_config import (
    H_HYPER,
    NIGHT_START_H, NIGHT_END_H,
    GAP_FACTOR,
    USE_GAP_MERGING, MERGE_GAP_MIN,
    LABEL_SLH,
)
from src.config import (
    TIME_COL, GLUCOSE_COL,
    FIG_DIR,
)

from src.visualization.style import (
    apply_lancet_style,
    enforce_lancet_minimums,
    save_figure,
    style_axes,
)
from src.visualization.visualization_helper import (
    shade_nights_on_axis,
    setup_relative_time_axis,
)

from src.counterfactual.build_counterfactual import build_full_counterfactual_signal  

from src.counterfactual.counterfactual_config import (
    APPLY_KERNEL,
    KERNEL_MIN,
    SELECTED_OFFSET_MIN,
)

from src.config.figure_config import (
    LANCET_MIN_FONT_PT,
    lancet_figsize,
)

from src.visualization.colors import (
COLOR_NIGHT, COLOR_ORIGINAL, COLOR_COUNTERFACTUAL, COLOR_THRESHOLD,COLOR_SLH_HIGHLIGHT
)
from src.visualization.glucose_axis import apply_glucose_axis
from src.utils.unit_conversion import format_glucose

# =============================================================================
# PARAMETERS 
# =============================================================================
FIGURES_OUTPUT_PATH = os.path.join(FIG_DIR, "Counterfactual_analysis")


ALPHA_SLH_HIGHLIGHT = 0.35        # Transparency for SLH shading

# ── Layout (full page width, 5 rows of 9 days) ────────────────────────
DAYS_PER_ROW = 9
ROWS = 5
ROW_HEIGHT_INCHES = 1.45

# =============================================================================
# 45-DAY OVERVIEW FIGURE
# =============================================================================
def generate_overview_figure(patient_id, df, all_episodes_df, slh_episodes, 
                             offset_min, timestep_min, output_prefix):
    """
    Generate 5-row 45-day overview comparing G(t) and C(t).
    
    Shows original signal, counterfactual signal, and highlights SLH episodes
    where counterfactual construction was applied.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier
    df : pd.DataFrame
        Patient data
    all_episodes_df : pd.DataFrame
        Classification DataFrame from classify_episodes() (for reference)
    slh_episodes : list of dict
        SLH episodes used for counterfactual construction
    offset_min : float
        Zone expansion offset δ
    timestep_min : float
        Sampling interval
    output_prefix : str
        Output path prefix without extension
    """
    apply_lancet_style()

    t_start = df[TIME_COL].iloc[0]
    row_starts = [t_start + pd.Timedelta(days=i * DAYS_PER_ROW) for i in range(ROWS)]
    row_ends = [rs + pd.Timedelta(days=DAYS_PER_ROW) for rs in row_starts]
    
    # Build full counterfactual signal
    c_signal = build_full_counterfactual_signal(df, slh_episodes, offset_min, timestep_min)
    
    fig, axes = plt.subplots(
        ROWS, 1,
        figsize=lancet_figsize(ROW_HEIGHT_INCHES * ROWS, columns="full"),
        constrained_layout=True,
    )

    # One glucose scale for every row: the rows are consecutive strips of the
    # same trace, so a per-row scale would make identical excursions look
    # different from one row to the next.
    g_all = df[GLUCOSE_COL].astype(float).dropna()
    y_min = max(40, float(g_all.min()) - 12) if not g_all.empty else 40
    y_max = min(450, float(g_all.max()) + 18) if not g_all.empty else 400

    
    for row_i, ax in enumerate(axes):
        rs, re = row_starts[row_i], row_ends[row_i]
        mask = (df[TIME_COL] >= rs) & (df[TIME_COL] < re)
        t_row = df.loc[mask, TIME_COL]
        g_row = df.loc[mask, GLUCOSE_COL].astype(float).copy()
        c_row = c_signal[mask.values].copy()
        
        # Insert NaN at gaps
        diffs = t_row.diff().dt.total_seconds().fillna(0) / 60
        gap_mask = diffs > timestep_min * GAP_FACTOR
        g_row[gap_mask] = np.nan
        c_row[gap_mask] = np.nan
        
        # Shade night periods
        flag_list = []
        shade_nights_on_axis(ax, rs, re, flag_list)
        
        # Highlight SLH episodes
        slh_labeled = False
        for ep in slh_episodes:
            if ep["offset_time"] < rs or ep["onset_time"] > re:
                continue
            
            label = "SLH episode" if not slh_labeled else "_"
            ax.axvspan(
                max(ep["onset_time"], rs),
                min(ep["offset_time"], re),
                color=COLOR_SLH_HIGHLIGHT,
                alpha=ALPHA_SLH_HIGHLIGHT,
                zorder=2,
                linewidth=0,
                label=label
            )
            slh_labeled = True
        
        # Plot threshold line
        ax.axhline(H_HYPER, color=COLOR_THRESHOLD, ls="--", lw=0.8,
                  alpha=0.65, zorder=3,
                  label=f"Hyperglycaemia threshold ({format_glucose(H_HYPER)})")
        
        # Plot glucose traces
        ax.plot(t_row, g_row, color=COLOR_ORIGINAL, lw=0.8, alpha=0.65,
               zorder=4, label="G(t) — original")
        ax.plot(t_row, c_row, color=COLOR_COUNTERFACTUAL, lw=1.0,
               zorder=5, label="C(t) — counterfactual")
        
        # Format axes
        ax.set_xlim(rs, re)
        ax.set_ylim(y_min, y_max)
        
        setup_relative_time_axis(ax, t_start, re, DAYS_PER_ROW)
        plt.setp(ax.xaxis.get_majorticklabels(),
                fontsize=LANCET_MIN_FONT_PT, rotation=0, ha="center")

        apply_glucose_axis(ax, fontsize=LANCET_MIN_FONT_PT, labelpad=2)
        ax.tick_params(labelsize=LANCET_MIN_FONT_PT)
        style_axes(ax)
        
        # Row label
        # ax.text(0.005, 0.97,
        #        f"Days {row_i*DAYS_PER_ROW+1}–{(row_i+1)*DAYS_PER_ROW}",
        #        transform=ax.transAxes, fontsize=6,
        #        color="#999999", va="top")
    
    # Build legend
    handles = [
        Line2D([0], [0], color=COLOR_ORIGINAL, lw=1.0, alpha=0.75,
              label="G(t) — original"),
        Line2D([0], [0], color=COLOR_COUNTERFACTUAL, lw=1.2,
              label="C(t) — counterfactual"),
        mpatches.Patch(facecolor=COLOR_SLH_HIGHLIGHT, alpha=ALPHA_SLH_HIGHLIGHT,
                      label=f"SLH episode (n={len(slh_episodes)})"),
        Line2D([0], [0], color=COLOR_THRESHOLD, ls="--", lw=0.9,
              label=f"Hyperglycaemia threshold ({format_glucose(H_HYPER)})"),
        mpatches.Patch(facecolor=COLOR_NIGHT, alpha=0.50,
                      label=f"Night ({NIGHT_START_H:02d}:00–{NIGHT_END_H:02d}:00)"),
    ]

    # "outside lower center" makes constrained_layout reserve room for the
    # legend inside the canvas, so the figure can be saved uncropped at
    # exactly the page width.
    fig.legend(handles=handles, loc="outside lower center", ncol=3,
              fontsize=LANCET_MIN_FONT_PT, frameon=False)

    fixed = enforce_lancet_minimums(fig)
    if any(fixed.values()):
        print(f"  Minimums enforced on {fixed['text']} text object(s) "
              f"and {fixed['lines']} line(s)")

    save_figure(fig, output_prefix, bbox_inches=None)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="45-day overview of the observed signal G(t) and the counterfactual C(t).",
    )
    parser.add_argument("--patient_id", default=None,
                        help="patient to show (default: the patient with the most SLH episodes)")
    parser.add_argument("--offset", type=float, default=SELECTED_OFFSET_MIN,
                        help=f"zone-expansion offset δ in minutes (default: {SELECTED_OFFSET_MIN:g})")
    parser.add_argument("--out_dir", default=FIGURES_OUTPUT_PATH,
                        help="output folder, a sub-folder per patient is created "
                             "(default: <CGM_DATA_ROOT>/Figures/Counterfactual_analysis)")
    return parser.parse_args(argv)


def main(argv=None):
    """Main execution pipeline."""
    args = parse_arguments(argv)
    offset_min, figures_dir = args.offset, args.out_dir
    merge_mode = "WITH gap merging" if USE_GAP_MERGING else "WITHOUT gap merging"
    
    print("=" * 70)
    print("  Linear Interpolation Counterfactual — 45-Day Overview")
    print("=" * 70)
    print(f"  Episode detection: {merge_mode}")
    if USE_GAP_MERGING:
        print(f"    MERGE_GAP_MIN = {MERGE_GAP_MIN} min")
    print(f"  Zone offset (δ): {offset_min:g} min")
    kernel_str = f"k={KERNEL_MIN} min" if APPLY_KERNEL else "no kernel"
    print(f"  Smoothing: {kernel_str}")
    print("=" * 70)
    
    # ═════════════════════════════════════════════════════════════════════════
    # REFACTORED: Classification now uses modular functions
    # ═════════════════════════════════════════════════════════════════════════
    
    # Load patient
    patient_id = resolve_patient_id(args.patient_id, role="most")
    df_raw = load_patient_window(patient_id)
    if df_raw is None:
        raise FileNotFoundError(f"Cannot load {patient_id}")
    df = prepare_valid_data(df_raw)
    
    # Detect episodes and classify
    raw_episodes = detect_hyperglycemia_episodes(df)
    classification_df = classify_episodes(df, raw_episodes)
    
    # Extract SLH episodes as list of dicts for counterfactual construction
    slh_df = classification_df[classification_df["classification"] == LABEL_SLH]
    slh_episodes = [
        {
            "onset_time": row["onset_time"],
            "offset_time": row["offset_time"],
        }
        for _, row in slh_df.iterrows()
    ]
    
    # Print summary
    timestep = detect_dominant_sampling_interval(df)
    print(f"\n  Patient: {patient_id}")
    print(f"  Timestep: {timestep:.0f} min")
    print(f"  Total episodes detected: {len(raw_episodes)}")
    print(f"  Episodes classified: {len(classification_df)}")
    print(f"  SLH episodes: {len(slh_episodes)}")
    
    # Generate figure
    os.makedirs(figures_dir, exist_ok=True)
    output_dir = os.path.join(figures_dir, patient_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # 45-day overview
    overview_prefix = os.path.join(output_dir, "overview_45d")
    generate_overview_figure(
        patient_id, df, classification_df, slh_episodes, 
        offset_min, timestep, overview_prefix
    )
    
    print("\n" + "=" * 70)
    print(f"Output: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()