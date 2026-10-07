"""
scripts/counterfactual/lininterp_offset_comp.py
===============================================
Linear Interpolation Counterfactual — Offset Comparison Figures
================================================================
Generates figures comparing different zone expansion offset (δ) values.

Usage:
    python3 -m scripts.counterfactual.lininterp_offset_comp

Authors: Guillaume Deside

Output
------
    <CGM_DATA_ROOT>/Figures/Counterfactual_analysis/<patient_id>/multipanel_epNN.{pdf,svg} and overlay_epNN.{pdf,svg}
"""
# =============================================================================
# STANDARD LIBRARY
# =============================================================================
import os

# =============================================================================
# THIRD-PARTY LIBRARIES
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =============================================================================
# PROJECT IMPORTS — Data & Processing
# =============================================================================
from src.utils.patient_selection import resolve_patient_id
from src.preprocessing.load_data import load_patient_window, prepare_valid_data
from src.preprocessing.timestep import detect_dominant_sampling_interval
from src.episodes.detection import detect_hyperglycemia_episodes
from src.episodes.criteria import classify_episodes
from src.counterfactual.build_counterfactual import build_full_counterfactual_signal

# =============================================================================
# PROJECT IMPORTS — Configuration
# =============================================================================
from src.config import (
    TIME_COL, 
    GLUCOSE_COL,
    FIG_DIR,
)

from src.config.figure_config import FIGURE_WIDTH_INCHES

from src.episodes.episode_config import (
    H_HYPER,
    USE_GAP_MERGING, 
    LABEL_SLH,
)

from src.counterfactual.counterfactual_config import (
    APPLY_KERNEL,
    KERNEL_MIN,
    OFFSET_VALUES,
)

# =============================================================================
# PROJECT IMPORTS — Visualization
# =============================================================================
from src.visualization.style import apply_publication_style, save_figure, style_axes
from src.visualization.colors import (
    OFFSET_COLORS, 
    COLOR_ORIGINAL,
    COLOR_THRESHOLD,
    COLOR_ZONE,
    COLOR_SLH_HIGHLIGHT,
)

from src.visualization.visualization_helper import (
    shade_nights_on_axis,
)
from src.visualization.glucose_axis import apply_glucose_axis
from src.utils.unit_conversion import format_glucose
# =============================================================================
# PARAMETERS 
# =============================================================================
# ── Paths ─────────────────────────────────────────────────────────────────────
FIGURES_OUTPUT_PATH = os.path.join(FIG_DIR, "Counterfactual_analysis")

# ── Patient selection ─────────────────────────────────────────────────────────
SELECTED_PID = None   # None = automatic (patient with the most SLH episodes); or set a patient ID

# ── Visualization parameters ──────────────────────────────────────────────────
PLOT_WINDOW_H = 3              # Hours of signal shown on each side of episode
N_EPISODES_TO_PLOT = 3         # Number of episodes to illustrate


def enrich_slh_episodes_with_peak_info(df: pd.DataFrame, classification_df: pd.DataFrame, 
                                       raw_episodes: list) -> list[dict]:
    """
    Enrich SLH episodes with peak glucose and peak time information.
    
    Parameters
    ----------
    df : pd.DataFrame
        Patient data with TIME_COL and GLUCOSE_COL
    classification_df : pd.DataFrame
        Classification results from classify_episodes()
    raw_episodes : list[dict]
        Raw episodes from detect_hyperglycemia_episodes()
        
    Returns
    -------
    list[dict]
        SLH episodes enriched with peak_glucose and peak_time fields
    """
    # Filter for SLH episodes
    slh_df = classification_df[classification_df["classification"] == LABEL_SLH].copy()
    
    enriched_episodes = []
    
    for _, row in slh_df.iterrows():
        # Find matching raw episode to get indices
        matching_ep = None
        for ep in raw_episodes:
            if ep["onset_time"] == row["onset_time"] and ep["offset_time"] == row["offset_time"]:
                matching_ep = ep
                break
        
        if matching_ep is None:
            # Fallback: find peak in time window directly
            mask = (df[TIME_COL] >= row["onset_time"]) & (df[TIME_COL] <= row["offset_time"])
            ep_data = df[mask]
            
            if len(ep_data) == 0:
                continue
            
            peak_idx = ep_data[GLUCOSE_COL].idxmax()
            peak_glucose = float(ep_data.loc[peak_idx, GLUCOSE_COL])
            peak_time = ep_data.loc[peak_idx, TIME_COL]
        else:
            # Use indices from raw episode
            ep_data = df.loc[matching_ep["idx"]]
            
            if ep_data[GLUCOSE_COL].dropna().empty:
                continue
            
            peak_idx = ep_data[GLUCOSE_COL].dropna().idxmax()
            peak_glucose = float(df.loc[peak_idx, GLUCOSE_COL])
            peak_time = df.loc[peak_idx, TIME_COL]
        
        # Build enriched episode dict
        enriched_episodes.append({
            "onset_time": row["onset_time"],
            "offset_time": row["offset_time"],
            "duration_min": row["duration_min"],
            "peak_glucose": peak_glucose,
            "peak_time": peak_time,
        })
    
    return enriched_episodes
 
 
# =============================================================================
# FIGURE GENERATION FUNCTIONS
# =============================================================================

def generate_offset_comparison_multipanel(episode, episode_num, df, offsets, 
                                         timestep_min, patient_id, output_prefix):
    """
    Generate multi-panel figure comparing different offset values.
    
    Creates separate panels for each offset value plus an overlay panel.
    
    Parameters
    ----------
    episode : dict
        Episode dictionary with onset_time, offset_time, duration_min, peak_glucose, peak_time
    episode_num : int
        Episode number for labeling
    df : pd.DataFrame
        Patient data
    offsets : list[int]
        Offset values (δ) in minutes to compare
    timestep_min : float
        Dominant sampling interval in minutes
    patient_id : str
        Patient identifier
    output_prefix : str
        Output file path without extension
    """
    apply_publication_style()
    
    window_delta = pd.Timedelta(hours=PLOT_WINDOW_H)
    t_start = episode["onset_time"] - window_delta
    t_end = episode["offset_time"] + window_delta
    
    mask = (df[TIME_COL] >= t_start) & (df[TIME_COL] <= t_end)
    t_window = df.loc[mask, TIME_COL]
    g_window = df.loc[mask, GLUCOSE_COL].astype(float).values
    
    y_min = max(50, float(np.nanmin(g_window)) - 15)
    y_max = min(420, float(np.nanmax(g_window)) + 25)
    y_range = y_max - y_min
    
    kernel_str = f"k={KERNEL_MIN} min" if APPLY_KERNEL else "no kernel"
    
    n_panels = len(offsets) + 1
    n_cols = min(n_panels, 3)
    n_rows = int(np.ceil(n_panels / n_cols))
    panel_w = FIGURE_WIDTH_INCHES / n_cols
    panel_h = panel_w * 0.90
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIGURE_WIDTH_INCHES, panel_h * n_rows),
                            constrained_layout=True)
    axes_flat = np.array(axes).reshape(-1)
    
    # Build counterfactuals for each offset
    cfacts = {}
    for offset in offsets:
        cfacts[offset] = build_full_counterfactual_signal(df, [episode], offset, timestep_min)[mask.values]
    
    fig.suptitle(
        f"Participant {patient_id}  ·  Ep. {episode_num}  ·  "
        f"{episode['duration_min']:.0f} min  ·  "
        f"{kernel_str}  ·  Comparison across δ",
        fontsize=6.5, color="#555555",
    )
    
    def draw_panel(ax, offset_val, c_array):
        """Draw individual panel for one offset value."""
        color = OFFSET_COLORS.get(offset_val, "#333333")
        delta_td = pd.Timedelta(minutes=offset_val)
        
        labeled_list = []
        shade_nights_on_axis(ax, t_start, t_end, labeled_list)
        
        # Zone shading
        if offset_val > 0:
            ax.axvspan(max(episode["onset_time"] - delta_td, t_start), episode["onset_time"],
                      color=COLOR_ZONE, alpha=0.18, zorder=1, linewidth=0)
            ax.axvspan(episode["offset_time"], min(episode["offset_time"] + delta_td, t_end),
                      color=COLOR_ZONE, alpha=0.18, zorder=1, linewidth=0)
        
        # Episode core
        ax.axvspan(episode["onset_time"], episode["offset_time"],
                  color=COLOR_SLH_HIGHLIGHT, alpha=0.25, zorder=1, linewidth=0)
        ax.axhline(H_HYPER, color=COLOR_THRESHOLD, ls="--", lw=0.8, alpha=0.65, zorder=2)
        
        # Signals
        ax.plot(t_window, g_window, color=COLOR_ORIGINAL, lw=0.8, alpha=0.40, zorder=3, label="G(t)")
        ax.plot(t_window, c_array, color=color, lw=1.4, zorder=5, label=f"C(t)  δ = {offset_val} min")
        
        # Peak annotation
        ts_float = t_window.values.astype("datetime64[s]").astype(np.float64)
        c_peak = float(np.interp(np.float64(episode["peak_time"].timestamp()), ts_float, c_array))
        delta_val = episode["peak_glucose"] - c_peak
        
        ax.scatter([episode["peak_time"]], [c_peak], color=color, s=25, marker="o", zorder=8)
        ax.text(episode["peak_time"], c_peak - y_range * 0.06, f"−{delta_val:.0f}",
               fontsize=6, color=color, ha="center", va="top", fontweight="bold")
        
        # Formatting
        ax.set_ylim(y_min, y_max)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.setp(ax.xaxis.get_majorticklabels(), fontsize=6.5, rotation=30, ha="right")
        apply_glucose_axis(ax, fontsize=7, labelpad=2)
        style_axes(ax)
        ax.legend(fontsize=6.5, loc="upper right", framealpha=0.90)
        ax.set_title(f"δ = {offset_val} min", fontsize=8, fontweight="bold", loc="left", pad=4)
    
    # Draw individual offset panels
    for idx, offset in enumerate(offsets):
        draw_panel(axes_flat[idx], offset, cfacts[offset])
    
    # Draw overlay panel
    ax_overlay = axes_flat[len(offsets)]
    labeled_list = []
    shade_nights_on_axis(ax_overlay, t_start, t_end, labeled_list)
    ax_overlay.axvspan(episode["onset_time"], episode["offset_time"],
                      color=COLOR_SLH_HIGHLIGHT, alpha=0.22, zorder=1, linewidth=0)
    ax_overlay.axhline(H_HYPER, color=COLOR_THRESHOLD, ls="--", lw=0.8, alpha=0.65, zorder=2)
    ax_overlay.plot(t_window, g_window, color=COLOR_ORIGINAL, lw=1.0, alpha=0.55, zorder=3, label="G(t)")
    
    for offset in offsets:
        color = OFFSET_COLORS.get(offset, "#333333")
        ax_overlay.plot(t_window, cfacts[offset], color=color, lw=1.2, zorder=5,
                       label=f"C(t)  δ = {offset} min")
    
    ax_overlay.set_ylim(y_min, y_max)
    ax_overlay.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax_overlay.xaxis.get_majorticklabels(), fontsize=6.5, rotation=30, ha="right")
    apply_glucose_axis(ax_overlay, fontsize=7, labelpad=2)
    style_axes(ax_overlay)
    ax_overlay.legend(fontsize=6.5, loc="upper right", framealpha=0.90)
    ax_overlay.set_title("All δ values — overlay", fontsize=8, fontweight="bold", loc="left", pad=4)
    
    # Hide unused panels
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)
    
    save_figure(fig, output_prefix)
 
 
def generate_offset_comparison_overlay(episode, episode_num, df, offsets, 
                                       timestep_min, patient_id, output_prefix):
    """
    Generate single-panel overlay figure comparing different offset values.
    
    All offset values shown on the same panel for direct comparison.
    
    Parameters
    ----------
    episode : dict
        Episode dictionary with onset_time, offset_time, duration_min, peak_glucose, peak_time
    episode_num : int
        Episode number for labeling
    df : pd.DataFrame
        Patient data
    offsets : list[int]
        Offset values (δ) in minutes to compare
    timestep_min : float
        Dominant sampling interval in minutes
    patient_id : str
        Patient identifier
    output_prefix : str
        Output file path without extension
    """
    apply_publication_style()
    
    window_delta = pd.Timedelta(hours=PLOT_WINDOW_H)
    t_start = episode["onset_time"] - window_delta
    t_end = episode["offset_time"] + window_delta
    
    mask = (df[TIME_COL] >= t_start) & (df[TIME_COL] <= t_end)
    t_window = df.loc[mask, TIME_COL]
    g_window = df.loc[mask, GLUCOSE_COL].astype(float).values
    
    y_min = max(50, float(np.nanmin(g_window)) - 15)
    y_max = min(420, float(np.nanmax(g_window)) + 25)
    
    kernel_str = f"k={KERNEL_MIN} min" if APPLY_KERNEL else "no kernel"
    
    # Build counterfactuals for each offset
    cfacts = {}
    for offset in offsets:
        cfacts[offset] = build_full_counterfactual_signal(df, [episode], offset, timestep_min)[mask.values]
    
    fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_WIDTH_INCHES * 0.45),
                          constrained_layout=True)
    
    fig.suptitle(
        f"Participant {patient_id}  ·  Ep. {episode_num}  ·  "
        f"{episode['duration_min']:.0f} min  ·  "
        f"peak {format_glucose(episode['peak_glucose'])}  ·  "
        f"{kernel_str}",
        fontsize=7, fontweight="bold",
    )
    
    labeled_list = []
    shade_nights_on_axis(ax, t_start, t_end, labeled_list)
    
    # Background elements
    ax.axvspan(episode["onset_time"], episode["offset_time"],
              color=COLOR_SLH_HIGHLIGHT, alpha=0.25, zorder=1, linewidth=0, label="SLH episode")
    ax.axhline(H_HYPER, color=COLOR_THRESHOLD, ls="--", lw=1.0, alpha=0.75, zorder=2,
              label=f"H = {format_glucose(H_HYPER)}")
    
    # Original signal
    ax.plot(t_window, g_window, color=COLOR_ORIGINAL, lw=1.2, alpha=0.65, zorder=3, label="G(t) — original")
    
    # Counterfactual signals for each offset
    for offset in offsets:
        color = OFFSET_COLORS.get(offset, "#333333")
        ax.plot(t_window, cfacts[offset], color=color, lw=1.4, zorder=5,
               label=f"C(t)  δ = {offset} min")
    
    # Formatting
    ax.set_xlim(t_start, t_end)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=7, rotation=30, ha="right")
    apply_glucose_axis(ax, fontsize=8, labelpad=3)
    
    style_axes(ax)
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.92, ncol=2)
    
    save_figure(fig, output_prefix)
 

# =============================================================================
# MAIN EXECUTION
# =============================================================================
 
def main():
    """Main execution pipeline."""
    merge_mode = "WITH gap merging" if USE_GAP_MERGING else "WITHOUT gap merging"
    
    print("=" * 70)
    print("  Linear Interpolation — Offset Comparison Figures")
    print("=" * 70)
    print(f"  Episode detection: {merge_mode}")
    kernel_str = f"k={KERNEL_MIN} min" if APPLY_KERNEL else "no kernel"
    print(f"  Smoothing: {kernel_str}")
    print(f"  Offsets (δ): {OFFSET_VALUES} min")
    print("=" * 70)
    
    # Load patient data
    selected_pid = resolve_patient_id(SELECTED_PID, role="most")
    df_raw = load_patient_window(selected_pid)
    if df_raw is None:
        raise FileNotFoundError(f"Cannot load {selected_pid}")
    
    df = prepare_valid_data(df_raw)
    timestep = detect_dominant_sampling_interval(df)
    
    # Detect and classify episodes
    raw_episodes = detect_hyperglycemia_episodes(df)
    classification_df = classify_episodes(df, raw_episodes)
    
    # Enrich SLH episodes with peak information for visualization
    slh_episodes = enrich_slh_episodes_with_peak_info(df, classification_df, raw_episodes)
    
    print(f"\n  Patient: {selected_pid}")
    print(f"  Timestep: {timestep:.0f} min")
    print(f"  Raw episodes detected: {len(raw_episodes)}")
    print(f"  Episodes classified: {len(classification_df)}")
    print(f"  SLH episodes: {len(slh_episodes)}")
    
    # Select episodes with sufficient margins
    window_delta = pd.Timedelta(hours=PLOT_WINDOW_H)
    zone_delta = pd.Timedelta(minutes=max(OFFSET_VALUES))
    
    eligible = [
        ep for ep in slh_episodes
        if ((ep["onset_time"] - window_delta >= df[TIME_COL].iloc[0]) and
            (ep["offset_time"] + window_delta <= df[TIME_COL].iloc[-1]) and
            (ep["onset_time"] - zone_delta >= df[TIME_COL].iloc[0]) and
            (ep["offset_time"] + zone_delta <= df[TIME_COL].iloc[-1]))
    ]
    
    n_selected = min(N_EPISODES_TO_PLOT, len(eligible))
    
    if n_selected == 0:
        print("  No eligible SLH episodes with sufficient margins.")
        return
    
    step = len(eligible) / n_selected
    episodes_to_plot = [eligible[int(i * step)] for i in range(n_selected)]
    
    print(f"  Eligible episodes: {len(eligible)}")
    print(f"  Selected for figures: {n_selected}")
    
    # Generate figures
    os.makedirs(FIGURES_OUTPUT_PATH, exist_ok=True)
    output_dir = os.path.join(FIGURES_OUTPUT_PATH, selected_pid)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n  Generating offset comparison figures...")
    for ep_i, ep in enumerate(episodes_to_plot, 1):
        print(f"\n    Episode {ep_i}/{n_selected}  "
              f"({ep['onset_time'].strftime('%-d %b %H:%M')}, {ep['duration_min']:.0f} min)")
        
        print(f"      Multi-panel figure...")
        output_prefix = os.path.join(output_dir, f"multipanel_ep{ep_i:02d}")
        generate_offset_comparison_multipanel(ep, ep_i, df, OFFSET_VALUES, timestep, selected_pid, output_prefix)
        
        print(f"      Overlay figure...")
        output_prefix = os.path.join(output_dir, f"overlay_ep{ep_i:02d}")
        generate_offset_comparison_overlay(ep, ep_i, df, OFFSET_VALUES, timestep, selected_pid, output_prefix)
    
    print("\n" + "=" * 70)
    print(f"Output: {output_dir}/")
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()