"""
Linear Interpolation Counterfactual — Step-by-Step Figure
==========================================================
Generates 4-panel vertical figure showing counterfactual construction steps:
  Panel A — Original signal G(t)
  Panel B — Zone expansion + local-minima anchors
  Panel C — Linear interpolation L(t)
  Panel D — Final counterfactual C(t) with rolling-mean smoothing

No title is drawn inside the artwork.

Every glucose number drawn on the figure — axis ticks, anchor values, peak
and counterfactual annotations — is rendered in the current display unit, so
``CGM_GLUCOSE_UNIT=mmol/L`` produces a figure with no mg/dL left in it.

Usage:
    python -m scripts.counterfactual.lininterp_stepbystep

Authors: Guillaume Deside
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from src.utils.patient_selection import resolve_patient_id
from src.preprocessing.load_data import load_patient_window, prepare_valid_data
from src.preprocessing.timestep import detect_dominant_sampling_interval
from src.episodes.detection import detect_hyperglycemia_episodes
from src.episodes.criteria import classify_episodes

from src.episodes.episode_config import (
    H_HYPER,
    USE_GAP_MERGING,
    MERGE_GAP_MIN,
    MERGE_INTERNAL_GAP_MIN,
    LABEL_SLH,
)

from src.config import (
    TIME_COL, GLUCOSE_COL,
    PREPROCESSING_PATH,
    FIG_DIR,
)

from src.config.figure_config import (
    LANCET_MIN_FONT_PT,
    lancet_figsize,
)

from src.counterfactual.build_counterfactual import find_local_minimum, build_linear_interpolation, build_counterfactual

from src.visualization.style import (
    apply_lancet_style,
    enforce_lancet_minimums,
    save_figure,
)

from src.visualization.colors import (
    COLOR_ORIGINAL,
    COLOR_THRESHOLD,
    COLOR_ZONE,
    COLOR_COUNTERFACTUAL,
    COLOR_LININTERP,
    COLOR_ANCHOR,
    COLOR_EPISODE,
)

from src.visualization.visualization_helper import shade_nights_on_axis

from src.counterfactual.counterfactual_config import (
    APPLY_KERNEL,
    KERNEL_MIN,
    SELECTED_OFFSET_MIN,
)
from src.visualization.glucose_axis import apply_glucose_axis
from src.utils.unit_conversion import format_glucose, glucose_unit_label

# =============================================================================
# PARAMETERS 
# =============================================================================
FIGURES_OUTPUT_PATH = os.path.join(FIG_DIR, "Counterfactual_analysis")

# ── Patient selection ─────────────────────────────────────────────────────────
SELECTED_PID = None   # None = automatic (patient with the most SLH episodes); or set a patient ID

# ── Visualization parameters ──────────────────────────────────────────────────
PLOT_WINDOW_H = 3              # Hours of signal shown on each side of episode
N_EPISODES_TO_PLOT = 3         # Number of episodes to illustrate

# ── Type sizes (nothing below 7 pt at final size) ────────────────────
FONT_ANNOTATION = LANCET_MIN_FONT_PT        # In-plot annotations and legends
FONT_PANEL_TITLE = LANCET_MIN_FONT_PT + 1   # Panel headings
FONT_PANEL_LETTER = LANCET_MIN_FONT_PT + 2  # Panel letters (bold capitals)
PANEL_ROW_HEIGHT_INCHES = 1.45

# ─────────────────────────────────────────────────────────────────────────────
# EPISODE ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────
def enrich_slh_episodes_with_peak_info(df: pd.DataFrame, classification_df: pd.DataFrame, 
                                       raw_episodes: list) -> list[dict]:
    """
    Enrich SLH episodes with peak glucose and peak time information.
    
    Parameters
    ----------
    df : pd.DataFrame
        Patient data
    classification_df : pd.DataFrame
        Classification results from classify_episodes()
    raw_episodes : list
        Raw episodes from detect_hyperglycemia_episodes()
        
    Returns
    -------
    list of dict
        SLH episodes enriched with peak_glucose and peak_time
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP-BY-STEP FIGURE GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_stepbystep_figure(episode, episode_num, df, offset_min, 
                               timestep_min, patient_id, output_prefix):
    """
    Generate 4-panel vertical figure showing counterfactual construction steps.
    
    Panels:
      A — Original signal G(t)
      B — Zone expansion + local-minima anchors
      C — Linear interpolation L(t)
      D — Final counterfactual C(t)
    """
    apply_lancet_style()

    # Time window for plot
    window_delta = pd.Timedelta(hours=PLOT_WINDOW_H)
    t_start = episode["onset_time"] - window_delta
    t_end = episode["offset_time"] + window_delta
    
    mask = (df[TIME_COL] >= t_start) & (df[TIME_COL] <= t_end)
    t_window = df.loc[mask, TIME_COL]
    g_window = df.loc[mask, GLUCOSE_COL].astype(float).values
    
    # Build L(t) without kernel (raw linear interpolation)
    signal_lin, anchor_info = build_linear_interpolation(df, episode, offset_min, timestep_min)
    l_window = signal_lin[mask.values]
    
    # Build C(t) with kernel (if enabled)
    signal_ctf, _ = build_counterfactual(df, episode, offset_min, timestep_min)
    c_window = signal_ctf[mask.values]
    
    # Calculate y-limits
    y_min = max(50, float(np.nanmin(g_window)) - 15)
    y_max = min(420, float(np.nanmax(g_window)) + 25)
    y_range = y_max - y_min
    
    delta_td = pd.Timedelta(minutes=offset_min)
    
    # Create figure.  No title is drawn inside the artwork.
    fig, axes = plt.subplots(
        4, 1,
        figsize=lancet_figsize(PANEL_ROW_HEIGHT_INCHES * 4, columns="full"),
        constrained_layout=True,
    )

    
    def setup_background(ax, show_zones):
        """Add background shading and reference lines."""
        flag_list = []
        shade_nights_on_axis(ax, t_start, t_end, flag_list)
        
        if show_zones and offset_min > 0:
            # Left shoulder
            ax.axvspan(
                max(episode["onset_time"] - delta_td, t_start),
                min(episode["onset_time"], t_end),
                color=COLOR_ZONE, alpha=0.22, zorder=1, linewidth=0
            )
            # Right shoulder
            ax.axvspan(
                max(episode["offset_time"], t_start),
                min(episode["offset_time"] + delta_td, t_end),
                color=COLOR_ZONE, alpha=0.22, zorder=1, linewidth=0
            )
        
        # Episode core
        ax.axvspan(
            max(episode["onset_time"], t_start),
            min(episode["offset_time"], t_end),
            color=COLOR_EPISODE, alpha=0.30, zorder=1, linewidth=0
        )
        
        # Threshold line
        ax.axhline(H_HYPER, color=COLOR_THRESHOLD, ls="--", lw=0.9,
                  alpha=0.70, zorder=2)
        
        # Episode boundaries
        for tm in (episode["onset_time"], episode["offset_time"]):
            ax.axvline(tm, color="#777777", lw=0.6, alpha=0.50, zorder=3)
        
        ax.set_ylim(y_min, y_max)
        apply_glucose_axis(ax, fontsize=FONT_PANEL_TITLE, labelpad=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(length=3, width=0.8, labelsize=LANCET_MIN_FONT_PT)
    
    def add_panel_label(ax, label, x=-0.12, y=1.05):
        """Add panel letter to the top-left corner of an axis."""
        ax.text(x, y, label, transform=ax.transAxes,
                fontsize=FONT_PANEL_LETTER, fontweight="bold",
                va="top", ha="right")
    
    # ── Panel A: Original signal ──────────────────────────────────────────────
    ax = axes[0]
    setup_background(ax, show_zones=False)
    ax.plot(t_window, g_window, color=COLOR_ORIGINAL, lw=1.0, zorder=4)
    ax.scatter(
        [episode["peak_time"]], [episode["peak_glucose"]],
        color=COLOR_THRESHOLD, s=30, marker="^", zorder=9
    )
    # Left of the peak and clamped below the top of the axes: to its right sits
    # the descending limb, and above it the panel heading.
    ax.annotate(
        f"Peak {format_glucose(episode['peak_glucose'])}\n"
        f"{episode['duration_min']:.0f} min",
        xy=(episode["peak_time"], episode["peak_glucose"]),
        xytext=(episode["peak_time"] - pd.Timedelta(minutes=25),
               min(episode["peak_glucose"] + y_range * 0.08,
                   y_max - y_range * 0.09)),
        fontsize=FONT_ANNOTATION, color=COLOR_THRESHOLD, ha="right",
        arrowprops=dict(arrowstyle="->", color=COLOR_THRESHOLD, lw=0.7),
        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                 ec=COLOR_THRESHOLD, lw=0.6, alpha=0.90),
    )
    add_panel_label(ax, "A")
    ax.set_title("Original CGM signal  G(t)", fontsize=FONT_PANEL_TITLE,
                 fontweight="bold", loc="left", pad=4)
    
    plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
    
    # ── Panel B: Zone expansion + anchors ─────────────────────────────────────
    ax = axes[1]
    setup_background(ax, show_zones=True)
    ax.plot(t_window, g_window, color=COLOR_ORIGINAL, lw=0.8, alpha=0.50, zorder=4)
    
    if anchor_info:
        ax.scatter(
            [anchor_info["t_a"], anchor_info["t_b"]],
            [anchor_info["g_a"], anchor_info["g_b"]],
            color=COLOR_ANCHOR, s=55, marker="v", zorder=9,
            label="Local-min anchor"
        )
        
        for (t_anc, g_anc, tag) in [
            (anchor_info["t_a"], anchor_info["g_a"], "La"),
            (anchor_info["t_b"], anchor_info["g_b"], "Lb")
        ]:
            ax.annotate(
                f"{tag} = {format_glucose(g_anc, with_unit=False)}",
                xy=(t_anc, g_anc),
                xytext=(t_anc + pd.Timedelta(minutes=15),
                       g_anc + y_range * 0.11),
                fontsize=FONT_ANNOTATION, color=COLOR_ANCHOR,
                arrowprops=dict(arrowstyle="->", color=COLOR_ANCHOR, lw=0.7)
            )
    
    add_panel_label(ax, "B")
    ax.set_title(f"Contextual windows [δ = {offset_min:g} min] and "
                 f"local-minimum anchors",
                 fontsize=FONT_PANEL_TITLE, fontweight="bold", loc="left", pad=4)

    plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
    
    # ── Panel C: Linear interpolation ─────────────────────────────────────────
    ax = axes[2]
    setup_background(ax, show_zones=True)
    ax.plot(t_window, g_window, color=COLOR_ORIGINAL, lw=0.8, alpha=0.40, zorder=4)
    ax.plot(t_window, l_window, color=COLOR_LININTERP, lw=1.4, ls="--", zorder=6)
    
    if anchor_info:
        ax.scatter(
            [anchor_info["t_a"], anchor_info["t_b"]],
            [anchor_info["g_a"], anchor_info["g_b"]],
            color=COLOR_ANCHOR, s=55, marker="v", zorder=9
        )
    
    add_panel_label(ax, "C")
    ax.set_title("Linear interpolation baseline  L(t)",
                 fontsize=FONT_PANEL_TITLE, fontweight="bold", loc="left", pad=4)

    plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
    
    # ── Panel D: Final counterfactual ─────────────────────────────────────────
    ax = axes[3]
    setup_background(ax, show_zones=True)
    ax.plot(t_window, g_window, color=COLOR_ORIGINAL, lw=0.8, alpha=0.35, zorder=4)
    ax.plot(t_window, l_window, color=COLOR_LININTERP, lw=0.8, ls="--",
           alpha=0.50, zorder=5)
    ax.plot(t_window, c_window, color=COLOR_COUNTERFACTUAL, lw=1.4, zorder=7)
    
    # Δ annotation at peak
    ts_float = t_window.values.astype("datetime64[s]").astype(np.float64)
    c_peak = float(np.interp(
        np.float64(episode["peak_time"].timestamp()),
        ts_float, c_window
    ))
    delta_val = episode["peak_glucose"] - c_peak
    
    ax.annotate(
        f"G = {format_glucose(episode['peak_glucose'], with_unit=False)}",
        xy=(episode["peak_time"], episode["peak_glucose"]),
        xytext=(episode["peak_time"] + pd.Timedelta(minutes=20),
               min(episode["peak_glucose"] + y_range * 0.07,
                   y_max - y_range * 0.07)),
        fontsize=FONT_ANNOTATION, color=COLOR_ORIGINAL,
        arrowprops=dict(arrowstyle="->", color=COLOR_ORIGINAL, lw=0.7)
    )
    # Δ is a difference: mg/dL and mmol/L differ by a scale factor with no
    # offset, so the same conversion applies to it as to a concentration.
    ax.annotate(
        f"C = {format_glucose(c_peak, with_unit=False)}  "
        f"(Δ = −{format_glucose(delta_val)})",
        xy=(episode["peak_time"], c_peak),
        xytext=(episode["peak_time"], c_peak - y_range * 0.24),
        fontsize=FONT_ANNOTATION, color=COLOR_COUNTERFACTUAL, ha="center",
        arrowprops=dict(arrowstyle="->", color=COLOR_COUNTERFACTUAL, lw=0.7),
        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                 ec=COLOR_COUNTERFACTUAL, lw=0.6, alpha=0.90)
    )

    d_title = (
        f"Counterfactual  C(t)  [{KERNEL_MIN} min centred rolling mean]"
        if APPLY_KERNEL else
        "Counterfactual  C(t)  [pure linear interpolation, no smoothing]"
    )
    add_panel_label(ax, "D")
    ax.set_title(d_title, fontsize=FONT_PANEL_TITLE, fontweight="bold",
                 loc="left", pad=4)

    # Hourly clock labels, horizontal: at full page width they fit without the
    # rotation, which house style avoids.
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(),
            fontsize=LANCET_MIN_FONT_PT, rotation=0, ha="center")
    ax.set_xlabel("Time of day (h:min)", fontsize=LANCET_MIN_FONT_PT, labelpad=3)

    axes[0].set_xlim(t_start, t_end)

    # One key for the whole figure rather than four boxes floating over the
    # panels, where they collided with the peak annotation and the headings.
    # "outside lower center" makes constrained_layout reserve the space, so
    # the figure saves uncropped at exactly the page width.
    figure_handles = [
        Line2D([0], [0], color=COLOR_ORIGINAL, lw=1.0, label="G(t) — observed"),
        Line2D([0], [0], color=COLOR_LININTERP, lw=1.4, ls="--",
              label="L(t) — linear interpolation"),
        Line2D([0], [0], color=COLOR_COUNTERFACTUAL, lw=1.4,
              label="C(t) — counterfactual"),
        Line2D([0], [0], color=COLOR_THRESHOLD, ls="--", lw=0.9,
              label=f"Hyperglycaemia threshold ({format_glucose(H_HYPER)})"),
        mpatches.Patch(facecolor=COLOR_EPISODE, alpha=0.40,
                      label="SLH episode"),
        Line2D([0], [0], color=COLOR_ANCHOR, ls="none", marker="v",
              markersize=5, label="Local-minimum anchor"),
    ]
    if offset_min > 0:
        figure_handles.append(
            mpatches.Patch(facecolor=COLOR_ZONE, alpha=0.40,
                          label=f"Contextual window (δ = {offset_min:g} min)")
        )

    fig.legend(handles=figure_handles, loc="outside lower center", ncol=4,
              fontsize=FONT_ANNOTATION, frameon=False)

    fixed = enforce_lancet_minimums(fig)
    if any(fixed.values()):
        print(f"      Minimums enforced on {fixed['text']} text "
              f"object(s) and {fixed['lines']} line(s)")

    save_figure(fig, output_prefix, bbox_inches=None)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Main execution pipeline."""
    merge_mode = "WITH gap merging" if USE_GAP_MERGING else "WITHOUT gap merging"
    
    print("=" * 70)
    print("  Linear Interpolation Counterfactual — Step-by-Step Figure")
    print("=" * 70)
    print(f"  Episode detection: {merge_mode}")
    if USE_GAP_MERGING:
        print(f"    MERGE_GAP_MIN = {MERGE_GAP_MIN} min")
        print(f"    MERGE_INTERNAL_GAP_MIN = {MERGE_INTERNAL_GAP_MIN} min")
    print(f"  Zone offset (δ): {SELECTED_OFFSET_MIN} min")
    kernel_str = f"k={KERNEL_MIN} min" if APPLY_KERNEL else "no kernel"
    print(f"  Smoothing: {kernel_str}")
    print("=" * 70)
    
    # Load patient
    patient_id = resolve_patient_id(SELECTED_PID, role="most")
    df_raw = load_patient_window(patient_id)
    if df_raw is None:
        raise FileNotFoundError(f"Cannot load {patient_id}")
    df = prepare_valid_data(df_raw)
    
    # Detect episodes and classify
    timestep = detect_dominant_sampling_interval(df)
    raw_episodes = detect_hyperglycemia_episodes(df)
    classification_df = classify_episodes(df, raw_episodes)
    
    # Enrich SLH episodes with peak information for visualization
    slh_episodes = enrich_slh_episodes_with_peak_info(df, classification_df, raw_episodes)
    
    # Print summary
    print(f"\n  Patient: {patient_id}")
    print(f"  Timestep: {timestep:.0f} min")
    print(f"  Raw episodes detected: {len(raw_episodes)}")
    print(f"  Episodes classified: {len(classification_df)}")
    print(f"  SLH episodes: {len(slh_episodes)}")
    
    # Select episodes to illustrate (with sufficient margins)
    window_delta = pd.Timedelta(hours=PLOT_WINDOW_H)
    zone_delta = pd.Timedelta(minutes=SELECTED_OFFSET_MIN)
    
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
    output_dir = os.path.join(FIGURES_OUTPUT_PATH, patient_id)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n  Generating step-by-step figures...")
    for ep_i, ep in enumerate(episodes_to_plot, 1):
        print(f"    Episode {ep_i}/{n_selected}  "
              f"({ep['onset_time'].strftime('%-d %b %H:%M')}, "
              f"{ep['duration_min']:.0f} min)")
        
        output_prefix = os.path.join(output_dir, f"steps_ep{ep_i:02d}")
        generate_stepbystep_figure(
            ep, ep_i, df, SELECTED_OFFSET_MIN, timestep, patient_id, output_prefix
        )
    
    print("\n" + "=" * 70)
    print(f"Output: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()