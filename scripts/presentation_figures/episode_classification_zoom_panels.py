"""
scripts/presentation_figures/episode_classification_zoom_panels.py
==================================================================

Generate zoom panels from generate_zoom_panels.py as individual and grouped figures.

For a given patient, one representative episode per classification stage
(C1–C6 + SLH) is selected.  Figures are saved in three forms:

  1. Individual panels — one panel per classification stage (7 separate figures)
  2. C4 + C5 side-by-side — two-panel figure comparing C4 and C5 episodes
  3. C1 + C2 + C3 side-by-side — three-panel figure comparing C1, C2, and C3 episodes
  4. C1 + C3 side-by-side — two-panel figure comparing C1 and C3 (no C2)

All panels use the same styling and presentation standards, with titles showing
only the episode description (no "C#" prefix).

Usage
-----
    python3 -m scripts.presentation_figures.episode_classification_zoom_panels [--patient_id ID]
                                                          [--out_dir DIR]

Output
------
    scripts/presentation_figures/output/episode_classification_zoom_panels/  (git-ignored; --out_dir to change)
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Episode detection & classification (reuses generate_zoom_panels logic)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.generate_zoom_panels import (
    classify_episodes_for_visualization,
    select_representative_episodes,
    DRAW_ORDER,
    STAGE_LABELS,
    ZOOM_WINDOW_H,
    ALPHA_EPISODE,
    ALPHA_NIGHT,
    _shade_nights,
)

from src.utils.patient_selection import resolve_patient_id
from src.preprocessing.load_data import load_patient_window
from src.preprocessing.timestep import detect_dominant_sampling_interval
from src.episodes.detection import detect_hyperglycemia_episodes
from src.episodes.episode_config import (
    H_HYPER, H_CONTEXT, CONTEXT_WIN_MIN, GAP_FACTOR,
)
from src.config import TIME_COL, GLUCOSE_COL, FIG_DIR
from src.visualization.style import apply_publication_style, save_figure
from src.visualization.colors import (
    STAGE_COLORS, COLOR_HYPER, COLOR_CONTEXT, COLOR_ORIGINAL,
)

DEFAULT_PATIENT_ID = None   # None = automatic: patient with the most SLH episodes
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "output", "episode_classification_zoom_panels"
)


def plot_individual_panel(df: pd.DataFrame, episode: dict, stage: str,
                          timestep_min: float,
                          window_hours: float = ZOOM_WINDOW_H) -> plt.Figure:
    """
    Build and return a standalone figure for one zoom panel.

    Mirrors ``plot_single_zoom_panel`` from generate_zoom_panels.py but
    wraps it in its own figure so it can be saved independently.
    """
    fig, ax = plt.subplots(figsize=(5.0, 3.2), constrained_layout=True)

    color  = STAGE_COLORS[stage]
    win_td = pd.Timedelta(hours=window_hours)
    ctx    = pd.Timedelta(minutes=CONTEXT_WIN_MIN)

    t_lo = max(episode["onset"]  - win_td, df[TIME_COL].iloc[0])
    t_hi = min(episode["offset"] + win_td, df[TIME_COL].iloc[-1])

    mask  = (df[TIME_COL] >= t_lo) & (df[TIME_COL] <= t_hi)
    t_win = df.loc[mask, TIME_COL]
    g_win = df.loc[mask, GLUCOSE_COL].astype(float)

    # Break trace at gaps
    diffs  = t_win.diff().dt.total_seconds().fillna(0) / 60
    g_plot = g_win.copy()
    g_plot[diffs > timestep_min * GAP_FACTOR] = np.nan

    _shade_nights(ax, t_lo, t_hi)

    # Context windows
    ax.axvspan(episode["onset"] - ctx, episode["onset"],
               color=COLOR_CONTEXT, alpha=0.09, zorder=1)
    ax.axvspan(episode["offset"], episode["offset"] + ctx,
               color=COLOR_CONTEXT, alpha=0.09, zorder=1)

    # Episode core
    ax.axvspan(episode["onset"], episode["offset"],
               color=color, alpha=ALPHA_EPISODE + 0.10, zorder=2)

    # Threshold lines
    ax.axhline(H_HYPER,   color=COLOR_HYPER,   ls="--", lw=1.0, zorder=4,
               label=f"H = {H_HYPER} mg/dL")
    ax.axhline(H_CONTEXT, color=COLOR_CONTEXT, ls=":",  lw=1.0, zorder=4,
               label=f"Hc = {H_CONTEXT} mg/dL")

    # Onset / offset markers
    for tm in (episode["onset"], episode["offset"]):
        ax.axvline(tm, color="#777777", lw=0.7, ls="-", alpha=0.45, zorder=5)

    # CGM trace
    ax.plot(t_win, g_plot, color=COLOR_ORIGINAL, lw=1.2, zorder=6)
    ax.scatter(t_win, g_win, color=COLOR_ORIGINAL, s=5, alpha=0.4, zorder=7)

    # Peak annotation
    pk_t, pk_v = episode["peak_time"], episode["peak_glucose"]
    ax.scatter([pk_t], [pk_v], marker="^", s=25, color=COLOR_HYPER, zorder=9)

    annot_text = f"{pk_v:.0f} mg/dL\n{episode['duration_min']:.0f} min"
    if "completeness_episode" in episode:
        annot_text += f"\nData: {episode['completeness_episode']:.0%}"

    ax.annotate(
        annot_text,
        xy=(pk_t, pk_v),
        xytext=(pk_t + pd.Timedelta(minutes=18), pk_v + 25),
        fontsize=7, color=COLOR_HYPER, ha="left", va="bottom",
        arrowprops=dict(arrowstyle="->", color=COLOR_HYPER, lw=0.7),
        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                  ec=COLOR_HYPER, lw=0.6, alpha=0.92),
        zorder=10,
    )

    # Axes
    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(70, 350)

    span_min = (t_hi - t_lo).total_seconds() / 60
    interval = 1 if span_min <= 300 else 2
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=7, rotation=30, ha="right")

    ax.set_ylabel("Glucose (mg/dL)", fontsize=8, labelpad=2)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=0.22, lw=0.5)

    # Remove C0-C6 prefix from label (keep only the description)
    label = STAGE_LABELS[stage]
    # Strip "C#  " or "SLH  " prefix if present
    if label.startswith("C") and "  " in label:
        label = label.split("  ", 1)[1]
    elif label.startswith("SLH") and "  " in label:
        label = label.split("  ", 1)[1]

    ax.set_title(label, fontsize=8, fontweight="bold",
                 pad=5, loc="left", color=color)

    ax.legend(fontsize=6.5, framealpha=0.85, loc="upper left")

    return fig


def plot_multi_panel_figure(df: pd.DataFrame, episodes_and_stages: list,
                             patient_id: str, timestep_min: float,
                             window_hours: float = ZOOM_WINDOW_H) -> plt.Figure:
    """
    Build and return a multi-panel figure with episodes side by side.

    Parameters
    ----------
    df : pd.DataFrame
        CGM data
    episodes_and_stages : list of (episode, stage) tuples
        Episodes to plot
    patient_id : str
        Patient ID for title
    timestep_min : float
        Dominant sampling interval
    window_hours : float
        Window size around episode

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_panels = len(episodes_and_stages)
    fig_w = 5.0 * n_panels + 0.5  # Scale width with number of panels

    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 3.2),
                              constrained_layout=True)

    # Handle single panel case (axes is not array)
    if n_panels == 1:
        axes = [axes]

    for ax, (episode, stage) in zip(axes, episodes_and_stages):
        color  = STAGE_COLORS[stage]
        win_td = pd.Timedelta(hours=window_hours)
        ctx    = pd.Timedelta(minutes=CONTEXT_WIN_MIN)

        t_lo = max(episode["onset"]  - win_td, df[TIME_COL].iloc[0])
        t_hi = min(episode["offset"] + win_td, df[TIME_COL].iloc[-1])

        mask  = (df[TIME_COL] >= t_lo) & (df[TIME_COL] <= t_hi)
        t_win = df.loc[mask, TIME_COL]
        g_win = df.loc[mask, GLUCOSE_COL].astype(float)

        # Break trace at gaps
        diffs  = t_win.diff().dt.total_seconds().fillna(0) / 60
        g_plot = g_win.copy()
        g_plot[diffs > timestep_min * GAP_FACTOR] = np.nan

        _shade_nights(ax, t_lo, t_hi)

        # Context windows
        ax.axvspan(episode["onset"] - ctx, episode["onset"],
                   color=COLOR_CONTEXT, alpha=0.09, zorder=1)
        ax.axvspan(episode["offset"], episode["offset"] + ctx,
                   color=COLOR_CONTEXT, alpha=0.09, zorder=1)

        # Episode core
        ax.axvspan(episode["onset"], episode["offset"],
                   color=color, alpha=ALPHA_EPISODE + 0.10, zorder=2)

        # Threshold lines
        ax.axhline(H_HYPER,   color=COLOR_HYPER,   ls="--", lw=1.0, zorder=4,
                   label=f"H = {H_HYPER} mg/dL")
        ax.axhline(H_CONTEXT, color=COLOR_CONTEXT, ls=":",  lw=1.0, zorder=4,
                   label=f"Hc = {H_CONTEXT} mg/dL")

        # Onset / offset markers
        for tm in (episode["onset"], episode["offset"]):
            ax.axvline(tm, color="#777777", lw=0.7, ls="-", alpha=0.45, zorder=5)

        # CGM trace
        ax.plot(t_win, g_plot, color=COLOR_ORIGINAL, lw=1.2, zorder=6)
        ax.scatter(t_win, g_win, color=COLOR_ORIGINAL, s=5, alpha=0.4, zorder=7)

        # Peak annotation
        pk_t, pk_v = episode["peak_time"], episode["peak_glucose"]
        ax.scatter([pk_t], [pk_v], marker="^", s=25, color=COLOR_HYPER, zorder=9)

        annot_text = f"{pk_v:.0f} mg/dL\n{episode['duration_min']:.0f} min"
        if "completeness_episode" in episode:
            annot_text += f"\nData: {episode['completeness_episode']:.0%}"

        ax.annotate(
            annot_text,
            xy=(pk_t, pk_v),
            xytext=(pk_t + pd.Timedelta(minutes=18), pk_v + 25),
            fontsize=7, color=COLOR_HYPER, ha="left", va="bottom",
            arrowprops=dict(arrowstyle="->", color=COLOR_HYPER, lw=0.7),
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec=COLOR_HYPER, lw=0.6, alpha=0.92),
            zorder=10,
        )

        # Axes
        ax.set_xlim(t_lo, t_hi)
        ax.set_ylim(70, 350)

        span_min = (t_hi - t_lo).total_seconds() / 60
        interval = 1 if span_min <= 300 else 2
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.setp(ax.xaxis.get_majorticklabels(), fontsize=7, rotation=30, ha="right")

        ax.set_ylabel("Glucose (mg/dL)", fontsize=8, labelpad=2)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", ls=":", alpha=0.22, lw=0.5)

        # Remove C0-C6 prefix from label (keep only the description)
        label = STAGE_LABELS[stage]
        # Strip "C#  " or "SLH  " prefix if present
        if label.startswith("C") and "  " in label:
            label = label.split("  ", 1)[1]
        elif label.startswith("SLH") and "  " in label:
            label = label.split("  ", 1)[1]

        ax.set_title(label, fontsize=8, fontweight="bold",
                     pad=5, loc="left", color=color)

        ax.legend(fontsize=6.5, framealpha=0.85, loc="upper left")

    return fig


def generate_individual_panels(patient_id: str, output_dir: str) -> None:
    """Load data, classify episodes, and save one figure per panel."""
    df = load_patient_window(patient_id)
    if df is None:
        print(f"[!] Could not load data for {patient_id}")
        return

    print(f"Loaded {len(df):,} data points for {patient_id}")

    timestep_min = detect_dominant_sampling_interval(df)
    raw_episodes = detect_hyperglycemia_episodes(df)
    print(f"Detected {len(raw_episodes)} raw hyperglycemia episodes")

    if not raw_episodes:
        print("[!] No episodes detected")
        return

    classified = classify_episodes_for_visualization(df, raw_episodes)

    print("\nEpisode classification:")
    for stage in DRAW_ORDER:
        count = len(classified[stage])
        if count:
            print(f"  {stage}: {count}")

    selected = select_representative_episodes(classified)
    print(f"\nSelected {len(selected)} representative episodes")

    apply_publication_style()
    os.makedirs(output_dir, exist_ok=True)

    # Generate individual panels
    print("\nGenerating individual panels...")
    for episode, stage in selected:
        fig = plot_individual_panel(df, episode, stage, timestep_min)
        out_path = os.path.join(output_dir, f"{patient_id}_{stage}")
        save_figure(fig, out_path)
        plt.close(fig)
        print(f"  Saved: {out_path}.pdf/.png")

    # Generate multi-panel grouped figures
    print("\nGenerating grouped multi-panel figures...")

    # Create a dictionary for quick lookup by stage (handle both exact and prefix matches)
    selected_dict = {stage: episode for episode, stage in selected}

    # Helper function to find stage by prefix
    def find_stage_by_prefix(prefix: str) -> tuple:
        """Find stage starting with prefix. Returns (stage_name, episode) or (None, None)."""
        for stage, episode in selected_dict.items():
            if stage.startswith(prefix):
                return (stage, episode)
        return (None, None)

    # C4 + C5 side by side
    c4_stage, c4_episode = find_stage_by_prefix("C4")
    c5_stage, c5_episode = find_stage_by_prefix("C5")

    if c4_stage and c5_stage:
        episodes_list = [
            (c4_episode, c4_stage),
            (c5_episode, c5_stage),
        ]
        fig = plot_multi_panel_figure(df, episodes_list, patient_id, timestep_min)
        out_path = os.path.join(output_dir, f"{patient_id}_C4_C5")
        save_figure(fig, out_path)
        plt.close(fig)
        print(f"  Saved: {out_path}.pdf/.png (C4 + C5)")
    else:
        missing = [s for s in ["C4", "C5"] if (s == "C4" and not c4_stage) or (s == "C5" and not c5_stage)]
        print(f"  [!] Skipped C4+C5 figure — missing {missing}")

    # C1 + C2 + C3 side by side
    c1_stage, c1_episode = find_stage_by_prefix("C1")
    c2_stage, c2_episode = find_stage_by_prefix("C2")
    c3_stage, c3_episode = find_stage_by_prefix("C3")

    if c1_stage and c2_stage and c3_stage:
        episodes_list = [
            (c1_episode, c1_stage),
            (c2_episode, c2_stage),
            (c3_episode, c3_stage),
        ]
        fig = plot_multi_panel_figure(df, episodes_list, patient_id, timestep_min)
        out_path = os.path.join(output_dir, f"{patient_id}_C1_C2_C3")
        save_figure(fig, out_path)
        plt.close(fig)
        print(f"  Saved: {out_path}.pdf/.png (C1 + C2 + C3)")
    else:
        missing = [s for s in ["C1", "C2", "C3"]
                   if (s == "C1" and not c1_stage) or (s == "C2" and not c2_stage) or (s == "C3" and not c3_stage)]
        print(f"  [!] Skipped C1+C2+C3 figure — missing {missing}")

    # C1 + C3 side by side (no C2)
    if c1_stage and c3_stage:
        episodes_list = [
            (c1_episode, c1_stage),
            (c3_episode, c3_stage),
        ]
        fig = plot_multi_panel_figure(df, episodes_list, patient_id, timestep_min)
        out_path = os.path.join(output_dir, f"{patient_id}_C1_C3")
        save_figure(fig, out_path)
        plt.close(fig)
        print(f"  Saved: {out_path}.pdf/.png (C1 + C3)")
    else:
        missing = [s for s in ["C1", "C3"]
                   if (s == "C1" and not c1_stage) or (s == "C3" and not c3_stage)]
        print(f"  [!] Skipped C1+C3 figure — missing {missing}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Save each zoom panel as an individual figure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--patient_id", default=DEFAULT_PATIENT_ID,
                        help="Patient ID")
    parser.add_argument("--out_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory")
    return parser


def main():
    args = parse_arguments().parse_args()
    args.patient_id = resolve_patient_id(args.patient_id)
    print("\n" + "=" * 60)
    print("  Zoom Panels — Individual Figures")
    print("=" * 60)
    print(f"  Patient : {args.patient_id}")
    print(f"  Output  : {args.out_dir}")
    print("=" * 60 + "\n")

    generate_individual_panels(args.patient_id, args.out_dir)

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
