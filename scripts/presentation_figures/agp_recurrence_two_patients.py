"""
scripts/presentation_figures/agp_recurrence_two_patients.py
===========================================================

Side-by-side presentation figure showing AGP + recurrence for two patients.

Layout (2 patients × stacked panels):
  ┌──────────────────┬──────────────────┐
  │  Patient A (AGP) │  Patient B (AGP) │
  ├──────────────────┼──────────────────┤
  │  Patient A (rec) │  Patient B (rec) │
  └──────────────────┴──────────────────┘

Where each "AGP" panel shows:
  • 5–95 percentile band, IQR, median (and optionally mean)
  • Hypo/hyper threshold lines with right-margin labels
  • Shared x-axis (00:00→24:00 time of day)

And each "rec" (recurrence) panel shows:
  • % of days with hyperglycaemic episodes (SLH / Not isolated / Night / LLH stacked — same categories as manuscript Figure 2)
  • 50% systematic threshold line
  • Shared x-axis with AGP panel

Both patients use the same AGP and recurrence styling; shared legend
at bottom center combines percentile bands, median, classifications,
and thresholds.

Usage
-----
    python3 -m scripts.presentation_figures.agp_recurrence_two_patients [--patient_a ID] \
                                                         [--patient_b ID] \
                                                         [--resolution 5] \
                                                         [--no_mean] \
                                                         [--out_dir DIR]

Output
------
    scripts/presentation_figures/output/agp_recurrence_two_patients/  (git-ignored; --out_dir to change)
"""

import os
import sys
import argparse
from datetime import datetime, time as _dtime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.patient_selection import resolve_patient_pair
from src.preprocessing.load_data import load_patient_window, prepare_valid_data
from src.preprocessing.timestep import detect_dominant_sampling_interval
from src.episodes.load_episodes import load_patient_episodes
from src.agp.compute_agp import compute_agp_profile
from src.config import (
    TIME_COL, GLUCOSE_COL,
    THRESHOLD_HYPER, THRESHOLD_HYPO,
)
from src.visualization.style import apply_publication_style, save_figure
from scripts.paper_figures.generate_figure_2_classification_summary import (
    assign_fine_category, filter_data_quality, CATEGORY_ORDER, CATEGORY_COLORS,
)
from src.visualization.colors import (
    COLOR_AGP_PERCENTILE_95, COLOR_AGP_PERCENTILE_IQR,
    COLOR_AGP_MEDIAN, COLOR_AGP_MEAN,
    COLOR_AGP_TARGET_ZONE,
    COLOR_THRESHOLD_HYPER, COLOR_THRESHOLD_HYPO,
    COLOR_RECURRENCE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Episode categories — same classification as manuscript Figure 2
# (SLH / Not isolated / Night / LLH; episodes failing C2 are dropped)
# ---------------------------------------------------------------------------
def _categorise(episodes: pd.DataFrame) -> pd.DataFrame:
    """Add the Figure 2 ``category`` column (empty frame if there is no usable episode)."""
    if episodes is None or len(episodes) == 0:
        return pd.DataFrame(columns=["onset_time", "offset_time", "duration_min", "category"])
    kept, _ = filter_data_quality(episodes)
    if len(kept) == 0:
        return pd.DataFrame(columns=["onset_time", "offset_time", "duration_min", "category"])
    return assign_fine_category(kept)


def _recurrence_by_category(episodes: pd.DataFrame, resolution_min: int, n_days: int,
                            start_date) -> dict:
    """% of days with an episode of each category in each time-of-day slot."""
    n_slots = int(24 * 60 / resolution_min)
    occurrence = {c: np.zeros(n_slots) for c in CATEGORY_ORDER}
    for _, ep in episodes.iterrows():
        cat = ep["category"]
        t, seen = ep["onset_time"], set()
        while t <= ep["offset_time"]:
            key = ((t.date() - start_date).days,
                   min(int((t.hour * 60 + t.minute) / resolution_min), n_slots - 1))
            if key not in seen:
                occurrence[cat][key[1]] += 1
                seen.add(key)
            t += pd.Timedelta(minutes=resolution_min)
    return {c: v / n_days * 100 for c, v in occurrence.items()}


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PATIENT_A   = None
DEFAULT_PATIENT_B   = None
DEFAULT_RESOLUTION  = 5          # minutes
DEFAULT_OUTPUT_DIR  = os.path.join(
    os.path.dirname(__file__), "output", "agp_recurrence_two_patients"
)

# ---------------------------------------------------------------------------
# Figure geometry (16:9 slide-friendly, two columns × two rows)
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 16.0, 9.0     # 16:9 aspect ratio

# Height ratio: AGP (taller) over recurrence heatmap (shorter)
AGP_HEIGHT_RATIO     = 3
REC_HEIGHT_RATIO     = 2
TOTAL_HEIGHT_RATIO   = AGP_HEIGHT_RATIO + REC_HEIGHT_RATIO

# Presentation font sizes
FS_BASE   = 12
FS_LABEL  = 13
FS_TICK   = 11
FS_LEGEND = 10
FS_ANNOT  = 10

# AGP y-axis bounds
AGP_Y_MIN, AGP_Y_MAX = 50, 330

# Recurrence systematic threshold
SYSTEMATIC_PCT = 50


# =============================================================================
# PANEL DRAWERS (each takes an existing Axes)
# =============================================================================

def _draw_agp(ax, agp: pd.DataFrame, show_mean: bool = True,
              show_legend: bool = False, show_right_labels: bool = False) -> None:
    """
    Draw the Ambulatory Glucose Profile onto *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes for the AGP plot.
    agp : pd.DataFrame
        AGP profile with columns: time_str, q5, q25, q75, q95, median, mean.
    show_mean : bool
        Whether to overlay the mean glucose line.
    show_legend : bool
        Whether to draw legend on this axes (only for one panel per patient column).
    show_right_labels : bool
        Whether to show threshold value labels on the right margin.

    Notes
    -----
    X-axis is in float hours (0–24) to allow sharex with the recurrence panel.
    """
    # Convert "HH:MM" time strings to fractional hours
    x = np.array([
        int(t.split(":")[0]) + int(t.split(":")[1]) / 60
        for t in agp["time_str"]
    ])

    # ── Target zone ──────────────────────────────────────────────────────────
    ax.axhspan(THRESHOLD_HYPO, THRESHOLD_HYPER,
               color=COLOR_AGP_TARGET_ZONE, alpha=0.10, zorder=0)

    # ── Percentile bands ─────────────────────────────────────────────────────
    ax.fill_between(x, agp["q5"], agp["q95"],
                    color=COLOR_AGP_PERCENTILE_95, alpha=0.20, zorder=1,
                    label="5–95th percentile")
    ax.fill_between(x, agp["q25"], agp["q75"],
                    color=COLOR_AGP_PERCENTILE_IQR, alpha=0.38, zorder=2,
                    label="25–75th percentile (IQR)")

    # Soft IQR edge lines
    ax.plot(x, agp["q25"], color=COLOR_AGP_PERCENTILE_IQR,
            lw=0.9, ls=":", alpha=0.60, zorder=3)
    ax.plot(x, agp["q75"], color=COLOR_AGP_PERCENTILE_IQR,
            lw=0.9, ls=":", alpha=0.60, zorder=3)

    # ── Median ───────────────────────────────────────────────────────────────
    ax.plot(x, agp["median"], color=COLOR_AGP_MEDIAN,
            lw=2.8, ls="-", label="Median", zorder=5)

    # ── Mean (optional) ───────────────────────────────────────────────────────
    if show_mean:
        ax.plot(x, agp["mean"], color=COLOR_AGP_MEAN,
                lw=2.0, ls="--", alpha=0.85, label="Mean", zorder=4)

    # ── Threshold lines ───────────────────────────────────────────────────────
    ax.axhline(THRESHOLD_HYPER, color=COLOR_THRESHOLD_HYPER,
               lw=1.6, ls="--", alpha=0.90, zorder=6)
    ax.axhline(THRESHOLD_HYPO, color=COLOR_THRESHOLD_HYPO,
               lw=1.6, ls="--", alpha=0.90, zorder=6)

    # Right-margin labels (only on rightmost panel)
    if show_right_labels:
        for val, color, text in [
            (THRESHOLD_HYPER, COLOR_THRESHOLD_HYPER, f"{THRESHOLD_HYPER} mg/dL"),
            (THRESHOLD_HYPO,  COLOR_THRESHOLD_HYPO,  f"{THRESHOLD_HYPO} mg/dL"),
        ]:
            ax.annotate(
                text,
                xy=(24, val), xytext=(6, 0),
                textcoords="offset points",
                fontsize=FS_ANNOT, color=color, fontweight="bold",
                va="center", annotation_clip=False,
            )

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim(0, 24)
    ax.set_ylim(AGP_Y_MIN, AGP_Y_MAX)

    yticks = [THRESHOLD_HYPO, 100, 140, THRESHOLD_HYPER, 250, 300]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(v) for v in yticks], fontsize=FS_TICK)

    ax.set_ylabel("Glucose (mg/dL)", fontsize=FS_LABEL, labelpad=6)
    ax.tick_params(axis="x", which="both", labelbottom=False)  # hidden — shared axis

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, ls=":", lw=0.5, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)


def _draw_recurrence(ax, occurrence: dict, n_days: int,
                     resolution_min: float,
                     show_legend: bool = False,
                     show_right_labels: bool = False) -> None:
    """
    Draw the stacked recurrence heatmap (SLH / Not isolated / Night / LLH) onto *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes for the recurrence plot.
    occurrence : dict
        Dictionary with the four Figure 2 categories as keys, each holding
        numpy arrays of percentage values for each time slot.
    n_days : int
        Number of days in the data window (for scaling).
    resolution_min : float
        Time-slot resolution in minutes.
    show_legend : bool
        Whether to draw legend on this axes.
    show_right_labels : bool
        Whether to show threshold percentage label on the right margin.

    Notes
    -----
    X-axis is in float hours (0–24), matching the AGP panel.
    Y-axis shows percentage of days with a hyperglycaemia episode.
    """
    n_slots = int(24 * 60 / resolution_min)
    x       = np.arange(n_slots) * resolution_min / 60  # float hours

    bottom = np.zeros(n_slots)

    # SLH drawn first (bottom layer), then Not isolated / Night / LLH stacked on top
    for label, color, display_label in [(c, CATEGORY_COLORS[c], c) for c in CATEGORY_ORDER]:
        vals = occurrence.get(label, np.zeros(n_slots))
        ax.fill_between(x, bottom, bottom + vals,
                        color=color, alpha=0.88,
                        label=display_label, zorder=2,
                        edgecolor="white", linewidth=0.4)
        bottom += vals

    # Systematic threshold
    ax.axhline(SYSTEMATIC_PCT, color=COLOR_RECURRENCE_THRESHOLD,
               ls="--", lw=1.5, zorder=5)

    if show_right_labels:
        ax.annotate(
            f"{SYSTEMATIC_PCT}%",
            xy=(24, SYSTEMATIC_PCT), xytext=(6, 0),
            textcoords="offset points",
            fontsize=FS_ANNOT, color=COLOR_RECURRENCE_THRESHOLD,
            fontweight="bold", va="center", annotation_clip=False,
        )

    # ── Axes — y-axis focused at ≤50% (zoom on the systematic threshold zone) ──
    y_max = max(SYSTEMATIC_PCT * 1.15, float(bottom.max()) * 1.08)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, y_max)

    # X-axis: shared formatting (major every 3 h, minor every 1 h)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels(
        [f"{h:02d}:00" for h in range(0, 25, 3)],
        fontsize=FS_TICK,
    )
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))

    ax.set_xlabel("Time of day", fontsize=FS_LABEL, labelpad=6)
    ax.set_ylabel("Days with episode (%)", fontsize=FS_LABEL, labelpad=6)
    ax.tick_params(axis="y", labelsize=FS_TICK)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, ls=":", lw=0.5, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)


# =============================================================================
# FIGURE ASSEMBLY
# =============================================================================

def plot_agp_recurrence_two_patients(
    df_a: pd.DataFrame, episodes_a: pd.DataFrame,
    df_b: pd.DataFrame, episodes_b: pd.DataFrame,
    resolution_min: float,
    show_mean: bool = True
) -> plt.Figure:
    """
    Build the 4-panel figure (2 columns × 2 rows):
      • Column A: Patient A's AGP and recurrence
      • Column B: Patient B's AGP and recurrence

    Parameters
    ----------
    df_a, df_b : pd.DataFrame
        Preprocessed CGM data for each patient.
    episodes_a, episodes_b : pd.DataFrame
        Episode tables for each patient.
    resolution_min : float
        AGP / recurrence time-slot width in minutes.
    show_mean : bool
        Whether to overlay the mean glucose line on the AGP.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ── Compute profiles for both patients ────────────────────────────────────
    agp_a = compute_agp_profile(df_a, int(resolution_min))
    agp_b = compute_agp_profile(df_b, int(resolution_min))

    start_date_a = df_a[TIME_COL].dt.date.min()
    end_date_a   = df_a[TIME_COL].dt.date.max()
    n_days_a     = (end_date_a - start_date_a).days + 1

    start_date_b = df_b[TIME_COL].dt.date.min()
    end_date_b   = df_b[TIME_COL].dt.date.max()
    n_days_b     = (end_date_b - start_date_b).days + 1

    occurrence_a = _recurrence_by_category(
        episodes_a, int(resolution_min), n_days_a, start_date_a
    )
    occurrence_b = _recurrence_by_category(
        episodes_b, int(resolution_min), n_days_b, start_date_b
    )

    # ── Style ─────────────────────────────────────────────────────────────────
    apply_publication_style()
    matplotlib.rcParams.update({
        "font.size":        FS_BASE,
        "axes.labelsize":   FS_LABEL,
        "xtick.labelsize":  FS_TICK,
        "ytick.labelsize":  FS_TICK,
        "legend.fontsize":  FS_LEGEND,
    })

    # ── Layout with GridSpec (2 columns × 2 rows) ────────────────────────────
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[AGP_HEIGHT_RATIO, REC_HEIGHT_RATIO],
        width_ratios=[1, 1],
        hspace=0.08,
        wspace=0.25,
    )

    # ── Create axes ───────────────────────────────────────────────────────────
    ax_agp_a = fig.add_subplot(gs[0, 0])  # Patient A, AGP
    ax_rec_a = fig.add_subplot(gs[1, 0])  # Patient A, Recurrence
    ax_agp_b = fig.add_subplot(gs[0, 1])  # Patient B, AGP
    ax_rec_b = fig.add_subplot(gs[1, 1])  # Patient B, Recurrence

    # ── Draw panels ───────────────────────────────────────────────────────────
    _draw_agp(ax_agp_a, agp_a, show_mean=show_mean,
              show_legend=False, show_right_labels=False)
    _draw_recurrence(ax_rec_a, occurrence_a, n_days_a, resolution_min,
                     show_legend=False, show_right_labels=False)

    _draw_agp(ax_agp_b, agp_b, show_mean=show_mean,
              show_legend=False, show_right_labels=True)
    _draw_recurrence(ax_rec_b, occurrence_b, n_days_b, resolution_min,
                     show_legend=False, show_right_labels=True)

    # Align x-ticks for all AGP panels with recurrence panels
    for ax in [ax_agp_a, ax_agp_b]:
        ax.set_xticks(range(0, 25, 3))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))

    # ── Shared legend at bottom center ────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor=COLOR_AGP_PERCENTILE_95, alpha=0.35,
                       label="5–95th percentile"),
        mpatches.Patch(facecolor=COLOR_AGP_PERCENTILE_IQR, alpha=0.55,
                       label="25–75th percentile (IQR)"),
        Line2D([0], [0], color=COLOR_AGP_MEDIAN, lw=2.8,
               label="Median"),
    ]
    if show_mean:
        handles.append(
            Line2D([0], [0], color=COLOR_AGP_MEAN, lw=2.0, ls="--",
                   label="Mean")
        )
    handles += [
        *[mpatches.Patch(facecolor=CATEGORY_COLORS[c], alpha=0.85, label=c) for c in CATEGORY_ORDER],
        Line2D([0], [0], color=COLOR_THRESHOLD_HYPER, lw=1.6, ls="--",
               label=f"Hyper  ({THRESHOLD_HYPER} mg/dL)"),
        Line2D([0], [0], color=COLOR_THRESHOLD_HYPO, lw=1.6, ls="--",
               label=f"Hypo  ({THRESHOLD_HYPO} mg/dL)"),
    ]

    fig.legend(handles=handles, loc="lower center",
               fontsize=FS_LEGEND, framealpha=0.92,
               ncol=4, columnspacing=1.2, handlelength=1.6,
               bbox_to_anchor=(0.5, -0.01), borderpad=0.5)

    # Add bottom margin for legend
    fig.subplots_adjust(bottom=0.12)

    return fig


# =============================================================================
# PIPELINE
# =============================================================================

def generate(patient_a: str, patient_b: str, resolution_min: int,
             show_mean: bool, output_dir: str) -> None:
    """Load data for both patients → classify → draw → save."""
    print(f"\n  Loading Patient A ({patient_a})...")
    raw_a = load_patient_window(patient_a)
    if raw_a is None:
        print(f"  [!] No CGM data found for {patient_a}")
        return
    df_a = prepare_valid_data(raw_a)
    print(f"    CGM  : {len(df_a):,} valid points")

    episodes_a = load_patient_episodes(patient_a)
    if episodes_a is None or len(episodes_a) == 0:
        print(f"    [!] No episodes found for {patient_a}")
    episodes_a = _categorise(episodes_a)
    print("    Episodes : " + "  /  ".join(
        f"{(episodes_a['category'] == c).sum()} {c}" for c in CATEGORY_ORDER))

    print(f"\n  Loading Patient B ({patient_b})...")
    raw_b = load_patient_window(patient_b)
    if raw_b is None:
        print(f"  [!] No CGM data found for {patient_b}")
        return
    df_b = prepare_valid_data(raw_b)
    print(f"    CGM  : {len(df_b):,} valid points")

    episodes_b = load_patient_episodes(patient_b)
    if episodes_b is None or len(episodes_b) == 0:
        print(f"    [!] No episodes found for {patient_b}")
    episodes_b = _categorise(episodes_b)
    print("    Episodes : " + "  /  ".join(
        f"{(episodes_b['category'] == c).sum()} {c}" for c in CATEGORY_ORDER))

    res = resolution_min
    print(f"\n  Resolution : {res} min")

    fig = plot_agp_recurrence_two_patients(
        df_a, episodes_a, df_b, episodes_b, res, show_mean=show_mean
    )

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"{patient_a}_vs_{patient_b}_agp_recurrence_{res}min")
    save_figure(fig, out)
    print(f"\n  Saved → {out}.pdf / .svg")


# =============================================================================
# CLI
# =============================================================================

def parse_arguments():
    p = argparse.ArgumentParser(
        description="AGP + recurrence heatmap — two-patient side-by-side presentation figure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--patient_a", default=DEFAULT_PATIENT_A,
                   help="Patient A ID")
    p.add_argument("--patient_b", default=DEFAULT_PATIENT_B,
                   help="Patient B ID")
    p.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                   help="Time-slot resolution in minutes (AGP + recurrence)")
    p.add_argument("--no_mean", action="store_true",
                   help="Hide mean line on AGP panels")
    p.add_argument("--out_dir", default=DEFAULT_OUTPUT_DIR,
                   help="Output directory")
    return p


def main():
    args = parse_arguments().parse_args()
    args.patient_a, args.patient_b = resolve_patient_pair(args.patient_a, args.patient_b)

    print("\n" + "=" * 60)
    print("  AGP + Recurrence — Two-Patient Comparison")
    print("=" * 60)
    print(f"  Patient A  : {args.patient_a}")
    print(f"  Patient B  : {args.patient_b}")
    print(f"  Resolution : {args.resolution} min")
    print(f"  Mean line  : {'no' if args.no_mean else 'yes'}")
    print(f"  Output     : {args.out_dir}")
    print("=" * 60)

    generate(args.patient_a, args.patient_b, args.resolution,
             show_mean=not args.no_mean,
             output_dir=args.out_dir)

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
