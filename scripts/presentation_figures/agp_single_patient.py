"""
scripts/presentation_figures/agp_single_patient.py
==================================================

Ambulatory Glucose Profile — clean presentation figure.

Produces a single panel containing only the AGP for one patient:
  · 5–95 percentile band   (light blue fill)
  · 25–75 IQR band         (medium blue fill, no hatch)
  · Median line            (dark blue, solid)
  · Mean line              (green, dashed)   — toggleable
  · Target range           (70–180 mg/dL, very light green fill)
  · 70 mg/dL threshold     (dashed, hypo colour)
  · 180 mg/dL threshold    (dashed, hyper colour)

No patient ID, no extra panels, no tables — presentation-ready.

Usage
-----
    python3 -m scripts.presentation_figures.agp_single_patient [--patient_id ID]
                                              [--resolution 15]
                                              [--no_mean]
                                              [--out_dir DIR]
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
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.patient_selection import resolve_patient_id
from src.preprocessing.load_data import load_patient_window
from src.agp.compute_agp import compute_agp_profile
from src.config import TIME_COL, GLUCOSE_COL, THRESHOLD_HYPER, THRESHOLD_HYPO
from src.config import FIG_DIR
from src.visualization.style import apply_publication_style, save_figure
from src.visualization.colors import (
    COLOR_AGP_PERCENTILE_95,
    COLOR_AGP_PERCENTILE_IQR,
    COLOR_AGP_MEDIAN,
    COLOR_AGP_MEAN,
    COLOR_AGP_TARGET_ZONE,
    COLOR_THRESHOLD_HYPER,
    COLOR_THRESHOLD_HYPO,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PATIENT_ID  = None
DEFAULT_RESOLUTION  = 5           # minutes — standard AGP resolution
DEFAULT_OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output", "agp_single_patient")

# ---------------------------------------------------------------------------
# Figure geometry (16:9 slide-friendly)
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 10.0, 5.5

# Presentation font sizes
FS_BASE   = 13
FS_LABEL  = 14
FS_TICK   = 12
FS_LEGEND = 11
FS_ANNOT  = 11

# Y-axis bounds
Y_MIN, Y_MAX = 50, 330

# ---------------------------------------------------------------------------
# Legend placement (below the axes)
# ---------------------------------------------------------------------------
#: Columns in the legend strip under the figure.
LEGEND_NCOL = 4

#: Figure fraction reserved at the bottom for the legend strip.
LEGEND_ROOM = 0.16

#: Legend anchor in axes coordinates — negative y puts it under the x-label.
LEGEND_Y_OFFSET = -0.20


# =============================================================================
# PLOT
# =============================================================================

def plot_agp(df: pd.DataFrame, resolution_min: int = DEFAULT_RESOLUTION,
             show_mean: bool = True) -> plt.Figure:
    """
    Build a presentation-ready AGP figure for the given patient data.

    Parameters
    ----------
    df : pd.DataFrame
        Full CGM data (TIME_COL, GLUCOSE_COL).
    resolution_min : int
        Time-slot width in minutes used to compute the AGP (default 15).
    show_mean : bool
        Whether to overlay the mean line (default True).

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ── Compute profile ───────────────────────────────────────────────────────
    agp = compute_agp_profile(df, resolution_min)

    # Build datetime x-values anchored to today (only time-of-day matters)
    today = datetime.today().date()
    x_vals = [
        datetime.combine(today, datetime.strptime(t, "%H:%M").time())
        for t in agp["time_str"]
    ]
    day_start = datetime.combine(today, _dtime(0, 0))
    day_end   = day_start + timedelta(days=1)

    # ── Style ─────────────────────────────────────────────────────────────────
    apply_publication_style()
    matplotlib.rcParams.update({
        "font.size":        FS_BASE,
        "axes.labelsize":   FS_LABEL,
        "xtick.labelsize":  FS_TICK,
        "ytick.labelsize":  FS_TICK,
        "legend.fontsize":  FS_LEGEND,
    })

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # ── 1. Target zone (70–180 mg/dL)  ───────────────────────────────────────
    ax.axhspan(THRESHOLD_HYPO, THRESHOLD_HYPER,
               color=COLOR_AGP_TARGET_ZONE, alpha=0.10, zorder=0,
               label=f"Target range  ({THRESHOLD_HYPO}–{THRESHOLD_HYPER} mg/dL)")

    # ── 2. Percentile bands ───────────────────────────────────────────────────
    ax.fill_between(
        x_vals, agp["q5"], agp["q95"],
        color=COLOR_AGP_PERCENTILE_95,
        alpha=0.20,
        zorder=1,
        label="5–95th percentile",
    )
    ax.fill_between(
        x_vals, agp["q25"], agp["q75"],
        color=COLOR_AGP_PERCENTILE_IQR,
        alpha=0.38,
        zorder=2,
        label="25–75th percentile (IQR)",
    )

    # Soft boundary lines for IQR edges
    ax.plot(x_vals, agp["q25"], color=COLOR_AGP_PERCENTILE_IQR,
            lw=0.9, ls=":", alpha=0.6, zorder=3)
    ax.plot(x_vals, agp["q75"], color=COLOR_AGP_PERCENTILE_IQR,
            lw=0.9, ls=":", alpha=0.6, zorder=3)

    # ── 3. Median line ───────────────────────────────────────────────────────
    ax.plot(x_vals, agp["median"],
            color=COLOR_AGP_MEDIAN, lw=2.8, ls="-",
            label="Median", zorder=5)

    # ── 4. Mean line (optional) ───────────────────────────────────────────────
    if show_mean:
        ax.plot(x_vals, agp["mean"],
                color=COLOR_AGP_MEAN, lw=2.0, ls="--", alpha=0.85,
                label="Mean", zorder=4)

    # ── 5. Threshold lines ────────────────────────────────────────────────────
    ax.axhline(THRESHOLD_HYPER, color=COLOR_THRESHOLD_HYPER,
               lw=1.6, ls="--", alpha=0.90, zorder=6)
    ax.axhline(THRESHOLD_HYPO, color=COLOR_THRESHOLD_HYPO,
               lw=1.6, ls="--", alpha=0.90, zorder=6)

    # ── 6. Threshold annotations (right margin) ────────────────────────────────
    x_annot = mdates.date2num(day_end)
    ax.annotate(
        f"{THRESHOLD_HYPER} mg/dL",
        xy=(x_annot, THRESHOLD_HYPER), xytext=(6, 0),
        textcoords="offset points",
        fontsize=FS_ANNOT, color=COLOR_THRESHOLD_HYPER,
        fontweight="bold", va="center", annotation_clip=False,
    )
    ax.annotate(
        f"{THRESHOLD_HYPO} mg/dL",
        xy=(x_annot, THRESHOLD_HYPO), xytext=(6, 0),
        textcoords="offset points",
        fontsize=FS_ANNOT, color=COLOR_THRESHOLD_HYPO,
        fontweight="bold", va="center", annotation_clip=False,
    )

    # ── 7. Axes formatting ────────────────────────────────────────────────────
    ax.set_xlim(day_start, day_end)
    ax.set_ylim(Y_MIN, Y_MAX)

    # X-axis: major ticks every 3 h, minor every 1 h
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=FS_TICK)

    # Y-axis: annotate key glucose values
    yticks = [THRESHOLD_HYPO, 100, 140, THRESHOLD_HYPER, 250, 300]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(v) for v in yticks], fontsize=FS_TICK)

    ax.set_xlabel("Time of day", fontsize=FS_LABEL, labelpad=6)
    ax.set_ylabel("Glucose (mg/dL)", fontsize=FS_LABEL, labelpad=6)

    # ── 8. Legend ─────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=COLOR_AGP_PERCENTILE_95, alpha=0.35,
                       label="5–95th percentile"),
        mpatches.Patch(facecolor=COLOR_AGP_PERCENTILE_IQR, alpha=0.55,
                       label="25–75th percentile (IQR)"),
        Line2D([0], [0], color=COLOR_AGP_MEDIAN, lw=2.8,
               label="Median"),
    ]
    if show_mean:
        legend_handles.append(
            Line2D([0], [0], color=COLOR_AGP_MEAN, lw=2.0, ls="--",
                   label="Mean")
        )
    legend_handles += [
        mpatches.Patch(facecolor=COLOR_AGP_TARGET_ZONE, alpha=0.25,
                       label=f"Target range  ({THRESHOLD_HYPO}–{THRESHOLD_HYPER} mg/dL)"),
        Line2D([0], [0], color=COLOR_THRESHOLD_HYPER, lw=1.6, ls="--",
               label=f"Hyper threshold  ({THRESHOLD_HYPER} mg/dL)"),
        Line2D([0], [0], color=COLOR_THRESHOLD_HYPO, lw=1.6, ls="--",
               label=f"Hypo threshold  ({THRESHOLD_HYPO} mg/dL)"),
    ]

    # Anchored in axes coordinates, so the strip stays centred on the plot
    # rather than on the figure — the figure carries a right margin for the
    # threshold labels, which would otherwise pull the legend off-centre.
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, LEGEND_Y_OFFSET),
        fontsize=FS_LEGEND,
        frameon=False,
        ncol=LEGEND_NCOL,
        columnspacing=1.6,
        handlelength=1.8,
        borderpad=0.7,
    )

    # ── 9. Clean spines + subtle grid ─────────────────────────────────────────
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, ls=":", lw=0.5, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)

    # Right margin for the threshold labels, bottom margin for the legend strip
    # (tight_layout does not measure a legend drawn outside the axes).
    fig.tight_layout(rect=[0, LEGEND_ROOM, 0.90, 1.0])
    return fig


# =============================================================================
# PIPELINE
# =============================================================================

def generate(patient_id: str, resolution_min: int,
             show_mean: bool, output_dir: str) -> None:
    """Load data, compute AGP, draw figure, save."""
    df = load_patient_window(patient_id)
    if df is None:
        print(f"[!] Could not load data for {patient_id}")
        return

    n_days = int(
        (df[TIME_COL].max() - df[TIME_COL].min()).total_seconds() / 86400
    )
    print(f"  Loaded {len(df):,} data points  ({n_days} days)")
    print(f"  Resolution : {resolution_min} min")

    fig = plot_agp(df, resolution_min=resolution_min, show_mean=show_mean)

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{resolution_min}min"
    out    = os.path.join(output_dir, f"{patient_id}_agp{suffix}")
    save_figure(fig, out)
    print(f"  Saved → {out}.pdf / .svg")


# =============================================================================
# CLI
# =============================================================================

def parse_arguments():
    p = argparse.ArgumentParser(
        description="AGP-only presentation figure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--patient_id",  default=DEFAULT_PATIENT_ID,
                   help="Patient ID (default: chosen automatically — most / fewest SLH episodes)")
    p.add_argument("--resolution",  type=int, default=DEFAULT_RESOLUTION,
                   help="AGP time-slot resolution in minutes")
    p.add_argument("--no_mean",     action="store_true",
                   help="Hide the mean line (show median only)")
    p.add_argument("--out_dir",     default=DEFAULT_OUTPUT_DIR,
                   help="Output directory")
    return p


def main():
    args = parse_arguments().parse_args()
    args.patient_id = resolve_patient_id(args.patient_id)

    print("\n" + "=" * 55)
    print("  AGP — Presentation Figure")
    print("=" * 55)
    print(f"  Patient    : {args.patient_id}")
    print(f"  Resolution : {args.resolution} min")
    print(f"  Mean line  : {'no' if args.no_mean else 'yes'}")
    print(f"  Output     : {args.out_dir}")
    print("=" * 55 + "\n")

    generate(args.patient_id, args.resolution,
             show_mean=not args.no_mean,
             output_dir=args.out_dir)

    print("\n" + "=" * 55 + "\n")


if __name__ == "__main__":
    main()
