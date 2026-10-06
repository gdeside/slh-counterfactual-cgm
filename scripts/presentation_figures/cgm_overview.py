"""
scripts/presentation_figures/cgm_overview.py
============================================

7-day CGM trace for a single patient — presentation-ready figure.

Glucose zones (horizontal bands)
---------------------------------
  < 70 mg/dL     Hypoglycaemia              light red
  70–140 mg/dL   Normal range (healthy)     light green
  140–180 mg/dL  Caution / elevated         light amber
  > 180 mg/dL    Hyperglycaemia             light orange

Reference lines
---------------
  70 mg/dL   dashed red      Hypoglycaemia threshold
  140 mg/dL  dotted green    Normal upper limit (non-diabetic)
  180 mg/dL  dashed orange   Hyperglycaemia threshold (T1D)

Other features
--------------
  · Night bands (22:00–06:00)   pale grey
  · CGM trace split at data gaps (no false interpolation)
  · Relative time axis:  Day 1, Day 2 … (patient-privacy safe)
  · Best 7-day window selected by maximising glucose range excursions
    (number of points outside 70–180) → shows all three zones visually

Usage
-----
    python3 -m scripts.presentation_figures.cgm_overview [--patient_id ID]
                                          [--n_days 7]
                                          [--out_dir DIR]

Output
------
    scripts/presentation_figures/output/cgm_overview/  (git-ignored; --out_dir to change)
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
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.patient_selection import resolve_patient_id
from src.preprocessing.load_data import load_patient_window
from src.preprocessing.timestep import detect_dominant_sampling_interval
from src.episodes.episode_config import (
    H_HYPO, H_CONTEXT, H_HYPER,
    NIGHT_START_H, NIGHT_END_H,
    GAP_FACTOR,
)
from src.config import TIME_COL, GLUCOSE_COL
from src.visualization.style import apply_publication_style, save_figure

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PATIENT_ID = None   # None = automatic: patient with the most SLH episodes
N_DAYS             = 7
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "cgm_overview")

# ---------------------------------------------------------------------------
# Presentation colour palette
# ---------------------------------------------------------------------------
# Zone fill colours  (horizontal glucose bands)
COLOR_ZONE_HYPO    = "#FFCDD2"   # light red    — below 70
COLOR_ZONE_NORMAL  = "#E8F5E9"   # light green  — 70–140 (healthy range)
COLOR_ZONE_CAUTION = "#FFF9C4"   # light yellow — 140–180 (elevated, above healthy)
COLOR_ZONE_HYPER   = "#FFE0B2"   # light orange — above 180

# Threshold line colours
COLOR_LINE_HYPO    = "#C62828"   # deep red
COLOR_LINE_NORMAL  = "#2E7D32"   # deep green
COLOR_LINE_HYPER   = "#E65100"   # deep orange

# CGM signal
COLOR_TRACE        = "#1C1C1C"   # near-black
COLOR_NIGHT        = "#BDBDBD"   # mid grey for night overlay

# Y-axis bounds
Y_MIN, Y_MAX = 40, 380

# Figure dimensions for 16:9 slides  (wide format)
FIG_W, FIG_H = 14.0, 4.8

# Presentation font size base
FS = 12


# =============================================================================
# HELPERS
# =============================================================================

def _split_by_gaps(df: pd.DataFrame, gap_min: float) -> list[pd.DataFrame]:
    """Split a time-sorted DataFrame into gap-free segments."""
    if len(df) < 2:
        return [df]
    diffs    = df[TIME_COL].diff().dt.total_seconds() / 60
    gap_mask = diffs > gap_min
    gap_idxs = df.index[gap_mask].tolist()
    if not gap_idxs:
        return [df]
    segments, start = [], df.index[0]
    for gi in gap_idxs:
        seg = df.loc[start : gi - 1]
        if len(seg):
            segments.append(seg)
        start = gi
    final = df.loc[start:]
    if len(final):
        segments.append(final)
    return segments


def _shade_nights(ax, t_zero: pd.Timestamp,
                  t_start: pd.Timestamp, t_end: pd.Timestamp) -> None:
    """Add semi-transparent night bands (relative hours on x-axis)."""
    cur = t_start.normalize()
    end = t_end.normalize() + timedelta(days=1)
    while cur <= end:
        for h_lo, h_hi in [(0, NIGHT_END_H), (NIGHT_START_H, 24)]:
            ns = pd.Timestamp(datetime.combine(cur.date(), _dtime(h_lo % 24, 0)))
            ne = pd.Timestamp(datetime.combine(cur.date(), _dtime(0, 0))) + pd.Timedelta(hours=h_hi)
            if ne > t_start and ns < t_end:
                x0 = (max(ns, t_start) - t_zero).total_seconds() / 3600
                x1 = (min(ne, t_end)   - t_zero).total_seconds() / 3600
                ax.axvspan(x0, x1,
                           color=COLOR_NIGHT, alpha=0.30, zorder=1, linewidth=0)
        cur += timedelta(days=1)


def select_best_window(df: pd.DataFrame, n_days: int = N_DAYS):
    """
    Choose an n_days consecutive window that maximises visible excursions
    outside the 70–180 mg/dL target band (hypo + hyper events), so the
    figure illustrates all three glucose zones.

    Falls back to the first n_days if the dataset is too short.
    """
    t_first = df[TIME_COL].min()
    t_last  = df[TIME_COL].max()
    d_first = t_first.normalize()
    n_available = int((t_last - d_first).total_seconds() / 86400) + 1

    if n_available <= n_days:
        return d_first, d_first + pd.Timedelta(days=n_days)

    best_score, best_start = -1, d_first

    for i in range(n_available - n_days + 1):
        ws = d_first + pd.Timedelta(days=i)
        we = ws + pd.Timedelta(days=n_days)
        mask  = (df[TIME_COL] >= ws) & (df[TIME_COL] < we)
        gvals = df.loc[mask, GLUCOSE_COL].dropna()
        if gvals.empty:
            continue
        # Reward hypo events strongly, hyper events moderately
        n_hypo  = (gvals < H_HYPO).sum()
        n_hyper = (gvals > H_HYPER).sum()
        score   = 3 * n_hypo + n_hyper
        if score > best_score:
            best_score, best_start = score, ws

    return best_start, best_start + pd.Timedelta(days=n_days)


# =============================================================================
# MAIN PLOT
# =============================================================================

def plot_cgm_overview(patient_id: str, df: pd.DataFrame,
                      window_start: pd.Timestamp, window_end: pd.Timestamp,
                      timestep_min: float) -> plt.Figure:
    """
    Build and return the 7-day CGM overview figure.

    Parameters
    ----------
    patient_id : str
    df : pd.DataFrame
        Full patient data (TIME_COL, GLUCOSE_COL).
    window_start, window_end : pd.Timestamp
        Exact [start, end) of the display window.
    timestep_min : float
        Dominant CGM sampling interval (for gap detection).
    """
    # ── Apply a slightly upscaled publication style ──────────────────────────
    apply_publication_style()
    matplotlib.rcParams.update({
        "font.size":        FS,
        "axes.titlesize":   FS + 2,
        "axes.labelsize":   FS + 1,
        "xtick.labelsize":  FS,
        "ytick.labelsize":  FS,
        "legend.fontsize":  FS - 1,
    })

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    t_zero = window_start
    x_max  = (window_end - window_start).total_seconds() / 3600  # hours

    # ── 1.  Glucose zone fills  ──────────────────────────────────────────────
    # Draw from bottom to top so colours are layered correctly.
    zone_specs = [
        (Y_MIN,    H_HYPO,    COLOR_ZONE_HYPO,    "Hypoglycaemia  (< 70 mg/dL)"),
        (H_HYPO,   H_CONTEXT, COLOR_ZONE_NORMAL,  f"Normal range for healthy people  (70–{H_CONTEXT} mg/dL)"),
        (H_CONTEXT,H_HYPER,   COLOR_ZONE_CAUTION, f"Caution zone  ({H_CONTEXT}–{H_HYPER} mg/dL)"),
        (H_HYPER,  Y_MAX,     COLOR_ZONE_HYPER,   f"Hyperglycaemia  (> {H_HYPER} mg/dL)"),
    ]
    for y_lo, y_hi, color, label in zone_specs:
        ax.axhspan(y_lo, y_hi, color=color, alpha=1.0, zorder=0, label=label)

    # ── 2.  Night shading  ──────────────────────────────────────────────────
    _shade_nights(ax, t_zero, window_start, window_end)

    # ── 3.  CGM trace  ──────────────────────────────────────────────────────
    mask      = (df[TIME_COL] >= window_start) & (df[TIME_COL] < window_end)
    df_win    = df[mask].sort_values(TIME_COL).reset_index(drop=True)
    gap_min   = timestep_min * GAP_FACTOR

    segments = _split_by_gaps(df_win, gap_min)
    for seg in segments:
        x_h = (seg[TIME_COL] - t_zero).dt.total_seconds() / 3600
        ax.plot(x_h, seg[GLUCOSE_COL],
                color=COLOR_TRACE, lw=1.4, alpha=0.90, zorder=3)

    # ── 4.  Threshold lines  ─────────────────────────────────────────────────
    thresholds = [
        (H_HYPO,    "--", COLOR_LINE_HYPO,   "lw=1.8",
         f"Hypo threshold  {H_HYPO} mg/dL"),
        (H_CONTEXT, ":",  COLOR_LINE_NORMAL, "lw=2.0",
         f"Healthy upper limit  {H_CONTEXT} mg/dL"),
        (H_HYPER,   "--", COLOR_LINE_HYPER,  "lw=1.8",
         f"Hyper threshold  {H_HYPER} mg/dL"),
    ]
    for val, ls, color, _, lbl in thresholds:
        lw = 2.0 if ls == ":" else 1.8
        ax.axhline(val, ls=ls, lw=lw, color=color, alpha=0.95, zorder=4)

    # ── 5.  Threshold labels (right margin) ──────────────────────────────────
    label_specs = [
        (H_HYPO,    COLOR_LINE_HYPO,   f"{H_HYPO} mg/dL\n(hypo)"),
        (H_CONTEXT, COLOR_LINE_NORMAL, f"{H_CONTEXT} mg/dL\n(healthy)"),
        (H_HYPER,   COLOR_LINE_HYPER,  f"{H_HYPER} mg/dL\n(hyper)"),
    ]
    for val, color, text in label_specs:
        ax.annotate(
            text,
            xy=(x_max, val), xytext=(6, 0),
            textcoords="offset points",
            fontsize=FS - 2, color=color, va="center",
            fontweight="bold", annotation_clip=False,
        )

    # ── 6.  Day separator lines  ─────────────────────────────────────────────
    n_days = int(x_max / 24)
    for d in range(1, n_days + 1):
        ax.axvline(d * 24, color="#9E9E9E", lw=0.7, ls="-", alpha=0.6, zorder=2)

    # ── 7.  Axes formatting  ─────────────────────────────────────────────────
    ax.set_xlim(0, x_max)
    ax.set_ylim(Y_MIN, Y_MAX)

    # X-axis: major ticks every 24 h (day boundary), minor every 6 h
    def day_formatter(x, pos):
        if x == 0:
            return "Day 1"
        d = int(x / 24)
        r = x % 24
        if r == 0:
            return f"Day {d + 1}"
        return ""

    ax.xaxis.set_major_locator(mticker.MultipleLocator(24))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(6))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(day_formatter))
    ax.tick_params(axis="x", which="major", pad=6)

    # Y-axis: annotate all key glucose values
    yticks = [H_HYPO, 100, H_CONTEXT, H_HYPER, 250, 300]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(v) for v in yticks])

    ax.set_xlabel("Time", fontsize=FS + 1, labelpad=6)
    ax.set_ylabel("Blood glucose (mg/dL)", fontsize=FS + 1, labelpad=6)

    # ── 8.  Legend  ──────────────────────────────────────────────────────────
    # Zone patches + night patch
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    legend_handles = [
        mpatches.Patch(facecolor=COLOR_ZONE_HYPO,    label=f"Hypoglycaemia  (< {H_HYPO} mg/dL)"),
        mpatches.Patch(facecolor=COLOR_ZONE_NORMAL,  label=f"Normal range — healthy people  (70–{H_CONTEXT} mg/dL)"),
        mpatches.Patch(facecolor=COLOR_ZONE_CAUTION, label=f"Above healthy, below hyper  ({H_CONTEXT}–{H_HYPER} mg/dL)"),
        mpatches.Patch(facecolor=COLOR_ZONE_HYPER,   label=f"Hyperglycaemia  (> {H_HYPER} mg/dL)"),
        mpatches.Patch(facecolor=COLOR_NIGHT, alpha=0.55,
                       label=f"Night  ({NIGHT_START_H:02d}:00–{NIGHT_END_H:02d}:00)"),
        Line2D([0], [0], color=COLOR_TRACE, lw=1.4, label="CGM signal"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=FS - 2,
        ncol=2,
        framealpha=0.92,
        columnspacing=0.8,
        handlelength=1.6,
        borderpad=0.6,
    )

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    # Light horizontal grid (only at key glucose values, very subtle)
    ax.yaxis.grid(True, ls=":", lw=0.4, color="#CCCCCC", zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout(rect=[0, 0, 0.88, 1.0])  # leave right margin for labels

    return fig


# =============================================================================
# PIPELINE
# =============================================================================

def generate(patient_id: str, n_days: int, output_dir: str) -> None:
    """Full pipeline: load → select window → plot → save."""
    df = load_patient_window(patient_id)
    if df is None:
        print(f"[!] Could not load data for {patient_id}")
        return

    print(f"  Loaded {len(df):,} data points for {patient_id}")

    timestep_min             = detect_dominant_sampling_interval(df)
    window_start, window_end = select_best_window(df, n_days)

    print(f"  Selected window : {window_start.date()} → {window_end.date()}")
    print(f"  Sampling        : {timestep_min:.0f} min")

    # Data coverage in window
    mask = (df[TIME_COL] >= window_start) & (df[TIME_COL] < window_end)
    df_w = df[mask]
    gvals = df_w[GLUCOSE_COL].dropna()
    pct_hypo  = 100 * (gvals < H_HYPO).mean()
    pct_hyper = 100 * (gvals > H_HYPER).mean()
    pct_normal= 100 * ((gvals >= H_HYPO) & (gvals <= H_CONTEXT)).mean()
    pct_caution = 100 * ((gvals > H_CONTEXT) & (gvals <= H_HYPER)).mean()
    print(f"  Glucose distribution in window:")
    print(f"    Hypo   (< {H_HYPO}):        {pct_hypo:.1f}%")
    print(f"    Normal (70–{H_CONTEXT}): {pct_normal:.1f}%")
    print(f"    Caution({H_CONTEXT}–{H_HYPER}):  {pct_caution:.1f}%")
    print(f"    Hyper  (> {H_HYPER}):       {pct_hyper:.1f}%")

    fig = plot_cgm_overview(patient_id, df, window_start, window_end, timestep_min)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{patient_id}_cgm_overview_{n_days}d")
    save_figure(fig, out_path)
    print(f"  Saved → {out_path}.pdf / .svg")


# =============================================================================
# CLI
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="7-day CGM overview — presentation figure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--patient_id", default=DEFAULT_PATIENT_ID,
                        help="Patient ID (default: patient with the most SLH episodes)")
    parser.add_argument("--n_days", type=int, default=N_DAYS,
                        help="Number of consecutive days to display")
    parser.add_argument("--out_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory")
    return parser


def main():
    args = parse_arguments().parse_args()
    args.patient_id = resolve_patient_id(args.patient_id)

    print("\n" + "=" * 60)
    print("  CGM Overview — Presentation Figure")
    print("=" * 60)
    print(f"  Patient : {args.patient_id}")
    print(f"  Days    : {args.n_days}")
    print(f"  Output  : {args.out_dir}")
    print("=" * 60 + "\n")

    generate(args.patient_id, args.n_days, args.out_dir)

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
