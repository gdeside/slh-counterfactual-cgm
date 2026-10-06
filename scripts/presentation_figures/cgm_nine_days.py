"""
scripts/presentation_figures/cgm_nine_days.py
=============================================

Generates two standalone presentation figures for a single patient
over the same 9-day window:

  Figure 1  (suffix *_raw*)
    Raw CGM trace with the 70 mg/dL and 180 mg/dL threshold lines only.
    No background zone colouring — clean baseline view.

  Figure 2  (suffix *_c1_episodes*)
    Identical CGM trace + thresholds, with every hyperglycaemia episode
    that satisfies criterion C1 (duration in [D_MIN_MIN, D_MAX_MIN] min,
    i.e. 10–60 min by default) highlighted as a coloured vertical span.
    The trace segment inside each episode is drawn in the highlight colour.

Both figures share the same relative time axis (no absolute timestamps)
and the same night shading.

Usage
-----
    python3 -m scripts.presentation_figures.cgm_nine_days [--patient_id ID]
                                           [--n_days 9]
                                           [--out_dir DIR]

Output
------
    scripts/presentation_figures/output/cgm_nine_days/  (git-ignored; --out_dir to change)
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
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.patient_selection import resolve_patient_id
from src.preprocessing.load_data import load_patient_window
from src.preprocessing.timestep import detect_dominant_sampling_interval
from src.episodes.detection import detect_hyperglycemia_episodes
from src.episodes.episode_config import (
    H_HYPER, H_HYPO,
    D_MIN_MIN, D_MAX_MIN,
    GAP_FACTOR,
    NIGHT_START_H, NIGHT_END_H,
)
from src.config import TIME_COL, GLUCOSE_COL
from src.visualization.style import apply_publication_style, save_figure
from src.visualization.colors import (
    COLOR_THRESHOLD_HYPER, COLOR_THRESHOLD_HYPO,
    COLOR_SLH,           # navy  — C1 episode highlight
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PATIENT_ID = None   # None = automatic: patient with the most SLH episodes
DEFAULT_N_DAYS     = 9
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "cgm_nine_days")

# ---------------------------------------------------------------------------
# Figure geometry
# ---------------------------------------------------------------------------
FIG_H = 4.8          # height constant; width scales with n_days

# Presentation font sizes
FS_LABEL  = 14
FS_TICK   = 12
FS_LEGEND = 11
FS_ANNOT  = 11

# CGM trace colour (no-episode portions)
COLOR_TRACE = "#1C1C1C"

# Night shading
COLOR_NIGHT = "#BDBDBD"
ALPHA_NIGHT = 0.28

# C1 episode highlight
COLOR_C1        = COLOR_SLH      # navy
ALPHA_C1_SPAN   = 0.18           # background span
ALPHA_C1_TRACE  = 1.0            # trace segment inside episode

# Y-axis
Y_MIN, Y_MAX = 40, 370


# =============================================================================
# HELPERS
# =============================================================================

def _split_by_gaps(df: pd.DataFrame, gap_min: float) -> list[pd.DataFrame]:
    """Return gap-free segments of *df* (TIME_COL sorted)."""
    if len(df) < 2:
        return [df]
    diffs    = df[TIME_COL].diff().dt.total_seconds() / 60
    gap_idxs = df.index[diffs > gap_min].tolist()
    if not gap_idxs:
        return [df]
    segs, start = [], df.index[0]
    for gi in gap_idxs:
        seg = df.loc[start: gi - 1]
        if len(seg):
            segs.append(seg)
        start = gi
    tail = df.loc[start:]
    if len(tail):
        segs.append(tail)
    return segs


def _to_rel_h(timestamps: pd.Series, t_zero: pd.Timestamp) -> np.ndarray:
    """Convert timestamps to fractional hours since *t_zero*."""
    return (timestamps - t_zero).dt.total_seconds() / 3600


def _shade_nights(ax, t_zero, t_start, t_end) -> None:
    """Shade night hours (NIGHT_START_H → NIGHT_END_H) on a relative-hour axis."""
    cur = t_start.normalize()
    end = t_end.normalize() + timedelta(days=1)
    while cur <= end:
        for h_lo, h_hi in [(0, NIGHT_END_H), (NIGHT_START_H, 24)]:
            ns = pd.Timestamp(datetime.combine(cur.date(), _dtime(h_lo % 24, 0)))
            ne = pd.Timestamp(datetime.combine(cur.date(), _dtime(0, 0))) + pd.Timedelta(hours=h_hi)
            if ne > t_start and ns < t_end:
                x0 = (max(ns, t_start) - t_zero).total_seconds() / 3600
                x1 = (min(ne, t_end)   - t_zero).total_seconds() / 3600
                ax.axvspan(x0, x1, color=COLOR_NIGHT, alpha=ALPHA_NIGHT,
                           zorder=0, linewidth=0)
        cur += timedelta(days=1)


# =============================================================================
# WINDOW SELECTION
# =============================================================================

def select_best_window(df: pd.DataFrame, c1_episodes: list[dict],
                       n_days: int = DEFAULT_N_DAYS):
    """
    Pick the *n_days* consecutive-day window that contains the most
    C1-passing episodes (so the highlighted figure is informative).
    Falls back to the first *n_days* of data when the dataset is short.
    """
    t_first = df[TIME_COL].min().normalize()
    t_last  = df[TIME_COL].max()
    n_avail = int((t_last - t_first).total_seconds() / 86400) + 1

    if n_avail <= n_days:
        return t_first, t_first + pd.Timedelta(days=n_days)

    best_score, best_start = -1, t_first
    for i in range(n_avail - n_days + 1):
        ws = t_first + pd.Timedelta(days=i)
        we = ws   + pd.Timedelta(days=n_days)
        score = sum(
            1 for ep in c1_episodes
            if ep["onset_time"] >= ws and ep["offset_time"] <= we
        )
        if score > best_score:
            best_score, best_start = score, ws

    return best_start, best_start + pd.Timedelta(days=n_days)


# =============================================================================
# SHARED AXIS SETUP
# =============================================================================

def _format_x_axis(ax, x_max: float, n_days: int) -> None:
    """Apply common relative-time x-axis formatting to *ax*."""
    def _day_label(x, pos):
        if x == 0:
            return "Day 1"
        d, r = divmod(x, 24)
        return f"Day {int(d) + 1}" if r == 0 else ""

    ax.xaxis.set_major_locator(mticker.MultipleLocator(24))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(6))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_day_label))
    ax.tick_params(axis="x", which="major", pad=5, labelsize=FS_TICK)
    ax.set_xlim(0, x_max)

    # Vertical day-separator lines
    for d in range(1, n_days + 1):
        ax.axvline(d * 24, color="#BBBBBB", lw=0.7, ls="-", alpha=0.55, zorder=1)


def _format_y_axis(ax) -> None:
    yticks = [H_HYPO, 100, 140, H_HYPER, 250, 300]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(v) for v in yticks], fontsize=FS_TICK)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_ylabel("Glucose (mg/dL)", fontsize=FS_LABEL, labelpad=6)
    ax.yaxis.grid(True, ls=":", lw=0.5, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_thresholds(ax, x_max: float) -> None:
    """Draw 70 and 180 mg/dL threshold lines with right-margin labels."""
    ax.axhline(H_HYPER, color=COLOR_THRESHOLD_HYPER,
               lw=1.6, ls="--", alpha=0.90, zorder=5)
    ax.axhline(H_HYPO,  color=COLOR_THRESHOLD_HYPO,
               lw=1.6, ls="--", alpha=0.90, zorder=5)
    for val, color, text in [
        (H_HYPER, COLOR_THRESHOLD_HYPER, f"{H_HYPER} mg/dL"),
        (H_HYPO,  COLOR_THRESHOLD_HYPO,  f"{H_HYPO} mg/dL"),
    ]:
        ax.annotate(
            text,
            xy=(x_max, val), xytext=(5, 0),
            textcoords="offset points",
            fontsize=FS_ANNOT, color=color, fontweight="bold",
            va="center", annotation_clip=False,
        )


def _draw_cgm_trace(ax, df_win: pd.DataFrame, t_zero: pd.Timestamp,
                    gap_min: float, color: str = COLOR_TRACE,
                    lw: float = 1.3, alpha: float = 0.90, zorder: int = 3) -> None:
    """Plot gap-safe CGM trace on *ax* using relative hours."""
    for seg in _split_by_gaps(df_win, gap_min):
        xh = _to_rel_h(seg[TIME_COL], t_zero)
        ax.plot(xh, seg[GLUCOSE_COL], color=color, lw=lw, alpha=alpha, zorder=zorder)


# =============================================================================
# FIGURE 1 — raw 9-day CGM
# =============================================================================

def _fig_raw(df_win: pd.DataFrame, t_zero: pd.Timestamp,
             x_max: float, n_days: int, gap_min: float) -> plt.Figure:
    """Build Figure 1: raw CGM trace + threshold lines only."""
    fig_w = max(12, n_days * 1.7)
    fig, ax = plt.subplots(figsize=(fig_w, FIG_H))

    _shade_nights(ax, t_zero, t_zero, t_zero + pd.Timedelta(hours=x_max))
    _draw_cgm_trace(ax, df_win, t_zero, gap_min)
    _draw_thresholds(ax, x_max)
    _format_x_axis(ax, x_max, n_days)
    _format_y_axis(ax)

    ax.set_xlabel("Time", fontsize=FS_LABEL, labelpad=6)

    # Minimal legend
    ax.legend(
        handles=[
            Line2D([0], [0], color=COLOR_TRACE, lw=1.3, label="CGM signal"),
            Line2D([0], [0], color=COLOR_THRESHOLD_HYPER, lw=1.6, ls="--",
                   label=f"Hyper  ({H_HYPER} mg/dL)"),
            Line2D([0], [0], color=COLOR_THRESHOLD_HYPO, lw=1.6, ls="--",
                   label=f"Hypo  ({H_HYPO} mg/dL)"),
            mpatches.Patch(facecolor=COLOR_NIGHT, alpha=0.50,
                           label=f"Night  ({NIGHT_START_H:02d}:00–{NIGHT_END_H:02d}:00)"),
        ],
        loc="upper right", fontsize=FS_LEGEND,
        framealpha=0.90, handlelength=1.8,
    )

    fig.tight_layout(rect=[0, 0, 0.93, 1.0])
    return fig


# =============================================================================
# FIGURE 2 — 9-day CGM with C1 episodes highlighted
# =============================================================================

def _fig_c1(df_win: pd.DataFrame, t_zero: pd.Timestamp,
            x_max: float, n_days: int, gap_min: float,
            c1_window: list[dict]) -> plt.Figure:
    """
    Build Figure 2: same CGM trace + C1 episode highlights.

    C1 episodes = hyperglycaemia episodes with duration in [D_MIN_MIN, D_MAX_MIN].
    """
    fig_w = max(12, n_days * 1.7)
    fig, ax = plt.subplots(figsize=(fig_w, FIG_H))

    _shade_nights(ax, t_zero, t_zero, t_zero + pd.Timedelta(hours=x_max))

    # ── Draw grey baseline trace (full window) ────────────────────────────────
    _draw_cgm_trace(ax, df_win, t_zero, gap_min,
                    color=COLOR_TRACE, lw=1.3, alpha=0.55, zorder=3)

    # ── Draw C1 episode highlights ────────────────────────────────────────────
    for ep in c1_window:
        x_on  = _to_rel_h(pd.Series([ep["onset_time"]]),  t_zero).item()
        x_off = _to_rel_h(pd.Series([ep["offset_time"]]), t_zero).item()

        # Background span
        ax.axvspan(x_on, x_off, color=COLOR_C1, alpha=ALPHA_C1_SPAN, zorder=2)

        # Onset / offset markers
        for xv in (x_on, x_off):
            ax.axvline(xv, color=COLOR_C1, lw=0.8, ls="--", alpha=0.45, zorder=3)

        # Coloured trace inside the episode
        ep_mask = (df_win[TIME_COL] >= ep["onset_time"]) & \
                  (df_win[TIME_COL] <= ep["offset_time"])
        ep_seg  = df_win[ep_mask].sort_values(TIME_COL)
        if not ep_seg.empty:
            xh = _to_rel_h(ep_seg[TIME_COL], t_zero)
            ax.plot(xh, ep_seg[GLUCOSE_COL],
                    color=COLOR_C1, lw=2.0, alpha=ALPHA_C1_TRACE, zorder=4)

    _draw_thresholds(ax, x_max)
    _format_x_axis(ax, x_max, n_days)
    _format_y_axis(ax)

    ax.set_xlabel("Time", fontsize=FS_LABEL, labelpad=6)

    n_c1 = len(c1_window)
    ax.legend(
        handles=[
            Line2D([0], [0], color=COLOR_TRACE, lw=1.3, label="CGM signal"),
            mpatches.Patch(facecolor=COLOR_C1, alpha=0.35,
                           label=f"C1 episode  ({D_MIN_MIN}–{D_MAX_MIN} min)  n = {n_c1}"),
            Line2D([0], [0], color=COLOR_THRESHOLD_HYPER, lw=1.6, ls="--",
                   label=f"Hyper  ({H_HYPER} mg/dL)"),
            Line2D([0], [0], color=COLOR_THRESHOLD_HYPO, lw=1.6, ls="--",
                   label=f"Hypo  ({H_HYPO} mg/dL)"),
            mpatches.Patch(facecolor=COLOR_NIGHT, alpha=0.50,
                           label=f"Night  ({NIGHT_START_H:02d}:00–{NIGHT_END_H:02d}:00)"),
        ],
        loc="upper right", fontsize=FS_LEGEND,
        framealpha=0.90, handlelength=1.8,
    )

    fig.tight_layout(rect=[0, 0, 0.93, 1.0])
    return fig


# =============================================================================
# PIPELINE
# =============================================================================

def generate(patient_id: str, n_days: int, output_dir: str) -> None:
    """Full pipeline: load → detect → select window → draw → save."""
    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_patient_window(patient_id)
    if df is None:
        print(f"[!] No data found for {patient_id}")
        return
    print(f"  Loaded {len(df):,} data points")

    timestep_min = detect_dominant_sampling_interval(df)
    gap_min      = timestep_min * GAP_FACTOR
    print(f"  Sampling interval : {timestep_min:.0f} min")

    # ── Detect C1 episodes (all data, then filter to window later) ────────────
    all_episodes = detect_hyperglycemia_episodes(df)
    c1_episodes  = [
        ep for ep in all_episodes
        if D_MIN_MIN <= ep["duration_min"] <= D_MAX_MIN
    ]
    print(f"  Raw episodes      : {len(all_episodes)}")
    print(f"  C1-passing        : {len(c1_episodes)}  ({D_MIN_MIN}–{D_MAX_MIN} min)")

    # ── Best 9-day window ─────────────────────────────────────────────────────
    t_start, t_end = select_best_window(df, c1_episodes, n_days)
    x_max = n_days * 24.0   # hours
    t_zero = t_start
    print(f"  Window            : {t_start.date()} → {t_end.date()}")

    # C1 episodes inside the window
    c1_window = [
        ep for ep in c1_episodes
        if ep["onset_time"] >= t_start and ep["offset_time"] <= t_end
    ]
    print(f"  C1 in window      : {len(c1_window)}")

    # ── Slice CGM data to window ──────────────────────────────────────────────
    mask   = (df[TIME_COL] >= t_start) & (df[TIME_COL] < t_end)
    df_win = df[mask].sort_values(TIME_COL).reset_index(drop=True)

    # ── Style ─────────────────────────────────────────────────────────────────
    apply_publication_style()
    matplotlib.rcParams.update({
        "font.size":      FS_TICK,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize":FS_TICK,
        "ytick.labelsize":FS_TICK,
        "legend.fontsize":FS_LEGEND,
    })

    os.makedirs(output_dir, exist_ok=True)

    # ── Figure 1 — raw ────────────────────────────────────────────────────────
    fig1 = _fig_raw(df_win, t_zero, x_max, n_days, gap_min)
    out1 = os.path.join(output_dir, f"{patient_id}_cgm_{n_days}d_raw")
    save_figure(fig1, out1)
    print(f"  Saved Fig 1 → {out1}.pdf / .svg")

    # ── Figure 2 — C1 highlights ──────────────────────────────────────────────
    fig2 = _fig_c1(df_win, t_zero, x_max, n_days, gap_min, c1_window)
    out2 = os.path.join(output_dir, f"{patient_id}_cgm_{n_days}d_c1_episodes")
    save_figure(fig2, out2)
    print(f"  Saved Fig 2 → {out2}.pdf / .svg")


# =============================================================================
# CLI
# =============================================================================

def parse_arguments():
    p = argparse.ArgumentParser(
        description="9-day CGM — raw + C1 episodes, two separate figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--patient_id", default=DEFAULT_PATIENT_ID)
    p.add_argument("--n_days",     type=int, default=DEFAULT_N_DAYS,
                   help="Number of consecutive days to display")
    p.add_argument("--out_dir",    default=DEFAULT_OUTPUT_DIR)
    return p


def main():
    args = parse_arguments().parse_args()
    args.patient_id = resolve_patient_id(args.patient_id)

    print("\n" + "=" * 60)
    print("  CGM — Raw + C1 Episodes  (presentation figures)")
    print("=" * 60)
    print(f"  Patient : {args.patient_id}")
    print(f"  Days    : {args.n_days}")
    print(f"  C1 rule : {D_MIN_MIN}–{D_MAX_MIN} min above {H_HYPER} mg/dL")
    print(f"  Output  : {args.out_dir}")
    print("=" * 60 + "\n")

    generate(args.patient_id, args.n_days, args.out_dir)

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
