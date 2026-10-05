"""
scripts/presentation_figures/nine_days_two_patients.py
======================================================

Side-by-side 9-day CGM overviews for two patients.

Generates TWO figures:

  Figure 1 (suffix *_raw*):
    · Raw CGM trace with threshold lines only (70 & 180 mg/dL)
    · No episode highlighting — clean baseline view
    · Night shading and day separators

  Figure 2 (suffix *_with_duration*):
    · Same CGM trace + thresholds
    · C1 episodes (duration 10–60 min) highlighted as coloured spans
    · Trace segments inside episodes drawn in highlight colour
    · Legend shows "Duration (10–60 min)" instead of "C1"
    · Night shading and day separators

Both figures: 16:9 presentation-ready layout, side-by-side panels.

Usage
-----
    python3 -m scripts.presentation_figures.nine_days_two_patients \
        [--patient_1 ID] \
        [--patient_2 ID] \
        [--n_days 9] \
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
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.patient_selection import resolve_patient_pair
from src.preprocessing.load_data import load_patient_window
from src.preprocessing.timestep import detect_dominant_sampling_interval
from src.episodes.detection import detect_hyperglycemia_episodes
from src.episodes.criteria import classify_episodes
from src.episodes.episode_config import (
    H_HYPER, H_HYPO,
    D_MIN_MIN, D_MAX_MIN,
    GAP_FACTOR,
    NIGHT_START_H, NIGHT_END_H,
    LABEL_SLH,
)
from src.config import TIME_COL, GLUCOSE_COL, COL_FLAG_TIME_ERROR, COL_FLAG_OUTLIER_GLUCOSE
from src.counterfactual.build_counterfactual import build_full_counterfactual_signal
from src.visualization.style import apply_publication_style, save_figure
from src.visualization.colors import (
    COLOR_THRESHOLD_HYPER, COLOR_THRESHOLD_HYPO,
    COLOR_SLH,           # navy — episode highlight
    OFFSET_COLORS,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PATIENT_1   = None
DEFAULT_PATIENT_2   = None
DEFAULT_N_DAYS      = 9
DEFAULT_OUTPUT_DIR  = os.path.join(
    os.path.dirname(__file__), "output", "nine_days_two_patients"
)

# ---------------------------------------------------------------------------
# Figure geometry — 16:9 with 2 columns
# ---------------------------------------------------------------------------
FIG_H = 8.0           # height — enlarged to fill presentation frame

# Presentation font sizes
FS_LABEL  = 15
FS_TICK   = 13
FS_LEGEND = 12
FS_ANNOT  = 12
FS_TITLE  = 16

# CGM trace colour (original / no-episode portions)
COLOR_TRACE    = "#1C1C1C"
COLOR_ORIGINAL = "#666666"   # lighter grey for G(t) in counterfactual view

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

def select_best_window(df: pd.DataFrame, n_days: int = DEFAULT_N_DAYS):
    """
    Pick the *n_days* consecutive-day window from the available data.
    Falls back to the first *n_days* of data when the dataset is short.
    """
    t_first = df[TIME_COL].min().normalize()
    t_last  = df[TIME_COL].max()
    n_avail = int((t_last - t_first).total_seconds() / 86400) + 1

    if n_avail <= n_days:
        return t_first, t_first + pd.Timedelta(days=n_days)

    return t_first, t_first + pd.Timedelta(days=n_days)


# =============================================================================
# AXIS SETUP
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


def _draw_thresholds(ax, x_max: float, show_labels: bool = True) -> None:
    """Draw 70 and 180 mg/dL threshold lines with optional right-margin labels."""
    ax.axhline(H_HYPER, color=COLOR_THRESHOLD_HYPER,
               lw=1.6, ls="--", alpha=0.90, zorder=5)
    ax.axhline(H_HYPO,  color=COLOR_THRESHOLD_HYPO,
               lw=1.6, ls="--", alpha=0.90, zorder=5)

    if show_labels:
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
# PLOT FUNCTION
# =============================================================================

def plot_cgm_panel(ax, df_win: pd.DataFrame, t_zero: pd.Timestamp,
                   patient_id: str, x_max: float, n_days: int,
                   gap_min: float, c1_episodes: list = None,
                   show_threshold_labels: bool = True,
                   g_ctf: np.ndarray = None, offset_min: float = None) -> None:
    """
    Plot a single 9-day CGM panel.

    When *g_ctf* is provided, draws G(t) (dashed, lighter) + C(t) (solid,
    colour-coded) in counterfactual mode instead of C1-episode highlighting.
    """
    if c1_episodes is None:
        c1_episodes = []

    _shade_nights(ax, t_zero, t_zero, t_zero + pd.Timedelta(hours=x_max))

    xh_all = _to_rel_h(df_win[TIME_COL], t_zero)

    if g_ctf is not None:
        # ── Counterfactual mode ──────────────────────────────────────────────
        color_ctf = OFFSET_COLORS.get(offset_min, "#0072B2")

        # Original trace: dashed, lighter
        for seg in _split_by_gaps(df_win, gap_min):
            xh = _to_rel_h(seg[TIME_COL], t_zero)
            ax.plot(xh, seg[GLUCOSE_COL],
                    color=COLOR_ORIGINAL, lw=0.9, ls="--", alpha=0.60, zorder=3,
                    label="G(t)" if seg is _split_by_gaps(df_win, gap_min)[0] else "_")

        # Counterfactual trace: solid, colour-coded — same gap-splitting
        df_ctf_win = df_win.copy()
        df_ctf_win[GLUCOSE_COL] = g_ctf
        for seg in _split_by_gaps(df_ctf_win, gap_min):
            xh = _to_rel_h(seg[TIME_COL], t_zero)
            ax.plot(xh, seg[GLUCOSE_COL],
                    color=color_ctf, lw=1.1, ls="-", alpha=0.85, zorder=4)

        # Fake legend handles for this panel
        from matplotlib.lines import Line2D as _L2D
        ax.plot([], [], color=COLOR_ORIGINAL, lw=0.9, ls="--", alpha=0.70,
                label="G(t)")
        ax.plot([], [], color=color_ctf, lw=1.1, ls="-",
                label=f"C(t) δ={offset_min} min")
        ax.legend(fontsize=FS_LEGEND - 1, loc="upper right",
                  framealpha=0.88, edgecolor="#AAAAAA", fancybox=False)

    else:
        # ── Standard mode: grey trace + C1 episode highlighting ─────────────
        _draw_cgm_trace(ax, df_win, t_zero, gap_min,
                        color=COLOR_TRACE, lw=1.3, alpha=0.55, zorder=3)

        for ep in c1_episodes:
            x_on  = _to_rel_h(pd.Series([ep["onset_time"]]),  t_zero).item()
            x_off = _to_rel_h(pd.Series([ep["offset_time"]]), t_zero).item()
            ax.axvspan(x_on, x_off, color=COLOR_C1, alpha=ALPHA_C1_SPAN, zorder=2)
            for xv in (x_on, x_off):
                ax.axvline(xv, color=COLOR_C1, lw=0.8, ls="--", alpha=0.45, zorder=3)
            ep_mask = (df_win[TIME_COL] >= ep["onset_time"]) & \
                      (df_win[TIME_COL] <= ep["offset_time"])
            ep_seg  = df_win[ep_mask].sort_values(TIME_COL)
            if not ep_seg.empty:
                xh = _to_rel_h(ep_seg[TIME_COL], t_zero)
                ax.plot(xh, ep_seg[GLUCOSE_COL],
                        color=COLOR_C1, lw=2.0, alpha=ALPHA_C1_TRACE, zorder=4)

    _draw_thresholds(ax, x_max, show_labels=show_threshold_labels)
    _format_x_axis(ax, x_max, n_days)
    _format_y_axis(ax)

    ax.set_xlabel("Time", fontsize=FS_LABEL, labelpad=6)
    ax.set_title(f"Participant {patient_id}", fontsize=FS_TITLE, fontweight="bold", pad=12)


# =============================================================================
# FIGURE
# =============================================================================

def plot_two_cgm_nine_days(df1: pd.DataFrame, df2: pd.DataFrame,
                           patient_id_1: str, patient_id_2: str,
                           n_days: int = DEFAULT_N_DAYS,
                           show_episodes: bool = False) -> plt.Figure:
    """
    Build a 2-panel side-by-side 9-day CGM figure.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        Full CGM data for each patient.
    patient_id_1, patient_id_2 : str
        Patient identifiers.
    n_days : int
        Number of days to display.
    show_episodes : bool
        Whether to highlight Duration episodes (default False for raw only).

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ── Detect sampling intervals ─────────────────────────────────────────────
    ts1 = detect_dominant_sampling_interval(df1)
    ts2 = detect_dominant_sampling_interval(df2)
    gap_min_1 = ts1 * GAP_FACTOR
    gap_min_2 = ts2 * GAP_FACTOR

    # ── Detect hyperglycemia episodes and filter for C1 ─────────────────────────
    all_episodes_1 = detect_hyperglycemia_episodes(df1)
    all_episodes_2 = detect_hyperglycemia_episodes(df2)

    # Filter for C1 episodes (duration in [D_MIN_MIN, D_MAX_MIN])
    def get_c1_episodes(episodes):
        c1 = []
        for ep in episodes:
            dur_min = (ep["offset_time"] - ep["onset_time"]).total_seconds() / 60
            if D_MIN_MIN <= dur_min <= D_MAX_MIN:
                c1.append(ep)
        return c1

    c1_episodes_1 = get_c1_episodes(all_episodes_1)
    c1_episodes_2 = get_c1_episodes(all_episodes_2)

    # ── Select best 9-day windows ─────────────────────────────────────────────
    t_start_1, t_end_1 = select_best_window(df1, n_days)
    t_start_2, t_end_2 = select_best_window(df2, n_days)

    df_win_1 = df1[(df1[TIME_COL] >= t_start_1) & (df1[TIME_COL] < t_end_1)].copy()
    df_win_2 = df2[(df2[TIME_COL] >= t_start_2) & (df2[TIME_COL] < t_end_2)].copy()

    # Filter episodes to window
    c1_win_1 = [ep for ep in c1_episodes_1
                if ep["onset_time"] >= t_start_1 and ep["offset_time"] <= t_end_1]
    c1_win_2 = [ep for ep in c1_episodes_2
                if ep["onset_time"] >= t_start_2 and ep["offset_time"] <= t_end_2]

    x_max = n_days * 24

    # ── Style ─────────────────────────────────────────────────────────────────
    apply_publication_style()
    matplotlib.rcParams.update({
        "font.size": FS_LABEL,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
    })

    # ── Figure: 2 panels side by side ────────────────────────────────────────
    fig_w = 22.0
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, FIG_H))
    fig.subplots_adjust(left=0.05, right=0.96, top=0.93, bottom=0.14, wspace=0.20)

    # ── Plot left panel (Patient 1) ────────────────────────────────────────────
    c1_ep_1 = c1_win_1 if show_episodes else None
    plot_cgm_panel(axes[0], df_win_1, t_start_1, patient_id_1,
                   x_max, n_days, gap_min_1, c1_episodes=c1_ep_1,
                   show_threshold_labels=False)

    # ── Plot right panel (Patient 2) ───────────────────────────────────────────
    c1_ep_2 = c1_win_2 if show_episodes else None
    plot_cgm_panel(axes[1], df_win_2, t_start_2, patient_id_2,
                   x_max, n_days, gap_min_2, c1_episodes=c1_ep_2,
                   show_threshold_labels=True)

    # ── Shared legend (bottom center) ──────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color=COLOR_TRACE, lw=1.3, label="CGM signal"),
    ]

    # Add Duration only if showing episodes
    if show_episodes:
        legend_handles.append(
            mpatches.Patch(facecolor=COLOR_C1, alpha=0.35,
                           label=f"Duration  ({D_MIN_MIN}–{D_MAX_MIN} min)")
        )

    legend_handles.append(
        mpatches.Patch(facecolor=COLOR_NIGHT, alpha=0.50,
                       label=f"Night  ({NIGHT_START_H:02d}:00–{NIGHT_END_H:02d}:00)")
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        fontsize=FS_LEGEND,
        framealpha=0.90,
        ncol=len(legend_handles),
        bbox_to_anchor=(0.5, 0.01),
        columnspacing=1.5,
        handlelength=2.0,
        borderpad=0.6,
    )

    return fig


# =============================================================================
# COUNTERFACTUAL TWO-PATIENT FIGURE
# =============================================================================

def _prepare_valid(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if COL_FLAG_TIME_ERROR in df.columns:
        mask &= ~df[COL_FLAG_TIME_ERROR].fillna(False).astype(bool)
    if COL_FLAG_OUTLIER_GLUCOSE in df.columns:
        mask &= ~df[COL_FLAG_OUTLIER_GLUCOSE].fillna(False).astype(bool)
    return df[mask][[TIME_COL, GLUCOSE_COL]].dropna().sort_values(TIME_COL).reset_index(drop=True)


def _build_ctf_for_window(df: pd.DataFrame, offset_min: float,
                           t_start: pd.Timestamp, t_end: pd.Timestamp,
                           gap_min: float):
    """Return (df_win, g_ctf_win) — both aligned to the 9-day window."""
    timestep = detect_dominant_sampling_interval(df)
    raw_episodes = detect_hyperglycemia_episodes(df)
    cls_df = classify_episodes(df, raw_episodes)
    slh_df = cls_df[cls_df["classification"] == LABEL_SLH]
    slh_episodes = [
        {"onset_time": r["onset_time"], "offset_time": r["offset_time"]}
        for _, r in slh_df.iterrows()
    ]

    c_signal = build_full_counterfactual_signal(df, slh_episodes, offset_min, timestep)

    mask = (df[TIME_COL] >= t_start) & (df[TIME_COL] < t_end)
    df_win = df[mask].copy().reset_index(drop=True)
    g_ctf  = c_signal[mask.values].copy()

    # Blank across large gaps (same logic as cgm_7day_counterfactual)
    diffs = df_win[TIME_COL].diff().dt.total_seconds().fillna(0) / 60
    gap_idx = diffs > gap_min
    df_win.loc[gap_idx, GLUCOSE_COL] = np.nan
    g_ctf[gap_idx.values] = np.nan

    return df_win, g_ctf, timestep


def plot_two_cgm_nine_days_counterfactual(
    df1: pd.DataFrame, df2: pd.DataFrame,
    patient_id_1: str, patient_id_2: str,
    offset_min: float,
    n_days: int = DEFAULT_N_DAYS,
) -> plt.Figure:
    """1 row × 2 columns: each panel shows G(t) + C(t) over *n_days*."""
    ts1 = detect_dominant_sampling_interval(df1)
    ts2 = detect_dominant_sampling_interval(df2)

    t_start_1, t_end_1 = select_best_window(df1, n_days)
    t_start_2, t_end_2 = select_best_window(df2, n_days)

    df_win_1, g_ctf_1, _ = _build_ctf_for_window(
        df1, offset_min, t_start_1, t_end_1, ts1 * GAP_FACTOR)
    df_win_2, g_ctf_2, _ = _build_ctf_for_window(
        df2, offset_min, t_start_2, t_end_2, ts2 * GAP_FACTOR)

    x_max = n_days * 24

    apply_publication_style()
    matplotlib.rcParams.update({
        "font.size": FS_LABEL, "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
    })

    fig, axes = plt.subplots(1, 2, figsize=(24.0, FIG_H))
    fig.subplots_adjust(left=0.05, right=0.96, top=0.90, bottom=0.12, wspace=0.20)

    plot_cgm_panel(axes[0], df_win_1, t_start_1, patient_id_1,
                   x_max, n_days, ts1 * GAP_FACTOR,
                   show_threshold_labels=False,
                   g_ctf=g_ctf_1, offset_min=offset_min)

    plot_cgm_panel(axes[1], df_win_2, t_start_2, patient_id_2,
                   x_max, n_days, ts2 * GAP_FACTOR,
                   show_threshold_labels=True,
                   g_ctf=g_ctf_2, offset_min=offset_min)

    return fig


# =============================================================================
# PIPELINE
# =============================================================================

def generate(patient_id_1: str, patient_id_2: str, n_days: int,
             output_dir: str, offset_min: int = None) -> None:
    """Load data for two patients and generate 9-day figure(s).

    When *offset_min* is given, produces a counterfactual comparison figure.
    Otherwise produces the raw + duration-annotated figures.
    """
    print(f"\n  Loading Patient {patient_id_1}...")
    df1_raw = load_patient_window(patient_id_1)
    if df1_raw is None:
        print(f"[!] No data found for {patient_id_1}")
        return
    df1 = _prepare_valid(df1_raw)
    print(f"    {len(df1):,} data points")

    print(f"\n  Loading Patient {patient_id_2}...")
    df2_raw = load_patient_window(patient_id_2)
    if df2_raw is None:
        print(f"[!] No data found for {patient_id_2}")
        return
    df2 = _prepare_valid(df2_raw)
    print(f"    {len(df2):,} data points")

    os.makedirs(output_dir, exist_ok=True)

    if offset_min is not None:
        # ── Counterfactual figure ─────────────────────────────────────────────
        print(f"\n  Building counterfactual 9-day figure (δ={offset_min} min)...")
        fig = plot_two_cgm_nine_days_counterfactual(
            df1, df2, patient_id_1, patient_id_2, offset_min, n_days=n_days)
        out = os.path.join(
            output_dir,
            f"cgm_{n_days}d_{patient_id_1}_vs_{patient_id_2}_ctf_delta{offset_min}"
        )
        save_figure(fig, out)
        print(f"  Saved → {out}.pdf / .svg\n")
        return

    # ── Figure 1: Raw CGM only (no episode highlighting) ─────────────────────
    print(f"\n  Building Figure 1: Raw 9-day CGM (no Duration highlighting)...")
    fig_raw = plot_two_cgm_nine_days(df1_raw, df2_raw, patient_id_1, patient_id_2,
                                     n_days=n_days, show_episodes=False)
    out_raw = os.path.join(output_dir, f"cgm_{n_days}d_{patient_id_1}_vs_{patient_id_2}_raw")
    save_figure(fig_raw, out_raw)
    print(f"  Saved → {out_raw}.pdf / .svg")

    # ── Figure 2: With Duration episodes highlighted ────────────────────────
    print(f"\n  Building Figure 2: 9-day CGM with Duration information...")
    fig_duration = plot_two_cgm_nine_days(df1_raw, df2_raw, patient_id_1, patient_id_2,
                                          n_days=n_days, show_episodes=True)
    out_duration = os.path.join(output_dir, f"cgm_{n_days}d_{patient_id_1}_vs_{patient_id_2}_with_duration")
    save_figure(fig_duration, out_duration)
    print(f"  Saved → {out_duration}.pdf / .svg\n")


# =============================================================================
# CLI
# =============================================================================

def parse_arguments():
    p = argparse.ArgumentParser(
        description="Side-by-side 9-day CGM overview for two patients",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--patient_1", default=DEFAULT_PATIENT_1,
                   help="First patient ID (default: chosen automatically — most / fewest SLH episodes)")
    p.add_argument("--patient_2", default=DEFAULT_PATIENT_2,
                   help="Second patient ID (default: chosen automatically — most / fewest SLH episodes)")
    p.add_argument("--n_days", type=int, default=DEFAULT_N_DAYS,
                   help="Number of days to display (e.g. 9)")
    p.add_argument("--offset_min", type=int, default=None,
                   help="Zone-expansion offset δ in minutes — enables counterfactual mode")
    p.add_argument("--out_dir", default=DEFAULT_OUTPUT_DIR,
                   help="Output directory")
    return p


def main():
    args = parse_arguments().parse_args()
    args.patient_1, args.patient_2 = resolve_patient_pair(args.patient_1, args.patient_2)

    print("\n" + "=" * 65)
    mode = f"counterfactual δ={args.offset_min} min" if args.offset_min else "raw + duration"
    print(f"  9-Day CGM — Two Patients ({mode})")
    print("=" * 65)
    print(f"  Patient 1  : {args.patient_1}")
    print(f"  Patient 2  : {args.patient_2}")
    print(f"  Duration   : {args.n_days} days")
    if args.offset_min:
        print(f"  Offset δ   : {args.offset_min} min")
    print(f"  Output     : {args.out_dir}")
    print("=" * 65)

    generate(args.patient_1, args.patient_2, args.n_days, args.out_dir,
             offset_min=args.offset_min)

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
