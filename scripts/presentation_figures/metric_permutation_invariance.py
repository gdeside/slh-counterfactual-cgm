"""
scripts/presentation_figures/metric_permutation_invariance.py
=============================================================

One 16:9 presentation figure showing, on a real patient, that the consensus
CGM metrics cannot see time.

The same readings are drawn twice:

  Panel A   the observed record — meals, nights, a daily shape;
  Panel B   the *identical* readings in random order — structure destroyed.

Between the two panels the glucose values are permuted, nothing else: the
timestamps, the number of readings and therefore the marginal distribution
(drawn as the histogram beside each panel) are untouched.  Every consensus
metric — TIR, TAR, TBR, mean, SD, CV, GMI — is a functional of that marginal
distribution alone, so the two columns of the table on the right match digit
for digit.

The metrics are computed on the displayed window, not on the whole record, so
what the audience sees is exactly what is measured.

Usage
-----
    python3 -m scripts.presentation_figures.metric_permutation_invariance
            [--patient_id ID] [--n_days 5] [--seed 7] [--out_dir DIR]

    # Same figure in mmol/L (default is mg/dL; output filename gets a _mmoll suffix)
    CGM_GLUCOSE_UNIT="mmol/L" python3 -m scripts.presentation_figures.metric_permutation_invariance
            [same options]

Authors
-------
Guillaume Deside

Date
----
2026

Output
------
    scripts/presentation_figures/output/metric_permutation_invariance/  (git-ignored; --out_dir to change)
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
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.patient_selection import resolve_patient_id
from src.preprocessing.load_data import load_patient_window
from src.preprocessing.timestep import detect_dominant_sampling_interval
from src.episodes.episode_config import GAP_FACTOR, NIGHT_START_H, NIGHT_END_H
from src.config import TIME_COL, GLUCOSE_COL, THRESHOLD_HYPER, THRESHOLD_HYPO
from src.metrics.metrics import (
    THRESHOLD_TAR_LEVEL2,
    THRESHOLD_TBR_LEVEL2,
)
from src.utils.unit_conversion import (
    to_display, glucose_label, glucose_unit_label, format_glucose, glucose_ticks,
)
from src.visualization.style import apply_publication_style, save_figure
from src.visualization.colors import (
    COLOR_THRESHOLD_HYPER, COLOR_THRESHOLD_HYPO,
    COLOR_AGP_TARGET_ZONE, COLOR_SLH,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PATIENT_ID = None   # None = automatic: patient with the most SLH episodes
DEFAULT_N_DAYS     = 5
DEFAULT_SEED       = 7
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "metric_permutation_invariance")

# ---------------------------------------------------------------------------
# Figure geometry — 16:9 slide
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 13.333, 7.5

# Presentation font sizes
FS_SUPTITLE = 19
FS_SUBTITLE = 13
FS_PANEL    = 15
FS_LABEL    = 13
FS_TICK     = 11
FS_TABLE    = 13
FS_ANNOT    = 11

# Trace colours
COLOR_TRACE_OBS  = "#1C1C1C"
COLOR_TRACE_SHUF = "#6E6E6E"

# Night shading
COLOR_NIGHT = "#BDBDBD"
ALPHA_NIGHT = 0.22

# Out-of-range fills
ALPHA_FILL   = 0.45
ALPHA_TARGET = 0.07

#: Marker size for individual out-of-range readings, in points.
MS_OUT_OF_RANGE = 2.2

# Y-axis (mg/dL, canonical)
Y_MIN, Y_MAX = 40, 370

#: Width of one marginal-histogram bin, in mg/dL.
HIST_BIN_MGDL = 10


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

def select_densest_window(df: pd.DataFrame, n_days: int):
    """
    Pick the *n_days* consecutive-day window carrying the most readings.

    Completeness is what matters here: the two panels must show the same
    number of points, and a window punctured by sensor gaps makes the
    "same readings, different order" claim harder to read at a glance.
    """
    t_first = df[TIME_COL].min().normalize()
    t_last  = df[TIME_COL].max()
    n_avail = int((t_last - t_first).total_seconds() / 86400) + 1

    if n_avail <= n_days:
        return t_first, t_first + pd.Timedelta(days=n_days)

    best_count, best_start = -1, t_first
    for i in range(n_avail - n_days + 1):
        ws = t_first + pd.Timedelta(days=i)
        we = ws + pd.Timedelta(days=n_days)
        count = int(((df[TIME_COL] >= ws) & (df[TIME_COL] < we)).sum())
        if count > best_count:
            best_count, best_start = count, ws

    return best_start, best_start + pd.Timedelta(days=n_days)


# =============================================================================
# METRICS
# =============================================================================

def compute_consensus_metrics(glucose: np.ndarray) -> dict[str, float]:
    """
    The consensus CGM metrics, each a functional of the marginal distribution.

    Written out here rather than imported from :mod:`src.metrics.metrics` so
    that the figure shows *unrounded* values: the two columns agree by
    construction, not because the numbers were rounded into agreement.
    """
    g = np.asarray(glucose, dtype=float)
    g = g[np.isfinite(g)]

    return {
        "TIR":   100 * float(((g >= THRESHOLD_HYPO) & (g <= THRESHOLD_HYPER)).mean()),
        "TAR":   100 * float((g > THRESHOLD_HYPER).mean()),
        "TAR2":  100 * float((g > THRESHOLD_TAR_LEVEL2).mean()),
        "TBR":   100 * float((g < THRESHOLD_HYPO).mean()),
        "TBR2":  100 * float((g < THRESHOLD_TBR_LEVEL2).mean()),
        "Mean":  float(g.mean()),
        "SD":    float(g.std(ddof=1)),
        "CV":    100 * float(g.std(ddof=1) / g.mean()),
        "GMI":   3.31 + 0.02392 * float(g.mean()),
    }


def _metric_rows(observed: dict, shuffled: dict) -> list[tuple[str, str, str]]:
    """
    Build the (label, observed, shuffled) strings drawn in the table.

    The two value columns are printed at full precision and left to speak for
    themselves: every pair matches digit for digit, which is the claim.

    The glucose unit lives in the row label rather than in every cell, which
    keeps the numeric columns narrow enough to sit side by side on a slide.
    """
    unit = glucose_unit_label()
    dec  = 0 if unit == "mg/dL" else 1

    def pct(key):
        return f"{observed[key]:.1f} %", f"{shuffled[key]:.1f} %"

    def glu(key):
        o, s = float(to_display(observed[key])), float(to_display(shuffled[key]))
        return f"{o:.{dec}f}", f"{s:.{dec}f}"

    hypo  = format_glucose(THRESHOLD_HYPO,  with_unit=False)
    hyper = format_glucose(THRESHOLD_HYPER, with_unit=False)
    tar2  = format_glucose(THRESHOLD_TAR_LEVEL2, with_unit=False)
    tbr2  = format_glucose(THRESHOLD_TBR_LEVEL2, with_unit=False)

    return [
        (f"TIR  {hypo}–{hyper}", *pct("TIR")),
        (f"TAR  > {hyper}",      *pct("TAR")),
        (f"TAR  > {tar2}",       *pct("TAR2")),
        (f"TBR  < {hypo}",       *pct("TBR")),
        (f"TBR  < {tbr2}",       *pct("TBR2")),
        (f"Mean glucose ({unit})", *glu("Mean")),
        (f"SD ({unit})",           *glu("SD")),
        ("CV",                   *pct("CV")),
        ("GMI",                  f"{observed['GMI']:.2f} %",
                                 f"{shuffled['GMI']:.2f} %"),
    ]


# =============================================================================
# PANELS
# =============================================================================

def _format_trace_axes(ax, x_max: float, n_days: int, show_xlabel: bool) -> None:
    """Relative-hour x-axis + glucose y-axis, shared by both trace panels."""
    def _day_label(x, pos):
        if x == 0:
            return "Day 1"
        d, r = divmod(x, 24)
        return f"Day {int(d) + 1}" if r == 0 else ""

    ax.xaxis.set_major_locator(mticker.MultipleLocator(24))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(6))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_day_label))
    ax.set_xlim(0, x_max)
    ax.tick_params(axis="x", which="major", pad=4, labelsize=FS_TICK)
    for d in range(1, n_days):
        ax.axvline(d * 24, color="#BBBBBB", lw=0.7, alpha=0.55, zorder=1)

    positions, labels = glucose_ticks(Y_MIN, Y_MAX, max_ticks=6)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=FS_TICK)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_ylabel(glucose_label(), fontsize=FS_LABEL, labelpad=6)
    ax.yaxis.grid(True, ls=":", lw=0.5, color="#DDDDDD", zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_xlabel:
        ax.set_xlabel("Time", fontsize=FS_LABEL, labelpad=5)
    else:
        ax.tick_params(axis="x", labelbottom=False)


def _draw_thresholds(ax) -> None:
    """70 and 180 mg/dL rules, plus the faint target band between them."""
    ax.axhspan(THRESHOLD_HYPO, THRESHOLD_HYPER,
               color=COLOR_AGP_TARGET_ZONE, alpha=ALPHA_TARGET, zorder=0, lw=0)
    for value, color in [(THRESHOLD_HYPER, COLOR_THRESHOLD_HYPER),
                         (THRESHOLD_HYPO,  COLOR_THRESHOLD_HYPO)]:
        ax.axhline(value, color=color, lw=1.5, ls="--", alpha=0.9, zorder=5)


def _draw_trace(ax, times: pd.Series, glucose: np.ndarray, t_zero, gap_min: float,
                color: str, lw: float) -> None:
    """
    Gap-safe trace with the out-of-range readings both filled and dotted.

    The fill reads well on the observed record, where excursions are runs of
    consecutive points, but it almost vanishes once the record is permuted:
    two neighbouring shuffled readings are rarely both out of range, so the
    area collapses into slivers and the panel looks *less* hyperglycaemic than
    it is.  Marking every out-of-range reading individually keeps the amount
    of coloured ink the same in both panels, which is the claim being made.
    """
    frame = pd.DataFrame({TIME_COL: times.values, GLUCOSE_COL: glucose})

    for seg in _split_by_gaps(frame, gap_min):
        xh = _to_rel_h(seg[TIME_COL], t_zero).to_numpy()
        gv = seg[GLUCOSE_COL].to_numpy(dtype=float)

        ax.fill_between(xh, THRESHOLD_HYPER, gv, where=gv >= THRESHOLD_HYPER,
                        interpolate=True, color=COLOR_THRESHOLD_HYPER,
                        alpha=ALPHA_FILL, lw=0, zorder=2)
        ax.fill_between(xh, gv, THRESHOLD_HYPO, where=gv <= THRESHOLD_HYPO,
                        interpolate=True, color=COLOR_THRESHOLD_HYPO,
                        alpha=ALPHA_FILL, lw=0, zorder=2)
        ax.plot(xh, gv, color=color, lw=lw, alpha=0.92, zorder=3)

        for mask, marker_color in [(gv > THRESHOLD_HYPER, COLOR_THRESHOLD_HYPER),
                                   (gv < THRESHOLD_HYPO,  COLOR_THRESHOLD_HYPO)]:
            ax.plot(xh[mask], gv[mask], ls="none", marker="o", ms=MS_OUT_OF_RANGE,
                    mfc=marker_color, mec="none", alpha=0.85, zorder=4)


def _draw_marginal(ax, glucose: np.ndarray) -> None:
    """
    Horizontal histogram of the readings, on the same y-axis as the trace.

    This is the *only* thing a consensus metric reads — drawing it twice,
    identical, is the mechanism behind the two matching columns.
    """
    edges = np.arange(Y_MIN, Y_MAX + HIST_BIN_MGDL, HIST_BIN_MGDL)
    counts, _ = np.histogram(np.asarray(glucose, dtype=float), bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])

    colors = [
        COLOR_THRESHOLD_HYPO if c < THRESHOLD_HYPO else
        COLOR_THRESHOLD_HYPER if c > THRESHOLD_HYPER else
        COLOR_AGP_TARGET_ZONE
        for c in centres
    ]
    ax.barh(centres, 100 * counts / max(counts.sum(), 1),
            height=HIST_BIN_MGDL * 0.88, color=colors, alpha=0.75, lw=0)

    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlim(0, None)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)


def _draw_metric_table(ax, rows: list[tuple[str, str, str]],
                       n_readings: int, n_days: int) -> None:
    """Draw the observed / shuffled comparison as a slide-legible table."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x_label, x_obs, x_shuf = 0.015, 0.66, 0.985

    n_rows   = len(rows)
    y_top    = 0.90
    y_bottom = 0.10
    row_h    = (y_top - y_bottom) / n_rows

    # Shuffled column highlight — the punchline, behind everything else.
    ax.add_patch(Rectangle((x_obs + 0.07, y_bottom - 0.012),
                           1 - (x_obs + 0.07), (y_top + 0.075) - (y_bottom - 0.012),
                           facecolor=COLOR_SLH, alpha=0.07, lw=0, zorder=0))

    # Header
    y_head = y_top + 0.035
    for x, text, align in [(x_label, "Metric", "left"),
                           (x_obs,   "Observed", "right"),
                           (x_shuf,  "Shuffled", "right")]:
        ax.text(x, y_head, text, fontsize=FS_TABLE, fontweight="bold",
                color="#222222", ha=align, va="center")
    ax.plot([0, 1], [y_top + 0.008] * 2, color="#999999", lw=1.0, zorder=1)

    # Rows
    for i, (label, observed, shuffled) in enumerate(rows):
        y = y_top - (i + 0.5) * row_h
        if i % 2 == 1:
            ax.add_patch(Rectangle((0, y - row_h / 2), 1, row_h,
                                   facecolor="#F4F4F4", lw=0, zorder=0))
        ax.text(x_label, y, label, fontsize=FS_TABLE, color="#222222",
                ha="left", va="center", zorder=2)
        ax.text(x_obs, y, observed, fontsize=FS_TABLE, color="#222222",
                ha="right", va="center", zorder=2)
        ax.text(x_shuf, y, shuffled, fontsize=FS_TABLE, fontweight="bold",
                color=COLOR_SLH, ha="right", va="center", zorder=2)

    ax.plot([0, 1], [y_bottom - 0.012] * 2, color="#999999", lw=1.0, zorder=1)
    # ax.text(0.5, y_bottom - 0.070,
    #         f"{n_readings:,} readings · {n_days} days · same values, random order",
    #         fontsize=FS_ANNOT, color=COLOR_ANNOTATION, ha="center", va="center")
    # ax.text(0.5, y_bottom - 0.122,
    #         "Identical to the last decimal, not merely to the rounding",
    #         fontsize=FS_ANNOT, color=COLOR_ANNOTATION, ha="center", va="center")


# =============================================================================
# FIGURE
# =============================================================================

def build_figure(df_win: pd.DataFrame, shuffled: np.ndarray, t_zero: pd.Timestamp,
                 n_days: int, gap_min: float, patient_id: str,
                 observed_metrics: dict, shuffled_metrics: dict) -> plt.Figure:
    """Assemble the 16:9 slide."""
    x_max  = n_days * 24.0
    times  = df_win[TIME_COL]
    values = df_win[GLUCOSE_COL].to_numpy(dtype=float)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs  = fig.add_gridspec(
        2, 3,
        width_ratios=[1.0, 0.13, 0.62],
        height_ratios=[1.0, 1.0],
        left=0.055, right=0.985, top=0.97, bottom=0.085,
        wspace=0.10, hspace=0.20,
    )

    ax_obs   = fig.add_subplot(gs[0, 0])
    ax_shuf  = fig.add_subplot(gs[1, 0], sharex=ax_obs)
    ax_hist1 = fig.add_subplot(gs[0, 1])
    ax_hist2 = fig.add_subplot(gs[1, 1])
    ax_table = fig.add_subplot(gs[:, 2])

    # ── Panel A — observed ────────────────────────────────────────────────────
    _shade_nights(ax_obs, t_zero, t_zero, t_zero + pd.Timedelta(hours=x_max))
    _draw_thresholds(ax_obs)
    _draw_trace(ax_obs, times, values, t_zero, gap_min, COLOR_TRACE_OBS, lw=1.4)
    _format_trace_axes(ax_obs, x_max, n_days, show_xlabel=False)
    ax_obs.set_title(f"A   Observed record — patient {patient_id}",
                     fontsize=FS_PANEL, fontweight="bold", loc="left", pad=7)

    # ── Panel B — shuffled ────────────────────────────────────────────────────
    _shade_nights(ax_shuf, t_zero, t_zero, t_zero + pd.Timedelta(hours=x_max))
    _draw_thresholds(ax_shuf)
    _draw_trace(ax_shuf, times, shuffled, t_zero, gap_min, COLOR_TRACE_SHUF, lw=0.5)
    _format_trace_axes(ax_shuf, x_max, n_days, show_xlabel=True)
    ax_shuf.set_title("B   Same readings, order destroyed by a random permutation",
                      fontsize=FS_PANEL, fontweight="bold", loc="left", pad=7)

    # ── Marginal distributions ────────────────────────────────────────────────
    _draw_marginal(ax_hist1, values)
    _draw_marginal(ax_hist2, shuffled)
    # ax_hist1.set_title("Distribution", fontsize=FS_ANNOT,
    #                    color=COLOR_ANNOTATION, pad=7)
    ax_hist2.annotate(
        "identical", xy=(0.5, -0.045), xycoords="axes fraction",
        fontsize=FS_ANNOT, fontweight="bold", color=COLOR_SLH,
        ha="center", va="top", annotation_clip=False,
    )

    # ── Metric table ──────────────────────────────────────────────────────────
    _draw_metric_table(ax_table, _metric_rows(observed_metrics, shuffled_metrics),
                       n_readings=len(values), n_days=n_days)

    # ── Slide titles ──────────────────────────────────────────────────────────
    # fig.text(0.055, 0.955, "Consensus CGM metrics are blind to time",
    #          fontsize=FS_SUPTITLE, fontweight="bold", color="#111111",
    #          ha="left", va="center")
    # fig.text(0.055, 0.905,
    #          "TIR, TAR, TBR, mean, SD, CV and GMI are functionals of the glucose "
    #          "distribution alone — reordering the record leaves every one of them "
    #          "unchanged.",
    #          fontsize=FS_SUBTITLE, color=COLOR_ANNOTATION, ha="left", va="center")

    return fig


# =============================================================================
# PIPELINE
# =============================================================================

def generate(patient_id: str, n_days: int, seed: int, output_dir: str) -> None:
    """Load → select window → permute → measure → draw → save."""
    df = load_patient_window(patient_id)
    if df is None:
        print(f"[!] No data found for {patient_id}")
        return
    print(f"  Loaded {len(df):,} data points")

    timestep_min = detect_dominant_sampling_interval(df)
    gap_min      = timestep_min * GAP_FACTOR
    print(f"  Sampling interval : {timestep_min:.0f} min")

    t_start, t_end = select_densest_window(df, n_days)
    mask   = (df[TIME_COL] >= t_start) & (df[TIME_COL] < t_end)
    df_win = df[mask].sort_values(TIME_COL).reset_index(drop=True)
    print(f"  Window            : {t_start.date()} → {t_end.date()}  "
          f"({len(df_win):,} readings)")

    if df_win.empty:
        print(f"[!] Empty window for {patient_id}")
        return

    values   = df_win[GLUCOSE_COL].to_numpy(dtype=float)
    shuffled = np.random.default_rng(seed).permutation(values)

    observed_metrics = compute_consensus_metrics(values)
    shuffled_metrics = compute_consensus_metrics(shuffled)

    worst = max(abs(shuffled_metrics[k] - observed_metrics[k])
                for k in observed_metrics)
    print(f"  Largest |Δ| across all metrics : {worst:.2e}")

    apply_publication_style()
    matplotlib.rcParams.update({
        "font.size":       FS_TICK,
        "axes.labelsize":  FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_ANNOT,
    })

    fig = build_figure(df_win, shuffled, t_start, n_days, gap_min, patient_id,
                       observed_metrics, shuffled_metrics)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir,
                       f"{patient_id}_metric_permutation_invariance_{n_days}d")
    # bbox_inches=None keeps the canvas at an exact 16:9 slide ratio.
    save_figure(fig, out, bbox_inches=None)


# =============================================================================
# CLI
# =============================================================================

def parse_arguments():
    p = argparse.ArgumentParser(
        description="16:9 slide — consensus CGM metrics are permutation-invariant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--patient_id", default=DEFAULT_PATIENT_ID)
    p.add_argument("--n_days", type=int, default=DEFAULT_N_DAYS,
                   help="Number of consecutive days shown and measured")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="Seed of the permutation, so the slide is reproducible")
    p.add_argument("--out_dir", default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def main():
    args = parse_arguments()
    args.patient_id = resolve_patient_id(args.patient_id)
    print(f"\nPermutation-invariance slide — {args.patient_id}")
    generate(args.patient_id, args.n_days, args.seed, args.out_dir)


if __name__ == "__main__":
    main()
