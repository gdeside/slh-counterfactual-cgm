"""
scripts/presentation_figures/slh_twin_comparison.py
===================================================

One 16:9 presentation figure arguing why the SLH count is worth reporting.

Two records are placed side by side:

  the **observed** record of a real patient, and a **constructed twin** built
  from it that every standard CGM readout calls the same record — identical
  TIR, TAR, TBR, mean, SD, CV and GMI, and an AGP that superimposes on the
  original — yet that contains a very different number of short-lasting
  hyperglycaemia episodes.

The twin is synthetic and the figure says so on its face.  It is not a second
participant, and it is not presented as one.

Construction
------------
The twin must satisfy three things at once: the same marginal distribution
(so every consensus metric is untouched), the same clock-time profile (so the
AGP is untouched), and a different arrangement of the hyperglycaemic frequency
*within* each day (so the SLH count moves).  It is built in two steps:

1. **Smooth.**  A centred rolling mean over ``SMOOTH_WINDOW_MIN`` gives a
   provisional signal in which brief post-meal spikes have merged into
   broader, slower excursions.  This step alone changes the values, so it
   cannot be the end of it.
2. **Rank-match within clock-time bins.**  Inside each ``CLOCK_BIN_MIN`` bin
   of the day, the original readings of that bin are re-dealt across days
   following the *ranks* of the smoothed signal.  Because the readings are
   only permuted, and only among readings that share a clock bin:

   * the whole record keeps the same multiset of values, so TIR, TAR, TBR,
     mean, SD, CV and GMI are identical to the last decimal — by
     construction, not by tuning;
   * each clock bin keeps its own multiset, so the AGP percentiles barely
     move (they are not exactly invariant: the AGP matches a reading per day
     per slot by nearest timestamp, and days with gaps contribute unevenly);
   * the minute-to-minute variability stays realistic, because neighbouring
     clock bins hold near-identical value pools and the smoothed signal
     orders them consistently.

What moves is the *shape* of the excursions: the same time above 180 mg/dL is
delivered in fewer, longer episodes, so it drains out of the 10–60 min window
that defines an SLH.

The twin therefore isolates exactly the axis the consensus metrics cannot see,
which is the argument the SLH count exists to make.

Usage
-----
    python3 -m scripts.presentation_figures.slh_twin_comparison
            [--patient_id ID] [--n_days 3] [--smooth_min 120] [--out_dir DIR]

    # Same figure in mmol/L (default is mg/dL; output filename gets a _mmoll suffix)
    CGM_GLUCOSE_UNIT="mmol/L" python3 -m scripts.presentation_figures.slh_twin_comparison
            [same options]

Authors
-------
Guillaume Deside

Date
----
2026

Output
------
    scripts/presentation_figures/output/slh_twin_comparison/  (git-ignored; --out_dir to change)
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
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.patient_selection import resolve_patient_id
from src.preprocessing.load_data import load_patient_window
from src.preprocessing.timestep import detect_dominant_sampling_interval
from src.episodes.detection import detect_hyperglycemia_episodes
from src.episodes.criteria import classify_episodes
from src.episodes.episode_config import (
    GAP_FACTOR, NIGHT_START_H, NIGHT_END_H, D_MIN_MIN, D_MAX_MIN, LABEL_SLH,
)
from src.agp.compute_agp import compute_agp_profile
from src.config import TIME_COL, GLUCOSE_COL, THRESHOLD_HYPER, THRESHOLD_HYPO
from src.metrics.metrics import THRESHOLD_TAR_LEVEL2
from src.utils.unit_conversion import (
    to_display, glucose_label, glucose_unit_label, format_glucose, glucose_ticks,
)
from src.visualization.style import apply_publication_style, save_figure
from src.visualization.colors import (
    COLOR_THRESHOLD_HYPER, COLOR_THRESHOLD_HYPO, COLOR_AGP_TARGET_ZONE,
    COLOR_ANNOTATION, COLOR_SLH, WONG_COLORS,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PATIENT_ID = None   # None = automatic: patient with the most SLH episodes
DEFAULT_N_DAYS     = 3
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "slh_twin_comparison")

#: Rolling-mean window used to merge brief excursions into longer ones.
SMOOTH_WINDOW_MIN = 120

#: Clock-bin width the readings are re-dealt within, in minutes.
CLOCK_BIN_MIN = 15

#: Resolution of the AGP panel, in minutes.
AGP_RESOLUTION_MIN = 15

# ---------------------------------------------------------------------------
# Figure geometry — 16:9 slide
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 13.333, 7.5

FS_SUPTITLE = 19
FS_SUBTITLE = 12.5
FS_PANEL    = 13.5
FS_LABEL    = 11.5
FS_TICK     = 10
FS_TABLE    = 11
FS_ANNOT    = 10
FS_BIGNUM   = 27

# ---------------------------------------------------------------------------
# Record colours — the observed record and its twin
# ---------------------------------------------------------------------------
COLOR_OBSERVED = COLOR_SLH                     # navy
COLOR_TWIN     = WONG_COLORS["bluish_green"]   # teal, clear of both thresholds

#: SLH episodes are highlighted the same way in both panels, so the eye
#: compares counts rather than colours.
COLOR_SLH_SPAN = WONG_COLORS["yellow"]
ALPHA_SLH_SPAN = 0.70

COLOR_NIGHT = "#BDBDBD"
ALPHA_NIGHT = 0.20

# Y-axis (mg/dL, canonical)
Y_MIN, Y_MAX = 40, 370
Y_MIN_AGP, Y_MAX_AGP = 50, 320


# =============================================================================
# TWIN CONSTRUCTION
# =============================================================================

def build_twin(df: pd.DataFrame, smooth_min: int = SMOOTH_WINDOW_MIN,
               bin_min: int = CLOCK_BIN_MIN) -> pd.DataFrame:
    """
    Build the constructed twin of *df* (see the module docstring).

    Parameters
    ----------
    df : pd.DataFrame
        Observed record, columns ``[TIME_COL, GLUCOSE_COL]``, time-sorted.
    smooth_min : int
        Width of the centred rolling mean, in minutes.  Wider merges more
        excursions, so it moves the SLH count further and the AGP slightly
        more.
    bin_min : int
        Clock-bin width the readings are permuted within, in minutes.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``GLUCOSE_COL`` replaced by the twin's readings.
        The timestamps, the number of readings and the multiset of values
        are unchanged.
    """
    values = df[GLUCOSE_COL].to_numpy(dtype=float)

    series   = pd.Series(values, index=df[TIME_COL])
    smoothed = series.rolling(f"{smooth_min}min", center=True,
                              min_periods=1).mean().to_numpy()

    minute_of_day = (df[TIME_COL].dt.hour * 60 + df[TIME_COL].dt.minute).to_numpy()
    clock_bin     = minute_of_day // bin_min

    twin = values.copy()
    for this_bin in np.unique(clock_bin):
        idx = np.flatnonzero(clock_bin == this_bin)
        # Rank of each day's smoothed value inside this clock bin, then deal
        # the bin's own observed values out in that order.
        ranks = np.argsort(np.argsort(smoothed[idx], kind="stable"), kind="stable")
        twin[idx] = np.sort(values[idx])[ranks]

    out = df.copy()
    out[GLUCOSE_COL] = twin
    return out


# =============================================================================
# MEASUREMENT
# =============================================================================

def compute_consensus_metrics(glucose: np.ndarray) -> dict[str, float]:
    """The permutation-invariant consensus metrics, unrounded."""
    g = np.asarray(glucose, dtype=float)
    g = g[np.isfinite(g)]
    return {
        "TIR":  100 * float(((g >= THRESHOLD_HYPO) & (g <= THRESHOLD_HYPER)).mean()),
        "TAR":  100 * float((g > THRESHOLD_HYPER).mean()),
        "TAR2": 100 * float((g > THRESHOLD_TAR_LEVEL2).mean()),
        "TBR":  100 * float((g < THRESHOLD_HYPO).mean()),
        "Mean": float(g.mean()),
        "SD":   float(g.std(ddof=1)),
        "CV":   100 * float(g.std(ddof=1) / g.mean()),
        "GMI":  3.31 + 0.02392 * float(g.mean()),
    }


def episode_table(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and classify every hyperglycaemia episode in *df*."""
    episodes = detect_hyperglycemia_episodes(df)
    if not episodes:
        return classify_episodes(df, [])
    return classify_episodes(df, episodes)


def slh_episodes(table: pd.DataFrame) -> pd.DataFrame:
    """The rows of a classification table that are confirmed SLH."""
    if table.empty:
        return table
    return table[table["classification"] == LABEL_SLH]


# =============================================================================
# SHARED DRAWING HELPERS
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
    """Shade night hours on a relative-hour axis."""
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


def _style_glucose_axis(ax, y_min: float, y_max: float, show_ylabel: bool = True) -> None:
    """Glucose y-axis with unit-aware ticks."""
    positions, labels = glucose_ticks(y_min, y_max, max_ticks=5)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=FS_TICK)
    ax.set_ylim(y_min, y_max)
    if show_ylabel:
        ax.set_ylabel(glucose_label(), fontsize=FS_LABEL, labelpad=4)
    ax.yaxis.grid(True, ls=":", lw=0.5, color="#DDDDDD", zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_thresholds(ax, target_band: bool = True) -> None:
    """The 70 and 180 mg/dL rules, and optionally the target band."""
    if target_band:
        ax.axhspan(THRESHOLD_HYPO, THRESHOLD_HYPER, color=COLOR_AGP_TARGET_ZONE,
                   alpha=0.07, zorder=0, lw=0)
    for value, color in [(THRESHOLD_HYPER, COLOR_THRESHOLD_HYPER),
                         (THRESHOLD_HYPO,  COLOR_THRESHOLD_HYPO)]:
        ax.axhline(value, color=color, lw=1.3, ls="--", alpha=0.9, zorder=5)


# =============================================================================
# PANEL A — AGP overlay
# =============================================================================

def _draw_agp_panel(ax, agp_observed: pd.DataFrame, agp_twin: pd.DataFrame,
                    median_rmse: float) -> None:
    """Overlay the two AGPs: the 'they look the same' evidence."""
    hours = np.arange(len(agp_observed)) * (24.0 / len(agp_observed))

    _draw_thresholds(ax)

    for agp, color, label in [(agp_observed, COLOR_OBSERVED, "Observed"),
                              (agp_twin,     COLOR_TWIN,     "Twin")]:
        ax.fill_between(hours, agp["q25"], agp["q75"],
                        color=color, alpha=0.16, lw=0, zorder=2)
        ax.plot(hours, agp["median"], color=color, lw=2.2, zorder=4, label=label)

    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 6))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 6)], fontsize=FS_TICK)
    ax.set_xlabel("Time of day", fontsize=FS_LABEL, labelpad=3)
    _style_glucose_axis(ax, Y_MIN_AGP, Y_MAX_AGP)

    ax.set_title("A   Same AGP", fontsize=FS_PANEL, fontweight="bold",
                 loc="left", pad=6)

    # The RMSE arrives in mg/dL like every other glucose quantity here; the
    # mg/dL → mmol/L conversion is a pure ratio, so it applies to a difference
    # unchanged.  Converting the label without the value would misstate it by
    # a factor of 18.
    rmse_display = float(to_display(median_rmse))
    decimals = 1 if glucose_unit_label() == "mg/dL" else 2
    ax.annotate(
        f"median curves differ by {rmse_display:.{decimals}f} "
        f"{glucose_unit_label()} RMSE\n(bands: 25–75th percentile)",
        xy=(0.03, 0.965), xycoords="axes fraction", fontsize=FS_ANNOT,
        color=COLOR_ANNOTATION, ha="left", va="top",
    )


# =============================================================================
# PANEL B — metric table
# =============================================================================

def _metric_rows(observed: dict, twin: dict) -> list[tuple[str, str, str]]:
    """The (label, observed, twin) strings of the comparison table."""
    unit = glucose_unit_label()
    dec  = 0 if unit == "mg/dL" else 1

    def pct(key):
        return f"{observed[key]:.1f}", f"{twin[key]:.1f}"

    def glu(key):
        return (f"{float(to_display(observed[key])):.{dec}f}",
                f"{float(to_display(twin[key])):.{dec}f}")

    hypo  = format_glucose(THRESHOLD_HYPO,  with_unit=False)
    hyper = format_glucose(THRESHOLD_HYPER, with_unit=False)
    tar2  = format_glucose(THRESHOLD_TAR_LEVEL2, with_unit=False)

    return [
        (f"TIR {hypo}–{hyper}  (%)", *pct("TIR")),
        (f"TAR > {hyper}  (%)",      *pct("TAR")),
        (f"TAR > {tar2}  (%)",       *pct("TAR2")),
        (f"TBR < {hypo}  (%)",       *pct("TBR")),
        (f"Mean  ({unit})",          *glu("Mean")),
        (f"SD  ({unit})",            *glu("SD")),
        ("CV  (%)",                  *pct("CV")),
        ("GMI  (%)",                 f"{observed['GMI']:.2f}", f"{twin['GMI']:.2f}"),
    ]


def _draw_metric_panel(ax, rows: list[tuple[str, str, str]]) -> None:
    """The 'they measure the same' evidence."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x_label, x_obs, x_twin = 0.02, 0.74, 0.99
    y_top, y_bottom = 0.86, 0.06
    row_h = (y_top - y_bottom) / len(rows)

    ax.text(0.0, 0.985, "B   Same metrics", fontsize=FS_PANEL, fontweight="bold",
            color="#111111", ha="left", va="top", transform=ax.transAxes)

    y_head = y_top + 0.045
    for x, text, color in [(x_obs, "Observed", COLOR_OBSERVED),
                           (x_twin, "Twin", COLOR_TWIN)]:
        ax.text(x, y_head, text, fontsize=FS_TABLE, fontweight="bold",
                color=color, ha="right", va="center")
    ax.plot([0, 1], [y_top + 0.015] * 2, color="#999999", lw=0.9)

    for i, (label, observed, twin) in enumerate(rows):
        y = y_top - (i + 0.5) * row_h
        if i % 2 == 1:
            ax.add_patch(Rectangle((0, y - row_h / 2), 1, row_h,
                                   facecolor="#F4F4F4", lw=0, zorder=0))
        ax.text(x_label, y, label, fontsize=FS_TABLE, color="#222222",
                ha="left", va="center", zorder=2)
        ax.text(x_obs, y, observed, fontsize=FS_TABLE, color=COLOR_OBSERVED,
                ha="right", va="center", zorder=2)
        ax.text(x_twin, y, twin, fontsize=FS_TABLE, fontweight="bold",
                color=COLOR_TWIN, ha="right", va="center", zorder=2)

    ax.plot([0, 1], [y_bottom - 0.015] * 2, color="#999999", lw=0.9)
    # ax.text(0.5, y_bottom - 0.075, "identical to the last decimal",
    #         fontsize=FS_ANNOT, color=COLOR_ANNOTATION, ha="center", va="center")


# =============================================================================
# PANEL C — the SLH count
# =============================================================================

def _draw_count_panel(ax, n_slh_observed: int, n_slh_twin: int,
                      n_ep_observed: int, n_ep_twin: int, n_days_total: int) -> None:
    """The 'but they are not the same patient' punchline."""
    counts = [n_slh_observed, n_slh_twin]
    colors = [COLOR_OBSERVED, COLOR_TWIN]
    labels = ["Observed", "Twin"]

    bars = ax.bar([0, 1], counts, width=0.58, color=colors, alpha=0.90, zorder=3)

    for bar, count, total in zip(bars, counts, [n_ep_observed, n_ep_twin]):
        ax.annotate(f"{count}", xy=(bar.get_x() + bar.get_width() / 2, count),
                    xytext=(0, 6), textcoords="offset points",
                    fontsize=FS_BIGNUM, fontweight="bold", color=bar.get_facecolor(),
                    ha="center", va="bottom")
        ax.annotate(f"of {total}\nexcursions",
                    xy=(bar.get_x() + bar.get_width() / 2, 0),
                    xytext=(0, -17), textcoords="offset points",
                    fontsize=FS_ANNOT, color=COLOR_ANNOTATION, linespacing=1.35,
                    ha="center", va="top", annotation_clip=False)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=FS_LABEL, fontweight="bold")
    for tick, color in zip(ax.get_xticklabels(), colors):
        tick.set_color(color)

    ax.set_xlim(-0.65, 1.65)
    ax.set_ylim(0, max(counts) * 1.42)
    ax.set_ylabel(f"SLH episodes in {n_days_total} days", fontsize=FS_LABEL, labelpad=4)
    ax.yaxis.grid(True, ls=":", lw=0.5, color="#DDDDDD", zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title("C   Different SLH count", fontsize=FS_PANEL,
                 fontweight="bold", loc="left", pad=6)

    if n_slh_observed > 0:
        drop = 100 * (n_slh_twin - n_slh_observed) / n_slh_observed
        ax.annotate(f"{drop:+.0f} %".replace("-", "\u2212"),
                    xy=(0.5, 0.97), xycoords="axes fraction",
                    fontsize=FS_PANEL, fontweight="bold", color=COLOR_ANNOTATION,
                    ha="center", va="top")




# =============================================================================
# PANELS D / E — the traces
# =============================================================================

def _draw_trace_panel(ax, df_win: pd.DataFrame, table: pd.DataFrame,
                      t_zero: pd.Timestamp, x_max: float, n_days: int,
                      gap_min: float, color: str, title: str,
                      show_xlabel: bool) -> None:
    """A few days of one record, with its SLH episodes highlighted."""
    _shade_nights(ax, t_zero, t_zero, t_zero + pd.Timedelta(hours=x_max))
    _draw_thresholds(ax, target_band=False)

    # SLH episodes, highlighted identically in both panels so that the eye
    # compares how many there are rather than what colour they are.
    window_slh = [
        row for _, row in slh_episodes(table).iterrows()
        if row["offset_time"] >= t_zero
        and row["onset_time"] <= t_zero + pd.Timedelta(hours=x_max)
    ]
    for row in window_slh:
        x_on  = _to_rel_h(pd.Series([row["onset_time"]]),  t_zero).item()
        x_off = _to_rel_h(pd.Series([row["offset_time"]]), t_zero).item()
        ax.axvspan(x_on, x_off, color=COLOR_SLH_SPAN, alpha=ALPHA_SLH_SPAN,
                   zorder=1, lw=0)

    for seg in _split_by_gaps(df_win, gap_min):
        xh = _to_rel_h(seg[TIME_COL], t_zero).to_numpy()
        ax.plot(xh, seg[GLUCOSE_COL].to_numpy(dtype=float),
                color=color, lw=1.5, alpha=0.95, zorder=3)

    def _day_label(x, pos):
        if x % 24:
            return ""
        return f"Day {int(x // 24) + 1}"

    ax.xaxis.set_major_locator(mticker.MultipleLocator(24))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(6))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_day_label))
    ax.set_xlim(0, x_max)
    for d in range(1, n_days):
        ax.axvline(d * 24, color="#BBBBBB", lw=0.7, alpha=0.55, zorder=1)

    _style_glucose_axis(ax, Y_MIN, Y_MAX)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    if not show_xlabel:
        ax.tick_params(axis="x", labelbottom=False)

    ax.set_title(f"{title}   —   {len(window_slh)} SLH in these {n_days} days",
                 fontsize=FS_PANEL, fontweight="bold", loc="left", pad=5,
                 color=color)


# =============================================================================
# WINDOW SELECTION
# =============================================================================

def select_window(df: pd.DataFrame, table: pd.DataFrame, n_days: int):
    """
    Pick the *n_days* window richest in observed SLH episodes.

    The twin is drawn over the same window — the two records share their
    timestamps, so the comparison is like for like.
    """
    t_first = df[TIME_COL].min().normalize()
    t_last  = df[TIME_COL].max()
    n_avail = int((t_last - t_first).total_seconds() / 86400) + 1

    if n_avail <= n_days:
        return t_first

    slh = slh_episodes(table)
    best_score, best_start = -1, t_first
    for i in range(n_avail - n_days + 1):
        ws = t_first + pd.Timedelta(days=i)
        we = ws + pd.Timedelta(days=n_days)
        n_slh = int(((slh["onset_time"] >= ws) & (slh["offset_time"] <= we)).sum()) \
            if not slh.empty else 0
        n_pts = int(((df[TIME_COL] >= ws) & (df[TIME_COL] < we)).sum())
        # Completeness breaks ties: a punctured window makes a poor slide.
        score = (n_slh, n_pts)
        if score > (best_score if isinstance(best_score, tuple) else (-1, -1)):
            best_score, best_start = score, ws

    return best_start


# =============================================================================
# FIGURE
# =============================================================================

def build_figure(patient_id: str, df: pd.DataFrame, df_twin: pd.DataFrame,
                 table_obs: pd.DataFrame, table_twin: pd.DataFrame,
                 metrics_obs: dict, metrics_twin: dict,
                 agp_obs: pd.DataFrame, agp_twin: pd.DataFrame,
                 median_rmse: float, t_zero: pd.Timestamp,
                 n_days: int, gap_min: float, smooth_min: int) -> plt.Figure:
    """Assemble the 16:9 slide."""
    x_max = n_days * 24.0
    mask  = (df[TIME_COL] >= t_zero) & (df[TIME_COL] < t_zero + pd.Timedelta(days=n_days))
    win_obs  = df[mask].reset_index(drop=True)
    win_twin = df_twin[mask].reset_index(drop=True)

    n_days_total = int(round(
        (df[TIME_COL].max() - df[TIME_COL].min()).total_seconds() / 86400))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(
        3, 3,
        width_ratios=[1.05, 0.92, 0.62],
        height_ratios=[1.30, 0.70, 0.70],
        left=0.052, right=0.985, top=0.822, bottom=0.085,
        wspace=0.30, hspace=0.45,
    )

    ax_agp   = fig.add_subplot(gs[0, 0])
    ax_tab   = fig.add_subplot(gs[0, 1])
    ax_count = fig.add_subplot(gs[0, 2])
    ax_obs   = fig.add_subplot(gs[1, :])
    ax_twin  = fig.add_subplot(gs[2, :], sharex=ax_obs)

    _draw_agp_panel(ax_agp, agp_obs, agp_twin, median_rmse)
    ax_agp.legend(loc="upper right", fontsize=FS_ANNOT, frameon=False,
                  handlelength=1.6)

    _draw_metric_panel(ax_tab, _metric_rows(metrics_obs, metrics_twin))

    _draw_count_panel(
        ax_count,
        n_slh_observed=len(slh_episodes(table_obs)),
        n_slh_twin=len(slh_episodes(table_twin)),
        n_ep_observed=len(table_obs), n_ep_twin=len(table_twin),
        n_days_total=n_days_total,
    )

    _draw_trace_panel(ax_obs, win_obs, table_obs, t_zero, x_max, n_days, gap_min,
                      COLOR_OBSERVED, f"D   Observed — {patient_id}",
                      show_xlabel=False)
    _draw_trace_panel(ax_twin, win_twin, table_twin, t_zero, x_max, n_days, gap_min,
                      COLOR_TWIN, "E   Constructed twin", show_xlabel=True)

    # ── Slide titles ──────────────────────────────────────────────────────────
    fig.text(0.052, 0.958,
             "Identical CGM report, different disease pattern",
             fontsize=FS_SUPTITLE, fontweight="bold", color="#111111",
             ha="left", va="center")
    # fig.text(0.052, 0.912,
    #          f"The same readings as {patient_id}, redistributed in time: "
    #          f"every consensus metric and the AGP are reproduced — the SLH count is not.",
    #          fontsize=FS_SUBTITLE, color=COLOR_ANNOTATION, ha="left", va="center")

    # The yellow bands need defining somewhere; in-panel it covered the trace.
    fig.legend(
        handles=[mpatches.Patch(facecolor=COLOR_SLH_SPAN, alpha=ALPHA_SLH_SPAN,
                                label=f"SLH — isolated daytime excursion above "
                                      f"{format_glucose(THRESHOLD_HYPER)} "
                                      f"lasting {D_MIN_MIN}–{D_MAX_MIN} min"),
                 Line2D([0], [0], color=COLOR_THRESHOLD_HYPER, lw=1.3, ls="--",
                        label=format_glucose(THRESHOLD_HYPER)),
                 Line2D([0], [0], color=COLOR_THRESHOLD_HYPO, lw=1.3, ls="--",
                        label=format_glucose(THRESHOLD_HYPO))],
        loc="lower left", bbox_to_anchor=(0.052, 0.862), fontsize=FS_ANNOT,
        frameon=False, ncol=3, handlelength=1.7, columnspacing=1.8,
    )
    # fig.text(0.985, 0.012,
    #          f"Twin is synthetic, not a second participant: {patient_id}'s own readings "
    #          f"re-dealt within {CLOCK_BIN_MIN}-min clock bins, ordered by a "
    #          f"{smooth_min}-min rolling mean.",
    #          fontsize=FS_ANNOT - 0.5, color=COLOR_ANNOTATION, ha="right", va="bottom",
    #          style="italic")

    return fig


# =============================================================================
# PIPELINE
# =============================================================================

def generate(patient_id: str, n_days: int, smooth_min: int, output_dir: str) -> None:
    """Load → build twin → measure both → draw → save."""
    df = load_patient_window(patient_id)
    if df is None:
        print(f"[!] No data found for {patient_id}")
        return
    print(f"  Loaded {len(df):,} data points")

    timestep_min = detect_dominant_sampling_interval(df)
    gap_min      = timestep_min * GAP_FACTOR

    df_twin = build_twin(df, smooth_min=smooth_min, bin_min=CLOCK_BIN_MIN)

    metrics_obs  = compute_consensus_metrics(df[GLUCOSE_COL].to_numpy(dtype=float))
    metrics_twin = compute_consensus_metrics(df_twin[GLUCOSE_COL].to_numpy(dtype=float))
    worst = max(abs(metrics_twin[k] - metrics_obs[k]) for k in metrics_obs)
    print(f"  Largest |Δ| across consensus metrics : {worst:.2e}")

    table_obs  = episode_table(df)
    table_twin = episode_table(df_twin)
    n_slh_obs, n_slh_twin = len(slh_episodes(table_obs)), len(slh_episodes(table_twin))
    print(f"  SLH  observed={n_slh_obs}  twin={n_slh_twin}  "
          f"(of {len(table_obs)} / {len(table_twin)} excursions)")
    if not table_obs.empty and not table_twin.empty:
        print(f"  Median excursion duration : "
              f"{table_obs.duration_min.median():.0f} → "
              f"{table_twin.duration_min.median():.0f} min")

    agp_obs  = compute_agp_profile(df,      AGP_RESOLUTION_MIN)
    agp_twin = compute_agp_profile(df_twin, AGP_RESOLUTION_MIN)
    median_rmse = float(np.sqrt(np.nanmean(
        (agp_obs["median"].to_numpy() - agp_twin["median"].to_numpy()) ** 2)))
    print(f"  AGP median RMSE : {median_rmse:.2f} mg/dL")

    t_zero = select_window(df, table_obs, n_days)
    print(f"  Trace window : {t_zero.date()} (+{n_days} d)")

    apply_publication_style()
    matplotlib.rcParams.update({
        "font.size":       FS_TICK,
        "axes.labelsize":  FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_ANNOT,
    })

    fig = build_figure(patient_id, df, df_twin, table_obs, table_twin,
                       metrics_obs, metrics_twin, agp_obs, agp_twin,
                       median_rmse, t_zero, n_days, gap_min, smooth_min)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"{patient_id}_slh_twin_comparison")
    # bbox_inches=None keeps the canvas at an exact 16:9 slide ratio.
    save_figure(fig, out, bbox_inches=None)


# =============================================================================
# CLI
# =============================================================================

def parse_arguments():
    p = argparse.ArgumentParser(
        description="16:9 slide — same metrics, same AGP, different SLH count",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--patient_id", default=DEFAULT_PATIENT_ID)
    p.add_argument("--n_days", type=int, default=DEFAULT_N_DAYS,
                   help="Days shown in the two trace panels")
    p.add_argument("--smooth_min", type=int, default=SMOOTH_WINDOW_MIN,
                   help="Rolling-mean window used to build the twin (minutes)")
    p.add_argument("--out_dir", default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def main():
    args = parse_arguments()
    args.patient_id = resolve_patient_id(args.patient_id)
    print(f"\nSLH twin comparison — {args.patient_id}")
    generate(args.patient_id, args.n_days, args.smooth_min, args.out_dir)


if __name__ == "__main__":
    main()
