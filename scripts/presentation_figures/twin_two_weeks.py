"""
scripts/presentation_figures/twin_two_weeks.py
==============================================

Introduction slides: two CGM records that every standard readout calls the
same record, and the one thing that separates them.

Three 16:9 figures, meant to be shown in order and before any definition has
been given — so nothing here names a criterion, a classification or an
acronym.  The whole argument is carried by what the audience can count.

  1. ``*_two_weeks_plain``
     Two weeks of glucose from two records, side by side, fourteen days per
     column.  Nothing is marked.  The records look like ordinary neighbours.

  2. ``*_two_weeks_marked``
     The same two weeks, with every hyperglycaemic excursion lasting at most
     ``MAX_DURATION_MIN`` minutes shaded.  One column fills with marks, the
     other stays nearly empty.

  3. ``*_agp_metrics``
     The ambulatory glucose profile of both records, overlaid, next to the
     consensus metrics of both.  They agree — which is the point.

Record B is constructed from Record A and the figures say so.  It is not a
second participant.

Construction
------------
Record B must reproduce record A's marginal distribution (so every consensus
metric is untouched) and its clock-time profile (so the AGP is untouched),
while arranging the hyperglycaemic frequency differently *within* each day.  It
is built in two steps:

1. **Smooth.**  A centred rolling mean over ``SMOOTH_WINDOW_MIN`` gives a
   provisional signal in which brief post-meal spikes have merged into
   broader, slower excursions.
2. **Rank-match within clock-time bins.**  Inside each ``CLOCK_BIN_MIN`` bin
   of the day, record A's readings for that bin are re-dealt across days
   following the *ranks* of the smoothed signal.  Because the readings are
   only permuted, and only among readings sharing a clock bin:

   * the record keeps the same multiset of values, so TIR, TAR, TBR, mean,
     SD, CV and GMI are identical to the last decimal — by construction, not
     by tuning;
   * each clock bin keeps its own multiset, so the AGP percentiles barely
     move (they are not exactly invariant: the AGP matches one reading per
     day per slot by nearest timestamp, and days with gaps contribute
     unevenly);
   * minute-to-minute variability stays realistic, because neighbouring
     clock bins hold near-identical value pools.

What moves is the *shape* of the excursions: the same time above the
hyperglycaemia threshold, delivered in fewer and longer episodes.

Usage
-----
    python3 -m scripts.presentation_figures.twin_two_weeks
            [--patient_id ID] [--n_days 14] [--max_dur 60]
            [--smooth_min 120] [--out_dir DIR]

    # Same figure in mmol/L (default is mg/dL; output filename gets a _mmoll suffix)
    CGM_GLUCOSE_UNIT="mmol/L" python3 -m scripts.presentation_figures.twin_two_weeks
            [same options]

Authors
-------
Guillaume Deside

Date
----
2026

Output
------
    scripts/presentation_figures/output/twin_two_weeks/  (git-ignored; --out_dir to change)
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.patient_selection import resolve_patient_id
from src.preprocessing.load_data import load_patient_window
from src.preprocessing.timestep import detect_dominant_sampling_interval
from src.episodes.detection import detect_hyperglycemia_episodes
from src.episodes.episode_config import GAP_FACTOR, D_MAX_MIN
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
DEFAULT_N_DAYS     = 14
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "twin_two_weeks")

#: Longest excursion that still counts as brief, in minutes.  Matches the
#: project's ``D_MAX_MIN`` so that this introduction lines up with the formal
#: definition given later in the talk, and is inclusive for the same reason.
MAX_DURATION_MIN = D_MAX_MIN

#: Rolling-mean window used to merge brief excursions into longer ones.
SMOOTH_WINDOW_MIN = 120

#: Clock-bin width the readings are re-dealt within, in minutes.  Narrower
#: than the AGP resolution on purpose: the fortnight gives each bin only 14
#: values to permute, and a narrow bin keeps the clock-time profile tighter.
CLOCK_BIN_MIN = 10

#: Coverage a candidate fortnight must reach before it is considered.
MIN_WINDOW_COVERAGE = 0.98

#: Resolution of the AGP, in minutes.
AGP_RESOLUTION_MIN = 15

# ---------------------------------------------------------------------------
# Figure geometry — 16:9 slide
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 13.333, 7.5

FS_SUPTITLE = 20
FS_SUBTITLE = 13
FS_COLHEAD  = 15
FS_PANEL    = 14
FS_LABEL    = 12
FS_TICK     = 10.5
FS_TABLE    = 12.5
FS_ANNOT    = 10.5
FS_DAY      = 8.5
FS_BIGNUM   = 30

# ---------------------------------------------------------------------------
# Record colours
# ---------------------------------------------------------------------------
COLOR_A = COLOR_SLH                     # navy
COLOR_B = WONG_COLORS["bluish_green"]   # teal, clear of both threshold colours

#: Brief excursions are shaded the same way in both columns, so the eye
#: compares how many there are rather than what colour they are.
COLOR_MARK = WONG_COLORS["yellow"]
ALPHA_MARK = 0.85

# Y-axis (mg/dL, canonical)
#: Padding added below the lowest and above the highest displayed reading.
DAY_Y_PAD = 12
Y_MIN_AGP, Y_MAX_AGP = 50, 320


# =============================================================================
# RECORD CONSTRUCTION
# =============================================================================

def build_twin(df: pd.DataFrame, smooth_min: int = SMOOTH_WINDOW_MIN,
               bin_min: int = CLOCK_BIN_MIN) -> pd.DataFrame:
    """
    Build record B from record A (see the module docstring).

    Parameters
    ----------
    df : pd.DataFrame
        Record A, columns ``[TIME_COL, GLUCOSE_COL]``, time-sorted.
    smooth_min : int
        Width of the centred rolling mean, in minutes.  Wider merges more
        excursions, so it moves the count of brief excursions further and
        the AGP slightly more.
    bin_min : int
        Clock-bin width the readings are permuted within, in minutes.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``GLUCOSE_COL`` replaced.  Timestamps, number of
        readings and multiset of values are unchanged.
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


def brief_excursions(df: pd.DataFrame, max_dur: int = MAX_DURATION_MIN) -> pd.DataFrame:
    """
    Hyperglycaemic excursions lasting at most *max_dur* minutes.

    No other condition is applied: this is an introduction figure, shown
    before any criterion has been defined, so duration is all it may use.
    """
    episodes = detect_hyperglycemia_episodes(df)
    if not episodes:
        return pd.DataFrame(columns=["onset_time", "offset_time", "duration_min"])
    table = pd.DataFrame(episodes)
    return table[table["duration_min"] <= max_dur].reset_index(drop=True)


# =============================================================================
# SHARED HELPERS
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


def _hours_of_day(timestamps: pd.Series) -> np.ndarray:
    """Fractional hour of day for each timestamp."""
    return (timestamps.dt.hour
            + timestamps.dt.minute / 60
            + timestamps.dt.second / 3600).to_numpy()


def select_window(df: pd.DataFrame, excursions: pd.DataFrame, n_days: int,
                  timestep_min: float):
    """
    Pick the fortnight to show.

    Candidates must first clear ``MIN_WINDOW_COVERAGE`` — a fortnight
    punctured by sensor gaps makes a poor slide, and missing days would make
    the two columns hard to compare.  Among those, the one richest in the
    *observed* record's brief excursions wins.

    The criterion deliberately looks only at record A, which is real data:
    choosing the window by how well record B happens to come out would be
    selecting on the very effect the figure claims to show.
    """
    t_first = df[TIME_COL].min().normalize()
    t_last  = df[TIME_COL].max()
    n_avail = int((t_last - t_first).total_seconds() / 86400) + 1

    if n_avail <= n_days:
        return t_first

    expected = n_days * 24 * 60 / timestep_min
    best_score, best_start, best_coverage = (-1, -1), t_first, 0.0

    for i in range(n_avail - n_days + 1):
        ws = t_first + pd.Timedelta(days=i)
        we = ws + pd.Timedelta(days=n_days)
        n_pts = int(((df[TIME_COL] >= ws) & (df[TIME_COL] < we)).sum())
        coverage = n_pts / expected
        if coverage < MIN_WINDOW_COVERAGE:
            continue
        n_exc = 0 if excursions.empty else int(
            ((excursions["onset_time"] >= ws) & (excursions["offset_time"] < we)).sum())
        score = (n_exc, n_pts)
        if score > best_score:
            best_score, best_start, best_coverage = score, ws, coverage

    if best_score == (-1, -1):        # nothing cleared the gate — take the fullest
        return select_window(df, excursions, n_days, timestep_min * 1e9)

    print(f"  Fortnight coverage : {100 * best_coverage:.1f} %")
    return best_start


# =============================================================================
# FIGURES 1a / 1b — two weeks, side by side
# =============================================================================

def _day_ylim(records: list[tuple], t_zero: pd.Timestamp, n_days: int) -> tuple[float, float]:
    """
    A single y-range covering every reading both columns will display.

    Derived from the data rather than fixed, so no peak is clipped (a
    flat-topped excursion invites the wrong question) and no row wastes half
    its height on an empty band.  Shared across both columns, because two
    records drawn on different scales cannot be compared by eye.
    """
    window_end = t_zero + pd.Timedelta(days=n_days)
    lows, highs = [], []
    for _, _, df_rec, _ in records:
        window = df_rec[(df_rec[TIME_COL] >= t_zero) & (df_rec[TIME_COL] < window_end)]
        if window.empty:
            continue
        lows.append(float(window[GLUCOSE_COL].min()))
        highs.append(float(window[GLUCOSE_COL].max()))
    if not lows:
        return 40.0, 360.0
    return min(lows) - DAY_Y_PAD, max(highs) + DAY_Y_PAD


def _draw_day_row(ax, day_data: pd.DataFrame, marks: pd.DataFrame,
                  gap_min: float, color: str, show_xticks: bool,
                  day_label: str | None, ylim: tuple[float, float]) -> None:
    """One day of one record, as a row of the fortnight column."""
    for hour in (6, 12, 18):
        ax.axvline(hour, color="#E8E8E8", lw=0.5, zorder=0)

    ax.axhline(THRESHOLD_HYPER, color=COLOR_THRESHOLD_HYPER, lw=0.8, ls="--",
               alpha=0.85, zorder=2)

    for _, mark in marks.iterrows():
        x_on  = mark["onset_time"].hour + mark["onset_time"].minute / 60
        x_off = mark["offset_time"].hour + mark["offset_time"].minute / 60
        if x_off <= x_on:            # excursion running past midnight
            x_off = 24.0
        ax.axvspan(x_on, x_off, color=COLOR_MARK, alpha=ALPHA_MARK, lw=0, zorder=1)

    for seg in _split_by_gaps(day_data, gap_min):
        ax.plot(_hours_of_day(seg[TIME_COL]),
                seg[GLUCOSE_COL].to_numpy(dtype=float),
                color=color, lw=1.0, alpha=0.95, zorder=3)

    ax.set_xlim(0, 24)
    ax.set_ylim(*ylim)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#DDDDDD")
    ax.grid(False)

    if show_xticks:
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "24:00"],
                           fontsize=FS_TICK)
        ax.tick_params(axis="x", length=2.5, pad=2)
    else:
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.tick_params(axis="x", labelbottom=False, length=0)

    if day_label is not None:
        ax.annotate(day_label, xy=(0, 0.5), xycoords="axes fraction",
                    xytext=(-8, 0), textcoords="offset points",
                    fontsize=FS_DAY, color=COLOR_ANNOTATION,
                    ha="right", va="center", annotation_clip=False)


def build_two_weeks_figure(patient_id: str, records: list[tuple], t_zero: pd.Timestamp,
                           n_days: int, gap_min: float, max_dur: int,
                           marked: bool) -> plt.Figure:
    """
    Both records' fortnights, one column each.

    Parameters
    ----------
    records : list of (title, colour, dataframe, excursion table)
    marked : bool
        Whether to shade the brief excursions.  The unmarked version is the
        same slide with nothing highlighted, so the two can be shown in
        sequence and only the marks change.
    """
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    top, bottom = 0.790, 0.072
    gs = fig.add_gridspec(
        n_days, 2, left=0.075, right=0.975, top=top, bottom=bottom,
        wspace=0.13, hspace=0.11,
    )

    ylim = _day_ylim(records, t_zero, n_days)
    counts = []
    for col, (title, color, df_rec, excursions) in enumerate(records):
        n_marked = 0
        for day in range(n_days):
            day_start = t_zero + pd.Timedelta(days=day)
            day_end   = day_start + pd.Timedelta(days=1)

            day_data = df_rec[(df_rec[TIME_COL] >= day_start)
                              & (df_rec[TIME_COL] < day_end)].reset_index(drop=True)

            if marked and not excursions.empty:
                day_marks = excursions[(excursions["onset_time"] >= day_start)
                                       & (excursions["onset_time"] < day_end)]
            else:
                day_marks = excursions.iloc[0:0]
            n_marked += len(day_marks)

            ax = fig.add_subplot(gs[day, col])
            _draw_day_row(
                ax, day_data, day_marks, gap_min, color,
                show_xticks=(day == n_days - 1),
                day_label=f"{day + 1}" if col == 0 else None,
                ylim=ylim,
            )
        counts.append(n_marked)

    # ── Column headings ───────────────────────────────────────────────────────
    for col, (title, color, _, _) in enumerate(records):
        x = 0.075 + (0.975 - 0.075) * (0.25 + 0.5 * col)
        fig.text(x, top + 0.035, title, fontsize=FS_COLHEAD, fontweight="bold",
                 color=color, ha="center", va="bottom")
        if marked:
            fig.text(x, top + 0.005,
                     f"{counts[col]} brief excursions in {n_days} days",
                     fontsize=FS_LABEL, color=color, ha="center", va="bottom")

    fig.text(0.02, 0.5, f"Day (1–{n_days})", fontsize=FS_LABEL,
             color=COLOR_ANNOTATION, rotation=90, ha="center", va="center")

    # ── Slide titles ──────────────────────────────────────────────────────────
    if marked:
        headline = (f"Every hyperglycaemic excursion of {max_dur} minutes or less")
        subtitle = ("The same two weeks. One record spends its time above range in "
                    "brief bursts, the other in long stretches.")
    else:
        headline = f"Two weeks of glucose, two records"
        subtitle = ("Fourteen consecutive days each, same scale, "
                    f"same threshold at {format_glucose(THRESHOLD_HYPER)}.")

    fig.text(0.075, 0.955, headline, fontsize=FS_SUPTITLE, fontweight="bold",
             color="#111111", ha="left", va="center")
    fig.text(0.075, 0.908, subtitle, fontsize=FS_SUBTITLE, color=COLOR_ANNOTATION,
             ha="left", va="center")

    handles = [Line2D([0], [0], color=COLOR_THRESHOLD_HYPER, lw=0.9, ls="--",
                      label=f"{format_glucose(THRESHOLD_HYPER)}")]
    if marked:
        handles.insert(0, mpatches.Patch(
            facecolor=COLOR_MARK, alpha=ALPHA_MARK,
            label=f"excursion above {format_glucose(THRESHOLD_HYPER)} "
                  f"lasting ≤ {max_dur} min"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.075, 0.862),
               fontsize=FS_ANNOT, frameon=False, ncol=2,
               handlelength=1.7, columnspacing=1.8)

    fig.text(0.975, 0.012,
             f"Record B is constructed from Record A's own readings, redistributed "
             f"in time — not a second participant.",
             fontsize=FS_ANNOT - 0.5, color=COLOR_ANNOTATION, ha="right",
             va="bottom", style="italic")

    return fig


# =============================================================================
# FIGURE 2 — AGP + metrics
# =============================================================================

def _metric_rows(metrics_a: dict, metrics_b: dict) -> list[tuple[str, str, str]]:
    """The (label, record A, record B) strings of the comparison table."""
    unit = glucose_unit_label()
    dec  = 0 if unit == "mg/dL" else 1

    def pct(key):
        return f"{metrics_a[key]:.1f}", f"{metrics_b[key]:.1f}"

    def glu(key):
        return (f"{float(to_display(metrics_a[key])):.{dec}f}",
                f"{float(to_display(metrics_b[key])):.{dec}f}")

    hypo  = format_glucose(THRESHOLD_HYPO,  with_unit=False)
    hyper = format_glucose(THRESHOLD_HYPER, with_unit=False)
    tar2  = format_glucose(THRESHOLD_TAR_LEVEL2, with_unit=False)

    return [
        (f"Time in range {hypo}–{hyper}  (%)", *pct("TIR")),
        (f"Time above {hyper}  (%)",           *pct("TAR")),
        (f"Time above {tar2}  (%)",            *pct("TAR2")),
        (f"Time below {hypo}  (%)",            *pct("TBR")),
        (f"Mean glucose  ({unit})",            *glu("Mean")),
        (f"SD  ({unit})",                      *glu("SD")),
        ("CV  (%)",                            *pct("CV")),
        ("GMI  (%)",  f"{metrics_a['GMI']:.2f}", f"{metrics_b['GMI']:.2f}"),
    ]


def _draw_agp_panel(ax, agp_a: pd.DataFrame, agp_b: pd.DataFrame,
                    labels: tuple[str, str], median_rmse: float) -> None:
    """The two ambulatory glucose profiles, overlaid."""
    hours = np.arange(len(agp_a)) * (24.0 / len(agp_a))

    ax.axhspan(THRESHOLD_HYPO, THRESHOLD_HYPER, color=COLOR_AGP_TARGET_ZONE,
               alpha=0.07, zorder=0, lw=0)
    for value, color in [(THRESHOLD_HYPER, COLOR_THRESHOLD_HYPER),
                         (THRESHOLD_HYPO,  COLOR_THRESHOLD_HYPO)]:
        ax.axhline(value, color=color, lw=1.3, ls="--", alpha=0.9, zorder=5)

    for agp, color, label in [(agp_a, COLOR_A, labels[0]),
                              (agp_b, COLOR_B, labels[1])]:
        ax.fill_between(hours, agp["q25"], agp["q75"],
                        color=color, alpha=0.16, lw=0, zorder=2)
        ax.plot(hours, agp["median"], color=color, lw=2.6, zorder=4, label=label)

    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)], fontsize=FS_TICK)
    ax.set_xlabel("Time of day", fontsize=FS_LABEL, labelpad=4)

    positions, labels_y = glucose_ticks(Y_MIN_AGP, Y_MAX_AGP, max_ticks=6)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels_y, fontsize=FS_TICK)
    ax.set_ylim(Y_MIN_AGP, Y_MAX_AGP)
    ax.set_ylabel(glucose_label(), fontsize=FS_LABEL, labelpad=4)
    ax.yaxis.grid(True, ls=":", lw=0.5, color="#DDDDDD", zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title("Ambulatory glucose profile", fontsize=FS_PANEL,
                 fontweight="bold", loc="left", pad=6)

    # The RMSE arrives in mg/dL like every other glucose quantity here; the
    # mg/dL → mmol/L conversion is a pure ratio, so it applies to a difference
    # unchanged.  Converting the label without the value would misstate it by
    # a factor of 18.
    rmse_display = float(to_display(median_rmse))
    decimals = 1 if glucose_unit_label() == "mg/dL" else 2
    ax.annotate(
        f"median curves differ by {rmse_display:.{decimals}f} "
        f"{glucose_unit_label()} RMSE\nbands: 25–75th percentile",
        xy=(0.02, 0.975), xycoords="axes fraction", fontsize=FS_ANNOT,
        color=COLOR_ANNOTATION, ha="left", va="top",
    )
    ax.legend(loc="upper right", fontsize=FS_LABEL, frameon=False, handlelength=1.7)


def _draw_metric_panel(ax, rows: list[tuple[str, str, str]],
                       labels: tuple[str, str]) -> None:
    """The consensus metrics of both records, side by side."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x_label, x_a, x_b = 0.02, 0.76, 0.99
    y_top, y_bottom = 0.87, 0.10
    row_h = (y_top - y_bottom) / len(rows)

    ax.text(0.0, 1.0, "Consensus metrics", fontsize=FS_PANEL, fontweight="bold",
            color="#111111", ha="left", va="top", transform=ax.transAxes)

    y_head = y_top + 0.045
    for x, text, color in [(x_a, labels[0], COLOR_A), (x_b, labels[1], COLOR_B)]:
        ax.text(x, y_head, text, fontsize=FS_TABLE, fontweight="bold",
                color=color, ha="right", va="center")
    ax.plot([0, 1], [y_top + 0.015] * 2, color="#999999", lw=0.9)

    for i, (label, value_a, value_b) in enumerate(rows):
        y = y_top - (i + 0.5) * row_h
        if i % 2 == 1:
            ax.add_patch(Rectangle((0, y - row_h / 2), 1, row_h,
                                   facecolor="#F4F4F4", lw=0, zorder=0))
        ax.text(x_label, y, label, fontsize=FS_TABLE, color="#222222",
                ha="left", va="center", zorder=2)
        ax.text(x_a, y, value_a, fontsize=FS_TABLE, color=COLOR_A,
                ha="right", va="center", zorder=2)
        ax.text(x_b, y, value_b, fontsize=FS_TABLE, fontweight="bold",
                color=COLOR_B, ha="right", va="center", zorder=2)

    ax.plot([0, 1], [y_bottom - 0.015] * 2, color="#999999", lw=0.9)
    ax.text(0.5, y_bottom - 0.080, "identical to the last decimal",
            fontsize=FS_ANNOT, color=COLOR_ANNOTATION, ha="center", va="center")


def build_agp_figure(agp_a: pd.DataFrame, agp_b: pd.DataFrame,
                     metrics_a: dict, metrics_b: dict,
                     labels: tuple[str, str], median_rmse: float,
                     n_days_total: int) -> plt.Figure:
    """The 'every standard readout agrees' slide."""
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 0.88],
                          left=0.055, right=0.975, top=0.775, bottom=0.115,
                          wspace=0.22)

    _draw_agp_panel(fig.add_subplot(gs[0, 0]), agp_a, agp_b, labels, median_rmse)
    _draw_metric_panel(fig.add_subplot(gs[0, 1]), _metric_rows(metrics_a, metrics_b),
                       labels)

    fig.text(0.055, 0.945, "The standard report cannot tell them apart",
             fontsize=FS_SUPTITLE, fontweight="bold", color="#111111",
             ha="left", va="center")
    fig.text(0.055, 0.893,
             f"The same {n_days_total} days as the previous slide: the two records "
             f"superimpose on the profile clinicians read, and agree on every metric.",
             fontsize=FS_SUBTITLE, color=COLOR_ANNOTATION, ha="left", va="center")

    fig.text(0.975, 0.012,
             "Record B is constructed from Record A's own readings, redistributed "
             "in time — not a second participant.",
             fontsize=FS_ANNOT - 0.5, color=COLOR_ANNOTATION, ha="right",
             va="bottom", style="italic")

    return fig


# =============================================================================
# PIPELINE
# =============================================================================

def generate(patient_id: str, n_days: int, max_dur: int, smooth_min: int,
             output_dir: str) -> None:
    """Load → build record B → measure both → draw the three slides."""
    df_a = load_patient_window(patient_id)
    if df_a is None:
        print(f"[!] No data found for {patient_id}")
        return
    print(f"  Loaded {len(df_a):,} data points")

    timestep_min = detect_dominant_sampling_interval(df_a)
    gap_min      = timestep_min * GAP_FACTOR

    # Everything on all three slides must describe the same fourteen days, so
    # the window is chosen first and record B is built from that slice alone.
    # Built from the whole record instead, B's fortnight would draw values from
    # days outside it and the metrics would no longer match exactly.
    exc_full = brief_excursions(df_a, max_dur)
    t_zero = select_window(df_a, exc_full, n_days, timestep_min)
    window_end = t_zero + pd.Timedelta(days=n_days)

    df_a = df_a[(df_a[TIME_COL] >= t_zero)
                & (df_a[TIME_COL] < window_end)].reset_index(drop=True)
    df_b = build_twin(df_a, smooth_min=smooth_min, bin_min=CLOCK_BIN_MIN)
    print(f"  Fortnight : {t_zero.date()} (+{n_days} d, {len(df_a):,} readings)")

    metrics_a = compute_consensus_metrics(df_a[GLUCOSE_COL].to_numpy(dtype=float))
    metrics_b = compute_consensus_metrics(df_b[GLUCOSE_COL].to_numpy(dtype=float))
    worst = max(abs(metrics_b[k] - metrics_a[k]) for k in metrics_a)
    print(f"  Largest |Δ| across consensus metrics : {worst:.2e}")

    exc_a = brief_excursions(df_a, max_dur)
    exc_b = brief_excursions(df_b, max_dur)
    print(f"  Excursions ≤ {max_dur} min : A={len(exc_a)}  B={len(exc_b)}")

    agp_a = compute_agp_profile(df_a, AGP_RESOLUTION_MIN)
    agp_b = compute_agp_profile(df_b, AGP_RESOLUTION_MIN)
    median_rmse = float(np.sqrt(np.nanmean(
        (agp_a["median"].to_numpy() - agp_b["median"].to_numpy()) ** 2)))
    print(f"  AGP median RMSE : {median_rmse:.2f} mg/dL")

    labels = ("Record A", "Record B")
    records = [(labels[0], COLOR_A, df_a, exc_a),
               (labels[1], COLOR_B, df_b, exc_b)]

    apply_publication_style()
    matplotlib.rcParams.update({
        "font.size":       FS_TICK,
        "axes.labelsize":  FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_ANNOT,
    })

    os.makedirs(output_dir, exist_ok=True)

    for marked, suffix in [(False, "two_weeks_plain"), (True, "two_weeks_marked")]:
        fig = build_two_weeks_figure(patient_id, records, t_zero, n_days,
                                     gap_min, max_dur, marked=marked)
        # bbox_inches=None keeps the canvas at an exact 16:9 slide ratio.
        save_figure(fig, os.path.join(output_dir, f"{patient_id}_{suffix}"),
                    bbox_inches=None)

    fig = build_agp_figure(agp_a, agp_b, metrics_a, metrics_b, labels,
                           median_rmse, n_days)
    save_figure(fig, os.path.join(output_dir, f"{patient_id}_agp_metrics"),
                bbox_inches=None)


# =============================================================================
# CLI
# =============================================================================

def parse_arguments():
    p = argparse.ArgumentParser(
        description="Introduction slides — two records, same report, different pattern",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--patient_id", default=DEFAULT_PATIENT_ID)
    p.add_argument("--n_days", type=int, default=DEFAULT_N_DAYS,
                   help="Days per column in the fortnight figures")
    p.add_argument("--max_dur", type=int, default=MAX_DURATION_MIN,
                   help="Longest excursion still marked as brief (minutes)")
    p.add_argument("--smooth_min", type=int, default=SMOOTH_WINDOW_MIN,
                   help="Rolling-mean window used to build record B (minutes)")
    p.add_argument("--out_dir", default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def main():
    args = parse_arguments()
    args.patient_id = resolve_patient_id(args.patient_id)
    print(f"\nTwo-week introduction slides — {args.patient_id}")
    generate(args.patient_id, args.n_days, args.max_dur, args.smooth_min,
             args.out_dir)


if __name__ == "__main__":
    main()
