"""
src/visualization/agp_plot.py
=============================

Panel-level plotting functions for the ambulatory glucose profile (AGP)
figure and its companion panels.

Functions
---------
plot_agp_panel
    Draw the AGP percentile bands, median/mean lines, and clinical
    threshold annotations on a single Axes.
plot_recurrence_classification_panel
    Stacked-area recurrence heatmap coloured by SLH / LLH classification.
plot_recurrence_duration_panel
    Stacked-area recurrence heatmap coloured by episode-duration bins.
plot_7day_trace
    Seven-day CGM trace with episode shading and relative-day x-axis.
add_panel_label
    Stamp a bold panel letter (A, B, C …) in the top-left corner of an Axes.
plot_metrics_table
    Render a transposed metrics comparison table for two patients.

All colour constants are sourced from ``src.visualization.colors``; clinical
thresholds, column names, and classification labels are imported from
``src.config``; AGP axis limits come from ``src.config.figure_config``.

Glucose data and axis limits stay in mg/dL throughout; the unit shown on
the axes, annotations and tables follows the display unit (see
``src.visualization.glucose_axis``).
"""

from datetime import datetime, time, timedelta

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    TIME_COL,
    GLUCOSE_COL,
    THRESHOLD_HYPER,
    THRESHOLD_HYPO,
    COL_EPISODE_START,
    COL_EPISODE_END,
    COL_EPISODE_CLASSIFICATION,
    CLASSIFICATION_SLH,
    CLASSIFICATION_LLH,
)
from src.config.figure_config import AGP_Y_MIN, AGP_Y_MAX
from src.utils.unit_conversion import format_glucose, glucose_unit_label
from src.visualization.glucose_axis import apply_glucose_axis, glucose_cell

from src.visualization.colors import (
    COLOR_AGP_PERCENTILE_95,
    COLOR_AGP_PERCENTILE_IQR,
    COLOR_AGP_TARGET_ZONE,
    COLOR_AGP_MEDIAN,
    COLOR_AGP_MEAN,
    COLOR_THRESHOLD_HYPER,
    COLOR_THRESHOLD_HYPO,
    COLOR_SLH,
    COLOR_LLH,
    COLOR_RECURRENCE_THRESHOLD,
    COLOR_DURATION_BINS,
)
from src.visualization.style import style_axes


# =============================================================================
# RELATIVE DAY FORMATTER
# =============================================================================
class RelativeDayFormatter(mdates.DateFormatter):
    """Format dates as 'Day N' relative to a start date."""

    def __init__(self, t_start):
        super().__init__("")
        if hasattr(t_start, "tz") and t_start.tz is not None:
            self.day_start = t_start.tz_localize(None).normalize()
        else:
            self.day_start = t_start.normalize()

    def __call__(self, x, pos=None):
        dt = mdates.num2date(x)
        if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
            dt_pd = pd.Timestamp(dt).tz_localize(None)
        else:
            dt_pd = pd.Timestamp(dt)
        dt_normalized = dt_pd.normalize()
        day_num = (dt_normalized - self.day_start).days + 1
        return f"Day {day_num}"


# =============================================================================
# VISUALIZATION CONSTANTS
# =============================================================================
ALPHA_AGP_PERCENTILE_95 = 0.12
ALPHA_AGP_PERCENTILE_IQR = 0.28
ALPHA_AGP_MEAN = 0.85
ALPHA_AGP_TARGET_ZONE = 0.04
ALPHA_THRESHOLD = 0.75
ALPHA_RECURRENCE_FILL = 0.88

TRACE_N_DAYS = 7                       # Days to display in glucose trace

CLASSIFICATION_LABELS = [CLASSIFICATION_SLH, CLASSIFICATION_LLH]

# Default shading for plot_7day_trace's episode spans -- the pipeline's binary
# SLH/LLH pair. Callers with a finer category scheme (e.g. Figure 2/3's
# SLH / Not isolated / Night / LLH) pass their own category_col/colors/alphas.
DEFAULT_TRACE_CATEGORY_COLORS = {CLASSIFICATION_SLH: COLOR_SLH, CLASSIFICATION_LLH: COLOR_LLH}
DEFAULT_TRACE_CATEGORY_ALPHAS = {CLASSIFICATION_SLH: 0.25, CLASSIFICATION_LLH: 0.20}

AGP_TIME_SLOT_TOLERANCE = 0.4          # Fraction of resolution for time matching
AGP_X_TICK_INTERVAL_HOURS = 3          # X-axis tick interval (hours)

RECURRENCE_SYSTEMATIC_THRESHOLD = 50  # % threshold for systematic recurrence
RECURRENCE_Y_MAX_BUFFER = 1.08        # Buffer above max bar for y-axis limit
RECURRENCE_X_TICK_INTERVAL_HOURS = 3  # X-axis tick interval (hours)

DURATION_LABELS = ["0–15", "15–30", "30–45", "45–60", "60–75", "75–90", "90+"]


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def plot_agp_panel(ax, agp: pd.DataFrame, patient_id: str,
                   show_legend: bool = True, show_threshold_annotations: bool = True):
    """
    Plot AGP (Ambulatory Glucose Profile) on given axes with enhanced styling.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    agp : pd.DataFrame
        AGP profile data from compute_agp_profile().
    patient_id : str
        Patient identifier for title.
    show_legend : bool, optional
        Whether to display the legend (default: True).
    show_threshold_annotations : bool, optional
        Whether to display threshold value annotations (default: True).
    """
    today = datetime.today().date()
    x_values = [
        datetime.combine(today, datetime.strptime(t, "%H:%M").time())
        for t in agp["time_str"]
    ]

    ax.fill_between(
        x_values, agp["q5"], agp["q95"],
        alpha=ALPHA_AGP_PERCENTILE_95,
        color=COLOR_AGP_PERCENTILE_95,
        label="5–95 percentile",
        zorder=1,
    )

    ax.fill_between(
        x_values, agp["q25"], agp["q75"],
        alpha=0.30,
        color=COLOR_AGP_PERCENTILE_IQR,
        label="IQR (25–75 percentile)",
        zorder=2,
        hatch="///",
        edgecolor=COLOR_AGP_PERCENTILE_IQR,
    )

    ax.plot(x_values, agp["q25"], color=COLOR_AGP_PERCENTILE_IQR, lw=0.8,
            ls=":", alpha=0.7, zorder=3)
    ax.plot(x_values, agp["q75"], color=COLOR_AGP_PERCENTILE_IQR, lw=0.8,
            ls=":", alpha=0.7, zorder=3)

    ax.plot(
        x_values, agp["median"],
        color=COLOR_AGP_MEDIAN,
        linewidth=2.0,
        ls="-",
        label="Median",
        zorder=5,
    )
    ax.plot(
        x_values, agp["mean"],
        color=COLOR_AGP_MEAN,
        linewidth=1.5,
        linestyle="--",
        alpha=ALPHA_AGP_MEAN,
        label="Mean",
        zorder=4,
    )

    ax.axhline(THRESHOLD_HYPER, color=COLOR_THRESHOLD_HYPER, linewidth=1.2,
               alpha=ALPHA_THRESHOLD, linestyle="--", zorder=0)
    ax.axhline(THRESHOLD_HYPO, color=COLOR_THRESHOLD_HYPO, linewidth=1.2,
               alpha=ALPHA_THRESHOLD, linestyle="--", zorder=0)

    ax.axhspan(THRESHOLD_HYPO, THRESHOLD_HYPER, alpha=ALPHA_AGP_TARGET_ZONE,
               color=COLOR_AGP_TARGET_ZONE, zorder=0)

    # ── X-axis: full 24-hour span 00:00 → 24:00 ──────────────────────────────
    day_start = datetime.combine(today, time(0, 0))
    day_end = day_start + timedelta(days=1)
    ax.set_xlim(day_start, day_end)
    ax.set_ylim(AGP_Y_MIN, AGP_Y_MAX)

    tick_times = [day_start + timedelta(hours=h)
                  for h in range(0, 25, AGP_X_TICK_INTERVAL_HOURS)]
    tick_labels = [f"{h:02d}:00" for h in range(0, 25, AGP_X_TICK_INTERVAL_HOURS)]
    ax.set_xticks(tick_times)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel("Time of day", fontsize=9, labelpad=4)
    apply_glucose_axis(ax, fontsize=9, labelpad=4)
    ax.set_title(f"{patient_id}", fontsize=8, fontweight="bold", pad=6)
    ax.tick_params(axis="both", which="major", labelsize=8)

    if show_threshold_annotations:
        x_pos = mdates.date2num(day_end)
        ax.text(x_pos, THRESHOLD_HYPER, f"  {format_glucose(THRESHOLD_HYPER)}",
                fontsize=7.5, va="center", ha="left",
                color=COLOR_THRESHOLD_HYPER, fontweight="bold", clip_on=False)
        ax.text(x_pos, THRESHOLD_HYPO, f"  {format_glucose(THRESHOLD_HYPO)}",
                fontsize=7.5, va="center", ha="left",
                color=COLOR_THRESHOLD_HYPO, fontweight="bold", clip_on=False)

    if show_legend:
        ax.legend(loc="upper right", fontsize=6, ncol=2, framealpha=0.88,
                  handlelength=1.4, columnspacing=0.8)

    style_axes(ax)


def plot_recurrence_classification_panel(ax, occurrence: dict, patient_id: str,
                                         resolution_min: int,
                                         show_legend: bool = True,
                                         show_threshold_annotation: bool = True):
    """
    Plot recurrence heatmap by SLH/LLH classification.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    occurrence : dict
        Mapping from classification label to recurrence array.
    patient_id : str
        Patient identifier for title.
    resolution_min : int
        Time resolution in minutes.
    show_legend : bool, optional
        Whether to display the legend (default: True).
    show_threshold_annotation : bool, optional
        Whether to display threshold value annotation (default: True).
    """
    n_slots = int(24 * 60 / resolution_min)
    time_hours = np.arange(n_slots) * resolution_min / 60
    bottom = np.zeros(n_slots)

    colors = {CLASSIFICATION_SLH: COLOR_SLH, CLASSIFICATION_LLH: COLOR_LLH}

    for label in CLASSIFICATION_LABELS:
        ax.fill_between(
            time_hours, bottom, bottom + occurrence[label],
            color=colors[label],
            alpha=ALPHA_RECURRENCE_FILL,
            label=label,
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        bottom += occurrence[label]

    ax.axhline(RECURRENCE_SYSTEMATIC_THRESHOLD, color=COLOR_RECURRENCE_THRESHOLD,
               linestyle="--", linewidth=1.5, zorder=5)

    y_max = max(100, float(bottom.max()) * RECURRENCE_Y_MAX_BUFFER)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, y_max)
    ax.set_xticks(range(0, 25, RECURRENCE_X_TICK_INTERVAL_HOURS))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, RECURRENCE_X_TICK_INTERVAL_HOURS)])
    ax.set_xlabel("Time of day", fontsize=9, labelpad=4)
    ax.set_ylabel("% of days with hyperglycemia", fontsize=8, labelpad=4)
    ax.tick_params(axis="both", which="major", labelsize=8)

    if show_threshold_annotation:
        ax.text(24, RECURRENCE_SYSTEMATIC_THRESHOLD,
                f"  {RECURRENCE_SYSTEMATIC_THRESHOLD}%",
                fontsize=7.5, va="center", ha="left",
                color=COLOR_RECURRENCE_THRESHOLD, fontweight="bold", clip_on=False)

    if show_legend:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.88)

    style_axes(ax)


def plot_recurrence_duration_panel(ax, occurrence: dict, patient_id: str,
                                   resolution_min: int,
                                   show_legend: bool = True,
                                   show_threshold_annotation: bool = True):
    """
    Plot recurrence heatmap by duration bins.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    occurrence : dict
        Mapping from duration label to recurrence array.
    patient_id : str
        Patient identifier for title.
    resolution_min : int
        Time resolution in minutes.
    show_legend : bool, optional
        Whether to display the legend (default: True).
    show_threshold_annotation : bool, optional
        Whether to display threshold value annotation (default: True).
    """
    n_slots = int(24 * 60 / resolution_min)
    time_hours = np.arange(n_slots) * resolution_min / 60
    bottom = np.zeros(n_slots)

    for label, color in zip(DURATION_LABELS, COLOR_DURATION_BINS):
        ax.fill_between(
            time_hours, bottom, bottom + occurrence[label],
            color=color,
            alpha=ALPHA_RECURRENCE_FILL,
            label=f"{label} min",
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        bottom += occurrence[label]

    ax.axhline(RECURRENCE_SYSTEMATIC_THRESHOLD, color=COLOR_RECURRENCE_THRESHOLD,
               linestyle="--", linewidth=1.5, zorder=5)

    y_max = max(100, float(bottom.max()) * RECURRENCE_Y_MAX_BUFFER)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, y_max)
    ax.set_xticks(range(0, 25, RECURRENCE_X_TICK_INTERVAL_HOURS))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, RECURRENCE_X_TICK_INTERVAL_HOURS)])
    ax.set_xlabel("Time of day", fontsize=9, labelpad=4)
    ax.set_ylabel("% of days with hyperglycemia", fontsize=9, labelpad=4)
    ax.set_title(
        f"Recurrence by episode duration — Patient {patient_id}",
        fontsize=9, fontweight="bold", pad=6,
    )
    ax.tick_params(axis="both", which="major", labelsize=8)

    if show_threshold_annotation:
        ax.text(24, RECURRENCE_SYSTEMATIC_THRESHOLD,
                f"  {RECURRENCE_SYSTEMATIC_THRESHOLD}%",
                fontsize=7.5, va="center", ha="left",
                color=COLOR_RECURRENCE_THRESHOLD, fontweight="bold", clip_on=False)

    if show_legend:
        ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.88)

    style_axes(ax)


def plot_7day_trace(ax, valid_data: pd.DataFrame, episodes: pd.DataFrame,
                    patient_id: str, show_threshold_annotations: bool = True,
                    category_col: str = COL_EPISODE_CLASSIFICATION,
                    category_colors: dict | None = None,
                    category_alphas: dict | None = None,
                    category_order: list | None = None,
                    show_category_legend: bool = False):
    """
    Plot 7-day glucose trace with relative time (Day 1, Day 2, etc.).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    valid_data : pd.DataFrame
        Valid CGM data.
    episodes : pd.DataFrame
        Episode data with classifications.
    patient_id : str
        Patient identifier.
    show_threshold_annotations : bool, optional
        Whether to display threshold value annotations (default: True).
    category_col : str, optional
        Column in `episodes` used to colour each shaded span (default: the
        pipeline's binary classification column).
    category_colors : dict, optional
        Mapping from category value to shading colour (default: the binary
        SLH/LLH pair, `DEFAULT_TRACE_CATEGORY_COLORS`). Pass a finer mapping
        (e.g. Figure 2/3's `CATEGORY_COLORS`) to shade by more categories.
    category_alphas : dict, optional
        Per-category fill alpha (default: `DEFAULT_TRACE_CATEGORY_ALPHAS`,
        falling back to 0.20 for any category not listed).
    category_order : list, optional
        Category order for the legend (default: `category_colors` insertion
        order).
    show_category_legend : bool, optional
        Whether to draw a legend mapping each shaded colour to its category
        (default: False, matching the previous unlabelled shading).
    """
    category_colors = category_colors or DEFAULT_TRACE_CATEGORY_COLORS
    category_alphas = category_alphas or DEFAULT_TRACE_CATEGORY_ALPHAS
    category_order = category_order or list(category_colors.keys())
    start_time = valid_data[TIME_COL].min()
    end_time = start_time + timedelta(days=TRACE_N_DAYS)

    trace_data = valid_data[
        (valid_data[TIME_COL] >= start_time) &
        (valid_data[TIME_COL] <= end_time)
    ].copy()

    if len(trace_data) == 0:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                fontsize=9, transform=ax.transAxes)
        return

    ax.plot(
        trace_data[TIME_COL], trace_data[GLUCOSE_COL],
        color="#333333", linewidth=0.8, alpha=0.7, zorder=3,
    )

    ax.axhline(THRESHOLD_HYPER, color=COLOR_THRESHOLD_HYPER, linewidth=1.2,
               alpha=ALPHA_THRESHOLD, linestyle="--", zorder=1)
    ax.axhline(THRESHOLD_HYPO, color=COLOR_THRESHOLD_HYPO, linewidth=1.2,
               alpha=ALPHA_THRESHOLD, linestyle="--", zorder=1)

    ax.axhspan(THRESHOLD_HYPO, THRESHOLD_HYPER, alpha=ALPHA_AGP_TARGET_ZONE,
               color=COLOR_AGP_TARGET_ZONE, zorder=0)

    trace_episodes = episodes[
        (episodes[COL_EPISODE_START] >= start_time) &
        (episodes[COL_EPISODE_START] <= end_time)
    ].copy()

    for _, episode in trace_episodes.iterrows():
        cat = episode[category_col]
        if cat not in category_colors:
            continue
        ax.axvspan(episode[COL_EPISODE_START], episode[COL_EPISODE_END],
                   alpha=category_alphas.get(cat, 0.20), color=category_colors[cat], zorder=2)

    ax.set_xlim(start_time, end_time)
    ax.set_ylim(AGP_Y_MIN, AGP_Y_MAX)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(RelativeDayFormatter(start_time))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))

    ax.set_xlabel("Time (days)", fontsize=9, labelpad=4)
    apply_glucose_axis(ax, fontsize=9, labelpad=4)
    ax.tick_params(axis="both", which="major", labelsize=8)
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=8, rotation=0, ha="center")

    if show_threshold_annotations:
        ax.text(1.02, THRESHOLD_HYPER / AGP_Y_MAX, f"  {format_glucose(THRESHOLD_HYPER)}",
                fontsize=7.5, va="center", ha="left",
                color=COLOR_THRESHOLD_HYPER, fontweight="bold",
                transform=ax.transAxes, clip_on=False)
        ax.text(1.02, THRESHOLD_HYPO / AGP_Y_MAX, f"  {format_glucose(THRESHOLD_HYPO)}",
                fontsize=7.5, va="center", ha="left",
                color=COLOR_THRESHOLD_HYPO, fontweight="bold",
                transform=ax.transAxes, clip_on=False)

    if show_category_legend:
        handles = [mpatches.Patch(facecolor=category_colors[cat],
                                  alpha=category_alphas.get(cat, 0.20), label=cat)
                   for cat in category_order if cat in category_colors]
        ax.legend(handles=handles, loc="upper right", fontsize=6.5, ncol=2,
                 framealpha=0.88, handlelength=1.4, columnspacing=0.8)

    ax.grid(True, alpha=0.25, axis="both", linestyle=":")
    style_axes(ax)


def add_panel_label(ax, label: str):
    """
    Add panel label (A, B, C, etc.) to subplot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add label to.
    label : str
        Panel letter (e.g., 'A', 'B', 'C').
    """
    ax.text(
        -0.12, 1.05, label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="right",
    )


def plot_metrics_table(ax, metrics_a: dict, metrics_b: dict,
                       counts_a: dict, counts_b: dict,
                       patient_a: str, patient_b: str,
                       include_classification: bool = False):
    """
    Plot transposed metrics comparison table for two patients.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to render the table on (will have its axis turned off).
    metrics_a : dict
        Glucose metrics for patient A.
    metrics_b : dict
        Glucose metrics for patient B.
    counts_a : dict
        Episode counts by classification for patient A.
    counts_b : dict
        Episode counts by classification for patient B.
    patient_a : str
        Label for patient A row.
    patient_b : str
        Label for patient B row.
    include_classification : bool, optional
        Whether to include SLH/LLH split columns (default: False).
    """
    ax.axis("off")

    n_total_a = counts_a[CLASSIFICATION_SLH] + counts_a[CLASSIFICATION_LLH]
    n_total_b = counts_b[CLASSIFICATION_SLH] + counts_b[CLASSIFICATION_LLH]

    if include_classification:
        col_labels = [
            "n\nH", "n\nSLH", "n\nLLH",
            f"Mean\n({glucose_unit_label()})", f"Median\n({glucose_unit_label()})",
            f"SD\n({glucose_unit_label()})", "CV\n(%)",
            "TIR 70-180\n(%)", "TAR >180\n(%)", "TAR >250\n(%)", "TBR <70\n(%)",
        ]
        row_a = [
            n_total_a, counts_a[CLASSIFICATION_SLH], counts_a[CLASSIFICATION_LLH],
            glucose_cell(metrics_a.get("Mean", "-")),
            glucose_cell(metrics_a.get("Median", "-")),
            glucose_cell(metrics_a.get("SD", "-")),
            metrics_a.get("CV", "-"),
            metrics_a.get("TIR70-180", "-"), metrics_a.get("TAR>180", "-"),
            metrics_a.get("TAR>250", "-"), metrics_a.get("TBR<70", "-"),
        ]
        row_b = [
            n_total_b, counts_b[CLASSIFICATION_SLH], counts_b[CLASSIFICATION_LLH],
            glucose_cell(metrics_b.get("Mean", "-")),
            glucose_cell(metrics_b.get("Median", "-")),
            glucose_cell(metrics_b.get("SD", "-")),
            metrics_b.get("CV", "-"),
            metrics_b.get("TIR70-180", "-"), metrics_b.get("TAR>180", "-"),
            metrics_b.get("TAR>250", "-"), metrics_b.get("TBR<70", "-"),
        ]
    else:
        col_labels = [
            "n_episodes",
            f"Mean\n({glucose_unit_label()})", f"Median\n({glucose_unit_label()})",
            f"SD\n({glucose_unit_label()})", "CV\n(%)",
            "TIR 70-180\n(%)", "TAR >180\n(%)", "TAR >250\n(%)", "TBR <70\n(%)",
        ]
        row_a = [
            n_total_a,
            glucose_cell(metrics_a.get("Mean", "-")),
            glucose_cell(metrics_a.get("Median", "-")),
            glucose_cell(metrics_a.get("SD", "-")),
            metrics_a.get("CV", "-"),
            metrics_a.get("TIR70-180", "-"), metrics_a.get("TAR>180", "-"),
            metrics_a.get("TAR>250", "-"), metrics_a.get("TBR<70", "-"),
        ]
        row_b = [
            n_total_b,
            glucose_cell(metrics_b.get("Mean", "-")),
            glucose_cell(metrics_b.get("Median", "-")),
            glucose_cell(metrics_b.get("SD", "-")),
            metrics_b.get("CV", "-"),
            metrics_b.get("TIR70-180", "-"), metrics_b.get("TAR>180", "-"),
            metrics_b.get("TAR>250", "-"), metrics_b.get("TBR<70", "-"),
        ]

    table_data = [row_a, row_b]
    row_labels = [patient_a, patient_b]

    COLOR_HEADER_BG = "#E8E8E8"
    COLOR_HEADER_TEXT = "#1A1A1A"
    COLOR_HIGHLIGHT_BG = "#D6EAF8"
    COLOR_ALT_ROW_BG = "#F9F9F9"
    COLOR_DATA_TEXT = "#2C2C2C"

    table = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.5)

    for i in range(len(col_labels)):
        cell = table[(0, i)]
        cell.set_facecolor(COLOR_HEADER_BG)
        cell.set_text_props(weight="bold", color=COLOR_HEADER_TEXT)

    for i in range(len(row_labels)):
        cell = table[(i + 1, -1)]
        cell.set_facecolor(COLOR_HEADER_BG)
        cell.set_text_props(weight="bold", color=COLOR_HEADER_TEXT)

    for i in range(2):
        cell = table[(i + 1, 0)]
        cell.set_facecolor(COLOR_HIGHLIGHT_BG)
        cell.set_text_props(weight="bold", color=COLOR_DATA_TEXT)

    for i in range(2):
        for j in range(1, len(col_labels)):
            cell = table[(i + 1, j)]
            if i % 2 == 0:
                cell.set_facecolor(COLOR_ALT_ROW_BG)
            cell.set_text_props(color=COLOR_DATA_TEXT)
