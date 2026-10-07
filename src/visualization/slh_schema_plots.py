"""
src/visualization/slh_schema_plots.py
=======================================

Panel-level plotting functions for the SLH episode schema figure.

The figure uses relative time on both axes (no absolute timestamps) to
comply with patient-confidentiality requirements.  Data gaps are detected
and plotted as disconnected segments rather than interpolated lines.

Functions
---------
detect_sampling_interval
    Infer the dominant CGM sampling interval (5, 10, or 15 min) from the
    inter-measurement deltas.
split_by_gaps
    Split a time-sorted DataFrame into continuous segments separated by
    gaps larger than a given threshold.
timestamp_to_relative_hours
    Convert a Series of timestamps to fractional hours since a reference.
get_episode_color
    Return the fill color for a given episode classification label.
get_episode_linewidth
    Return the line width for a given episode classification label.
shade_nights_on_axis_relative
    Add semi-transparent night-hour bands to an Axes using relative hours.
draw_overview_panel
    Multi-day overview panel: raw CGM trace coloured by episode type,
    reference threshold lines, day separators, and a classification legend.
draw_zoom_panel
    Single-episode zoom panel: context windows, episode fill, duration
    annotation, and threshold reference lines.

Color and threshold constants are imported from ``src.visualization.colors``
and ``src.episodes.episode_config``.

Traces and thresholds are plotted in mg/dL; the y-axis and the threshold
annotations are rendered in the display unit (see
``src.visualization.glucose_axis``).
"""
from datetime import datetime, time as _dtime, timedelta
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

from src.config import TIME_COL, GLUCOSE_COL

from src.episodes.episode_config import H_HYPER, H_CONTEXT, H_HYPO
from src.episodes.episode_config import CONTEXT_WIN_MIN
from src.episodes.episode_config import NIGHT_START_H, NIGHT_END_H

from src.visualization.colors import COLOR_SLH, COLOR_LLH, COLOR_EXCLUDED
from src.visualization.colors import COLOR_BEFORE_WINDOW, COLOR_AFTER_WINDOW
from src.visualization.colors import COLOR_NIGHT, COLOR_TRACE, COLOR_HYPER, COLOR_CONTEXT, COLOR_HYPO

from src.visualization.style import style_axes
from src.visualization.glucose_axis import apply_glucose_axis
from src.utils.unit_conversion import format_glucose


# Transparency levels
ALPHA_EPISODE_FILL = 0.22
ALPHA_CONTEXT_WINDOW = 0.10

# Figure parameters
FONT_SIZE_BASE = 7

# Gap detection threshold (multiple of expected sampling interval)
GAP_THRESHOLD_FACTOR = 2.0  # Gap if Δt > 2.0 × expected interval


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def detect_sampling_interval(df: pd.DataFrame) -> float:
    """
    Detect the dominant CGM sampling interval in minutes.

    Computes inter-measurement deltas, snaps each to the nearest canonical
    interval (5, 10, or 15 min), and returns the mode.

    Parameters
    ----------
    df : pd.DataFrame
        CGM data containing a ``TIME_COL`` column sorted by time.

    Returns
    -------
    float
        Dominant sampling interval in minutes (typically 5.0, 10.0, or 15.0).
    """
    diffs = df[TIME_COL].diff().dt.total_seconds().dropna() / 60
    valid = diffs[(diffs > 1) & (diffs <= 60)]
    
    if valid.empty:
        return 5.0
    
    # Round to common intervals
    canonical = np.array([5, 10, 15])
    snapped = valid.apply(lambda x: canonical[np.argmin(np.abs(canonical - x))])
    
    return float(snapped.value_counts().idxmax())


def split_by_gaps(df: pd.DataFrame, gap_threshold_min: float) -> list[pd.DataFrame]:
    """
    Split DataFrame into continuous segments (no gaps > threshold).
    
    Parameters
    ----------
    df : pd.DataFrame
        Data sorted by TIME_COL
    gap_threshold_min : float
        Maximum allowed gap in minutes
        
    Returns
    -------
    list of pd.DataFrame
        Continuous segments
    """
    if len(df) < 2:
        return [df]
    
    # Calculate time differences
    diffs = df[TIME_COL].diff().dt.total_seconds() / 60
    
    # Find gap indices (where diff > threshold)
    gap_mask = diffs > gap_threshold_min
    gap_indices = df.index[gap_mask].tolist()
    
    if not gap_indices:
        return [df]
    
    # Split at gaps
    segments = []
    start_idx = df.index[0]
    
    for gap_idx in gap_indices:
        # Segment ends just before gap
        segment = df.loc[start_idx:gap_idx-1]
        if len(segment) > 0:
            segments.append(segment)
        start_idx = gap_idx
    
    # Add final segment
    final_segment = df.loc[start_idx:]
    if len(final_segment) > 0:
        segments.append(final_segment)
    
    return segments


def timestamp_to_relative_hours(timestamps: pd.Series, t_zero: pd.Timestamp) -> np.ndarray:
    """
    Convert timestamps to fractional hours elapsed since *t_zero*.

    Parameters
    ----------
    timestamps : pd.Series
        Series of ``pd.Timestamp`` values.
    t_zero : pd.Timestamp
        Reference time (t = 0).

    Returns
    -------
    np.ndarray
        Array of elapsed hours (float).
    """
    return (timestamps - t_zero).dt.total_seconds() / 3600


def get_episode_color(episode_type: str) -> str:
    """
    Return the fill color hex string for a given episode classification.

    Parameters
    ----------
    episode_type : str
        Classification label — ``"SLH"``, ``"LLH"``, or anything else
        (falls back to the excluded color).

    Returns
    -------
    str
        Hex color string.
    """
    return {
        "SLH": COLOR_SLH,
        "LLH": COLOR_LLH,
    }.get(episode_type, COLOR_EXCLUDED)


def get_episode_linewidth(episode_type: str) -> float:
    """
    Return the line width for plotting a given episode classification.

    Parameters
    ----------
    episode_type : str
        Classification label — ``"SLH"``, ``"LLH"``, or anything else
        (falls back to a thin line).

    Returns
    -------
    float
        Line width in points.
    """
    return {
        "SLH": 1.6,
        "LLH": 1.2,
    }.get(episode_type, 0.9)


# =============================================================================
# OVERVIEW PANEL (TOP PANEL)
# =============================================================================
def shade_nights_on_axis_relative(ax, t_zero, t_start, t_end):
    """
    Add semi-transparent night-hour bands to an Axes using relative hours.

    Night periods are defined by ``NIGHT_START_H`` and ``NIGHT_END_H`` from
    the episode configuration.  The x-axis is assumed to represent hours
    elapsed since *t_zero*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to shade.
    t_zero : pd.Timestamp
        Reference time corresponding to x = 0 on the axis.
    t_start : pd.Timestamp
        Absolute start of the display window (used to clip shading).
    t_end : pd.Timestamp
        Absolute end of the display window (used to clip shading).
    """
    current_date = t_start.date()
    end_date = t_end.date() + timedelta(days=1)
    
    while current_date <= end_date:
        for hour_start, hour_end in [(0, NIGHT_END_H), (NIGHT_START_H, 24)]:
            night_start = pd.Timestamp(datetime.combine(current_date, _dtime(hour_start % 24, 0)))
            night_end = pd.Timestamp(datetime.combine(current_date, _dtime(0, 0))) + pd.Timedelta(hours=hour_end)
            
            if night_end > t_start and night_start < t_end:
                # Convert to relative hours
                x_start = (max(night_start, t_start) - t_zero).total_seconds() / 3600
                x_end = (min(night_end, t_end) - t_zero).total_seconds() / 3600
                
                ax.axvspan(x_start, x_end, color=COLOR_NIGHT, alpha=1.0, zorder=0, linewidth=0)
        
        current_date += timedelta(days=1)


def draw_overview_panel(ax, df, classified, window_start, window_end):
    """
    Draw a multi-day overview panel with relative time and gap-safe rendering.

    The x-axis shows hours elapsed since *window_start* (e.g. "0h", "24h").
    Data gaps larger than ``GAP_THRESHOLD_FACTOR`` × sampling interval are
    rendered as disconnected line segments rather than interpolated.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    df : pd.DataFrame
        Full CGM data containing ``TIME_COL`` and ``GLUCOSE_COL`` columns.
    classified : list of dict
        Episode records as returned by the classification pipeline.  Each
        dict must contain ``"onset_time"``, ``"offset_time"``, and ``"type"``.
    window_start : pd.Timestamp
        Start of the display window (also the t = 0 reference).
    window_end : pd.Timestamp
        End of the display window.
    """
    # Reference time (t=0)
    t_zero = window_start
    
    # Night shading
    shade_nights_on_axis_relative(ax, t_zero, window_start, window_end)
    
    # Extract window data
    mask = (df[TIME_COL] >= window_start) & (df[TIME_COL] < window_end)
    df_window = df[mask].sort_values(TIME_COL).copy()
    
    # Detect sampling interval and gap threshold
    sampling_interval_min = detect_sampling_interval(df_window)
    gap_threshold_min = sampling_interval_min * GAP_THRESHOLD_FACTOR
    
    # Split into continuous segments
    segments = split_by_gaps(df_window, gap_threshold_min)
    
    # Convert to relative time
    df_window['time_rel_h'] = timestamp_to_relative_hours(df_window[TIME_COL], t_zero)
    
    # Plot raw trace (grey) - segment by segment
    for segment in segments:
        segment_rel = segment.copy()
        segment_rel['time_rel_h'] = timestamp_to_relative_hours(segment_rel[TIME_COL], t_zero)
        ax.plot(segment_rel['time_rel_h'], segment_rel[GLUCOSE_COL],
               color=COLOR_TRACE, lw=0.6, alpha=0.55, zorder=2)
    
    # Plot episode segments (colored by type)
    for ep in classified:
        if ep["onset_time"] >= window_end or ep["offset_time"] < window_start:
            continue
        
        color = get_episode_color(ep["type"])
        linewidth = get_episode_linewidth(ep["type"])
        
        ep_mask = (
            (df[TIME_COL] >= max(ep["onset_time"], window_start)) &
            (df[TIME_COL] <= min(ep["offset_time"], window_end))
        )
        segment = df.loc[ep_mask].sort_values(TIME_COL).copy()
        
        if segment.empty:
            continue
        
        # Convert to relative time
        segment['time_rel_h'] = timestamp_to_relative_hours(segment[TIME_COL], t_zero)
        
        # Split episode segment by gaps
        ep_segments = split_by_gaps(segment, gap_threshold_min)
        
        for ep_seg in ep_segments:
            ep_seg_rel = ep_seg.copy()
            ep_seg_rel['time_rel_h'] = timestamp_to_relative_hours(ep_seg_rel[TIME_COL], t_zero)
            
            ax.plot(ep_seg_rel['time_rel_h'], ep_seg_rel[GLUCOSE_COL],
                   color=color, lw=linewidth,
                   zorder=4 if ep["type"] == "SLH" else 3)
            
            above = ep_seg_rel[GLUCOSE_COL] >= H_HYPER
            ax.fill_between(ep_seg_rel['time_rel_h'], H_HYPER, ep_seg_rel[GLUCOSE_COL],
                           where=above, color=color, alpha=ALPHA_EPISODE_FILL, zorder=3)
    
    # Reference lines
    ax.axhline(H_HYPER, color=COLOR_HYPER, lw=1.0, ls="--", alpha=0.85, zorder=5)
    ax.axhline(H_CONTEXT, color=COLOR_CONTEXT, lw=0.8, ls=":", alpha=0.75, zorder=5)
    ax.axhline(H_HYPO, color=COLOR_HYPO, lw=0.6, ls=":", alpha=0.55, zorder=5)
    
    # Vertical day separators (at 24h intervals)
    duration_h = (window_end - window_start).total_seconds() / 3600
    for day in range(1, int(duration_h / 24) + 1):
        ax.axvline(day * 24, color="#cccccc", lw=0.5, ls="-", zorder=1)
    
    # Axes
    x_max = (window_end - t_zero).total_seconds() / 3600
    ax.set_xlim(0, x_max)
    ax.set_ylim(40, 380)
    
    # X-axis formatter: relative time
    def hours_formatter(x, pos):
        """Format x-axis as 'Day N' or 'Nh'."""
        if x == 0:
            return "0h"
        elif x % 24 == 0:
            return f"Day {int(x/24)}"
        else:
            return f"{int(x)}h"
    
    ax.xaxis.set_major_formatter(FuncFormatter(hours_formatter))
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(24))  # Every 24h
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(6))   # Every 6h
    
    ax.set_xlabel("Time (h)", fontsize=FONT_SIZE_BASE, labelpad=2)
    
    # Y-axis: clinical ticks in mg/dL, round mmol/L ticks otherwise
    apply_glucose_axis(
        ax,
        mgdl_ticks=[H_HYPO, 100, 140, H_HYPER, 250, 300],
        mgdl_ticklabels=["70", "100", "140", "180", "250", "300"],
        fontsize=FONT_SIZE_BASE + 1,
        labelpad=3,
    )
    
    # Threshold annotations (right margin)
    for val, label, color in [
        (H_HYPER, f"$H$ = {format_glucose(H_HYPER, with_unit=False)}", COLOR_HYPER),
        (H_CONTEXT, f"$H_c$ = {format_glucose(H_CONTEXT, with_unit=False)}", COLOR_CONTEXT),
    ]:
        ax.annotate(label,
                   xy=(x_max, val), xytext=(4, 0),
                   textcoords="offset points",
                   fontsize=5.5, color=color, va="center",
                   annotation_clip=False)
    
    # Legend
    handles = [
        mpatches.Patch(facecolor=COLOR_SLH, alpha=0.85, label="SLH"),
                mpatches.Patch(facecolor=COLOR_LLH, alpha=0.85, label=f"LLH"),
        # mpatches.Patch(facecolor=COLOR_LLH, alpha=0.85, label=f"LLH (> {D_MAX_MIN} min)"),
        mpatches.Patch(facecolor=COLOR_EXCLUDED, alpha=0.85, label="Excluded"),
        mpatches.Patch(facecolor=COLOR_NIGHT, edgecolor="#cccccc", lw=0.5,
                      label=f"Night ({NIGHT_START_H:02d}:00–{NIGHT_END_H:02d}:00)"),
        # Line2D([0], [0], color=COLOR_HYPER, lw=1.0, ls="--",
        #       label=f"$H$ = {H_HYPER:.0f} mg/dL"),
        # Line2D([0], [0], color=COLOR_CONTEXT, lw=0.8, ls=":",
        #       label=f"$H_c$ = {H_CONTEXT:.0f} mg/dL"),
    ]
    ax.legend(handles=handles, fontsize=6, loc="upper right", ncol=3,
             framealpha=0.92, columnspacing=0.7, handlelength=1.4)
    
    style_axes(ax)


# =============================================================================
# ZOOM PANEL (DETAILED VIEW)
# =============================================================================

def draw_zoom_panel(ax, df, episode, timestep_min, pad_min):
    """
    Draw a single-episode zoom panel with relative time and gap-safe rendering.

    The x-axis shows minutes relative to episode onset (negative = before,
    zero = onset, positive = after).  Context windows before and after the
    episode are shaded.  Data gaps larger than ``GAP_THRESHOLD_FACTOR`` ×
    *timestep_min* are rendered as disconnected segments.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    df : pd.DataFrame
        Full CGM data containing ``TIME_COL`` and ``GLUCOSE_COL`` columns.
    episode : dict
        Episode record with keys ``"onset_time"``, ``"offset_time"``, and
        ``"duration_min"``.
    timestep_min : float
        CGM sampling interval in minutes (used for gap detection).
    pad_min : float
        Extra time (minutes) to display before onset and after offset.
    """
    ctx = pd.Timedelta(minutes=CONTEXT_WIN_MIN)
    pad = pd.Timedelta(minutes=pad_min)
    
    onset = episode["onset_time"]
    offset = episode["offset_time"]
    duration = episode["duration_min"]
    
    zoom_start = onset - pad
    zoom_end = offset + pad
    
    # Reference time (t=0 at episode onset)
    t_zero = onset
    
    # Night shading (relative time)
    shade_nights_on_axis_relative(ax, t_zero, zoom_start, zoom_end)
    
    # Context window shading (relative coordinates)
    before_window_start_rel = -CONTEXT_WIN_MIN  # -60 min
    before_window_end_rel = 0                   # 0 min (onset)
    after_window_start_rel = (offset - onset).total_seconds() / 60  # offset relative to onset
    after_window_end_rel = after_window_start_rel + CONTEXT_WIN_MIN
    
    ax.axvspan(max(before_window_start_rel, -(pad_min)), before_window_end_rel,
              color=COLOR_BEFORE_WINDOW, alpha=ALPHA_CONTEXT_WINDOW,
              zorder=1, linewidth=0)
    
    ax.axvspan(after_window_start_rel, min(after_window_end_rel, duration + pad_min),
              color=COLOR_AFTER_WINDOW, alpha=ALPHA_CONTEXT_WINDOW,
              zorder=1, linewidth=0)
    
    # Context window boundary markers (relative)
    for t_rel, color in [
        (before_window_start_rel, COLOR_BEFORE_WINDOW),
        (before_window_end_rel, "#555555"),
        (after_window_start_rel, "#555555"),
        (after_window_end_rel, COLOR_AFTER_WINDOW)
    ]:
        ax.axvline(t_rel, color=color, lw=0.7, ls="--", alpha=0.55, zorder=3)
    
    # Extract data
    mask = (df[TIME_COL] >= zoom_start) & (df[TIME_COL] <= zoom_end)
    segment = df[mask].sort_values(TIME_COL).copy()
    
    # Convert to relative time (minutes from onset)
    segment['time_rel_min'] = (segment[TIME_COL] - t_zero).dt.total_seconds() / 60
    
    # Detect gaps and split
    gap_threshold_min = timestep_min * GAP_THRESHOLD_FACTOR
    segments = split_by_gaps(segment, gap_threshold_min)
    
    # Plot raw trace (grey) - segment by segment
    for seg in segments:
        seg_rel = seg.copy()
        seg_rel['time_rel_min'] = (seg_rel[TIME_COL] - t_zero).dt.total_seconds() / 60
        ax.plot(seg_rel['time_rel_min'], seg_rel[GLUCOSE_COL],
               color=COLOR_TRACE, lw=0.9, alpha=0.65, zorder=4)
    
    # Episode trace + fill
    ep_mask = (df[TIME_COL] >= onset) & (df[TIME_COL] <= offset)
    ep_segment = df[ep_mask].sort_values(TIME_COL).copy()
    
    if not ep_segment.empty:
        ep_segment['time_rel_min'] = (ep_segment[TIME_COL] - t_zero).dt.total_seconds() / 60
        
        # Split episode by gaps
        ep_segments = split_by_gaps(ep_segment, gap_threshold_min)
        
        for ep_seg in ep_segments:
            ep_seg_rel = ep_seg.copy()
            ep_seg_rel['time_rel_min'] = (ep_seg_rel[TIME_COL] - t_zero).dt.total_seconds() / 60
            
            ax.plot(ep_seg_rel['time_rel_min'], ep_seg_rel[GLUCOSE_COL],
                   color=COLOR_SLH, lw=1.6, zorder=5)
            
            above = ep_seg_rel[GLUCOSE_COL] >= H_HYPER
            ax.fill_between(ep_seg_rel['time_rel_min'], H_HYPER, ep_seg_rel[GLUCOSE_COL],
                           where=above, color=COLOR_SLH, alpha=0.30, zorder=4)
    
    # Reference lines
    ax.axhline(H_HYPER, color=COLOR_HYPER, lw=0.9, ls="--", alpha=0.85, zorder=6)
    ax.axhline(H_CONTEXT, color=COLOR_CONTEXT, lw=0.9, ls=":", alpha=0.85, zorder=6)
    ax.axhline(H_HYPO, color=COLOR_HYPO, lw=0.6, ls=":", alpha=0.50, zorder=6)
    
    # Duration annotation
    ep_midpoint_rel = duration / 2
    y_annotation = H_HYPER + (ax.get_ylim()[1] - H_HYPER) * 0.8 if len(segment) > 0 else H_HYPER + 30
    
    annot_text = f"{duration:.0f} min"
    
    ax.annotate(
        annot_text,
        xy=(ep_midpoint_rel, H_HYPER+20),
        xytext=(ep_midpoint_rel, y_annotation),
        ha="center", va="bottom",
        fontsize=6, color=COLOR_SLH, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=COLOR_SLH, lw=0.7),
        annotation_clip=False,
    )
    
    # Context window labels
    if len(segment) > 0:
        # y_label = ax.get_ylim()[0] + (H_HYPO - ax.get_ylim()[0]) * 0.8
        y_label = 200
    else:
        y_label = 200
    
    for midpoint_rel, color, text in [
        (before_window_start_rel / 2, COLOR_BEFORE_WINDOW, "60 min\nbefore"),
        (after_window_start_rel + CONTEXT_WIN_MIN / 2, COLOR_AFTER_WINDOW, "60 min\nafter"),
    ]:
        ax.text(midpoint_rel, y_label, text,
               ha="center", va="center", fontsize=5.0,
               color=color, style="italic")
    
    # Axes
    x_min = -(pad_min)
    x_max = duration + pad_min
    ax.set_xlim(x_min, x_max)
    
    if len(segment) > 0:
        y_min = max(30, float(segment[GLUCOSE_COL].min()) - 20)
        y_max = min(400, float(segment[GLUCOSE_COL].max()) + 35)
    else:
        y_min, y_max = 40, 300
    
    ax.set_ylim(y_min, y_max)
    
    # X-axis formatter: relative minutes
    def minutes_formatter(x, pos):
        """Format as '+X min' or '-X min'."""
        if x == 0:
            return "0\n(onset)"
        elif abs(x - duration) < 0.1:
            return f"{int(x)}\n(offset)"
        elif x > 0:
            return f"+{int(x)}"
        else:
            return f"{int(x)}"
    
    # ax.xaxis.set_major_formatter(FuncFormatter(minutes_formatter))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=8, integer=True))
    
    apply_glucose_axis(ax, fontsize=FONT_SIZE_BASE + 1, labelpad=3)
    ax.set_xlabel("Time (min)", fontsize=FONT_SIZE_BASE, labelpad=2)
    
    # Threshold annotations (right)
    for val, lbl, color in [
        (H_HYPER, f"$H$={format_glucose(H_HYPER, with_unit=False)}", COLOR_HYPER),
        (H_CONTEXT, f"$H_c$={format_glucose(H_CONTEXT, with_unit=False)}", COLOR_CONTEXT),
    ]:
        if ax.get_ylim()[0] < val < ax.get_ylim()[1]:
            ax.annotate(lbl,
                       xy=(x_max, val), xytext=(3, 0),
                       textcoords="offset points",
                       fontsize=5, color=color, va="center",
                       annotation_clip=False)
    
    # Legend
    # handles = [
    #     Line2D([0], [0], color=COLOR_SLH, lw=1.6, label="SLH episode"),
    #     Line2D([0], [0], color=COLOR_HYPER, lw=0.9, ls="--",
    #           label=f"$H$ = {H_HYPER:.0f}"),
    #     Line2D([0], [0], color=COLOR_CONTEXT, lw=0.9, ls=":",
    #           label=f"$H_c$ = {H_CONTEXT:.0f}"),
    # ]
    # ax.legend(handles=handles, fontsize=5.5, loc="upper right",
    #          framealpha=0.92, handlelength=1.2)
    
    style_axes(ax)