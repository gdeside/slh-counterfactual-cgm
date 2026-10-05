"""
src/visualization/visualization_helper.py
==========================================

Shared axis-setup helpers used across multiple figure modules.

Functions
---------
shade_nights_on_axis
    Add semi-transparent night-period bands to an Axes using absolute
    timestamps; tracks labelling state to avoid duplicate legend entries.
setup_relative_time_axis
    Configure a datetime Axes with per-day 'Day N' major labels and
    6-hourly minor ticks.

Classes
-------
RelativeDayFormatter
    ``matplotlib.dates.DateFormatter`` subclass that formats tick positions
    as 'Day N' relative to a given start timestamp.

All project-level configuration (night hours, figure widths) is imported
from ``src.config`` and ``src.config.figure_config``.
"""

import os
from datetime import datetime, time as dtime, timedelta

# =============================================================================
# THIRD-PARTY LIBRARIES
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =============================================================================
# PROJECT IMPORTS — Data & Processing
# =============================================================================
from src.preprocessing.load_data import load_patient_window, prepare_valid_data
from src.preprocessing.timestep import detect_dominant_sampling_interval
from src.episodes.detection import detect_hyperglycemia_episodes
from src.episodes.criteria import classify_episodes
from src.counterfactual.build_counterfactual import build_full_counterfactual_signal

# =============================================================================
# PROJECT IMPORTS — Configuration
# =============================================================================
from src.config import (
    TIME_COL, 
    GLUCOSE_COL,
    FIG_DIR,
)

from src.config.figure_config import FIGURE_WIDTH_INCHES

from src.episodes.episode_config import (
    H_HYPER,
    NIGHT_START_H, 
    NIGHT_END_H,
    USE_GAP_MERGING,
    LABEL_SLH,
)

from src.counterfactual.counterfactual_config import (
    APPLY_KERNEL,
    KERNEL_MIN,
    OFFSET_VALUES,
)

# =============================================================================
# PROJECT IMPORTS — Visualization
# =============================================================================
from src.visualization.style import apply_publication_style, save_figure, style_axes
from src.visualization.colors import (
    OFFSET_COLORS, 
    COLOR_NIGHT,
    COLOR_ORIGINAL,
    COLOR_THRESHOLD,
    COLOR_ZONE,
    COLOR_SLH_HIGHLIGHT,
)


def shade_nights_on_axis(ax, t_start, t_end, labeled_list):
    """
    Add night period shading to a matplotlib axis.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to shade
    t_start : pd.Timestamp
        Start of time window
    t_end : pd.Timestamp
        End of time window
    labeled_list : list
        Mutable list tracking if label has been added (modified in place)
    """
    current_date = t_start.date()
    end_date = t_end.date() + timedelta(days=1)
    
    while current_date <= end_date:
        for hour_start, hour_end in [(0, NIGHT_END_H), (NIGHT_START_H, 24)]:
            night_start = pd.Timestamp(datetime.combine(current_date, dtime(hour_start % 24, 0)))
            night_end = pd.Timestamp(datetime.combine(current_date, dtime(0, 0))) + pd.Timedelta(hours=hour_end)
            
            if night_end > t_start and night_start < t_end:
                label = f"Night ({NIGHT_START_H:02d}–{NIGHT_END_H:02d}h)" if not labeled_list else "_"
                ax.axvspan(
                    max(night_start, t_start), min(night_end, t_end),
                    color=COLOR_NIGHT, alpha=0.45, zorder=0, linewidth=0, label=label
                )
                labeled_list.append(True)
        
        current_date += timedelta(days=1)


class RelativeDayFormatter(mdates.DateFormatter):
    """Format x-axis tick as 'Day N' relative to a given start timestamp."""

    def __init__(self, t_start: pd.Timestamp):
        super().__init__("")
        if hasattr(t_start, 'tz') and t_start.tz is not None:
            self.day_start = t_start.tz_localize(None).normalize()
        else:
            self.day_start = t_start.normalize()

    def __call__(self, x, pos=None):
        dt = mdates.num2date(x)
        if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
            dt_pd = pd.Timestamp(dt).tz_localize(None)
        else:
            dt_pd = pd.Timestamp(dt)
        day_num = (dt_pd.normalize() - self.day_start).days + 1
        return f"Day {day_num}"


def setup_relative_time_axis(ax, t_start: pd.Timestamp, t_end: pd.Timestamp,
                             days_in_row: int):
    """
    Configure a datetime Axes with per-day 'Day N' major labels and 6-hourly minor ticks.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes whose x-axis encodes absolute datetimes.
    t_start : pd.Timestamp
        Start of the display window; used as the Day 1 reference.
    t_end : pd.Timestamp
        End of the display window.
    days_in_row : int
        Number of days shown on this row (currently unused internally but
        kept for API consistency with multi-row layouts).
    """
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(RelativeDayFormatter(t_start))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=6.5, rotation=0, ha="center")