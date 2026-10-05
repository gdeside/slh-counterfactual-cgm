"""
src/preprocessing/timestep.py
==============================

CGM sampling interval utilities.

Functions to analyse and standardise CGM sampling intervals.  Used in
preprocessing to filter patients by sampling frequency and to normalise
irregular intervals before coverage computation.

Functions
---------
snap_to_canonical
    Snap a measured sampling interval to the nearest canonical value
    (1, 5, 10, 15, 20, or 30 min) within a ±JITTER tolerance.
dominant_step_of_series
    Compute the median inter-measurement interval for a timestamp series.
detect_dominant_sampling_interval
    Infer the mode of gap-to-canonical-interval mappings from a DataFrame.

Author
------
Guillaume Deside

Date
----
2026-03-28
"""

# =============================================================================
# THIRD-PARTY LIBRARIES
# =============================================================================
import numpy as np
import pandas as pd

# =============================================================================
# PROJECT IMPORTS
# =============================================================================
from src.config import TIME_COL


# =============================================================================
# CONSTANTS
# =============================================================================
# Canonical sampling intervals (minutes)
CANONICAL_STEPS = np.array([1, 5, 10, 15, 20, 30])

# Tolerance for snapping (minutes)
JITTER = 2


# =============================================================================
# CORE FUNCTIONS
# =============================================================================
def snap_to_canonical(step_min: float) -> int | None:
    """
    Snap a measured sampling interval to the nearest canonical value.

    Parameters
    ----------
    step_min : float
        Measured sampling interval (minutes)

    Returns
    -------
    int or None
        Closest canonical interval if within tolerance (±JITTER),
        otherwise None
    """
    diffs = np.abs(CANONICAL_STEPS - step_min)
    idx = np.argmin(diffs)

    if diffs[idx] <= JITTER:
        return int(CANONICAL_STEPS[idx])

    return None


def dominant_step_of_series(times: pd.Series) -> float:
    """
    Compute median inter-measurement interval for a time series.

    Parameters
    ----------
    times : pd.Series
        Timestamp series (datetime)

    Returns
    -------
    float
        Median sampling interval in minutes, or NaN if unavailable

    Notes
    -----
    - Filters unrealistic gaps:
        * excludes < 1 min (noise)
        * excludes > 60 min (large gaps)
    """
    diffs = (
        times.sort_values()
        .diff()
        .dt.total_seconds()
        .dropna() / 60
    )

    valid = diffs[(diffs > 1) & (diffs <= 60)]

    return float(valid.median()) if not valid.empty else np.nan


def detect_dominant_sampling_interval(df: pd.DataFrame) -> float:
    """
    Infer dominant sampling interval using mode of rounded gaps.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing TIME_COL column

    Returns
    -------
    float
        Dominant sampling interval (minutes)

    Notes
    -----
    - Each gap is mapped to nearest canonical value
    - The most frequent (mode) interval is returned
    - Defaults to 5.0 if no valid intervals found
    """
    diffs = (
        df[TIME_COL]
        .diff()
        .dt.total_seconds()
        .dropna() / 60
    )

    valid = diffs[(diffs > 1) & (diffs <= 60)]

    if valid.empty:
        return 5.0

    # Map each gap to nearest canonical step
    rounded = valid.apply(
        lambda x: CANONICAL_STEPS[np.argmin(np.abs(CANONICAL_STEPS - x))]
    )

    return float(rounded.value_counts().idxmax())