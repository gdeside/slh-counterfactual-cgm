"""
src/preprocessing/quality.py
=============================

CGM data quality flagging.

Functions to annotate CGM time series with row-level data-quality flags
covering time inconsistencies, data gaps, and glucose outliers.  No rows
are removed — flags are purely descriptive and consumed downstream by
window selection and analysis steps.

Functions
---------
attach_quality_flags
    Annotate a CGM DataFrame with ``flag_time_error``, ``flag_gap_before``,
    ``gap_before_min``, and ``flag_outlier_glucose`` columns.
prepare_valid_data
    Parse and filter a CGM DataFrame, removing rows with invalid timestamps,
    non-numeric glucose, or active quality flags.

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
from src.config import (
    TIME_COL,
    GLUCOSE_COL,
    TIME_ERROR_RATIO,
    GAP_MULTIPLE,
    GLUCOSE_MIN,
    GLUCOSE_MAX,
    COL_FLAG_TIME_ERROR,
    COL_FLAG_OUTLIER_GLUCOSE
)


# =============================================================================
# QUALITY FLAGGING
# =============================================================================
def attach_quality_flags(
    df: pd.DataFrame,
    dominant_step_min: float,
) -> pd.DataFrame:
    """
    Attach data-quality flags to CGM time series.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing at least:
        - TIME_COL (datetime)
        - GLUCOSE_COL (numeric)
    dominant_step_min : float
        Dominant sampling interval (minutes)

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with additional columns:

        - flag_time_error : bool
            True if time difference is unexpectedly small
            (step ≤ TIME_ERROR_RATIO × dominant_step)

        - flag_gap_before : bool
            True if gap before measurement is large
            (step > GAP_MULTIPLE × dominant_step)

        - gap_before_min : float
            Duration of gap (minutes), NaN if no gap

        - flag_outlier_glucose : bool
            True if glucose value is outside valid range
            [GLUCOSE_MIN, GLUCOSE_MAX]

    Notes
    -----
    - No rows are removed.
    - Flags are computed row-wise using time differences.
    - The first row will always have NaN time difference.
    """
    df = df.copy()

    # -------------------------------------------------------------------------
    # Compute time differences (minutes)
    # -------------------------------------------------------------------------
    diffs_min = df[TIME_COL].diff().dt.total_seconds() / 60

    # -------------------------------------------------------------------------
    # Time error flag (unexpectedly short intervals)
    # -------------------------------------------------------------------------
    error_threshold = TIME_ERROR_RATIO * dominant_step_min

    df["flag_time_error"] = (
        (diffs_min <= error_threshold) &
        diffs_min.notna()
    )

    # -------------------------------------------------------------------------
    # Gap flag (unexpectedly large intervals)
    # -------------------------------------------------------------------------
    gap_threshold = GAP_MULTIPLE * dominant_step_min

    df["flag_gap_before"] = (
        (diffs_min > gap_threshold) &
        diffs_min.notna()
    )

    df["gap_before_min"] = np.where(
        df["flag_gap_before"],
        diffs_min,
        np.nan,
    )

    # -------------------------------------------------------------------------
    # Glucose outlier flag
    # -------------------------------------------------------------------------
    df["flag_outlier_glucose"] = (
        (df[GLUCOSE_COL] < GLUCOSE_MIN) |
        (df[GLUCOSE_COL] > GLUCOSE_MAX)
    )

    return df


# =============================================================================
# DATA PREPARATION
# =============================================================================
def prepare_valid_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and filter CGM data for metrics computation.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with:
        - valid timestamps
        - numeric glucose
        - no flagged rows
    """
    df = df.copy()

    # Parse columns
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], dayfirst=True, errors="coerce")
    df[GLUCOSE_COL] = pd.to_numeric(df[GLUCOSE_COL], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=[TIME_COL, GLUCOSE_COL])

    # Remove flagged rows (if present)
    if COL_FLAG_TIME_ERROR in df.columns:
        df = df[~df[COL_FLAG_TIME_ERROR].astype(bool)]

    if COL_FLAG_OUTLIER_GLUCOSE in df.columns:
        df = df[~df[COL_FLAG_OUTLIER_GLUCOSE].astype(bool)]

    return df
