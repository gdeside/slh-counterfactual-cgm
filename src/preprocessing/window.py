"""
src/preprocessing/window.py
============================

CGM optimal window extraction.

Functions to identify and extract the best fixed-duration window from a
CGM time series by maximising valid data coverage.  The search uses a
sliding window with binary-search counting (O(n log n)) and writes the
result to Excel together with provenance metadata.

Functions
---------
find_optimal_window
    Slide a fixed-duration window over the CGM recording and return the
    position with the highest valid-data coverage that meets
    ``MIN_COVERAGE_PCT``.
save_patient_window
    Save the extracted window DataFrame to Excel and return a metadata
    dictionary describing the saved file.

Author
------
Guillaume Deside

Date
----
2026-03-28
"""

# =============================================================================
# STANDARD LIBRARIES
# =============================================================================
import os
from datetime import timedelta

# =============================================================================
# THIRD-PARTY LIBRARIES
# =============================================================================
import numpy as np
import pandas as pd

# =============================================================================
# PROJECT IMPORTS
# =============================================================================
from src.config import (
    WINDOW_DAYS,
    MIN_COVERAGE_PCT,
    SLIDE_FREQ,
    TIME_COL,
)

from src.preprocessing.coverage import compute_coverage_percentage


# =============================================================================
# CORE WINDOW SEARCH
# =============================================================================
def find_optimal_window(
    df_flagged: pd.DataFrame,
    sampling_min: float,
) -> dict | None:
    """
    Find the optimal fixed-duration window maximizing data coverage.

    Parameters
    ----------
    df_flagged : pd.DataFrame
        DataFrame with quality flags:
        - flag_time_error
        - flag_outlier_glucose
    sampling_min : float
        Sampling interval (minutes)

    Returns
    -------
    dict or None
        Dictionary containing:
        - win_start
        - win_end
        - coverage_pct
        - n_valid
        - n_total
        - df_window

        Returns None if no window satisfies MIN_COVERAGE_PCT.

    Notes
    -----
    - Uses sliding windows of length WINDOW_DAYS
    - Window positions spaced by SLIDE_FREQ
    - Valid data excludes flagged rows
    - Efficient counting via binary search on timestamps
    """
    window_delta = timedelta(days=WINDOW_DAYS)

    # -------------------------------------------------------------------------
    # Identify valid timestamps
    # -------------------------------------------------------------------------
    valid_mask = (
        ~df_flagged["flag_time_error"] &
        ~df_flagged["flag_outlier_glucose"]
    )

    times_valid = df_flagged.loc[valid_mask, TIME_COL].values

    if len(times_valid) == 0:
        return None

    # Convert timestamps to int64 (nanoseconds)
    t_valid_ns = times_valid.astype("datetime64[ns]").astype("int64")

    # -------------------------------------------------------------------------
    # Define search range
    # -------------------------------------------------------------------------
    t_min = df_flagged[TIME_COL].min()
    t_max = df_flagged[TIME_COL].max()
    latest_start = t_max - window_delta

    # Data that already ARE one window (e.g. the released open dataset: 45 days minus one sample) are accepted
    # when they fall short of the window length by at most two sampling steps; anything shorter is rejected.
    if latest_start < t_min <= latest_start + timedelta(minutes=2 * sampling_min):
        latest_start = t_min

    if t_min > latest_start:
        return None  # insufficient total span

    candidates = pd.date_range(
        start=t_min,
        end=latest_start,
        freq=SLIDE_FREQ,
    )

    # -------------------------------------------------------------------------
    # Search best window
    # -------------------------------------------------------------------------
    best = {
        "coverage": -1.0,
        "start": None,
        "end": None,
        "n_valid": 0,
    }

    for start in candidates:
        end = start + window_delta

        lo = np.searchsorted(t_valid_ns, start.value, side="left")
        hi = np.searchsorted(t_valid_ns, end.value, side="right")

        n_valid = hi - lo

        coverage = compute_coverage_percentage(
            n_valid,
            WINDOW_DAYS,
            sampling_min,
        )

        if coverage > best["coverage"]:
            best.update({
                "coverage": coverage,
                "start": start,
                "end": end,
                "n_valid": n_valid,
            })

    # -------------------------------------------------------------------------
    # Validate threshold
    # -------------------------------------------------------------------------
    if best["coverage"] < MIN_COVERAGE_PCT:
        return None

    # -------------------------------------------------------------------------
    # Extract window data
    # -------------------------------------------------------------------------
    mask = (
        (df_flagged[TIME_COL] >= best["start"]) &
        (df_flagged[TIME_COL] < best["end"])
    )

    df_window = df_flagged.loc[mask].copy().reset_index(drop=True)

    return {
        "win_start": best["start"],
        "win_end": best["end"],
        "coverage_pct": round(best["coverage"], 2),
        "n_valid": best["n_valid"],
        "n_total": len(df_window),
        "df_window": df_window,
    }


# =============================================================================
# OUTPUT SAVING
# =============================================================================
def save_patient_window(
    patient_id: str,
    window_result: dict,
    sampling_min: float,
    output_base_path: str,
) -> dict:
    """
    Save extracted window to Excel and return metadata.

    Parameters
    ----------
    patient_id : str
        Patient identifier
    window_result : dict
        Output from find_optimal_window()
    sampling_min : float
        Sampling interval (minutes)
    output_base_path : str
        Base directory for saving results

    Returns
    -------
    dict
        Metadata describing the saved window

    Notes
    -----
    Filename format:
        {patient_id}_win1_<start>_to_<end>_covXXpct.xlsx
    """
    # -------------------------------------------------------------------------
    # Create patient directory
    # -------------------------------------------------------------------------
    patient_dir = os.path.join(output_base_path, patient_id)
    os.makedirs(patient_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Format filename
    # -------------------------------------------------------------------------
    start_str = window_result["win_start"].strftime("%Y-%m-%d")

    end_str = (
        window_result["win_end"] - timedelta(seconds=1)
    ).strftime("%Y-%m-%d")

    coverage_str = str(int(round(window_result["coverage_pct"])))

    filename = (
        f"{patient_id}_win1_{start_str}_to_{end_str}_cov{coverage_str}pct.xlsx"
    )

    filepath = os.path.join(patient_dir, filename)

    # -------------------------------------------------------------------------
    # Save Excel file
    # -------------------------------------------------------------------------
    window_result["df_window"].to_excel(filepath, index=False)

    # -------------------------------------------------------------------------
    # Compute flag rate
    # -------------------------------------------------------------------------
    n_total = window_result["n_total"]
    n_valid = window_result["n_valid"]

    flag_rate = (
        100 * (n_total - n_valid) / n_total
        if n_total > 0 else 0.0
    )

    return {
        "patient_id": patient_id,
        "win_start": window_result["win_start"],
        "win_end": window_result["win_end"],
        "coverage_pct": window_result["coverage_pct"],
        "n_valid": n_valid,
        "n_total": n_total,
        "flag_rate_pct": round(flag_rate, 2),
        "dominant_step_min": sampling_min,
        "filename": filename,
        "filepath": filepath,
    }