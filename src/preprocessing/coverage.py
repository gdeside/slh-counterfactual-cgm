"""
src/preprocessing/coverage.py
==============================

CGM coverage utilities.

Helper functions to compute expected measurements and data coverage
for CGM time series within fixed-duration windows.

Functions
---------
compute_expected_measurements
    Compute the expected number of CGM readings in a window given its
    duration and sampling interval.
compute_coverage_percentage
    Express valid measurement count as a percentage of the expected
    count, capped at 100 %.

Author
------
Guillaume Deside

Date
----
2026-03-28
"""

# =============================================================================
# COVERAGE COMPUTATIONS
# =============================================================================
def compute_expected_measurements(window_days: int, sampling_min: float) -> int:
    """
    Compute the expected number of measurements in a time window.

    Parameters
    ----------
    window_days : int
        Duration of the window (in days)
    sampling_min : float
        Sampling interval (in minutes)

    Returns
    -------
    int
        Expected number of measurements

    Notes
    -----
    Assumes uniform sampling over 24 hours per day.
    """
    return int(window_days * 24 * 60 / sampling_min)


def compute_coverage_percentage(
    n_valid: int,
    window_days: int,
    sampling_min: float,
) -> float:
    """
    Compute coverage percentage relative to expected measurements.

    Parameters
    ----------
    n_valid : int
        Number of valid (non-missing) measurements
    window_days : int
        Duration of the window (in days)
    sampling_min : float
        Sampling interval (in minutes)

    Returns
    -------
    float
        Coverage percentage, capped at 100%

    Notes
    -----
    Coverage is defined as:

        coverage (%) = 100 × (n_valid / expected)

    where expected is computed from the sampling frequency.
    """
    expected = compute_expected_measurements(window_days, sampling_min)

    if expected == 0:
        return 0.0

    return min(100.0, 100.0 * n_valid / expected)