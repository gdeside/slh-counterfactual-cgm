"""
src/metrics/__init__.py
=======================

CGM metrics computation and loading module.

Exposes functions to compute standardised CGM metrics (statistics, time-in-range,
period medians) and to load pre-computed metrics from disk.

    from src.metrics import (
        compute_basic_statistics,
        compute_time_in_range_metrics,
        compute_time_based_metrics,
        compute_period_medians,
        create_empty_metrics,
        classify_time_of_day,
    )
    from src.metrics import compute_rhythm_metrics, compute_clock_lock
    from src.metrics import ALL_METRIC_NAMES, MIN_ROWS_FOR_METRICS
    from src.metrics import load_patient_metrics
"""

from .metrics import (
    # Functions
    classify_time_of_day,
    create_empty_metrics,
    compute_basic_statistics,
    compute_time_in_range_metrics,
    compute_time_based_metrics,
    compute_period_medians,
    # Constants – metric names
    ALL_METRIC_NAMES,
    MIN_ROWS_FOR_METRICS,
    METRIC_MEAN_GLUCOSE,
    METRIC_MEDIAN_GLUCOSE,
    METRIC_MIN_GLUCOSE,
    METRIC_MAX_GLUCOSE,
    METRIC_SD,
    METRIC_CV,
    METRIC_TIR70180,
    METRIC_TIR70140,
    METRIC_TIR60140,
    METRIC_TBR70,
    METRIC_TBR54,
    METRIC_TAR180,
    METRIC_TAR250,
    METRIC_CGM_DAYS,
    METRIC_MEDIAN_MORNING,
    METRIC_MEDIAN_NOON,
    METRIC_MEDIAN_AFTERNOON,
    METRIC_MEDIAN_EVENING,
    METRIC_MEDIAN_NIGHT,
)
from .rhythm import (
    # Functions
    build_day_clock_matrix,
    burden_concentration,
    ceiling_concentration,
    compute_clock_lock,
    compute_rhythm_metrics,
    peak_clock_hours,
    # Constants
    RHYTHM_METRIC_NAMES,
    N_CLOCK_BINS,
    TOP_FRACTION,
    MIN_DAYS_FOR_RHYTHM,
    METRIC_CLOCK_LOCK_HYPER,
    METRIC_CLOCK_LOCK_HYPO,
    METRIC_CLOCK_LOCK_P_HYPER,
    METRIC_CLOCK_LOCK_P_HYPO,
)
from .load_metrics import load_patient_metrics

__all__ = [
    # Functions
    "classify_time_of_day",
    "create_empty_metrics",
    "compute_basic_statistics",
    "compute_time_in_range_metrics",
    "compute_time_based_metrics",
    "compute_period_medians",
    "load_patient_metrics",
    "build_day_clock_matrix",
    "burden_concentration",
    "ceiling_concentration",
    "compute_clock_lock",
    "compute_rhythm_metrics",
    "peak_clock_hours",
    # Constants
    "ALL_METRIC_NAMES",
    "MIN_ROWS_FOR_METRICS",
    "METRIC_MEAN_GLUCOSE",
    "METRIC_MEDIAN_GLUCOSE",
    "METRIC_MIN_GLUCOSE",
    "METRIC_MAX_GLUCOSE",
    "METRIC_SD",
    "METRIC_CV",
    "METRIC_TIR70180",
    "METRIC_TIR70140",
    "METRIC_TIR60140",
    "METRIC_TBR70",
    "METRIC_TBR54",
    "METRIC_TAR180",
    "METRIC_TAR250",
    "METRIC_CGM_DAYS",
    "METRIC_MEDIAN_MORNING",
    "METRIC_MEDIAN_NOON",
    "METRIC_MEDIAN_AFTERNOON",
    "METRIC_MEDIAN_EVENING",
    "METRIC_MEDIAN_NIGHT",
    "RHYTHM_METRIC_NAMES",
    "N_CLOCK_BINS",
    "TOP_FRACTION",
    "MIN_DAYS_FOR_RHYTHM",
    "METRIC_CLOCK_LOCK_HYPER",
    "METRIC_CLOCK_LOCK_HYPO",
    "METRIC_CLOCK_LOCK_P_HYPER",
    "METRIC_CLOCK_LOCK_P_HYPO",
]
