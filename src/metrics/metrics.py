"""
CGM Metrics Computation Utilities
==================================

Provides functions to compute standardized CGM metrics:
- Basic statistics (mean, median, SD, CV)
- Time in range (TIR / TBR / TAR)
- Time-based metrics
- Period-based medians
- Time-of-day classification utilities
"""

import numpy as np
from src.config import TIME_COL, GLUCOSE_COL, THRESHOLD_HYPO, THRESHOLD_HYPER

# =============================================================================
# CONFIGURATION
# =============================================================================

ROUNDING_DECIMALS_GLUCOSE = 1  # Round glucose values to 2 decimal places
ROUNDING_DECIMALS_FRACTIONS = 3  # Round fractions to 3 decimal places

MIN_ROWS_FOR_METRICS = 10  # Minimum number of glucose readings required

# ── Additional glucose thresholds (mg/dL) not in src.config ──────────────────
# THRESHOLD_HYPO (70) and THRESHOLD_HYPER (180) are imported from src.config
THRESHOLD_TIR_TIGHT_UPPER = 140
THRESHOLD_TIR_LOWER_RELAXED = 60

THRESHOLD_TBR_LEVEL2 = 54

THRESHOLD_TAR_LEVEL2 = 250

# ── Time-of-day periods (minutes since midnight) ─────────────────────────────
TIME_PERIOD_MORNING = (390, 690)      # 06:30 – 11:30
TIME_PERIOD_NOON = (690, 930)         # 11:30 – 15:30
TIME_PERIOD_AFTERNOON = (930, 1110)   # 15:30 – 18:30
TIME_PERIOD_EVENING = (1110, 1350)    # 18:30 – 22:30

PERIOD_NAME_MORNING = "morning"
PERIOD_NAME_NOON = "noon"
PERIOD_NAME_AFTERNOON = "afternoon"
PERIOD_NAME_EVENING = "evening"
PERIOD_NAME_NIGHT = "night"

TIME_PERIOD_NAMES = [
    PERIOD_NAME_MORNING,
    PERIOD_NAME_NOON,
    PERIOD_NAME_AFTERNOON,
    PERIOD_NAME_EVENING,
    PERIOD_NAME_NIGHT,
]

# =============================================================================
# METRIC NAMES
# =============================================================================

METRIC_MEAN_GLUCOSE = "Mean_glycemia"
METRIC_MEDIAN_GLUCOSE = "Median_glycemia"
METRIC_MIN_GLUCOSE = "Minimum_glycemia"
METRIC_MAX_GLUCOSE = "Maximum_glycemia"
METRIC_SD = "SD"
METRIC_CV = "CV"

METRIC_TIR70180 = "TIR70180"
METRIC_TIR70140 = "TIR70140"
METRIC_TIR60140 = "TIR60140"
METRIC_TBR70 = "TBR70"
METRIC_TBR54 = "TBR54"
METRIC_TAR180 = "TAR180"
METRIC_TAR250 = "TAR250"

METRIC_CGM_DAYS = "CGM_days_measured"

METRIC_MEDIAN_MORNING = "Median_morning"
METRIC_MEDIAN_NOON = "Median_noon"
METRIC_MEDIAN_AFTERNOON = "Median_afternoon"
METRIC_MEDIAN_EVENING = "Median_evening"
METRIC_MEDIAN_NIGHT = "Median_night"

ALL_METRIC_NAMES = [
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
]

# =============================================================================
# TIME CLASSIFICATION
# =============================================================================

def classify_time_of_day(timestamp) -> str:
    """
    Classify timestamp into time-of-day period.

    Returns one of:
    morning, noon, afternoon, evening, night
    """
    minutes = timestamp.hour * 60 + timestamp.minute

    if TIME_PERIOD_MORNING[0] <= minutes < TIME_PERIOD_MORNING[1]:
        return PERIOD_NAME_MORNING
    elif TIME_PERIOD_NOON[0] <= minutes < TIME_PERIOD_NOON[1]:
        return PERIOD_NAME_NOON
    elif TIME_PERIOD_AFTERNOON[0] <= minutes < TIME_PERIOD_AFTERNOON[1]:
        return PERIOD_NAME_AFTERNOON
    elif TIME_PERIOD_EVENING[0] <= minutes < TIME_PERIOD_EVENING[1]:
        return PERIOD_NAME_EVENING
    else:
        return PERIOD_NAME_NIGHT


# =============================================================================
# BASE UTILITIES
# =============================================================================

def create_empty_metrics() -> dict:
    """
    Create empty metrics dictionary filled with NaN.
    """
    return {metric_name: np.nan for metric_name in ALL_METRIC_NAMES}


# =============================================================================
# BASIC STATISTICS
# =============================================================================

def compute_basic_statistics(glucose_values):
    """
    Compute basic glucose statistics.
    """
    mean_glucose = glucose_values.mean()
    median_glucose = glucose_values.median()
    min_glucose = glucose_values.min()
    max_glucose = glucose_values.max()
    sd_glucose = glucose_values.std()

    cv_glucose = (sd_glucose / mean_glucose) if mean_glucose > 0 else np.nan

    return {
        METRIC_MEAN_GLUCOSE: round(mean_glucose, ROUNDING_DECIMALS_GLUCOSE),
        METRIC_MEDIAN_GLUCOSE: round(median_glucose, ROUNDING_DECIMALS_GLUCOSE),
        METRIC_MIN_GLUCOSE: round(min_glucose, ROUNDING_DECIMALS_GLUCOSE),
        METRIC_MAX_GLUCOSE: round(max_glucose, ROUNDING_DECIMALS_GLUCOSE),
        METRIC_SD: round(sd_glucose, ROUNDING_DECIMALS_GLUCOSE),
        METRIC_CV: round(cv_glucose, ROUNDING_DECIMALS_FRACTIONS),
    }


# =============================================================================
# TIME-OF-DAY METRICS
# =============================================================================

def compute_period_medians(df):
    """
    Compute median glucose per time-of-day period.
    """
    df = df.copy()
    df["time_period"] = df[TIME_COL].apply(classify_time_of_day)

    period_medians = (
        df.groupby("time_period")[GLUCOSE_COL]
        .median()
        .reindex(TIME_PERIOD_NAMES)
    )

    return {
        METRIC_MEDIAN_MORNING: round(period_medians[PERIOD_NAME_MORNING], ROUNDING_DECIMALS_GLUCOSE),
        METRIC_MEDIAN_NOON: round(period_medians[PERIOD_NAME_NOON], ROUNDING_DECIMALS_GLUCOSE),
        METRIC_MEDIAN_AFTERNOON: round(period_medians[PERIOD_NAME_AFTERNOON], ROUNDING_DECIMALS_GLUCOSE),
        METRIC_MEDIAN_EVENING: round(period_medians[PERIOD_NAME_EVENING], ROUNDING_DECIMALS_GLUCOSE),
        METRIC_MEDIAN_NIGHT: round(period_medians[PERIOD_NAME_NIGHT], ROUNDING_DECIMALS_GLUCOSE),
    }


# =============================================================================
# TIME-BASED METRICS
# =============================================================================

def compute_time_based_metrics(df):
    """
    Compute CGM duration in days.
    """
    cgm_days = df[TIME_COL].dt.date.nunique()

    return {
        METRIC_CGM_DAYS: int(cgm_days),
    }


# =============================================================================
# TIME IN RANGE METRICS
# =============================================================================

def compute_time_in_range_metrics(glucose_values):
    """
    Compute TIR / TBR / TAR metrics (fractions 0–1).
    """
    n_total = len(glucose_values)

    tir_70_180 = ((glucose_values >= THRESHOLD_HYPO) &
                  (glucose_values <= THRESHOLD_HYPER)).sum() / n_total

    tir_70_140 = ((glucose_values >= THRESHOLD_HYPO) &
                  (glucose_values <= THRESHOLD_TIR_TIGHT_UPPER)).sum() / n_total

    tir_60_140 = ((glucose_values >= THRESHOLD_TIR_LOWER_RELAXED) &
                  (glucose_values <= THRESHOLD_TIR_TIGHT_UPPER)).sum() / n_total

    tbr_70 = (glucose_values < THRESHOLD_HYPO).sum() / n_total
    tbr_54 = (glucose_values < THRESHOLD_TBR_LEVEL2).sum() / n_total

    tar_180 = (glucose_values > THRESHOLD_HYPER).sum() / n_total
    tar_250 = (glucose_values > THRESHOLD_TAR_LEVEL2).sum() / n_total

    return {
        METRIC_TIR70180: round(tir_70_180, ROUNDING_DECIMALS_FRACTIONS),
        METRIC_TIR70140: round(tir_70_140, ROUNDING_DECIMALS_FRACTIONS),
        METRIC_TIR60140: round(tir_60_140, ROUNDING_DECIMALS_FRACTIONS),
        METRIC_TBR70: round(tbr_70, ROUNDING_DECIMALS_FRACTIONS),
        METRIC_TBR54: round(tbr_54, ROUNDING_DECIMALS_FRACTIONS),
        METRIC_TAR180: round(tar_180, ROUNDING_DECIMALS_FRACTIONS),
        METRIC_TAR250: round(tar_250, ROUNDING_DECIMALS_FRACTIONS),
    }