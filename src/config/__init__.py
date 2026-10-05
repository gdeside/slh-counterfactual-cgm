"""
src/config/__init__.py
======================

Central configuration hub for the counterfactual CGM-SLH project.

This file exposes all scientific constants, column names, clinical thresholds,
and figure settings so that scripts and modules can import them cleanly:

    from src.config import (
        PATIENT_START, PATIENT_END,
        THRESHOLD_HYPO, THRESHOLD_HYPER,
        FIG_PREPROCESSING_DIR, DPI_SAVE,
        ...
    )

"""

# =============================================================================
# Patient cohort & run parameters
# =============================================================================
from .constants import (
    # Column names
    PATIENT_ID_COL,
    TIME_COL,
    GLUCOSE_COL,
    
    # Quality flags
    COL_FLAG_TIME_ERROR,
    COL_FLAG_OUTLIER_GLUCOSE,

    COL_COVERAGE,
    COL_SAMPLING,

    COL_HAS_WINDOW,
    COL_COVERAGE_PCT,
    COL_DOMINANT_STEP,
    COL_WIN_START,
    COL_WIN_END ,
    COL_FILEPATH,

    COL_EPISODE_START,
    COL_EPISODE_END,
    COL_EPISODE_DURATION,
    COL_EPISODE_CLASSIFICATION,
    CLASSIFICATION_SLH,
    CLASSIFICATION_LLH,
    
    # Glucose unit conversion
    MGDL_PER_MMOLL,

    # Glucose & clinical thresholds
    THRESHOLD_HYPO,
    THRESHOLD_HYPER,
    GLUCOSE_MIN,
    GLUCOSE_MAX,
    
    # Preprocessing parameters
    WINDOW_DAYS,
    MIN_COVERAGE_PCT,
    SLIDE_FREQ,
    REQUIRED_SAMPLING_MIN,
    TIME_ERROR_RATIO,
    GAP_MULTIPLE,
    
    # Night definition
    NIGHT_START_H,
    NIGHT_END_H,
)

# =============================================================================
# Figure & visualization configuration
# =============================================================================
from .figure_config import (
    FIG_DIR,
    FIG_PREPROCESSING_DIR,
    FIG_FILTERING_DIR,
    FIG_METRICS_DIR,
    DPI_SAVE,
    FIGURE_WIDTH_INCHES,
    FIGURE_WIDTH_SINGLE_INCHES,
    FIGURE_HEIGHT_INCHES,
    FIGURE_HEIGHT_RATIO,
    LANCET_WIDTH_1COL_INCHES,
    LANCET_WIDTH_2COL_INCHES,
    LANCET_WIDTH_FULL_INCHES,
    LANCET_MAX_HEIGHT_INCHES,
    LANCET_MIN_FONT_PT,
    LANCET_MIN_LINEWIDTH_PT,
    LANCET_MIN_DPI,
    lancet_figsize,
)

from .paths import (
    BASE_PATH,
    DATA_PATH,
    RAW_PATH,
    PREPROCESSING_PATH,
    FILTER_DIR_PATH,
    METRICS_PATH,
    EPISODES_PATH,
    SENSITIVITY_PATH,
    SENSITIVITY_LININTERP_SIGNALS,
    SENSITIVITY_LININTERP_DATA,
    SENSITIVITY_LININTERP_FIGURES,
    COHORT_SENSITIVITY_CSV,
    QUARTILE_ANALYSIS_PATH,
    QUARTILE_FIGURES_PATH,
    QUARTILE_STATS_PATH,
    QUARTILE_ASSIGNMENTS_CSV,
    SCATTER_ANALYSIS_PATH,
    SCATTER_FIGURES_PATH,
    SCATTER_STATS_CSV,
    PATIENT_START,
    PATIENT_END,
    WINDOW_SUMMARY_CSV,
    FILTER_STATUS_CSV,
    RANDOM_SEED,
    METRICS_CSV,
    PATIENTS_STATUS_CSV,
    EPISODES_SUMMARY_CSV,
)

# =============================================================================
# Utility functions
# =============================================================================
from src.utils.time_utils import is_night
from src.utils.unit_conversion import (
    # Value conversion
    mgdl_to_mmoll,
    mmoll_to_mgdl,
    convert_glucose_column,
    convert_dataframe_units,
    convert_metric_table,
    is_glucose_valued_column,
    normalize_unit,
    # Unit names
    UNIT_MGDL,
    UNIT_MMOLL,
    CANONICAL_UNIT,
    # Display unit (presentation only — computation stays in mg/dL)
    get_display_unit,
    set_display_unit,
    display_unit,
    to_display,
    from_display,
    glucose_label,
    glucose_unit_label,
    format_glucose,
    glucose_ticks,
)


# Make all imported names available at package level
__all__ = [
    # Paths
    "BASE_PATH", "DATA_PATH", "RAW_PATH", "PREPROCESSING_PATH", "FILTER_DIR_PATH", "METRICS_PATH", "EPISODES_PATH", "SENSITIVITY_PATH",
    "SENSITIVITY_LININTERP_PATH", "SENSITIVITY_LININTERP_SIGNALS", "SENSITIVITY_LININTERP_DATA", "SENSITIVITY_LININTERP_FIGURES",
    "COHORT_SENSITIVITY_CSV",
    "QUARTILE_ANALYSIS_PATH", "QUARTILE_FIGURES_PATH", "QUARTILE_STATS_PATH", "QUARTILE_ASSIGNMENTS_CSV",
    "SCATTER_ANALYSIS_PATH", "SCATTER_FIGURES_PATH", "SCATTER_STATS_CSV",
    "FILTER_STATUS_CSV", "RANDOM_SEED","WINDOW_SUMMARY_CSV", "METRICS_CSV", "PATIENTS_STATUS_CSV", "EPISODES_SUMMARY_CSV",

    # Column names
    "PATIENT_ID_COL", "TIME_COL", "GLUCOSE_COL",
    
    # Quality flags
    "COL_FLAG_TIME_ERROR", "COL_FLAG_OUTLIER_GLUCOSE",

    "COL_COVERAGE","COL_SAMPLING",

    "COL_HAS_WINDOW", "COL_COVERAGE_PCT", "COL_DOMINANT_STEP", "COL_WIN_START", "COL_WIN_END",
    "COL_FILEPATH",

    "COL_EPISODE_START", "COL_EPISODE_END", "COL_EPISODE_DURATION", "COL_EPISODE_CLASSIFICATION",
    "CLASSIFICATION_SLH", "CLASSIFICATION_LLH",
    
    # Glucose unit conversion
    "MGDL_PER_MMOLL",

    # Thresholds
    "THRESHOLD_HYPO", "THRESHOLD_HYPER",
    "GLUCOSE_MIN", "GLUCOSE_MAX",
    
    # Preprocessing
    "WINDOW_DAYS", "MIN_COVERAGE_PCT", "SLIDE_FREQ",
    "REQUIRED_SAMPLING_MIN", "TIME_ERROR_RATIO", "GAP_MULTIPLE",
    
    # Night
    "NIGHT_START_H", "NIGHT_END_H",
    
    # Figure settings
    "FIG_DIR", "FIG_PREPROCESSING_DIR", "FIG_FILTERING_DIR", "FIG_METRICS_DIR",
    "DPI_SAVE", "FIGURE_WIDTH_INCHES", "FIGURE_WIDTH_SINGLE_INCHES", "FIGURE_HEIGHT_INCHES", "FIGURE_HEIGHT_RATIO",

    # Lancet artwork specification
    "LANCET_WIDTH_1COL_INCHES", "LANCET_WIDTH_2COL_INCHES", "LANCET_WIDTH_FULL_INCHES",
    "LANCET_MAX_HEIGHT_INCHES", "LANCET_MIN_FONT_PT", "LANCET_MIN_LINEWIDTH_PT",
    "LANCET_MIN_DPI", "lancet_figsize",
    
    # Patient range
    "PATIENT_START", "PATIENT_END",
    
    # Utilities
    "is_night",

    # Unit conversion & display unit
    "mgdl_to_mmoll", "mmoll_to_mgdl", "convert_glucose_column",
    "convert_dataframe_units", "convert_metric_table",
    "is_glucose_valued_column", "normalize_unit",
    "UNIT_MGDL", "UNIT_MMOLL", "CANONICAL_UNIT",
    "get_display_unit", "set_display_unit", "display_unit",
    "to_display", "from_display",
    "glucose_label", "glucose_unit_label", "format_glucose", "glucose_ticks",
]