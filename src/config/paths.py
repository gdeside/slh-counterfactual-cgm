"""
src/config/paths.py
===================

Project-specific paths and runtime configuration.

These are separated from scientific constants because they depend on the local machine
and data location.
"""

from pathlib import Path

# =============================================================================
# BASE PATH — set the CGM_DATA_ROOT environment variable
# =============================================================================
import os

# CGM_DATA_ROOT is the folder holding Data/ and Results/ (e.g. the folder produced by
# `make sample_data`). Without it, ./data (relative to the working directory) is used.
_root = os.environ.get("CGM_DATA_ROOT", "data")
BASE_PATH = Path(_root)

# =============================================================================
# DATA DIRECTORIES
# =============================================================================

DATA_PATH          = BASE_PATH / "Data"
RAW_PATH           = DATA_PATH / "Raw"
PREPROCESSING_PATH = DATA_PATH / "Preprocessed"
FILTER_DIR_PATH    = DATA_PATH / "Filtered"
METRICS_PATH       = DATA_PATH / "Metrics"
EPISODES_PATH      = DATA_PATH / "Episodes"
RESULTS_PATH       = BASE_PATH / "Results"
SENSITIVITY_PATH     = RESULTS_PATH / "Sensitivity_Analysis"

SENSITIVITY_LININTERP_SIGNALS = SENSITIVITY_PATH / "Signals"
SENSITIVITY_LININTERP_DATA    = SENSITIVITY_PATH / "Data"
SENSITIVITY_LININTERP_FIGURES = SENSITIVITY_PATH / "Figures"
COHORT_SENSITIVITY_CSV        = SENSITIVITY_LININTERP_DATA / "cohort_sensitivity.csv"

QUARTILE_ANALYSIS_PATH   = RESULTS_PATH / "Quartile_Analysis"
QUARTILE_FIGURES_PATH    = QUARTILE_ANALYSIS_PATH / "Figures"
QUARTILE_STATS_PATH      = QUARTILE_ANALYSIS_PATH / "Statistics"
QUARTILE_ASSIGNMENTS_CSV = QUARTILE_ANALYSIS_PATH / "quartile_assignments.csv"

SCATTER_ANALYSIS_PATH = RESULTS_PATH / "Scatter_Analysis"
SCATTER_FIGURES_PATH  = SCATTER_ANALYSIS_PATH / "Figures"
SCATTER_STATS_CSV     = SCATTER_ANALYSIS_PATH / "scatter_statistics.csv"

# =============================================================================
# CLINICAL TABLE (provided by the user in CGM_DATA_ROOT, not pipeline-generated)
# =============================================================================
FINAL_COHORT_CSV = DATA_PATH / "Final cohort SLH.csv"

PATIENTS_STATUS_CSV = PREPROCESSING_PATH / "patients_status.csv"
METRICS_CSV = METRICS_PATH / "metrics_all_patients.csv"
FILTER_STATUS_CSV = FILTER_DIR_PATH / "median_glucose_110to170_patient_status.csv"
WINDOW_SUMMARY_CSV = PREPROCESSING_PATH /  "windows_summary.csv"
EPISODES_SUMMARY_CSV = EPISODES_PATH / "episodes_summary.csv"


RANDOM_SEED = 42

