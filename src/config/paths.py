"""
src/config/paths.py
===================

Project-specific paths and runtime configuration.

These are separated from scientific constants because they depend on the local machine
and data location.
"""

from pathlib import Path

# =============================================================================
# BASE PATH — EDIT THIS FOR YOUR MACHINE
# =============================================================================
import os

# Override with the CGM_DATA_ROOT environment variable (e.g. the folder produced by
# `make sample_data`); falls back to the default below.
BASE_PATH = Path(os.environ.get(
    "CGM_DATA_ROOT",
    "/Users/guillaume/Documents/Projects_Lysy/counterfactual-cgm-slh_echec",
))
#BASE_PATH = Path("/Users/guillaume/Library/CloudStorage/OneDrive-UCL/Shared - Zain - Philippe - Laura - Guillaume")

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
# CLINICAL COHORT FILE (shared OneDrive, not pipeline-generated)
# =============================================================================
FINAL_COHORT_CSV = DATA_PATH / "Final cohort SLH.csv"

# =============================================================================
# CLINICAL REGRESSION RESULTS (written to OneDrive, under BASE_PATH/Results)
# =============================================================================
POISSON_REGRESSION_PATH = RESULTS_PATH / "Poisson_Regression_SLH_Predictors"
NB_REGRESSION_PATH      = RESULTS_PATH / "NB_Regression_SLH_Predictors"
LINEAR_REGRESSION_PATH  = RESULTS_PATH / "Linear_Regression_SLH_Predictors"

# =============================================================================
# OTHER RUN-TIME PARAMETERS
# =============================================================================
PATIENT_START = 1
PATIENT_END   = 245


PATIENTS_STATUS_CSV = PREPROCESSING_PATH / "patients_status.csv"
METRICS_CSV = METRICS_PATH / "metrics_all_patients.csv"
FILTER_STATUS_CSV = FILTER_DIR_PATH / "median_glucose_110to170_patient_status.csv"
WINDOW_SUMMARY_CSV = PREPROCESSING_PATH /  "windows_summary.csv"
EPISODES_SUMMARY_CSV = EPISODES_PATH / "episodes_summary.csv"


RANDOM_SEED = 42

