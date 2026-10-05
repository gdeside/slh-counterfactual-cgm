"""
src/preprocessing
=================

CGM data preprocessing pipeline for the counterfactual CGM-SLH project.

The pipeline runs in four stages:

1. **Loading** — read raw or preprocessed patient files from disk.
2. **Timestep detection** — infer the dominant sampling interval.
3. **Quality flagging** — annotate rows with time-error, gap, and
   glucose-outlier flags without dropping any data.
4. **Window extraction** — find and save the optimal fixed-duration window
   maximising valid data coverage.

Modules
-------
coverage
    Expected-measurement and coverage-percentage computations.
load_data
    Raw and preprocessed CGM data loaders; patient discovery.
quality
    Row-level quality flagging and valid-data filtering.
timestep
    Dominant sampling-interval detection and canonical-value snapping.
window
    Sliding-window search and window-file saving.

Public API
----------
Commonly used symbols are re-exported here for convenient
``from src.preprocessing import X`` imports.

Author
------
Guillaume Deside
"""

# Coverage utilities
from src.preprocessing.coverage import (
    compute_expected_measurements,
    compute_coverage_percentage,
)

# Data loaders
from src.preprocessing.load_data import (
    load_raw_time_column,
    load_patient_raw_data,
    load_window_data,
    load_patient_window,
    prepare_valid_data,
    discover_patients,
)

# Quality flagging
from src.preprocessing.quality import (
    attach_quality_flags,
    prepare_valid_data as prepare_valid_data_quality,
)

# Sampling interval utilities
from src.preprocessing.timestep import (
    snap_to_canonical,
    dominant_step_of_series,
    detect_dominant_sampling_interval,
)

# Window extraction
from src.preprocessing.window import (
    find_optimal_window,
    save_patient_window,
)

__all__ = [
    # coverage
    "compute_expected_measurements",
    "compute_coverage_percentage",
    # load_data
    "load_raw_time_column",
    "load_patient_raw_data",
    "load_window_data",
    "load_patient_window",
    "prepare_valid_data",
    "discover_patients",
    # quality
    "attach_quality_flags",
    "prepare_valid_data_quality",
    # timestep
    "snap_to_canonical",
    "dominant_step_of_series",
    "detect_dominant_sampling_interval",
    # window
    "find_optimal_window",
    "save_patient_window",
]
