"""
src/counterfactual/__init__.py
==============================

Public API for the counterfactual CGM-SLH module.

Exposes configuration constants, signal-building functions, and data-loading
helpers so that callers can import everything they need from a single location:

    from src.counterfactual import (
        APPLY_KERNEL, KERNEL_MIN, SELECTED_OFFSET_MIN, OFFSET_VALUES,
        build_counterfactual,
        build_full_counterfactual_signal,
        load_patient_signals,
        load_sensitivity_results,
        load_cohort_sensitivity_results,
    )
"""

from src.counterfactual.counterfactual_config import (
    APPLY_KERNEL,
    KERNEL_MIN,
    SELECTED_OFFSET_MIN,
    OFFSET_VALUES,
)

from src.counterfactual.build_counterfactual import (
    find_local_minimum,
    build_linear_interpolation,
    apply_rolling_mean_smoothing,
    build_counterfactual,
    build_full_counterfactual_signal,
)

from src.counterfactual.load_counterfactual import (
    load_patient_signals,
    load_sensitivity_results,
    load_cohort_sensitivity_results,
    extract_delta_values,
)

__all__ = [
    # Configuration
    "APPLY_KERNEL",
    "KERNEL_MIN",
    "SELECTED_OFFSET_MIN",
    "OFFSET_VALUES",
    # Build
    "find_local_minimum",
    "build_linear_interpolation",
    "apply_rolling_mean_smoothing",
    "build_counterfactual",
    "build_full_counterfactual_signal",
    # Load
    "load_patient_signals",
    "load_sensitivity_results",
    "load_cohort_sensitivity_results",
    "extract_delta_values",
]
