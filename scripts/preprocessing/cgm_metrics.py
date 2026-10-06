"""
scripts/preprocessing/cgm_metrics.py
====================================

CGM Metrics Computation
----------------------
Compute standardized CGM metrics from preprocessed 45-day windows.

Input
-----
- Data/Preprocessed/windows_summary.csv
- Data/Preprocessed/patients_status.csv
- Data/Preprocessed/{PID}/{PID}_win1_*.xlsx

Output
------
- Data/Metrics/metrics_all_patients.csv

Description
-----------
For each patient:
1. Load preprocessed CGM window
2. Filter invalid measurements (flags + NaNs)
3. Compute 19 standardized CGM metrics
4. Aggregate results into a cohort-level CSV

Author
------
Guillaume Deside

Date
----
2026-03-28

Usage
-----
    python3 -m scripts.preprocessing.cgm_metrics [--help for options]
    (or: make metrics_preprocessing)
"""

# =============================================================================
# STANDARD LIBRARIES
# =============================================================================
import os
from typing import Dict, List

# =============================================================================
# THIRD-PARTY LIBRARIES
# =============================================================================
import pandas as pd
from tqdm import tqdm

# =============================================================================
# PROJECT IMPORTS
# =============================================================================
from src.config import (
    GLUCOSE_COL,
    PATIENT_ID_COL,
    COL_HAS_WINDOW,
    COL_COVERAGE_PCT,
    COL_DOMINANT_STEP,
    COL_WIN_START,
    COL_WIN_END,
    COL_FILEPATH,
    WINDOW_SUMMARY_CSV,
    METRICS_CSV,
    PATIENTS_STATUS_CSV,
    METRICS_PATH,
)

from src.preprocessing.load_data import load_window_data
from src.preprocessing.quality import prepare_valid_data

from src.metrics.metrics import (
    compute_basic_statistics,
    compute_time_in_range_metrics,
    compute_time_based_metrics,
    compute_period_medians,
    create_empty_metrics,
    ALL_METRIC_NAMES,
    MIN_ROWS_FOR_METRICS,
    METRIC_MEAN_GLUCOSE,
    METRIC_TIR70180,
    METRIC_CV,
)


from src.visualization.metrics_plots import view1_control_overview, view2_hypo_hyper_risk, view3_diurnal_pattern, view4_variability, view5_correlation_heatmap, plot_distribution_metrics

# =============================================================================
# METRICS COMPUTATION
# =============================================================================
def compute_all_metrics(filepath: str) -> Dict:
    """
    Compute CGM metrics for a single patient window.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    dict
        Dictionary of metrics (NaN-filled if failure)
    """
    df = load_window_data(filepath)

    if df is None:
        return create_empty_metrics()

    df = prepare_valid_data(df)

    if len(df) < MIN_ROWS_FOR_METRICS:
        return create_empty_metrics()

    try:
        glucose = df[GLUCOSE_COL]

        metrics = {}
        metrics.update(compute_basic_statistics(glucose))
        metrics.update(compute_time_in_range_metrics(glucose))
        metrics.update(compute_time_based_metrics(df))
        metrics.update(compute_period_medians(df))

        return metrics

    except Exception:
        return create_empty_metrics()


# =============================================================================
# OUTPUT UTILITIES
# =============================================================================
def create_output_dataframe(rows: List[Dict]) -> pd.DataFrame:
    """
    Create final structured output DataFrame.

    Parameters
    ----------
    rows : list of dict

    Returns
    -------
    pd.DataFrame
    """
    column_order = [
        PATIENT_ID_COL,
        COL_HAS_WINDOW,
        COL_COVERAGE_PCT,
        COL_DOMINANT_STEP,
        COL_WIN_START,
        COL_WIN_END,
    ] + ALL_METRIC_NAMES

    return pd.DataFrame(rows)[column_order]


def print_cohort_summary(df: pd.DataFrame) -> None:
    """Print summary statistics for cohort."""
    valid_df = df[df[COL_HAS_WINDOW]]

    if valid_df.empty:
        return

    print("\n  Cohort summary:")

    mean_glucose = valid_df[METRIC_MEAN_GLUCOSE]
    print(f"    Mean glucose: {mean_glucose.mean():.1f} ± {mean_glucose.std():.1f}")

    print(f"    Median TIR70-180: {(valid_df[METRIC_TIR70180].median()*100):.1f}%")
    print(f"    Median CV: {(valid_df[METRIC_CV].median()*100):.1f}%")


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main() -> None:
    """Run CGM metrics computation pipeline."""
    print("=" * 70)
    print("  CGM METRICS COMPUTATION")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------
    if not os.path.exists(WINDOW_SUMMARY_CSV):
        raise FileNotFoundError("windows_summary.csv missing")

    if not os.path.exists(PATIENTS_STATUS_CSV):
        raise FileNotFoundError("patients_status.csv missing")

    os.makedirs(METRICS_PATH, exist_ok=True)

    summary_df = pd.read_csv(WINDOW_SUMMARY_CSV)
    status_df = pd.read_csv(PATIENTS_STATUS_CSV)

    # Build lookup dictionary (FASTER than loop)
    window_info = summary_df.set_index(PATIENT_ID_COL).to_dict("index")

    print(f"\n  Patients total: {len(status_df)}")
    print(f"  With windows:   {len(summary_df)}\n")

    rows = []
    errors = []

    # -------------------------------------------------------------------------
    # Process patients
    # -------------------------------------------------------------------------
    for _, row in tqdm(status_df.iterrows(), total=len(status_df), ncols=72):
        pid = row[PATIENT_ID_COL]
        has_window = bool(row[COL_HAS_WINDOW])

        base = {
            PATIENT_ID_COL: pid,
            COL_HAS_WINDOW: has_window,
        }

        # Skip missing windows
        if not has_window or pid not in window_info:
            base.update({
                COL_COVERAGE_PCT: None,
                COL_DOMINANT_STEP: None,
                COL_WIN_START: None,
                COL_WIN_END: None,
            })
            rows.append({**base, **create_empty_metrics()})
            continue

        win = window_info[pid]
        base.update(win)

        filepath = win[COL_FILEPATH]

        if not os.path.exists(filepath):
            errors.append(f"{pid}: missing file")
            rows.append({**base, **create_empty_metrics()})
            continue

        metrics = compute_all_metrics(filepath)
        rows.append({**base, **metrics})

    # -------------------------------------------------------------------------
    # Save output
    # -------------------------------------------------------------------------
    df_out = create_output_dataframe(rows)
    df_out.to_csv(METRICS_CSV, index=False)


    # -------------------------------------------------------------------------
    # Metrics visualization
    # -------------------------------------------------------------------------
    view1_control_overview(df_out)
    view2_hypo_hyper_risk(df_out)
    view3_diurnal_pattern(df_out)
    view4_variability(df_out)
    view5_correlation_heatmap(df_out)
    plot_distribution_metrics(df_out)

    # -------------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------------
    if errors:
        print("\nWarnings:")
        for e in errors:
            print(f"{e}")

    print("\n" + "=" * 70)
    print("  METRICS COMPUTATION COMPLETE")
    print("=" * 70)

    print(f"  Patients with metrics: {df_out[COL_HAS_WINDOW].sum()}")
    print(f"  Output: {METRICS_CSV}")

    print_cohort_summary(df_out)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()