"""
scripts/preprocessing/cgm_filtered_metrics_distribution.py
==========================================================

CGM Metrics Distribution — Filtered Cohort
-------------------------------------------
Compute distribution statistics (mean, SD, min, Q1, median, Q3, max) for
each CGM metric, restricted to the patients retained after the median
glucose filter (cgm_filtering.py).

Input
-----
- Data/Metrics/metrics_all_patients.csv
- Data/Filtered/median_glucose_110to170_patient_status.csv

Output
------
- Data/Filtered/filtered_cohort_metrics_distribution.csv

Description
-----------
For each of the 19 standardized CGM metrics:
1. Restrict metrics_all_patients.csv to the filtered/selected patients
2. Compute n, n_missing, mean, SD, min, Q1, median, Q3, max
3. Save one row per metric to a cohort-level CSV

Author
------
Guillaume Deside

Date
----
2026-08-26

Usage
-----
    python3 -m scripts.preprocessing.cgm_filtered_metrics_distribution [--help for options]
    (or: make filtered_metrics_distribution)
"""

# =============================================================================
# STANDARD LIBRARIES
# =============================================================================
import os

# =============================================================================
# THIRD-PARTY LIBRARIES
# =============================================================================
import pandas as pd

# =============================================================================
# PROJECT IMPORTS
# =============================================================================
from src.config import (
    PATIENT_ID_COL,
    METRICS_CSV,
    FILTER_STATUS_CSV,
    FILTER_DIR_PATH,
)

from src.metrics.metrics import ALL_METRIC_NAMES
from src.metrics.load_metrics import load_patient_metrics
from src.filtering.load_filtering import load_filtered_patient_list

# =============================================================================
# CONSTANTS
# =============================================================================
ROUNDING_DECIMALS = 2

DISTRIBUTION_CSV = os.path.join(
    str(FILTER_DIR_PATH), "filtered_cohort_metrics_distribution.csv"
)


# =============================================================================
# DISTRIBUTION COMPUTATION
# =============================================================================
def compute_metric_distribution(df: pd.DataFrame, metric_cols: list) -> pd.DataFrame:
    """
    Compute distribution statistics for each metric column.

    Parameters
    ----------
    df : pd.DataFrame
        Patient-level metrics (one row per patient)
    metric_cols : list of str
        Metric columns to summarize

    Returns
    -------
    pd.DataFrame
        One row per metric, with columns:
        metric, n, n_missing, mean, std, min, q1, median, q3, max
    """
    desc = df[metric_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc = desc.rename(columns={"25%": "q1", "50%": "median", "75%": "q3"})

    dist = pd.DataFrame({
        "metric":    metric_cols,
        "n":         desc["count"].astype(int).values,
        "n_missing": (len(df) - desc["count"]).astype(int).values,
        "mean":      desc["mean"].values,
        "std":       desc["std"].values,
        "min":       desc["min"].values,
        "q1":        desc["q1"].values,
        "median":    desc["median"].values,
        "q3":        desc["q3"].values,
        "max":       desc["max"].values,
    })

    numeric_cols = ["mean", "std", "min", "q1", "median", "q3", "max"]
    dist[numeric_cols] = dist[numeric_cols].round(ROUNDING_DECIMALS)

    return dist


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main() -> None:
    """Run the filtered-cohort metrics distribution pipeline."""
    print("=" * 70)
    print("  CGM METRICS DISTRIBUTION — FILTERED COHORT")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------
    if not os.path.exists(METRICS_CSV):
        raise FileNotFoundError(
            f"metrics_all_patients.csv not found:\n  {METRICS_CSV}\n"
            "Run cgm_metrics.py first."
        )

    if not os.path.exists(FILTER_STATUS_CSV):
        raise FileNotFoundError(
            f"Filter status CSV not found:\n  {FILTER_STATUS_CSV}\n"
            "Run cgm_filtering.py first."
        )

    os.makedirs(FILTER_DIR_PATH, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load metrics and restrict to the filtered/selected patients
    # -------------------------------------------------------------------------
    df_all = load_patient_metrics(METRICS_CSV)

    filtered_pids = load_filtered_patient_list(str(FILTER_STATUS_CSV))
    df_filtered = df_all[df_all[PATIENT_ID_COL].isin(filtered_pids)].copy()

    print(f"\n  Patients with valid metrics: {len(df_all)}")
    print(f"  Patients after filtering:    {len(df_filtered)}\n")

    # -------------------------------------------------------------------------
    # Compute and save distribution
    # -------------------------------------------------------------------------
    dist_df = compute_metric_distribution(df_filtered, ALL_METRIC_NAMES)
    dist_df.to_csv(DISTRIBUTION_CSV, index=False)

    # -------------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("  METRICS DISTRIBUTION COMPLETE")
    print("=" * 70)
    print(f"  Metrics summarized: {len(dist_df)}")
    print(f"  Output: {DISTRIBUTION_CSV}")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
