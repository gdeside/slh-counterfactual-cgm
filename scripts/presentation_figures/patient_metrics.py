"""
scripts/presentation_figures/patient_metrics.py
===============================================

CGM metrics of one patient, written to a CSV (one row per metric).

Metrics are the project's standard set from ``src.metrics.metrics``, computed
on the patient's whole preprocessed record (``load_patient_window``):

  · Glucose statistics   mean, median, min, max, SD            (glucose unit)
                         CV                                    (fraction)
  · Time in range        TIR 70–180 / 70–140 / 60–140          (fractions)
  · Time below range     TBR < 70 / < 54                       (fractions)
  · Time above range     TAR > 180 / > 250                     (fractions)
  · Time of day          median glucose: morning, noon, afternoon, evening, night
  · CGM_days_measured    number of calendar days with data
  · GMI                  3.31 + 0.02392 × mean glucose (mg/dL)  (%)

Glucose-valued rows are in mg/dL by default; --unit mmol/L converts them
(fractions, CV, GMI and day counts are unit-free and unchanged).

Usage
-----
    python3 -m scripts.presentation_figures.patient_metrics [--patient_id ID]
                                                            [--unit mg/dL|mmol/L]
                                                            [--out_dir DIR]

Output
------
    scripts/presentation_figures/output/patient_metrics/<patient_id>_metrics_<mgdl|mmoll>.csv
    columns: patient_id, metric, value, unit
    (git-ignored; --out_dir to change)
"""

import os
import sys
import argparse

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.patient_selection import resolve_patient_id
from src.preprocessing.load_data import load_patient_window
from src.config import GLUCOSE_COL
from src.metrics.metrics import (
    MIN_ROWS_FOR_METRICS,
    compute_basic_statistics,
    compute_time_in_range_metrics,
    compute_time_based_metrics,
    compute_period_medians,
    METRIC_MEAN_GLUCOSE,
    METRIC_CV,
)
from src.utils.unit_conversion import (
    is_glucose_valued_column,
    mgdl_to_mmoll,
    normalize_unit,
    unit_tag,
    UNIT_MGDL,
    UNIT_MMOLL,
)

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "patient_metrics")

METRIC_GMI = "GMI"
GMI_INTERCEPT, GMI_SLOPE = 3.31, 0.02392   # GMI (%) from mean glucose in mg/dL


# =============================================================================
# COMPUTATION
# =============================================================================

def compute_patient_metrics(df: pd.DataFrame) -> dict:
    """All metrics of one patient, in mg/dL (fractions for time-in-range)."""
    glucose = df[GLUCOSE_COL].dropna()
    if len(glucose) < MIN_ROWS_FOR_METRICS:
        raise ValueError(f"only {len(glucose)} readings; "
                         f"need at least {MIN_ROWS_FOR_METRICS}")

    metrics = {}
    metrics.update(compute_basic_statistics(glucose))
    metrics.update(compute_time_in_range_metrics(glucose))
    metrics.update(compute_time_based_metrics(df))
    metrics.update(compute_period_medians(df))
    metrics[METRIC_GMI] = round(
        GMI_INTERCEPT + GMI_SLOPE * metrics[METRIC_MEAN_GLUCOSE], 2)
    return metrics


def metrics_table(patient_id: str, metrics: dict, unit: str) -> pd.DataFrame:
    """Long-format table (patient_id, metric, value, unit) in the chosen unit."""
    rows = []
    for name, value in metrics.items():
        if is_glucose_valued_column(name):
            if unit == UNIT_MMOLL:
                value = round(float(mgdl_to_mmoll(value)), 2)
            row_unit = unit
        elif name == METRIC_GMI:
            row_unit = "%"
        elif name == "CGM_days_measured":
            row_unit = "days"
        else:
            row_unit = "fraction"      # CV and time-in-range metrics (0–1)
        rows.append({"patient_id": patient_id, "metric": name,
                     "value": value, "unit": row_unit})
    return pd.DataFrame(rows)


# =============================================================================
# CLI
# =============================================================================

def parse_arguments():
    p = argparse.ArgumentParser(
        description="CGM metrics of one patient -> CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--patient_id", default=None,
                   help="Patient ID (default: chosen automatically — most SLH episodes)")
    p.add_argument("--unit", default=UNIT_MGDL, choices=[UNIT_MGDL, UNIT_MMOLL],
                   type=normalize_unit, help="Unit of the glucose-valued metrics")
    p.add_argument("--out_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    return p


def main():
    args = parse_arguments().parse_args()
    patient_id = resolve_patient_id(args.patient_id)

    df = load_patient_window(patient_id)
    if df is None:
        print(f"[!] Could not load data for {patient_id}")
        sys.exit(1)

    table = metrics_table(patient_id, compute_patient_metrics(df), args.unit)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir,
                       f"{patient_id}_metrics_{unit_tag(args.unit)}.csv")
    table.to_csv(out, index=False)
    print(f"  Patient : {patient_id}  ({len(df):,} readings)")
    print(f"  Saved   → {out}")


if __name__ == "__main__":
    main()
