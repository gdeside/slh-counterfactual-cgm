"""
scripts/preprocessing/cgm_preprocessing.py
==========================================

CGM Data Preprocessing — Optimal 45-Day Window Extraction (5-min only)
----------------------------------------------------------------------
End-to-end preprocessing pipeline for CGM datasets.

For each patient (A001–A245):
    1. Load raw data
    2. Detect dominant sampling interval → KEEP ONLY 5-min patients
    3. Attach quality flags (time errors, gaps, glucose outliers)
    4. Extract optimal 45-day window (≥ required coverage)
    5. Save processed window to Excel

Outputs
-------
Data/Preprocessed/
├── A001/
│   └── A001_win1_<start>_to_<end>_covXXpct.xlsx
├── ...
├── windows_summary.csv
├── patients_status.csv
├── exclusion_summary.txt

This script is part of the counterfactual CGM-SLH project 
Author
------
Guillaume Deside

Date
----
2026-04-15

Usage
-----
    python3 -m scripts.preprocessing.cgm_preprocessing [--help for options]
    (or: make preprocessing)
"""

# =============================================================================
# LIBRARIES
# =============================================================================
from pathlib import Path
from typing import Tuple, Dict, Optional

import pandas as pd
from tqdm import tqdm

# =============================================================================
# PROJECT IMPORTS
# =============================================================================
from src.config import (
    # Paths
    RAW_PATH,
    PREPROCESSING_PATH,
    # Patient range
    PATIENT_START,
    PATIENT_END,
    # Preprocessing parameters
    WINDOW_DAYS,
    MIN_COVERAGE_PCT,
    REQUIRED_SAMPLING_MIN,
    GLUCOSE_MIN,
    GLUCOSE_MAX,
    TIME_ERROR_RATIO,
    GAP_MULTIPLE,
)

# Core preprocessing modules
from src.preprocessing.load_data import load_patient_raw_data
from src.preprocessing.timestep import detect_dominant_sampling_interval
from src.preprocessing.quality import attach_quality_flags
from src.preprocessing.window import (
    find_optimal_window,
    save_patient_window,
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def print_configuration() -> None:
    """Print pipeline configuration for reproducibility."""
    print("=" * 70)
    print("  CGM PREPROCESSING — CONFIGURATION")
    print("=" * 70)
    print(f"  Raw data directory : {RAW_PATH}")
    print(f"  Output directory   : {PREPROCESSING_PATH}")
    print(f"  Window length      : {WINDOW_DAYS} days")
    print(f"  Minimum coverage   : {MIN_COVERAGE_PCT} %")
    print(f"  Required sampling  : {REQUIRED_SAMPLING_MIN:.0f} min ONLY")
    print(f"  Valid glucose range: [{GLUCOSE_MIN}, {GLUCOSE_MAX}] mg/dL")
    print(f"  Time-error ratio   : step < {TIME_ERROR_RATIO} × dominant")
    print(f"  Gap threshold      : step > {GAP_MULTIPLE} × dominant")
    print(f"  Patients analysed  : A{PATIENT_START:03d}–A{PATIENT_END:03d}")
    print("=" * 70)


# =============================================================================
# CORE PIPELINE
# =============================================================================
def process_single_patient(patient_id: str) -> Tuple[Optional[Dict], str]:
    """
    Run full preprocessing pipeline for one patient.

    Returns
    -------
    (metadata, reason)
        metadata : dict or None
        reason   : exclusion reason (empty string if successful)
    """
    # Step 1 — Load raw data
    df_raw = load_patient_raw_data(patient_id, RAW_PATH)

    if df_raw is None or len(df_raw) < 100:
        return None, "missing_or_too_short"

    # Step 2 — Sampling interval filtering (keep only 5-min)
    sampling_interval = detect_dominant_sampling_interval(df_raw)

    if sampling_interval != REQUIRED_SAMPLING_MIN:
        return None, f"non_5min_interval_{int(sampling_interval)}min"

    # Step 3 — Attach quality flags
    df_flagged = attach_quality_flags(df_raw, sampling_interval)

    # Step 4 — Optimal window extraction
    window_result = find_optimal_window(df_flagged, sampling_interval)

    if window_result is None:
        return None, "insufficient_coverage"

    # Step 5 — Save processed window
    metadata = save_patient_window(
        patient_id,
        window_result,
        sampling_interval,
        PREPROCESSING_PATH,
    )

    return metadata, ""


# =============================================================================
# REPORTING
# =============================================================================
def summarize_exclusions(exclusion_reasons: Dict[str, str]) -> Dict[str, int]:
    """
    Aggregate per-patient exclusion reasons into the three pipeline-stage
    buckets: missing/too-short raw data, non-5min sampling interval, and
    insufficient window coverage (no valid 45-day window found).
    """
    exclusion_counts: Dict[str, int] = {}
    for reason in exclusion_reasons.values():
        exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1

    return {
        "missing_or_too_short": exclusion_counts.get("missing_or_too_short", 0),
        "non_5min_sampling": sum(
            c for r, c in exclusion_counts.items()
            if r.startswith("non_5min_interval_")
        ),
        "insufficient_coverage": exclusion_counts.get("insufficient_coverage", 0),
    }


def save_exclusion_summary(
    exclusion_reasons: Dict[str, str],
    output_path: Path | str,
    n_total: int,
    n_included: int,
) -> None:
    """Save detailed exclusion statistics to a text report."""
    exclusion_counts: Dict[str, int] = {}
    for reason in exclusion_reasons.values():
        exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1

    stage_counts = summarize_exclusions(exclusion_reasons)
    missing = stage_counts["missing_or_too_short"]
    insufficient = stage_counts["insufficient_coverage"]

    non_5min = {
        r.replace("non_5min_interval_", ""): c
        for r, c in exclusion_counts.items()
        if r.startswith("non_5min_interval_")
    }
    total_non_5min = sum(non_5min.values())

    summary_path = Path(output_path) / "exclusion_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("CGM PREPROCESSING — EXCLUSION SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total patients analysed : {n_total}\n")
        f.write(f"Included with valid window: {n_included} "
                f"({100 * n_included / n_total:.1f}%)\n")
        f.write(f"Excluded                : {n_total - n_included}\n\n")

        f.write("BREAKDOWN OF EXCLUSIONS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Missing or too short data : {missing}\n")
        f.write(f"Non-5 min sampling interval: {total_non_5min}\n")

        for interval in sorted(non_5min, key=lambda x: int(x.replace("min", ""))):
            f.write(f"   {interval} min : {non_5min[interval]}\n")

        f.write(f"\nInsufficient coverage (< {MIN_COVERAGE_PCT}%) : {insufficient}\n")

    print(f"exclusion_summary.txt: {summary_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main() -> None:
    """Run the full CGM preprocessing pipeline."""
    print_configuration()

    # Ensure output directory exists
    PREPROCESSING_PATH.mkdir(parents=True, exist_ok=True)

    patient_ids = [
        f"A{n:03d}" for n in range(PATIENT_START, PATIENT_END + 1)
    ]

    windows_metadata = []
    exclusion_reasons: Dict[str, str] = {}

    print(f"\nProcessing {len(patient_ids)} patients...\n")

    for pid in tqdm(patient_ids, desc="Patients", ncols=80):
        metadata, reason = process_single_patient(pid)

        if metadata:
            windows_metadata.append(metadata)
        else:
            exclusion_reasons[pid] = reason

    print("\n" + "=" * 70)

    # Save windows summary
    if windows_metadata:
        df_summary = pd.DataFrame(windows_metadata)
        cols = [
            "patient_id", "win_start", "win_end", "coverage_pct",
            "flag_rate_pct", "n_valid", "n_total",
            "dominant_step_min", "filename", "filepath"
        ]
        df_summary = df_summary[cols]

        summary_path = PREPROCESSING_PATH / "windows_summary.csv"
        df_summary.to_csv(summary_path, index=False)
        print(f"windows_summary.csv → {summary_path}")
    else:
        print(" No valid windows extracted.")

    # Save patient status
    rows = [
        {"patient_id": pid, "has_window": False, "exclusion_reason": reason}
        for pid, reason in sorted(exclusion_reasons.items())
    ]
    rows += [
        {"patient_id": m["patient_id"], "has_window": True, "exclusion_reason": ""}
        for m in windows_metadata
    ]

    status_df = pd.DataFrame(rows)
    status_path = PREPROCESSING_PATH / "patients_status.csv"
    status_df.to_csv(status_path, index=False)
    print(f"patients_status.csv → {status_path}")

    # Save exclusion report
    save_exclusion_summary(
        exclusion_reasons,
        PREPROCESSING_PATH,
        len(patient_ids),
        len(windows_metadata),
    )

    # Final summary
    exclusion_counts = summarize_exclusions(exclusion_reasons)
    print("=" * 70)
    print(f"  Total patients   : {len(patient_ids)}")
    print(f"  Included         : {len(windows_metadata)}")
    print(f"  Excluded         : {len(exclusion_reasons)}")
    print(f"    Missing/too short data      : {exclusion_counts['missing_or_too_short']}")
    print(f"    Non-5min sampling interval  : {exclusion_counts['non_5min_sampling']}")
    print(f"    No valid window (coverage)  : {exclusion_counts['insufficient_coverage']}")
    print(f"  Output directory : {PREPROCESSING_PATH}")
    print("=" * 70)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()