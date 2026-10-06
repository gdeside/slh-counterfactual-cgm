"""
scripts/counterfactual/sensitivity_analysis.py
==============================================
CGM Sensitivity Analysis — Linear Interpolation Counterfactual (MODULAR)
=========================================================================
Uses modular episode detection and counterfactual construction from src/.

Counterfactual method
---------------------
  Step 1  Linear interpolation between local-minima anchors
  Step 2  Optional rolling-mean smoothing (APPLY_KERNEL)

Episode detection
-----------------
  Full C1-C6 criterion pipeline from src.episodes.criteria
  Detects and classifies episodes as SLH or LLH

Sensitivity grid
----------------
  delta (zone-expansion offset): OFFSET_VALUES from counterfactual_config
                                 (currently [0, 15, 30, 45, 60, 75, 90] minutes)
  kernel width: fixed at KERNEL_MIN (60 min)

Metrics
-------
  Computed separately for:
    - Full 24h signal
    - Daytime only (06:00-22:00)

  Also computed, per offset (not split all-day/daytime -- see
  compute_max_median_diff): the largest time-of-day difference between the
  original and counterfactual median AGP curves (magnitude, time of day, and
  direction).

Output structure (single-patient mode)
---------------------------------------
  {SENSITIVITY_PATH}/Data/{pid}_sensitivity.csv
    Contains metrics for all delta values (all-day + daytime columns)
  
  {SENSITIVITY_PATH}/Signals/{pid}_signals.csv
    Contains time series data:
    - time: timestamps
    - glucose_original: original signal
    - glucose_ctf_delta_0: counterfactual with δ=0 min
    - glucose_ctf_delta_15: counterfactual with δ=15 min
    - glucose_ctf_delta_30: counterfactual with δ=30 min
    - glucose_ctf_delta_45: counterfactual with δ=45 min
    - glucose_ctf_delta_60: counterfactual with δ=60 min
    - glucose_ctf_delta_75: counterfactual with δ=75 min
    - glucose_ctf_delta_90: counterfactual with δ=90 min

Output structure (cohort ALL mode)
------------------------------------
  {SENSITIVITY_PATH}/Data/cohort_sensitivity.csv
    Aggregated metrics across all patients
  
  {SENSITIVITY_PATH}/Signals/{pid}_signals.csv (one file per patient)
    Individual patient signal files

Usage:
    python3 -m scripts.counterfactual.sensitivity_analysis --patient_id ALL

Authors: Guillaume Deside
"""

import os
import argparse
import traceback

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# Import configuration
from src.config import (
    TIME_COL, GLUCOSE_COL,
    PREPROCESSING_PATH,
    SENSITIVITY_PATH,
    FILTER_STATUS_CSV,
)

from src.preprocessing.load_data import load_window_data, discover_patients
from src.preprocessing.timestep import detect_dominant_sampling_interval

from src.filtering.load_filtering import load_filtered_patient_list

from src.episodes.load_episodes import prepare_episode_data
from src.episodes.detection import detect_hyperglycemia_episodes
from src.episodes.criteria import classify_episodes

from src.utils.time_utils import is_daytime_mask, is_night
from src.counterfactual.build_counterfactual import build_full_counterfactual_signal
from src.agp.compute_agp import compute_agp_profile


from src.episodes.episode_config import (
    H_HYPER, H_CONTEXT, H_HYPO,
    D_MIN_MIN, D_MAX_MIN,
    MIN_DATA_COMPLETENESS,
    CONTEXT_WIN_MIN,
    NIGHT_START_H, NIGHT_END_H,
    LABEL_SLH, LABEL_LLH,
)

from src.counterfactual.counterfactual_config import (
    OFFSET_VALUES,
    APPLY_KERNEL,
    KERNEL_MIN,
)

# ── Metrics keys ─────────────────────────────────────────────────────────────
METRICS_KEYS = ["mean", "median", "sd", "cv", "TIR70-180", "TAR>180", "TAR>250", "TBR<70"]

# ── Max median-profile-difference parameters ──────────────────────────────────
DIFF_SIGN_EPS = 0.5   # mg/dL -- differences smaller than this are treated as "no change"


def hhmm_to_hour_float(hhmm: str) -> float:
    """'HH:MM' -> hour as a float (e.g. '08:30' -> 8.5)."""
    h, m = map(int, hhmm.split(":"))
    return h + m / 60.0


def compute_max_median_diff(agp_original: pd.DataFrame, agp_ctf: pd.DataFrame) -> dict | None:
    """
    Time-slot-wise Delta-median = median_original(t) - median_ctf(t), and the
    slot where |Delta| is largest.

    A positive signed difference means the counterfactual median is LOWER
    than the original at that slot (removing SLH episodes reduced glucose
    there). 

    Returns
    -------
    dict or None
        {"max_abs_diff", "signed_diff_at_max", "time_of_max"}, or None if
        no time slot has valid data in both profiles.
    """
    diff = agp_original["median"].to_numpy() - agp_ctf["median"].to_numpy()
    valid = ~np.isnan(diff)
    if not valid.any():
        return None

    abs_diff = np.where(valid, np.abs(diff), -np.inf)
    slot_idx = int(np.argmax(abs_diff))

    return {
        "max_abs_diff": float(abs(diff[slot_idx])),
        "signed_diff_at_max": float(diff[slot_idx]),
        "time_of_max": agp_original["time_str"].iloc[slot_idx],
    }



# =============================================================================
# METRICS COMPUTATION
# =============================================================================
def compute_metrics(glucose: np.ndarray, time_series: pd.Series = None, 
                   daytime_only: bool = False) -> dict:
    """
    Compute CGM metrics.
    
    Parameters
    ----------
    glucose : np.ndarray
        Glucose values (mg/dL)
    time_series : pd.Series, optional
        Timestamps corresponding to glucose values
    daytime_only : bool
        If True and time_series provided, compute metrics only for daytime hours (06:00-22:00)
    
    Returns
    -------
    dict
        Metrics dictionary
    """
    # Apply daytime filter if requested
    if daytime_only and time_series is not None:
        daytime_mask = is_daytime_mask(time_series)
        glucose = glucose[daytime_mask]
    
    # Filter out NaN
    g = glucose[~np.isnan(glucose)]
    if len(g) < 10:
        return {k: np.nan for k in METRICS_KEYS}
    
    n = len(g)
    mean_val = float(np.mean(g))
    
    return {
        "mean": round(mean_val, 2),
        "median": round(float(np.median(g)), 2),
        "sd": round(float(np.std(g, ddof=1)), 2),
        "cv": round(100 * float(np.std(g, ddof=1) / mean_val) if mean_val > 0 else np.nan, 4),
        "TIR70-180": round(100 * ((g >= H_HYPO) & (g <= H_HYPER)).sum() / n, 4),
        "TAR>180": round(100 * (g > H_HYPER).sum() / n, 4),
        "TAR>250": round(100 * (g > 250).sum() / n, 4),
        "TBR<70": round(100 * (g < H_HYPO).sum() / n, 4),
    }


def process_patient(patient_id: str, input_dir: str) -> tuple:
    """
    Process single patient: detect episodes and compute sensitivity metrics.
    
    Returns
    -------
    (df, timestep, slh_episodes, sensitivity_dict) or (None, None, None, None)
    """
    # Load patient data
    patient_dir = os.path.join(input_dir, patient_id)
    import glob
    files = glob.glob(os.path.join(patient_dir, f"{patient_id}_win1_*.xlsx"))
    
    if not files:
        return None, None, None, None
    
    df_raw = load_window_data(files[0])
    if df_raw is None:
        return None, None, None, None
    
    # Prepare data (remove flagged rows)
    df = prepare_episode_data(df_raw)
    if len(df) < 50:
        return None, None, None, None
    
    # Detect timestep
    timestep = detect_dominant_sampling_interval(df)
    
    # Detect and classify episodes
    raw_episodes = detect_hyperglycemia_episodes(df)
    classified_df = classify_episodes(df, raw_episodes)
    
    # Extract SLH episodes only
    slh_mask = classified_df["classification"] == LABEL_SLH
    slh_episodes = []
    
    for _, row in classified_df[slh_mask].iterrows():
        slh_episodes.append({
            "onset_time": row["onset_time"],
            "offset_time": row["offset_time"],
            "duration_min": row["duration_min"],
        })
    
    if len(slh_episodes) == 0:
        return df, timestep, [], {}
    
    # Compute original metrics (all-day + daytime)
    orig_glucose = df[GLUCOSE_COL].values.astype(float)
    orig_time = df[TIME_COL]
    
    orig_metrics = compute_metrics(orig_glucose, orig_time, daytime_only=False)
    orig_metrics_daytime = compute_metrics(orig_glucose, orig_time, daytime_only=True)

    # Original AGP median profile -- reused across all offsets below to find
    # the time-of-day slot where the counterfactual median differs most.
    agp_original = compute_agp_profile(df, timestep)

    # Sensitivity grid: compute counterfactual for each delta
    sensitivity = {}

    for offset in OFFSET_VALUES:
        ctf_signal = build_full_counterfactual_signal(df, slh_episodes, offset, timestep)

        df_ctf = df.copy()
        df_ctf[GLUCOSE_COL] = ctf_signal
        agp_ctf = compute_agp_profile(df_ctf, timestep)

        sensitivity[offset] = {
            "metrics": compute_metrics(ctf_signal, orig_time, daytime_only=False),
            "metrics_daytime": compute_metrics(ctf_signal, orig_time, daytime_only=True),
            "max_median_diff": compute_max_median_diff(agp_original, agp_ctf),
        }
    
    # Package results
    results = {
        "orig_metrics": orig_metrics,
        "orig_metrics_daytime": orig_metrics_daytime,
        "sensitivity": sensitivity,
    }
    
    return df, timestep, slh_episodes, results


def format_sensitivity_rows(patient_id: str, results: dict) -> list:
    """
    Format sensitivity results as CSV rows.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier
    results : dict
        Results dict from process_patient
        
    Returns
    -------
    list of dict
        Rows for CSV output
    """
    orig_metrics = results["orig_metrics"]
    orig_metrics_daytime = results["orig_metrics_daytime"]
    sensitivity = results["sensitivity"]
    
    rows = []
    
    for offset in OFFSET_VALUES:
        entry = sensitivity.get(offset, {})
        
        row = {
            "patient_id": patient_id,
            "offset": offset,
            "kernel": KERNEL_MIN if APPLY_KERNEL else 0,
        }
        
        # All-day metrics
        for k in METRICS_KEYS:
            ov = orig_metrics.get(k, np.nan)
            cv = entry.get("metrics", {}).get(k, np.nan)
            row[f"orig_{k}"] = ov
            row[f"ctf_{k}"] = cv
            row[f"delta_{k}"] = (cv - ov if np.isfinite(ov) and np.isfinite(cv) else np.nan)
        
        # Daytime-only metrics
        for k in METRICS_KEYS:
            ov = orig_metrics_daytime.get(k, np.nan)
            cv = entry.get("metrics_daytime", {}).get(k, np.nan)
            row[f"daytime_orig_{k}"] = ov
            row[f"daytime_ctf_{k}"] = cv
            row[f"daytime_delta_{k}"] = (cv - ov if np.isfinite(ov) and np.isfinite(cv) else np.nan)

        # Max median-profile difference: largest |Delta median AGP(t)| between
        # original and counterfactual, and the time of day it occurs at.
        max_diff = entry.get("max_median_diff")
        if max_diff is not None:
            signed_diff = max_diff["signed_diff_at_max"]
            time_of_max = max_diff["time_of_max"]
            if abs(signed_diff) < DIFF_SIGN_EPS:
                direction = "No change"
            elif signed_diff > 0:
                direction = "Counterfactual < Original"
            else:
                direction = "Counterfactual > Original"

            row["max_median_diff_abs"] = round(max_diff["max_abs_diff"], 2)
            row["max_median_diff_signed"] = round(signed_diff, 2)
            row["max_median_diff_time"] = time_of_max
            row["max_median_diff_time_hour"] = round(hhmm_to_hour_float(time_of_max), 3)
            row["max_median_diff_direction"] = direction
        else:
            row["max_median_diff_abs"] = np.nan
            row["max_median_diff_signed"] = np.nan
            row["max_median_diff_time"] = None
            row["max_median_diff_time_hour"] = np.nan
            row["max_median_diff_direction"] = None

        rows.append(row)
    
    return rows


def save_signal_data(patient_id: str, df: pd.DataFrame, 
                    slh_episodes: list[dict], timestep: float,
                    output_dir: str):
    """
    Save original and counterfactual signals to CSV file.
    
    Creates a single file with columns:
    - time: timestamp
    - glucose_original: original signal
    - glucose_ctf_delta_{X}: counterfactual signal for each delta value
    
    Parameters
    ----------
    patient_id : str
        Patient identifier
    df : pd.DataFrame
        Original patient data
    slh_episodes : list of dict
        SLH episodes
    timestep : float
        Sampling interval
    output_dir : str
        Output directory
    """
    # Build output dataframe starting with time and original glucose
    output_df = pd.DataFrame({
        TIME_COL: df[TIME_COL],
        "glucose_original": df[GLUCOSE_COL].values.astype(float),
    })
    
    # Add counterfactual signal for each delta value
    for offset in OFFSET_VALUES:
        ctf_signal = build_full_counterfactual_signal(df, slh_episodes, offset, timestep)
        output_df[f"glucose_ctf_delta_{offset}"] = ctf_signal
    
    # Save to CSV
    signals_dir = os.path.join(output_dir, "Signals")
    os.makedirs(signals_dir, exist_ok=True)
    
    output_path = os.path.join(signals_dir, f"{patient_id}_signals.csv")
    output_df.to_csv(output_path, index=False)
    
    return output_path


# =============================================================================
# SENSITIVITY SUMMARY
# =============================================================================
def _mean_ci95(vals: pd.Series) -> tuple:
    """Mean and 95% CI of the mean (t-distribution), given a sample."""
    n = len(vals)
    mean_val = float(np.mean(vals))
    if n < 2:
        return mean_val, np.nan, np.nan
    sem = float(np.std(vals, ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return mean_val, mean_val - tcrit * sem, mean_val + tcrit * sem


def _describe_by_offset(cohort_df: pd.DataFrame, col: str, time_window: str, metric: str) -> list:
    """n, mean, CI95, median, Q1, Q3, IQR, min, max of `col`, one record per offset value."""
    if col not in cohort_df.columns:
        return []

    records = []
    for offset, grp in cohort_df.groupby("offset"):
        vals = grp[col].dropna()
        if vals.empty:
            continue
        q1, q3 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        mean_val, ci95_low, ci95_high = _mean_ci95(vals)
        records.append({
            "time_window": time_window,
            "metric": metric,
            "offset_min": int(offset),
            "n": len(vals),
            "mean": round(mean_val, 4),
            "ci95_low": round(ci95_low, 4) if np.isfinite(ci95_low) else np.nan,
            "ci95_high": round(ci95_high, 4) if np.isfinite(ci95_high) else np.nan,
            "median": round(float(np.median(vals)), 4),
            "Q1": round(q1, 4),
            "Q3": round(q3, 4),
            "IQR": round(q3 - q1, 4),
            "min": round(float(vals.min()), 4),
            "max": round(float(vals.max()), 4),
        })
    return records


def summarize_sensitivity(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute article-ready summary statistics of delta values across patients.

    For each combination of (time_window, metric, offset) computes:
      n, mean, ci95_low, ci95_high, median, Q1, Q3, IQR, min, max of the
      delta column. The CI95 is the 95% confidence interval of the mean
      (t-distribution).

    Also includes the max median-profile-difference magnitude and its
    time-of-day (time_window="profile_diff"), which -- unlike the metrics
    above -- are not split into all-day/daytime variants: they already
    describe a single time-of-day comparison across the full 24h profile
    (see compute_max_median_diff).

    Parameters
    ----------
    cohort_df : pd.DataFrame
        Cohort sensitivity DataFrame as produced by format_sensitivity_rows.

    Returns
    -------
    pd.DataFrame
        Long-format summary with columns:
        time_window, metric, offset, n, mean, ci95_low, ci95_high,
        median, Q1, Q3, IQR, min, max
    """
    records = []

    for time_window, prefix in [("all_day", "delta_"), ("daytime", "daytime_delta_")]:
        for metric in METRICS_KEYS:
            records.extend(_describe_by_offset(cohort_df, f"{prefix}{metric}", time_window, metric))

    records.extend(_describe_by_offset(
        cohort_df, "max_median_diff_abs", "profile_diff", "max_median_diff_abs"
    ))
    records.extend(_describe_by_offset(
        cohort_df, "max_median_diff_time_hour", "profile_diff", "max_median_diff_time_hour"
    ))

    summary = pd.DataFrame(records)
    summary = summary.sort_values(["time_window", "metric", "offset_min"]).reset_index(drop=True)
    return summary


def save_sensitivity_summary(cohort_df: pd.DataFrame, output_dir: str) -> str:
    """
    Compute summary stats and save to CSV.

    Parameters
    ----------
    cohort_df : pd.DataFrame
        Cohort sensitivity DataFrame.
    output_dir : str
        Base output directory (summary saved to <output_dir>/Data/).

    Returns
    -------
    str
        Path to the saved CSV file.
    """
    summary = summarize_sensitivity(cohort_df)

    out_path = os.path.join(output_dir, "Data", "sensitivity_summary.csv")
    summary.to_csv(out_path, index=False)
    return out_path


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main execution pipeline."""
    parser = argparse.ArgumentParser(
        description="CGM sensitivity analysis -- modular linear interpolation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--patient_id", default="ALL", help="Patient ID or ALL")
    parser.add_argument("--input_dir", default=PREPROCESSING_PATH)
    parser.add_argument("--output_dir", default=SENSITIVITY_PATH)
    parser.add_argument("--filter_csv", default=FILTER_STATUS_CSV)
    parser.add_argument("--no_filter", action="store_true")
    
    args = parser.parse_args()
    
    kernel_str = f"k={KERNEL_MIN} min" if APPLY_KERNEL else "no kernel"
    
    print("\n" + "=" * 70)
    print("  CGM Sensitivity Analysis — Modular Linear Interpolation")
    print("=" * 70)
    print(f"  Input:   {args.input_dir}")
    print(f"  Output:  {args.output_dir}")
    print(f"  Patient: {args.patient_id}")
    print(f"  Kernel:  {kernel_str}")
    print(f"  Delta:   {OFFSET_VALUES} min")
    print(f"  Daytime: 06:00-22:00 (separate metrics)")
    print("=" * 70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "Data"), exist_ok=True)
    
    # Single patient mode
    if args.patient_id.strip().upper() != "ALL":
        pid = args.patient_id.strip()
        print(f"\n  Single-patient mode: {pid}\n")
        
        df, timestep, slh_episodes, results = process_patient(pid, args.input_dir)
        
        if df is None:
            print(f"  [{pid}] SKIP -- no data or processing failed")
            return
        
        n_slh = len(slh_episodes)
        print(f"  [{pid}]  timestep={timestep:.0f} min  rows={len(df)}  SLH={n_slh}")
        
        if n_slh > 0:
            # Save metrics CSV
            rows = format_sensitivity_rows(pid, results)
            output_path = os.path.join(args.output_dir, "Data", f"{pid}_sensitivity.csv")
            pd.DataFrame(rows).to_csv(output_path, index=False)
            print(f"  Saved metrics: {output_path}")
            
            # Save signal data (original + counterfactuals)
            signals_path = save_signal_data(pid, df, slh_episodes, timestep, args.output_dir)
            print(f"  Saved signals: {signals_path}")
        else:
            print(f"  [{pid}] No SLH episodes detected")
    
    # Cohort mode
    else:
        filtered = (None if args.no_filter else load_filtered_patient_list(args.filter_csv))
        all_pids = discover_patients(args.input_dir)
        
        if not all_pids:
            print(f"  [ERROR] No patients found in {args.input_dir}")
            return
        
        pids = ([p for p in all_pids if p in filtered] if filtered is not None else all_pids)
        print(f"\n  Cohort mode: {len(pids)} patients\n")
        
        all_rows = []
        n_ok = 0
        n_fail = 0
        n_signals_saved = 0
        
        for pid in tqdm(pids, desc="  Processing", ncols=70):
            try:
                df, timestep, slh_episodes, results = process_patient(pid, args.input_dir)
                
                if df is None:
                    n_fail += 1
                    continue
                
                n_slh = len(slh_episodes)
                
                if n_slh > 0:
                    # Format metrics rows
                    rows = format_sensitivity_rows(pid, results)
                    
                    # Add to cohort metrics
                    all_rows.extend(rows)
                    
                    # Save individual patient metrics CSV
                    output_path = os.path.join(args.output_dir, "Data", f"{pid}_sensitivity.csv")
                    pd.DataFrame(rows).to_csv(output_path, index=False)
                    
                    # Save individual patient signal data
                    save_signal_data(pid, df, slh_episodes, timestep, args.output_dir)
                    n_signals_saved += 1
                
                n_ok += 1
                
            except Exception as exc:
                print(f"\n  [{pid}] ERROR: {exc}")
                traceback.print_exc()
                n_fail += 1
        
        # Save cohort metrics
        if all_rows:
            cohort_df = pd.DataFrame(all_rows)
            cohort_csv = os.path.join(args.output_dir, "Data", "cohort_sensitivity.csv")
            cohort_df.to_csv(cohort_csv, index=False)
            print(f"\n  Cohort metrics CSV ({len(cohort_df)} rows) -> {cohort_csv}")

            summary_csv = save_sensitivity_summary(cohort_df, args.output_dir)
            print(f"  Sensitivity summary CSV -> {summary_csv}")
        
        print(f"\n  Processed: {n_ok}  |  Failed: {n_fail}")
        print(f"  Signal files saved: {n_signals_saved}")
    
    print("\n" + "=" * 70)
    print(f"Output directories:")
    print(f"    Metrics:  {os.path.join(args.output_dir, 'Data')}/")
    print(f"    Signals:  {os.path.join(args.output_dir, 'Signals')}/")
    print(f"    Summary:  {os.path.join(args.output_dir, 'Data', 'sensitivity_summary.csv')}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()