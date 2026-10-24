"""
scripts/statistics/slh_llh_cohort_descriptive_stats.py
======================================================
Cohort-level descriptive statistics for SLH / LLH episodes.

Computes and reports the following for the Results section:
  - Total raw hyperglycemia episodes (>180 mg/dL) detected, before classification
  - Total SLH episodes and per-patient distribution (median, IQR, min-max)
  - Proportion of raw episodes classified as SLH (cohort-level and per-patient
    range), i.e. what fraction of a patient's hyperglycemia peaks are SLH
  - Total "night SLH" episodes (pass every SLH criterion except daytime, C3)
    and the same per-patient distribution/proportion statistics as SLH
  - SLH episode duration characteristics (median, IQR, range)
  - SLH onset time distribution (hour of day across all patients)
  - Recurrent SLH pattern prevalence (≥ 50 % of days)
  - LLH episode duration characteristics (median, IQR, range)
  - Explicit breakdown of the 3 conceptual classification criteria used to
    define daytime SLH, mapped onto the underlying C1-C6 boolean columns:
      1. Short duration        -> C1
      2. Isolated episode      -> C4 AND C5 AND C6
      3. Daytime occurrence    -> C3
    (C2, data completeness, is a quality gate applied before these three are
    evaluated — it is not itself one of the "3 classification criteria".)

Inputs
------
  Episodes/<patient_id>_episodes.csv  — per-patient episode files
  Episodes/episodes_summary.csv       — cohort summary (optional)

Outputs
-------
  Results/Cohort_Descriptive_Stats/cohort_per_patient.csv
  Results/Cohort_Descriptive_Stats/summary_statistics.txt
  Results/Cohort_Descriptive_Stats/fig_cohort_distributions.{pdf,png}
  Results/Cohort_Descriptive_Stats/fig_cohort_funnel.{pdf,png}
  Results/Cohort_Descriptive_Stats/fig_simplified_funnel.{pdf,png}

Usage
-----
    python3 -m scripts.statistics.slh_llh_cohort_descriptive_stats [--no_filter]

Authors
-------
Guillaume Deside, Laura Symul, Philippe Lysy, Zain Syed
UCL — slh-counterfactual-cgm project

Date
----
2025
"""

import os
import argparse
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    DPI_SAVE,
    FIGURE_WIDTH_INCHES,
    EPISODES_PATH,
    FILTER_STATUS_CSV,
)
from src.config.paths import RESULTS_PATH
from src.episodes.episode_config import (
    LABEL_SLH,
    LABEL_LLH,
    D_MIN_MIN,
    D_MAX_MIN,
)
from src.filtering.load_filtering import load_filtered_patient_list
from src.episodes.load_episodes import load_patient_episodes, discover_episode_files
from src.episodes.episode_config import (
    H_CONTEXT, MIN_DATA_COMPLETENESS, CONTEXT_WIN_MIN,
)
from src.visualization.colors import WONG_COLORS, FUNNEL_COLORS, CROSSING_COLORS
from src.visualization.style import apply_publication_style
from src.visualization.style import FONT_SIZE_BASE

# =============================================================================
# CONFIGURATION
# =============================================================================
EPISODES_DIR = EPISODES_PATH
OUTPUT_DIR   = RESULTS_PATH / "Cohort_Descriptive_Stats"

# Fraction of days with ≥ 1 SLH episode to qualify as a recurrent pattern
RECURRENCE_THRESHOLD = 0.50


# =============================================================================
# RECURRENCE CALCULATION
# =============================================================================
def calculate_recurrence_from_episodes(episodes_df: pd.DataFrame,
                                       window_days: int = 45) -> tuple[float, bool]:
    """
    Calculate SLH recurrence: percentage of days with at least one SLH episode.

    Parameters
    ----------
    episodes_df : pd.DataFrame
        SLH episode rows with onset_time and offset_time columns.
    window_days : int
        Minimum window length used as denominator when the observed range is shorter.

    Returns
    -------
    recurrence_pct : float
        Percentage (0–100) of days containing at least one SLH episode.
    has_recurrent_pattern : bool
        True when recurrence_pct ≥ RECURRENCE_THRESHOLD × 100.
    """
    if len(episodes_df) == 0:
        return 0.0, False

    start_date = episodes_df["onset_time"].dt.date.min()
    end_date   = episodes_df["offset_time"].dt.date.max()
    n_days     = max((end_date - start_date).days + 1, window_days)

    days_with_slh: set = set()
    for _, ep in episodes_df.iterrows():
        current = ep["onset_time"].date()
        ep_end  = ep["offset_time"].date()
        while current <= ep_end:
            days_with_slh.add(current)
            current += timedelta(days=1)

    recurrence_pct      = 100.0 * len(days_with_slh) / n_days if n_days > 0 else 0.0
    has_recurrent_pattern = recurrence_pct >= RECURRENCE_THRESHOLD * 100

    return recurrence_pct, has_recurrent_pattern


# =============================================================================
# PATIENT-LEVEL PROCESSING
# =============================================================================
def process_patient_statistics(patient_id: str) -> dict | None:
    """
    Compute episode statistics for a single patient.

    Uses the shared ``load_patient_episodes`` loader from
    ``src.episodes.load_episodes``, which reads from the configured
    ``EPISODES_PATH``.

    Parameters
    ----------
    patient_id : str
        Patient identifier.

    Returns
    -------
    dict or None
        Per-patient statistics, or None when no episodes are available.
    """
    episodes_df = load_patient_episodes(patient_id)

    if episodes_df is None or len(episodes_df) == 0:
        print(f"  [{patient_id}] SKIP — no episodes")
        return None

    slh_episodes = episodes_df[episodes_df["classification"] == LABEL_SLH]
    llh_episodes = episodes_df[episodes_df["classification"] == LABEL_LLH]

    n_slh = len(slh_episodes)
    n_llh = len(llh_episodes)

    slh_durations = slh_episodes["duration_min"].values if n_slh > 0 else []
    llh_durations = llh_episodes["duration_min"].values if n_llh > 0 else []

    recurrence_pct, has_recurrent = calculate_recurrence_from_episodes(slh_episodes)

    slh_onset_hours = (
        (slh_episodes["onset_time"].dt.hour + slh_episodes["onset_time"].dt.minute / 60).tolist()
        if n_slh > 0 else []
    )

    # ── Funnel counts from C1–C6 boolean columns ──────────────────────────────
    has_criteria_cols = all(c in episodes_df.columns for c in ("C1", "C2", "C3", "C4", "C5", "C6"))
    if has_criteria_cols:
        c = episodes_df
        n_raw_ep = len(c)
        n_c1_ep  = int(c["C1"].sum())
        n_c2_ep  = int((c["C1"] & c["C2"]).sum())
        n_c3_ep  = int((c["C1"] & c["C2"] & c["C3"]).sum())
        n_c4_ep  = int((c["C1"] & c["C2"] & c["C3"] & c["C4"]).sum())
        n_c5_ep  = int((c["C1"] & c["C2"] & c["C3"] & c["C4"] & c["C5"]).sum())
        n_slh_ep = int((c["C1"] & c["C2"] & c["C3"] & c["C4"] & c["C5"] & c["C6"]).sum())

        # Simplified funnel: C1 → C4+C5+C6 → C2 → C3
        n_c1_c456_ep    = int((c["C1"] & c["C4"] & c["C5"] & c["C6"]).sum())
        n_c1_c2_c456_ep = int((c["C1"] & c["C2"] & c["C4"] & c["C5"] & c["C6"]).sum())

        # Night SLH: passes all criteria except C3 (daytime)
        night_mask = c["C1"] & c["C2"] & ~c["C3"] & c["C4"] & c["C5"] & c["C6"]
        n_night_slh_ep = int(night_mask.sum())
        night_eps = c[night_mask]
        night_slh_onset_hours = (
            (night_eps["onset_time"].dt.hour + night_eps["onset_time"].dt.minute / 60).tolist()
            if n_night_slh_ep > 0 else []
        )
        night_slh_durations = night_eps["duration_min"].values.tolist() if n_night_slh_ep > 0 else []
    else:
        n_raw_ep = n_slh + n_llh
        n_c1_ep = n_c2_ep = n_c3_ep = n_c4_ep = n_c5_ep = n_slh_ep = 0
        n_c1_c456_ep = n_c1_c2_c456_ep = n_night_slh_ep = 0
        night_slh_onset_hours = []
        night_slh_durations = []

    # C6 crossing breakdown (episodes that pass C1–C5 and are evaluated for C6)
    has_crossing_cols = has_criteria_cols and "c6_crosses_before" in episodes_df.columns
    if has_crossing_cols:
        reaches_c6 = c["C1"] & c["C2"] & c["C3"] & c["C4"] & c["C5"]
        c6_ep = c[reaches_c6]
        n_c6_both        = int((c6_ep["c6_crosses_before"] & c6_ep["c6_crosses_after"]).sum())
        n_c6_before_only = int((c6_ep["c6_crosses_before"] & ~c6_ep["c6_crosses_after"]).sum())
        n_c6_after_only  = int((~c6_ep["c6_crosses_before"] & c6_ep["c6_crosses_after"]).sum())
        n_c6_no_cross    = int((~c6_ep["c6_crosses_before"] & ~c6_ep["c6_crosses_after"]).sum())
    else:
        n_c6_both = n_c6_before_only = n_c6_after_only = n_c6_no_cross = 0

    # Per-patient proportion of raw hyperglycemia peaks classified as SLH /
    # night-SLH, i.e. "what fraction of THIS patient's peaks are SLH" — the
    # cohort-level min/max of these two columns is the per-patient proportion
    # RANGE quoted in the Results text (distinct from the single cohort-wide
    # percentage total_slh / funnel_n_raw).
    prop_slh_pct       = 100.0 * n_slh_ep       / n_raw_ep if n_raw_ep > 0 else np.nan
    prop_night_slh_pct = 100.0 * n_night_slh_ep / n_raw_ep if n_raw_ep > 0 else np.nan

    result = {
        "patient_id":          patient_id,
        "n_slh":               n_slh,
        "n_llh":               n_llh,
        "slh_median_duration": np.median(slh_durations)         if len(slh_durations) > 0 else np.nan,
        "slh_q25_duration":    np.percentile(slh_durations, 25) if len(slh_durations) > 0 else np.nan,
        "slh_q75_duration":    np.percentile(slh_durations, 75) if len(slh_durations) > 0 else np.nan,
        "recurrence_pct":      recurrence_pct,
        "has_recurrent_pattern": has_recurrent,
        "llh_median_duration": np.median(llh_durations)         if len(llh_durations) > 0 else np.nan,
        "llh_q25_duration":    np.percentile(llh_durations, 25) if len(llh_durations) > 0 else np.nan,
        "llh_q75_duration":    np.percentile(llh_durations, 75) if len(llh_durations) > 0 else np.nan,
        # stored for cohort pooling; excluded from the per-patient CSV
        "slh_durations":          list(slh_durations),
        "llh_durations":          list(llh_durations),
        "slh_onset_hours":        slh_onset_hours,
        "night_slh_onset_hours":  night_slh_onset_hours,
        "night_slh_durations":    night_slh_durations,
        # funnel counts (sequential C1 → SLH)
        "n_raw_ep":               n_raw_ep,
        "n_c1_ep":                n_c1_ep,
        "n_c2_ep":                n_c2_ep,
        "n_c3_ep":                n_c3_ep,
        "n_c4_ep":                n_c4_ep,
        "n_c5_ep":                n_c5_ep,
        "n_slh_ep":               n_slh_ep,
        # simplified funnel counts (C1 → C4+C5+C6 → C2 → C3)
        "n_c1_c456_ep":           n_c1_c456_ep,
        "n_c1_c2_c456_ep":        n_c1_c2_c456_ep,
        "n_night_slh_ep":         n_night_slh_ep,
        "n_c6_both":              n_c6_both,
        "n_c6_before_only":       n_c6_before_only,
        "n_c6_after_only":        n_c6_after_only,
        "n_c6_no_cross":          n_c6_no_cross,
        # per-patient proportion of raw peaks that are SLH / night-SLH (%)
        "prop_slh_pct":           prop_slh_pct,
        "prop_night_slh_pct":     prop_night_slh_pct,
    }

    print(f"  [{patient_id}] SLH={n_slh:3d}  LLH={n_llh:3d}  "
          f"Recurrence={recurrence_pct:.1f}%  "
          f"Recurrent={'Yes' if has_recurrent else 'No'}")

    return result


# =============================================================================
# COHORT-LEVEL STATISTICS
# =============================================================================
def compute_cohort_statistics(results: list[dict]) -> dict:
    """
    Aggregate per-patient results into cohort-level descriptive statistics.

    Parameters
    ----------
    results : list of dict
        Per-patient statistics as returned by ``process_patient_statistics``.

    Returns
    -------
    dict
        Cohort summary statistics (totals, medians, IQRs, recurrence rates).
    """
    df = pd.DataFrame(results)

    slh_per_patient       = df["n_slh"].values
    night_slh_per_patient = df["n_night_slh_ep"].values
    raw_per_patient       = df["n_raw_ep"].values
    prop_slh_per_patient       = df["prop_slh_pct"].dropna().values
    prop_night_slh_per_patient = df["prop_night_slh_pct"].dropna().values
    all_slh_durations = [d for r in results for d in r["slh_durations"]]
    all_llh_durations = [d for r in results for d in r["llh_durations"]]
    all_slh_onset_hours = [h for r in results for h in r["slh_onset_hours"]]

    # Cohort-level funnel totals
    fn_raw = int(df["n_raw_ep"].sum())
    fn_c1  = int(df["n_c1_ep"].sum())
    fn_c2  = int(df["n_c2_ep"].sum())
    fn_c3  = int(df["n_c3_ep"].sum())
    fn_c4  = int(df["n_c4_ep"].sum())
    fn_c5  = int(df["n_c5_ep"].sum())
    fn_slh = int(df["n_slh_ep"].sum())
    fn_c1_c456    = int(df["n_c1_c456_ep"].sum())
    fn_c1_c2_c456 = int(df["n_c1_c2_c456_ep"].sum())
    fn_night_slh  = int(df["n_night_slh_ep"].sum())
    fn_c6_both       = int(df["n_c6_both"].sum())
    fn_c6_before_only = int(df["n_c6_before_only"].sum())
    fn_c6_after_only  = int(df["n_c6_after_only"].sum())
    fn_c6_no_cross   = int(df["n_c6_no_cross"].sum())

    # "3 classification criteria" of daytime SLH, mapped onto C1-C6 (see
    # module docstring): criterion 1 = C1 (duration), criterion 2 = C4+C5+C6
    # (isolated), criterion 3 = C3 (daytime). C2 (data completeness) is a
    # quality gate applied first, evaluated on the raw episode pool.
    crit1_excluded = fn_raw - fn_c1        # fails duration (C1)
    crit2_excluded = fn_c1  - fn_c1_c456   # fails isolation (C4/C5/C6), among C1-passing
    crit3_excluded = fn_night_slh          # fails daytime (C3), among C1+isolated+C2-passing == night SLH

    return {
        "n_patients":             len(df),
        "total_slh":              int(df["n_slh"].sum()),
        "median_slh_per_patient": np.median(slh_per_patient),
        "q25_slh_per_patient":    np.percentile(slh_per_patient, 25),
        "q75_slh_per_patient":    np.percentile(slh_per_patient, 75),
        "min_slh_per_patient":    int(np.min(slh_per_patient))  if len(slh_per_patient) > 0 else np.nan,
        "max_slh_per_patient":    int(np.max(slh_per_patient))  if len(slh_per_patient) > 0 else np.nan,
        "median_slh_duration":    np.median(all_slh_durations)         if all_slh_durations else np.nan,
        "q25_slh_duration":       np.percentile(all_slh_durations, 25) if all_slh_durations else np.nan,
        "q75_slh_duration":       np.percentile(all_slh_durations, 75) if all_slh_durations else np.nan,
        "min_slh_duration":       np.min(all_slh_durations)            if all_slh_durations else np.nan,
        "max_slh_duration":       np.max(all_slh_durations)            if all_slh_durations else np.nan,
        "n_recurrent":            int(df["has_recurrent_pattern"].sum()),
        "recurrent_pct":          100.0 * df["has_recurrent_pattern"].mean() if len(df) > 0 else 0.0,
        "total_llh":              int(df["n_llh"].sum()),
        "median_llh_duration":    np.median(all_llh_durations)         if all_llh_durations else np.nan,
        "q25_llh_duration":       np.percentile(all_llh_durations, 25) if all_llh_durations else np.nan,
        "q75_llh_duration":       np.percentile(all_llh_durations, 75) if all_llh_durations else np.nan,
        "min_llh_duration":       np.min(all_llh_durations)            if all_llh_durations else np.nan,
        "max_llh_duration":       np.max(all_llh_durations)            if all_llh_durations else np.nan,

        # ── "SLH but night": episodes that pass every SLH criterion except
        # daytime (C3) — i.e. would be SLH if they hadn't occurred at night.
        "total_night_slh":              int(fn_night_slh),
        "median_night_slh_per_patient": np.median(night_slh_per_patient),
        "q25_night_slh_per_patient":    np.percentile(night_slh_per_patient, 25),
        "q75_night_slh_per_patient":    np.percentile(night_slh_per_patient, 75),
        "min_night_slh_per_patient":    int(np.min(night_slh_per_patient)) if len(night_slh_per_patient) > 0 else np.nan,
        "max_night_slh_per_patient":    int(np.max(night_slh_per_patient)) if len(night_slh_per_patient) > 0 else np.nan,

        # ── Proportions: what fraction of raw hyperglycemia peaks are SLH /
        # night-SLH — both the single cohort-wide percentage (pooled across
        # all episodes) and the per-patient range (min-max across patients),
        # which is the "varied greatly, ranging from X% to X%" statistic.
        "pct_slh_of_raw":         100.0 * fn_slh       / fn_raw if fn_raw > 0 else np.nan,
        "pct_night_slh_of_raw":   100.0 * fn_night_slh / fn_raw if fn_raw > 0 else np.nan,
        "median_prop_slh_of_raw_per_patient": np.median(prop_slh_per_patient)       if len(prop_slh_per_patient) > 0 else np.nan,
        "min_prop_slh_of_raw_per_patient":    np.min(prop_slh_per_patient)          if len(prop_slh_per_patient) > 0 else np.nan,
        "max_prop_slh_of_raw_per_patient":    np.max(prop_slh_per_patient)          if len(prop_slh_per_patient) > 0 else np.nan,
        "median_prop_night_slh_of_raw_per_patient": np.median(prop_night_slh_per_patient) if len(prop_night_slh_per_patient) > 0 else np.nan,
        "min_prop_night_slh_of_raw_per_patient":    np.min(prop_night_slh_per_patient)    if len(prop_night_slh_per_patient) > 0 else np.nan,
        "max_prop_night_slh_of_raw_per_patient":    np.max(prop_night_slh_per_patient)    if len(prop_night_slh_per_patient) > 0 else np.nan,

        # funnel
        "funnel_n_raw": fn_raw,
        "funnel_n_c1":  fn_c1,
        "funnel_n_c2":  fn_c2,
        "funnel_n_c3":  fn_c3,
        "funnel_n_c4":  fn_c4,
        "funnel_n_c5":  fn_c5,
        "funnel_n_slh": fn_slh,
        "funnel_excl_c1": fn_raw - fn_c1,
        "funnel_excl_c2": fn_c1  - fn_c2,
        "funnel_excl_c3": fn_c2  - fn_c3,
        "funnel_excl_c4": fn_c3  - fn_c4,
        "funnel_excl_c5": fn_c4  - fn_c5,
        "funnel_excl_c6": fn_c5  - fn_slh,
        # simplified funnel: C1 → C4+C5+C6 → C2 → C3
        "funnel_n_c1_c456":        fn_c1_c456,
        "funnel_n_c1_c2_c456":     fn_c1_c2_c456,
        "funnel_n_night_slh":      fn_night_slh,
        # "3 classification criteria" breakdown (see module docstring)
        "criterion1_duration_excluded":  crit1_excluded,   # C1
        "criterion2_isolation_excluded": crit2_excluded,   # C4 AND C5 AND C6
        "criterion3_daytime_excluded":   crit3_excluded,   # C3 (== night SLH)
        "funnel_excl_c456_simp":   fn_c1       - fn_c1_c456,
        "funnel_excl_c2_simp":     fn_c1_c456  - fn_c1_c2_c456,
        "funnel_excl_c3_simp":     fn_c1_c2_c456 - fn_slh,
        "c6_both":                fn_c6_both,
        "c6_before_only":         fn_c6_before_only,
        "c6_after_only":          fn_c6_after_only,
        "c6_no_cross":            fn_c6_no_cross,
    }


# =============================================================================
# OUTPUT GENERATION
# =============================================================================
def add_panel_label(ax, label, x=-0.12, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="right")


def generate_summary_table(stats: dict, output_path: str) -> str:
    """
    Write a human-readable summary table and return it as a string.

    Parameters
    ----------
    stats : dict
        Cohort statistics from ``compute_cohort_statistics``.
    output_path : str
        Destination file path.

    Returns
    -------
    str
        The formatted summary text.
    """
    lines = [
        "=" * 70,
        "  COHORT DESCRIPTIVE STATISTICS",
        "=" * 70,
        "",
        f"Cohort size:                      {stats['n_patients']} patients",
        "",
        "SLH EPISODES (daytime — pass all 3 classification criteria):",
        f"  Total episodes:                 {stats['total_slh']:,}",
        f"  Proportion of raw peaks:        {stats['pct_slh_of_raw']:.1f}%  (cohort-wide: {stats['total_slh']:,} / {stats['funnel_n_raw']:,})",
        f"  Per patient (median):           {stats['median_slh_per_patient']:.1f}",
        f"  Per patient (IQR):              {stats['q25_slh_per_patient']:.1f}–{stats['q75_slh_per_patient']:.1f}",
        f"  Per patient (range):            {stats['min_slh_per_patient']}–{stats['max_slh_per_patient']}",
        f"  Per-patient proportion of raw peaks that are SLH:",
        f"    median {stats['median_prop_slh_of_raw_per_patient']:.1f}%,  "
        f"range {stats['min_prop_slh_of_raw_per_patient']:.1f}%–{stats['max_prop_slh_of_raw_per_patient']:.1f}%",
        f"  Duration (median):              {stats['median_slh_duration']:.1f} min",
        f"  Duration (IQR):                 {stats['q25_slh_duration']:.1f}–{stats['q75_slh_duration']:.1f} min",
        f"  Duration (range):               {stats['min_slh_duration']:.1f}–{stats['max_slh_duration']:.1f} min",
        f"  Recurrent patterns (≥50%):      {stats['recurrent_pct']:.1f}% ({stats['n_recurrent']} patients)",
        "",
        "SLH BUT NIGHT (pass criteria 1 & 2 — duration & isolation — but occur",
        "at night, i.e. fail criterion 3 / C3; would be SLH had they occurred",
        "during the day):",
        f"  Total episodes:                 {stats['total_night_slh']:,}",
        f"  Proportion of raw peaks:        {stats['pct_night_slh_of_raw']:.1f}%  (cohort-wide: {stats['total_night_slh']:,} / {stats['funnel_n_raw']:,})",
        f"  Per patient (median):           {stats['median_night_slh_per_patient']:.1f}",
        f"  Per patient (IQR):              {stats['q25_night_slh_per_patient']:.1f}–{stats['q75_night_slh_per_patient']:.1f}",
        f"  Per patient (range):            {stats['min_night_slh_per_patient']}–{stats['max_night_slh_per_patient']}",
        f"  Per-patient proportion of raw peaks that are night-SLH:",
        f"    median {stats['median_prop_night_slh_of_raw_per_patient']:.1f}%,  "
        f"range {stats['min_prop_night_slh_of_raw_per_patient']:.1f}%–{stats['max_prop_night_slh_of_raw_per_patient']:.1f}%",
        "",
        "LLH EPISODES:",
        f"  Total episodes:                 {stats['total_llh']:,}",
        f"  Duration (median):              {stats['median_llh_duration']:.1f} min",
        f"  Duration (IQR):                 {stats['q25_llh_duration']:.1f}–{stats['q75_llh_duration']:.1f} min",
        f"  Duration (range):               {stats['min_llh_duration']:.1f}–{stats['max_llh_duration']:.1f} min",
        "",
        "\"3 CLASSIFICATION CRITERIA\" OF DAYTIME SLH (mapped onto C1–C6 — see",
        "module docstring; C2 data-completeness is a quality gate, not one of",
        "the 3 criteria, and is applied to the raw pool before they are checked):",
        f"  Criterion 1 — Short duration  (C1):        −{stats['criterion1_duration_excluded']:,} excluded",
        f"  Criterion 2 — Isolated episode (C4+C5+C6): −{stats['criterion2_isolation_excluded']:,} excluded",
        f"  Criterion 3 — Daytime occurrence (C3):     −{stats['criterion3_daytime_excluded']:,} excluded (= SLH but night, above)",
        "",
        "EPISODE CLASSIFICATION FUNNEL (sequential C1 → C2 → C3 → C4 → C5 → C6):",
        f"  Raw hyperglycemia episodes:     {stats['funnel_n_raw']:,}",
        f"  After C1 (duration bounds):     {stats['funnel_n_c1']:,}  (−{stats['funnel_excl_c1']:,})",
        f"  After C2 (data completeness):   {stats['funnel_n_c2']:,}  (−{stats['funnel_excl_c2']:,})",
        f"  After C3 (daytime only):        {stats['funnel_n_c3']:,}  (−{stats['funnel_excl_c3']:,})",
        f"  After C4 (no pre-hyper):        {stats['funnel_n_c4']:,}  (−{stats['funnel_excl_c4']:,})",
        f"  After C5 (no post-hyper):       {stats['funnel_n_c5']:,}  (−{stats['funnel_excl_c5']:,})",
        f"  After C6 (OR-crossing):  SLH   {stats['funnel_n_slh']:,}  (−{stats['funnel_excl_c6']:,})",
        "",
        "=" * 70,
    ]
    summary = "\n".join(lines)

    with open(output_path, "w") as fh:
        fh.write(summary)

    print(f"  Summary table saved to: {output_path}")
    return summary


def plot_distributions(results: list[dict], stats: dict, output_dir: str) -> None:
    """
    Create a 4-panel publication-quality figure for episode distributions.

    Panels
    ------
    A : SLH episodes per patient (histogram showing median 19, IQR 13-25.5)
    B : SLH episode duration (histogram with median 35 min, IQR 20-48.8)
    C : SLH onset time distribution (shows daytime clustering, median 13.8h)
    D : SLH vs LLH duration comparison (side-by-side violin plots on log scale)

    Parameters
    ----------
    results : list of dict
        Per-patient statistics.
    stats : dict
        Cohort statistics from ``compute_cohort_statistics``.
    output_dir : str
        Directory where PDF and PNG files are written.
    """
    apply_publication_style()

    all_slh_durations       = [d for r in results for d in r["slh_durations"]]
    all_llh_durations       = [d for r in results for d in r["llh_durations"]]
    all_slh_onset_hours     = [h for r in results for h in r["slh_onset_hours"]]
    all_night_slh_onset_hours = [h for r in results for h in r.get("night_slh_onset_hours", [])]
    all_night_slh_durations   = [d for r in results for d in r.get("night_slh_durations", [])]
    slh_per_patient         = [r["n_slh"] for r in results]
    night_slh_per_patient   = [r.get("n_night_slh_ep", 0) for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_INCHES, FIGURE_WIDTH_INCHES * 0.90))

    # ── Panel A: SLH Episodes Per Patient ────────────────────────────────────
    ax = axes[0, 0]
    if slh_per_patient:
        bins = np.arange(0, max(slh_per_patient) + 5, 2)
        ax.hist(slh_per_patient, bins=bins, color=WONG_COLORS["orange"],
                alpha=0.75, edgecolor=WONG_COLORS["black"], linewidth=0.5,
                label="SLH (daytime)")
        if any(v > 0 for v in night_slh_per_patient):
            ax.hist(night_slh_per_patient, bins=bins, color=WONG_COLORS["blue"],
                    alpha=0.75, edgecolor=WONG_COLORS["black"], linewidth=0.5,
                    label="SLH-like (night)")

        median_slh = stats["median_slh_per_patient"]

        ax.axvline(median_slh, color=WONG_COLORS["vermillion"],
                   linestyle="--", linewidth=1.8, label=f"Median = {median_slh:.1f}")

        n_night_total = stats.get("funnel_n_night_slh", sum(night_slh_per_patient))
        ax.text(0.97, 0.97,
                f"n = {stats['n_patients']} patients\n"
                f"SLH: {stats['total_slh']:,}  |  Night: {n_night_total:,}",
                transform=ax.transAxes, fontsize=6.5,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                         edgecolor="#cccccc", alpha=0.9))

    ax.set_xlabel("Episodes per patient", fontsize=7.5)
    ax.set_ylabel("Number of patients", fontsize=7.5)
    ax.legend(fontsize=6, loc="upper left", framealpha=0.9)
    add_panel_label(ax, "A")
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.4)

    # ── Panel B: SLH Episode Duration ────────────────────────────────────────
    ax = axes[0, 1]
    if all_slh_durations:
        bins = np.arange(D_MIN_MIN, D_MAX_MIN + 2, 2)
        ax.hist(all_slh_durations, bins=bins, color=WONG_COLORS["bluish_green"],
                alpha=0.75, edgecolor=WONG_COLORS["black"], linewidth=0.5,
                label="SLH (daytime)")
        if all_night_slh_durations:
            ax.hist(all_night_slh_durations, bins=bins, color=WONG_COLORS["sky_blue"],
                    alpha=0.65, edgecolor=WONG_COLORS["black"], linewidth=0.5,
                    label="SLH-like (night)")

        median_dur = stats["median_slh_duration"]

        ax.axvline(median_dur, color=WONG_COLORS["vermillion"],
                   linestyle="--", linewidth=1.8, label=f"Median = {median_dur:.1f} min")

    ax.set_xlabel("Episode duration (min)", fontsize=7.5)
    ax.set_ylabel("Number of episodes", fontsize=7.5)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.9)
    add_panel_label(ax, "B")
    ax.set_xlim(D_MIN_MIN - 2, D_MAX_MIN + 2)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.4)

    # ── Panel C: SLH Onset Time (Daytime Clustering) ─────────────────────────
    ax = axes[1, 0]
    if all_slh_onset_hours or all_night_slh_onset_hours:
        # Shade night periods first (behind histograms)
        ax.axvspan(0, 6,  alpha=0.10, color=WONG_COLORS["sky_blue"], zorder=0)
        ax.axvspan(22, 24, alpha=0.10, color=WONG_COLORS["sky_blue"], zorder=0)
        ax.axvspan(6, 22,  alpha=0.05, color=WONG_COLORS["orange"], zorder=0)

        if all_night_slh_onset_hours:
            ax.hist(all_night_slh_onset_hours, bins=24, range=(0, 24),
                    color=WONG_COLORS["sky_blue"], alpha=0.80,
                    edgecolor=WONG_COLORS["black"], linewidth=0.5,
                    label=f"SLH-like night (n={len(all_night_slh_onset_hours):,})")

        if all_slh_onset_hours:
            ax.hist(all_slh_onset_hours, bins=24, range=(0, 24),
                    color=WONG_COLORS["orange"], alpha=0.75,
                    edgecolor=WONG_COLORS["black"], linewidth=0.5,
                    label=f"SLH daytime (n={len(all_slh_onset_hours):,})")

            median_onset = np.median(all_slh_onset_hours)
            ax.axvline(median_onset, color=WONG_COLORS["vermillion"],
                       linestyle="--", linewidth=1.8,
                       label=f"Median SLH = {median_onset:.1f} h")

        ax.text(14, ax.get_ylim()[1] * 0.97, "C3: daytime",
                fontsize=5.5, ha="center", color="#666666")
        ax.text(3, ax.get_ylim()[1] * 0.97, "night",
                fontsize=5.5, ha="center", color="#555555")
        ax.text(23, ax.get_ylim()[1] * 0.97, "night",
                fontsize=5.5, ha="center", color="#555555")

        ax.legend(fontsize=6, loc="upper left", framealpha=0.9)

    ax.set_xlabel("Time of onset (clock hour)", fontsize=7.5)
    ax.set_ylabel("Number of episodes", fontsize=7.5)
    add_panel_label(ax, "C")
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "24:00"], fontsize=6.5)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.4)

    # ── Panel D: SLH vs LLH Duration Comparison ──────────────────────────────
    ax = axes[1, 1]
    
    # Prepare data for violin plots
    data_to_plot = []
    labels = []
    colors_violin = []
    
    if all_slh_durations:
        data_to_plot.append(all_slh_durations)
        labels.append("SLH")
        colors_violin.append(WONG_COLORS["blue"])
    
    if all_llh_durations:
        data_to_plot.append(all_llh_durations)
        labels.append("LLH")
        colors_violin.append(WONG_COLORS["reddish_purple"])
    
    if data_to_plot:
        positions = np.arange(len(data_to_plot))
        
        # Create violin plots
        parts = ax.violinplot(data_to_plot, positions=positions, widths=0.6,
                              showmeans=False, showmedians=False, showextrema=False)
        
        for i, (pc, color) in enumerate(zip(parts['bodies'], colors_violin)):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor(WONG_COLORS["black"])
            pc.set_linewidth(0.8)
        
        # Add median lines and markers
        for i, data in enumerate(data_to_plot):
            median_val = np.median(data)
            q25 = np.percentile(data, 25)
            q75 = np.percentile(data, 75)
            
            # Median line
            ax.plot([i - 0.25, i + 0.25], [median_val, median_val], 
                   color=WONG_COLORS["vermillion"], linewidth=2.5, zorder=10)
            
            # IQR markers
            ax.plot([i, i], [q25, q75], color=WONG_COLORS["black"], 
                   linewidth=1.2, alpha=0.6, zorder=9)
            
            # Add text annotations
            label_text = f"Median = {median_val:.0f} min\nIQR = {q25:.0f}–{q75:.0f} min"
            ax.text(i, ax.get_ylim()[1] * 0.92, label_text,
                   ha="center", va="top", fontsize=6,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                            edgecolor=colors_violin[i], alpha=0.9))
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yscale("log")
        
        # Add episode counts
        # ax.text(0.03, 0.97, f"n = {len(all_slh_durations):,}", 
        #        transform=ax.transAxes, fontsize=6, va="top", ha="left")
        # if all_llh_durations:
        #     ax.text(0.97, 0.97, f"n = {len(all_llh_durations):,}", 
        #            transform=ax.transAxes, fontsize=6, va="top", ha="right")
    
    ax.set_ylabel("Episode duration (min, log scale)", fontsize=7.5)
    add_panel_label(ax, "D")
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.4, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    fig_base = os.path.join(output_dir, "fig_cohort_distributions")
    for ext in ("pdf", "png"):
        fig.savefig(f"{fig_base}.{ext}", dpi=DPI_SAVE, bbox_inches="tight")
    plt.close(fig)

    print(f"  Figure saved to: {fig_base}.{{pdf,png}}")


def plot_cohort_funnel(stats: dict, output_dir: str) -> None:
    """
    Create a cohort-level episode classification funnel figure.

    Panel A: waterfall funnel (n_raw → SLH).
    Panel B: episodes excluded at each criterion.
    Panel C: C6 OR-crossing breakdown (both / before-only / after-only / no-cross).
    """
    apply_publication_style()

    n_raw = max(stats["funnel_n_raw"], 1)
    n_ref = n_raw

    steps = [
        ("All hyperglycemia episodes",
         stats["funnel_n_raw"], "#555555"),
        (f"C1  Duration [{D_MIN_MIN}–{D_MAX_MIN} min]",
         stats["funnel_n_c1"], FUNNEL_COLORS["C1"]),
        (f"C2  + Data completeness ≥{MIN_DATA_COMPLETENESS*100:.0f}%",
         stats["funnel_n_c2"], FUNNEL_COLORS["C2"]),
        ("C3  + Daytime (06:00–22:00)",
         stats["funnel_n_c3"], FUNNEL_COLORS["C3"]),
        (f"C4  + No hyperglycemia {CONTEXT_WIN_MIN} min before",
         stats["funnel_n_c4"], FUNNEL_COLORS["C4"]),
        (f"C5  + No hyperglycemia {CONTEXT_WIN_MIN} min after",
         stats["funnel_n_c5"], FUNNEL_COLORS["C5"]),
        (f"SLH  + OR-crossing ≤{H_CONTEXT} mg/dL",
         stats["funnel_n_slh"], FUNNEL_COLORS["SLH"]),
    ]

    labels = [s[0] for s in steps]
    values = [s[1] for s in steps]
    colors = [s[2] for s in steps]
    deltas = [None] + [values[i - 1] - values[i] for i in range(1, len(values))]

    excl_keys   = ["funnel_excl_c1", "funnel_excl_c2", "funnel_excl_c3",
                   "funnel_excl_c4", "funnel_excl_c5", "funnel_excl_c6"]
    excl_labels = ["C1", "C2", "C3", "C4", "C5", "C6"]
    excl_colors = [FUNNEL_COLORS["C1"], FUNNEL_COLORS["C2"], FUNNEL_COLORS["C3"],
                   FUNNEL_COLORS["C4"], FUNNEL_COLORS["C5"], FUNNEL_COLORS["SLH"]]
    excl_values = [stats[k] for k in excl_keys]

    fig, (ax_A, ax_B, ax_C) = plt.subplots(
        1, 3,
        figsize=(11.0, 3.70),
        gridspec_kw={"width_ratios": [2.2, 1.0, 1.4], "wspace": 0.55},
    )
    fig.subplots_adjust(left=0.28, right=0.97, top=0.93, bottom=0.12)

    # ── Panel A: Waterfall funnel ─────────────────────────────────────────────
    y     = np.arange(len(steps))[::-1]
    x_max = max(values) * 1.42 if max(values) > 0 else 10
    ax_A.set_xlim(0, x_max)

    bars = ax_A.barh(y, values, color=colors, height=0.52,
                     edgecolor="white", linewidth=0.4, zorder=3)

    for i, (bar, val, pct, delta) in enumerate(
            zip(bars, values, [v / n_ref * 100 for v in values], deltas)):
        ax_A.text(bar.get_width() + x_max * 0.012,
                  bar.get_y() + bar.get_height() / 2,
                  f"n = {val:,}  ({pct:.0f}%)",
                  va="center", fontsize=7, color="#333333")
        if delta is not None and delta > 0:
            ax_A.text(bar.get_width() / 2,
                      bar.get_y() + bar.get_height() + 0.07,
                      f"−{delta:,}",
                      va="bottom", ha="center", fontsize=6,
                      color="#B5121B", fontweight="bold")

    ax_A.set_yticks(y)
    ax_A.set_yticklabels(labels, fontsize=7.5)
    ax_A.spines["top"].set_visible(False)
    ax_A.spines["right"].set_visible(False)
    ax_A.spines["left"].set_linewidth(0.8)
    ax_A.spines["bottom"].set_linewidth(0.8)
    ax_A.tick_params(length=3, width=0.8, labelsize=FONT_SIZE_BASE)
    ax_A.set_xlabel(
        f"Number of surviving episodes  (cohort: {stats['n_patients']} patients)",
        fontsize=FONT_SIZE_BASE + 1, labelpad=4,
    )
    add_panel_label(ax_A, "A")

    # ── Panel B: Exclusions by criterion ─────────────────────────────────────
    y_b   = np.arange(len(excl_labels))[::-1]
    x_max_b = max(excl_values) * 1.45 if max(excl_values) > 0 else 10
    ax_B.set_xlim(0, x_max_b)

    bars_b = ax_B.barh(y_b, excl_values, color=excl_colors, height=0.52,
                       edgecolor="white", linewidth=0.4, zorder=3)

    for bar, val in zip(bars_b, excl_values):
        pct = 100.0 * val / n_raw
        ax_B.text(bar.get_width() + x_max_b * 0.02,
                  bar.get_y() + bar.get_height() / 2,
                  f"{val:,}  ({pct:.1f}%)",
                  va="center", fontsize=6.5, color="#333333")

    ax_B.set_yticks(y_b)
    ax_B.set_yticklabels(excl_labels, fontsize=7.5)
    ax_B.spines["top"].set_visible(False)
    ax_B.spines["right"].set_visible(False)
    ax_B.spines["left"].set_linewidth(0.8)
    ax_B.spines["bottom"].set_linewidth(0.8)
    ax_B.tick_params(length=3, width=0.8, labelsize=FONT_SIZE_BASE)
    ax_B.set_xlabel("Episodes excluded per criterion", fontsize=FONT_SIZE_BASE + 1, labelpad=4)
    add_panel_label(ax_B, "B")

    # ── Panel C: C6 OR-crossing breakdown (vertical barplot) ──────────────────
    n_both        = stats.get("c6_both", 0)
    n_before_only = stats.get("c6_before_only", 0)
    n_after_only  = stats.get("c6_after_only", 0)
    n_no_cross    = stats.get("c6_no_cross", 0)
    n_c6_total    = n_both + n_before_only + n_after_only + n_no_cross

    crossing_cats = [
        ("Both",        n_both,        CROSSING_COLORS["both"]),
        ("Before only", n_before_only, CROSSING_COLORS["before_only"]),
        ("After only",  n_after_only,  CROSSING_COLORS["after_only"]),
        ("No crossing", n_no_cross,    "#BBBBBB"),
    ]

    if n_c6_total > 0:
        cat_labels = [cc[0] for cc in crossing_cats]
        cat_values = [cc[1] for cc in crossing_cats]
        cat_colors = [cc[2] for cc in crossing_cats]

        x_pos = np.arange(len(crossing_cats))
        bars_c = ax_C.bar(x_pos, cat_values, color=cat_colors,
                          edgecolor="white", linewidth=0.6, width=0.65, zorder=3)

        y_top = max(cat_values) if max(cat_values) > 0 else 1
        for bar, val in zip(bars_c, cat_values):
            pct = 100.0 * val / n_c6_total
            ax_C.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + y_top * 0.03,
                      f"{val:,}\n({pct:.0f}%)",
                      ha="center", va="bottom", fontsize=6.0, color="#333333")

        ax_C.set_xticks(x_pos)
        ax_C.set_xticklabels(cat_labels, fontsize=6.5, rotation=20, ha="right")
        ax_C.set_ylabel("Number of episodes", fontsize=FONT_SIZE_BASE + 1, labelpad=4)
        ax_C.set_xlabel(
            f"C6 crossing pattern\n(n = {n_c6_total:,} reaching C6)",
            fontsize=FONT_SIZE_BASE + 1, labelpad=4,
        )
        ax_C.set_ylim(0, y_top * 1.28)
    else:
        ax_C.text(0.5, 0.5, "C6 crossing detail\nnot available\n(rerun episode detection)",
                  ha="center", va="center", transform=ax_C.transAxes,
                  fontsize=7, color="#888888", style="italic")
        ax_C.set_axis_off()

    ax_C.spines["top"].set_visible(False)
    ax_C.spines["right"].set_visible(False)
    ax_C.tick_params(length=3, width=0.8, labelsize=FONT_SIZE_BASE)
    add_panel_label(ax_C, "C")

    fig_base = os.path.join(output_dir, "fig_cohort_funnel")
    for ext in ("pdf", "png"):
        fig.savefig(f"{fig_base}.{ext}", dpi=DPI_SAVE, bbox_inches="tight")
    plt.close(fig)

    print(f"  Funnel figure saved to: {fig_base}.{{pdf,png}}")


def plot_simplified_funnel(stats: dict, output_dir: str) -> None:
    """
    Create a simplified funnel figure with reordered criteria.

    Criteria order: C1 (duration) → C4+C5+C6 (no pre/post-hyper + OR-crossing)
                    → C2 (data completeness) → C3 (daytime) → SLH.

    This order reveals how many night episodes meet all other criteria (night SLH).

    Panel A: waterfall funnel (5 steps).
    Panel B: episodes excluded at each of the 4 criteria, with night SLH annotation.
    """
    apply_publication_style()

    n_raw         = max(stats["funnel_n_raw"], 1)
    n_after_c1    = stats["funnel_n_c1"]
    n_after_c456  = stats.get("funnel_n_c1_c456", stats["funnel_n_slh"])
    n_after_c2    = stats.get("funnel_n_c1_c2_c456", stats["funnel_n_slh"])
    n_slh         = stats["funnel_n_slh"]
    n_night_slh   = stats.get("funnel_n_night_slh", n_after_c2 - n_slh)

    excl_c1   = n_raw        - n_after_c1
    excl_c456 = n_after_c1   - n_after_c456
    excl_c2   = n_after_c456 - n_after_c2
    excl_c3   = n_after_c2   - n_slh       # == n_night_slh

    steps = [
        ("All hyperglycemia episodes",
         n_raw,        "#555555"),
        (f"Duration [{D_MIN_MIN}–{D_MAX_MIN} min]",
         n_after_c1,   FUNNEL_COLORS["C1"]),
        ("Context analysis  No pre/post-hyper · OR-crossing",
         n_after_c456, FUNNEL_COLORS["C4"]),
        (f"Data completeness ≥{MIN_DATA_COMPLETENESS*100:.0f}%",
         n_after_c2,   FUNNEL_COLORS["C2"]),
        ("Daytime (06:00–22:00)  →  SLH",
         n_slh,        FUNNEL_COLORS["SLH"]),
    ]

    labels  = [s[0] for s in steps]
    values  = [s[1] for s in steps]
    colors  = [s[2] for s in steps]
    deltas  = [None] + [values[i - 1] - values[i] for i in range(1, len(values))]

    excl_labels = ["Duration", "Context\nanalysis", "Data\ncompleteness", "Daytime"]
    excl_values = [excl_c1, excl_c456, excl_c2, excl_c3]
    excl_colors = [FUNNEL_COLORS["C1"], FUNNEL_COLORS["C4"],
                   FUNNEL_COLORS["C2"], FUNNEL_COLORS["C3"]]

    fig, (ax_A, ax_B) = plt.subplots(
        1, 2,
        figsize=(7.80, 3.20),
        gridspec_kw={"width_ratios": [2.4, 1.0], "wspace": 0.52},
    )
    fig.subplots_adjust(left=0.38, right=0.97, top=0.93, bottom=0.14)

    # ── Panel A: Waterfall funnel ─────────────────────────────────────────────
    y     = np.arange(len(steps))[::-1]
    x_max = max(values) * 1.44 if max(values) > 0 else 10
    ax_A.set_xlim(0, x_max)

    bars = ax_A.barh(y, values, color=colors, height=0.52,
                     edgecolor="white", linewidth=0.4, zorder=3)

    for i, (bar, val, pct, delta) in enumerate(
            zip(bars, values, [v / n_raw * 100 for v in values], deltas)):
        ax_A.text(bar.get_width() + x_max * 0.012,
                  bar.get_y() + bar.get_height() / 2,
                  f"n = {val:,}  ({pct:.0f}%)",
                  va="center", fontsize=7, color="#333333")
        if delta is not None and delta > 0:
            label_txt = f"−{delta:,}"
            # Annotate C3 exclusion as "night SLH"
            if i == len(steps) - 1 and n_night_slh > 0:
                label_txt += f"\n(night SLH-like: {n_night_slh:,})"
            ax_A.text(bar.get_width() / 2,
                      bar.get_y() + bar.get_height() + 0.07,
                      label_txt,
                      va="bottom", ha="center", fontsize=6,
                      color="#B5121B", fontweight="bold")

    ax_A.set_yticks(y)
    ax_A.set_yticklabels(labels, fontsize=7.5)
    ax_A.spines["top"].set_visible(False)
    ax_A.spines["right"].set_visible(False)
    ax_A.spines["left"].set_linewidth(0.8)
    ax_A.spines["bottom"].set_linewidth(0.8)
    ax_A.tick_params(length=3, width=0.8, labelsize=FONT_SIZE_BASE)
    ax_A.set_xlabel(
        f"Number of surviving episodes  (cohort: {stats['n_patients']} patients)",
        fontsize=FONT_SIZE_BASE + 1, labelpad=4,
    )
    add_panel_label(ax_A, "A")

    # ── Panel B: Exclusions per criterion ────────────────────────────────────
    y_b     = np.arange(len(excl_labels))[::-1]
    x_max_b = max(excl_values) * 1.55 if max(excl_values) > 0 else 10
    ax_B.set_xlim(0, x_max_b)

    bars_b = ax_B.barh(y_b, excl_values, color=excl_colors, height=0.52,
                       edgecolor="white", linewidth=0.4, zorder=3)

    for i, (bar, val) in enumerate(zip(bars_b, excl_values)):
        pct = 100.0 * val / n_raw
        label_txt = f"{val:,}  ({pct:.1f}%)"
        # Annotate C3 bar with night SLH count
        if excl_labels[i].startswith("Daytime") and n_night_slh > 0:
            label_txt += f"\n← night SLH"
        ax_B.text(bar.get_width() + x_max_b * 0.02,
                  bar.get_y() + bar.get_height() / 2,
                  label_txt,
                  va="center", fontsize=6.5, color="#333333")

    ax_B.set_yticks(y_b)
    ax_B.set_yticklabels(excl_labels, fontsize=7.5)
    ax_B.spines["top"].set_visible(False)
    ax_B.spines["right"].set_visible(False)
    ax_B.spines["left"].set_linewidth(0.8)
    ax_B.spines["bottom"].set_linewidth(0.8)
    ax_B.tick_params(length=3, width=0.8, labelsize=FONT_SIZE_BASE)
    ax_B.set_xlabel("Episodes excluded per criterion", fontsize=FONT_SIZE_BASE + 1, labelpad=4)
    add_panel_label(ax_B, "B")

    fig_base = os.path.join(output_dir, "fig_simplified_funnel")
    for ext in ("pdf", "png"):
        fig.savefig(f"{fig_base}.{ext}", dpi=DPI_SAVE, bbox_inches="tight")
    plt.close(fig)

    print(f"  Simplified funnel figure saved to: {fig_base}.{{pdf,png}}")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    """Run the cohort descriptive statistics pipeline."""
    parser = argparse.ArgumentParser(
        description="Compute cohort-level descriptive statistics from pre-computed episodes"
    )
    parser.add_argument("--episodes_dir", default=str(EPISODES_DIR),
                        help="Directory containing episode CSV files (default: config EPISODES_PATH)")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR),
                        help="Output directory (default: Results/Cohort_Descriptive_Stats)")
    parser.add_argument("--filter_csv", default=str(FILTER_STATUS_CSV),
                        help="Patient filter CSV")
    parser.add_argument("--no_filter", action="store_true",
                        help="Process all patients (ignore filter)")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  Cohort Descriptive Statistics (SLH/LLH Episodes)")
    print("=" * 70)
    print(f"  Episodes: {args.episodes_dir}")
    print(f"  Output:   {args.output_dir}")
    print("=" * 70 + "\n")

    if not os.path.isdir(args.episodes_dir):
        print(f"  [ERROR] Episodes directory not found: {args.episodes_dir}")
        print("  Run episode_detection.py first to generate episode files.")
        return

    filtered = None if args.no_filter else load_filtered_patient_list(args.filter_csv)
    all_pids  = discover_episode_files(args.episodes_dir)

    if not all_pids:
        print(f"[ERROR] No episode files found in {args.episodes_dir}")
        return

    pids = [p for p in all_pids if p in filtered] if filtered is not None else all_pids
    print(f"  Cohort: {len(pids)} patients\n")

    results = []
    n_ok = n_fail = 0

    for pid in pids:
        try:
            result = process_patient_statistics(pid)
            if result is not None:
                results.append(result)
                n_ok += 1
        except Exception as exc:
            print(f"  [{pid}] ERROR: {exc}")
            n_fail += 1

    print(f"\n  Processed: {n_ok}  |  Failed: {n_fail}\n")

    if not results:
        print("[ERROR] No results to analyze")
        return

    stats = compute_cohort_statistics(results)

    os.makedirs(args.output_dir, exist_ok=True)

    # Save per-patient table (exclude raw lists stored only for cohort pooling)
    _exclude_keys = {"slh_durations", "llh_durations", "slh_onset_hours",
                     "night_slh_onset_hours", "night_slh_durations"}
    df_out = pd.DataFrame([
        {k: v for k, v in r.items() if k not in _exclude_keys}
        for r in results
    ])
    csv_path = os.path.join(args.output_dir, "cohort_per_patient.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"  Per-patient data saved to: {csv_path}")

    print("\n  Generating outputs...")
    summary_path = os.path.join(args.output_dir, "summary_statistics.txt")
    summary = generate_summary_table(stats, summary_path)
    plot_distributions(results, stats, args.output_dir)
    plot_cohort_funnel(stats, args.output_dir)
    plot_simplified_funnel(stats, args.output_dir)

    print("\n" + summary)
    print("\n" + "=" * 70)
    print(f"  Done! Output saved to: {os.path.abspath(args.output_dir)}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()