"""
scripts/local_slh_analysis_cohort.py
=====================================

Batch local SLH region analysis across the filtered patient cohort.

For every patient that passes the study inclusion filter this script runs the
full six-step local-SLH pipeline (see scripts/supplementary_code docstring)
and writes per-patient artefacts to ``RESULT_PATH / <patient_id> /``:

Output files per patient
------------------------
fig_recurrence_dominance_<pid>.pdf
    Three-panel figure: SLH/LLH recurrence curves (top), smoothed dominance
    index with detected peaks and regions (middle), AGP with region overlays
    (bottom).

fig_glucose_zoom_<pid>.pdf
    Per-region glucose zoom (±CONTEXT_H hours around each region): grey daily
    shadows, orange SLH episode traces, blue all-data AGP lines, orange
    SLH-only AGP lines, green region shading.

fig_agp_counterfactual_<pid>.pdf
    AGP comparison per counterfactual scenario (original + per-region + combined).

metrics_<pid>.csv
    Wide-format table with one row per scenario (original, R1, R2, …, combined)
    and columns for all CGM metrics plus #SLH removed and delta vs. original.

A summary file ``cohort_local_slh_summary.csv`` is written at ``RESULT_PATH``
root with one row per patient, including the number of regions detected and
key delta metrics for the combined (all-regions-removed) scenario.

Pipeline steps
--------------
1. Load filtered patient list (``FILTER_STATUS_CSV``).
2. For each patient: load preprocessed CGM window and detect/classify episodes.
3. Compute SLH and LLH recurrence profiles at ``RESOLUTION_MIN`` resolution.
4. Build masked + Gaussian-smoothed dominance index; apply threshold.
5. Detect peaks and expand region boundaries (adaptive or fixed width).
6. Build per-region counterfactual signals and compute CGM metrics.
7. Save figures and CSV.

Parameters
----------
All algorithm parameters are defined in the ``# ── Parameters`` section below.
They intentionally mirror the notebook so that notebook and batch results are
directly comparable.

Usage
-----
Run from the project root::

    python scripts/local_slh_analysis_cohort.py

Author
------
Guillaume Deside

Date
----
2026-06-30
"""

# =============================================================================
# STANDARD LIBRARY
# =============================================================================
import os
import sys
import traceback
import warnings
from pathlib import Path

# =============================================================================
# THIRD-PARTY
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for batch runs
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# =============================================================================
# PROJECT IMPORTS — make src/ importable when called from project root
# =============================================================================
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    TIME_COL, GLUCOSE_COL,
    COL_EPISODE_START, COL_EPISODE_END,
    COL_EPISODE_DURATION, COL_EPISODE_CLASSIFICATION,
    CLASSIFICATION_SLH, CLASSIFICATION_LLH,
    FILTER_STATUS_CSV, PREPROCESSING_PATH,
)
from src.config.paths import RESULTS_PATH
from src.preprocessing.load_data import (
    load_patient_window, discover_patients,
)
from src.filtering.load_filtering import load_filtered_patient_list
from src.episodes.detection import detect_hyperglycemia_episodes
from src.episodes.criteria import classify_episodes
from src.agp.compute_agp import (
    compute_agp_profile,
    compute_recurrence_by_classification,
)
from src.counterfactual.build_counterfactual import (
    build_full_counterfactual_signal,
)
from src.counterfactual.counterfactual_config import (
    SELECTED_OFFSET_MIN, KERNEL_MIN,
)
from src.metrics.metrics import (
    compute_basic_statistics,
    compute_time_in_range_metrics,
    METRIC_MEAN_GLUCOSE, METRIC_SD, METRIC_CV,
    METRIC_TIR70180, METRIC_TAR180, METRIC_TBR70,
)

# =============================================================================
# ── Parameters — edit here to change pipeline behaviour ──────────────────────
# =============================================================================

# Output root: one sub-folder per patient will be created inside
RESULT_PATH = RESULTS_PATH / "Local_SLH_Analysis"

# Time resolution for recurrence profiles (minutes). Must divide 1440.
RESOLUTION_MIN = 15

# Dominance threshold mode.
#   "relative"  →  τ = THRESHOLD_VALUE × max(dom̃)   [recommended]
#   "absolute"  →  τ = THRESHOLD_VALUE
THRESHOLD_MODE  = "relative"
THRESHOLD_VALUE = 0.75

# Minimum SLH recurrence (% of days) below which a slot is masked to dom = 0.
MIN_SLH_RECURRENCE_PCT = 2.0

# Gaussian smoothing kernel width in time slots (0 = no smoothing).
# At 15 min resolution: SMOOTH_SIGMA_SLOTS=2 → 30 min kernel.
SMOOTH_SIGMA_SLOTS = 2

# Minimum time between two detected peaks (minutes).
MIN_PEAK_DISTANCE_MIN = 60

# Region boundary mode: "auto" (adaptive) or a number (fixed half-width in min).
REGION_WIDTH_MODE = "auto"

# Context window for glucose zoom plots (hours before/after each region).
CONTEXT_H = 2.0

# Counterfactual parameters (mirror counterfactual_config defaults).
OFFSET_MIN  = SELECTED_OFFSET_MIN   # anchor search half-window (min)
_KERNEL_MIN = KERNEL_MIN            # rolling-mean smoothing (min)

# Colour palette (colour-blind safe, mirrors notebook).
C_SLH    = "#E69F00"
C_LLH    = "#0072B2"
C_DOM    = "#CC79A7"
C_REGION = "#009E73"
C_GLUC   = "#56B4E9"

# Metrics to include in the output CSV.
DISPLAY_METRICS = [
    METRIC_MEAN_GLUCOSE, METRIC_SD, METRIC_CV,
    METRIC_TIR70180, METRIC_TAR180, METRIC_TBR70,
]
DISPLAY_LABELS = {
    METRIC_MEAN_GLUCOSE: "Mean glucose (mg/dL)",
    METRIC_SD:           "SD (mg/dL)",
    METRIC_CV:           "CV (%)",
    METRIC_TIR70180:     "TIR 70-180 (%)",
    METRIC_TAR180:       "TAR >180 (%)",
    METRIC_TBR70:        "TBR <70 (%)",
}


# =============================================================================
# HELPERS
# =============================================================================

def _minutes_to_hhmm(minutes: float) -> str:
    """Convert a float number of minutes since midnight to 'HH:MM' string."""
    h = int(minutes) // 60
    m = int(minutes) % 60
    return f"{h:02d}:{m:02d}"


def _compute_metrics(glucose_values: np.ndarray) -> dict:
    """
    Compute all display metrics for a glucose array.

    Returns a flat dict with metric names as keys and float values.
    TIR / TAR / TBR are converted to percentages (×100).
    """
    stats = compute_basic_statistics(pd.Series(glucose_values))
    tir   = compute_time_in_range_metrics(pd.Series(glucose_values))
    # Convert fractions → percentages for the three range metrics
    tir[METRIC_TIR70180] = round(tir[METRIC_TIR70180] * 100, 1)
    tir[METRIC_TAR180]   = round(tir[METRIC_TAR180]   * 100, 1)
    tir[METRIC_TBR70]    = round(tir[METRIC_TBR70]    * 100, 1)
    return {**stats, **tir}


def _onset_in_region(ts: pd.Timestamp, start_min: float, end_min: float) -> bool:
    """Return True if the time-of-day of *ts* falls in [start_min, end_min]."""
    m = ts.hour * 60 + ts.minute + ts.second / 60
    return start_min <= m <= end_min


def _auto_region_bounds(
    signal: np.ndarray,
    peak_slot: int,
    threshold: float,
    resolution_min: int,
) -> tuple[float, float]:
    """
    Expand outward from *peak_slot* while signal >= threshold.

    Returns (start_min, end_min) in minutes since midnight.
    """
    lo = peak_slot
    while lo > 0 and signal[lo - 1] >= threshold:
        lo -= 1
    hi = peak_slot
    n  = len(signal)
    while hi < n - 1 and signal[hi + 1] >= threshold:
        hi += 1
    return lo * resolution_min, (hi + 1) * resolution_min


def _slh_glucose_profile(
    df: pd.DataFrame,
    slh_df: pd.DataFrame,
    start_min: float,
    end_min: float,
    resolution_min: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-slot glucose median and mean using only SLH episode time points
    whose onset falls inside [start_min, end_min].

    Returns (median_array, mean_array) of length n_slots=1440/resolution_min.
    NaN for slots with no data.
    """
    n_slots = int(1440 / resolution_min)
    slot_glucose: list[list[float]] = [[] for _ in range(n_slots)]

    for _, ep in slh_df.iterrows():
        if not _onset_in_region(ep[COL_EPISODE_START], start_min, end_min):
            continue
        mask = (df[TIME_COL] >= ep[COL_EPISODE_START]) & (df[TIME_COL] <= ep[COL_EPISODE_END])
        for _, row in df[mask].iterrows():
            frac = row[TIME_COL].hour + row[TIME_COL].minute / 60
            slot_glucose[int(frac * 60 / resolution_min) % n_slots].append(row[GLUCOSE_COL])

    med = np.array([np.median(v) if v else np.nan for v in slot_glucose])
    mn  = np.array([np.mean(v)   if v else np.nan for v in slot_glucose])
    return med, mn


# =============================================================================
# FIGURE 1 — three-panel recurrence / dominance / AGP
# =============================================================================

def _fig_recurrence_dominance(
    patient_id: str,
    df: pd.DataFrame,
    r_slh: np.ndarray,
    r_llh: np.ndarray,
    dom_masked: np.ndarray,
    dom_smooth: np.ndarray,
    effective_threshold: float,
    peak_slots: np.ndarray,
    regions: list[dict],
    agp: pd.DataFrame,
    resolution_min: int,
) -> plt.Figure:
    """
    Build the three-panel recurrence + dominance + AGP figure.

    Parameters mirror the notebook §6 cell.

    Returns
    -------
    matplotlib.figure.Figure
    """
    slot_hours = np.arange(len(r_slh)) * resolution_min / 60
    agp_hours  = np.arange(len(agp))   * resolution_min / 60
    peak_hours = peak_slots * resolution_min / 60

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(
        f"SLH-predominant region detection — Patient {patient_id}",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: recurrence curves ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(slot_hours, r_slh, color=C_SLH, lw=1.8, label="SLH recurrence")
    ax.plot(slot_hours, r_llh, color=C_LLH, lw=1.8, label="LLH recurrence")
    ax.set_ylabel("Recurrence (% of days)", fontsize=10)
    ax.set_title("SLH and LLH episode recurrence by time of day", fontsize=10)
    ax.legend(fontsize=9); ax.set_ylim(bottom=0); ax.grid(True, alpha=0.3)
    for r in regions:
        ax.axvspan(r["start_min"] / 60, r["end_min"] / 60, alpha=0.15, color=C_REGION)

    # ── Panel 2: dominance index ──────────────────────────────────────────────
    ax = axes[1]
    ax.fill_between(slot_hours, dom_smooth, alpha=0.3, color=C_DOM)
    ax.plot(slot_hours, dom_smooth, color=C_DOM, lw=1.8,
            label=r"$\widetilde{\mathrm{dom}}(t)$ smoothed")
    ax.plot(slot_hours, dom_masked, color=C_DOM, lw=0.8, ls="--", alpha=0.6,
            label=r"$\mathrm{dom}(t)$ masked")
    ax.axhline(effective_threshold, ls="--", color="gray", lw=1.2,
               label=f"Threshold ({effective_threshold:.3f})")
    ax.plot(peak_hours, dom_smooth[peak_slots], "v",
            color=C_REGION, ms=9, zorder=5, label="Detected peaks")
    for r in regions:
        ax.axvspan(r["start_min"] / 60, r["end_min"] / 60, alpha=0.15, color=C_REGION)
        ax.annotate(
            f"R{r['region_id']}\n{_minutes_to_hhmm(r['centre_min'])}",
            xy=(r["centre_min"] / 60, r["peak_dom"]),
            xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=8.5, color=C_REGION, fontweight="bold",
        )
    ax.set_ylabel("SLH dominance index", fontsize=10)
    ax.set_title(
        r"$\mathrm{dom}(t)=r_\mathrm{SLH}/(r_\mathrm{SLH}+r_\mathrm{LLH})$",
        fontsize=10,
    )
    ax.set_ylim(0, min(1.1, dom_smooth.max() * 1.35))
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Panel 3: AGP with region overlays ────────────────────────────────────
    ax = axes[2]
    ax.fill_between(agp_hours, agp["q25"], agp["q75"], alpha=0.25, color=C_GLUC)
    ax.fill_between(agp_hours, agp["q5"],  agp["q95"], alpha=0.12, color=C_GLUC)
    ax.plot(agp_hours, agp["median"], color=C_GLUC, lw=2, label="Median glucose")
    ax.axhline(180, ls="--", color="#CC4444", lw=1, alpha=0.7, label="180 mg/dL")
    ax.axhline(70,  ls="--", color="#4444CC", lw=1, alpha=0.7, label="70 mg/dL")
    for r in regions:
        ax.axvspan(
            r["start_min"] / 60, r["end_min"] / 60,
            alpha=0.18, color=C_REGION,
            label=f"R{r['region_id']}: {_minutes_to_hhmm(r['start_min'])}"
                  f"–{_minutes_to_hhmm(r['end_min'])}",
        )
    ax.set_ylabel("Glucose (mg/dL)", fontsize=10)
    ax.set_title("AGP with detected SLH-predominant regions", fontsize=10)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 2))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], fontsize=8)
    axes[-1].set_xlabel("Time of day", fontsize=10)
    fig.tight_layout()
    return fig


# =============================================================================
# FIGURE 2 — per-region glucose zoom
# =============================================================================

def _fig_glucose_zoom(
    patient_id: str,
    df: pd.DataFrame,
    slh_df: pd.DataFrame,
    regions: list[dict],
    agp: pd.DataFrame,
    resolution_min: int,
) -> plt.Figure:
    """
    Build per-region glucose zoom plots (±CONTEXT_H hours of context).

    Each subplot shows: grey daily shadows, orange SLH episode segments,
    blue all-data median/mean, orange SLH-only median/mean, green region shading.

    Returns
    -------
    matplotlib.figure.Figure
    """
    agp_hours_arr = np.arange(len(agp)) * resolution_min / 60

    # Pre-group daily glucose traces once (avoids repeated filtering)
    df = df.copy()
    df["_date"]      = df[TIME_COL].dt.date
    df["_hour_frac"] = df[TIME_COL].dt.hour + df[TIME_COL].dt.minute / 60
    daily_groups     = {d: g for d, g in df.groupby("_date")}

    n_r   = len(regions)
    ncols = min(n_r, 3)
    nrows = int(np.ceil(n_r / ncols)) if n_r else 1
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4.5 * nrows),
        squeeze=False, sharey=False,
    )
    fig.suptitle(
        f"Glucose profile zoom (±{int(CONTEXT_H*60)} min context)"
        f" — Patient {patient_id}",
        fontsize=12, fontweight="bold",
    )

    for idx, r in enumerate(regions):
        ax = axes[idx // ncols][idx % ncols]

        x_lo = max(0.0,  r["start_min"] / 60 - CONTEXT_H)
        x_hi = min(24.0, r["end_min"]   / 60 + CONTEXT_H)

        # Grey shadows: one trace per day
        for day_df in daily_groups.values():
            win = day_df[(day_df["_hour_frac"] >= x_lo) & (day_df["_hour_frac"] <= x_hi)]
            if not win.empty:
                ax.plot(win["_hour_frac"], win[GLUCOSE_COL],
                        color="#AAAAAA", lw=0.7, alpha=0.35, zorder=1)

        # Orange SLH episode segments whose onset falls in the region
        def _in_region(t):
            m = t.hour * 60 + t.minute + t.second / 60
            return r["start_min"] <= m <= r["end_min"]

        for _, ep in slh_df.iterrows():
            if not _in_region(ep[COL_EPISODE_START]):
                continue
            ep_data = df[
                (df[TIME_COL] >= ep[COL_EPISODE_START]) &
                (df[TIME_COL] <= ep[COL_EPISODE_END])
            ]
            if not ep_data.empty:
                ax.plot(ep_data["_hour_frac"], ep_data[GLUCOSE_COL],
                        color=C_SLH, lw=1.0, alpha=0.5, zorder=2)

        # AGP slot indices for the zoomed window
        s_slot_ext = int(x_lo * 60 / resolution_min)
        e_slot_ext = min(int(np.ceil(x_hi * 60 / resolution_min)), len(agp) - 1)
        x_agp = agp_hours_arr[s_slot_ext: e_slot_ext + 1]

        # Blue: all-data median and mean
        ax.plot(x_agp, agp["median"].values[s_slot_ext: e_slot_ext + 1],
                color=C_GLUC, lw=2.2, zorder=4, label="All — median")
        ax.plot(x_agp, agp["mean"].values[s_slot_ext: e_slot_ext + 1],
                color=C_GLUC, lw=1.5, ls="--", zorder=4, label="All — mean")

        # Orange: SLH-only median and mean (only within region)
        slh_med, slh_mn = _slh_glucose_profile(
            df, slh_df, r["start_min"], r["end_min"], resolution_min,
        )
        s_slot_r = int(r["start_min"] / resolution_min)
        e_slot_r = min(int(np.ceil(r["end_min"] / resolution_min)), len(agp) - 1)
        x_r = agp_hours_arr[s_slot_r: e_slot_r + 1]
        ax.plot(x_r, slh_med[s_slot_r: e_slot_r + 1],
                color=C_SLH, lw=2.2, zorder=5, label="SLH only — median")
        ax.plot(x_r, slh_mn[s_slot_r: e_slot_r + 1],
                color=C_SLH, lw=1.5, ls="--", zorder=5, label="SLH only — mean")

        # Green region shading
        ax.axvspan(r["start_min"] / 60, r["end_min"] / 60,
                   alpha=0.12, color=C_REGION, zorder=0, label="Region")
        ax.axvline(r["start_min"] / 60, color=C_REGION, lw=1.0, ls=":", zorder=3)
        ax.axvline(r["end_min"]   / 60, color=C_REGION, lw=1.0, ls=":", zorder=3)

        ax.axhline(180, ls=":", color="#CC4444", lw=0.8, alpha=0.7)
        ax.axhline(70,  ls=":", color="#4444CC", lw=0.8, alpha=0.7)

        # X-axis: HH:MM ticks
        x_ticks = np.arange(np.floor(x_lo), np.ceil(x_hi) + 0.5, 0.5)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(
            [f"{int(h):02d}:{int((h % 1)*60):02d}" for h in x_ticks],
            fontsize=7, rotation=45,
        )
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel("Time of day", fontsize=8)
        ax.set_ylabel("Glucose (mg/dL)", fontsize=8)
        ax.set_title(
            f"R{r['region_id']}: {_minutes_to_hhmm(r['start_min'])}"
            f"–{_minutes_to_hhmm(r['end_min'])}",
            fontsize=9,
        )
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.25)

    # Hide unused axes
    for idx in range(n_r, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    return fig


# =============================================================================
# FIGURE 3 — AGP counterfactual comparison
# =============================================================================

def _agp_from_array(
    glucose_array: np.ndarray,
    time_series: pd.Series,
    resolution_min: int,
) -> pd.DataFrame:
    """
    Compute an AGP profile (median, mean, q5/q25/q75/q95) from a raw numpy
    signal paired with a datetime series.

    Returns a DataFrame with one row per time slot.
    """
    n_slots = int(1440 / resolution_min)
    slot_vals: list[list[float]] = [[] for _ in range(n_slots)]
    for t, g in zip(time_series, glucose_array):
        slot = int((t.hour * 60 + t.minute) / resolution_min) % n_slots
        slot_vals[slot].append(g)

    rows = []
    for vals in slot_vals:
        arr = np.array(vals) if vals else np.array([np.nan])
        rows.append({
            "median": np.nanmedian(arr),
            "mean":   np.nanmean(arr),
            "q5":     np.nanpercentile(arr, 5),
            "q25":    np.nanpercentile(arr, 25),
            "q75":    np.nanpercentile(arr, 75),
            "q95":    np.nanpercentile(arr, 95),
        })
    return pd.DataFrame(rows)


def _fig_agp_counterfactual(
    patient_id: str,
    df: pd.DataFrame,
    scenarios: dict[str, np.ndarray],
    regions: list[dict],
    resolution_min: int,
) -> plt.Figure:
    """
    Build AGP comparison figure: one subplot per scenario (original +
    per-region + combined), each overlaid on the original AGP.

    Parameters
    ----------
    scenarios : dict
        Keys are scenario labels; values are glucose arrays (same length as df).

    Returns
    -------
    matplotlib.figure.Figure
    """
    hours = np.arange(int(1440 / resolution_min)) * resolution_min / 60
    agp_orig = _agp_from_array(
        df[GLUCOSE_COL].values, df[TIME_COL], resolution_min,
    )

    region_colors = ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7"]
    scenario_items = [
        (label, arr, region_colors[i % len(region_colors)])
        for i, (label, arr) in enumerate(scenarios.items())
        if label != "original"
    ]
    n = len(scenario_items)
    fig, axes = plt.subplots(
        n, 1, figsize=(14, 3.5 * n), sharex=True,
    )
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"AGP counterfactual comparison — Patient {patient_id}",
        fontsize=12, fontweight="bold",
    )

    def _plot_panel(ax, agp_cf, color, title):
        ax.fill_between(hours, agp_orig["q25"], agp_orig["q75"],
                        alpha=0.15, color=C_GLUC)
        ax.plot(hours, agp_orig["median"], color=C_GLUC, lw=1.5,
                ls="--", label="Original median")
        ax.fill_between(hours, agp_cf["q25"], agp_cf["q75"],
                        alpha=0.25, color=color)
        ax.plot(hours, agp_cf["median"], color=color, lw=2,
                label="Counterfactual median")
        ax.axhline(180, ls=":", color="#CC4444", lw=0.8, alpha=0.7)
        ax.axhline(70,  ls=":", color="#4444CC", lw=0.8, alpha=0.7)
        ax.set_ylabel("Glucose (mg/dL)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.25)

    for ax, (label, arr, color) in zip(axes, scenario_items):
        agp_cf = _agp_from_array(arr, df[TIME_COL], resolution_min)
        # Shade regions belonging to this scenario
        for r in regions:
            if label.startswith(f"R{r['region_id']}") or label == "combined":
                ax.axvspan(r["start_min"] / 60, r["end_min"] / 60,
                           alpha=0.10, color=color)
        _plot_panel(ax, agp_cf, color, label)

    axes[-1].set_xlim(0, 24)
    axes[-1].set_xticks(range(0, 25, 2))
    axes[-1].set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], fontsize=8)
    axes[-1].set_xlabel("Time of day", fontsize=10)
    fig.tight_layout()
    return fig


# =============================================================================
# CORE: single-patient analysis
# =============================================================================

def analyse_patient(patient_id: str, out_dir: Path) -> dict | None:
    """
    Run the full local-SLH pipeline for one patient and save artefacts.

    Parameters
    ----------
    patient_id : str
        Patient identifier (e.g., 'A001').
    out_dir : Path
        Directory where figures and CSV are written.

    Returns
    -------
    dict or None
        Summary row (patient-level metrics) to append to the cohort CSV,
        or None if the patient cannot be processed.
    """
    # ── 1. Load CGM data ──────────────────────────────────────────────────────
    df = load_patient_window(patient_id)
    if df is None:
        print(f"  [{patient_id}] No window data — skip")
        return None

    df = df[[TIME_COL, GLUCOSE_COL]].dropna().sort_values(TIME_COL).reset_index(drop=True)
    n_days = (df[TIME_COL].dt.date.max() - df[TIME_COL].dt.date.min()).days + 1

    # ── 2. Detect and classify hyperglycaemia episodes ────────────────────────
    raw_eps  = detect_hyperglycemia_episodes(df)
    if not raw_eps:
        print(f"  [{patient_id}] No episodes detected — skip")
        return None
    eps_df = classify_episodes(df, raw_eps)
    slh_df = eps_df[eps_df[COL_EPISODE_CLASSIFICATION] == CLASSIFICATION_SLH].copy()
    if slh_df.empty:
        print(f"  [{patient_id}] No SLH episodes — skip")
        return None

    # ── 3. Recurrence profiles ────────────────────────────────────────────────
    start_date = df[TIME_COL].dt.date.min()
    rec = compute_recurrence_by_classification(
        eps_df, RESOLUTION_MIN, n_days, start_date,
    )
    r_slh = rec.get(CLASSIFICATION_SLH, np.zeros(int(1440 / RESOLUTION_MIN)))
    r_llh = rec.get(CLASSIFICATION_LLH, np.zeros(int(1440 / RESOLUTION_MIN)))

    # ── 4. Dominance index: mask → smooth ────────────────────────────────────
    total = r_slh + r_llh
    with np.errstate(invalid="ignore", divide="ignore"):
        dom_raw = np.where(total > 0, r_slh / total, 0.0)
    dom_masked = np.where(r_slh >= MIN_SLH_RECURRENCE_PCT, dom_raw, 0.0)
    dom_smooth = (
        gaussian_filter1d(dom_masked, sigma=SMOOTH_SIGMA_SLOTS, mode="wrap")
        if SMOOTH_SIGMA_SLOTS > 0 else dom_masked.copy()
    )
    # Threshold is calibrated on the smoothed signal (stable across patients),
    # but peak *locations* are detected on the masked signal so that fine peaks
    # blurred out by smoothing (e.g. a 14h cluster hidden in a broad saddle)
    # are still found. Region *boundaries* are still walked on dom_smooth so
    # that noisy single-slot spikes in dom_masked don't inflate region width.
    dom_max = dom_smooth.max()
    if dom_max == 0:
        print(f"  [{patient_id}] Dominance is zero — skip")
        return None

    effective_threshold = (
        THRESHOLD_VALUE * dom_max if THRESHOLD_MODE == "relative"
        else THRESHOLD_VALUE
    )

    # ── 5. Peak detection on dom_masked (fine signal) ────────────────────────
    # Using dom_masked rather than dom_smooth preserves local maxima that the
    # Gaussian kernel merges into a single broad hump. The height filter still
    # uses effective_threshold (derived from dom_smooth) so spurious low-value
    # noise spikes in dom_masked are rejected.
    min_dist_slots = max(1, int(np.ceil(MIN_PEAK_DISTANCE_MIN / RESOLUTION_MIN)))
    peak_slots, _ = find_peaks(
        dom_masked,
        height=effective_threshold,
        distance=min_dist_slots,
    )
    if len(peak_slots) == 0:
        print(f"  [{patient_id}] No peaks detected — skip")
        return None

    # ── 6. Region boundaries ──────────────────────────────────────────────────
    regions: list[dict] = []
    for rid, ps in enumerate(peak_slots, start=1):
        if isinstance(REGION_WIDTH_MODE, str) and REGION_WIDTH_MODE == "auto":
            s_min, e_min = _auto_region_bounds(
                dom_smooth, ps, effective_threshold, RESOLUTION_MIN,
            )
        else:
            half_w = float(REGION_WIDTH_MODE)
            s_min  = max(0.0, ps * RESOLUTION_MIN - half_w)
            e_min  = min(1440.0, ps * RESOLUTION_MIN + half_w)

        centre_min = ps * RESOLUTION_MIN
        regions.append({
            "region_id":  rid,
            "start_min":  s_min,
            "end_min":    e_min,
            "centre_min": centre_min,
            "peak_dom":   float(dom_smooth[ps]),
        })

    # ── 7. AGP profile ────────────────────────────────────────────────────────
    agp = compute_agp_profile(df, RESOLUTION_MIN)

    # ── 8. Counterfactual signals ─────────────────────────────────────────────
    # Estimate CGM timestep (most common interval)
    dt_min = df[TIME_COL].diff().dt.total_seconds().div(60).dropna()
    timestep_min = float(dt_min.mode().iloc[0]) if not dt_min.empty else 5.0

    slh_eps_list = [
        {
            "onset_time":  row[COL_EPISODE_START],
            "offset_time": row[COL_EPISODE_END],
        }
        for _, row in slh_df.iterrows()
    ]

    # Original metrics
    orig_glucose = df[GLUCOSE_COL].values
    orig_metrics = _compute_metrics(orig_glucose)

    scenarios: dict[str, np.ndarray] = {"original": orig_glucose}
    metric_rows: list[dict] = [{
        "scenario":        "Original",
        "n_slh_removed":   0,
        **{DISPLAY_LABELS[m]: orig_metrics[m] for m in DISPLAY_METRICS},
    }]

    for r in regions:
        # Filter SLH episodes whose onset is in this region
        eps_in = [
            e for e in slh_eps_list
            if _onset_in_region(e["onset_time"], r["start_min"], r["end_min"])
        ]
        if not eps_in:
            cf_arr = orig_glucose.copy()
        else:
            cf_arr = build_full_counterfactual_signal(
                df, eps_in, OFFSET_MIN, timestep_min,
            )

        label = (
            f"R{r['region_id']}: {_minutes_to_hhmm(r['start_min'])}"
            f"–{_minutes_to_hhmm(r['end_min'])}"
        )
        scenarios[label] = cf_arr
        cf_metrics = _compute_metrics(cf_arr)
        row = {
            "scenario":      label,
            "n_slh_removed": len(eps_in),
        }
        for m in DISPLAY_METRICS:
            row[DISPLAY_LABELS[m]]           = cf_metrics[m]
            row[f"delta_{DISPLAY_LABELS[m]}"] = round(cf_metrics[m] - orig_metrics[m], 2)
        metric_rows.append(row)

    # Combined scenario: remove SLH from all regions simultaneously
    all_eps_in_regions = [
        e for e in slh_eps_list
        if any(_onset_in_region(e["onset_time"], r["start_min"], r["end_min"])
               for r in regions)
    ]
    cf_combined = (
        build_full_counterfactual_signal(df, all_eps_in_regions, OFFSET_MIN, timestep_min)
        if all_eps_in_regions else orig_glucose.copy()
    )
    scenarios["combined"] = cf_combined
    cf_comb_metrics = _compute_metrics(cf_combined)
    comb_row = {
        "scenario":      "combined",
        "n_slh_removed": len(all_eps_in_regions),
    }
    for m in DISPLAY_METRICS:
        comb_row[DISPLAY_LABELS[m]]           = cf_comb_metrics[m]
        comb_row[f"delta_{DISPLAY_LABELS[m]}"] = round(cf_comb_metrics[m] - orig_metrics[m], 2)
    metric_rows.append(comb_row)

    # ── 9. Save figures ───────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1 = _fig_recurrence_dominance(
        patient_id, df, r_slh, r_llh,
        dom_masked, dom_smooth, effective_threshold,
        peak_slots, regions, agp, RESOLUTION_MIN,
    )
    fig1.savefig(out_dir / f"fig_recurrence_dominance_{patient_id}.pdf",
                 dpi=150, bbox_inches="tight")
    plt.close(fig1)

    fig2 = _fig_glucose_zoom(patient_id, df, slh_df, regions, agp, RESOLUTION_MIN)
    fig2.savefig(out_dir / f"fig_glucose_zoom_{patient_id}.pdf",
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)

    fig3 = _fig_agp_counterfactual(patient_id, df, scenarios, regions, RESOLUTION_MIN)
    fig3.savefig(out_dir / f"fig_agp_counterfactual_{patient_id}.pdf",
                 dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # ── 10. Save per-patient metrics CSV ──────────────────────────────────────
    pd.DataFrame(metric_rows).to_csv(
        out_dir / f"metrics_{patient_id}.csv", index=False,
    )

    # ── 11. Build summary row for cohort CSV ──────────────────────────────────
    summary = {
        "patient_id":  patient_id,
        "n_days":      n_days,
        "n_slh_total": len(slh_df),
        "n_regions":   len(regions),
        "region_ids":  ";".join(
            f"R{r['region_id']}({_minutes_to_hhmm(r['start_min'])}"
            f"-{_minutes_to_hhmm(r['end_min'])})"
            for r in regions
        ),
        "threshold_mode":  THRESHOLD_MODE,
        "effective_tau":   round(effective_threshold, 4),
    }
    # Add combined-scenario deltas to the summary row
    for m in DISPLAY_METRICS:
        summary[f"combined_delta_{DISPLAY_LABELS[m]}"] = comb_row[f"delta_{DISPLAY_LABELS[m]}"]

    print(
        f"  [{patient_id}] OK — {len(regions)} region(s), "
        f"{len(slh_df)} SLH eps, "
        f"DTIR={comb_row.get(f'delta_{DISPLAY_LABELS[METRIC_TIR70180]}', 'n/a'):+.1f}%pt"
    )
    return summary


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    """
    Run the batch local-SLH analysis for all filtered patients.

    Iterates over the filtered patient list, calls ``analyse_patient`` for
    each, collects summary rows, and writes ``cohort_local_slh_summary.csv``.
    """
    warnings.filterwarnings("ignore", category=UserWarning)

    RESULT_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {RESULT_PATH}\n")

    # Load the filtered patient list
    filtered = load_filtered_patient_list(str(FILTER_STATUS_CSV))
    if filtered is None:
        # Fallback: use all patients that have a preprocessed window
        print("Falling back to all available patients (no filter CSV).")
        filtered = set(discover_patients(str(PREPROCESSING_PATH)))

    all_pids = sorted(filtered)
    print(f"Patients to process: {len(all_pids)}\n")

    summary_rows: list[dict] = []
    failed: list[str] = []

    for i, pid in enumerate(all_pids, start=1):
        print(f"[{i}/{len(all_pids)}] {pid}")
        out_dir = RESULT_PATH / pid
        try:
            row = analyse_patient(pid, out_dir)
            if row is not None:
                summary_rows.append(row)
        except Exception:
            print(f"  [{pid}] ERROR — see traceback below")
            traceback.print_exc()
            failed.append(pid)

    # Write cohort-level summary
    summary_csv = RESULT_PATH / "cohort_local_slh_summary.csv"
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        print(f"\nCohort summary saved: {summary_csv} ({len(summary_rows)} patients)")
    else:
        print("\nNo patients produced output.")

    if failed:
        print(f"\nFailed patients ({len(failed)}): {', '.join(failed)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
