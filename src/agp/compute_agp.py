"""
src/agp/compute_agp.py
======================

Ambulatory Glucose Profile (AGP) and episode-recurrence computation.

Functions
---------
compute_agp_profile
    Compute the median daily glucose profile (with percentiles) from raw CGM data.
compute_recurrence_by_duration
    Compute per-slot episode recurrence rates stratified by episode duration.
compute_recurrence_by_classification
    Compute per-slot episode recurrence rates stratified by SLH/LLH classification.
"""

from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd

from src.config import (
    TIME_COL,
    GLUCOSE_COL,
    CLASSIFICATION_SLH,
    CLASSIFICATION_LLH,
    COL_EPISODE_START,
    COL_EPISODE_END,
    COL_EPISODE_DURATION,
    COL_EPISODE_CLASSIFICATION,
)

# ── AGP computation parameters ────────────────────────────────────────────────
AGP_TIME_SLOT_TOLERANCE = 0.4   # Fraction of resolution for nearest-point matching

# ── Episode duration bins (minutes) ──────────────────────────────────────────
DURATION_BINS   = [0, 15, 30, 45, 60, 75, 90, np.inf]
DURATION_LABELS = ["0–15", "15–30", "30–45", "45–60", "60–75", "75–90", "90+"]

CLASSIFICATION_LABELS = [CLASSIFICATION_SLH, CLASSIFICATION_LLH]


def compute_agp_profile(df: pd.DataFrame, resolution_min: int) -> pd.DataFrame:
    """
    Compute Ambulatory Glucose Profile (median daily profile with percentiles).

    Parameters
    ----------
    df : pd.DataFrame
        Valid glucose data with TIME_COL and GLUCOSE_COL columns.
    resolution_min : int
        Time slot resolution in minutes.

    Returns
    -------
    pd.DataFrame
        AGP profile with columns:
        - time_str: time of day (HH:MM)
        - median, mean: central tendency
        - q5, q25, q75, q95: percentiles
        - count: number of observations per slot
    """
    n_slots = int(24 * 60 / resolution_min)
    tolerance = AGP_TIME_SLOT_TOLERANCE * resolution_min

    df = df.copy()
    df["_date"] = df[TIME_COL].dt.date
    unique_dates = df["_date"].unique()

    # Collect glucose values for each time slot across all days
    slot_values = {slot: [] for slot in range(n_slots)}

    for date in unique_dates:
        day_data = df[df["_date"] == date].copy()
        day_min = pd.Timestamp(datetime.combine(date, time(0, 0)))

        for slot in range(n_slots):
            slot_time = day_min + pd.Timedelta(minutes=slot * resolution_min)
            day_data["_diff"] = (day_data[TIME_COL] - slot_time).abs().dt.total_seconds() / 60

            if day_data.empty:
                slot_values[slot].append(np.nan)
                continue

            closest_idx = day_data["_diff"].idxmin()
            min_diff = day_data.loc[closest_idx, "_diff"]

            if min_diff <= tolerance:
                slot_values[slot].append(day_data.loc[closest_idx, GLUCOSE_COL])
            else:
                slot_values[slot].append(np.nan)

    # Compute statistics for each slot
    records = []
    for slot in range(n_slots):
        values = np.array(slot_values[slot], dtype=float)
        valid = values[~np.isnan(values)]
        count = len(valid)

        slot_time_str = (datetime.min + timedelta(minutes=slot * resolution_min)).strftime("%H:%M")

        records.append({
            "time_str": slot_time_str,
            "median": np.median(valid) if count > 0 else np.nan,
            "mean":   np.mean(valid)   if count > 0 else np.nan,
            "q5":     np.percentile(valid,  5) if count > 0 else np.nan,
            "q25":    np.percentile(valid, 25) if count > 0 else np.nan,
            "q75":    np.percentile(valid, 75) if count > 0 else np.nan,
            "q95":    np.percentile(valid, 95) if count > 0 else np.nan,
            "count":  count,
        })

    return pd.DataFrame(records)


def compute_recurrence_by_duration(episodes: pd.DataFrame,
                                   resolution_min: int,
                                   n_days: int,
                                   start_date) -> dict:
    """
    Compute recurrence by episode duration bins.

    For each duration bin and time slot, calculates:
    (number of distinct days with an episode) / n_days × 100

    Parameters
    ----------
    episodes : pd.DataFrame
        Episode data.
    resolution_min : int
        Time slot resolution in minutes.
    n_days : int
        Total number of days in the analysis window.
    start_date : datetime.date
        First date of the analysis window.

    Returns
    -------
    dict
        Mapping from duration label to array of recurrence % values (length n_slots).
    """
    n_slots = int(24 * 60 / resolution_min)
    occurrence = {label: np.zeros(n_slots) for label in DURATION_LABELS}

    episodes = episodes.copy()
    episodes["_bin"] = pd.cut(
        episodes[COL_EPISODE_DURATION],
        bins=DURATION_BINS,
        labels=DURATION_LABELS,
        right=False,
    )

    for _, episode in episodes.iterrows():
        label = episode["_bin"]
        if pd.isna(label):
            continue

        current_time = episode[COL_EPISODE_START]
        seen_keys = set()

        while current_time <= episode[COL_EPISODE_END]:
            day_offset = (current_time.date() - start_date).days
            slot = min(
                int((current_time.hour * 60 + current_time.minute) / resolution_min),
                n_slots - 1,
            )

            key = (day_offset, slot)
            if key not in seen_keys:
                occurrence[label][slot] += 1
                seen_keys.add(key)

            current_time += timedelta(minutes=resolution_min)

    for label in DURATION_LABELS:
        occurrence[label] = occurrence[label] / n_days * 100

    return occurrence


def compute_recurrence_by_classification(episodes: pd.DataFrame,
                                         resolution_min: int,
                                         n_days: int,
                                         start_date) -> dict:
    """
    Compute recurrence by SLH/LLH classification.

    Parameters
    ----------
    episodes : pd.DataFrame
        Episode data with a classification column.
    resolution_min : int
        Time slot resolution in minutes.
    n_days : int
        Total number of days in the analysis window.
    start_date : datetime.date
        First date of the analysis window.

    Returns
    -------
    dict
        Mapping from classification label to array of recurrence % values (length n_slots).
    """
    n_slots = int(24 * 60 / resolution_min)
    occurrence = {label: np.zeros(n_slots) for label in CLASSIFICATION_LABELS}

    for _, episode in episodes.iterrows():
        classification = episode[COL_EPISODE_CLASSIFICATION]
        if classification not in CLASSIFICATION_LABELS:
            continue

        current_time = episode[COL_EPISODE_START]
        seen_keys = set()

        while current_time <= episode[COL_EPISODE_END]:
            day_offset = (current_time.date() - start_date).days
            slot = min(
                int((current_time.hour * 60 + current_time.minute) / resolution_min),
                n_slots - 1,
            )

            key = (day_offset, slot)
            if key not in seen_keys:
                occurrence[classification][slot] += 1
                seen_keys.add(key)

            current_time += timedelta(minutes=resolution_min)

    for label in CLASSIFICATION_LABELS:
        occurrence[label] = occurrence[label] / n_days * 100

    return occurrence
