"""
src/episodes/load_episodes.py
=============================

I/O utilities for CGM episode data.

Provides three categories of helpers:

* **Preprocessing** – ``prepare_episode_data`` cleans and sorts a raw CGM
  DataFrame so it is ready for episode detection (type coercion, flag
  removal, time-sorting).

* **Per-patient loading** – ``load_patient_episodes`` reads a patient's
  episode CSV from ``EPISODES_PATH`` and parses timestamp columns.

* **Cohort-level loading** – ``load_episode_summary`` reads the aggregated
  summary CSV produced by the detection pipeline; ``discover_episode_files``
  scans the episodes directory and returns available patient IDs.

Column name constants (``COL_EPISODE_START``, ``COL_EPISODE_END``) and path
constants (``EPISODES_PATH``, ``EPISODES_SUMMARY_CSV``) are imported from
``src.config`` to keep a single source of truth.
"""

import os
import glob
import pandas as pd

from src.config import (
    TIME_COL,
    GLUCOSE_COL,
    COL_FLAG_TIME_ERROR,
    COL_FLAG_OUTLIER_GLUCOSE,
    COL_EPISODE_START,
    COL_EPISODE_END,
    EPISODES_PATH,
    EPISODES_SUMMARY_CSV,
)


def prepare_episode_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare CGM data for episode detection.

    Steps:

    1. Parse and coerce time and glucose columns.
    2. Drop rows with missing values.
    3. Keep only valid rows (exclude flagged time errors and outliers).
    4. Sort by time.

    Parameters
    ----------
    df : pd.DataFrame
        Raw window data.

    Returns
    -------
    pd.DataFrame
        Cleaned and sorted data ready for episode detection.
    """
    df = df.copy()

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], dayfirst=True, errors="coerce")
    df[GLUCOSE_COL] = pd.to_numeric(df[GLUCOSE_COL], errors="coerce")

    df = df.dropna(subset=[TIME_COL, GLUCOSE_COL])

    if COL_FLAG_TIME_ERROR in df.columns:
        df = df[~df[COL_FLAG_TIME_ERROR].astype(bool)]

    if COL_FLAG_OUTLIER_GLUCOSE in df.columns:
        df = df[~df[COL_FLAG_OUTLIER_GLUCOSE].astype(bool)]

    df = df.sort_values(TIME_COL).reset_index(drop=True)

    return df


def load_patient_episodes(patient_id: str) -> pd.DataFrame | None:
    """
    Load episode classification data for a patient.

    Parameters
    ----------
    patient_id : str
        Patient identifier.

    Returns
    -------
    pd.DataFrame or None
        Episode data with parsed timestamps, or None if the file is not found.
    """
    filepath = os.path.join(str(EPISODES_PATH), f"{patient_id}_episodes.csv")
    if not os.path.exists(filepath):
        return None

    try:
        df = pd.read_csv(filepath)
        df[COL_EPISODE_START] = pd.to_datetime(df[COL_EPISODE_START])
        df[COL_EPISODE_END] = pd.to_datetime(df[COL_EPISODE_END])
        return df
    except Exception:
        return None


def load_episode_summary() -> pd.DataFrame:
    """
    Load cohort episode summary (n_SLH per patient).

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the summary CSV does not exist.
    """
    path = str(EPISODES_SUMMARY_CSV)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Episodes summary not found: {path}\n"
            "Run episode_detection.py first."
        )
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} patients from episodes summary")
    return df


def discover_episode_files(episodes_dir: str) -> list[str]:
    """
    Discover all patient IDs that have an episode file in *episodes_dir*.

    Parameters
    ----------
    episodes_dir : str
        Directory containing ``*_episodes.csv`` files.

    Returns
    -------
    list of str
        Sorted patient IDs.
    """
    files = glob.glob(os.path.join(episodes_dir, "*_episodes.csv"))
    pids = [os.path.basename(f).replace("_episodes.csv", "") for f in files]
    return sorted(pids)
