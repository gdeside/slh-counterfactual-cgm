"""
src/preprocessing/load_data.py
===============================

CGM data loading utilities.

Utility functions for loading raw and preprocessed CGM datasets.
All loaders handle missing files safely, normalise column names
(case-insensitive), return clean sorted DataFrames, and return
``None`` on failure for robust pipeline integration.

Functions
---------
_find_column
    Locate a column by name (case-insensitive) from a list of candidates.
load_raw_time_column
    Load the time column only from a raw patient Excel file.
load_patient_raw_data
    Load and minimally clean raw CGM data (time + glucose) for one patient.
load_window_data
    Load a preprocessed CGM window from an Excel file.
load_patient_window
    Load, clean, and validate the preprocessed window for one patient.
prepare_valid_data
    Filter a window DataFrame to valid (non-flagged) glucose rows.
discover_patients
    List all patient IDs that have a preprocessed window available.

Author
------
Guillaume Deside

Date
----
2026-03-28
"""

# =============================================================================
# STANDARD LIBRARIES
# =============================================================================
import os
import glob
from pathlib import Path

# =============================================================================
# THIRD-PARTY LIBRARIES
# =============================================================================
import pandas as pd

# =============================================================================
# PROJECT IMPORTS
# =============================================================================
from src.config import (
    RAW_PATH,
    PREPROCESSING_PATH,
    TIME_COL,
    GLUCOSE_COL,
    COL_FLAG_TIME_ERROR,
    COL_FLAG_OUTLIER_GLUCOSE,
)


# =============================================================================
# INTERNAL HELPERS
# =============================================================================
def _find_column(columns, candidates) -> str | None:
    """
    Find a column name (case-insensitive) from a list of candidates.

    Parameters
    ----------
    columns : list-like
        Available column names
    candidates : tuple or list
        Possible column names to match

    Returns
    -------
    str or None
        Matching column name, or None if not found
    """
    return next((c for c in columns if c.lower() in candidates), None)


# =============================================================================
# RAW DATA LOADERS
# =============================================================================
def load_raw_time_column(patient_id: str) -> pd.DataFrame | None:
    """
    Load raw Excel file for a patient (time column only).

    Parameters
    ----------
    patient_id : str
        Patient identifier (e.g., 'A001')

    Returns
    -------
    pd.DataFrame or None
        DataFrame with a single column ['time'], sorted chronologically,
        or None if file is missing or invalid
    """
    filepath = os.path.join(RAW_PATH, f"{patient_id}.xlsx")

    if not os.path.exists(filepath):
        return None

    try:
        df = pd.read_excel(filepath)

        time_col = _find_column(df.columns, ("time",))
        if time_col is None:
            return None

        df = df[[time_col]].rename(columns={time_col: TIME_COL})

        df[TIME_COL] = pd.to_datetime(
            df[TIME_COL],
            dayfirst=True,
            errors="coerce",
        )

        return (
            df.dropna(subset=[TIME_COL])
              .sort_values(TIME_COL)
              .reset_index(drop=True)
        )

    except Exception:
        return None


def load_patient_raw_data(
    patient_id: str,
    raw_path: str,
) -> pd.DataFrame | None:
    """
    Load and minimally clean raw CGM data for one patient.

    Parameters
    ----------
    patient_id : str
        Patient identifier (e.g., 'A001')
    raw_path : str
        Directory containing raw Excel files

    Returns
    -------
    pd.DataFrame or None
        Clean DataFrame with columns [TIME_COL, GLUCOSE_COL],
        sorted by time, or None if loading fails
    """
    filepath = os.path.join(raw_path, f"{patient_id}.xlsx")

    if not os.path.exists(filepath):
        return None

    try:
        df = pd.read_excel(filepath)

        # Detect columns
        time_col = _find_column(df.columns, ("time",))
        glucose_col = _find_column(df.columns, ("glycemie", "glycemia", "glucose"))

        if time_col is None or glucose_col is None:
            return None

        # Rename to standard schema
        df = df.rename(columns={
            time_col: TIME_COL,
            glucose_col: GLUCOSE_COL,
        })

        # Parse time
        df[TIME_COL] = pd.to_datetime(
            df[TIME_COL],
            dayfirst=True,
            errors="coerce",
        )

        # Parse glucose
        df[GLUCOSE_COL] = pd.to_numeric(
            df[GLUCOSE_COL],
            errors="coerce",
        )

        # Drop invalid rows
        df = df.dropna(subset=[TIME_COL, GLUCOSE_COL])

        return (
            df[[TIME_COL, GLUCOSE_COL]]
            .sort_values(TIME_COL)
            .reset_index(drop=True)
        )

    except Exception:
        return None


# =============================================================================
# PREPROCESSED DATA LOADERS
# =============================================================================
def load_window_data(filepath: str) -> pd.DataFrame | None:
    """
    Load preprocessed CGM window data from Excel.

    Parameters
    ----------
    filepath : str
        Path to window Excel file

    Returns
    -------
    pd.DataFrame or None
        DataFrame with standardized columns [TIME_COL, GLUCOSE_COL],
        or None if file is invalid
    """
    try:
        df = pd.read_excel(filepath)

        # Normalize column names
        column_map = {}

        for col in df.columns:
            col_lower = col.lower()

            if col_lower == "time":
                column_map[col] = TIME_COL
            elif col_lower in ("glycemie", "glycemia", "glucose"):
                column_map[col] = GLUCOSE_COL

        df = df.rename(columns=column_map)

        # Validate required columns
        if TIME_COL not in df.columns or GLUCOSE_COL not in df.columns:
            return None

        return df

    except Exception:
        return None
    

def load_patient_window(patient_id: str) -> pd.DataFrame | None:
    """
    Load and clean preprocessed CGM window data for one patient.

    Parameters
    ----------
    patient_id : str
        Patient identifier (e.g., 'A001')

    Returns
    -------
    pd.DataFrame or None
        Clean DataFrame with columns [TIME_COL, GLUCOSE_COL],
        filtered for valid data, sorted by time, and containing at least
        100 observations, or None if loading fails or data is insufficient
    """
    patient_dir = os.path.join(PREPROCESSING_PATH, patient_id)
    if not os.path.isdir(patient_dir):
        return None
    
    files = glob.glob(os.path.join(patient_dir, f"{patient_id}_win1_*.xlsx"))
    if not files:
        return None
    
    try:
        df = pd.read_excel(files[0])
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        
        # Remove flagged rows
        for col in (COL_FLAG_TIME_ERROR, COL_FLAG_OUTLIER_GLUCOSE):
            if col in df.columns:
                df = df[~df[col].fillna(False).astype(bool)]
        
        # Clean glucose data
        df[GLUCOSE_COL] = pd.to_numeric(df[GLUCOSE_COL], errors="coerce")
        df = df.dropna(subset=[GLUCOSE_COL])
        df = df.sort_values(TIME_COL).reset_index(drop=True)
        
        return df if len(df) >= 100 else None
        
    except Exception as e:
        print(f"error: {patient_id}: {e}")
        return None
    


def prepare_valid_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract valid glucose measurements (exclude flagged rows).
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw window data
        
    Returns
    -------
    pd.DataFrame
        Valid data with time and glucose columns only
    """
    mask = pd.Series(True, index=df.index)
    
    if COL_FLAG_TIME_ERROR in df.columns:
        mask &= ~df[COL_FLAG_TIME_ERROR].astype(bool)
    
    if COL_FLAG_OUTLIER_GLUCOSE in df.columns:
        mask &= ~df[COL_FLAG_OUTLIER_GLUCOSE].astype(bool)
    
    return df[mask][[TIME_COL, GLUCOSE_COL]].dropna().sort_values(TIME_COL).copy()


def discover_raw_patients(raw_path=None) -> list[str]:
    """
    List the patients that have a raw CGM file: every ``<patient_id>.xlsx`` in ``RAW_PATH``.

    Patient IDs are the file names (any format: ``A001``, ``OD_001``, ...); the cohort is whatever is on disk,
    so there is no hard-coded number of patients. Excel lock files (``~$...``) and hidden files are ignored.

    Returns
    -------
    list of str
        Sorted patient identifiers.
    """
    raw_path = RAW_PATH if raw_path is None else raw_path
    return sorted(
        p.stem for p in Path(raw_path).glob("*.xlsx")
        if not p.name.startswith(("~$", "."))
    )


def resolve_patients(requested=None, raw_path=None) -> list[str]:
    """
    Patients to process: all patients found in the raw folder, or the ``requested`` subset.

    Raises
    ------
    SystemExit
        If no raw file exists, or a requested ID has no raw file.
    """
    available = discover_raw_patients(raw_path)
    if not available:
        raise SystemExit(
            f"No raw CGM file (<patient_id>.xlsx) found in {RAW_PATH if raw_path is None else raw_path}. "
            "Check CGM_DATA_ROOT and the Data/Raw folder."
        )
    if not requested:
        return available
    unknown = [p for p in requested if p not in available]
    if unknown:
        raise SystemExit(f"No raw file for: {', '.join(unknown)}. Available: {', '.join(available[:10])}"
                         f"{' ...' if len(available) > 10 else ''}")
    return sorted(set(requested))


def discover_patients(input_dir: str) -> list[str]:
    """
    Discover all patient IDs that have at least one preprocessed window file.

    Parameters
    ----------
    input_dir : str
        Base preprocessing directory (``PREPROCESSING_PATH``).

    Returns
    -------
    list of str
        Sorted list of patient identifiers (any sub-directory name, e.g. ``A001`` or ``OD_001``)
        for which a ``{pid}_win1_*.xlsx`` file exists.
    """
    all_dirs = [
        os.path.basename(d) for d in glob.glob(os.path.join(input_dir, "*"))
        if os.path.isdir(d) and not os.path.basename(d).startswith((".", "_"))
    ]
    
    patients = []
    for pid in all_dirs:
        # Check if patient has a window file
        files = glob.glob(os.path.join(input_dir, pid, f"{pid}_win1_*.xlsx"))
        if files:
            patients.append(pid)
    
    return sorted(patients)