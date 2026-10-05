import pandas as pd
import os

from src.config.constants import TIME_COL

from src.config.paths import (
    SENSITIVITY_LININTERP_SIGNALS,
    SENSITIVITY_LININTERP_DATA,
    COHORT_SENSITIVITY_CSV,
)

def load_patient_signals(patient_id: str) -> pd.DataFrame:
    """
    Load saved signal data (original + counterfactuals) for patient.

    Returns
    -------
    pd.DataFrame with columns:
        - time
        - glucose_original
        - glucose_ctf_delta_0, glucose_ctf_delta_15, etc.
    or None if file not found
    """
    signals_path = os.path.join(
        SENSITIVITY_LININTERP_SIGNALS, f"{patient_id}_signals.csv"
    )

    if not os.path.exists(signals_path):
        return None

    df = pd.read_csv(signals_path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    return df


def load_sensitivity_results(patient_id: str) -> pd.DataFrame:
    """
    Load sensitivity analysis results CSV for a patient.

    Returns
    -------
    pd.DataFrame or None if file not found
    """
    csv_path = os.path.join(
        SENSITIVITY_LININTERP_DATA, f"{patient_id}_sensitivity.csv"
    )

    if not os.path.exists(csv_path):
        return None

    return pd.read_csv(csv_path)


def load_cohort_sensitivity_results(offset: int) -> pd.DataFrame:
    """
    Load cohort sensitivity results CSV filtered to a single offset.

    Parameters
    ----------
    offset : int
        Zone-expansion offset in minutes to filter for.

    Returns
    -------
    pd.DataFrame
    """
    path = str(COHORT_SENSITIVITY_CSV)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Sensitivity results not found: {path}\n"
            "Run sensitivity_analysis.py --patient_id ALL first."
        )
    df = pd.read_csv(path)
    df = df[df["offset"] == offset].copy()
    print(f"  Loaded {len(df)} patients from sensitivity results (δ={offset} min)")
    return df


def extract_delta_values(signals_df: pd.DataFrame) -> list:
    """Extract available delta values from signal dataframe columns."""
    delta_values = []

    for col in signals_df.columns:
        if col.startswith("glucose_ctf_delta_"):
            delta_str = col.replace("glucose_ctf_delta_", "")
            try:
                delta_values.append(int(delta_str))
            except ValueError:
                continue

    return sorted(delta_values)