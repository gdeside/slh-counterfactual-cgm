"""
src/metrics/load_metrics.py
============================

Utilities for loading pre-computed patient metrics from CSV files.
"""

import os
import pandas as pd
from src.config import COL_HAS_WINDOW

def load_patient_metrics(csv_path):
    """
    Load patient metrics from CSV file.
    
    Parameters
    ----------
    csv_path : str
        Path to metrics CSV file
        
    Returns
    -------
    pd.DataFrame
        Metrics for patients with valid windows
        
    Raises
    ------
    FileNotFoundError
        If metrics CSV does not exist
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"metrics_all_patients.csv not found:\n  {csv_path}\n"
            "Run cgm_metrics.py first."
        )
    
    df = pd.read_csv(csv_path)
    df = df[df[COL_HAS_WINDOW] == True].copy().reset_index(drop=True)
    
    print(f"  Loaded {len(df)} patients with valid metrics")
    
    return df