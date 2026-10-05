from src.config import PATIENT_ID_COL
import csv
import os

def load_filtered_patient_list(filter_csv: str) -> set | None:
    """
    Load list of filtered patients from CSV file.

    Parameters
    ----------
    filter_csv : str
        Path to filter status CSV

    Returns
    -------
    set or None
        Set of patient IDs, or None if file not found
    """
    if not os.path.isfile(filter_csv):
        print(f"  [WARN] Filter CSV not found at: {filter_csv}")
        return None

    with open(filter_csv, newline="", encoding="utf-8") as f:
        pids = [r[PATIENT_ID_COL] for r in csv.DictReader(f)
                if r.get("selected", "").strip().lower() == "true"]

    print(f"  Filter CSV: {len(pids)} selected patients")
    return set(pids)