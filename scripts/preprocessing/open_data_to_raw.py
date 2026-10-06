"""
scripts/preprocessing/open_data_to_raw.py
=========================================

Turns the released open dataset (``cohort_open_data.csv``: ``patient_id``, ``time_h``,
``glucose_mgdl`` or ``glucose_mmoll``) into the raw layout read by the pipeline:
``<CGM_DATA_ROOT>/Data/Raw/<patient_id>.xlsx`` with the columns ``time`` and ``glycemie`` (mg/dL).

Open-data time is relative (hours from monitoring start); it is mapped onto an arbitrary,
synthetic origin (2000-01-01 00:00), so no real date is involved. Glucose in mmol/L is
converted back to mg/dL.

Then run the pipeline on it::

    export CGM_DATA_ROOT=/path/to/folder
    python3 -m scripts.preprocessing.open_data_to_raw --input /path/to/cohort_open_data.csv
    make pipeline

Output
------
    <CGM_DATA_ROOT>/Data/Raw/<patient_id>.xlsx  (columns time, glycemie in mg/dL)
"""
import argparse
from pathlib import Path

import pandas as pd

from src.config import RAW_PATH
from src.utils.unit_conversion import MGDL_PER_MMOLL

ORIGIN = pd.Timestamp("2000-01-01 00:00")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=Path("Results/Open_Data/cohort_open_data.csv"),
                    help="cohort_open_data.csv (default: Results/Open_Data/cohort_open_data.csv under cwd)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if "glucose_mgdl" in df.columns:
        glucose = df["glucose_mgdl"]
    elif "glucose_mmoll" in df.columns:
        glucose = df["glucose_mmoll"] * MGDL_PER_MMOLL
    else:
        raise SystemExit("Expected a 'glucose_mgdl' or 'glucose_mmoll' column.")

    out = pd.DataFrame({
        "patient_id": df["patient_id"],
        "time": ORIGIN + pd.to_timedelta(df["time_h"], unit="h"),
        "glycemie": glucose,
    })
    RAW_PATH.mkdir(parents=True, exist_ok=True)
    for pid, g in out.groupby("patient_id"):
        g.drop(columns="patient_id").sort_values("time").round({"glycemie": 1}).to_excel(
            RAW_PATH / f"{pid}.xlsx", index=False)
    print(f"Wrote {out['patient_id'].nunique()} patients to {RAW_PATH}")


if __name__ == "__main__":
    main()
