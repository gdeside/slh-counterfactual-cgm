"""
scripts/examples/convert_brist1d.py
===================================

Turns the original BrisT1D-Open files (``example_data/BrisT1D/raw/P01.csv`` ...) into the raw layout read by the
pipeline and by the notebooks: one file per participant with only the glucose.

What is changed (the converted files are NOT the original BrisT1D files):
  * columns: everything but the timestamp and the glucose ``bg`` (insulin, carbs, heart rate, steps, device ...)
    is dropped; rows without a glucose value are dropped
  * units: ``bg`` mmol/L -> mg/dL (x 18.0182), rounded to 0.1
  * time grid: each reading is moved to the nearest 5-minute mark (the original timestamps drift by a few minutes);
    readings that land on the same mark are averaged. No value is interpolated: 15-minute sensors
    (FreeStyle Libre 2) stay 15-minute.

Usage
-----
    python3 -m scripts.examples.convert_brist1d [--input_dir example_data/BrisT1D/raw]
                                                [--out_root example_data/BrisT1D/converted]
    (or: make brist1d)

Then run the notebooks / the pipeline on it::

    export CGM_DATA_ROOT=$PWD/example_data/BrisT1D/converted
    make pipeline

Output
------
    <out_root>/Data/Raw/<participant>.xlsx   columns ``time``, ``glycemie`` (mg/dL)
"""
import argparse
from pathlib import Path

import pandas as pd

from src.utils.unit_conversion import MGDL_PER_MMOLL

GRID = "5min"


def convert_file(path: Path) -> pd.DataFrame:
    """Original BrisT1D file -> DataFrame[time, glycemie] (mg/dL, 5-min grid, glucose only)."""
    df = pd.read_csv(path, usecols=["timestamp", "bg"], parse_dates=["timestamp"]).dropna(subset=["bg"])
    df["time"] = df["timestamp"].dt.round(GRID)
    df["glycemie"] = df["bg"] * MGDL_PER_MMOLL
    out = df.groupby("time", as_index=False)["glycemie"].mean()
    out["glycemie"] = out["glycemie"].round(1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", type=Path, default=Path("example_data/BrisT1D/raw"))
    ap.add_argument("--out_root", type=Path, default=Path("example_data/BrisT1D/converted"))
    args = ap.parse_args()

    files = sorted(args.input_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No .csv file in {args.input_dir} - put the original BrisT1D-Open files there.")
    raw_dir = args.out_root / "Data" / "Raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        out = convert_file(f)
        out.to_excel(raw_dir / f"{f.stem}.xlsx", index=False)
        print(f"{f.stem}: {len(out)} readings, {out['time'].min():%Y-%m-%d} -> {out['time'].max():%Y-%m-%d}")
    print(f"Wrote {len(files)} participants to {raw_dir}")


if __name__ == "__main__":
    main()
