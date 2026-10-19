"""
scripts/examples/convert_shanghai_t1dm.py
=========================================

Put the ShanghaiT1DM example data into the format and units of this project.

Source (not redistributed with this repository)
    Zhu, Jinhao (2022). Diabetes Datasets-ShanghaiT1DM and ShanghaiT2DM. figshare. Dataset.
    https://doi.org/10.6084/m9.figshare.20444397.v3        (licence: CC BY 4.0)

THE DATA ARE MODIFIED. The converted files are NOT the original ShanghaiT1DM files; this script
  * keeps only the time stamp and the CGM column of each patient file (insulin, meals, ketones, CBG and the
    clinical summary table are dropped);
  * converts glucose to mg/dL if the file is in mmol/L (x 18.0182; the figshare v3 files are already in mg/dL);
  * resamples the 15-min CGM to the 5-min grid expected by this project, by linear interpolation in time,
    from the first reading of each patient; gaps longer than --max_gap minutes (default 20) are NOT filled;
  * names the columns ``time`` and ``glycemie``, one CSV per patient (the file name is the patient ID).
Interpolated points are not measurements. The episode counts obtained on these files are an illustration of the
code, not a result about ShanghaiT1DM. The original authors are not responsible for the modified data.

Folder layout (created if missing)
    example_data/ShanghaiT1DM/raw/        <- put the ORIGINAL figshare files here (one .xlsx per patient, e.g.
                                              1001_0_20210730.xlsx; the Shanghai_T1DM_Summary.xlsx is ignored)
    example_data/ShanghaiT1DM/converted/  <- written by this script; read by ``make example``

Usage
-----
    python3 -m scripts.examples.convert_shanghai_t1dm [--input_dir DIR] [--out_dir DIR] [--max_gap 20]
    (or: make shanghai_convert)  then:  make shanghai_example

Output (in --out_dir)
    <patient_id>.csv          columns time, glycemie (mg/dL), 5-min grid
    MODIFIED_DATA_NOTICE.md  source, licence and the list of changes above
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.unit_conversion import MGDL_PER_MMOLL

DEFAULT_IN = Path("example_data/ShanghaiT1DM/raw")
DEFAULT_OUT = Path("example_data/ShanghaiT1DM/converted")
TARGET_STEP_MIN = 5
DEFAULT_MAX_GAP_MIN = 20          # 15-min sampling + tolerance: longer gaps are left empty
SOURCE = ("Zhu, Jinhao (2022). Diabetes Datasets-ShanghaiT1DM and ShanghaiT2DM. figshare. Dataset. "
          "https://doi.org/10.6084/m9.figshare.20444397.v3 (CC BY 4.0)")
NOTICE = f"""These files are MODIFIED from the ShanghaiT1DM dataset:
{SOURCE}

Changes made by scripts/examples/convert_shanghai_t1dm.py (counterfactual-cgm-slh):
- only the time stamp and the CGM column were kept (insulin, meals, ketones, CBG, clinical table dropped);
- glucose in mg/dL (converted from mmol/L when needed);
- resampled from 15 min to a 5-min grid by linear interpolation (gaps > {{max_gap}} min left empty);
- columns renamed time / glycemie.
Interpolated points are not measurements. The original authors do not endorse these modified data.
"""


def find_cgm_column(columns) -> tuple:
    """(column, unit) of the CGM column of a ShanghaiT1DM patient file, e.g. 'CGM (mg / dl)'."""
    for c in columns:
        name = str(c).lower().replace(" ", "")
        if name.startswith("cgm"):
            return c, ("mmol/L" if "mmol" in name else "mg/dL")
    raise ValueError(f"no CGM column found. Columns: {list(columns)}")


def to_project_format(raw: pd.DataFrame, max_gap_min: float = DEFAULT_MAX_GAP_MIN) -> pd.DataFrame:
    """ShanghaiT1DM patient sheet -> DataFrame ``time`` / ``glycemie`` (mg/dL) on a 5-min grid."""
    gcol, unit = find_cgm_column(raw.columns)
    tcol = next((c for c in raw.columns if str(c).strip().lower() in ("date", "time", "timestamp")), None)
    if tcol is None:
        raise ValueError(f"no time column found. Columns: {list(raw.columns)}")
    s = pd.Series(pd.to_numeric(raw[gcol], errors="coerce").to_numpy(),
                  index=pd.to_datetime(raw[tcol], errors="coerce"))
    s = s[s.index.notna()].dropna()
    s = s[~s.index.duplicated()].sort_index()
    if unit == "mmol/L":
        s = s * MGDL_PER_MMOLL
    if len(s) < 2:
        raise ValueError("fewer than 2 CGM readings")

    grid = pd.date_range(s.index.min(), s.index.max(), freq=f"{TARGET_STEP_MIN}min")
    filled = s.reindex(s.index.union(grid)).interpolate("time", limit_area="inside").reindex(grid)
    # a grid point is kept only if the real readings around it are at most max_gap_min apart
    t = pd.Series(s.index, index=s.index)
    prev_t = t.reindex(grid, method="ffill")
    next_t = t.reindex(grid, method="bfill")
    gap_min = (next_t - prev_t).dt.total_seconds() / 60
    keep = (gap_min <= max_gap_min) | (prev_t == next_t)
    out = filled[keep.to_numpy()].round(1).rename_axis("time").rename("glycemie").reset_index()
    return out


def patient_files(input_dir: Path) -> list:
    return sorted(p for p in input_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in (".xlsx", ".xls", ".csv")
                  and "summary" not in p.stem.lower() and not p.name.startswith((".", "~$")))


def convert_folder(input_dir: Path, out_dir: Path, max_gap_min: float = DEFAULT_MAX_GAP_MIN) -> int:
    input_dir, out_dir = Path(input_dir), Path(out_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"{input_dir} not found: put the ShanghaiT1DM patient files there.")
    files = patient_files(input_dir)
    if not files:
        raise SystemExit(f"No patient file in {input_dir}. Download the ShanghaiT1DM files from "
                         f"https://doi.org/10.6084/m9.figshare.20444397.v3 and put them there.")
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in files:
        try:
            raw = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
            df = to_project_format(raw, max_gap_min)
        except Exception as exc:                       # one bad file must not stop the others
            print(f"SKIPPED {path.name}: {exc}")
            continue
        df.to_csv(out_dir / f"{path.stem}.csv", index=False)
        print(f"{path.name}: {len(df)} readings (5 min) -> {path.stem}.csv")
        n += 1
    (out_dir / "MODIFIED_DATA_NOTICE.md").write_text(NOTICE.format(max_gap=f"{max_gap_min:g}"), encoding="utf-8")
    print(f"\n{n} patient(s) written to {out_dir}/  - MODIFIED data, see MODIFIED_DATA_NOTICE.md")
    return n


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", type=Path, default=DEFAULT_IN, help=f"original files (default {DEFAULT_IN})")
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT, help=f"converted files (default {DEFAULT_OUT})")
    ap.add_argument("--max_gap", type=float, default=DEFAULT_MAX_GAP_MIN,
                    help="longest gap (min) between real readings that is interpolated (default 20)")
    args = ap.parse_args(argv)
    convert_folder(args.input_dir, args.out_dir, args.max_gap)


if __name__ == "__main__":
    main()
