"""
scripts/preprocessing/generate_open_data.py
===========================================
Generate an anonymised open-access CGM dataset.

Privacy-by-design anonymisation pipeline compliant with:

  * **HIPAA Safe Harbor**  — 45 CFR §164.514(b)  (United States)
  * **GDPR**              — Regulation (EU) 2016/679, Recital 26,
                            Art. 5(1)(b)(c), Art. 25  (European Union)

Anonymisation steps
-------------------
1. **Patient re-identification**
   Original patient codes (which can function as medical-record numbers,
   a HIPAA category-2 identifier) are replaced by randomly-ordered
   sequential codes ``OD_001 … OD_N``.  The mapping is computed in
   memory and **never written to disk**.

2. **Date / time removal**  [HIPAA Safe Harbor identifier #3]
   Absolute timestamps — including calendar dates and clock times — are
   replaced by a floating-point *relative time* in hours measured from
   the first valid reading of each patient's monitoring window.
   No date, year, or wall-clock information is retained in the output.
   An optional random phase shift  Δt ∈ [0, 24 h)  can be applied
   (``--phase_shift``) to further obscure the exact monitoring start-time
   while preserving circadian structure.

3. **Glucose signal perturbation**
   Independent Gaussian noise  N(0, σ²)  (default σ = 2.0 mg/dL) is
   added to every glucose reading.  This magnitude is within the typical
   CGM device measurement uncertainty (≈ 10–15 % MARD) and does not
   materially distort population-level statistics (TIR, mean, SD), while
   preventing exact signal matching against the source data.
   Values are clipped to the physiological plausibility range
   [20, 600] mg/dL and rounded to one decimal place.

4. **Metadata removal**
   Internal quality flags (flag_time_error, flag_outlier_glucose, …),
   window filenames, and any other pipeline artefacts are excluded from
   the output.  Only the two scientific columns are published.

5. **Cohort-order shuffling**
   Patients are assigned to anonymous codes in a randomly permuted order
   (reproducible with ``--seed``), so rank-position in the original
   cohort is not recoverable from the output file ordering.

Residual re-identification risk
---------------------------------
After the above steps the data satisfy the definition of *anonymised*
data under GDPR Recital 26 (the data controller and any third party
cannot reasonably identify the natural person) and the HIPAA Safe Harbor
standard (all 18 enumerated identifiers removed).  Nevertheless, users
remain responsible for verifying compliance with applicable local law
before onward redistribution.  We recommend coupling this output with a
restrictive data-use licence (e.g., Creative Commons BY 4.0 or the
PhysioNet Credentialed Health Data Licence 1.5.0).

Outputs
-------
  Results/Open_Data/
    ├── per_patient/
    │   ├── OD_001.csv
    │   ⋮
    │   └── OD_N.csv
    ├── cohort_open_data.csv   — all patients concatenated
    └── DATACARD.md            — dataset description and terms of use

CSV columns
-----------
  patient_id   : anonymous code (OD_001 … OD_N)
  time_h       : relative time from monitoring start (hours ≥ 0, float)
  glucose_mgdl : glucose concentration + noise (mg/dL, float)

With ``--unit mmol/L`` the glucose column is published in mmol/L and named
``glucose_mmoll`` instead.  Anonymisation is unchanged: the noise σ and the
clipping range are applied in mg/dL, and the unit conversion is the last
step, so both releases describe exactly the same perturbed signal.

Usage
-----
    # Standard run (filtered cohort, default noise)
    python3 -m scripts.preprocessing.generate_open_data

    # Publish the same release in mmol/L
    python3 -m scripts.preprocessing.generate_open_data --unit mmol/L

    # All patients, stronger noise, random phase shift, custom seed
    python3 -m scripts.preprocessing.generate_open_data --no_filter --noise_std 3.0 \
                                                         --phase_shift --seed 123

Authors
-------
Guillaume Deside, Laura Symul, Philippe Lysy, Zain Syed
UCL — counterfactual-cgm-slh project

Date
----
2025
"""

import os
import argparse
import textwrap
from datetime import date

import numpy as np
import pandas as pd

from src.config import (
    GLUCOSE_MIN,
    GLUCOSE_MAX,
    TIME_COL,
    GLUCOSE_COL,
    FILTER_STATUS_CSV,
)
from src.config.paths import RESULTS_PATH, PREPROCESSING_PATH
from src.preprocessing.load_data import load_patient_window, discover_patients
from src.filtering.load_filtering import load_filtered_patient_list
from src.utils.unit_conversion import (
    UNIT_MGDL,
    UNIT_MMOLL,
    mgdl_to_mmoll,
    normalize_unit,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR    = RESULTS_PATH / "Open_Data"
PER_PATIENT_DIR = OUTPUT_DIR / "per_patient"

# Default noise standard deviation (mg/dL).
# CGM device MARD is typically 10–15 %, corresponding to ~10–20 mg/dL at
# hyperglycaemic levels.  σ = 2 mg/dL is a conservative lower bound that
# is imperceptible clinically yet prevents exact signal matching.
DEFAULT_NOISE_STD = 2.0

# Anonymous patient ID prefix and zero-padding
ANON_PREFIX = "OD"


# =============================================================================
# ANONYMISATION HELPERS
# =============================================================================
def _make_anon_ids(n: int, prefix: str = ANON_PREFIX) -> list[str]:
    """Return zero-padded anonymous ID strings OD_001 … OD_N."""
    width = max(3, len(str(n)))
    return [f"{prefix}_{str(i + 1).zfill(width)}" for i in range(n)]


def anonymise_patient(
    df: pd.DataFrame,
    anon_id: str,
    noise_std: float,
    rng: np.random.Generator,
    phase_shift: bool,
    unit: str = UNIT_MGDL,
) -> pd.DataFrame:
    """
    Transform one patient's preprocessed window into its open-access form.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed window with at least TIME_COL and GLUCOSE_COL columns.
        Quality-flagged rows must have been removed upstream.
    anon_id : str
        Anonymous patient identifier (e.g. ``OD_042``).
    noise_std : float
        Standard deviation of Gaussian noise added to glucose (mg/dL).
    rng : np.random.Generator
        Seeded random number generator (ensures reproducibility).
    phase_shift : bool
        If True, shift the relative time by a random Δt ∈ [0, 24 h) so
        that the exact monitoring start-time cannot be inferred.
    unit : str, default "mg/dL"
        Unit of the published glucose column.  Noise and clipping are always
        applied in mg/dL; conversion happens last, so the released signal is
        identical in both units.

    Returns
    -------
    pd.DataFrame
        Anonymised DataFrame with columns [patient_id, time_h, glucose_<unit>].
    """
    # ── 1. Keep only the two scientific columns ───────────────────────────────
    out = df[[TIME_COL, GLUCOSE_COL]].copy()

    # ── 2. Relative time in hours (HIPAA identifier #3 removed) ──────────────
    t0 = out[TIME_COL].min()
    out["time_h"] = (out[TIME_COL] - t0).dt.total_seconds() / 3600.0

    # Optional phase shift: add a uniform random offset in [0, 24) hours.
    # This further obscures the exact monitoring start while preserving the
    # internal relative structure and circadian patterns.
    if phase_shift:
        delta = rng.uniform(0.0, 24.0)
        out["time_h"] = out["time_h"] + delta

    out["time_h"] = out["time_h"].round(3)

    # ── 3. Glucose perturbation ───────────────────────────────────────────────
    noise = rng.normal(loc=0.0, scale=noise_std, size=len(out))
    out["glucose_mgdl"] = (out[GLUCOSE_COL] + noise).clip(GLUCOSE_MIN, GLUCOSE_MAX)
    out["glucose_mgdl"] = out["glucose_mgdl"].round(1)

    # ── 4. Optional unit conversion (after noise and clipping) ───────────────
    glucose_col = "glucose_mgdl"
    if normalize_unit(unit) == UNIT_MMOLL:
        glucose_col = "glucose_mmoll"
        out[glucose_col] = mgdl_to_mmoll(out["glucose_mgdl"], decimals=2)

    # ── 5. Assign anonymous ID and select output columns ─────────────────────
    out["patient_id"] = anon_id
    return out[["patient_id", "time_h", glucose_col]].reset_index(drop=True)


# =============================================================================
# DATACARD WRITER
# =============================================================================
def write_datacard(output_path: str, n_patients: int, noise_std: float,
                   phase_shift: bool, seed: int, unit: str = UNIT_MGDL) -> None:
    """
    Write a DATACARD.md file describing the open dataset.

    The datacard follows the Datasheets for Datasets (Gebru et al., 2021)
    template and the PhysioNet publication guidelines.

    Parameters
    ----------
    output_path : str
        Destination path for the DATACARD.md file.
    n_patients : int
        Number of patients included in the release.
    noise_std : float
        Gaussian noise σ applied to glucose values.
    phase_shift : bool
        Whether a random phase shift was applied to monitoring times.
    seed : int
        Random seed used for reproducibility.
    unit : str, default "mg/dL"
        Unit of the published glucose column.
    """
    today = date.today().isoformat()

    unit = normalize_unit(unit)
    glucose_col = "glucose_mmoll" if unit == UNIT_MMOLL else "glucose_mgdl"
    clip_lo, clip_hi = (
        (f"{mgdl_to_mmoll(GLUCOSE_MIN, decimals=1)}",
         f"{mgdl_to_mmoll(GLUCOSE_MAX, decimals=1)}")
        if unit == UNIT_MMOLL else (f"{GLUCOSE_MIN}", f"{GLUCOSE_MAX}")
    )
    card = textwrap.dedent(f"""\
    # Dataset Card — Anonymised CGM Open-Access Release

    **Generated:** {today}
    **Version:** 1.0.0

    ---

    ## Overview

    This dataset contains anonymised continuous glucose monitoring (CGM) signals
    derived from a clinical study of hyperglycaemic episodes in people with type 1
    diabetes.  The original data were collected with 5-minute CGM devices over
    45-day monitoring windows.

    The anonymisation pipeline is implemented in
    `scripts/preprocessing/generate_open_data.py` and has been designed to comply with:

    - **HIPAA Safe Harbor** (45 CFR §164.514(b), United States)
    - **GDPR** Regulation (EU) 2016/679 — Recital 26, Art. 5(1)(b)(c), Art. 25
      (European Union)

    ---

    ## Cohort

    | Parameter                | Value           |
    |--------------------------|-----------------|
    | Number of patients       | {n_patients}    |
    | Monitoring window        | 45 days         |
    | Nominal sampling rate    | 5 minutes       |
    | Glucose range (clipped)  | {clip_lo} – {clip_hi} {unit} |

    ---

    ## Columns

    | Column        | Type   | Description                                      |
    |---------------|--------|--------------------------------------------------|
    | `patient_id`  | string | Anonymous code (OD\\_001 … OD\\_N); the mapping   |
    |               |        | to original IDs was never persisted              |
    | `time_h`      | float  | Relative time from first valid reading (hours);  |
    |               |        | no calendar date or clock time is encoded        |
    | `{glucose_col}`| float | CGM glucose + Gaussian noise ({unit})            |

    ---

    ## Anonymisation Steps Applied

    1. **Patient re-identification** — original patient codes replaced by
       randomly-ordered sequential codes; mapping discarded.
    2. **Date / time removal** — absolute timestamps converted to relative
       hours from monitoring start (HIPAA identifier #3 removed).
       {"A random phase shift Δt ∈ [0, 24 h) was applied per patient." if phase_shift else "No phase shift applied (monitoring start is at time_h = 0)."}
    3. **Glucose perturbation** — independent Gaussian noise
       N(0, {noise_std}²) mg/dL added; values clipped to
       [{GLUCOSE_MIN}, {GLUCOSE_MAX}] mg/dL.  Noise and clipping are applied
       in mg/dL; the published column is expressed in {unit}.
    4. **Metadata removal** — quality flags and pipeline artefacts excluded.
    5. **Cohort shuffling** — patient order randomly permuted
       (seed = {seed}).

    ---

    ## Intended Use

    This dataset is released to support reproducible research in:

    - Continuous glucose monitoring signal processing
    - Hyperglycaemic episode detection and classification (SLH / LLH)
    - Counterfactual signal construction and sensitivity analysis
    - Development and benchmarking of CGM analysis algorithms

    ### Out-of-scope uses

    - Clinical diagnosis or treatment decisions
    - Re-identification of individuals
    - Commercial use without a separate data-use agreement

    ---

    ## Licence

    This dataset is released under the **Creative Commons Attribution 4.0
    International (CC BY 4.0)** licence.
    You are free to share and adapt the material for any purpose, provided
    appropriate credit is given.
    See <https://creativecommons.org/licenses/by/4.0/> for details.

    Users remain responsible for verifying compliance with applicable local
    regulations before onward redistribution.

    ---

    ## Citation

    If you use this dataset, please cite:

    ```
    Deside G., Symul L., Lysy P., Syed Z. (2025).
    Counterfactual CGM Signals for Short-Lasting Hyperglycaemia Episodes.
    UCL. [Dataset]. doi: TBD
    ```

    ---

    ## Contact

    For questions regarding data access or use, please open an issue at
    <https://github.com/gdeside/counterfactual-cgm-slh>.
    """)

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(card)

    print(f"  DATACARD written to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    """Run the open-data anonymisation pipeline."""
    parser = argparse.ArgumentParser(
        description="Anonymise CGM windows for open-access release (HIPAA + GDPR compliant)"
    )
    parser.add_argument(
        "--no_filter", action="store_true",
        help="Include all patients (ignore median-glucose filter)",
    )
    parser.add_argument(
        "--noise_std", type=float, default=DEFAULT_NOISE_STD,
        help=f"Gaussian noise σ added to glucose readings (mg/dL, default {DEFAULT_NOISE_STD})",
    )
    parser.add_argument(
        "--phase_shift", action="store_true",
        help="Apply a random phase shift Δt ∈ [0, 24 h) to each patient's time axis",
    )
    parser.add_argument(
        "--unit", default=UNIT_MGDL,
        help=f"Unit of the published glucose column "
             f"('{UNIT_MGDL}' or '{UNIT_MMOLL}', default {UNIT_MGDL})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default 42)",
    )
    parser.add_argument(
        "--output_dir", default=str(OUTPUT_DIR),
        help="Root output directory",
    )
    parser.add_argument(
        "--filter_csv", default=str(FILTER_STATUS_CSV),
        help="Patient filter status CSV",
    )
    args = parser.parse_args()

    unit = normalize_unit(args.unit)
    rng = np.random.default_rng(args.seed)

    print("\n" + "=" * 70)
    print("  Open-Data Anonymisation Pipeline")
    print("  HIPAA Safe Harbor (45 CFR §164.514(b)) + GDPR (EU) 2016/679")
    print("=" * 70)
    print(f"  Source     : {PREPROCESSING_PATH}")
    print(f"  Output     : {args.output_dir}")
    print(f"  Noise σ    : {args.noise_std} mg/dL")
    print(f"  Unit       : {unit}")
    print(f"  Phase shift: {'yes' if args.phase_shift else 'no'}")
    print(f"  Seed       : {args.seed}")
    print("=" * 70 + "\n")

    # ── Discover patients ─────────────────────────────────────────────────────
    filtered = None if args.no_filter else load_filtered_patient_list(args.filter_csv)
    all_pids  = discover_patients(str(PREPROCESSING_PATH))

    if not all_pids:
        print(f"  [ERROR] No preprocessed patient directories found in {PREPROCESSING_PATH}")
        return

    pids = [p for p in all_pids if p in filtered] if filtered is not None else all_pids

    if not pids:
        print("  [ERROR] No patients remain after filtering.")
        return

    # ── Random permutation → anonymous IDs (mapping never saved) ──────────────
    permuted = rng.permutation(pids).tolist()
    anon_ids  = _make_anon_ids(len(permuted))
    id_map    = dict(zip(permuted, anon_ids))   # original → anonymous; in-memory only

    print(f"  Cohort: {len(pids)} patients\n")

    # ── Create output directories ─────────────────────────────────────────────
    per_patient_dir = os.path.join(args.output_dir, "per_patient")
    os.makedirs(per_patient_dir, exist_ok=True)

    # ── Process patients ──────────────────────────────────────────────────────
    all_chunks: list[pd.DataFrame] = []
    n_ok = n_fail = 0

    for orig_pid in permuted:
        anon_id = id_map[orig_pid]

        df = load_patient_window(orig_pid)
        if df is None:
            print(f"  [{orig_pid}] SKIP — no valid window")
            n_fail += 1
            continue

        anon_df = anonymise_patient(df, anon_id, args.noise_std, rng,
                                    args.phase_shift, unit=unit)

        # Per-patient file
        out_path = os.path.join(per_patient_dir, f"{anon_id}.csv")
        anon_df.to_csv(out_path, index=False)

        all_chunks.append(anon_df)
        n_ok += 1
        n_rows = len(anon_df)
        t_max  = anon_df["time_h"].max()
        print(f"  {anon_id}  ({orig_pid:>6}) — {n_rows:5d} rows, {t_max:.1f} h")

    print(f"\n  Processed: {n_ok}  |  Skipped: {n_fail}\n")

    if not all_chunks:
        print("  [ERROR] No output generated.")
        return

    # ── Cohort-wide CSV ───────────────────────────────────────────────────────
    cohort_df = pd.concat(all_chunks, ignore_index=True)
    cohort_path = os.path.join(args.output_dir, "cohort_open_data.csv")
    cohort_df.to_csv(cohort_path, index=False)
    print(f"  Cohort CSV saved : {cohort_path}")
    print(f"  Total rows       : {len(cohort_df):,}")

    # ── DATACARD ──────────────────────────────────────────────────────────────
    datacard_path = os.path.join(args.output_dir, "DATACARD.md")
    write_datacard(
        datacard_path,
        n_patients=n_ok,
        unit=unit,
        noise_std=args.noise_std,
        phase_shift=args.phase_shift,
        seed=args.seed,
    )

    print("\n" + "=" * 70)
    print(f"  Done!  Output → {os.path.abspath(args.output_dir)}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
