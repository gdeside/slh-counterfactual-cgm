"""
scripts/sample_data/generate_sample_data_from_real.py
=====================================================

Build 4 FAKE patients (A001–A004, 50 days, 5-min sampling) from the shape of REAL CGM data, for a more
realistic illustration dataset than the purely simulated one (`generate_sample_data.py`).

Run it on the machine that holds the real data (CGM_DATA_ROOT = real data root, `make preprocessing`
and `make episode_detection` already done). The output contains NO real patient: it is a re-assembly.

How a fake patient is built
---------------------------
1. SOURCE POOLS — the filtered real cohort is ranked by number of SLH episodes and cut into 4 groups
   (high, mid-high, mid-low, low burden); fake patient k draws only from group k, so the four fake patients
   keep different SLH burdens. Each pool must contain at least ``--min_sources`` (default 5) real patients.
2. DAY MIXING — real patients' 24-h segments (03:00 → 03:00, so the cut falls in the quiet night; days with
   < 95 % of readings are discarded) are drawn at random, each used at most once, in random order. A fake
   patient therefore interleaves days from many different real patients: no real trajectory is kept.
3. NOISE — a patient-specific glucose scale in [0.95, 1.05], plus Gaussian noise (σ = 2 mg/dL by default)
   on every reading; values clipped to 40–400 mg/dL.
4. RELATIVE TIME — timestamps are replaced by an arbitrary origin (2000-01-01 03:00) on a regular 5-min grid.
5. FRESH IDS — A001…A004, assigned in a fixed order; no mapping to real patients is written anywhere
   (the console only prints aggregate numbers, never real patient IDs).

Output (same layout as `generate_sample_data.py`)
------------------------------------------------
    <out>/Data/Raw/A001.xlsx … A004.xlsx     columns: time, glycemie (mg/dL)
    <out>.zip                                 with --zip (release asset)

IMPORTANT — before publishing the result: data derived from real patients stays personal data unless the
re-assembly is judged anonymous. Get the data-protection officer's / ethics committee's approval, keep the seed
private if the DPO asks for it, and never commit the real inputs.

Usage
-----
    CGM_DATA_ROOT=/path/to/real/data python3 -m scripts.sample_data.generate_sample_data_from_real \\
        [--out sample_data_real] [--seed 2024] [--noise_std 2.0] [--n_days 50] [--min_sources 5] [--zip]
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

STEP_MIN = 5
SLOTS_PER_DAY = 24 * 60 // STEP_MIN          # 288
DAY_START_H = 3                              # cut days at 03:00 (quiet night)
MIN_DAY_COVERAGE = 0.95
GLUCOSE_MIN, GLUCOSE_MAX = 40.0, 400.0
SCALE_RANGE = (0.95, 1.05)
ORIGIN = pd.Timestamp("2000-01-01") + pd.Timedelta(hours=DAY_START_H)   # arbitrary t = 0
N_FAKE = 4


# =============================================================================
# BUILDING BLOCKS (pure functions, unit-tested)
# =============================================================================
def split_days(df: pd.DataFrame, start_hour: int = DAY_START_H, min_coverage: float = MIN_DAY_COVERAGE) -> list:
    """Cut a trace (columns ``time``, ``glycemie``) into 24-h segments starting at ``start_hour``.

    Each segment is a length-288 array on a regular 5-min grid (NaN where a reading is missing);
    segments with less than ``min_coverage`` of valid readings are dropped."""
    if df is None or len(df) == 0:
        return []
    t = pd.to_datetime(df["time"]).dt.floor(f"{STEP_MIN}min")
    shifted = t - pd.Timedelta(hours=start_hour)
    day = shifted.dt.floor("D")
    slot = ((shifted - day).dt.total_seconds() // (STEP_MIN * 60)).astype(int)
    work = pd.DataFrame({"day": day, "slot": slot, "g": df["glycemie"].to_numpy(dtype=float)})
    work = work.groupby(["day", "slot"], as_index=False)["g"].mean()
    out = []
    for _, grp in work.groupby("day"):
        arr = np.full(SLOTS_PER_DAY, np.nan)
        arr[grp["slot"].to_numpy()] = grp["g"].to_numpy()
        if np.isfinite(arr).mean() >= min_coverage:
            out.append(arr)
    return out


def assign_pools(ranked_pids: list, n_pools: int = N_FAKE, min_sources: int = 5) -> list:
    """Split patients, ranked from highest to lowest SLH burden, into ``n_pools`` consecutive groups."""
    if len(ranked_pids) < n_pools * min_sources:
        raise ValueError(f"Need at least {n_pools * min_sources} patients ({min_sources} per fake patient), "
                         f"found {len(ranked_pids)}. Lower --min_sources or use the full cohort.")
    return [list(chunk) for chunk in np.array_split(np.asarray(ranked_pids, dtype=object), n_pools)]


def build_fake_patient(day_pool: list, n_days: int, rng: np.random.Generator,
                       noise_std: float = 2.0) -> pd.DataFrame:
    """Assemble one fake patient from a pool of real 24-h segments (each segment used at most once)."""
    if len(day_pool) < n_days:
        raise ValueError(f"Pool has {len(day_pool)} usable days, {n_days} needed.")
    order = rng.permutation(len(day_pool))[:n_days]
    trace = np.concatenate([day_pool[i] for i in order])
    trace = trace * rng.uniform(*SCALE_RANGE)
    trace = trace + rng.normal(0.0, noise_std, size=trace.size)
    trace = np.clip(trace, GLUCOSE_MIN, GLUCOSE_MAX).round(1)
    time = ORIGIN + pd.to_timedelta(np.arange(trace.size) * STEP_MIN, unit="min")
    df = pd.DataFrame({"time": time, "glycemie": trace}).dropna().reset_index(drop=True)   # keep real sensor gaps
    return df


# =============================================================================
# DRIVER
# =============================================================================
def load_ranked_cohort() -> list:
    """Filtered real cohort ranked by number of SLH episodes (descending)."""
    from src.config import FILTER_STATUS_CSV
    from src.episodes.load_episodes import load_episode_summary
    from src.filtering.load_filtering import load_filtered_patient_list

    summary = load_episode_summary()
    keep = load_filtered_patient_list(str(FILTER_STATUS_CSV))
    if keep:
        summary = summary[summary["patient_id"].isin(keep)]
    # tie-break by a seeded-independent order (patient id) so the ranking is deterministic
    return summary.sort_values(["n_SLH", "patient_id"], ascending=[False, True])["patient_id"].tolist()


def load_days(pid: str) -> list:
    from src.preprocessing.load_data import load_patient_window, prepare_valid_data
    raw = load_patient_window(pid)
    return split_days(prepare_valid_data(raw)) if raw is not None else []


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="sample_data_real", help="output data root (default: ./sample_data_real)")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--noise_std", type=float, default=2.0, help="Gaussian noise σ in mg/dL")
    ap.add_argument("--n_days", type=int, default=50)
    ap.add_argument("--min_sources", type=int, default=5, help="minimum real patients per fake patient")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    pools = assign_pools(load_ranked_cohort(), N_FAKE, args.min_sources)

    out = Path(args.out)
    raw_dir = out / "Data" / "Raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for k, pool_pids in enumerate(pools):
        day_pool = [d for pid in pool_pids for d in load_days(pid)]
        fake = build_fake_patient(day_pool, args.n_days, rng, args.noise_std)
        pid = f"A{k + 1:03d}"
        fake.to_excel(raw_dir / f"{pid}.xlsx", index=False)
        print(f"{pid}: built from {len(pool_pids)} real patients, {len(day_pool)} candidate days -> "
              f"{len(fake)} readings, median {fake['glycemie'].median():.0f} mg/dL")

    if args.zip:
        z = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
        print(f"Wrote {z}")
    print(f"\nNext: export CGM_DATA_ROOT=$(pwd)/{out}   (then `make pipeline`).")
    print("Reminder: get the DPO's approval before publishing data derived from real patients.")


if __name__ == "__main__":
    main()
