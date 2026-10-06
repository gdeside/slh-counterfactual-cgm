"""
scripts/sample_data/generate_sample_data.py
===========================================

Synthetic CGM sample dataset (no real patient data) so that reviewers can run the
core pipeline (preprocessing -> metrics -> filtering -> episode detection ->
counterfactual / sensitivity) in minutes.

For each patient A001..A00N it writes ``<DIR>/Data/Raw/A00N.xlsx`` with the two
columns the raw loader expects (``time``, ``glycemie`` in mg/dL), sampled every
5 min for 50 days with a few sensor gaps.

The signal is a patient-specific baseline + circadian drift + AR(1) noise + meal
excursions. Excursion size/width differ per patient so the sample contains
SLH (10-60 min), LLH (> 60 min) and night-time episodes, with a different SLH
count per patient.

Run the pipeline on it with::

    CGM_DATA_ROOT=<DIR> make preprocessing metrics_preprocessing filtering episode_detection

Fixed seed: the output is reproducible.

Usage
-----
    python3 -m scripts.sample_data.generate_sample_data [--help for options]
    (or: make sample_data)

Output
------
    <out>/Data/Raw/A00N.xlsx for each fake patient (+ <out>.zip with --zip)
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DAYS = 50
STEP_MIN = 5
START = pd.Timestamp("2000-01-01 00:00")   # arbitrary t=0: synthetic, relative time (not a real date)

# baseline mg/dL, meal-spike amplitude (mg/dL), spike width (min, Gaussian sigma)
PROFILES = [
    dict(base=125, amp=(50, 110), sigma=(12, 30), p_meal=0.85),   # frequent short spikes
    dict(base=135, amp=(30, 70),  sigma=(15, 45), p_meal=0.80),   # mixed short/long
    dict(base=115, amp=(60, 120), sigma=(10, 22), p_meal=0.90),   # mostly short spikes
    dict(base=140, amp=(20, 50),  sigma=(20, 50), p_meal=0.70),   # fewer, slower excursions
    dict(base=130, amp=(40, 90),  sigma=(12, 35), p_meal=0.85),
]
MEALS_H = (7.5, 10.5, 12.5, 16.0, 19.5)


def synth_patient(rng: np.random.Generator, prof: dict) -> pd.DataFrame:
    idx = pd.date_range(START, periods=DAYS * 24 * 60 // STEP_MIN, freq=f"{STEP_MIN}min")
    t_min = np.arange(len(idx)) * STEP_MIN
    hour = (t_min / 60) % 24

    g = prof["base"] + 8 * np.sin(2 * np.pi * (hour - 4) / 24)
    noise = np.zeros(len(idx))
    for i in range(1, len(idx)):                       # AR(1) sensor/physiology noise
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 2.5)
    g = g + noise

    for day in range(DAYS):
        for mh in MEALS_H:
            if rng.random() > prof["p_meal"]:
                continue
            centre = day * 1440 + mh * 60 + rng.normal(0, 20)
            amp = rng.uniform(*prof["amp"])
            sig = rng.uniform(*prof["sigma"])
            g += amp * np.exp(-0.5 * ((t_min - centre) / sig) ** 2)

    g = np.clip(np.round(g), 40, 400)
    df = pd.DataFrame({"time": idx, "glycemie": g})

    # a few sensor gaps (keeps coverage ~ 95 %)
    for _ in range(rng.integers(3, 7)):
        s = rng.integers(0, len(df) - 60)
        df = df.drop(df.index[s:s + rng.integers(6, 48)])
    return df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default="sample_data", help="Data root (default: ./sample_data)")
    ap.add_argument("--n-patients", type=int, default=4, choices=range(1, len(PROFILES) + 1))
    ap.add_argument("--zip", action="store_true", help="Also write <out>.zip (release asset)")
    args = ap.parse_args()

    out = Path(args.out)
    raw = out / "Data" / "Raw"
    raw.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for i in range(args.n_patients):
        df = synth_patient(rng, PROFILES[i])
        df.to_excel(raw / f"A{i + 1:03d}.xlsx", index=False)
        print(f"A{i + 1:03d}: {len(df)} rows, median {df.glycemie.median():.0f} mg/dL")

    if args.zip:
        z = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
        print(f"Wrote {z}")
    print(f"\nNext: export CGM_DATA_ROOT=$(pwd)/{out}")


if __name__ == "__main__":
    main()
