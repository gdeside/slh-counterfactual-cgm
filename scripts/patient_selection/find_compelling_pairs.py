"""
scripts/patient_selection/find_compelling_pairs.py
===================================================
Select pairs of participants that illustrate, in the article, that the number
of SLH episodes carries information beyond standard CGM metrics.

Every pair of participants of the filtered cohort is scored once; five
strategies then pick the most informative pairs (patient_1 is always the
participant with the FEWER SLH episodes):

  extreme_divergence             similar overall CGM profile (standardised
                                 metrics), very different n_SLH
  TIR_matched                    nearly identical TIR 70-180, different n_SLH
  threshold_borderline           both close to the 70% TIR target, different n_SLH
  representative_vs_extreme      one "typical" participant (closest quartile to the
                                 cohort centroid) vs one in the top n_SLH quartile
  matched_mean_diff_variability  same mean glucose, different CV, different n_SLH

Each selected pair is annotated with the percentile of its n_SLH in the
cohort and a TIR control category (>=70% good, >=50% adequate, else poor).

Inputs
------
  Data/Metrics/metrics_all_patients.csv
  Data/Episodes/episodes_summary.csv
  Data/Filtered/..._patient_status.csv      (skipped with --no_filter)

Outputs (RESULTS_PATH / Patient_Pair_Selection)
-----------------------------------------------
  compelling_pairs.csv

Usage
-----
    env/bin/python3 -m scripts.patient_selection.find_compelling_pairs [--no_filter] [--n_pairs 5]

Authors: Guillaume Deside
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from sklearn.preprocessing import StandardScaler

from src.config import PATIENT_ID_COL, FILTER_STATUS_CSV, METRICS_CSV
from src.config.paths import RESULTS_PATH
from src.episodes.load_episodes import load_episode_summary
from src.filtering.load_filtering import load_filtered_patient_list
from src.metrics.load_metrics import load_patient_metrics

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = RESULTS_PATH / "Patient_Pair_Selection"
OUTPUT_NAME = "compelling_pairs.csv"

# Metrics (columns of metrics_all_patients.csv) describing the "overall CGM profile"
PROFILE_METRICS = ["Mean_glycemia", "Median_glycemia", "SD", "CV", "TIR70180", "TAR180", "TAR250"]

N_PAIRS_PER_STRATEGY = 5

TIR_MATCH_TOLERANCE = 0.02        # |ΔTIR| (fraction) for "TIR_matched"
TIR_TARGET = 0.70                 # clinical TIR target
TIR_TARGET_TOLERANCE = 0.05       # both participants within ±5 points of the target
MEAN_MATCH_TOLERANCE_MGDL = 5.0   # |Δmean glucose| for "matched mean"
MIN_CV_DIFF = 0.02                # minimum |ΔCV| (fraction) for "matched mean"
REPRESENTATIVE_QUANTILE = 0.25    # closest quartile to the cohort centroid
EXTREME_SLH_QUANTILE = 0.75       # top n_SLH quartile

TIR_GOOD, TIR_ADEQUATE = 0.70, 0.50


# =============================================================================
# DATA LOADING
# =============================================================================
def load_cohort(apply_filter: bool = True) -> pd.DataFrame:
    """One row per participant: CGM metrics + n_SLH (filtered cohort by default)."""
    metrics = load_patient_metrics(METRICS_CSV)
    episodes = load_episode_summary()[[PATIENT_ID_COL, "n_SLH"]]
    cohort = metrics.merge(episodes, on=PATIENT_ID_COL, how="inner")

    if apply_filter:
        selected = load_filtered_patient_list(str(FILTER_STATUS_CSV))
        if selected is not None:
            cohort = cohort[cohort[PATIENT_ID_COL].isin(selected)]

    cohort = cohort.dropna(subset=PROFILE_METRICS + ["n_SLH"]).reset_index(drop=True)
    return cohort


# =============================================================================
# PAIR TABLE
# =============================================================================
def build_pair_table(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    All unordered pairs of participants, with patient_1 = the one with fewer SLH episodes.

    Columns: patient_1/2, n_SLH_1/2, n_SLH_diff, metric_distance (Euclidean, standardised
    PROFILE_METRICS), TIR_1/2 and mean/CV of both participants.
    """
    z = StandardScaler().fit_transform(cohort[PROFILE_METRICS].to_numpy())
    centroid_distance = np.linalg.norm(z - z.mean(axis=0), axis=1)

    i, j = np.triu_indices(len(cohort), k=1)
    swap = cohort["n_SLH"].to_numpy()[i] > cohort["n_SLH"].to_numpy()[j]
    i, j = np.where(swap, j, i), np.where(swap, i, j)          # patient_1 has fewer SLH

    def side(col, idx):
        return cohort[col].to_numpy()[idx]

    pairs = pd.DataFrame({
        "patient_1": side(PATIENT_ID_COL, i), "patient_2": side(PATIENT_ID_COL, j),
        "n_SLH_1": side("n_SLH", i), "n_SLH_2": side("n_SLH", j),
        "TIR_1": side("TIR70180", i), "TIR_2": side("TIR70180", j),
        "mean_glucose_1": side("Mean_glycemia", i), "mean_glucose_2": side("Mean_glycemia", j),
        "CV_1": side("CV", i), "CV_2": side("CV", j),
        "centroid_distance_1": centroid_distance[i], "centroid_distance_2": centroid_distance[j],
        "metric_distance": np.linalg.norm(z[i] - z[j], axis=1),
    })
    pairs["n_SLH_diff"] = pairs["n_SLH_2"] - pairs["n_SLH_1"]
    pairs.attrs["typical_distance_cut"] = float(np.quantile(centroid_distance, REPRESENTATIVE_QUANTILE))
    return pairs


# =============================================================================
# STRATEGIES (each takes the pair table and returns the selected pairs)
# =============================================================================
def extreme_divergence(pairs: pd.DataFrame, cohort: pd.DataFrame, n: int) -> pd.DataFrame:
    """Large n_SLH difference relative to the distance between the CGM profiles."""
    sel = pairs[pairs["metric_distance"] > 0].copy()
    sel["divergence_score"] = sel["n_SLH_diff"] / sel["metric_distance"]
    return sel.nlargest(n, "divergence_score").assign(type="extreme_divergence")


def tir_matched(pairs: pd.DataFrame, cohort: pd.DataFrame, n: int) -> pd.DataFrame:
    """Same TIR (gold-standard metric), different SLH burden."""
    close = (pairs["TIR_1"] - pairs["TIR_2"]).abs() <= TIR_MATCH_TOLERANCE
    return pairs[close & (pairs["n_SLH_diff"] > 0)].nlargest(n, "n_SLH_diff").assign(type="TIR_matched")


def threshold_borderline(pairs: pd.DataFrame, cohort: pd.DataFrame, n: int) -> pd.DataFrame:
    """Both participants close to the 70% TIR target, different SLH burden."""
    near = lambda col: (pairs[col] - TIR_TARGET).abs() <= TIR_TARGET_TOLERANCE
    return (pairs[near("TIR_1") & near("TIR_2") & (pairs["n_SLH_diff"] > 0)]
            .nlargest(n, "n_SLH_diff").assign(type="threshold_borderline"))


def representative_vs_extreme(pairs: pd.DataFrame, cohort: pd.DataFrame, n: int) -> pd.DataFrame:
    """A typical participant (near the cohort centroid) versus one with extreme SLH burden."""
    typical_cut = pairs.attrs["typical_distance_cut"]
    extreme_cut = cohort["n_SLH"].quantile(EXTREME_SLH_QUANTILE)
    typical = pairs["centroid_distance_1"] <= typical_cut
    extreme = pairs["n_SLH_2"] >= extreme_cut
    return (pairs[typical & extreme & (pairs["n_SLH_diff"] > 0)]
            .nlargest(n, "n_SLH_diff").assign(type="representative_vs_extreme"))


def matched_mean_different_variability(pairs: pd.DataFrame, cohort: pd.DataFrame, n: int) -> pd.DataFrame:
    """Same mean glucose, different CV and SLH burden: does SLH capture what CV misses?"""
    same_mean = (pairs["mean_glucose_1"] - pairs["mean_glucose_2"]).abs() <= MEAN_MATCH_TOLERANCE_MGDL
    diff_cv = (pairs["CV_1"] - pairs["CV_2"]).abs() > MIN_CV_DIFF
    return (pairs[same_mean & diff_cv & (pairs["n_SLH_diff"] > 0)]
            .nlargest(n, "n_SLH_diff").assign(type="matched_mean_diff_variability"))


STRATEGIES = {
    "extreme SLH divergence": extreme_divergence,
    "TIR matched": tir_matched,
    "clinical threshold (TIR ~70%)": threshold_borderline,
    "representative vs extreme": representative_vs_extreme,
    "matched mean, different variability": matched_mean_different_variability,
}


# =============================================================================
# ANNOTATION
# =============================================================================
def tir_category(tir: float) -> str:
    if tir >= TIR_GOOD:
        return "Good control"
    if tir >= TIR_ADEQUATE:
        return "Adequate control"
    return "Poor control"


def annotate_pairs(pairs: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Add the pair label, the n_SLH cohort percentiles and the TIR control categories."""
    out = pairs.copy()
    out.insert(0, "pair", out["patient_1"] + " vs " + out["patient_2"])
    for k in (1, 2):
        out[f"n_SLH_percentile_{k}"] = [percentileofscore(cohort["n_SLH"], v) for v in out[f"n_SLH_{k}"]]
        out[f"TIR_status_{k}"] = out[f"TIR_{k}"].map(tir_category)
    out = out.rename(columns={"type": "compelling_factor"})
    for col in ("TIR_1", "TIR_2", "CV_1", "CV_2"):          # fractions -> %
        out[col] = out[col] * 100
    return out


# =============================================================================
# MAIN
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--no_filter", action="store_true", help="Use all participants instead of the filtered cohort.")
    p.add_argument("--n_pairs", type=int, default=N_PAIRS_PER_STRATEGY, help="Pairs kept per strategy.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("  Compelling participant pairs for article illustration")
    print("=" * 70)

    cohort = load_cohort(apply_filter=not args.no_filter)
    print(f"  Cohort: {len(cohort)} participants, n_SLH {cohort['n_SLH'].min():.0f}–{cohort['n_SLH'].max():.0f}")
    if len(cohort) < 2:
        print("  At least two participants are needed.")
        return

    pairs = build_pair_table(cohort)
    print(f"  Pairs scored: {len(pairs)}\n")

    selected = []
    for name, strategy in STRATEGIES.items():
        found = strategy(pairs, cohort, args.n_pairs)
        print(f"  {name:<38s} {len(found)} pair(s)")
        selected.append(found)

    result = pd.concat(selected, ignore_index=True).drop_duplicates(subset=["patient_1", "patient_2"])
    result = annotate_pairs(result.reset_index(drop=True), cohort)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / OUTPUT_NAME
    result.to_csv(out_csv, index=False)

    print("\n  Top pairs (fewer SLH → more SLH):")
    cols = ["pair", "n_SLH_1", "n_SLH_2", "compelling_factor", "TIR_status_1", "TIR_status_2"]
    print(result[cols].head(10).to_string(index=False))
    print(f"\n  {len(result)} unique pairs → {out_csv}")


if __name__ == "__main__":
    main()
