"""
CGM Episode Detection and Classification
=========================================
Input       : Data/Data_Preprocessed_45d/{PID}/{PID}_win1_*.xlsx
              Data/Preprocessed/windows_summary.csv
              Data/Filtered/median_glucose_110to170_patient_status.csv
              (only patients selected by cgm_filtering.py are processed)
Output      : Data/Episodes/{PID}_episodes.csv
              Data/Episodes/episodes_summary.csv
              Data/Episodes/episode_type_summary.csv
                (total count, n patients with >=1, and per-patient
                 mean/SD/min/Q1/median/Q3/max for each episode type:
                 the original binary classification, the paper's 3-way
                 slh_category, and the SLH daytime/nighttime split)

Episode classification criteria
--------------------------------
Episodes are classified as Short-Lasting Hyperglycemia (SLH) or Long-Lasting
Hyperglycemia (LLH) based on sequential criteria:

  C1  Duration ∈ [D_MIN_MIN, D_MAX_MIN] measured as
      n_pts_above_H_HYPER × timestep  (gap-robust)
  
  C2  Data completeness ≥ MIN_DATA_COMPLETENESS
      (episode + context windows must have ≥75% expected measurements)
  
  C3  Daytime: onset outside [NIGHT_START_H, NIGHT_END_H)
  
  C4  No glucose > H_HYPER in CONTEXT_WIN_MIN before onset
      (episode not preceded by sustained hyperglycaemia)
  
  C5  No glucose > H_HYPER in CONTEXT_WIN_MIN after offset
      (glucose returns below H_HYPER after episode)
  
  C6  ∃ glucose ≤ H_CONTEXT in before OR after window
      (OR-crossing condition)

  Classification:
    - SLH: Passes ALL criteria (C1–C6 all True)
    - LLH: Fails ANY criterion

Episode detection thresholds
-----------------------------
  H_HYPER = 180 mg/dL   — Hyperglycemia threshold
  H_CONTEXT = 140 mg/dL — Context threshold for crossing checks

Gap merging parameters (if USE_GAP_MERGING = True)
--------------------------------------------------
  MERGE_GAP_MIN = 30 min          — Maximum gap to merge episodes
  MERGE_INTERNAL_GAP_MIN = 15 min — Maximum internal gap within episode
  GAP_FACTOR = 1.5                — Gap detection factor (× timestep)

Context windows
---------------
  Pre-episode:  60 min before onset
  Post-episode: 60 min after offset
  Data completeness check: core + ±60 min windows

"""
import os


import pandas as pd
from tqdm import tqdm

from src.config.paths import (
    WINDOW_SUMMARY_CSV,
    FILTER_STATUS_CSV,
    EPISODES_SUMMARY_CSV,
    EPISODES_PATH,
)

from src.config.constants import (
        PATIENT_ID_COL,
        COL_FILEPATH
)


from src.preprocessing.load_data import load_window_data
from src.filtering.load_filtering import load_filtered_patient_list

from src.episodes.episode_config import H_HYPER, H_CONTEXT
from src.episodes.episode_config import USE_GAP_MERGING, MERGE_GAP_MIN, MERGE_INTERNAL_GAP_MIN, GAP_FACTOR
from src.episodes.episode_config import APPLY_C1, APPLY_C2, APPLY_C3, APPLY_C4, APPLY_C5, APPLY_C6
from src.episodes.episode_config import D_MIN_MIN, D_MAX_MIN
from src.episodes.episode_config import MIN_DATA_COMPLETENESS
from src.episodes.episode_config import NIGHT_START_H, NIGHT_END_H
from src.episodes.episode_config import MIN_EPISODE_POINTS
from src.episodes.episode_config import LABEL_SLH, LABEL_LLH
from src.episodes.episode_config import LABEL_SLH_NON_ISOLATED
from src.episodes.episode_config import LABEL_DAYTIME, LABEL_NIGHTTIME

from src.episodes.criteria import classify_episodes
from src.episodes.detection import detect_hyperglycemia_episodes
from src.episodes.load_episodes import prepare_episode_data

# ─────────────────────────────────────────────────────────────────────────────
# EPISODE TYPE SUMMARY — output path and column definitions
# ─────────────────────────────────────────────────────────────────────────────
EPISODE_TYPE_SUMMARY_CSV = os.path.join(str(EPISODES_PATH), "episode_type_summary.csv")

# Per-patient count columns feeding the type summary, in report order
EPISODE_TYPE_COLUMNS = [
    "classification_SLH",
    "classification_LLH",
    "category_SLH",
    "category_Non_isolated_SLH",
    "category_LLH",
    "category_SLH_Daytime",
    "category_SLH_Nighttime",
]


def count_episode_types(episode_df: pd.DataFrame | None, patient_id: str) -> dict:
    """
    Count episodes of each type for one patient (0 for every type if the
    patient has no episodes at all).

    Covers both the original binary label (``classification``) and the
    paper-definition 3-way label plus day/night split (``slh_category`` /
    ``onset_period``), so both can be compared side by side.
    """
    if episode_df is None or len(episode_df) == 0:
        return {PATIENT_ID_COL: patient_id, **{col: 0 for col in EPISODE_TYPE_COLUMNS}}

    is_cat_slh = episode_df["slh_category"] == LABEL_SLH

    return {
        PATIENT_ID_COL: patient_id,
        "classification_SLH": int((episode_df["classification"] == LABEL_SLH).sum()),
        "classification_LLH": int((episode_df["classification"] == LABEL_LLH).sum()),
        "category_SLH": int(is_cat_slh.sum()),
        "category_Non_isolated_SLH": int((episode_df["slh_category"] == LABEL_SLH_NON_ISOLATED).sum()),
        "category_LLH": int((episode_df["slh_category"] == LABEL_LLH).sum()),
        "category_SLH_Daytime": int((is_cat_slh & (episode_df["onset_period"] == LABEL_DAYTIME)).sum()),
        "category_SLH_Nighttime": int((is_cat_slh & (episode_df["onset_period"] == LABEL_NIGHTTIME)).sum()),
    }


def compute_episode_type_summary(type_counts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the cohort-level episode-type summary: one row per episode type,
    with the total episode count, how many patients have at least one, and
    the distribution (mean, SD, min, Q1, median, Q3, max) of the per-patient
    counts across the WHOLE filtered cohort (patients with zero episodes of
    a type count as 0, they are not excluded from the distribution).
    """
    n_patients = len(type_counts_df)
    desc = type_counts_df[EPISODE_TYPE_COLUMNS].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc = desc.rename(columns={"25%": "q1", "50%": "median", "75%": "q3"})

    summary = pd.DataFrame({
        "episode_type": EPISODE_TYPE_COLUMNS,
        "total_episodes": type_counts_df[EPISODE_TYPE_COLUMNS].sum().values,
        "n_patients": n_patients,
        "n_patients_with_episode": (type_counts_df[EPISODE_TYPE_COLUMNS] > 0).sum().values,
        "mean": desc["mean"].values,
        "std": desc["std"].values,
        "min": desc["min"].values,
        "q1": desc["q1"].values,
        "median": desc["median"].values,
        "q3": desc["q3"].values,
        "max": desc["max"].values,
    })

    numeric_cols = ["mean", "std", "min", "q1", "median", "q3", "max"]
    summary[numeric_cols] = summary[numeric_cols].round(2)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# PATIENT-LEVEL PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_patient_episodes(filepath: str, patient_id: str) -> pd.DataFrame | None:
    """
    Detect and classify episodes for a single patient.
    
    Parameters
    ----------
    filepath : str
        Path to preprocessed window file
    patient_id : str
        Patient identifier
        
    Returns
    -------
    pd.DataFrame or None
        Episode classification table, or None if processing fails
    """
    # Load data
    #df = load_patient_window(filepath)
    df = load_window_data(filepath)  # Use the standardized loading function
    if df is None:
        return None
    
    # Prepare for episode detection
    df = prepare_episode_data(df)
    if len(df) < MIN_EPISODE_POINTS:
        return None
    
    # Detect episodes
    episodes = detect_hyperglycemia_episodes(df)
    
    # Classify episodes
    episode_df = classify_episodes(df, episodes)
    
    # Add patient ID
    episode_df.insert(0, PATIENT_ID_COL, patient_id)
    
    return episode_df


def save_patient_episodes(episode_df: pd.DataFrame, patient_id: str):
    """
    Save episode classification table to CSV.
    
    Parameters
    ----------
    episode_df : pd.DataFrame
        Episode classification table
    patient_id : str
        Patient identifier
    """
    os.makedirs(EPISODES_PATH, exist_ok=True)
    
    output_path = os.path.join(EPISODES_PATH, f"{patient_id}_episodes.csv")
    episode_df.to_csv(output_path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Main episode detection and classification pipeline."""
    merge_mode = "WITH gap merging" if USE_GAP_MERGING else "WITHOUT gap merging (continuous runs)"
    
    print("=" * 70)
    print("  CGM Episode Detection and Classification")
    print("=" * 70)
    print(f"  H_HYPER = {H_HYPER} mg/dL  (episode detection threshold)")
    print(f"  H_CONTEXT = {H_CONTEXT} mg/dL  (context crossing threshold)")
    print()
    print(f"  Episode detection mode: {merge_mode}")
    if USE_GAP_MERGING:
        print(f"    MERGE_GAP_MIN = {MERGE_GAP_MIN} min")
        print(f"    MERGE_INTERNAL_GAP_MIN = {MERGE_INTERNAL_GAP_MIN} min")
        print(f"    GAP_FACTOR = {GAP_FACTOR}")
    print()
    print("  Criteria applied:")
    print(f"    C1: Duration {D_MIN_MIN}-{D_MAX_MIN} min         {'[ENABLED]' if APPLY_C1 else '[DISABLED]'}")
    print(f"    C2: Data completeness ≥{MIN_DATA_COMPLETENESS:.0%}      {'[ENABLED]' if APPLY_C2 else '[DISABLED]'}")
    print(f"    C3: Daytime only (not {NIGHT_START_H:02d}:00-{NIGHT_END_H:02d}:00) {'[ENABLED]' if APPLY_C3 else '[DISABLED]'}")
    print(f"    C4: No pre-hyperglycemia          {'[ENABLED]' if APPLY_C4 else '[DISABLED]'}")
    print(f"    C5: No post-hyperglycemia         {'[ENABLED]' if APPLY_C5 else '[DISABLED]'}")
    print(f"    C6: OR-crossing condition         {'[ENABLED]' if APPLY_C6 else '[DISABLED]'}")
    print("=" * 70)

    # Validate input files exist
    if not os.path.exists(WINDOW_SUMMARY_CSV):
        raise FileNotFoundError(
            f"windows_summary.csv not found at:\n  {WINDOW_SUMMARY_CSV}\n"
            "Run cgm_preprocessing.py first."
        )

    if not os.path.exists(FILTER_STATUS_CSV):
        raise FileNotFoundError(
            f"Filter status CSV not found:\n  {FILTER_STATUS_CSV}\n"
            "Run cgm_metrics.py and cgm_filtering.py first."
        )

    # Load patient list, restricted to the filtered/selected patients
    summary_df = pd.read_csv(WINDOW_SUMMARY_CSV)
    n_with_windows = len(summary_df)

    filtered_pids = load_filtered_patient_list(str(FILTER_STATUS_CSV))
    summary_df = summary_df[summary_df[PATIENT_ID_COL].isin(filtered_pids)].reset_index(drop=True)

    print(f"\n  {n_with_windows} patients with valid windows")
    print(f"  {len(summary_df)} patients after filtering")

    # Process all patients
    total_episodes = 0
    total_slh = 0
    total_llh = 0
    errors = []
    patient_summary = []  # NEW: Collect per-patient summary
    type_counts = []      # Collect per-patient counts for episode_type_summary.csv

    for _, row in tqdm(summary_df.iterrows(),
                       total=len(summary_df),
                       desc="  Detecting episodes",
                       ncols=72):

        patient_id = row[PATIENT_ID_COL]
        filepath = row[COL_FILEPATH]

        try:
            # Process patient
            episode_df = process_patient_episodes(filepath, patient_id)

            if episode_df is None or len(episode_df) == 0:
                # NEW: Record zero episodes for this patient
                patient_summary.append({
                    PATIENT_ID_COL: patient_id,
                    "n_SLH": 0,
                    "n_LLH": 0,
                })
                type_counts.append(count_episode_types(None, patient_id))
                continue

            # Save results
            save_patient_episodes(episode_df, patient_id)

            # Update counts
            n_episodes = len(episode_df)
            n_slh = (episode_df["classification"] == LABEL_SLH).sum()
            n_llh = (episode_df["classification"] == LABEL_LLH).sum()

            total_episodes += n_episodes
            total_slh += n_slh
            total_llh += n_llh

            # NEW: Record per-patient counts
            patient_summary.append({
                PATIENT_ID_COL: patient_id,
                "n_SLH": int(n_slh),
                "n_LLH": int(n_llh),
            })
            type_counts.append(count_episode_types(episode_df, patient_id))

        except Exception as e:
            errors.append(f"  {patient_id}: {e}")

    # NEW: Save patient summary file
    if patient_summary:
        summary_output_df = pd.DataFrame(patient_summary)
        summary_output_df.to_csv(EPISODES_SUMMARY_CSV, index=False)

    # NEW: Save episode-type summary (counts + descriptive stats per type)
    if type_counts:
        type_counts_df = pd.DataFrame(type_counts)
        type_summary_df = compute_episode_type_summary(type_counts_df)
        type_summary_df.to_csv(EPISODE_TYPE_SUMMARY_CSV, index=False)

    # Print warnings if any
    if errors:
        print("\n  Warnings:")
        for error in errors:
            print(error)

    # Print summary
    print()
    print("=" * 70)
    print("  EPISODE DETECTION COMPLETE")
    print("=" * 70)
    print(f"  Mode: {merge_mode}")
    print(f"  Total episodes detected: {total_episodes}")
    print(f"    SLH (Short-Lasting):   {total_slh}  ({total_slh/total_episodes*100:.1f}%)" if total_episodes > 0 else "    SLH: 0")
    print(f"    LLH (Long-Lasting):    {total_llh}  ({total_llh/total_episodes*100:.1f}%)" if total_episodes > 0 else "    LLH: 0")
    print()
    print(f"  Output: {EPISODES_PATH}")
    print(f"  Files: {{patient_id}}_episodes.csv")
    print(f"  Summary: {EPISODES_SUMMARY_CSV}")  # NEW
    print(f"  Episode type summary: {EPISODE_TYPE_SUMMARY_CSV}")  # NEW
    print("=" * 70)


if __name__ == "__main__":
    main()