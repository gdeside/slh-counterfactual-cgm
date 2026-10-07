"""
src/episodes/criteria.py
========================
Criterion functions (C1–C6) for SLH/LLH episode classification.

Each criterion is implemented as a standalone function that accepts a list of
detected episode dicts and returns a parallel list of booleans.  The top-level
``classify_episodes`` and ``classify_episodes_sequential`` functions compose
them into final SLH / LLH labels.

Criteria
--------
C1 : Duration bounds        – episode length within [D_MIN_MIN, D_MAX_MIN]
C2 : Data completeness      – ≥ MIN_DATA_COMPLETENESS in core + context windows
C3 : Daytime only           – onset outside night hours
C4 : No pre-episode hyper   – no glucose > H_HYPER in context window before onset
C5 : No post-episode hyper  – no glucose > H_HYPER in context window after offset
C6 : Threshold crossing     – glucose ≤ H_CONTEXT in at least one context window

Authors
-------
Guillaume Deside

Date
----
2025
"""

import pandas as pd

from src.config import TIME_COL, GLUCOSE_COL


from src.episodes.episode_config import H_HYPER, H_CONTEXT
from src.episodes.episode_config import APPLY_C1, APPLY_C2, APPLY_C3, APPLY_C4, APPLY_C5, APPLY_C6
from src.episodes.episode_config import D_MIN_MIN, D_MAX_MIN
from src.episodes.episode_config import MIN_DATA_COMPLETENESS, CONTEXT_WIN_MIN
from src.episodes.episode_config import NIGHT_START_H, NIGHT_END_H
from src.episodes.episode_config import REQUIRED_SAMPLING_MIN, MIN_EPISODE_POINTS
from src.episodes.episode_config import LABEL_SLH, LABEL_LLH
from src.episodes.episode_config import LABEL_SLH_NON_ISOLATED
from src.episodes.episode_config import LABEL_DAYTIME, LABEL_NIGHTTIME

# ─────────────────────────────────────────────────────────────────────────────
# CRITERION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def check_c1_duration_bounds(episodes: list[dict]) -> list[bool]:
    """
    C1: Check episode duration is within bounds.
    
    Duration (measured as n_pts × timestep) must be: D_MIN_MIN ≤ duration ≤ D_MAX_MIN
    
    Parameters
    ----------
    episodes : list of dict
        Detected episodes
        
    Returns
    -------
    list of bool
        True if episode passes C1 (appropriate duration)
    """
    if not APPLY_C1:
        return [True] * len(episodes)
    
    passes = []
    
    for episode in episodes:
        duration = episode["duration_min"]
        passes.append(D_MIN_MIN <= duration <= D_MAX_MIN)
    
    return passes

def calculate_data_completeness(df: pd.DataFrame, t_start: pd.Timestamp,
                                t_end: pd.Timestamp, timestep_min: float) -> float:
    """
    Calculate data completeness as fraction of expected measurements present.
    """
    if t_end <= t_start:
        return 0.0
    
    duration_min = (t_end - t_start).total_seconds() / 60.0
    expected_n = int(duration_min / timestep_min) + 1
    
    if expected_n <= 0:
        return 1.0
    
    # Count actual non-NaN glucose values in the window
    mask = (df[TIME_COL] >= t_start) & (df[TIME_COL] <= t_end)
    actual_n = df.loc[mask, GLUCOSE_COL].notna().sum()
    
    return min(actual_n / expected_n, 1.0)

def check_c2_data_completeness(df: pd.DataFrame, episodes: list[dict]) -> list[bool]:
    """
    C2: Check data completeness in core episode + context windows.
    
    Requires ≥ MIN_DATA_COMPLETENESS coverage in ALL three parts:
    - Episode core:      [onset, offset]
    - Before context:    [onset - CONTEXT_WIN_MIN, onset]
    - After context:     [offset, offset + CONTEXT_WIN_MIN]
        
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data
    episodes : list of dict
        Detected episodes
        
    Returns
    -------
    list of bool
        True if episode passes C2 (adequate data coverage in core + both context windows)
    """
    if not APPLY_C2:
        return [True] * len(episodes)
    
    passes = []
    
    for episode in episodes:
        onset_time = episode["onset_time"]
        offset_time = episode["offset_time"]
        
        # Calculate completeness for each window separately
        completeness_episode = calculate_data_completeness(
            df, onset_time, offset_time, REQUIRED_SAMPLING_MIN
        )
        
        completeness_before = calculate_data_completeness(
            df,
            onset_time - pd.Timedelta(minutes=CONTEXT_WIN_MIN),
            onset_time,
            REQUIRED_SAMPLING_MIN
        )
        
        completeness_after = calculate_data_completeness(
            df,
            offset_time,
            offset_time + pd.Timedelta(minutes=CONTEXT_WIN_MIN),
            REQUIRED_SAMPLING_MIN
        )
        
        # Pass only if ALL three windows have sufficient data
        passes_c1 = (
            completeness_episode >= MIN_DATA_COMPLETENESS and
            completeness_before >= MIN_DATA_COMPLETENESS and
            completeness_after >= MIN_DATA_COMPLETENESS
        )
        
        passes.append(passes_c1)
    
    return passes

def check_c3_daytime_only(episodes: list[dict]) -> list[bool]:
    """
    C3: Check that episode occurs during daytime.

    Onset must be OUTSIDE night hours [NIGHT_START_H, NIGHT_END_H).
    Night is defined as 22:00–06:00. Only the onset hour is checked, matching
    the paper's day/night rule ("SLH episodes starting after 6 AM and before
    10 PM were classified as daytime SLH... nighttime SLH otherwise").

    Parameters
    ----------
    episodes : list of dict
        Detected episodes

    Returns
    -------
    list of bool
        True if episode passes C3 (occurs during daytime)
    """
    if not APPLY_C3:
        return [True] * len(episodes)

    passes = []

    for episode in episodes:
        onset_hour = episode["onset_time"].hour
        onset_is_night = onset_hour >= NIGHT_START_H or onset_hour < NIGHT_END_H
        passes.append(not onset_is_night)

    return passes


def check_c4_no_pre_episode_hyperglycemia(df: pd.DataFrame, episodes: list[dict]) -> list[bool]:
    """
    C4: Check no hyperglycemia before episode.
    
    No glucose > H_HYPER in CONTEXT_WIN_MIN before episode onset.
    Episode must not be preceded by sustained hyperglycemia.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data
    episodes : list of dict
        Detected episodes
        
    Returns
    -------
    list of bool
        True if episode passes C4 (no pre-episode hyperglycemia)
    """
    if not APPLY_C4:
        return [True] * len(episodes)
    
    passes = []
    
    for episode in episodes:
        onset_time = episode["onset_time"]
        
        # Define pre-episode window
        window_start = onset_time - pd.Timedelta(minutes=CONTEXT_WIN_MIN)
        window_end = onset_time
        
        # Extract data in window
        window_mask = (df[TIME_COL] >= window_start) & (df[TIME_COL] < window_end)
        window_data = df[window_mask]
        
        if len(window_data) == 0:
            # No data in window - cannot verify criterion
            passes.append(False)
            continue
        
        # Check if ANY glucose > H_HYPER
        has_hyperglycemia = (window_data[GLUCOSE_COL] > H_HYPER).any()
        
        # Episode passes if there is NO hyperglycemia
        passes.append(not has_hyperglycemia)
    
    return passes


def check_c5_no_post_episode_hyperglycemia(df: pd.DataFrame, episodes: list[dict]) -> list[bool]:
    """
    C5: Check glucose returns below H_HYPER after episode.
    
    No glucose > H_HYPER in CONTEXT_WIN_MIN after episode offset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data
    episodes : list of dict
        Detected episodes
        
    Returns
    -------
    list of bool
        True if episode passes C5 (no post-episode hyperglycemia)
    """
    if not APPLY_C5:
        return [True] * len(episodes)
    
    passes = []
    
    for episode in episodes:
        offset_time = episode["offset_time"]
        
        # Define post-episode window
        window_start = offset_time
        window_end = offset_time + pd.Timedelta(minutes=CONTEXT_WIN_MIN)
        
        # Extract data in window
        window_mask = (df[TIME_COL] > window_start) & (df[TIME_COL] <= window_end)
        window_data = df[window_mask]
        
        if len(window_data) == 0:
            # No data in window - cannot verify criterion
            passes.append(False)
            continue
        
        # Check if ANY glucose > H_HYPER
        has_hyperglycemia = (window_data[GLUCOSE_COL] > H_HYPER).any()
        
        # Episode passes if there is NO hyperglycemia
        passes.append(not has_hyperglycemia)
    
    return passes


def check_c6_or_crossing_condition(df: pd.DataFrame, episodes: list[dict]) -> list[bool]:
    """
    C6: Check OR-crossing condition.
    
    There exists glucose ≤ H_CONTEXT in before OR after window.
    Episode must cross the H_CONTEXT threshold in at least one direction.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data
    episodes : list of dict
        Detected episodes
        
    Returns
    -------
    list of bool
        True if episode passes C6 (crosses H_CONTEXT threshold)
    """
    if not APPLY_C6:
        return [True] * len(episodes)
    
    passes = []
    
    for episode in episodes:
        onset_time = episode["onset_time"]
        offset_time = episode["offset_time"]
        
        # Define pre-episode window
        before_start = onset_time - pd.Timedelta(minutes=CONTEXT_WIN_MIN)
        before_end = onset_time
        
        # Define post-episode window
        after_start = offset_time
        after_end = offset_time + pd.Timedelta(minutes=CONTEXT_WIN_MIN)
        
        # Extract before data
        before_mask = (df[TIME_COL] >= before_start) & (df[TIME_COL] < before_end)
        before_data = df[before_mask]
        
        # Extract after data
        after_mask = (df[TIME_COL] > after_start) & (df[TIME_COL] <= after_end)
        after_data = df[after_mask]
        
        # Check crossing in before window
        crosses_before = False
        if len(before_data) > 0:
            crosses_before = (before_data[GLUCOSE_COL] <= H_CONTEXT).any()
        
        # Check crossing in after window
        crosses_after = False
        if len(after_data) > 0:
            crosses_after = (after_data[GLUCOSE_COL] <= H_CONTEXT).any()
        
        # Episode passes if it crosses in EITHER before OR after (or both)
        passes.append(crosses_before or crosses_after)
    
    return passes


def _compute_c6_crossing_details(df: pd.DataFrame, episodes: list[dict]) -> list[tuple[bool, bool]]:
    """Return (crosses_before, crosses_after) for each episode, regardless of APPLY_C6."""
    ctx = pd.Timedelta(minutes=CONTEXT_WIN_MIN)
    details = []
    for ep in episodes:
        before_mask = (df[TIME_COL] >= ep["onset_time"] - ctx) & (df[TIME_COL] < ep["onset_time"])
        after_mask  = (df[TIME_COL] > ep["offset_time"])        & (df[TIME_COL] <= ep["offset_time"] + ctx)
        crosses_before = bool((df.loc[before_mask, GLUCOSE_COL] <= H_CONTEXT).any()) if before_mask.any() else False
        crosses_after  = bool((df.loc[after_mask,  GLUCOSE_COL] <= H_CONTEXT).any()) if after_mask.any()  else False
        details.append((crosses_before, crosses_after))
    return details


def _classify_onset_period(onset_time) -> str:
    """
    Daytime/Nighttime tag from onset hour alone (independent of APPLY_C3),
    so it always reflects the actual onset time regardless of whether the
    daytime criterion is toggled on/off for the main SLH/LLH label.

    Night is [NIGHT_START_H, NIGHT_END_H) = 22:00-06:00, matching the paper's
    rule (onset after 6 AM and before 10 PM -> daytime, else nighttime).
    """
    onset_hour = onset_time.hour
    is_night = onset_hour >= NIGHT_START_H or onset_hour < NIGHT_END_H
    return LABEL_NIGHTTIME if is_night else LABEL_DAYTIME


def _classify_slh_category(c1: bool, c2: bool, c4: bool, c5: bool, c6: bool) -> str:
    """
    3-way category matching the paper's definition verbatim (day/night is
    NOT part of this — see `_classify_onset_period` for that):

      1. Duration (C1) fails                    -> LLH
      2. Duration passes, isolation/completeness
         (C2 and C4 and C5 and C6) fails         -> Non-isolated SLH
      3. Duration, isolation, completeness all
         pass                                    -> SLH
    """
    if not c1:
        return LABEL_LLH

    isolated_and_complete = c2 and c4 and c5 and c6
    if not isolated_and_complete:
        return LABEL_SLH_NON_ISOLATED

    return LABEL_SLH


# ─────────────────────────────────────────────────────────────────────────────
# EPISODE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def classify_episodes(df: pd.DataFrame, episodes: list[dict]) -> pd.DataFrame:
    """
    Classify detected episodes as SLH or LLH based on criteria C1-C6.

    Episodes that pass all criteria (C1-C6, daytime included) are classified
    as SLH (Short-Lasting Hyperglycemia). Episodes that fail any criterion are
    classified as LLH (Long-Lasting Hyperglycemia). This is the original
    binary label, kept unchanged for backward compatibility with everything
    downstream that reads it (Num_SLH, Table 1, etc.).

    Two additional columns give the paper's literal definition on top of it,
    without changing the columns above:
    - `slh_category` : SLH / Non-isolated SLH / LLH, based on C1 (duration)
      and C2+C4+C5+C6 (completeness + isolation) only -- daytime is NOT a
      requirement here, matching the paper's 3-way definition.
    - `onset_period`  : Daytime / Nighttime, from onset hour alone -- the
      paper's sub-classification "among SLHs", but populated for every
      episode so it can be cross-tabulated with `slh_category` freely.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data
    episodes : list of dict
        Detected episodes

    Returns
    -------
    pd.DataFrame
        Episode classification table with columns:
        - episode_id: sequential ID
        - onset_time, offset_time: episode boundaries
        - duration_min: episode duration
        - n_points: number of measurements
        - C1, C2, C3, C4, C5, C6: criterion pass/fail flags
        - classification: SLH or LLH (original binary label, unchanged)
        - slh_category: SLH / Non-isolated SLH / LLH (paper's 3-way definition)
        - onset_period: Daytime / Nighttime (onset-hour based)
    """
    if len(episodes) == 0:
        # Return empty dataframe with proper columns
        return pd.DataFrame(columns=[
            "episode_id", "onset_time", "offset_time", "duration_min", "n_points",
            "C1", "C2", "C3", "C4", "C5", "C6",
            "c6_crosses_before", "c6_crosses_after", "classification",
            "slh_category", "onset_period",
        ])
    
    # Apply all criteria
    c1_results = check_c1_duration_bounds(episodes)
    c2_results = check_c2_data_completeness(df, episodes)
    c3_results = check_c3_daytime_only(episodes)
    c4_results = check_c4_no_pre_episode_hyperglycemia(df, episodes)
    c5_results = check_c5_no_post_episode_hyperglycemia(df, episodes)
    c6_results = check_c6_or_crossing_condition(df, episodes)
    c6_crossing_details = _compute_c6_crossing_details(df, episodes)

    # Build classification table
    rows = []
    for i, episode in enumerate(episodes):
        # Check if all criteria pass
        all_pass = (
            c1_results[i] and
            c2_results[i] and
            c3_results[i] and
            c4_results[i] and
            c5_results[i] and
            c6_results[i]
        )
        
        classification = LABEL_SLH if all_pass else LABEL_LLH

        slh_category = _classify_slh_category(
            c1_results[i], c2_results[i], c4_results[i], c5_results[i], c6_results[i]
        )
        onset_period = _classify_onset_period(episode["onset_time"])

        rows.append({
            "episode_id": i + 1,
            "onset_time": episode["onset_time"],
            "offset_time": episode["offset_time"],
            "duration_min": round(episode["duration_min"], 2),
            "n_points": episode["n_points"],
            "C1": c1_results[i],
            "C2": c2_results[i],
            "C3": c3_results[i],
            "C4": c4_results[i],
            "C5": c5_results[i],
            "C6": c6_results[i],
            "c6_crosses_before": c6_crossing_details[i][0],
            "c6_crosses_after":  c6_crossing_details[i][1],
            "classification": classification,
            "slh_category": slh_category,
            "onset_period": onset_period,
        })
    
    return pd.DataFrame(rows)



def classify_episodes_sequential(df: pd.DataFrame, episodes: list[dict]) -> dict:
    """
    Classify episodes sequentially and track statistics.
    
    Episodes are filtered C1 → C2 → C3 → C4 → C5 → C6, with counts
    tracked at each stage for visualization.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed CGM data with TIME_COL and GLUCOSE_COL
    episodes : list of dict
        Raw detected episodes from detect_hyperglycemia_episodes
        
    Returns
    -------
    dict
        Funnel statistics with keys:
        - n_raw: Total episodes detected
        - n_c1, n_c2, n_c3, n_c4, n_c5: Episodes surviving each criterion
        - n_slh: Final SLH count
        - n_before_only, n_after_only, n_both: OR-crossing breakdown
        - excluded_at_each_stage: Episodes excluded at each criterion
    """
    if len(episodes) == 0:
        return {
            "n_raw": 0, "n_c1": 0, "n_c2": 0, "n_c3": 0,
            "n_c4": 0, "n_c5": 0, "n_slh": 0,
            "n_before_only": 0, "n_after_only": 0, "n_both": 0,
            "n_no_cross": 0,
            "excluded_at_each_stage": {
                "C1": 0, "C2": 0, "C3": 0, "C4": 0, "C5": 0, "C6": 0
            }
        }
    
    n_raw = len(episodes)
    active = [True] * n_raw  # Episodes still being evaluated
    
    # Track exclusions
    excluded_at = {
        "C1": 0, "C2": 0, "C3": 0, "C4": 0, "C5": 0, "C6": 0
    }
    
    # Crossing breakdown for SLH episodes
    n_before_only = 0
    n_after_only = 0
    n_both = 0
    n_no_cross = 0
    
    # Helper to apply criterion
    def apply_criterion(criterion_name, check_function, episodes_subset):
        """Apply criterion and return results."""
        if criterion_name == "C1":
            return check_function(episodes_subset)
        elif criterion_name in ["C2", "C4", "C5", "C6"]:
            return check_function(df, episodes_subset)
        else:  # C3
            return check_function(episodes_subset)
    
    # Sequential criteria application
    criteria = [
        ("C1", check_c1_duration_bounds, APPLY_C1),
        ("C2", check_c2_data_completeness, APPLY_C2),
        ("C3", check_c3_daytime_only, APPLY_C3),
        ("C4", check_c4_no_pre_episode_hyperglycemia, APPLY_C4),
        ("C5", check_c5_no_post_episode_hyperglycemia, APPLY_C5),
    ]
    
    for criterion_name, check_function, apply_flag in criteria:
        if not apply_flag:
            continue
        
        # Get active episodes
        active_episodes = [ep for i, ep in enumerate(episodes) if active[i]]
        active_indices = [i for i in range(n_raw) if active[i]]
        
        if not active_episodes:
            break
        
        # Apply criterion
        results = apply_criterion(criterion_name, check_function, active_episodes)
        
        # Update active status
        for idx, result in zip(active_indices, results):
            if not result:
                active[idx] = False
                excluded_at[criterion_name] += 1
    
    # Count survivors after C1-C5
    n_c1 = n_raw - excluded_at["C1"]
    n_c2 = n_c1 - excluded_at["C2"]
    n_c3 = n_c2 - excluded_at["C3"]
    n_c4 = n_c3 - excluded_at["C4"]
    n_c5 = n_c4 - excluded_at["C5"]
    
    # C6: OR-crossing condition (special handling for breakdown)
    if APPLY_C6:
        active_episodes = [ep for i, ep in enumerate(episodes) if active[i]]
        active_indices = [i for i in range(n_raw) if active[i]]
        
        if active_episodes:
            c6_results = check_c6_or_crossing_condition(df, active_episodes)
            
            # For SLH episodes, check crossing pattern
            for idx, (ep, result) in zip(active_indices, zip(active_episodes, c6_results)):
                if result:
                    # Episode passes C6 - it's SLH
                    # Determine crossing pattern
                    ctx = pd.Timedelta(minutes=CONTEXT_WIN_MIN)
                    
                    # Check before window
                    mask_before = (df[TIME_COL] >= ep["onset_time"] - ctx) & \
                                 (df[TIME_COL] <= ep["onset_time"])
                    crosses_before = (df.loc[mask_before, GLUCOSE_COL] <= H_CONTEXT).any()
                    
                    # Check after window
                    mask_after = (df[TIME_COL] >= ep["offset_time"]) & \
                                (df[TIME_COL] <= ep["offset_time"] + ctx)
                    crosses_after = (df.loc[mask_after, GLUCOSE_COL] <= H_CONTEXT).any()
                    
                    if crosses_before and crosses_after:
                        n_both += 1
                    elif crosses_before:
                        n_before_only += 1
                    else:
                        n_after_only += 1
                else:
                    # Episode fails C6
                    active[idx] = False
                    excluded_at["C6"] += 1
                    n_no_cross += 1
    
    n_slh = n_c5 - excluded_at["C6"]
    
    return {
        "n_raw": n_raw,
        "n_c1": n_c1,
        "n_c2": n_c2,
        "n_c3": n_c3,
        "n_c4": n_c4,
        "n_c5": n_c5,
        "n_slh": n_slh,
        "n_before_only": n_before_only,
        "n_after_only": n_after_only,
        "n_both": n_both,
        "n_no_cross": n_no_cross,
        "excluded_at_each_stage": excluded_at,
    }