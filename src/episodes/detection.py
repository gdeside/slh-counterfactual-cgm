"""
src/episodes/detection.py
=========================
Hyperglycemia episode detection from preprocessed CGM time series.

Two detection strategies are provided and selected via ``USE_GAP_MERGING``:

* **Simple** (``detect_hyperglycemia_episodes_simple``) – identifies
  continuous runs of glucose > H_HYPER with no gap handling.

* **Gap-aware** (``detect_hyperglycemia_episodes_with_gaps``) – splits the
  signal at large sensor-dropout gaps, then merges short adjacent episodes
  that are separated by less than ``MERGE_GAP_MIN`` minutes.

The public entry point ``detect_hyperglycemia_episodes`` dispatches to the
appropriate strategy.

Authors
-------
Guillaume Deside

Date
----
2025
"""

import pandas as pd


from src.config import TIME_COL, GLUCOSE_COL


from src.episodes.episode_config import H_HYPER
from src.episodes.episode_config import USE_GAP_MERGING, MERGE_GAP_MIN, MERGE_INTERNAL_GAP_MIN, GAP_FACTOR

from src.episodes.episode_config import REQUIRED_SAMPLING_MIN, MIN_EPISODE_POINTS


def detect_hyperglycemia_episodes_simple(df: pd.DataFrame) -> list[dict]:
    """
    Detect hyperglycemia episodes as continuous runs above H_HYPER threshold.
    
    Simple detection: episodes are continuous sequences of glucose > H_HYPER
    with no gap merging. Any break in the sequence creates a new episode.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data with TIME_COL and GLUCOSE_COL columns
        
    Returns
    -------
    list of dict
        Each dict contains:
        - onset_idx: start index
        - offset_idx: end index (inclusive)
        - onset_time: start timestamp
        - offset_time: end timestamp
        - duration_min: duration in minutes
        - n_points: number of points in episode
        - idx: list of original DataFrame indices belonging to the episode
    """
    if len(df) == 0:
        return []
    
    episodes = []
    
    # Find runs above threshold
    above_threshold = df[GLUCOSE_COL] > H_HYPER
    
    # Identify episode boundaries
    in_episode = False
    onset_idx = None
    
    for idx in range(len(df)):
        if above_threshold.iloc[idx] and not in_episode:
            # Episode starts
            in_episode = True
            onset_idx = idx
            
        elif not above_threshold.iloc[idx] and in_episode:
            # Episode ends
            offset_idx = idx - 1
            
            # Check minimum duration
            if offset_idx - onset_idx + 1 >= MIN_EPISODE_POINTS:
                onset_time = df.loc[onset_idx, TIME_COL]
                offset_time = df.loc[offset_idx, TIME_COL]
                
                # Calculate duration as n_points × timestep (gap-robust)
                n_points = offset_idx - onset_idx + 1
                duration_min = n_points * REQUIRED_SAMPLING_MIN
                
                episodes.append({
                    "onset_idx": onset_idx,
                    "offset_idx": offset_idx,
                    "onset_time": onset_time,
                    "offset_time": offset_time,
                    "duration_min": duration_min,
                    "n_points": n_points,
                    "idx": list(range(onset_idx, offset_idx + 1)),
                })
            
            in_episode = False
            onset_idx = None
    
    # Handle episode that extends to end of data
    if in_episode and onset_idx is not None:
        offset_idx = len(df) - 1
        if offset_idx - onset_idx + 1 >= MIN_EPISODE_POINTS:
            onset_time = df.loc[onset_idx, TIME_COL]
            offset_time = df.loc[offset_idx, TIME_COL]
            
            n_points = offset_idx - onset_idx + 1
            duration_min = n_points * REQUIRED_SAMPLING_MIN
            
            episodes.append({
                "onset_idx": onset_idx,
                "offset_idx": offset_idx,
                "onset_time": onset_time,
                "offset_time": offset_time,
                "duration_min": duration_min,
                "n_points": n_points,
                "idx": list(range(onset_idx, offset_idx + 1)),
            })
    
    return episodes


def detect_hyperglycemia_episodes_with_gaps(df: pd.DataFrame) -> list[dict]:
    """
    Detect hyperglycemia episodes with gap-aware merging.
    
    This is a detection that handles:
    1. Splits data into blocks at large time gaps (> GAP_FACTOR x timestep)
    2. Within each block, identifies continuous runs above threshold
    3. Splits runs at internal gaps > MERGE_INTERNAL_GAP_MIN
    4. Merges adjacent episodes if separated by < MERGE_GAP_MIN
    
    This approach is more robust to sensor dropouts and recognizes that
    clustered hyperglycemic events may represent a single physiological episode.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data with TIME_COL and GLUCOSE_COL columns
        
    Returns
    -------
    list of dict
        Each dict contains:
        - onset_idx: start index
        - offset_idx: end index (inclusive)
        - onset_time: start timestamp
        - offset_time: end timestamp
        - duration_min: duration in minutes
        - n_points: number of points in episode
        - idx: list of original DataFrame indices belonging to the episode
    """
    if len(df) == 0:
        return []
    
    df = df.copy()
    
    # Split into blocks at large gaps
    diffs_min = df[TIME_COL].diff().dt.total_seconds() / 60
    df["_blk"] = (diffs_min > REQUIRED_SAMPLING_MIN * GAP_FACTOR).cumsum()
    
    episodes = []
    
    for _, blk in df.groupby("_blk", sort=True):
        above = blk[GLUCOSE_COL] > H_HYPER
        if not above.any():
            continue
        
        # Identify runs above threshold
        run_id = above.ne(above.shift()).cumsum()
        runs = []
        
        for _, grp in blk[above].groupby(run_id[above]):
            # Check for internal gaps
            t_sorted = grp[TIME_COL].sort_values()
            int_gaps = t_sorted.diff().dt.total_seconds().fillna(0) / 60
            large_gaps = int_gaps[int_gaps > MERGE_INTERNAL_GAP_MIN]
            
            if large_gaps.empty:
                # No internal gaps - single run
                runs.append({
                    "onset": grp[TIME_COL].iloc[0],
                    "offset": grp[TIME_COL].iloc[-1],
                    "idx": list(grp.index),
                })
            else:
                # Split at internal gaps
                sub_idx = list(grp.index)
                split_positions = [0] + [sub_idx.index(i) for i in large_gaps.index] + [len(sub_idx)]
                
                for a, b in zip(split_positions[:-1], split_positions[1:]):
                    piece = sub_idx[a:b]
                    if piece:
                        runs.append({
                            "onset": df.loc[piece[0], TIME_COL],
                            "offset": df.loc[piece[-1], TIME_COL],
                            "idx": piece,
                        })
        
        if not runs:
            continue
        
        # Merge adjacent runs if gap < MERGE_GAP_MIN
        merged = [runs[0].copy()]
        for nxt in runs[1:]:
            gap = (nxt["onset"] - merged[-1]["offset"]).total_seconds() / 60
            if gap <= MERGE_GAP_MIN:
                # Merge episodes
                merged[-1]["offset"] = nxt["offset"]
                merged[-1]["idx"].extend(nxt["idx"])
            else:
                # Keep as separate episode
                merged.append(nxt.copy())
        
        episodes.extend(merged)
    
    # Convert to standard format and calculate durations
    formatted_episodes = []
    for ep in episodes:
        if len(ep["idx"]) < MIN_EPISODE_POINTS:
            continue
        
        # Get indices as sorted list
        idx_list = sorted(ep["idx"])
        onset_idx = idx_list[0]
        offset_idx = idx_list[-1]
        
        # Calculate duration (gap-robust: count points above threshold)
        rows = df.loc[idx_list]
        n_points = int((rows[GLUCOSE_COL] > H_HYPER).sum())
        duration_min = n_points * REQUIRED_SAMPLING_MIN
        
        formatted_episodes.append({
            "onset_idx": onset_idx,
            "offset_idx": offset_idx,
            "onset_time": ep["onset"],
            "offset_time": ep["offset"],
            "duration_min": duration_min,
            "n_points": n_points,
            "idx": idx_list,
        })
    
    return formatted_episodes


def detect_hyperglycemia_episodes(df: pd.DataFrame) -> list[dict]:
    """
    Detect hyperglycemia episodes using the configured method.
    
    Dispatches to either simple continuous-run detection or gap-aware
    detection based on the USE_GAP_MERGING parameter.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data with TIME_COL and GLUCOSE_COL columns
        
    Returns
    -------
    list of dict
        Detected hyperglycemic episodes with full metadata
    """
    if USE_GAP_MERGING:
        return detect_hyperglycemia_episodes_with_gaps(df)
    else:
        return detect_hyperglycemia_episodes_simple(df)