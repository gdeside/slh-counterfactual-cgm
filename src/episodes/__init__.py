"""
src/episodes/__init__.py
========================

Public API for the episodes module.

Exposes configuration constants, detection functions, classification
functions, and I/O helpers so callers can import everything from a single
location:

    from src.episodes import (
        # Config
        H_HYPER, H_CONTEXT, H_HYPO,
        D_MIN_MIN, D_MAX_MIN,
        LABEL_SLH, LABEL_LLH,
        # Detection
        detect_hyperglycemia_episodes,
        # Classification
        classify_episodes,
        classify_episodes_sequential,
        # I/O
        prepare_episode_data,
        load_patient_episodes,
        load_episode_summary,
        discover_episode_files,
    )
"""

from src.episodes.episode_config import (
    H_HYPER,
    H_CONTEXT,
    H_HYPO,
    USE_GAP_MERGING,
    MERGE_GAP_MIN,
    MERGE_INTERNAL_GAP_MIN,
    GAP_FACTOR,
    APPLY_C1,
    APPLY_C2,
    APPLY_C3,
    APPLY_C4,
    APPLY_C5,
    APPLY_C6,
    D_MIN_MIN,
    D_MAX_MIN,
    MIN_DATA_COMPLETENESS,
    CONTEXT_WIN_MIN,
    NIGHT_START_H,
    NIGHT_END_H,
    REQUIRED_SAMPLING_MIN,
    MIN_EPISODE_POINTS,
    LABEL_SLH,
    LABEL_LLH,
)

from src.episodes.load_episodes import (
    prepare_episode_data,
    load_patient_episodes,
    load_episode_summary,
    discover_episode_files,
)

from src.episodes.detection import (
    detect_hyperglycemia_episodes,
    detect_hyperglycemia_episodes_simple,
    detect_hyperglycemia_episodes_with_gaps,
)

from src.episodes.criteria import (
    check_c1_duration_bounds,
    check_c2_data_completeness,
    check_c3_daytime_only,
    check_c4_no_pre_episode_hyperglycemia,
    check_c5_no_post_episode_hyperglycemia,
    check_c6_or_crossing_condition,
    classify_episodes,
    classify_episodes_sequential,
)

__all__ = [
    # Configuration
    "H_HYPER",
    "H_CONTEXT",
    "H_HYPO",
    "USE_GAP_MERGING",
    "MERGE_GAP_MIN",
    "MERGE_INTERNAL_GAP_MIN",
    "GAP_FACTOR",
    "APPLY_C1",
    "APPLY_C2",
    "APPLY_C3",
    "APPLY_C4",
    "APPLY_C5",
    "APPLY_C6",
    "D_MIN_MIN",
    "D_MAX_MIN",
    "MIN_DATA_COMPLETENESS",
    "CONTEXT_WIN_MIN",
    "NIGHT_START_H",
    "NIGHT_END_H",
    "REQUIRED_SAMPLING_MIN",
    "MIN_EPISODE_POINTS",
    "LABEL_SLH",
    "LABEL_LLH",
    # I/O
    "prepare_episode_data",
    "load_patient_episodes",
    "load_episode_summary",
    "discover_episode_files",
    # Detection
    "detect_hyperglycemia_episodes",
    "detect_hyperglycemia_episodes_simple",
    "detect_hyperglycemia_episodes_with_gaps",
    # Classification
    "check_c1_duration_bounds",
    "check_c2_data_completeness",
    "check_c3_daytime_only",
    "check_c4_no_pre_episode_hyperglycemia",
    "check_c5_no_post_episode_hyperglycemia",
    "check_c6_or_crossing_condition",
    "classify_episodes",
    "classify_episodes_sequential",
]
