H_HYPER = 180      # Hyperglycemia threshold for episode detection
H_CONTEXT = 140    # Context threshold for crossing checks (C6)
H_HYPO = 70        # Hypoglycemia threshold (for reference line)

# ── Gap-based episode merging ─────────────────────────────────────────────────
USE_GAP_MERGING = True        # Enable/disable gap-based episode merging
MERGE_GAP_MIN = 30             # Maximum gap to merge episodes (minutes)
MERGE_INTERNAL_GAP_MIN = 15    # Maximum internal gap within episode (minutes)
GAP_FACTOR = 1.5               # Gap detection factor (× timestep)

# ── Criterion toggles ─────────────────────────────────────────────────────────
# Set to False to disable a criterion (for sensitivity analysis)
APPLY_C1 = True    # Duration bounds
APPLY_C2 = True    # Data completeness
APPLY_C3 = True    # Daytime only
APPLY_C4 = True    # No pre-episode hyperglycemia
APPLY_C5 = True    # No post-episode hyperglycemia
APPLY_C6 = True    # OR-crossing condition

# ── C1: Duration bounds ───────────────────────────────────────────────────────
D_MIN_MIN = 10     # Minimum episode duration (minutes)
D_MAX_MIN = 60    # Maximum episode duration (minutes)

# ── C2: Data completeness parameters ─────────────────────────────────────────
MIN_DATA_COMPLETENESS = 0.75   # Minimum coverage fraction (75%)
CONTEXT_WIN_MIN = 60           # Context window duration (minutes)

# ── C3: Daytime restriction ───────────────────────────────────────────────────
NIGHT_START_H = 22     # Start of night (22:00)
NIGHT_END_H = 6        # End of night (06:00)


# ── Processing parameters ─────────────────────────────────────────────────────
REQUIRED_SAMPLING_MIN = 5.0    # Expected sampling interval (minutes)
MIN_EPISODE_POINTS = 2         # Minimum points to constitute an episode

# ── Episode classification labels ─────────────────────────────────────────────
LABEL_SLH = "SLH"              # Short-Lasting Hyperglycemia
LABEL_LLH = "LLH"              # Long-Lasting Hyperglycemia

# ── Paper-definition 3-way category labels (see `slh_category` column) ────────
LABEL_SLH_NON_ISOLATED = "Non-isolated SLH"

# ── Onset time-of-day labels (see `onset_period` column) ──────────────────────
LABEL_DAYTIME = "Daytime"
LABEL_NIGHTTIME = "Nighttime"
