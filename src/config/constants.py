# Column names
PATIENT_ID_COL = "patient_id"
TIME_COL = "time"
GLUCOSE_COL = "glycemie"


COL_COVERAGE = "coverage_pct"
COL_SAMPLING = "dominant_step_min"


COL_HAS_WINDOW = "has_window"
COL_COVERAGE_PCT = "coverage_pct"
COL_DOMINANT_STEP = "dominant_step_min"
COL_WIN_START = "win_start"
COL_WIN_END = "win_end"
COL_FILEPATH = "filepath"

# QUALITY FLAGS
COL_FLAG_TIME_ERROR = "flag_time_error"
COL_FLAG_OUTLIER_GLUCOSE = "flag_outlier_glucose"


# =============================================================================
# Episode column names
# =============================================================================
COL_EPISODE_START = "onset_time"
COL_EPISODE_END = "offset_time"
COL_EPISODE_DURATION = "duration_min"
COL_EPISODE_CLASSIFICATION = "classification"

# Episode classification labels
CLASSIFICATION_SLH = "SLH"
CLASSIFICATION_LLH = "LLH"



# Glucose unit conversion
# 1 mmol/L = 18.0182 mg/dL (molar mass of glucose: 180.156 g/mol)
from src.utils.unit_conversion import MGDL_PER_MMOLL

# Glucose thresholds (mg/dL)
THRESHOLD_HYPO = 70
THRESHOLD_HYPER = 180
GLUCOSE_MIN = 20
GLUCOSE_MAX = 500

# Preprocessing window
WINDOW_DAYS = 45
MIN_COVERAGE_PCT = 70.0
SLIDE_FREQ = "6h"
REQUIRED_SAMPLING_MIN = 5.0

# Quality control
TIME_ERROR_RATIO = 0.5
GAP_MULTIPLE = 2.0

# Night definition
NIGHT_START_H = 22
NIGHT_END_H = 6


H_HYPER = 180
H_HYPO = 70