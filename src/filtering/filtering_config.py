# ── Filtering criterion ───────────────────────────────────────────────────────
MEDIAN_GLUCOSE_MIN = 110   # mg/dL — lower bound
MEDIAN_GLUCOSE_MAX = 170   # mg/dL — upper bound

# ── Column names ──────────────────────────────────────────────────────────────
COL_MEDIAN_GLUCOSE = "Median_glycemia"
COL_CV = "CV"
COL_TIR = "TIR70180"

# ── Display labels ────────────────────────────────────────────────────────────
# Static mg/dL label. Figures that follow the display unit build their own
# label with src.utils.unit_conversion.glucose_label("Median glucose").
LABEL_MEDIAN_GLUCOSE = "Median glucose (mg/dL)"
LABEL_CV = "Coefficient of variation"
LABEL_TIR = "TIR 70–180 (%)"