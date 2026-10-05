"""
CGM Patient Filtering — Median Glucose Criterion
================================================
Input       : Data/Data_Metrics/metrics_all_patients.csv
Output      : Data/Filtered_Cohorts/
              Figures/Filtered_Cohorts/

Filtering criterion
-------------------
  Median glucose ∈ [110, 170] mg/dL

Figure outputs
--------------
  median_glucose_histogram.pdf/.png    — Distribution with selection
  median_vs_cv_scatter.pdf/.png        — Scatter plot colored by TIR
  median_vs_tir_scatter.pdf/.png       — Scatter plot colored by CV
  cv_vs_tir_scatter.pdf/.png           — Scatter plot colored by Median
"""
# =============================================================================
# IMPORTS
# =============================================================================
import os
import pandas as pd

# =============================================================================
# IMPORTS PROJECT
# =============================================================================
from src.config import (
    FIG_FILTERING_DIR,
    PATIENT_ID_COL,
    FILTER_DIR_PATH,
    FIG_FILTERING_DIR,
    METRICS_CSV,
)

from src.filtering.filtering_config import (
    MEDIAN_GLUCOSE_MIN,
    MEDIAN_GLUCOSE_MAX,
    COL_MEDIAN_GLUCOSE,
    COL_CV,
    COL_TIR,
    LABEL_MEDIAN_GLUCOSE,
    LABEL_CV,
    LABEL_TIR,
)

from src.filtering.filtering import apply_median_glucose_filter
from src.metrics.load_metrics import load_patient_metrics


from src.visualization.filtering_plots import (
    generate_histogram_figure,
    generate_scatter_figure,
)

# =============================================================================
# PATHS & CONSTANTS
# =============================================================================
COL_HAS_WINDOW = "has_window"

# =============================================================================
# UTILS
# =============================================================================
def print_filtering_summary(n_selected, n_total):
    """
    Print filtering summary to console.
    """
    pct_selected = n_selected / n_total * 100
    pct_excluded = 100 - pct_selected

    print("\n" + "═" * 70)
    print("  FILTERING SUMMARY — Median Glucose Criterion")
    print("═" * 70)
    print(
        f"  Criterion: Median glucose ∈ "
        f"[{MEDIAN_GLUCOSE_MIN}, {MEDIAN_GLUCOSE_MAX}] mg/dL"
    )
    print("  " + "─" * 68)
    print(f"  Selected:  {n_selected:>4d} / {n_total}  ({pct_selected:.1f}%)")
    print(f"  Excluded:  {n_total - n_selected:>4d} / {n_total}  ({pct_excluded:.1f}%)")
    print("═" * 70)


def save_patient_lists(df_all, df_selected):
    """
    Save patient status CSV and selected patient IDs text file.
    """
    os.makedirs(FILTER_DIR_PATH, exist_ok=True)

    selected_ids = set(df_selected[PATIENT_ID_COL])

    status_df = df_all.copy()
    status_df["selected"] = status_df[PATIENT_ID_COL].isin(selected_ids)

    status_df = pd.concat([
        status_df[status_df["selected"]],
        status_df[~status_df["selected"]],
    ]).reset_index(drop=True)

    csv_path = os.path.join(
        FILTER_DIR_PATH,
        f"median_glucose_{MEDIAN_GLUCOSE_MIN}to{MEDIAN_GLUCOSE_MAX}_patient_status.csv",
    )
    status_df.to_csv(csv_path, index=False)
    print(f"{os.path.basename(csv_path)}")

    txt_path = os.path.join(
        FILTER_DIR_PATH,
        f"median_glucose_{MEDIAN_GLUCOSE_MIN}to{MEDIAN_GLUCOSE_MAX}_selected_ids.txt",
    )
    with open(txt_path, "w") as f:
        for patient_id in sorted(selected_ids):
            f.write(f"{patient_id}\n")

    print(f"{os.path.basename(txt_path)}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    """Main filtering and visualization pipeline."""

    print("=" * 70)
    print("  CGM Patient Filtering — Median Glucose Criterion")
    print("=" * 70)
    print(
        f"  Criterion: Median glucose ∈ "
        f"[{MEDIAN_GLUCOSE_MIN}, {MEDIAN_GLUCOSE_MAX}] mg/dL"
    )
    print("=" * 70)

    os.makedirs(FILTER_DIR_PATH, exist_ok=True)
    os.makedirs(FIG_FILTERING_DIR, exist_ok=True)

    df_all = load_patient_metrics(METRICS_CSV)

    df_selected = apply_median_glucose_filter(df_all)

    print_filtering_summary(len(df_selected), len(df_all))

    save_patient_lists(df_all, df_selected)


    generate_histogram_figure(df_all, df_selected, FIG_FILTERING_DIR)

    generate_scatter_figure(
        df_all, df_selected,
        COL_MEDIAN_GLUCOSE, COL_CV, COL_TIR,
        FIG_FILTERING_DIR,
    )

    generate_scatter_figure(
        df_all, df_selected,
        COL_MEDIAN_GLUCOSE, COL_TIR, COL_CV,
        FIG_FILTERING_DIR,
    )

    generate_scatter_figure(
        df_all, df_selected,
        COL_CV, COL_TIR, COL_MEDIAN_GLUCOSE,
        FIG_FILTERING_DIR,
    )

    print("\n" + "=" * 70)
    print("  FILTERING COMPLETE")
    print("=" * 70)
    print(f"  Total patients:     {len(df_all)}")
    print(f"  Selected patients:  {len(df_selected)}")
    print(f"  Excluded patients:  {len(df_all) - len(df_selected)}")
    print()
    print(f"  Output files:")
    print(f"    Data: {FILTER_DIR_PATH}")
    print(f"    Figures: {FIG_FILTERING_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()