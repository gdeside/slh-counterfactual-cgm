"""
scripts/generate_table1.py
===========================
Generate Table 1 (study cohort characteristics) using pysummaries.

Each variable is assigned its statistical summary:
  - Continuous  -> Median [Q1-Q3], with a Missing count
  - Categorical -> n (%), with a Missing level

The table is exported as:
  - table1.html   (styled, for reports / copy-paste to Word)
  - table1.csv    (raw wide format for further editing)

Usage
-----
    python3 -m scripts.paper_figures.generate_table1

Outputs  (RESULTS_PATH/Table1/)
"""

import numpy as np
import pandas as pd
from pysummaries import get_table_summary
from pysummaries.table_summary import (
    numerical_median_iqr,
    numerical_missing,
)

from src.config.paths import FINAL_COHORT_CSV, RESULTS_PATH
from src.utils.unit_conversion import (
    format_glucose,
    glucose_label,
    is_glucose_valued_column,
    to_display,
)

OUTPUT_DIR = RESULTS_PATH / "Table1"

# ---------------------------------------------------------------------------
# Column encoding constants (match nb_regression scripts)
# ---------------------------------------------------------------------------
CODE_NO  = 1
CODE_YES = 2
INSULIN_PUMP_CODE = 4


# ---------------------------------------------------------------------------
# Helper: comma-decimal European strings -> float
# ---------------------------------------------------------------------------
def parse_eu_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )


# ---------------------------------------------------------------------------
# Load & clean
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(FINAL_COHORT_CSV, sep=None, engine="python")

    # ── Continuous variables (European decimal comma) ──────────────────────
    for col in ["age_study_entry", "diabetes_duration", "zscore",
                "HbA1c", "IDAA1c", "GTAA1c",
                "TIR70180", "TIR70140", "TAR180", "TAR250", "TBR70",
                "Mean_glycemia", "SD", "CV", "Median_glycemia"]:
        df[col] = parse_eu_float(df[col])

    # Total daily insulin dose (TIDD) stored in 'TIDD ' column
    df["TIDD"] = parse_eu_float(df["TIDD "])

    # Duration of SLH not in clinical CSV — use Num_SLH only
    df["Num_SLH"] = pd.to_numeric(df["Num_SLH"], errors="coerce")

    # ── Categorical variables ───────────────────────────────────────────────
    # Sex: keep M/F strings as-is — already readable
    df["Sex"] = df["Sex"].str.strip().map({"M": "Male", "F": "Female"})

    # Binary: 1=No, 2=Yes
    for col in ["comorbidities", "Family_history", "Physical_activity",
                "DKA_at_diagnosis", "DKA_after_diagnosis"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").map(
            {CODE_NO: "No", CODE_YES: "Yes"}
        )

    # SH (Severe hypoglycemia): 1=No, 2=Yes  (1 missing)
    df["SH"] = pd.to_numeric(df["SH"], errors="coerce").map(
        {CODE_NO: "No", CODE_YES: "Yes"}
    )

    # Insulin regimen: 1,2,3 = MDI variants, 4 = Pump
    def recode_regimen(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return np.nan
        if v in (1.0, 2.0, 3.0):
            return "Basal-prandial"
        if v == INSULIN_PUMP_CODE:
            return "Insulin pump"
        return np.nan

    df["Insulin_regimen"] = df["Insulin_regimen "].apply(recode_regimen)

    # Recode regimen more granularly to match table (includes "Two injections")
    def recode_regimen_detail(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return np.nan
        if v == 1.0:
            return "Basal-prandial"
        if v == 2.0:
            return "Two injections"
        if v == 3.0:
            return "Basal-prandial"
        if v == INSULIN_PUMP_CODE:
            return "Insulin pump"
        return np.nan

    df["Insulin_regimen"] = df["Insulin_regimen "].apply(recode_regimen_detail)

    # Make categoricals pd.Categorical so pysummaries sees all levels
    cat_cols = ["Sex", "comorbidities", "Family_history", "Physical_activity",
                "DKA_at_diagnosis", "Insulin_regimen", "DKA_after_diagnosis", "SH"]
    for col in cat_cols:
        df[col] = pd.Categorical(df[col])

    # The cohort file stores glucose in mg/dL; a table is text, so the values
    # themselves are converted here (a no-op while the display unit is mg/dL).
    for col in df.columns:
        if is_glucose_valued_column(col) and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = to_display(df[col])

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print(f"Loaded {len(df)} patients, {df.shape[1]} columns")

    # ── Column lists matching Table 1 structure ───────────────────────────
    continuous_cols = [
        "age_study_entry",
        "diabetes_duration",
        "zscore",
        "TIDD",
        # Glycemic variability
        "HbA1c",
        "IDAA1c",
        "GTAA1c",
        "TIR70180",
        "TIR70140",
        "TAR180",
        "TAR250",
        "TBR70",
        "Mean_glycemia",
        "SD",
        "CV",
        "Median_glycemia",
        # SLH
        "Num_SLH",
    ]

    categorical_cols = [
        "Sex",
        "comorbidities",
        "Family_history",
        "Physical_activity",
        "DKA_at_diagnosis",
        "Insulin_regimen",
        "DKA_after_diagnosis",
        "SH",
    ]

    columns_include = [
        "age_study_entry",
        "Sex",
        "diabetes_duration",
        "zscore",
        "comorbidities",
        "Family_history",
        "Physical_activity",
        "DKA_at_diagnosis",
        "Insulin_regimen",
        "TIDD",
        "DKA_after_diagnosis",
        "SH",
        "HbA1c",
        "IDAA1c",
        "GTAA1c",
        "TIR70180",
        "TIR70140",
        "TAR180",
        "TAR250",
        "TBR70",
        "Mean_glycemia",
        "SD",
        "CV",
        "Median_glycemia",
        "Num_SLH",
    ]

    labels = {
        "age_study_entry":  "Age at study entry (years)",
        "Sex":              "Sex",
        "diabetes_duration": "Duration of diabetes (years)",
        "zscore":           "BMI (z-score)",
        "comorbidities":    "Comorbidities",
        "Family_history":   "Family history of diabetes",
        "Physical_activity": "Physical activity",
        "DKA_at_diagnosis": "Ketoacidosis at diagnosis",
        "Insulin_regimen":  "Insulin regimen",
        "TIDD":             "Total daily insulin dose (units/kg)",
        "DKA_after_diagnosis": "Ketoacidosis after diagnosis",
        "SH":               "Severe hypoglycemia",
        "HbA1c":            "HbA1c (%)",
        "IDAA1c":           "IDAA1C",
        "GTAA1c":           "GTAA1C",
        "TIR70180":         f"TIR {format_glucose(70, with_unit=False)}–{format_glucose(180)}",
        "TIR70140":         f"TIR {format_glucose(70, with_unit=False)}–{format_glucose(140)}",
        "TAR180":           f"TAR > {format_glucose(180)}",
        "TAR250":           f"TAR > {format_glucose(250)}",
        "TBR70":            f"TBR < {format_glucose(70)}",
        "Mean_glycemia":    glucose_label("Mean glucose"),
        "SD":               glucose_label("Standard deviation"),
        "CV":               "Coefficient of variation",
        "Median_glycemia":  glucose_label("Median glucose"),
        "Num_SLH":          "Number of SLH episodes",
    }

    df_sub = df[columns_include].copy()

    # Print missing counts for visibility
    print("\nMissing values per variable:")
    for col in columns_include:
        n_missing = int(df_sub[col].isna().sum())
        print(f"  {col:30s}  missing={n_missing:3d} / {len(df_sub)}")

    # ── Which patients are missing which variables ─────────────────────────
    id_col = "Patient ID"
    print("\nPatients with missing values (by variable):")
    missing_long_rows = []
    for col in columns_include:
        missing_mask = df_sub[col].isna()
        missing_ids = df.loc[missing_mask, id_col].tolist()
        if missing_ids:
            print(f"  {labels[col]:40s} ({len(missing_ids)}): {missing_ids}")
        for pid in missing_ids:
            missing_long_rows.append({id_col: pid, "Variable": labels[col]})

    missing_long_df = pd.DataFrame(missing_long_rows, columns=[id_col, "Variable"])
    missing_long_path = OUTPUT_DIR / "table1_missing_by_patient.csv"
    missing_long_df.to_csv(missing_long_path, index=False)
    print(f"\nSaved: {missing_long_path}")

    # Wide matrix: one row per patient with >=1 missing value, one column per variable
    missing_wide = df_sub[columns_include].isna().rename(columns=labels)
    missing_wide.insert(0, id_col, df[id_col].values)
    missing_wide = missing_wide[missing_wide.drop(columns=id_col).any(axis=1)]
    missing_wide_path = OUTPUT_DIR / "table1_missing_matrix.csv"
    missing_wide.to_csv(missing_wide_path, index=False)
    print(f"Saved: {missing_wide_path}  ({len(missing_wide)} patients with >=1 missing value)")

    # ── Export HTML ───────────────────────────────────────────────────────
    # pysummaries applies one numerical_functions dict to ALL numeric cols,
    # so we run two separate calls (continuous / categorical) and
    # concatenate the HTML blocks into one page.

    labels_with_stat = {}
    for col in continuous_cols:
        labels_with_stat[col] = f"{labels[col]}  [Median (Q1–Q3)]"
    for col in categorical_cols:
        labels_with_stat[col] = labels[col]

    # Run pysummaries for continuous cols (Median [Q1-Q3] + Missing)
    tbl_continuous = get_table_summary(
        df_sub[continuous_cols + categorical_cols],
        columns_include=continuous_cols,
        columns_labels={c: labels_with_stat[c] for c in continuous_cols},
        numerical_functions={"Median [Q1–Q3]": numerical_median_iqr,
                             "Missing":        numerical_missing},
        categorical_missing_level="Missing",
        rounding=2,
    )

    # Run pysummaries for categoricals only (for the master table)
    tbl_cat = get_table_summary(
        df_sub[categorical_cols],
        columns_include=categorical_cols,
        columns_labels={c: labels[c] for c in categorical_cols},
        categorical_missing_level="Missing",
        rounding=1,
    )

    # Save separate HTML files
    html_continuous = tbl_continuous.get_raw_html()
    html_cat        = tbl_cat.get_raw_html()

    # Build combined HTML
    combined_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Table 1 – Study cohort characteristics (N={len(df)})</title>
<style>
  body {{ font-family: Arial, sans-serif; font-size: 13px; margin: 30px; }}
  h1   {{ font-size: 15px; font-weight: bold; }}
  h2   {{ font-size: 13px; font-weight: bold; margin-top: 20px; border-top: 1px solid #ccc; padding-top: 8px; }}
  p    {{ font-size: 11px; color: #555; }}
</style>
</head>
<body>
<h1>Table 1: Characteristics of the study cohort (N = {len(df)})</h1>

<h2>Categorical variables — n (%)</h2>
{html_cat}

<h2>Continuous variables — Median [Q1–Q3]</h2>
{html_continuous}

<p>Continuous variables: Median [Q1–Q3], with number of missing values.
Categorical variables: n (%), with a Missing level when applicable.</p>
</body>
</html>"""

    html_path = OUTPUT_DIR / "table1.html"
    html_path.write_text(combined_html, encoding="utf-8")
    print(f"\nSaved: {html_path}")

    # ── Export CSV (raw stats for each variable) ──────────────────────────
    csv_rows = []
    for col in columns_include:
        lbl = labels[col]
        vals = df[col].dropna()
        n_total   = len(df)
        n_missing = int(df[col].isna().sum())

        if col in categorical_cols:
            counts = df[col].value_counts(dropna=False)
            for level, cnt in counts.items():
                level_lbl = "Missing" if pd.isna(level) else str(level)
                csv_rows.append({
                    "Variable":  lbl,
                    "Level":     level_lbl,
                    "Statistic": "n (%)",
                    "Value":     f"{cnt} ({100*cnt/n_total:.1f})",
                    "N_missing": n_missing,
                })
        else:
            q1, med, q3 = vals.quantile([0.25, 0.5, 0.75])
            stat_str = f"{med:.2f} [{q1:.2f}–{q3:.2f}]"
            csv_rows.append({
                "Variable":  lbl,
                "Level":     "",
                "Statistic": "Median [Q1–Q3]",
                "Value":     stat_str,
                "N_missing": n_missing,
            })
            csv_rows.append({
                "Variable":  lbl,
                "Level":     "Missing",
                "Statistic": "n",
                "Value":     str(n_missing),
                "N_missing": n_missing,
            })

    csv_df = pd.DataFrame(csv_rows)
    csv_path = OUTPUT_DIR / "table1.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    print(f"\nTable 1 complete — {len(csv_df)} rows exported.")


if __name__ == "__main__":
    main()
