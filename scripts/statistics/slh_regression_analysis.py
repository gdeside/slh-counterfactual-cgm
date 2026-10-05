"""
scripts/slh_regression_analysis.py
====================================
Random Forest regression analysis: CGM metrics + patient phenotype
as predictors of SLH episode count (n_SLH).

Pipeline
--------
  1. Load CGM metrics (pipeline) and phenotype (clinical CSV)
  2. Merge on patient_id, apply cohort filter
  3. Drop features with > MAX_MISSING_PCT missing values
  4. Impute remaining missing: median (continuous), mode (categorical)
  5. Ordinal-encode categorical features
  6. Fit Random Forest regressor (OOB score as internal validation)
  7. MDI feature importance (mean decrease in impurity, across all trees)
  8. Permutation importance (50 repeats on training set)
  9. Permutation test  -- shuffle y N_PERMUTATIONS times, compare CV R2 to null
 10. Partial Dependence Plots for top-k features
 11. Observed vs OOB-predicted scatter

Outputs  (RESULTS_PATH / RF_SLH_Predictors)
-------------------------------------------
  fig_missing_data.{pdf,png,svg}
  fig_importance_mdi.{pdf,png,svg}
  fig_importance_permutation.{pdf,png,svg}
  fig_permutation_test.{pdf,png,svg}
  fig_pdp_top<k>.{pdf,png,svg}
  fig_obs_vs_pred.{pdf,png,svg}
  feature_importance_mdi.csv
  feature_importance_permutation.csv
  missing_data_report.csv
  summary.txt

Usage
-----
    env/bin/python3 -m scripts.statistics.slh_regression_analysis [--no_filter] \\
        [--max_missing 0.20] [--n_permutations 1000] [--top_k 6]

Authors
-------
Guillaume Deside
UCL - counterfactual-cgm-slh project

Date
----
2025
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import OrdinalEncoder

from src.config import METRICS_CSV, FILTER_STATUS_CSV
from src.config.paths import FINAL_COHORT_CSV, RESULTS_PATH
from src.metrics.metrics import (
    METRIC_MEDIAN_GLUCOSE, METRIC_MEAN_GLUCOSE, METRIC_SD, METRIC_CV,
    METRIC_TIR70180, METRIC_TAR180, METRIC_TBR70,
)
from src.episodes.load_episodes import load_episode_summary
from src.metrics.load_metrics import load_patient_metrics
from src.filtering.load_filtering import load_filtered_patient_list
from src.visualization.colors import WONG_COLORS
from src.visualization.style import apply_publication_style, save_figure, FONT_SIZE_BASE

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = RESULTS_PATH / "RF_SLH_Predictors"

MAX_MISSING_PCT_DEFAULT = 0.20   # drop features with more missing than this
N_ESTIMATORS            = 200
MAX_FEATURES            = "sqrt"
MIN_SAMPLES_LEAF        = 3
RANDOM_STATE            = 42
N_PERMUTATIONS_DEFAULT  = 1000
N_CV_FOLDS              = 5
TOP_K_DEFAULT           = 6

RESPONSE       = "n_SLH"
RESPONSE_LABEL = "Number of SLH episodes"

CGM_METRICS = [
    METRIC_MEDIAN_GLUCOSE, METRIC_MEAN_GLUCOSE, METRIC_SD, METRIC_CV,
    METRIC_TIR70180, METRIC_TAR180, METRIC_TBR70,
]
PROPORTION_METRICS = [METRIC_CV, METRIC_TIR70180, METRIC_TAR180, METRIC_TBR70]

CGM_LABELS = {
    METRIC_MEDIAN_GLUCOSE: "Median glucose (mg/dL)",
    METRIC_MEAN_GLUCOSE:   "Mean glucose (mg/dL)",
    METRIC_SD:             "Glucose SD (mg/dL)",
    METRIC_CV:             "Glucose CV (%)",
    METRIC_TIR70180:       "TIR 70-180 (%)",
    METRIC_TAR180:         "TAR >180 (%)",
    METRIC_TBR70:          "TBR <70 (%)",
}

PHENO_CONTINUOUS = [
    "age_study_entry", "diabetes_duration", "zscore", "TIDD",
    "HbA1c", "IDAA1c", "GTAA1c",
]
PHENO_LABELS_CONT = {
    "age_study_entry":   "Age at study entry (years)",
    "diabetes_duration": "Diabetes duration (years)",
    "zscore":            "BMI z-score",
    "TIDD":              "Total daily insulin dose (U/kg)",
    "HbA1c":             "HbA1c (%)",
    "IDAA1c":            "IDAA1C",
    "GTAA1c":            "GTAA1C",
}

PHENO_CATEGORICAL = [
    "Sex", "comorbidities", "Physical_activity", "DKA_at_diagnosis",
    "Family_history", "Insulin_regimen", "DKA_after_diagnosis", "SH",
]
PHENO_LABELS_CAT = {
    "Sex":               "Sex",
    "comorbidities":     "Comorbidities",
    "Physical_activity": "Physical activity",
    "DKA_at_diagnosis":  "DKA at diagnosis",
    "Family_history":    "Family history of diabetes",
    "Insulin_regimen":   "Insulin regimen",
    "DKA_after_diagnosis": "DKA after diagnosis",
    "SH":                "Severe hypoglycemia",
}

CODE_NO = 1; CODE_YES = 2; INSULIN_PUMP_CODE = 4

ALL_LABELS = {**CGM_LABELS, **PHENO_LABELS_CONT, **PHENO_LABELS_CAT}


# =============================================================================
# DATA LOADING
# =============================================================================
def parse_eu_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )


def load_cgm(apply_filter: bool) -> pd.DataFrame:
    metrics_df  = load_patient_metrics(METRICS_CSV)
    episodes_df = load_episode_summary()
    merged = metrics_df.merge(episodes_df[["patient_id", RESPONSE]],
                               on="patient_id", how="inner")
    if apply_filter:
        pids   = load_filtered_patient_list(FILTER_STATUS_CSV)
        merged = merged[merged["patient_id"].isin(pids)].reset_index(drop=True)
    for col in PROPORTION_METRICS:
        if col in merged.columns:
            merged[col] = merged[col] * 100.0
    return merged[["patient_id", RESPONSE] + CGM_METRICS]


def load_phenotype() -> pd.DataFrame:
    df = pd.read_csv(FINAL_COHORT_CSV, sep=None, engine="python")

    for col in ["age_study_entry", "diabetes_duration", "zscore",
                "HbA1c", "IDAA1c", "GTAA1c"]:
        df[col] = parse_eu_float(df[col])
    df["TIDD"] = parse_eu_float(df["TIDD "])

    df["Sex"] = df["Sex"].str.strip().map({"M": "Male", "F": "Female"})
    for col in ["comorbidities", "Family_history", "Physical_activity",
                "DKA_at_diagnosis", "DKA_after_diagnosis"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").map(
            {CODE_NO: "No", CODE_YES: "Yes"})
    df["SH"] = pd.to_numeric(df["SH"], errors="coerce").map(
        {CODE_NO: "No", CODE_YES: "Yes"})

    def recode_regimen(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return np.nan
        mapping = {1.0: "Basal-prandial", 2.0: "Two injections",
                   3.0: "Basal-prandial", float(INSULIN_PUMP_CODE): "Insulin pump"}
        return mapping.get(v, np.nan)

    df["Insulin_regimen"] = df["Insulin_regimen "].apply(recode_regimen)

    pid_col = "Patient ID" if "Patient ID" in df.columns else df.columns[0]
    df = df.rename(columns={pid_col: "patient_id"})
    keep = ["patient_id"] + PHENO_CONTINUOUS + PHENO_CATEGORICAL
    return df[[c for c in keep if c in df.columns]]


def build_dataset(apply_filter: bool,
                  max_missing_pct: float) -> tuple:
    cgm   = load_cgm(apply_filter)
    pheno = load_phenotype()
    merged = cgm.merge(pheno, on="patient_id", how="inner")
    print(f"  Merged: {len(merged)} patients")

    feature_cols_raw = [c for c in CGM_METRICS + PHENO_CONTINUOUS + PHENO_CATEGORICAL
                        if c in merged.columns]

    missing_frac = merged[feature_cols_raw].isnull().mean()
    missing_rpt  = missing_frac.to_dict()

    drop_cols = [c for c, f in missing_frac.items() if f > max_missing_pct]
    if drop_cols:
        drop_labels = [ALL_LABELS.get(c, c) for c in drop_cols]
        print(f"  Dropped (>{max_missing_pct*100:.0f}% missing): {drop_labels}")
        merged = merged.drop(columns=drop_cols)
        feature_cols_raw = [c for c in feature_cols_raw if c not in drop_cols]

    kept_cont = [c for c in CGM_METRICS + PHENO_CONTINUOUS if c in feature_cols_raw]
    kept_cat  = [c for c in PHENO_CATEGORICAL if c in feature_cols_raw]

    # Impute
    for col in kept_cont:
        n_imp = merged[col].isnull().sum()
        if n_imp:
            med = merged[col].median()
            merged[col] = merged[col].fillna(med)
            print(f"  Imputed {ALL_LABELS.get(col, col)}: {n_imp} with median={med:.3f}")

    for col in kept_cat:
        n_imp = merged[col].isnull().sum()
        if n_imp:
            mode_val = merged[col].mode()
            if len(mode_val):
                merged[col] = merged[col].fillna(mode_val[0])
                print(f"  Imputed {ALL_LABELS.get(col, col)}: {n_imp} with mode='{mode_val[0]}'")

    # Ordinal encode categoricals
    if kept_cat:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        merged[kept_cat] = enc.fit_transform(merged[kept_cat].astype(str))

    feat_cols = kept_cont + kept_cat
    return merged, feat_cols, missing_rpt


# =============================================================================
# PLOTS
# =============================================================================
def plot_missing_report(missing_rpt: dict, max_missing_pct: float) -> None:
    rows = sorted([(ALL_LABELS.get(c, c), f)
                   for c, f in missing_rpt.items()],
                  key=lambda x: x[1], reverse=True)
    if not rows:
        return
    labels_m, fracs = zip(*rows)

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6.5, max(3.5, len(labels_m) * 0.38)))
    colors = [WONG_COLORS["vermillion"] if f > max_missing_pct
              else WONG_COLORS["blue"] for f in fracs]
    ax.barh(labels_m, [f * 100 for f in fracs],
            color=colors, edgecolor="white", alpha=0.85)
    ax.axvline(max_missing_pct * 100, color="black", lw=1.5, ls="--",
               label=f"Threshold ({max_missing_pct*100:.0f}%)")
    ax.set_xlabel("Missing values (%)", fontsize=FONT_SIZE_BASE)
    ax.set_title("Missing data by feature\n(red = dropped before modelling)",
                 fontsize=FONT_SIZE_BASE)
    ax.legend(fontsize=FONT_SIZE_BASE - 2, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig_missing_data")
    print("  Saved: fig_missing_data")


def plot_mdi_importance(rf: RandomForestRegressor,
                        feat_cols: list) -> None:
    imp = rf.feature_importances_
    std = np.std([t.feature_importances_ for t in rf.estimators_], axis=0)
    imp_df = pd.DataFrame({
        "feature": feat_cols,
        "label":   [ALL_LABELS.get(c, c) for c in feat_cols],
        "importance": imp, "std": std,
    }).sort_values("importance", ascending=True)
    imp_df.to_csv(OUTPUT_DIR / "feature_importance_mdi.csv", index=False)

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6.5, max(4, len(feat_cols) * 0.42)))
    ax.barh(imp_df["label"], imp_df["importance"],
            xerr=imp_df["std"], color=WONG_COLORS["blue"],
            edgecolor="white", alpha=0.85, capsize=3)
    ax.set_xlabel("MDI importance (mean +/- SD across trees)",
                  fontsize=FONT_SIZE_BASE)
    ax.set_title("Random Forest — Mean Decrease in Impurity",
                 fontsize=FONT_SIZE_BASE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig_importance_mdi")
    print("  Saved: fig_importance_mdi")


def compute_and_plot_permutation_importance(rf, X, y, feat_cols) -> pd.DataFrame:
    perm = permutation_importance(rf, X, y, n_repeats=50,
                                  scoring="r2", random_state=RANDOM_STATE,
                                  n_jobs=-1)
    perm_df = pd.DataFrame({
        "feature":        feat_cols,
        "label":          [ALL_LABELS.get(c, c) for c in feat_cols],
        "importance_mean": perm.importances_mean,
        "importance_std":  perm.importances_std,
    }).sort_values("importance_mean", ascending=True)
    perm_df.to_csv(OUTPUT_DIR / "feature_importance_permutation.csv", index=False)

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6.5, max(4, len(feat_cols) * 0.42)))
    colors = [WONG_COLORS["vermillion"] if v < 0 else WONG_COLORS["blue"]
              for v in perm_df["importance_mean"]]
    ax.barh(perm_df["label"], perm_df["importance_mean"],
            xerr=perm_df["importance_std"],
            color=colors, edgecolor="white", alpha=0.85, capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Mean decrease in R2 (50 repeats)", fontsize=FONT_SIZE_BASE)
    ax.set_title("Random Forest — permutation importance\n"
                 "(negative = feature adds noise)", fontsize=FONT_SIZE_BASE)
    handles = [
        mpatches.Patch(color=WONG_COLORS["blue"],       label="Positive contribution"),
        mpatches.Patch(color=WONG_COLORS["vermillion"],  label="Negative / noise"),
    ]
    ax.legend(handles=handles, fontsize=FONT_SIZE_BASE - 2, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig_importance_permutation")
    print("  Saved: fig_importance_permutation")
    return perm_df


def run_and_plot_permutation_test(X, y, true_r2, n_permutations) -> dict:
    rng  = np.random.default_rng(RANDOM_STATE)
    kf   = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    null_r2s = []

    for i in range(n_permutations):
        y_perm  = rng.permutation(y)
        rf_null = RandomForestRegressor(
            n_estimators=100, max_features=MAX_FEATURES,
            min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=-1, random_state=i,
        )
        scores = cross_val_score(rf_null, X, y_perm, cv=kf,
                                 scoring="r2", n_jobs=-1)
        null_r2s.append(scores.mean())
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{n_permutations} permutations done ...")

    null_r2s = np.array(null_r2s)
    p_value  = float((null_r2s >= true_r2).mean())
    result   = {"true_r2": true_r2, "null_r2s": null_r2s,
                "null_mean": float(null_r2s.mean()),
                "null_std":  float(null_r2s.std()),
                "p_value":   p_value, "n_perm": n_permutations}

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(null_r2s, bins=40, color=WONG_COLORS["bluish_green"],
            edgecolor="white", alpha=0.8, label="Null (shuffled y)")
    ax.axvline(true_r2, color=WONG_COLORS["vermillion"], lw=2.5,
               label=f"Observed R2 = {true_r2:.3f}")
    p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
    ax.set_xlabel(f"Cross-validated R2 ({N_CV_FOLDS}-fold)", fontsize=FONT_SIZE_BASE)
    ax.set_ylabel("Count", fontsize=FONT_SIZE_BASE)
    ax.set_title(f"Permutation test ({n_permutations} permutations)\n{p_text}",
                 fontsize=FONT_SIZE_BASE)
    ax.legend(fontsize=FONT_SIZE_BASE - 1, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig_permutation_test")
    print("  Saved: fig_permutation_test")
    return result


def plot_pdp(rf, X, feat_cols, perm_df, top_k) -> None:
    top_feats = (perm_df.sort_values("importance_mean", ascending=False)
                 .head(top_k)["feature"].tolist())
    top_idx = [feat_cols.index(f) for f in top_feats if f in feat_cols]
    if not top_idx:
        return

    n_cols = min(3, len(top_idx))
    n_rows = int(np.ceil(len(top_idx) / n_cols))
    feat_names = [ALL_LABELS.get(c, c) for c in feat_cols]

    apply_publication_style()
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 3.2))
    axes = np.array(axes).flatten()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        PartialDependenceDisplay.from_estimator(
            rf, X, features=top_idx, feature_names=feat_names,
            ax=axes[:len(top_idx)],
            line_kw={"color": WONG_COLORS["blue"], "lw": 2},
            random_state=RANDOM_STATE,
        )

    for ax in axes[:len(top_idx)]:
        ax.set_ylabel(RESPONSE_LABEL, fontsize=FONT_SIZE_BASE - 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[len(top_idx):]:
        ax.set_visible(False)

    fig.suptitle(f"Partial Dependence Plots — top {len(top_idx)} features\n"
                 "(marginal effect on predicted n_SLH)",
                 fontsize=FONT_SIZE_BASE)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / f"fig_pdp_top{len(top_idx)}")
    print(f"  Saved: fig_pdp_top{len(top_idx)}")


def plot_obs_vs_pred(rf, X, y) -> None:
    oob_pred = rf.oob_prediction_
    oob_r2   = rf.oob_score_
    rho, p   = scipy_stats.spearmanr(y, oob_pred)

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(y, oob_pred, color=WONG_COLORS["blue"],
               s=30, alpha=0.65, edgecolors="none")
    lims = [min(y.min(), oob_pred.min()) - 1,
            max(y.max(), oob_pred.max()) + 1]
    ax.plot(lims, lims, color="grey", lw=1, ls="--", alpha=0.7)
    ax.set_xlabel(f"Observed {RESPONSE_LABEL}", fontsize=FONT_SIZE_BASE)
    ax.set_ylabel(f"OOB predicted {RESPONSE_LABEL}", fontsize=FONT_SIZE_BASE)
    p_text = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    ax.set_title(
        f"Observed vs OOB-predicted\n"
        f"OOB R2={oob_r2:.3f}  Spearman rho={rho:.2f} ({p_text})",
        fontsize=FONT_SIZE_BASE,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig_obs_vs_pred")
    print("  Saved: fig_obs_vs_pred")


# =============================================================================
# MAIN
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no_filter", action="store_true")
    p.add_argument("--max_missing", type=float, default=MAX_MISSING_PCT_DEFAULT)
    p.add_argument("--n_permutations", type=int, default=N_PERMUTATIONS_DEFAULT)
    p.add_argument("--top_k", type=int, default=TOP_K_DEFAULT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}\n")

    # ── Data ────────────────────────────────────────────────────────────────
    print("[1/7] Loading and preparing data ...")
    df, feat_cols, missing_rpt = build_dataset(
        apply_filter=not args.no_filter,
        max_missing_pct=args.max_missing,
    )
    pd.DataFrame([{"feature": k, "label": ALL_LABELS.get(k, k),
                   "missing_pct": round(v*100, 1),
                   "dropped": v > args.max_missing}
                  for k, v in missing_rpt.items()]).to_csv(
        OUTPUT_DIR / "missing_data_report.csv", index=False)
    plot_missing_report(missing_rpt, args.max_missing)

    X = df[feat_cols].values.astype(float)
    y = df[RESPONSE].values.astype(float)
    print(f"  Features kept: {len(feat_cols)}  |  Patients: {len(y)}")

    # ── Random Forest ───────────────────────────────────────────────────────
    print("\n[2/7] Fitting Random Forest ...")
    rf = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, max_features=MAX_FEATURES,
        min_samples_leaf=MIN_SAMPLES_LEAF, oob_score=True,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    rf.fit(X, y)
    oob_r2 = rf.oob_score_

    kf    = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_r2 = cross_val_score(rf, X, y, cv=kf, scoring="r2", n_jobs=-1)
    print(f"  OOB R2 = {oob_r2:.3f}")
    print(f"  CV R2  = {cv_r2.mean():.3f} +/- {cv_r2.std():.3f}")

    # ── MDI importance ──────────────────────────────────────────────────────
    print("\n[3/7] MDI feature importance ...")
    plot_mdi_importance(rf, feat_cols)

    # ── Permutation importance ──────────────────────────────────────────────
    print("\n[4/7] Permutation importance (50 repeats) ...")
    perm_df = compute_and_plot_permutation_importance(rf, X, y, feat_cols)

    # ── Permutation test ────────────────────────────────────────────────────
    print(f"\n[5/7] Permutation test ({args.n_permutations} permutations) ...")
    perm_result = run_and_plot_permutation_test(
        X, y, true_r2=cv_r2.mean(), n_permutations=args.n_permutations)
    p_val  = perm_result["p_value"]
    p_text = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
    print(f"  True CV R2 = {perm_result['true_r2']:.3f}")
    print(f"  Null R2    = {perm_result['null_mean']:.3f} +/- {perm_result['null_std']:.3f}")
    print(f"  {p_text}")

    # ── PDP ─────────────────────────────────────────────────────────────────
    print(f"\n[6/7] Partial Dependence Plots (top {args.top_k}) ...")
    plot_pdp(rf, X, feat_cols, perm_df, top_k=args.top_k)

    # ── Obs vs predicted ────────────────────────────────────────────────────
    print("\n[7/7] Observed vs OOB-predicted ...")
    plot_obs_vs_pred(rf, X, y)

    # ── Summary ─────────────────────────────────────────────────────────────
    top5 = perm_df.sort_values("importance_mean", ascending=False).head(5)
    summary = (
        f"Random Forest Regression — SLH count predictors\n"
        f"{'='*52}\n"
        f"N patients    : {len(y)}\n"
        f"N features    : {len(feat_cols)}\n"
        f"  CGM         : {[c for c in feat_cols if c in CGM_METRICS]}\n"
        f"  Phenotype   : {[c for c in feat_cols if c not in CGM_METRICS]}\n"
        f"\nRF (n={N_ESTIMATORS} trees, max_features={MAX_FEATURES}, "
        f"min_leaf={MIN_SAMPLES_LEAF})\n"
        f"  OOB R2      : {oob_r2:.3f}\n"
        f"  CV R2       : {cv_r2.mean():.3f} +/- {cv_r2.std():.3f}\n"
        f"\nPermutation test ({args.n_permutations} permutations)\n"
        f"  True CV R2  : {perm_result['true_r2']:.3f}\n"
        f"  Null R2     : {perm_result['null_mean']:.3f} +/- {perm_result['null_std']:.3f}\n"
        f"  {p_text}\n"
        f"\nTop 5 by permutation importance:\n"
    )
    for _, row in top5.iterrows():
        summary += (f"  {row['label']:<40s}  "
                    f"{row['importance_mean']:.4f} +/- {row['importance_std']:.4f}\n")

    print("\n" + summary)
    (OUTPUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")

    print(f"\nDone. All outputs in:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
