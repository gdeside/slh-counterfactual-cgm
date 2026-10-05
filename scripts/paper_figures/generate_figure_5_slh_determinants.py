"""
scripts/paper_figures/generate_figure_5_slh_determinants.py
=============================================================
Manuscript figure: association of CGM metrics, patient phenotype and
glycaemic biomarkers with SLH burden (number of SLH episodes).

Panels
------
  A - Spearman correlation matrix: CGM metrics, phenotype and biomarkers
      (+ n_SLH), domain blocks delimited by separator lines
  B - Mutual information of every variable with n_SLH, coloured by domain
  C - Diabetes duration vs GTAA1C scatter, coloured by n_SLH
  D - Combined CGM + phenotype PCA - patient scores (coloured by n_SLH)
  E - Combined CGM + phenotype PCA - correlation circle
  F - n_SLH by insulin regimen (boxplot, Kruskal-Wallis)
  G - Diabetes duration vs GTAA1C scatter, coloured by n_SLH, marker shape
      = insulin regimen (same relationship as panel C, stratified by
      treatment modality)

Every panel is drawn by a plot_<name>(ax, ...) function that is reused
twice: once inside the combined GridSpec figure below, and once on its own
Axes to save an individual per-panel file (see save_individual_panel and the
Figure_Determinants/ subfolder in Outputs).

Variable sets and merge logic reuse the definitions established in:
  scripts/slh_baseline_metrics_multivariate_exploration.py   (CGM metrics)
  scripts/slh_patient_characteristics_exploration.py         (phenotype/biomarkers/
      insulin regimen recoding)
  scripts/slh_combined_cgm_phenotype_pca.py                  (combined PCA,
      GTAA1C/HbA1c/IDAA1c excluded there because GTAA1C is mechanically
      built from TIR - see that script's docstring)

Usage
-----
    env/bin/python3 -m scripts.paper_figures.generate_figure_5_slh_determinants [--no_filter]

Outputs (FIG_DIR / Manuscript_Figures)
---------------------------------------
  Figure5_SLH_Determinants.{pdf,svg}            - combined 7-panel figure
  Figure_Determinants/panel<X>_<name>.{pdf,svg} - each panel A-G saved standalone
  correlation_matrix_slh_determinants.csv       - full Spearman rho matrix
  correlation_pvalues_slh_determinants.csv      - matching p-value matrix
  spearman_vs_slh_determinants.csv              - each variable vs n_SLH only
                                                   (rho, p, domain), ranked by |rho|
  mutual_information_slh_determinants.csv       - each variable's MI with n_SLH
  scatter_duration_gtaa1c_stats_slh_determinants.csv - panels C/G: rho, p, fit
  pca_variance_explained_slh_determinants.csv   - PC1/PC2 % variance explained
  pca_loadings_slh_determinants.csv             - PC1/PC2 correlation loadings
  pca_scores_slh_determinants.csv               - per-patient PC1/PC2 scores
  pca_pc_vs_slh_determinants.csv                - PC1/PC2 vs n_SLH (rho, p)
  insulin_regimen_kruskal_slh_determinants.csv  - panel F: Kruskal-Wallis stat
  summary_statistics_slh_determinants.txt       - human-readable results digest

Authors: Guillaume Deside
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

from src.config import METRICS_CSV, FILTER_STATUS_CSV, FIG_DIR
from src.config.paths import FINAL_COHORT_CSV
from src.metrics.metrics import (
    METRIC_MEDIAN_GLUCOSE,
    METRIC_MEAN_GLUCOSE,
    METRIC_SD,
    METRIC_CV,
    METRIC_TIR70180,
    METRIC_TAR180,
    METRIC_TBR70,
)
from src.episodes.load_episodes import load_episode_summary
from src.metrics.load_metrics import load_patient_metrics
from src.filtering.load_filtering import load_filtered_patient_list
from src.visualization.style import apply_publication_style, save_figure, style_axes, FONT_SIZE_BASE
from src.visualization.colors import WONG_COLORS, DIVERGING, COLORMAP_CONTINUOUS

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR      = os.path.join(FIG_DIR, "Manuscript_Figures")
OUTPUT_NAME     = "Figure5_SLH_Determinants"
INDIVIDUAL_DIR  = os.path.join(OUTPUT_DIR, "Figure_Determinants")

RESPONSE       = "n_SLH"
RESPONSE_LABEL = "Number of SLH episodes"

# ── Insulin regimen (categorical) ────────────────────────────────────────────
# Same recoding as scripts/slh_patient_characteristics_exploration.py.
INSULIN_REGIMEN_COL = "Insulin_regimen"
INSULIN_PUMP_CODE   = 4

INSULIN_REGIMEN_MARKERS = {
    "Basal-prandial": "o",
    "Two injections": "^",
    "Insulin pump":   "s",
}
INSULIN_REGIMEN_ORDER  = list(INSULIN_REGIMEN_MARKERS)
INSULIN_REGIMEN_COLORS = [WONG_COLORS["blue"], WONG_COLORS["orange"], WONG_COLORS["bluish_green"]]

# ── Domain variable sets ─────────────────────────────────────────────────────
CGM_COLS = [
    METRIC_MEDIAN_GLUCOSE, METRIC_MEAN_GLUCOSE, METRIC_SD, METRIC_CV,
    METRIC_TIR70180, METRIC_TAR180, METRIC_TBR70,
]
PROPORTION_METRICS = [METRIC_CV, METRIC_TIR70180, METRIC_TAR180, METRIC_TBR70]

PHENOTYPE_COLS  = ["age_study_entry", "diabetes_duration", "zscore", "TIDD"]
BIOMARKER_COLS  = ["HbA1c", "IDAA1c", "GTAA1c"]

ALL_COLS = CGM_COLS + PHENOTYPE_COLS + BIOMARKER_COLS

LABELS_SHORT = {
    METRIC_MEDIAN_GLUCOSE: "Median glucose",
    METRIC_MEAN_GLUCOSE:   "Mean glucose",
    METRIC_SD:             "Glucose SD",
    METRIC_CV:             "Glucose CV",
    METRIC_TIR70180:       "TIR 70-180",
    METRIC_TAR180:         "TAR >180",
    METRIC_TBR70:          "TBR <70",
    "age_study_entry":     "Age at entry",
    "diabetes_duration":   "Diabetes duration",
    "zscore":              "BMI z-score",
    "TIDD":                "TIDD",
    "HbA1c":                "HbA1c",
    "IDAA1c":               "IDAA1C",
    "GTAA1c":                "GTAA1C",
}

LABELS_FULL = {
    "diabetes_duration": "Diabetes duration (years)",
    "GTAA1c":            "GTAA1C",
}

DOMAIN_OF = {c: "CGM" for c in CGM_COLS}
DOMAIN_OF.update({c: "Phenotype" for c in PHENOTYPE_COLS})
DOMAIN_OF.update({c: "Biomarker" for c in BIOMARKER_COLS})

DOMAIN_COLORS = {
    "CGM":       WONG_COLORS["blue"],
    "Phenotype": WONG_COLORS["bluish_green"],
    "Biomarker": WONG_COLORS["reddish_purple"],
}
DOMAIN_TITLES = {
    "CGM":       "CGM metrics",
    "Phenotype": "Phenotype",
    "Biomarker": "Biomarkers",
}

# ── Combined PCA variable set (identical to slh_combined_cgm_phenotype_pca.py)
PCA_CGM_COLS   = [METRIC_MEDIAN_GLUCOSE, METRIC_CV, METRIC_TIR70180]
PCA_PHENO_COLS = ["age_study_entry", "diabetes_duration", "zscore", "TIDD"]
PCA_COLS       = PCA_CGM_COLS + PCA_PHENO_COLS

PCA_VAR_COLORS = {
    METRIC_MEDIAN_GLUCOSE: WONG_COLORS["blue"],
    METRIC_CV:             WONG_COLORS["orange"],
    METRIC_TIR70180:       WONG_COLORS["vermillion"],
    "age_study_entry":     WONG_COLORS["bluish_green"],
    "diabetes_duration":   WONG_COLORS["reddish_purple"],
    "zscore":              WONG_COLORS["sky_blue"],
    "TIDD":                WONG_COLORS["yellow"],
}


# =============================================================================
# DATA LOADING
# =============================================================================
def parse_eu_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def recode_insulin_regimen(v) -> str | float:
    """Map the raw numeric regimen code to a short category label (codes 1
    and 3 both denote basal-prandial variants in the source coding)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return np.nan
    if v in (1.0, 3.0):
        return "Basal-prandial"
    if v == 2.0:
        return "Two injections"
    if v == INSULIN_PUMP_CODE:
        return "Insulin pump"
    return np.nan


def load_cgm(apply_filter: bool) -> pd.DataFrame:
    metrics_df  = load_patient_metrics(METRICS_CSV)
    episodes_df = load_episode_summary()
    merged = metrics_df.merge(episodes_df[["patient_id", RESPONSE]], on="patient_id", how="inner")

    if apply_filter:
        pids = load_filtered_patient_list(str(FILTER_STATUS_CSV))
        merged = merged[merged["patient_id"].isin(pids)].reset_index(drop=True)

    for col in PROPORTION_METRICS:
        if col in merged.columns:
            merged[col] = merged[col] * 100.0

    merged["patient_id"] = merged["patient_id"].astype(str).str.strip()
    return merged[["patient_id", RESPONSE] + CGM_COLS]


def load_phenotype_biomarkers() -> pd.DataFrame:
    df = pd.read_csv(FINAL_COHORT_CSV, sep=None, engine="python")
    for col in ["age_study_entry", "diabetes_duration", "zscore", "HbA1c", "IDAA1c", "GTAA1c"]:
        df[col] = parse_eu_float(df[col])
    df["TIDD"] = parse_eu_float(df["TIDD "])
    df[INSULIN_REGIMEN_COL] = df["Insulin_regimen "].apply(recode_insulin_regimen)

    pid_col = "Patient ID" if "Patient ID" in df.columns else df.columns[0]
    df = df.rename(columns={pid_col: "patient_id"})
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    return df[["patient_id"] + PHENOTYPE_COLS + BIOMARKER_COLS + [INSULIN_REGIMEN_COL]]


def build_dataset(apply_filter: bool = True) -> pd.DataFrame:
    cgm   = load_cgm(apply_filter)
    pheno = load_phenotype_biomarkers()
    merged = cgm.merge(pheno, on="patient_id", how="inner")
    merged = merged.dropna(subset=ALL_COLS + [RESPONSE]).reset_index(drop=True)
    print(f"  Merged, complete-case dataset: {len(merged)} patients")
    return merged


def add_panel_label(ax, label, x=-0.08, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="right")


# =============================================================================
# PANEL A: Spearman correlation matrix (CGM + phenotype + biomarkers + n_SLH)
# =============================================================================
def compute_correlation_stats(df: pd.DataFrame):
    """Full Spearman rho/p matrices over ALL_COLS + n_SLH, plus a ranked
    long-format table of each variable's association with n_SLH alone.
    Saves all three to CSV and returns them for plotting/summary reuse."""
    cols   = ALL_COLS + [RESPONSE]
    labels = [LABELS_SHORT[c] for c in ALL_COLS] + [RESPONSE_LABEL]
    n = len(cols)

    rho_mat = np.zeros((n, n))
    p_mat   = np.zeros((n, n))
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            r, p = stats.spearmanr(df[ci], df[cj])
            rho_mat[i, j] = r
            p_mat[i, j]   = p

    rho_df = pd.DataFrame(rho_mat, index=labels, columns=labels)
    rho_df.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix_slh_determinants.csv"))
    p_df = pd.DataFrame(p_mat, index=labels, columns=labels)
    p_df.to_csv(os.path.join(OUTPUT_DIR, "correlation_pvalues_slh_determinants.csv"))

    response_idx = n - 1  # RESPONSE is the last entry of `cols`
    spearman_vs_response_df = pd.DataFrame([
        {
            "variable":     col,
            "label":        LABELS_SHORT[col],
            "domain":       DOMAIN_OF[col],
            "spearman_rho": rho_mat[i, response_idx],
            "p_value":      p_mat[i, response_idx],
            "significant_p<0.05": p_mat[i, response_idx] < 0.05,
        }
        for i, col in enumerate(ALL_COLS)
    ]).sort_values("spearman_rho", key=np.abs, ascending=False).reset_index(drop=True)
    spearman_vs_response_df.to_csv(
        os.path.join(OUTPUT_DIR, "spearman_vs_slh_determinants.csv"), index=False)

    return rho_mat, p_mat, cols, labels, spearman_vs_response_df


def plot_correlation_matrix(ax, rho_mat: np.ndarray, p_mat: np.ndarray,
                             cols: list, labels: list) -> None:
    n = len(cols)

    im = ax.imshow(rho_mat, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, label="Spearman rho", shrink=0.75, pad=0.015)
    cbar.ax.tick_params(labelsize=FONT_SIZE_BASE - 2)
    cbar.set_label("Spearman rho", fontsize=FONT_SIZE_BASE - 1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=FONT_SIZE_BASE - 2)
    ax.set_yticklabels(labels, fontsize=FONT_SIZE_BASE - 2)

    tick_colors = [DOMAIN_COLORS[DOMAIN_OF[c]] for c in ALL_COLS] + ["black"]
    for tick, color in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(color)
        tick.set_fontweight("bold")
    for tick, color in zip(ax.get_yticklabels(), tick_colors):
        tick.set_color(color)
        tick.set_fontweight("bold")

    for i in range(n):
        for j in range(n):
            sig   = "*" if p_mat[i, j] < 0.05 else ""
            txt   = f"{rho_mat[i, j]:.2f}{sig}"
            color = "white" if abs(rho_mat[i, j]) > 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=FONT_SIZE_BASE - 4, color=color)

    for spine in ax.spines.values():
        spine.set_visible(False)

    n_cgm       = len(CGM_COLS)
    n_cgm_pheno = n_cgm + len(PHENOTYPE_COLS)
    n_domains   = len(ALL_COLS)

    for boundary in [n_cgm - 0.5, n_cgm_pheno - 0.5]:
        ax.axhline(boundary, color="black", lw=1.0, ls=":")
        ax.axvline(boundary, color="black", lw=1.0, ls=":")
    ax.axhline(n_domains - 0.5, color="black", lw=1.5, ls="--")
    ax.axvline(n_domains - 0.5, color="black", lw=1.5, ls="--")

    # ── Domain group labels above the matrix ────────────────────────────────
    ax.set_ylim(n - 0.5, -2.6)
    group_bounds = [
        ("CGM",       0,           n_cgm),
        ("Phenotype", n_cgm,       n_cgm_pheno),
        ("Biomarker", n_cgm_pheno, n_domains),
    ]
    for domain, start, end in group_bounds:
        center = (start + end - 1) / 2
        ax.text(center, -1.5, DOMAIN_TITLES[domain], ha="center", va="center",
                fontsize=FONT_SIZE_BASE - 1, fontweight="bold",
                color=DOMAIN_COLORS[domain])
        ax.plot([start - 0.4, end - 0.6], [-0.75, -0.75],
                color=DOMAIN_COLORS[domain], lw=2.2, solid_capstyle="butt",
                clip_on=False)

    ax.set_title(
        "Spearman correlation matrix\n"
        "(* p<0.05  |  dotted = domain boundary  |  dashed = SLH count)",
        fontsize=FONT_SIZE_BASE, fontweight="bold", loc="left", pad=6,
    )
    add_panel_label(ax, "A", x=-0.02, y=1.20)


# =============================================================================
# PANEL B: Mutual information (all 14 variables, coloured by domain)
# =============================================================================
def compute_mutual_information(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ALL_COLS:
        sub = df[[col, RESPONSE]].dropna()
        mi  = mutual_info_regression(sub[[col]], sub[RESPONSE], random_state=42)[0]
        rows.append({"variable": col, "label": LABELS_SHORT[col],
                     "MI": mi, "domain": DOMAIN_OF[col]})
    mi_df = pd.DataFrame(rows).sort_values("MI", ascending=True)
    mi_df.to_csv(os.path.join(OUTPUT_DIR, "mutual_information_slh_determinants.csv"),
                 index=False)
    return mi_df


def plot_mutual_information(ax, mi_df: pd.DataFrame) -> None:
    bar_colors = [DOMAIN_COLORS[d] for d in mi_df["domain"]]
    bars = ax.barh(mi_df["label"], mi_df["MI"], color=bar_colors,
                   edgecolor="white", alpha=0.88)

    for bar, val in zip(bars, mi_df["MI"]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=FONT_SIZE_BASE - 2)

    handles = [mpatches.Patch(color=c, label=DOMAIN_TITLES[d], alpha=0.88)
               for d, c in DOMAIN_COLORS.items()]
    ax.legend(handles=handles, fontsize=FONT_SIZE_BASE - 2, frameon=False,
              loc="lower right")
    ax.set_xlabel(f"Mutual information with {RESPONSE_LABEL.lower()}",
                  fontsize=FONT_SIZE_BASE)
    style_axes(ax)
    ax.set_title("Mutual information - all variables", fontweight="bold",
                 loc="left", pad=6, fontsize=FONT_SIZE_BASE)
    add_panel_label(ax, "B")


# =============================================================================
# PANEL C: Diabetes duration vs GTAA1C, coloured by n_SLH
# =============================================================================
def compute_scatter_stats(df: pd.DataFrame) -> dict:
    x = df["diabetes_duration"].values
    y = df["GTAA1c"].values
    r, p = stats.spearmanr(x, y)
    slope, intercept = np.polyfit(x, y, 1)

    scatter_stats = {
        "x_variable": "diabetes_duration", "y_variable": "GTAA1c", "n": len(x),
        "spearman_rho": r, "p_value": p, "ols_slope": slope, "ols_intercept": intercept,
    }
    pd.DataFrame([scatter_stats]).to_csv(
        os.path.join(OUTPUT_DIR, "scatter_duration_gtaa1c_stats_slh_determinants.csv"),
        index=False)
    return scatter_stats


def plot_scatter_duration_gtaa1c(ax, df: pd.DataFrame, scatter_stats: dict) -> None:
    x = df["diabetes_duration"].values
    y = df["GTAA1c"].values
    c = df[RESPONSE].values
    r, p = scatter_stats["spearman_rho"], scatter_stats["p_value"]

    sc = ax.scatter(x, y, c=c, cmap=COLORMAP_CONTINUOUS, s=42, alpha=0.85,
                    edgecolors="white", linewidths=0.5)

    b, a = scatter_stats["ols_slope"], scatter_stats["ols_intercept"]
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, a + b * x_line, color="black", lw=1.2, ls="--", alpha=0.6)

    cbar = plt.colorbar(sc, ax=ax, label=RESPONSE_LABEL, shrink=0.85)
    cbar.ax.tick_params(labelsize=FONT_SIZE_BASE - 2)

    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    p_text = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    ax.set_xlabel(LABELS_FULL["diabetes_duration"], fontsize=FONT_SIZE_BASE)
    ax.set_ylabel(LABELS_FULL["GTAA1c"], fontsize=FONT_SIZE_BASE)
    style_axes(ax)
    ax.set_title(f"Diabetes duration vs GTAA1C\n(Spearman rho = {r:.2f}{sig}, {p_text})",
                 fontweight="bold", loc="left", pad=6, fontsize=FONT_SIZE_BASE)
    add_panel_label(ax, "C")


# =============================================================================
# PANELS D/E: Combined CGM + phenotype PCA - scores and correlation circle
# =============================================================================
def compute_combined_pca(df: pd.DataFrame):
    X = df[PCA_COLS].values
    y = df[RESPONSE].values

    X_scaled = StandardScaler().fit_transform(X)
    pca      = PCA(n_components=2, random_state=42)
    coords   = pca.fit_transform(X_scaled)
    var_exp  = pca.explained_variance_ratio_ * 100
    loadings_corr = pca.components_.T * np.sqrt(pca.explained_variance_)

    r1, p1 = stats.spearmanr(coords[:, 0], y)
    r2, p2 = stats.spearmanr(coords[:, 1], y)
    print(f"  Combined PCA: PC1={var_exp[0]:.1f}%, PC2={var_exp[1]:.1f}%, "
          f"total={var_exp.sum():.1f}%")
    print(f"  Spearman(PC1, n_SLH) = {r1:.3f} (p={p1:.2e})")
    print(f"  Spearman(PC2, n_SLH) = {r2:.3f} (p={p2:.2e})")

    pd.DataFrame({
        "PC": ["PC1", "PC2"],
        "variance_explained_pct": var_exp,
        "cumulative_variance_pct": np.cumsum(var_exp),
    }).to_csv(os.path.join(OUTPUT_DIR, "pca_variance_explained_slh_determinants.csv"),
              index=False)

    pd.DataFrame({
        "variable": PCA_COLS,
        "label":    [LABELS_SHORT[c] for c in PCA_COLS],
        "domain":   ["CGM" if c in PCA_CGM_COLS else "Phenotype" for c in PCA_COLS],
        "PC1_loading": loadings_corr[:, 0],
        "PC2_loading": loadings_corr[:, 1],
    }).to_csv(os.path.join(OUTPUT_DIR, "pca_loadings_slh_determinants.csv"), index=False)

    pd.DataFrame({
        "patient_id": df["patient_id"].values,
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        RESPONSE: y,
    }).to_csv(os.path.join(OUTPUT_DIR, "pca_scores_slh_determinants.csv"), index=False)

    pc_vs_slh_stats = {"pc1_rho": r1, "pc1_p": p1, "pc2_rho": r2, "pc2_p": p2}
    pd.DataFrame({
        "PC": ["PC1", "PC2"],
        "spearman_rho_vs_nSLH": [r1, r2],
        "p_value": [p1, p2],
    }).to_csv(os.path.join(OUTPUT_DIR, "pca_pc_vs_slh_determinants.csv"), index=False)

    return coords, y, var_exp, loadings_corr, pc_vs_slh_stats


def plot_pca_scores(ax, coords, y, var_exp) -> None:
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=y, cmap=COLORMAP_CONTINUOUS,
                    s=32, alpha=0.8, edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax, label=RESPONSE_LABEL, shrink=0.85)
    cbar.ax.tick_params(labelsize=FONT_SIZE_BASE - 2)
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var.)", fontsize=FONT_SIZE_BASE)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var.)", fontsize=FONT_SIZE_BASE)
    style_axes(ax)
    ax.set_title("Combined CGM + phenotype PCA - scores", fontweight="bold",
                 loc="left", pad=6, fontsize=FONT_SIZE_BASE)
    add_panel_label(ax, "D")


MEDIAN_GLUCOSE_LABEL_NUDGE = (-0.10, 0.09)
AGE_ENTRY_LABEL_NUDGE      = (0.10, -0.09)


def plot_pca_circle(ax, loadings_corr, var_exp) -> None:
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="grey", lw=1, ls="--", alpha=0.5)
    ax.axhline(0, color="grey", lw=0.5, ls="--", alpha=0.5)
    ax.axvline(0, color="grey", lw=0.5, ls="--", alpha=0.5)

    # Median glucose and Age at entry point in nearly the same direction -
    # nudge their labels apart so they don't overlap.
    label_nudge = {
        METRIC_MEDIAN_GLUCOSE: MEDIAN_GLUCOSE_LABEL_NUDGE,
        "age_study_entry":     AGE_ENTRY_LABEL_NUDGE,
    }

    for i, col in enumerate(PCA_COLS):
        lx, ly = loadings_corr[i, 0], loadings_corr[i, 1]
        color  = PCA_VAR_COLORS[col]
        ls     = "solid" if col in PCA_CGM_COLS else (0, (4, 1.5))
        lw     = 2.4 if col in PCA_CGM_COLS else 2.0
        ax.annotate(
            "", xy=(lx, ly), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                            mutation_scale=12),
        )
        scale = 1.15
        nx, ny = label_nudge.get(col, (0.0, 0.0))
        ha = "left" if lx >= 0 else "right"
        va = "bottom" if ly >= 0 else "top"
        ax.text(lx * scale + nx, ly * scale + ny, LABELS_SHORT[col],
                ha=ha, va=va, fontsize=FONT_SIZE_BASE - 1, color=color,
                fontweight="bold")

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var.)", fontsize=FONT_SIZE_BASE)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var.)", fontsize=FONT_SIZE_BASE)
    style_axes(ax)
    ax.set_title("Combined CGM + phenotype PCA - correlation circle\n"
                 "(solid = CGM metric, dashed = phenotype variable)",
                 fontweight="bold", loc="left", pad=6, fontsize=FONT_SIZE_BASE)
    add_panel_label(ax, "E")


# =============================================================================
# PANEL F: n_SLH by insulin regimen (boxplot, Kruskal-Wallis)
# =============================================================================
def compute_insulin_regimen_stats(df: pd.DataFrame) -> dict:
    """Kruskal-Wallis test of n_SLH across insulin-regimen categories, and the
    per-level sample sizes needed to draw panel F. Saved to CSV and returned
    for plotting/summary reuse."""
    sub = df[[INSULIN_REGIMEN_COL, RESPONSE]].dropna()
    levels = [lvl for lvl in INSULIN_REGIMEN_ORDER if (sub[INSULIN_REGIMEN_COL] == lvl).any()]
    groups = [sub.loc[sub[INSULIN_REGIMEN_COL] == lvl, RESPONSE].values for lvl in levels]
    stat, p = stats.kruskal(*groups)

    result = {
        "test": "Kruskal-Wallis", "stat": stat, "p_value": p, "n": len(sub),
        "levels": levels, "n_per_level": [len(g) for g in groups],
    }
    pd.DataFrame([{k: v for k, v in result.items() if k not in ("levels", "n_per_level")}
                  | {"levels": ", ".join(levels),
                     "n_per_level": ", ".join(str(n) for n in result["n_per_level"])}]
                 ).to_csv(os.path.join(OUTPUT_DIR, "insulin_regimen_kruskal_slh_determinants.csv"),
                          index=False)
    return result


def plot_slh_by_insulin_regimen(ax, df: pd.DataFrame, regimen_stats: dict) -> None:
    sub = df[[INSULIN_REGIMEN_COL, RESPONSE]].dropna()
    levels = regimen_stats["levels"]
    groups = [sub.loc[sub[INSULIN_REGIMEN_COL] == lvl, RESPONSE].values for lvl in levels]
    colors = INSULIN_REGIMEN_COLORS[:len(levels)]

    bp = ax.boxplot(
        groups, patch_artist=True, notch=False,
        medianprops=dict(color="black", lw=2),
        whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    rng = np.random.default_rng(42)
    for i, (grp, color) in enumerate(zip(groups, colors), start=1):
        jitter = rng.uniform(-0.18, 0.18, size=len(grp))
        ax.scatter(np.full(len(grp), i) + jitter, grp, color=color, s=12, alpha=0.5, zorder=3)

    ax.set_xticks(range(1, len(levels) + 1))
    ax.set_xticklabels(
        [f"{lvl}\n(n={n})" for lvl, n in zip(levels, regimen_stats["n_per_level"])],
        fontsize=FONT_SIZE_BASE - 1,
    )
    ax.set_ylabel(RESPONSE_LABEL, fontsize=FONT_SIZE_BASE)

    p = regimen_stats["p_value"]
    p_text = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    ax.set_title(f"n_SLH by insulin regimen\n(Kruskal-Wallis, {p_text})",
                 fontweight="bold", loc="left", pad=6, fontsize=FONT_SIZE_BASE)
    style_axes(ax)
    add_panel_label(ax, "F")


# =============================================================================
# PANEL G: Diabetes duration vs GTAA1C, coloured by n_SLH, shaped by insulin regimen
# =============================================================================
def plot_scatter_duration_gtaa1c_by_insulin(ax, df: pd.DataFrame, scatter_stats: dict) -> None:
    """Same duration-vs-GTAA1C relationship as panel C, split by insulin
    regimen via marker shape - checks whether the association holds within
    each treatment modality rather than only in aggregate."""
    sub = df.dropna(subset=["diabetes_duration", "GTAA1c", RESPONSE, INSULIN_REGIMEN_COL])
    x = sub["diabetes_duration"].values
    y = sub["GTAA1c"].values
    c = sub[RESPONSE].values
    r, p = scatter_stats["spearman_rho"], scatter_stats["p_value"]

    norm = mcolors.Normalize(vmin=c.min(), vmax=c.max())
    sc = None
    for lvl, marker in INSULIN_REGIMEN_MARKERS.items():
        mask = (sub[INSULIN_REGIMEN_COL] == lvl).values
        if not mask.any():
            continue
        sc = ax.scatter(x[mask], y[mask], c=c[mask], cmap=COLORMAP_CONTINUOUS, norm=norm,
                        marker=marker, s=42, alpha=0.85, edgecolors="white",
                        linewidths=0.5, zorder=3)

    b, a = scatter_stats["ols_slope"], scatter_stats["ols_intercept"]
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, a + b * x_line, color="black", lw=1.2, ls="--", alpha=0.6, zorder=1)

    cbar = plt.colorbar(sc, ax=ax, label=RESPONSE_LABEL, shrink=0.85)
    cbar.ax.tick_params(labelsize=FONT_SIZE_BASE - 2)

    legend_handles = [
        Line2D([0], [0], marker=marker, color="w", markerfacecolor="#999999",
              markeredgecolor="black", markersize=7, label=lvl)
        for lvl, marker in INSULIN_REGIMEN_MARKERS.items()
        if (sub[INSULIN_REGIMEN_COL] == lvl).any()
    ]
    ax.legend(handles=legend_handles, title="Insulin regimen",
             fontsize=FONT_SIZE_BASE - 2, title_fontsize=FONT_SIZE_BASE - 2,
             loc="best", framealpha=0.9)

    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    p_text = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    ax.set_xlabel(LABELS_FULL["diabetes_duration"], fontsize=FONT_SIZE_BASE)
    ax.set_ylabel(LABELS_FULL["GTAA1c"], fontsize=FONT_SIZE_BASE)
    style_axes(ax)
    ax.set_title(
        f"Diabetes duration vs GTAA1C by insulin regimen\n(Spearman rho = {r:.2f}{sig}, {p_text})",
        fontweight="bold", loc="left", pad=6, fontsize=FONT_SIZE_BASE)
    add_panel_label(ax, "G")


# =============================================================================
# INDIVIDUAL PANEL EXPORT
# =============================================================================
def save_individual_panel(plot_fn, figsize: tuple, filename: str, *args, **kwargs) -> None:
    """Render one panel-drawing function standalone (its own figure) into
    Figure_Determinants/, reusing the exact drawing function used inside the
    combined Figure 5 GridSpec - no plotting logic is duplicated."""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    plot_fn(ax, *args, **kwargs)
    fig.tight_layout()
    save_figure(fig, os.path.join(INDIVIDUAL_DIR, filename))


# =============================================================================
# CONSOLIDATED HUMAN-READABLE SUMMARY
# =============================================================================
def write_summary_txt(df: pd.DataFrame, spearman_vs_response_df: pd.DataFrame,
                       mi_df: pd.DataFrame, var_exp: np.ndarray,
                       pc_vs_slh_stats: dict, scatter_stats: dict,
                       regimen_stats: dict) -> str:
    """Write a single human-readable digest of every statistic exported by
    this script, so the Results text can be written/checked without opening
    each individual CSV."""
    y = df[RESPONSE].values
    top_rho  = spearman_vs_response_df.head(5)
    top_mi   = mi_df.sort_values("MI", ascending=False).head(5)

    lines = [
        "=" * 70,
        "  FIGURE 5 - SLH DETERMINANTS: RESULTS DIGEST",
        "=" * 70,
        "",
        f"Cohort (complete cases across CGM + phenotype + biomarkers): n = {len(df)}",
        f"  {RESPONSE_LABEL}: mean={y.mean():.1f}, median={np.median(y):.0f}, "
        f"IQR={np.percentile(y, 25):.0f}-{np.percentile(y, 75):.0f}, "
        f"range={y.min():.0f}-{y.max():.0f}",
        "",
        "TOP 5 SPEARMAN CORRELATES OF n_SLH (see spearman_vs_slh_determinants.csv",
        "for all 14 variables):",
    ]
    for _, row in top_rho.iterrows():
        sig = "*" if row["significant_p<0.05"] else ""
        lines.append(f"  {row['label']:20s} ({row['domain']:10s})  "
                     f"rho={row['spearman_rho']:+.3f}{sig}  p={row['p_value']:.2e}")

    lines += ["", "TOP 5 BY MUTUAL INFORMATION (see mutual_information_slh_determinants.csv"
                    " for all 14 variables):"]
    for _, row in top_mi.iterrows():
        lines.append(f"  {row['label']:20s} ({row['domain']:10s})  MI={row['MI']:.3f}")

    lines += [
        "",
        f"SCATTER - Diabetes duration vs GTAA1C (see "
        f"scatter_duration_gtaa1c_stats_slh_determinants.csv):",
        f"  n={scatter_stats['n']}, Spearman rho={scatter_stats['spearman_rho']:+.3f}, "
        f"p={scatter_stats['p_value']:.2e}",
        f"  OLS fit: GTAA1C = {scatter_stats['ols_intercept']:.3f} + "
        f"{scatter_stats['ols_slope']:.3f} x diabetes_duration",
        "",
        "COMBINED CGM + PHENOTYPE PCA (see pca_variance_explained_slh_determinants.csv,",
        "pca_loadings_slh_determinants.csv, pca_scores_slh_determinants.csv):",
        f"  PC1: {var_exp[0]:.1f}% variance explained  |  "
        f"Spearman(PC1, n_SLH) = {pc_vs_slh_stats['pc1_rho']:+.3f} "
        f"(p={pc_vs_slh_stats['pc1_p']:.2e})",
        f"  PC2: {var_exp[1]:.1f}% variance explained  |  "
        f"Spearman(PC2, n_SLH) = {pc_vs_slh_stats['pc2_rho']:+.3f} "
        f"(p={pc_vs_slh_stats['pc2_p']:.2e})",
        f"  Total (PC1+PC2): {var_exp.sum():.1f}% variance explained",
        "",
        f"INSULIN REGIMEN - n_SLH by category (see "
        f"insulin_regimen_kruskal_slh_determinants.csv):",
        f"  levels: {', '.join(f'{lvl} (n={n})' for lvl, n in zip(regimen_stats['levels'], regimen_stats['n_per_level']))}",
        f"  Kruskal-Wallis stat={regimen_stats['stat']:.3f}, p={regimen_stats['p_value']:.3f}",
        "",
        "=" * 70,
    ]
    summary = "\n".join(lines)

    summary_path = os.path.join(OUTPUT_DIR, "summary_statistics_slh_determinants.txt")
    with open(summary_path, "w") as fh:
        fh.write(summary)
    print(f"  Saved: {summary_path}")
    return summary


# =============================================================================
# MAIN FIGURE GENERATION
# =============================================================================
def generate_figure_5(apply_filter: bool = True) -> None:
    apply_publication_style()

    print("\n" + "=" * 70)
    print("  GENERATING FIGURE 5 - SLH DETERMINANTS")
    print("  (CGM metrics x phenotype x biomarkers)")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n  Loading and merging CGM + phenotype + biomarker data ...")
    df = build_dataset(apply_filter=apply_filter)
    print(f"  n_SLH: mean={df[RESPONSE].mean():.1f}, median={df[RESPONSE].median():.0f}, "
          f"max={df[RESPONSE].max():.0f}, n={len(df)}")

    print("\n  Computing Spearman correlation matrix ...")
    rho_mat, p_mat, cols, labels, spearman_vs_response_df = compute_correlation_stats(df)

    print("\n  Computing mutual information ...")
    mi_df = compute_mutual_information(df)

    print("\n  Computing diabetes duration vs GTAA1C scatter statistics ...")
    scatter_stats = compute_scatter_stats(df)

    print("\n  Computing combined CGM + phenotype PCA ...")
    coords, y, var_exp, loadings_corr, pc_vs_slh_stats = compute_combined_pca(df)

    print("\n  Computing n_SLH by insulin regimen (Kruskal-Wallis) ...")
    regimen_stats = compute_insulin_regimen_stats(df)

    fig = plt.figure(figsize=(8.6, 14.2))
    gs = gridspec.GridSpec(
        4, 2, figure=fig, height_ratios=[1.25, 1.0, 1.0, 1.0],
        hspace=0.55, wspace=0.38,
        left=0.10, right=0.90, top=0.965, bottom=0.035,
    )

    ax_a = fig.add_subplot(gs[0, :])
    plot_correlation_matrix(ax_a, rho_mat, p_mat, cols, labels)

    ax_b = fig.add_subplot(gs[1, 0])
    plot_mutual_information(ax_b, mi_df)

    ax_c = fig.add_subplot(gs[1, 1])
    plot_scatter_duration_gtaa1c(ax_c, df, scatter_stats)

    ax_d = fig.add_subplot(gs[2, 0])
    plot_pca_scores(ax_d, coords, y, var_exp)

    ax_e = fig.add_subplot(gs[2, 1])
    plot_pca_circle(ax_e, loadings_corr, var_exp)

    ax_f = fig.add_subplot(gs[3, 0])
    plot_slh_by_insulin_regimen(ax_f, df, regimen_stats)

    ax_g = fig.add_subplot(gs[3, 1])
    plot_scatter_duration_gtaa1c_by_insulin(ax_g, df, scatter_stats)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    save_figure(fig, output_path)

    print("\n  Writing consolidated results digest ...")
    write_summary_txt(df, spearman_vs_response_df, mi_df, var_exp,
                       pc_vs_slh_stats, scatter_stats, regimen_stats)

    print("\n  Saving individual panels to Figure_Determinants/ ...")
    os.makedirs(INDIVIDUAL_DIR, exist_ok=True)
    save_individual_panel(
        lambda ax: plot_correlation_matrix(ax, rho_mat, p_mat, cols, labels),
        (9.5, 8.5), "panelA_correlation_matrix")
    save_individual_panel(
        lambda ax: plot_mutual_information(ax, mi_df),
        (6.5, 5.5), "panelB_mutual_information")
    save_individual_panel(
        lambda ax: plot_scatter_duration_gtaa1c(ax, df, scatter_stats),
        (5.5, 4.5), "panelC_scatter_duration_gtaa1c")
    save_individual_panel(
        lambda ax: plot_pca_scores(ax, coords, y, var_exp),
        (5.5, 4.5), "panelD_pca_scores")
    save_individual_panel(
        lambda ax: plot_pca_circle(ax, loadings_corr, var_exp),
        (5.5, 5.5), "panelE_pca_correlation_circle")
    save_individual_panel(
        lambda ax: plot_slh_by_insulin_regimen(ax, df, regimen_stats),
        (5.0, 4.5), "panelF_boxplot_insulin_regimen")
    save_individual_panel(
        lambda ax: plot_scatter_duration_gtaa1c_by_insulin(ax, df, scatter_stats),
        (6.0, 5.0), "panelG_scatter_duration_gtaa1c_insulin")
    print(f"  Saved: {INDIVIDUAL_DIR}/panel{{A-G}}_*.{{pdf,svg}}")

    print("\n" + "=" * 70)
    print(f"  SAVED: {output_path}")
    print("=" * 70 + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no_filter", action="store_true",
                   help="Skip the study-cohort filter (use all patients with valid metrics)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_figure_5(apply_filter=not args.no_filter)
