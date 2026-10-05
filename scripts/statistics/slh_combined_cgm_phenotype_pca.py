"""
scripts/slh_combined_cgm_phenotype_pca.py
===========================================
Combined CGM + phenotype PCA, aimed at characterising the "honeymoon phase"
signature of SLH: a single low-dimensional axis spanning both the patient's
actual glycaemic pattern (CGM metrics) and their clinical characteristics
(phenotype), rather than treating the two domains as separate PCAs.

Variable selection
-------------------
Combines the two *already independent* representative sets established by
the standalone CGM and phenotype explorations, MINUS one further exclusion
needed specifically because we're now merging domains:

  CGM_COLS   = Median glucose, CV, TIR 70-180
               (see scripts/slh_baseline_metrics_multivariate_exploration.py)
  PHENO_COLS = Age at study entry, Diabetes duration, BMI z-score, TIDD
               (see scripts/slh_patient_characteristics_exploration.py,
               MINUS GTAA1C/HbA1c/IDAA1c)

GTAA1C, HbA1c and IDAA1c are dropped here specifically because GTAA1C is
*defined* as HbA1c - 3 x TIR(70-180) - it is not just empirically correlated
with the CGM metrics, it is mechanically built from one of them. Confirmed
empirically too: rho(GTAA1C, TIR70180) = -0.71, rho(GTAA1C, Median glucose)
= +0.65 in this cohort - both far above the ~0.4-0.5 ceiling tolerated
elsewhere in this project's PCAs. Including it here would double-count the
TIR signal under a different name. TIDD/age/duration/BMI carry information
that is NOT CGM-derived, so they combine safely with the 3 CGM metrics
(max remaining pairwise |rho| among the 7 retained variables is 0.53,
CV vs TIDD - see fig_correlation_heatmap / correlation_matrix.csv).

Outputs (RESULTS_PATH / SLH_Combined_CGM_Phenotype_PCA)
---------------------------------------------------------
  fig_correlation_heatmap.{pdf,png,svg}
  fig_pca_scores.{pdf,png,svg}
  fig_pca_correlation_circle.{pdf,png,svg}
  correlation_matrix.csv

Usage
-----
    env/bin/python3 -m scripts.statistics.slh_combined_cgm_phenotype_pca [--no_filter]

Author
------
Guillaume Deside

Date
----
2026
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import METRICS_CSV, FILTER_STATUS_CSV
from src.config.paths import RESULTS_PATH, FINAL_COHORT_CSV
from src.metrics.metrics import METRIC_MEDIAN_GLUCOSE, METRIC_CV, METRIC_TIR70180
from src.episodes.load_episodes import load_episode_summary
from src.metrics.load_metrics import load_patient_metrics
from src.filtering.load_filtering import load_filtered_patient_list
from src.visualization.colors import WONG_COLORS
from src.visualization.style import apply_publication_style, save_figure, FONT_SIZE_BASE

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = RESULTS_PATH / "SLH_Combined_CGM_Phenotype_PCA"

RESPONSE       = "n_SLH"
RESPONSE_LABEL = "Number of SLH episodes"

CGM_COLS = [METRIC_MEDIAN_GLUCOSE, METRIC_CV, METRIC_TIR70180]
PROPORTION_METRICS = [METRIC_CV, METRIC_TIR70180]

PHENO_COLS = ["age_study_entry", "diabetes_duration", "zscore", "TIDD"]

COMBINED_COLS = CGM_COLS + PHENO_COLS

LABELS = {
    METRIC_MEDIAN_GLUCOSE: "Median glucose (mg/dL)",
    METRIC_CV:             "Glucose CV (%)",
    METRIC_TIR70180:       "TIR 70-180 (%)",
    "age_study_entry":     "Age at study entry (years)",
    "diabetes_duration":   "Diabetes duration (years)",
    "zscore":              "BMI z-score",
    "TIDD":                "Total daily insulin dose (U/kg)",
}
LABELS_SHORT = {
    METRIC_MEDIAN_GLUCOSE: "Median glucose",
    METRIC_CV:             "CV",
    METRIC_TIR70180:       "TIR 70-180",
    "age_study_entry":     "Age at entry",
    "diabetes_duration":   "Diabetes duration",
    "zscore":              "BMI z-score",
    "TIDD":                "TIDD",
}

# Colour-blind-safe: CGM metrics in blue/orange/vermillion (matches the
# standalone CGM PCA), phenotype variables in the remaining Wong colours.
VAR_COLORS = {
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


def load_phenotype() -> pd.DataFrame:
    df = pd.read_csv(FINAL_COHORT_CSV, sep=None, engine="python")
    for col in ["age_study_entry", "diabetes_duration", "zscore"]:
        df[col] = parse_eu_float(df[col])
    df["TIDD"] = parse_eu_float(df["TIDD "])

    pid_col = "Patient ID" if "Patient ID" in df.columns else df.columns[0]
    df = df.rename(columns={pid_col: "patient_id"})
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    return df[["patient_id"] + PHENO_COLS]


def build_dataset(apply_filter: bool = True) -> pd.DataFrame:
    cgm   = load_cgm(apply_filter)
    pheno = load_phenotype()
    merged = cgm.merge(pheno, on="patient_id", how="inner")
    merged = merged.dropna(subset=COMBINED_COLS + [RESPONSE]).reset_index(drop=True)
    print(f"  Merged, complete-case dataset: {len(merged)} patients")
    return merged


# =============================================================================
# PLOT 1: Spearman correlation heatmap (combined 7 variables + response)
# =============================================================================
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    cols   = COMBINED_COLS + [RESPONSE]
    labels = [LABELS_SHORT[c] for c in COMBINED_COLS] + [RESPONSE_LABEL]

    n = len(cols)
    rho_mat = np.zeros((n, n))
    p_mat   = np.zeros((n, n))
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            r, p = stats.spearmanr(df[ci], df[cj])
            rho_mat[i, j] = r
            p_mat[i, j]   = p

    rho_df = pd.DataFrame(rho_mat, index=labels, columns=labels)
    rho_df.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    cmap = matplotlib.colormaps["RdBu_r"]
    im = ax.imshow(rho_mat, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman rho", shrink=0.8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=FONT_SIZE_BASE - 1)
    ax.set_yticklabels(labels, fontsize=FONT_SIZE_BASE - 1)

    for i in range(n):
        for j in range(n):
            sig   = "*" if p_mat[i, j] < 0.05 else ""
            txt   = f"{rho_mat[i, j]:.2f}{sig}"
            color = "white" if abs(rho_mat[i, j]) > 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=FONT_SIZE_BASE - 2, color=color)

    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_visible(False)
    ax.axhline(len(CGM_COLS) - 0.5, color="black", lw=1.2, ls=":")
    ax.axvline(len(CGM_COLS) - 0.5, color="black", lw=1.2, ls=":")
    ax.axhline(n - 1.5, color="black", lw=1.5, ls="--")
    ax.axvline(n - 1.5, color="black", lw=1.5, ls="--")

    ax.set_title(
        "Spearman correlation: combined CGM + phenotype variables & SLH count\n"
        "(* p < 0.05  |  dotted line separates CGM from phenotype variables)",
        fontsize=FONT_SIZE_BASE,
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig_correlation_heatmap")
    print("  Saved: fig_correlation_heatmap")


# =============================================================================
# PLOT 2 & 3: PCA scores + correlation circle
# =============================================================================
def plot_pca(df: pd.DataFrame) -> None:
    X = df[COMBINED_COLS].values
    y = df[RESPONSE].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100

    loadings_corr = pca.components_.T * np.sqrt(pca.explained_variance_)

    # ── Scores ────────────────────────────────────────────────────────────────
    apply_publication_style()
    fig_sc, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=y, cmap="plasma",
                    s=35, alpha=0.75, edgecolors="none")
    plt.colorbar(sc, ax=ax, label=RESPONSE_LABEL, shrink=0.85)
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var.)", fontsize=FONT_SIZE_BASE)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var.)", fontsize=FONT_SIZE_BASE)
    ax.set_title("Combined CGM + phenotype PCA — patient scores\n(coloured by n_SLH)",
                 fontsize=FONT_SIZE_BASE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig_sc.tight_layout()
    save_figure(fig_sc, OUTPUT_DIR / "fig_pca_scores")
    print("  Saved: fig_pca_scores")

    # ── Correlation circle ────────────────────────────────────────────────────
    apply_publication_style()
    fig_cc, ax2 = plt.subplots(figsize=(6, 6))

    theta = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(theta), np.sin(theta), color="grey", lw=1, ls="--", alpha=0.5)
    ax2.axhline(0, color="grey", lw=0.5, ls="--", alpha=0.5)
    ax2.axvline(0, color="grey", lw=0.5, ls="--", alpha=0.5)

    for i, col in enumerate(COMBINED_COLS):
        lx, ly = loadings_corr[i, 0], loadings_corr[i, 1]
        color  = VAR_COLORS[col]
        style  = "-|>" if col in CGM_COLS else "-|>"
        lw     = 2.4 if col in CGM_COLS else 2.0
        ls     = "solid" if col in CGM_COLS else (0, (4, 1.5))
        ax2.annotate(
            "", xy=(lx, ly), xytext=(0, 0),
            arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls, mutation_scale=12),
        )
        scale = 1.13
        ha = "left" if lx >= 0 else "right"
        va = "bottom" if ly >= 0 else "top"
        ax2.text(lx * scale, ly * scale, LABELS_SHORT[col],
                 ha=ha, va=va, fontsize=FONT_SIZE_BASE - 1,
                 color=color, fontweight="bold")

    ax2.set_xlim(-1.3, 1.3)
    ax2.set_ylim(-1.3, 1.3)
    ax2.set_aspect("equal")
    ax2.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var.)", fontsize=FONT_SIZE_BASE)
    ax2.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var.)", fontsize=FONT_SIZE_BASE)
    ax2.set_title(
        "Combined CGM + phenotype PCA — correlation circle\n"
        "(solid = CGM metric, dashed = phenotype variable)",
        fontsize=FONT_SIZE_BASE,
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig_cc.tight_layout()
    save_figure(fig_cc, OUTPUT_DIR / "fig_pca_correlation_circle")
    print("  Saved: fig_pca_correlation_circle")
    print(f"  PCA: PC1={var_exp[0]:.1f}%, PC2={var_exp[1]:.1f}%, total={var_exp.sum():.1f}%")

    r1, p1 = stats.spearmanr(coords[:, 0], y)
    r2, p2 = stats.spearmanr(coords[:, 1], y)
    print(f"  Spearman(PC1, n_SLH) = {r1:.3f} (p={p1:.2e})")
    print(f"  Spearman(PC2, n_SLH) = {r2:.3f} (p={p2:.2e})")


# =============================================================================
# MAIN
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no_filter", action="store_true",
                   help="Skip the study-cohort filter (use all patients with valid metrics)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    print("\n[1/3] Loading and merging CGM + phenotype data ...")
    df = build_dataset(apply_filter=not args.no_filter)

    print("\n[2/3] Correlation heatmap ...")
    plot_correlation_heatmap(df)

    print("\n[3/3] PCA scores + correlation circle ...")
    plot_pca(df)

    print(f"\nDone. All outputs saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
