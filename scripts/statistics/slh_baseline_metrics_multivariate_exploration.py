"""
scripts/statistics/slh_baseline_metrics_multivariate_exploration.py
===================================================================
Non-parametric multivariate exploration of the relationship between
baseline CGM metrics and SLH episode count.

No regression models are fitted here. The goal is visual and
rank-based exploration:

  1. Spearman correlation heatmap  — all metrics x metrics + n_SLH (rank-based)
  2. Mutual information            — MI(metric, n_SLH) bar chart (all metrics)
  3. PCA scores                    — patients projected onto PC1/PC2,
                                     coloured by n_SLH (reduced metric set)
  4. PCA correlation circle        — proper correlation circle showing metric
                                     contributions (reduced metric set)
  5. Stratified box plots          — metrics split by SLH tertile
  6. Parallel coordinates          — each patient as a poly-line across z-scored
                                     metrics, coloured by n_SLH
  7. SLH-count distribution        — simple histogram for reference

Metric sets
-----------
  METRICS_ALL  — all 7 metrics; used for heatmap and MI to show collinearity
  METRICS      — 3 independent representatives retained for PCA:
                   Median glucose (level), CV (variability), TIR 70-180 (in-range time)
                 Mean glucose ~ Median, SD ~ CV x Mean, TAR/TBR partition TIR -> dropped

Inputs
------
  Data/Metrics/metrics_all_patients.csv
  Episodes/episodes_summary.csv
  Filtered/..._patient_status.csv

Outputs (written under RESULTS_PATH/SLH_Baseline_Exploration)
--------------------------------------------------------------
  fig_correlation_heatmap.{pdf,png,svg}
  fig_mutual_information.{pdf,png,svg}
  fig_pca_scores.{pdf,png,svg}
  fig_pca_correlation_circle.{pdf,png,svg}
  fig_stratified_boxplots.{pdf,png,svg}
  fig_parallel_coordinates.{pdf,png,svg}
  fig_slh_distribution.{pdf,png,svg}
  spearman_matrix.csv
  mutual_information.csv

Usage
-----
    python3 -m scripts.statistics.slh_baseline_metrics_multivariate_exploration [--no_filter]

Authors
-------
Guillaume Deside
UCL - counterfactual-cgm-slh project

Date
----
2025
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

from src.config import (
    METRICS_CSV,
    FILTER_STATUS_CSV,
)
from src.config.paths import RESULTS_PATH
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
from src.visualization.colors import WONG_COLORS
from src.visualization.style import apply_publication_style, save_figure, FONT_SIZE_BASE


# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = RESULTS_PATH / "SLH_Baseline_Exploration"

# All 7 metrics — used for correlation heatmap and MI to expose collinearity
METRICS_ALL = [
    METRIC_MEDIAN_GLUCOSE,
    METRIC_MEAN_GLUCOSE,
    METRIC_SD,
    METRIC_CV,
    METRIC_TIR70180,
    METRIC_TAR180,
    METRIC_TBR70,
]

# Reduced independent set for PCA / boxplots / parallel coordinates:
#   Mean glucose ~ Median glucose             -> drop Mean
#   SD ~ CV x Mean (linear transform)         -> drop SD
#   TAR >180 + TBR <70 partition TIR range    -> drop TAR, TBR
METRICS = [
    METRIC_MEDIAN_GLUCOSE,
    METRIC_CV,
    METRIC_TIR70180,
]

METRIC_LABELS = {
    METRIC_MEDIAN_GLUCOSE: "Median glucose\n(mg/dL)",
    METRIC_MEAN_GLUCOSE:   "Mean glucose\n(mg/dL)",
    METRIC_SD:             "Glucose SD\n(mg/dL)",
    METRIC_CV:             "Glucose CV\n(%)",
    METRIC_TIR70180:       "TIR 70-180\n(%)",
    METRIC_TAR180:         "TAR >180\n(%)",
    METRIC_TBR70:          "TBR <70\n(%)",
}

METRIC_LABELS_SHORT = {
    METRIC_MEDIAN_GLUCOSE: "Median glucose",
    METRIC_MEAN_GLUCOSE:   "Mean glucose",
    METRIC_SD:             "SD",
    METRIC_CV:             "CV",
    METRIC_TIR70180:       "TIR 70-180",
    METRIC_TAR180:         "TAR >180",
    METRIC_TBR70:          "TBR <70",
}

PROPORTION_METRICS = [METRIC_CV, METRIC_TIR70180, METRIC_TAR180, METRIC_TBR70]

RESPONSE       = "n_SLH"
RESPONSE_LABEL = "Number of SLH episodes"

N_TERTILES = 3
TERTILE_COLORS = [WONG_COLORS["blue"], WONG_COLORS["orange"], WONG_COLORS["vermillion"]]

# One distinct colour per retained metric, used in the correlation circle
METRIC_COLORS = {
    METRIC_MEDIAN_GLUCOSE: WONG_COLORS["blue"],
    METRIC_CV:             WONG_COLORS["orange"],
    METRIC_TIR70180:       WONG_COLORS["vermillion"],
}


# =============================================================================
# DATA LOADING
# =============================================================================
def load_and_merge(apply_filter: bool = True) -> pd.DataFrame:
    metrics_df  = load_patient_metrics(METRICS_CSV)
    episodes_df = load_episode_summary()

    merged = metrics_df.merge(
        episodes_df[["patient_id", RESPONSE]],
        on="patient_id",
        how="inner",
    )

    if apply_filter:
        filtered_pids = load_filtered_patient_list(FILTER_STATUS_CSV)
        before = len(merged)
        merged = merged[merged["patient_id"].isin(filtered_pids)].reset_index(drop=True)
        print(f"  Cohort filter: {before} -> {len(merged)} patients")

    for col in PROPORTION_METRICS:
        if col in merged.columns:
            merged[col] = merged[col] * 100.0

    merged = merged.dropna(subset=METRICS_ALL + [RESPONSE]).reset_index(drop=True)
    print(f"  Complete cases for analysis: {len(merged)} patients")
    return merged


def add_tertile(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["slh_tertile"] = pd.qcut(
        df[RESPONSE], q=N_TERTILES, labels=[0, 1, 2], duplicates="drop"
    ).astype(int)
    return df


def slh_colormap(values: np.ndarray):
    cmap = matplotlib.colormaps["plasma"]
    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
    return cmap, norm


# =============================================================================
# PLOT 1: Spearman correlation heatmap (all metrics + n_SLH)
# =============================================================================
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    # Use METRICS_ALL to show collinearity; retained metrics highlighted
    cols   = METRICS_ALL + [RESPONSE]
    labels = [METRIC_LABELS_SHORT[m] for m in METRICS_ALL] + [RESPONSE_LABEL]

    n = len(cols)
    rho_mat = np.zeros((n, n))
    p_mat   = np.zeros((n, n))
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            r, p = stats.spearmanr(df[ci], df[cj])
            rho_mat[i, j] = r
            p_mat[i, j]   = p

    rho_df = pd.DataFrame(rho_mat, index=labels, columns=labels)
    rho_df.to_csv(OUTPUT_DIR / "spearman_matrix.csv")

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
    # Dashed separator before n_SLH
    ax.axhline(n - 1.5, color="black", lw=1.5, ls="--")
    ax.axvline(n - 1.5, color="black", lw=1.5, ls="--")

    # Mark retained metrics on the diagonal
    for m in METRICS:
        ki = METRICS_ALL.index(m)
        rect = plt.Rectangle((ki - 0.5, ki - 0.5), 1, 1,
                              linewidth=2, edgecolor="black", facecolor="none")
        ax.add_patch(rect)

    ax.set_title(
        "Spearman correlation: all CGM metrics & SLH count\n"
        "(* p < 0.05  |  black square = retained for PCA)",
        fontsize=FONT_SIZE_BASE,
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig_correlation_heatmap")
    print("  Saved: fig_correlation_heatmap")


# =============================================================================
# PLOT 2: Mutual information bar chart (all metrics)
# =============================================================================
def plot_mutual_information(df: pd.DataFrame) -> None:
    X = df[METRICS_ALL].values
    y = df[RESPONSE].values

    mi = mutual_info_regression(X, y, random_state=42)

    mi_df = pd.DataFrame({
        "Metric":   [METRIC_LABELS_SHORT[m] for m in METRICS_ALL],
        "MI":       mi,
        "retained": [m in METRICS for m in METRICS_ALL],
    }).sort_values("MI", ascending=True)
    mi_df.to_csv(OUTPUT_DIR / "mutual_information.csv", index=False)

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    bar_colors = [WONG_COLORS["blue"] if r else WONG_COLORS["bluish_green"]
                  for r in mi_df["retained"]]
    bars = ax.barh(mi_df["Metric"], mi_df["MI"], color=bar_colors, edgecolor="white")

    for bar, val in zip(bars, mi_df["MI"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=FONT_SIZE_BASE - 1)

    handles = [
        mpatches.Patch(color=WONG_COLORS["blue"],        label="Retained for PCA"),
        mpatches.Patch(color=WONG_COLORS["bluish_green"], label="Excluded (correlated)"),
    ]
    ax.legend(handles=handles, fontsize=FONT_SIZE_BASE - 2, frameon=False)
    ax.set_xlabel("Mutual information with n_SLH", fontsize=FONT_SIZE_BASE)
    ax.set_title("Non-parametric association strength\n(mutual information, all metrics)",
                 fontsize=FONT_SIZE_BASE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig_mutual_information")
    print("  Saved: fig_mutual_information")


# =============================================================================
# PLOT 3 & 4: PCA — scores and correlation circle (separate files)
# =============================================================================
def plot_pca(df: pd.DataFrame) -> None:
    X = df[METRICS].values
    y = df[RESPONSE].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    var_exp = pca.explained_variance_ratio_ * 100
    total   = var_exp.sum()

    # Correlation circle loadings: corr(original variable, PC)
    # = eigenvector component * sqrt(eigenvalue)
    loadings_corr = pca.components_.T * np.sqrt(pca.explained_variance_)

    # ── Figure A: patient scores ──────────────────────────────────────────────
    apply_publication_style()
    fig_sc, ax = plt.subplots(figsize=(5, 4.5))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=y, cmap="plasma",
                    s=35, alpha=0.75, edgecolors="none")
    plt.colorbar(sc, ax=ax, label=RESPONSE_LABEL, shrink=0.85)
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var.)", fontsize=FONT_SIZE_BASE)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var.)", fontsize=FONT_SIZE_BASE)
    ax.set_title("PCA — patient scores\n(coloured by n_SLH)", fontsize=FONT_SIZE_BASE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig_sc.tight_layout()
    save_figure(fig_sc, OUTPUT_DIR / "fig_pca_scores")
    print("  Saved: fig_pca_scores")

    # ── Figure B: correlation circle ──────────────────────────────────────────
    apply_publication_style()
    fig_cc, ax2 = plt.subplots(figsize=(5, 5))

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(theta), np.sin(theta), color="grey", lw=1, ls="--", alpha=0.5)
    ax2.axhline(0, color="grey", lw=0.5, ls="--", alpha=0.5)
    ax2.axvline(0, color="grey", lw=0.5, ls="--", alpha=0.5)

    for i, metric in enumerate(METRICS):
        lx, ly = loadings_corr[i, 0], loadings_corr[i, 1]
        color  = METRIC_COLORS[metric]
        ax2.annotate(
            "", xy=(lx, ly), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=12),
        )
        # Position label slightly beyond tip; align to avoid overlap
        scale = 1.13
        ha = "left" if lx >= 0 else "right"
        va = "bottom" if ly >= 0 else "top"
        ax2.text(lx * scale, ly * scale, METRIC_LABELS_SHORT[metric],
                 ha=ha, va=va, fontsize=FONT_SIZE_BASE, color=color, fontweight="bold")

    ax2.set_xlim(-1.3, 1.3)
    ax2.set_ylim(-1.3, 1.3)
    ax2.set_aspect("equal")
    ax2.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var.)", fontsize=FONT_SIZE_BASE)
    ax2.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var.)", fontsize=FONT_SIZE_BASE)
    ax2.set_title(
        "PCA — correlation circle\n"
        "(arrow length = correlation of variable with PC)",
        fontsize=FONT_SIZE_BASE,
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig_cc.tight_layout()
    save_figure(fig_cc, OUTPUT_DIR / "fig_pca_correlation_circle")
    print("  Saved: fig_pca_correlation_circle")

    print(f"  PCA: PC1={var_exp[0]:.1f}%, PC2={var_exp[1]:.1f}%, total={total:.1f}%")


# =============================================================================
# PLOT 5: Stratified box plots (tertile split on n_SLH)
# =============================================================================
def plot_stratified_boxplots(df: pd.DataFrame) -> None:
    df        = add_tertile(df)
    n_metrics = len(METRICS)

    apply_publication_style()
    fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 3.2, 3.8))
    if n_metrics == 1:
        axes = [axes]

    tertile_counts = df["slh_tertile"].value_counts().sort_index()
    group_labels   = [
        f"T1\nn={tertile_counts.get(0, 0)}",
        f"T2\nn={tertile_counts.get(1, 0)}",
        f"T3\nn={tertile_counts.get(2, 0)}",
    ]

    for idx, metric in enumerate(METRICS):
        ax     = axes[idx]
        groups = [df[df["slh_tertile"] == t][metric].values for t in range(N_TERTILES)]

        bp = ax.boxplot(
            groups,
            patch_artist=True,
            notch=False,
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(lw=1.2),
            capprops=dict(lw=1.2),
            flierprops=dict(marker="o", markersize=3, alpha=0.5),
        )
        for patch, color in zip(bp["boxes"], TERTILE_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(group_labels, fontsize=FONT_SIZE_BASE - 2)
        ax.set_ylabel(METRIC_LABELS[metric].replace("\n", " "), fontsize=FONT_SIZE_BASE - 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        kw_stat, kw_p = stats.kruskal(*groups)
        p_text = f"KW p={kw_p:.3f}" if kw_p >= 0.001 else "KW p<0.001"
        ax.set_title(p_text, fontsize=FONT_SIZE_BASE - 2, color="grey")

    handles = [mpatches.Patch(facecolor=c, alpha=0.7, label=l)
               for c, l in zip(TERTILE_COLORS,
                                ["T1 (low SLH)", "T2 (mid)", "T3 (high SLH)"])]
    fig.legend(handles=handles, loc="lower center", fontsize=FONT_SIZE_BASE - 1,
               frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("CGM metrics by SLH tertile (Kruskal-Wallis p-value)",
                 fontsize=FONT_SIZE_BASE)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig_stratified_boxplots")
    print("  Saved: fig_stratified_boxplots")


# =============================================================================
# PLOT 6: Parallel coordinates
# =============================================================================
def plot_parallel_coordinates(df: pd.DataFrame) -> None:
    X     = df[METRICS].values.copy()
    scaler = StandardScaler()
    X_z   = scaler.fit_transform(X)
    y     = df[RESPONSE].values

    cmap, norm = slh_colormap(y)

    n_axes = len(METRICS)
    x_pos  = np.arange(n_axes)

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    order = np.argsort(y)
    for i in order:
        color = cmap(norm(y[i]))
        ax.plot(x_pos, X_z[i], color=color, alpha=0.35, lw=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([METRIC_LABELS_SHORT[m] for m in METRICS],
                       fontsize=FONT_SIZE_BASE)
    ax.set_ylabel("Z-score", fontsize=FONT_SIZE_BASE)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for xp in x_pos:
        ax.axvline(xp, color="grey", lw=0.5, alpha=0.4)

    sm = cm.ScalarMappable(cmap="plasma", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=RESPONSE_LABEL, shrink=0.85)

    ax.set_title("Parallel coordinates: CGM metrics per patient\n(coloured by n_SLH)",
                 fontsize=FONT_SIZE_BASE)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig_parallel_coordinates")
    print("  Saved: fig_parallel_coordinates")


# =============================================================================
# PLOT 7: SLH count distribution (reference)
# =============================================================================
def plot_slh_distribution(df: pd.DataFrame) -> None:
    y = df[RESPONSE].values

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(5, 3.5))

    max_slh = int(y.max())
    bins    = np.arange(0, max_slh + 2) - 0.5
    ax.hist(y, bins=bins, color=WONG_COLORS["blue"], edgecolor="white", alpha=0.85)
    ax.set_xlabel(RESPONSE_LABEL, fontsize=FONT_SIZE_BASE)
    ax.set_ylabel("Number of patients", fontsize=FONT_SIZE_BASE)
    ax.set_title(
        f"Distribution of SLH count\n"
        f"(n={len(y)}, median={np.median(y):.0f}, "
        f"IQR [{np.percentile(y, 25):.0f}-{np.percentile(y, 75):.0f}])",
        fontsize=FONT_SIZE_BASE,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig_slh_distribution")
    print("  Saved: fig_slh_distribution")


# =============================================================================
# MAIN
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no_filter", action="store_true",
                   help="Skip cohort filter and use all available patients")
    return p.parse_args()


def main() -> None:
    args         = parse_args()
    apply_filter = not args.no_filter

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    print("\n[1/7] Loading data ...")
    df = load_and_merge(apply_filter=apply_filter)
    print(f"\n  n_SLH: mean={df[RESPONSE].mean():.2f}, "
          f"median={df[RESPONSE].median():.0f}, "
          f"max={df[RESPONSE].max():.0f}, "
          f"n={len(df)}")

    print("\n[2/7] Spearman correlation heatmap (all metrics) ...")
    plot_correlation_heatmap(df)

    print("\n[3/7] Mutual information (all metrics) ...")
    plot_mutual_information(df)

    print("\n[4/7] PCA scores ...")
    print("\n[5/7] PCA correlation circle ...")
    plot_pca(df)

    print("\n[6/7] Stratified box plots ...")
    plot_stratified_boxplots(df)

    print("\n[7/7] Parallel coordinates ...")
    plot_parallel_coordinates(df)

    print("\n[+] SLH distribution ...")
    plot_slh_distribution(df)

    print(f"\nDone. All outputs saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
