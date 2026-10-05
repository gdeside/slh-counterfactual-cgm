"""
scripts/slh_patient_characteristics_exploration.py
====================================================
Non-parametric exploration of the link between patient phenotype /
clinical characteristics and SLH episode count (Num_SLH).

No regression models are fitted. The goal is visual and rank-based
exploration using methods suited to the mixed (continuous + categorical)
and non-normal nature of the data.

Analyses
--------
  1. Spearman correlation bar chart  — continuous characteristics vs n_SLH
  2. Mutual information bar chart    — all characteristics (continuous +
                                       dummy-encoded categorical) vs n_SLH
  3. n_SLH distribution by category — box plots for each binary/nominal
                                       variable (Mann-Whitney / Kruskal-Wallis p)
  4. PCA scores                      — patients in PC space on continuous
                                       characteristics, coloured by n_SLH
  5. PCA correlation circle          — variable contributions to each PC
  6. t-SNE embedding                 — non-linear layout, coloured by n_SLH
  7. Stratified box plots            — continuous characteristics split by
                                       SLH tertile (Kruskal-Wallis p)
  8. Parallel coordinates            — z-scored continuous characteristics
                                       per patient, coloured by n_SLH

Inputs
------
  FINAL_COHORT_CSV  (OneDrive / Data / Final cohort SLH.csv)

Outputs (RESULTS_PATH / SLH_Patient_Characteristics_Exploration)
----------------------------------------------------------------
  fig_spearman_bar.{pdf,png,svg}
  fig_mutual_information.{pdf,png,svg}
  fig_slh_by_category_<var>.{pdf,png,svg}
  fig_pca_scores.{pdf,png,svg}
  fig_pca_correlation_circle.{pdf,png,svg}
  fig_tsne.{pdf,png,svg}
  fig_stratified_boxplots.{pdf,png,svg}
  fig_parallel_coordinates.{pdf,png,svg}
  fig_scatter_age_duration_control.{pdf,png,svg}
  fig_scatter_gtaa1c_duration_insulin.{pdf,png,svg}
  spearman_continuous.csv
  mutual_information.csv
  kruskal_wallis_categorical.csv

Outputs (FIG_DIR / Supplementary)
----------------------------------
  SupplementaryFigure_SLH_by_Phenotype.{pdf,svg}
    n_SLH by every categorical phenotype variable, one subplot per variable
    (A, B, C, ...) -- combined version of fig_slh_by_category_<var> above.

Artwork specification
---------------------
No title is drawn inside the artwork; statistics that a reader needs while looking at
the plot stay on the panel.

Usage
-----
    env/bin/python3 -m scripts.statistics.slh_patient_characteristics_exploration [--no_filter]

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
from matplotlib.lines import Line2D
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

from src.config.paths import FINAL_COHORT_CSV, RESULTS_PATH
from src.config.figure_config import FIG_DIR, LANCET_MIN_FONT_PT, lancet_figsize
from src.visualization.colors import WONG_COLORS
from src.visualization.style import (
    apply_lancet_style,
    enforce_lancet_minimums,
    save_figure,
    style_axes,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = RESULTS_PATH / "SLH_Patient_Characteristics_Exploration"

CODE_NO           = 1
CODE_YES          = 2
INSULIN_PUMP_CODE = 4

RESPONSE       = "Num_SLH"
RESPONSE_LABEL = "Number of SLH episodes"

# Continuous phenotype characteristics used for Spearman, MI, stratified boxplots
CONTINUOUS_COLS = [
    "age_study_entry",
    "diabetes_duration",
    "zscore",
    "TIDD",
    "HbA1c",
    "IDAA1c",
    "GTAA1c",
]

CONTINUOUS_LABELS = {
    "age_study_entry":  "Age at study entry (years)",
    "diabetes_duration": "Diabetes duration (years)",
    "zscore":           "BMI z-score",
    "TIDD":             "Total daily insulin dose (U/kg)",
    "HbA1c":            "HbA1c (%)",
    "IDAA1c":           "IDAA1C",
    "GTAA1c":           "GTAA1C",
}

# PCA-only subset: HbA1c, IDAA1C and GTAA1C are highly collinear (rho up to
# 0.93 — see spearman_matrix.csv from plot_correlation_heatmap), and IDAA1C
# is itself HbA1c + 4*TIDD, so it duplicates TIDD's information too. Keep
# only GTAA1C (the strongest single SLH correlate of the three, rho=-0.49)
# as the sole composite-index representative, so PCA loadings reflect
# genuinely distinct dimensions rather than three near-duplicate axes.
# Max remaining pairwise |rho| among these five is 0.48 (duration vs TIDD),
# vs. 0.93 in the full CONTINUOUS_COLS set.
PCA_COLS = [
    "age_study_entry",
    "diabetes_duration",
    "zscore",
    "TIDD",
    "GTAA1c",
]

# Short names for the correlation circle.  The full CONTINUOUS_LABELS run off
# the panel at 7 pt, and their units would mislead there anyway: the PCA is
# fitted on standardised variables, so the circle shows correlations, not
# quantities in years or U/kg.
PCA_CIRCLE_LABELS = {
    "age_study_entry":   "Age at entry",
    "diabetes_duration": "Diabetes duration",
    "zscore":            "BMI z-score",
    "TIDD":              "TIDD",
    "GTAA1c":            "GTAA1C",
}

# Categorical variables — each will get its own n_SLH box plot
CATEGORICAL_COLS = [
    "Sex",
    "comorbidities",
    "Family_history",
    "Physical_activity",
    "DKA_at_diagnosis",
    "Insulin_regimen",
    "DKA_after_diagnosis",
    "SH",
]

CATEGORICAL_LABELS = {
    "Sex":               "Sex",
    "comorbidities":     "Comorbidities",
    "Family_history":    "Family history of diabetes",
    "Physical_activity": "Physical activity",
    "DKA_at_diagnosis":  "DKA at diagnosis",
    "Insulin_regimen":   "Insulin regimen",
    "DKA_after_diagnosis": "DKA after diagnosis",
    "SH":                "Severe hypoglycemia",
}

N_TERTILES     = 3
TERTILE_COLORS = [WONG_COLORS["blue"], WONG_COLORS["orange"], WONG_COLORS["vermillion"]]
TSNE_SEED      = 42

# ── Combined supplementary figure (all categorical variables, one subplot each)
SUPPLEMENTARY_OUTPUT_DIR = FIG_DIR / "Supplementary"
COMBINED_CATEGORY_FIGURE_NAME = "SupplementaryFigure_SLH_by_Phenotype"
PANEL_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# =============================================================================
# ARTWORK (column widths, nothing below 7 pt, no title in the figure)
# =============================================================================
FONT_BODY = LANCET_MIN_FONT_PT          # Tick labels, legends, in-plot values
FONT_TITLE = LANCET_MIN_FONT_PT + 1     # Panel headings
FONT_PANEL_LETTER = LANCET_MIN_FONT_PT + 2   # Panel letters (bold capitals)

#: Per-row height of a multi-panel figure, in inches.
PANEL_ROW_HEIGHT_INCHES = 2.15
#: Extra height reserved for a figure-level legend, in inches.
LEGEND_HEIGHT_INCHES = 0.35


def format_p_value(p: float) -> str:
    """
    Format a p value in house style: two significant figures, or ``p<0.0001``.

    Replaces both the ``*`` / ``**`` / ``***`` significance stars used in this
    script's figures and the ``f"p={p:.3f}"`` formatting, which rendered
    anything below 0.0005 as the impossible "p=0.000".
    """
    if p < 0.0001:
        return "p<0.0001"

    return f"p={float(f'{p:.2g}'):g}"


def save_lancet_figure(fig: plt.Figure, path) -> None:
    """
    Save a figure at exactly its column width.

    ``bbox_inches=None`` keeps the page width the figure was drawn at — the
    default "tight" crop would change it — so the layout must already leave
    room for everything it draws.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str or Path
        Destination without extension.
    """
    fixed = enforce_lancet_minimums(fig)
    if any(fixed.values()):
        print(f"    Minimums enforced on {fixed['text']} text object(s) "
              f"and {fixed['lines']} line(s)")

    save_figure(fig, path, bbox_inches=None)


# =============================================================================
# DATA LOADING
# =============================================================================
def parse_eu_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def load_data() -> pd.DataFrame:
    df = pd.read_csv(FINAL_COHORT_CSV, sep=None, engine="python")

    # Continuous
    for col in ["age_study_entry", "diabetes_duration", "zscore",
                "HbA1c", "IDAA1c", "GTAA1c"]:
        df[col] = parse_eu_float(df[col])
    df["TIDD"]    = parse_eu_float(df["TIDD "])
    df["Num_SLH"] = pd.to_numeric(df["Num_SLH"], errors="coerce")

    # Categorical
    df["Sex"] = df["Sex"].str.strip().map({"M": "Male", "F": "Female"})

    for col in ["comorbidities", "Family_history", "Physical_activity",
                "DKA_at_diagnosis", "DKA_after_diagnosis"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").map(
            {CODE_NO: "No", CODE_YES: "Yes"}
        )

    df["SH"] = pd.to_numeric(df["SH"], errors="coerce").map(
        {CODE_NO: "No", CODE_YES: "Yes"}
    )

    def recode_regimen(v):
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

    df["Insulin_regimen"] = df["Insulin_regimen "].apply(recode_regimen)

    needed = CONTINUOUS_COLS + [RESPONSE]
    df = df.dropna(subset=needed).reset_index(drop=True)
    print(f"  Patients with complete continuous data: {len(df)}")
    return df


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


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """
    Partial Spearman correlation between x and y, controlling for z.

    Implemented as the Pearson correlation between the residuals of x and y
    after each is linearly regressed on rank(z) (rank-transform equivalent
    of a partial correlation). Returns (rho, p).
    """
    rx, ry, rz = (scipy_stats.rankdata(v) for v in (x, y, z))
    rx_resid = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ry_resid = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return scipy_stats.pearsonr(rx_resid, ry_resid)


# =============================================================================
# PLOT 0: Spearman correlation heatmap (all continuous phenotype vars + n_SLH)
# =============================================================================
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    cols   = CONTINUOUS_COLS + [RESPONSE]
    labels = [CONTINUOUS_LABELS[c] for c in CONTINUOUS_COLS] + [RESPONSE_LABEL]

    sub = df[cols].dropna()
    n = len(cols)
    rho_mat = np.zeros((n, n))
    p_mat   = np.zeros((n, n))
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            r, p = scipy_stats.spearmanr(sub[ci], sub[cj])
            rho_mat[i, j] = r
            p_mat[i, j]   = p

    rho_df = pd.DataFrame(rho_mat, index=labels, columns=labels)
    rho_df.to_csv(OUTPUT_DIR / "spearman_matrix.csv")

    apply_lancet_style()
    fig, ax = plt.subplots(
        figsize=lancet_figsize(5.6, columns="full"), constrained_layout=True,
    )

    cmap = matplotlib.colormaps["RdBu_r"]
    im = ax.imshow(rho_mat, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Spearman ρ", fontsize=FONT_BODY)
    cbar.ax.tick_params(labelsize=FONT_BODY)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=FONT_BODY)
    ax.set_yticklabels(labels, fontsize=FONT_BODY)

    for i in range(n):
        for j in range(n):
            sig   = "*" if p_mat[i, j] < 0.05 else ""
            txt   = f"{rho_mat[i, j]:.2f}{sig}"
            color = "white" if abs(rho_mat[i, j]) > 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=FONT_BODY, color=color)

    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_visible(False)
    # Dashed separator before n_SLH
    ax.axhline(n - 1.5, color="black", lw=1.5, ls="--")
    ax.axvline(n - 1.5, color="black", lw=1.5, ls="--")

    save_lancet_figure(fig, OUTPUT_DIR / "fig_correlation_heatmap")
    print("  Saved: fig_correlation_heatmap")


# =============================================================================
# PLOT 1: Spearman correlation bar chart (continuous vs n_SLH)
# =============================================================================
def plot_spearman_bar(df: pd.DataFrame) -> None:
    rows = []
    for col in CONTINUOUS_COLS:
        vals = df[[col, RESPONSE]].dropna()
        r, p = scipy_stats.spearmanr(vals[col], vals[RESPONSE])
        rows.append({"col": col, "label": CONTINUOUS_LABELS[col],
                     "rho": r, "p": p, "n": len(vals)})
    result = pd.DataFrame(rows).sort_values("rho", key=abs, ascending=True)
    result.to_csv(OUTPUT_DIR / "spearman_continuous.csv", index=False)

    apply_lancet_style()
    fig, ax = plt.subplots(
        figsize=lancet_figsize(3.4, columns="full"), constrained_layout=True,
    )

    bar_colors = [WONG_COLORS["vermillion"] if r < 0 else WONG_COLORS["blue"]
                  for r in result["rho"]]
    bars = ax.barh(result["label"], result["rho"],
                   color=bar_colors, edgecolor="white", alpha=0.85)

    # Reported p values rather than significance stars, which say less than
    # the value and which is not used in their place.
    for bar, row in zip(bars, result.itertuples()):
        xpos = row.rho + (0.01 if row.rho >= 0 else -0.01)
        ha   = "left" if row.rho >= 0 else "right"
        label = f"{row.rho:.2f} ({format_p_value(row.p)}, n={row.n})"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha=ha, fontsize=FONT_BODY)

    # Room for those labels beyond the end of each bar, on whichever side the
    # bars actually extend — a symmetric range wastes half the panel when
    # every correlation has the same sign.
    lo, hi = float(result["rho"].min()), float(result["rho"].max())
    pad = float(result["rho"].abs().max()) * 0.45
    ax.set_xlim(min(0.0, lo) - (pad if lo < 0 else 0.03),
                max(0.0, hi) + (pad if hi > 0 else 0.03))

    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"Spearman ρ with the {RESPONSE_LABEL.lower().replace('slh', 'SLH')}",
                  fontsize=FONT_BODY)
    ax.tick_params(labelsize=FONT_BODY)
    style_axes(ax)

    # Only the directions actually present: a cohort where every association
    # is negative should not carry a "positive" key.
    handles = [
        mpatches.Patch(color=color, label=label)
        for color, label, present in [
            (WONG_COLORS["blue"], "Positive association", (result["rho"] >= 0).any()),
            (WONG_COLORS["vermillion"], "Negative association", (result["rho"] < 0).any()),
        ]
        if present
    ]
    # Lower left: with every bar running left from zero, the right-hand side
    # of the bottom row is occupied.
    ax.legend(handles=handles, fontsize=FONT_BODY, frameon=False, loc="lower left")

    save_lancet_figure(fig, OUTPUT_DIR / "fig_spearman_bar")
    print("  Saved: fig_spearman_bar")


# =============================================================================
# PLOT 2: Mutual information bar chart (continuous + dummy categorical)
# =============================================================================
def plot_mutual_information(df: pd.DataFrame) -> None:
    # Dummy-encode categorical variables (ignoring NaN rows per variable)
    mi_rows = []

    # Continuous
    for col in CONTINUOUS_COLS:
        sub = df[[col, RESPONSE]].dropna()
        mi  = mutual_info_regression(sub[[col]], sub[RESPONSE], random_state=42)[0]
        mi_rows.append({"label": CONTINUOUS_LABELS[col], "MI": mi,
                        "type": "Continuous"})

    # Categorical — one-hot then sum MI across dummies (proxy)
    for col in CATEGORICAL_COLS:
        sub = df[[col, RESPONSE]].dropna()
        if len(sub) < 10:
            continue
        dummies = pd.get_dummies(sub[col], drop_first=False).astype(float)
        if dummies.shape[1] == 0:
            continue
        mi_vals = mutual_info_regression(dummies, sub[RESPONSE], random_state=42)
        mi_rows.append({"label": CATEGORICAL_LABELS[col],
                        "MI": float(mi_vals.sum()), "type": "Categorical"})

    mi_df = pd.DataFrame(mi_rows).sort_values("MI", ascending=True)
    mi_df.to_csv(OUTPUT_DIR / "mutual_information.csv", index=False)

    type_colors = {"Continuous": WONG_COLORS["blue"],
                   "Categorical": WONG_COLORS["orange"]}
    bar_colors = [type_colors[t] for t in mi_df["type"]]

    apply_lancet_style()
    fig, ax = plt.subplots(
        figsize=lancet_figsize(4.2, columns="full"), constrained_layout=True,
    )
    bars = ax.barh(mi_df["label"], mi_df["MI"],
                   color=bar_colors, edgecolor="white", alpha=0.85)

    for bar, val in zip(bars, mi_df["MI"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=FONT_BODY)

    # Room for the value labels, which otherwise run off the right-hand edge.
    ax.set_xlim(0, float(mi_df["MI"].max()) * 1.18)

    handles = [mpatches.Patch(color=c, label=t, alpha=0.85)
               for t, c in type_colors.items()]
    ax.legend(handles=handles, fontsize=FONT_BODY, frameon=False,
             loc="lower right")
    ax.set_xlabel(
        f"Mutual information with the {RESPONSE_LABEL.lower().replace('slh', 'SLH')}",
        fontsize=FONT_BODY,
    )
    ax.tick_params(labelsize=FONT_BODY)
    style_axes(ax)

    save_lancet_figure(fig, OUTPUT_DIR / "fig_mutual_information")
    print("  Saved: fig_mutual_information")


# =============================================================================
# PLOT 3: n_SLH distribution by categorical variable
# =============================================================================
def plot_slh_by_category(df: pd.DataFrame) -> None:
    kw_rows = []

    for col in CATEGORICAL_COLS:
        sub = df[[col, RESPONSE]].dropna()
        levels = sorted(sub[col].dropna().unique())
        if len(levels) < 2:
            continue

        groups = [sub[sub[col] == lv][RESPONSE].values for lv in levels]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            continue

        if len(groups) == 2:
            stat, p = scipy_stats.mannwhitneyu(*groups, alternative="two-sided")
            test_name = "Mann-Whitney"
        else:
            stat, p = scipy_stats.kruskal(*groups)
            test_name = "Kruskal-Wallis"

        kw_rows.append({"Variable": CATEGORICAL_LABELS[col],
                        "Test": test_name, "stat": stat, "p": p,
                        "n_levels": len(groups)})

        apply_lancet_style()
        fig, ax = plt.subplots(
            figsize=lancet_figsize(3.2, columns="double"),
            constrained_layout=True,
        )

        level_colors = [TERTILE_COLORS[i % len(TERTILE_COLORS)]
                        for i in range(len(levels))]

        bp = ax.boxplot(
            groups,
            patch_artist=True,
            notch=False,
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(lw=1.2),
            capprops=dict(lw=1.2),
            flierprops=dict(marker="o", markersize=3, alpha=0.5),
        )
        for patch, color in zip(bp["boxes"], level_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # Jitter overlay
        for i, (grp, color) in enumerate(zip(groups, level_colors), start=1):
            jitter = np.random.default_rng(42).uniform(-0.18, 0.18, size=len(grp))
            ax.scatter(np.full(len(grp), i) + jitter, grp,
                       color=color, s=12, alpha=0.5, zorder=3)

        ns = [len(g) for g in groups]
        ax.set_xticks(range(1, len(levels) + 1))
        ax.set_xticklabels(
            [f"{lv}\n(n={n})" for lv, n in zip(levels, ns)],
            fontsize=FONT_BODY,
        )
        ax.set_xlabel(CATEGORICAL_LABELS[col], fontsize=FONT_BODY, labelpad=3)
        ax.set_ylabel(RESPONSE_LABEL, fontsize=FONT_BODY)
        ax.tick_params(labelsize=FONT_BODY)

        # The test result stays on the panel.
        ax.set_title(f"{test_name} {format_p_value(p)}",
                     fontsize=FONT_BODY, loc="left", pad=4)
        style_axes(ax)

        fname = f"fig_slh_by_{col.lower()}"
        save_lancet_figure(fig, OUTPUT_DIR / fname)
        print(f"  Saved: {fname}")

    kw_df = pd.DataFrame(kw_rows)
    kw_df.to_csv(OUTPUT_DIR / "kruskal_wallis_categorical.csv", index=False)
    print(f"  Saved: kruskal_wallis_categorical.csv")


# =============================================================================
# SUPPLEMENTARY FIGURE: n_SLH BY CATEGORY, ALL VARIABLES AS SUBPLOTS (A, B, ...)
# =============================================================================
def plot_slh_by_category_panel(ax, groups: list, levels: list, test_name: str,
                               p: float, var_label: str) -> None:
    """Draw one categorical variable's n_SLH boxplot (+ jitter) on a single Axes.
    Same statistics/styling as the individual-file version in
    plot_slh_by_category, just targeting a shared subplot instead of its own
    figure."""
    level_colors = [TERTILE_COLORS[i % len(TERTILE_COLORS)] for i in range(len(levels))]

    bp = ax.boxplot(
        groups,
        patch_artist=True,
        notch=False,
        medianprops=dict(color="black", lw=2),
        whiskerprops=dict(lw=1.2),
        capprops=dict(lw=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], level_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    for i, (grp, color) in enumerate(zip(groups, level_colors), start=1):
        jitter = np.random.default_rng(42).uniform(-0.18, 0.18, size=len(grp))
        ax.scatter(np.full(len(grp), i) + jitter, grp, color=color, s=10,
                  alpha=0.5, zorder=3)

    ns = [len(g) for g in groups]
    ax.set_xticks(range(1, len(levels) + 1))
    ax.set_xticklabels([f"{lv}\n(n={n})" for lv, n in zip(levels, ns)],
                       fontsize=FONT_BODY)
    ax.set_ylabel(RESPONSE_LABEL, fontsize=FONT_BODY, labelpad=2)

    ax.set_title(f"{var_label} — {test_name} {format_p_value(p)}",
                 fontsize=FONT_BODY, loc="left", pad=4)
    style_axes(ax)
    ax.tick_params(labelsize=FONT_BODY)


def plot_slh_by_category_combined(df: pd.DataFrame) -> None:
    """
    Supplementary figure: n_SLH distribution for every categorical phenotype
    variable in CATEGORICAL_COLS, each as its own subplot, labelled A, B,
    C, ... Complements (does not replace) the per-variable files saved by
    plot_slh_by_category.
    """
    apply_lancet_style()

    panels = []  # (var_label, groups, levels, test_name, p)
    for col in CATEGORICAL_COLS:
        sub = df[[col, RESPONSE]].dropna()
        levels = sorted(sub[col].dropna().unique())
        if len(levels) < 2:
            continue

        groups_levels = [(sub[sub[col] == lv][RESPONSE].values, lv) for lv in levels]
        groups_levels = [(g, lv) for g, lv in groups_levels if len(g) >= 2]
        if len(groups_levels) < 2:
            continue
        groups = [g for g, _ in groups_levels]
        levels = [lv for _, lv in groups_levels]

        if len(groups) == 2:
            stat, p = scipy_stats.mannwhitneyu(*groups, alternative="two-sided")
            test_name = "Mann-Whitney"
        else:
            stat, p = scipy_stats.kruskal(*groups)
            test_name = "Kruskal-Wallis"

        panels.append((CATEGORICAL_LABELS[col], groups, levels, test_name, p))

    n_panels = len(panels)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))

    # Title stays out of the artwork; the panels plus their headings fit one
    # page at this per-row height.
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=lancet_figsize(PANEL_ROW_HEIGHT_INCHES * n_rows, columns="full"),
        constrained_layout=True,
    )
    axes_flat = np.array(axes).reshape(-1)

    for i, (var_label, groups, levels, test_name, p) in enumerate(panels):
        ax = axes_flat[i]
        plot_slh_by_category_panel(ax, groups, levels, test_name, p, var_label)
        ax.text(-0.14, 1.06, PANEL_LETTERS[i], transform=ax.transAxes,
               fontsize=FONT_PANEL_LETTER, fontweight="bold",
               va="bottom", ha="right")

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    n_patients = int(df[RESPONSE].notna().sum())

    panel_list = "; ".join(
        f"({PANEL_LETTERS[i]}) {var_label.lower()}, {test_name} "
        f"{format_p_value(p)}"
        for i, (var_label, _, _, test_name, p) in enumerate(panels)
    )

    SUPPLEMENTARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUPPLEMENTARY_OUTPUT_DIR / COMBINED_CATEGORY_FIGURE_NAME
    save_lancet_figure(fig, out_path)
    print(f"  Saved: {out_path}")


# =============================================================================
# PLOT 4 & 5: PCA scores + correlation circle
# =============================================================================
def plot_pca(df: pd.DataFrame) -> None:
    sub = df[PCA_COLS + [RESPONSE]].dropna()
    X   = sub[PCA_COLS].values
    y   = sub[RESPONSE].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100

    # Correlation loadings: corr(var, PC) = eigenvector * sqrt(eigenvalue)
    loadings_corr = pca.components_.T * np.sqrt(pca.explained_variance_)

    # ── Scores ────────────────────────────────────────────────────────────────
    apply_lancet_style()
    fig_sc, ax = plt.subplots(
        figsize=lancet_figsize(3.4, columns="double"), constrained_layout=True,
    )
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=y, cmap="plasma",
                    s=18, alpha=0.75, edgecolors="none")
    cbar = fig_sc.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(RESPONSE_LABEL, fontsize=FONT_BODY)
    cbar.ax.tick_params(labelsize=FONT_BODY)
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% of variance)", fontsize=FONT_BODY)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% of variance)", fontsize=FONT_BODY)
    ax.tick_params(labelsize=FONT_BODY)
    style_axes(ax)

    variables = ", ".join(CONTINUOUS_LABELS[c].lower() for c in PCA_COLS)
    save_lancet_figure(fig_sc, OUTPUT_DIR / "fig_pca_scores")
    print("  Saved: fig_pca_scores")

    # ── Correlation circle ────────────────────────────────────────────────────
    arrow_colors = [
        WONG_COLORS["blue"], WONG_COLORS["orange"], WONG_COLORS["vermillion"],
        WONG_COLORS["bluish_green"], WONG_COLORS["reddish_purple"],
        WONG_COLORS["sky_blue"], WONG_COLORS["yellow"],
    ]

    apply_lancet_style()
    fig_cc, ax2 = plt.subplots(
        figsize=lancet_figsize(4.4, columns="double"), constrained_layout=True,
    )

    theta = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(theta), np.sin(theta), color="grey", lw=1, ls="--", alpha=0.5)
    ax2.axhline(0, color="grey", lw=0.5, ls="--", alpha=0.5)
    ax2.axvline(0, color="grey", lw=0.5, ls="--", alpha=0.5)

    for i, col in enumerate(PCA_COLS):
        lx, ly = loadings_corr[i, 0], loadings_corr[i, 1]
        color  = arrow_colors[i % len(arrow_colors)]
        ax2.annotate(
            "", xy=(lx, ly), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=12),
        )
        scale = 1.10
        ha = "left" if lx >= 0 else "right"
        va = "bottom" if ly >= 0 else "top"
        ax2.text(lx * scale, ly * scale, PCA_CIRCLE_LABELS[col],
                 ha=ha, va=va, fontsize=FONT_BODY,
                 color=color, fontweight="bold")

    # Margin around the unit circle for the labels set beyond the arrow tips.
    ax2.set_xlim(-1.55, 1.55)
    ax2.set_ylim(-1.55, 1.55)
    ax2.set_aspect("equal")
    ax2.set_xlabel(f"PC1 ({var_exp[0]:.1f}% of variance)", fontsize=FONT_BODY)
    ax2.set_ylabel(f"PC2 ({var_exp[1]:.1f}% of variance)", fontsize=FONT_BODY)
    ax2.tick_params(labelsize=FONT_BODY)
    style_axes(ax2)

    save_lancet_figure(fig_cc, OUTPUT_DIR / "fig_pca_correlation_circle")
    print("  Saved: fig_pca_correlation_circle")
    print(f"  PCA: PC1={var_exp[0]:.1f}%, PC2={var_exp[1]:.1f}%, "
          f"total={var_exp.sum():.1f}%")


# =============================================================================
# PLOT 6: t-SNE
# =============================================================================
def plot_tsne(df: pd.DataFrame, perplexity: float = 15.0) -> None:
    sub = df[CONTINUOUS_COLS + [RESPONSE]].dropna()
    X   = sub[CONTINUOUS_COLS].values
    y   = sub[RESPONSE].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    perp   = min(perplexity, len(X_scaled) - 1)
    tsne   = TSNE(n_components=2, perplexity=perp, random_state=TSNE_SEED,
                  max_iter=1000, init="pca")
    coords = tsne.fit_transform(X_scaled)

    apply_lancet_style()
    fig, ax = plt.subplots(
        figsize=lancet_figsize(3.4, columns="double"), constrained_layout=True,
    )
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=y, cmap="plasma",
                    s=18, alpha=0.75, edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(RESPONSE_LABEL, fontsize=FONT_BODY)
    cbar.ax.tick_params(labelsize=FONT_BODY)
    ax.set_xlabel("t-SNE dimension 1", fontsize=FONT_BODY)
    ax.set_ylabel("t-SNE dimension 2", fontsize=FONT_BODY)
    ax.tick_params(labelsize=FONT_BODY)
    style_axes(ax)

    save_lancet_figure(fig, OUTPUT_DIR / "fig_tsne")
    print("  Saved: fig_tsne")


# =============================================================================
# PLOT 7: Stratified box plots (continuous vars by SLH tertile)
# =============================================================================
def plot_stratified_boxplots(df: pd.DataFrame) -> None:
    df_t = add_tertile(df)

    tertile_counts = df_t["slh_tertile"].value_counts().sort_index()
    group_labels   = [
        f"T1 (low)\nn={tertile_counts.get(0, 0)}",
        f"T2 (mid)\nn={tertile_counts.get(1, 0)}",
        f"T3 (high)\nn={tertile_counts.get(2, 0)}",
    ]

    n_cols = len(CONTINUOUS_COLS)
    ncols  = 3
    nrows  = int(np.ceil(n_cols / ncols))

    # Three columns rather than four: at 175 mm, four leave each panel under
    # 40 mm wide for three boxes and a two-line tick label.
    apply_lancet_style()
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=lancet_figsize(
            PANEL_ROW_HEIGHT_INCHES * nrows + LEGEND_HEIGHT_INCHES,
            columns="full",
        ),
        constrained_layout=True,
    )
    axes = np.array(axes).flatten()

    p_texts = []

    for idx, col in enumerate(CONTINUOUS_COLS):
        ax     = axes[idx]
        groups = [df_t[df_t["slh_tertile"] == t][col].dropna().values
                  for t in range(N_TERTILES)]

        bp = ax.boxplot(
            groups, patch_artist=True, notch=False,
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
            flierprops=dict(marker="o", markersize=3, alpha=0.5),
        )
        for patch, color in zip(bp["boxes"], TERTILE_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(group_labels, fontsize=FONT_BODY)
        ax.set_ylabel(CONTINUOUS_LABELS[col], fontsize=FONT_BODY)
        ax.tick_params(labelsize=FONT_BODY)
        style_axes(ax)

        valid = [g for g in groups if len(g) >= 2]
        if len(valid) >= 2:
            _, kw_p = scipy_stats.kruskal(*valid)
            p_texts.append((CONTINUOUS_LABELS[col], kw_p))
            p_text = f"{PANEL_LETTERS[idx]}   Kruskal–Wallis {format_p_value(kw_p)}"
        else:
            p_text = PANEL_LETTERS[idx]
        ax.set_title(p_text, fontsize=FONT_BODY, fontweight="bold",
                     loc="left", pad=4)

    for ax in axes[n_cols:]:
        ax.set_visible(False)

    handles = [mpatches.Patch(facecolor=c, alpha=0.7, label=l)
               for c, l in zip(TERTILE_COLORS,
                                ["T1 (low SLH)", "T2 (mid)", "T3 (high SLH)"])]

    # "outside lower center" makes constrained_layout reserve room for the
    # legend inside the canvas, so the figure saves uncropped at exactly the
    # page width.
    fig.legend(handles=handles, loc="outside lower center", fontsize=FONT_BODY,
               frameon=False, ncol=3)

    panel_list = "; ".join(
        f"({PANEL_LETTERS[i]}) {label.lower()}, Kruskal–Wallis "
        f"{format_p_value(kw_p)}"
        for i, (label, kw_p) in enumerate(p_texts)
    )

    save_lancet_figure(fig, OUTPUT_DIR / "fig_stratified_boxplots")
    print("  Saved: fig_stratified_boxplots")


# =============================================================================
# PLOT 8: Parallel coordinates (continuous characteristics)
# =============================================================================
def plot_parallel_coordinates(df: pd.DataFrame) -> None:
    sub    = df[CONTINUOUS_COLS + [RESPONSE]].dropna()
    X      = sub[CONTINUOUS_COLS].values
    y      = sub[RESPONSE].values
    scaler = StandardScaler()
    X_z    = scaler.fit_transform(X)

    cmap, norm = slh_colormap(y)
    x_pos      = np.arange(len(CONTINUOUS_COLS))

    apply_lancet_style()
    fig, ax = plt.subplots(
        figsize=lancet_figsize(3.6, columns="full"), constrained_layout=True,
    )

    order = np.argsort(y)
    for i in order:
        ax.plot(x_pos, X_z[i], color=cmap(norm(y[i])), alpha=0.35, lw=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([CONTINUOUS_LABELS[c] for c in CONTINUOUS_COLS],
                       rotation=30, ha="right", fontsize=FONT_BODY)
    ax.set_ylabel("Standardised value (z-score)", fontsize=FONT_BODY)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
    for xp in x_pos:
        ax.axvline(xp, color="grey", lw=0.5, alpha=0.4)
    ax.tick_params(labelsize=FONT_BODY)
    style_axes(ax)

    sm = cm.ScalarMappable(cmap="plasma", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(RESPONSE_LABEL, fontsize=FONT_BODY)
    cbar.ax.tick_params(labelsize=FONT_BODY)

    save_lancet_figure(fig, OUTPUT_DIR / "fig_parallel_coordinates")
    print("  Saved: fig_parallel_coordinates")


# =============================================================================
# PLOT 9: Age at entry vs diabetes duration - scatter plots coloured by n_SLH
# =============================================================================
def _scatter_slh(ax, x: np.ndarray, y: np.ndarray, c: np.ndarray,
                  xlabel: str, ylabel: str, letter: str = "") -> tuple:
    """Draw one scatter panel; returns the mappable and its Spearman result."""
    r, p = scipy_stats.spearmanr(x, y)
    sc = ax.scatter(x, y, c=c, cmap="plasma", s=18, alpha=0.8,
                    edgecolors="white", linewidths=0.4)
    b, a = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, a + b * x_line, color="black", lw=1.2, ls="--", alpha=0.6)
    ax.set_xlabel(xlabel, fontsize=FONT_BODY)
    ax.set_ylabel(ylabel, fontsize=FONT_BODY)
    ax.tick_params(labelsize=FONT_BODY)
    heading = f"{letter}   " if letter else ""
    ax.set_title(f"{heading}Spearman ρ={r:.2f}, {format_p_value(p)}",
                 fontsize=FONT_BODY, fontweight="bold", loc="left", pad=4)
    style_axes(ax)

    return sc, r, p


def plot_age_duration_control_scatter(df: pd.DataFrame) -> None:
    cols = ["age_study_entry", "diabetes_duration", "GTAA1c", RESPONSE]
    sub = df[cols].dropna()

    apply_lancet_style()
    fig, axes = plt.subplots(
        1, 3, figsize=lancet_figsize(2.6, columns="full"),
        constrained_layout=True,
    )

    sc, r_ad, p_ad = _scatter_slh(
        axes[0], sub["age_study_entry"].values, sub["diabetes_duration"].values,
        sub[RESPONSE].values,
        CONTINUOUS_LABELS["age_study_entry"], CONTINUOUS_LABELS["diabetes_duration"],
        letter="A",
    )
    _, r_ag, p_ag = _scatter_slh(
        axes[1], sub["age_study_entry"].values, sub["GTAA1c"].values,
        sub[RESPONSE].values,
        CONTINUOUS_LABELS["age_study_entry"], CONTINUOUS_LABELS["GTAA1c"],
        letter="B",
    )
    _, r_dg, p_dg = _scatter_slh(
        axes[2], sub["diabetes_duration"].values, sub["GTAA1c"].values,
        sub[RESPONSE].values,
        CONTINUOUS_LABELS["diabetes_duration"], CONTINUOUS_LABELS["GTAA1c"],
        letter="C",
    )

    cbar = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(RESPONSE_LABEL, fontsize=FONT_BODY)
    cbar.ax.tick_params(labelsize=FONT_BODY)

    save_lancet_figure(fig, OUTPUT_DIR / "fig_scatter_age_duration_control")
    print("  Saved: fig_scatter_age_duration_control")

    # ── Partial correlations: disentangle age, duration, GTAA1C vs n_SLH ─────
    age, dur, gtaa, slh = (sub[c].values for c in
                            ["age_study_entry", "diabetes_duration", "GTAA1c", RESPONSE])

    rows = []
    r, p = scipy_stats.spearmanr(age, dur)
    rows.append({"pair": "Age entry vs Diabetes duration", "type": "zero-order",
                 "control_for": "-", "rho": r, "p": p, "n": len(sub)})

    r, p = scipy_stats.spearmanr(age, slh)
    rows.append({"pair": "Age entry vs n_SLH", "type": "zero-order",
                 "control_for": "-", "rho": r, "p": p, "n": len(sub)})
    r, p = partial_spearman(age, slh, dur)
    rows.append({"pair": "Age entry vs n_SLH", "type": "partial",
                 "control_for": "Diabetes duration", "rho": r, "p": p, "n": len(sub)})

    r, p = scipy_stats.spearmanr(dur, slh)
    rows.append({"pair": "Diabetes duration vs n_SLH", "type": "zero-order",
                 "control_for": "-", "rho": r, "p": p, "n": len(sub)})
    r, p = partial_spearman(dur, slh, age)
    rows.append({"pair": "Diabetes duration vs n_SLH", "type": "partial",
                 "control_for": "Age entry", "rho": r, "p": p, "n": len(sub)})

    r, p = scipy_stats.spearmanr(gtaa, slh)
    rows.append({"pair": "GTAA1C vs n_SLH", "type": "zero-order",
                 "control_for": "-", "rho": r, "p": p, "n": len(sub)})
    r, p = partial_spearman(gtaa, slh, age)
    rows.append({"pair": "GTAA1C vs n_SLH", "type": "partial",
                 "control_for": "Age entry", "rho": r, "p": p, "n": len(sub)})
    r, p = partial_spearman(gtaa, slh, dur)
    rows.append({"pair": "GTAA1C vs n_SLH", "type": "partial",
                 "control_for": "Diabetes duration", "rho": r, "p": p, "n": len(sub)})

    partial_df = pd.DataFrame(rows)
    partial_df.to_csv(OUTPUT_DIR / "partial_correlations_age_duration_gtaa1c.csv",
                       index=False)
    print("  Saved: partial_correlations_age_duration_gtaa1c.csv")
    print(partial_df.to_string(index=False))


# =============================================================================
# PLOT 10: GTAA1C vs diabetes duration - coloured by n_SLH, marker = insulin regimen
# =============================================================================
INSULIN_REGIMEN_MARKERS = {
    "Basal-prandial": "o",
    "Two injections": "^",
    "Insulin pump":   "s",
}


def plot_gtaa1c_duration_by_insulin(df: pd.DataFrame) -> None:
    """Same GTAA1C-vs-duration relationship as plot_age_duration_control_scatter's
    third panel, but with marker shape encoding insulin regimen — checks whether
    the honeymoon-phase (low duration / low GTAA1C / high n_SLH) signal holds
    within each treatment modality, not just in aggregate."""
    cols = ["diabetes_duration", "GTAA1c", RESPONSE, "Insulin_regimen"]
    sub = df[cols].dropna()

    x = sub["diabetes_duration"].values
    y = sub["GTAA1c"].values
    c = sub[RESPONSE].values
    r, p = scipy_stats.spearmanr(x, y)

    apply_lancet_style()
    fig, ax = plt.subplots(
        figsize=lancet_figsize(3.4, columns="double"), constrained_layout=True,
    )

    cmap, norm = slh_colormap(c)
    sc = None
    for regimen, marker in INSULIN_REGIMEN_MARKERS.items():
        mask = (sub["Insulin_regimen"] == regimen).values
        if not mask.any():
            continue
        sc = ax.scatter(x[mask], y[mask], c=c[mask], cmap=cmap, norm=norm,
                        marker=marker, s=24, alpha=0.85,
                        edgecolors="white", linewidths=0.5, zorder=3)

    b, a = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, a + b * x_line, color="black", lw=1.2, ls="--", alpha=0.6, zorder=1)

    ax.set_xlabel(CONTINUOUS_LABELS["diabetes_duration"], fontsize=FONT_BODY)
    ax.set_ylabel(CONTINUOUS_LABELS["GTAA1c"], fontsize=FONT_BODY)
    ax.set_title(f"Spearman ρ={r:.2f}, {format_p_value(p)}",
                 fontsize=FONT_BODY, loc="left", pad=4)
    ax.tick_params(labelsize=FONT_BODY)
    style_axes(ax)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(RESPONSE_LABEL, fontsize=FONT_BODY)
    cbar.ax.tick_params(labelsize=FONT_BODY)

    counts = {
        regimen: int((sub["Insulin_regimen"] == regimen).sum())
        for regimen in INSULIN_REGIMEN_MARKERS
    }
    legend_handles = [
        Line2D([0], [0], marker=marker, color="w", markerfacecolor="#999999",
               markeredgecolor="black", markersize=5,
               label=f"{regimen} (n={counts[regimen]})")
        for regimen, marker in INSULIN_REGIMEN_MARKERS.items()
        if counts[regimen]
    ]
    ax.legend(handles=legend_handles, title="Insulin regimen",
             fontsize=FONT_BODY, title_fontsize=FONT_BODY,
             loc="best", frameon=True, framealpha=1.0, edgecolor="#cccccc",
             borderpad=0.35, handletextpad=0.5)

    save_lancet_figure(fig, OUTPUT_DIR / "fig_scatter_gtaa1c_duration_insulin")
    print("  Saved: fig_scatter_gtaa1c_duration_insulin")


# =============================================================================
# MAIN
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tsne_perplexity", type=float, default=15.0,
                   help="t-SNE perplexity (default: 15)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    print("\n[1/9] Loading data ...")
    df = load_data()
    print(f"  n_SLH: mean={df[RESPONSE].mean():.1f}, "
          f"median={df[RESPONSE].median():.0f}, "
          f"max={df[RESPONSE].max():.0f}")

    print("\n[+] Correlation heatmap ...")
    plot_correlation_heatmap(df)

    print("\n[2/9] Spearman correlation bar chart ...")
    plot_spearman_bar(df)

    print("\n[3/9] Mutual information ...")
    plot_mutual_information(df)

    print("\n[4/9] n_SLH by categorical variable ...")
    plot_slh_by_category(df)

    print("\n[+] n_SLH by categorical variable -- combined supplementary figure ...")
    plot_slh_by_category_combined(df)

    print("\n[5/9] PCA scores ...")
    print("\n[6/9] PCA correlation circle ...")
    plot_pca(df)

    print("\n[7/9] t-SNE ...")
    plot_tsne(df, perplexity=args.tsne_perplexity)

    print("\n[8/9] Stratified box plots ...")
    plot_stratified_boxplots(df)

    print("\n[9/9] Parallel coordinates ...")
    plot_parallel_coordinates(df)

    print("\n[+] Age / duration / GTAA1C scatter + partial correlations ...")
    plot_age_duration_control_scatter(df)

    print("\n[+] GTAA1C vs diabetes duration by insulin regimen ...")
    plot_gtaa1c_duration_by_insulin(df)

    print(f"\nDone. All outputs saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
