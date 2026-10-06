"""
scripts/paper_figures/generate_supplementary_figure_phenotype_determinants.py
=============================================================================
Supplementary figure: patient-phenotype determinants of SLH frequency.

Panels
------
  A - Mutual information of every phenotype variable (continuous + one-hot
      encoded categorical) with n_SLH. Same computation as
      plot_mutual_information in
      scripts/statistics/slh_patient_characteristics_exploration.py.
  B - GTAA1C vs diabetes duration scatter, coloured by n_SLH, marker shape =
      insulin regimen (delivery type: basal-prandial / two injections /
      insulin pump). Same construction as plot_gtaa1c_duration_by_insulin
      in scripts/statistics/slh_patient_characteristics_exploration.py.
  C - Combined CGM + phenotype PCA -- scores.
  D - Combined CGM + phenotype PCA -- correlation circle.

Panels C/D reuse compute_combined_pca and the PCA_COLS/colour definitions
from scripts/paper_figures/generate_figure_6_slh_determinants.py verbatim
("the PCA as it was done before"). Only the plotting wrappers are
re-declared here, because the originals hardcode "D"/"E" panel labels that
were correct for that file's own 7-panel layout but not for this one.

Usage
-----
    env/bin/python3 -m scripts.paper_figures.generate_supplementary_figure_phenotype_determinants

Artwork specification
---------------------
No title is drawn inside the artwork.

Outputs (FIG_DIR / Supplementary)
----------------------------------
  SupplementaryFigure_Phenotype_Determinants.{pdf,svg}

Authors: Guillaume Deside
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

from src.config.figure_config import FIG_DIR, LANCET_MIN_FONT_PT, lancet_figsize
from src.metrics.metrics import METRIC_MEDIAN_GLUCOSE
from src.visualization.style import (
    apply_lancet_style,
    enforce_lancet_minimums,
    save_figure,
    style_axes,
)
from src.visualization.colors import WONG_COLORS, COLORMAP_CONTINUOUS

# Reuse the phenotype dataset/computation from the exploratory script rather
# than duplicating the CSV parsing and recoding logic.
from scripts.statistics.slh_patient_characteristics_exploration import (
    load_data,
    CONTINUOUS_COLS,
    CONTINUOUS_LABELS,
    CATEGORICAL_COLS,
    CATEGORICAL_LABELS,
    RESPONSE as RESPONSE_PHENO,
    RESPONSE_LABEL as RESPONSE_LABEL_PHENO,
    slh_colormap,
    INSULIN_REGIMEN_MARKERS,
)

# Reuse the combined CGM+phenotype PCA computation and colour/variable
# definitions from Figure 5 (SLH determinants) rather than duplicating them.
from scripts.paper_figures.generate_figure_6_slh_determinants import (
    build_dataset,
    compute_combined_pca,
    PCA_COLS,
    PCA_CGM_COLS,
    PCA_VAR_COLORS,
    LABELS_SHORT,
    RESPONSE_LABEL as RESPONSE_LABEL_PCA,
)


# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = os.path.join(FIG_DIR, "Supplementary")
OUTPUT_NAME = "SupplementaryFigure_Phenotype_Determinants"

MEDIAN_GLUCOSE_LABEL_NUDGE = (-0.10, 0.09)
AGE_ENTRY_LABEL_NUDGE      = (0.10, -0.09)
# Diabetes duration and BMI z-score point in nearly the same direction, so
# their labels land on top of each other; drop this one below the other.
DIABETES_DURATION_LABEL_NUDGE = (0.0, -0.16)

# Correlation-circle framing: the unit circle plus room for the variable
# labels set just beyond the arrow tips.
PCA_CIRCLE_LIM = 1.85
PCA_LABEL_SCALE = 1.12

# ── Type sizes (nothing below 7 pt at final size) ────────────────────
FONT_BODY = LANCET_MIN_FONT_PT          # Tick labels, legends, in-plot values
FONT_TITLE = LANCET_MIN_FONT_PT + 1     # Panel headings
FONT_PANEL_LETTER = LANCET_MIN_FONT_PT + 2   # Panel letters (bold capitals)

# Four panels on one full-width page.  The height uses most of the page depth
# the journal allows (230 mm), because at 175 mm wide the four panels are otherwise
# left with well under 50 mm of plotting area each.
FIGURE_HEIGHT_INCHES = 8.5

# Panel A spends ~35 mm of its cell on category names, so its row gets the
# extra width; the correlation circle in D is square and cannot use it.
TOP_ROW_WIDTH_RATIOS = [1.04, 0.96]
BOTTOM_ROW_WIDTH_RATIOS = [1.06, 0.94]


def format_p_value(p: float) -> str:
    """
    Format a p value in house style: two significant figures, or ``p<0.0001``.

    The previous ``f"p={p:.3f}"`` rendered anything below 0.0005 as the
    impossible "p=0.000".
    """
    if p < 0.0001:
        return "p<0.0001"

    return f"p={float(f'{p:.2g}'):g}"


def add_panel_label(ax, label, x=-0.12, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=FONT_PANEL_LETTER,
            fontweight="bold", va="bottom", ha="right")


# =============================================================================
# PANEL A: Mutual information (phenotype variables vs n_SLH)
# =============================================================================
def plot_mutual_information_panel(ax, df: pd.DataFrame) -> dict:
    """Same computation as plot_mutual_information in
    slh_patient_characteristics_exploration.py, drawn onto a shared subplot
    instead of its own figure.

    Returns
    -------
    dict
        ``n`` (participants contributing) and ``top`` (the three variables
        with the highest mutual information), if reported.
    """
    mi_rows = []
    for col in CONTINUOUS_COLS:
        sub = df[[col, RESPONSE_PHENO]].dropna()
        mi = mutual_info_regression(sub[[col]], sub[RESPONSE_PHENO], random_state=42)[0]
        mi_rows.append({"label": CONTINUOUS_LABELS[col], "MI": mi, "type": "Continuous"})

    for col in CATEGORICAL_COLS:
        sub = df[[col, RESPONSE_PHENO]].dropna()
        if len(sub) < 10:
            continue
        dummies = pd.get_dummies(sub[col], drop_first=False).astype(float)
        if dummies.shape[1] == 0:
            continue
        mi_vals = mutual_info_regression(dummies, sub[RESPONSE_PHENO], random_state=42)
        mi_rows.append({"label": CATEGORICAL_LABELS[col], "MI": float(mi_vals.sum()),
                        "type": "Categorical"})

    mi_df = pd.DataFrame(mi_rows).sort_values("MI", ascending=True)

    type_colors = {"Continuous": WONG_COLORS["blue"], "Categorical": WONG_COLORS["orange"]}
    bar_colors = [type_colors[t] for t in mi_df["type"]]

    bars = ax.barh(mi_df["label"], mi_df["MI"], color=bar_colors,
                   edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, mi_df["MI"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
               f"{val:.3f}", va="center", fontsize=FONT_BODY)

    # Room for the value labels, which otherwise run off the right-hand edge
    # once they are set at the 7 pt minimum.
    ax.set_xlim(0, mi_df["MI"].max() * 1.22)

    handles = [mpatches.Patch(color=c, label=t, alpha=0.85) for t, c in type_colors.items()]
    ax.legend(handles=handles, fontsize=FONT_BODY, frameon=False, loc="lower right")
    ax.set_xlabel(f"Mutual information with {RESPONSE_LABEL_PHENO}",
                  fontsize=FONT_BODY)
    ax.set_title("Mutual information with SLH frequency", fontweight="bold",
                loc="left", fontsize=FONT_TITLE)
    ax.tick_params(labelsize=FONT_BODY)
    style_axes(ax)

    return {
        "n": int(df[RESPONSE_PHENO].notna().sum()),
        "top": list(mi_df.sort_values("MI", ascending=False)["label"].head(3)),
    }


# =============================================================================
# PANEL B: GTAA1C vs diabetes duration, colour = n_SLH, marker = insulin regimen
# =============================================================================
def plot_gtaa1c_duration_by_insulin_regimen(ax, df: pd.DataFrame) -> dict:
    """Same construction as plot_gtaa1c_duration_by_insulin in
    slh_patient_characteristics_exploration.py: marker shape encodes insulin
    regimen (basal-prandial / two injections / insulin pump) -- i.e. the
    type of insulin delivery.

    Returns
    -------
    dict
        ``n``, Spearman ``rho`` and ``p`` if reported, plus ``handles``:
        the insulin-regimen key, which the caller places outside the panel.
    """
    cols = ["diabetes_duration", "GTAA1c", RESPONSE_PHENO, "Insulin_regimen"]
    sub = df[cols].dropna()

    x = sub["diabetes_duration"].values
    y = sub["GTAA1c"].values
    c = sub[RESPONSE_PHENO].values
    r, p = stats.spearmanr(x, y)

    cmap, norm = slh_colormap(c)
    sc = None
    for regimen, marker in INSULIN_REGIMEN_MARKERS.items():
        mask = (sub["Insulin_regimen"] == regimen).values
        if not mask.any():
            continue
        sc = ax.scatter(x[mask], y[mask], c=c[mask], cmap=cmap, norm=norm,
                        marker=marker, s=28, alpha=0.85,
                        edgecolors="white", linewidths=0.5, zorder=3)

    b, a = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, a + b * x_line, color="black", lw=1.2, ls="--", alpha=0.6, zorder=1)

    ax.set_xlabel(CONTINUOUS_LABELS["diabetes_duration"], fontsize=FONT_BODY)
    ax.set_ylabel(CONTINUOUS_LABELS["GTAA1c"], fontsize=FONT_BODY)
    ax.set_title("GTAA1C vs diabetes duration", fontweight="bold",
                loc="left", fontsize=FONT_TITLE)

    # The statistic belongs on the panel, not in its heading: house style
    # keeps headings to a descriptive phrase.
    ax.text(0.02, 0.02, f"Spearman ρ={r:.2f}, {format_p_value(p)}",
           transform=ax.transAxes, fontsize=FONT_BODY,
           va="bottom", ha="left", color="#333333",
           bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                    alpha=0.85))
    style_axes(ax)

    cbar = ax.figure.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=FONT_BODY)
    cbar.set_label(RESPONSE_LABEL_PHENO, fontsize=FONT_BODY)

    legend_handles = [
        Line2D([0], [0], marker=marker, color="w", markerfacecolor="#999999",
              markeredgecolor="black", markersize=4,
              label=f"{regimen} (n={(sub['Insulin_regimen'] == regimen).sum()})")
        for regimen, marker in INSULIN_REGIMEN_MARKERS.items()
        if (sub["Insulin_regimen"] == regimen).any()
    ]
    # The key is returned, not drawn here: inside the panel it covered the
    # top-right of the point cloud, and the colorbar occupies the right edge.
    # The caller attaches it to the row below the panel, where constrained
    # layout reserves room for it — which it does not do for a legend anchored
    # outside an Axes.
    return {
        "n": int(len(sub)),
        "rho": float(r),
        "p": float(p),
        "handles": legend_handles,
    }


# =============================================================================
# PANELS C/D: Combined CGM + phenotype PCA
# (computation reused verbatim from generate_figure_6_slh_determinants.py;
#  plotting re-declared only to control panel labelling for this layout)
# =============================================================================
def plot_pca_scores_panel(ax, coords, y, var_exp) -> None:
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=y, cmap=COLORMAP_CONTINUOUS,
                    s=18, alpha=0.8, edgecolors="none")
    cbar = ax.figure.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=FONT_BODY)
    cbar.set_label(RESPONSE_LABEL_PCA, fontsize=FONT_BODY)
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% of variance)", fontsize=FONT_BODY)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% of variance)", fontsize=FONT_BODY)
    ax.tick_params(labelsize=FONT_BODY)
    style_axes(ax)
    ax.set_title("CGM and phenotype PCA — scores", fontweight="bold",
                loc="left", pad=4, fontsize=FONT_TITLE)


def plot_pca_circle_panel(ax, loadings_corr, var_exp) -> None:
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="grey", lw=1, ls="--", alpha=0.5)
    ax.axhline(0, color="grey", lw=0.5, ls="--", alpha=0.5)
    ax.axvline(0, color="grey", lw=0.5, ls="--", alpha=0.5)

    label_nudge = {
        METRIC_MEDIAN_GLUCOSE: MEDIAN_GLUCOSE_LABEL_NUDGE,
        "age_study_entry":     AGE_ENTRY_LABEL_NUDGE,
        "diabetes_duration":   DIABETES_DURATION_LABEL_NUDGE,
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
        scale = PCA_LABEL_SCALE
        nx, ny = label_nudge.get(col, (0.0, 0.0))
        ha = "left" if lx >= 0 else "right"
        va = "bottom" if ly >= 0 else "top"
        ax.text(lx * scale + nx, ly * scale + ny, LABELS_SHORT[col],
                ha=ha, va=va, fontsize=FONT_BODY, color=color,
                fontweight="bold")

    ax.set_xlim(-PCA_CIRCLE_LIM, PCA_CIRCLE_LIM)
    ax.set_ylim(-PCA_CIRCLE_LIM, PCA_CIRCLE_LIM)
    ax.set_aspect("equal")
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% of variance)", fontsize=FONT_BODY)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% of variance)", fontsize=FONT_BODY)
    ax.tick_params(labelsize=FONT_BODY)
    style_axes(ax)
    ax.set_title("CGM and phenotype PCA — correlation circle",
                fontweight="bold", loc="left", pad=4, fontsize=FONT_TITLE)

    # The solid/dashed convention, previously a second title line.
    ax.legend(
        handles=[
            Line2D([0], [0], color="#555555", lw=1.6, ls="solid",
                  label="CGM metric"),
            Line2D([0], [0], color="#555555", lw=1.6, ls=(0, (4, 1.5)),
                  label="Phenotype variable"),
        ],
        fontsize=FONT_BODY, loc="lower left", frameon=True, framealpha=1.0,
        edgecolor="#cccccc", borderpad=0.35, handletextpad=0.5,
    )


# =============================================================================
# MAIN FIGURE GENERATION
# =============================================================================
def generate_supplementary_figure() -> None:
    apply_lancet_style()

    print("=" * 70)
    print("  Supplementary Figure: Phenotype Determinants of SLH")
    print("=" * 70)

    print("\n  Loading phenotype dataset (panels A, B)...")
    df_pheno = load_data()

    print("\n  Loading combined CGM + phenotype dataset (panels C, D)...")
    df_pca_data = build_dataset(apply_filter=True)
    coords, y, var_exp, loadings_corr, _ = compute_combined_pca(df_pca_data)

    # Constrained layout rather than hand-tuned margins: the figure is now
    # drawn at a fixed width and saved uncropped, so the panels have to
    # fit the canvas by themselves.
    #
    # One sub-figure per row, rather than a single 2x2 grid: in a shared grid
    # every axes in a column is aligned to the widest left-hand decoration, so
    # panel A's long category names pushed panel C ~35 mm to the right and cost
    # it that much plotting width for nothing.
    fig = plt.figure(
        figsize=lancet_figsize(FIGURE_HEIGHT_INCHES, columns="full"),
        constrained_layout=True,
    )
    row_top, row_bottom = fig.subfigures(2, 1)

    ax_a, ax_b = row_top.subplots(
        1, 2, gridspec_kw={"width_ratios": TOP_ROW_WIDTH_RATIOS},
    )
    ax_c, ax_d = row_bottom.subplots(
        1, 2, gridspec_kw={"width_ratios": BOTTOM_ROW_WIDTH_RATIOS},
    )

    mi_info = plot_mutual_information_panel(ax_a, df_pheno)
    add_panel_label(ax_a, "A")

    scatter_info = plot_gtaa1c_duration_by_insulin_regimen(ax_b, df_pheno)
    add_panel_label(ax_b, "B")

    # Panel B's marker key, just below its panel rather than over its data.
    row_top.legend(handles=scatter_info["handles"], title="Insulin regimen",
                  loc="outside lower right", ncol=3, frameon=False,
                  fontsize=FONT_BODY, title_fontsize=FONT_BODY,
                  handletextpad=0.4, columnspacing=1.2)

    plot_pca_scores_panel(ax_c, coords, y, var_exp)
    add_panel_label(ax_c, "C")

    plot_pca_circle_panel(ax_d, loadings_corr, var_exp)
    add_panel_label(ax_d, "D")

    fixed = enforce_lancet_minimums(fig)
    if any(fixed.values()):
        print(f"  Minimums enforced on {fixed['text']} text object(s) "
              f"and {fixed['lines']} line(s)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    save_figure(fig, output_path, bbox_inches=None)

    print("=" * 70 + "\n")


if __name__ == "__main__":
    generate_supplementary_figure()
