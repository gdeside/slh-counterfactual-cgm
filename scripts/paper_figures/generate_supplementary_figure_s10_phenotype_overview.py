"""
scripts/paper_figures/generate_supplementary_figure_s10_phenotype_overview.py
=============================================================================
Supplementary Figure S10: patient-phenotype determinants of SLH frequency.

Panels
------
  A   - Mutual information of every phenotype variable with n_SLH.
  B   - GTAA1C vs diabetes duration, colour = n_SLH, marker = insulin regimen.
  C-J - n_SLH by category, one box plot per categorical phenotype variable
        (Mann-Whitney for two levels, Kruskal-Wallis for three or more).

Panels A and B are drawn by the functions of
generate_supplementary_figure_phenotype_determinants.py, and the box plots by
plot_slh_by_category_panel of
scripts/statistics/slh_patient_characteristics_exploration.py, so the three
figures stay identical panel by panel.

Usage
-----
    python3 -m scripts.paper_figures.generate_supplementary_figure_s10_phenotype_overview

Requires the clinical phenotype table (not in the repository).

Artwork specification
---------------------
No title is drawn inside the artwork.

Outputs (FIG_DIR / Supplementary)
----------------------------------
  SupplementaryFigure_S10_Phenotype_Overview.{pdf,svg}

Authors: Guillaume Deside
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from src.config.figure_config import FIG_DIR, lancet_figsize
from src.visualization.style import (
    apply_lancet_style,
    enforce_lancet_minimums,
    save_figure_with_panels,
)
from scripts.statistics.slh_patient_characteristics_exploration import (
    load_data,
    CATEGORICAL_COLS,
    CATEGORICAL_LABELS,
    RESPONSE,
    plot_slh_by_category_panel,
    format_p_value,
)
from scripts.paper_figures.generate_supplementary_figure_phenotype_determinants import (
    plot_mutual_information_panel,
    plot_gtaa1c_duration_by_insulin_regimen,
    add_panel_label,
    FONT_BODY,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = os.path.join(FIG_DIR, "Supplementary")
OUTPUT_NAME = "SupplementaryFigure_S10_Phenotype_Overview"

FIGURE_HEIGHT_INCHES = 9.0
ROW_HEIGHT_RATIOS = [1.25, 2.0]   # panels A-B : box plots
BOX_N_COLS = 3


# =============================================================================
# BOX PLOTS BY CATEGORY
# =============================================================================
def category_test_panels(df: pd.DataFrame) -> list:
    """(variable label, groups, levels, test name, p) for every usable categorical variable."""
    panels = []
    for col in CATEGORICAL_COLS:
        sub = df[[col, RESPONSE]].dropna()
        kept = [(sub.loc[sub[col] == lv, RESPONSE].values, lv)
                for lv in sorted(sub[col].unique())]
        kept = [(g, lv) for g, lv in kept if len(g) >= 2]
        if len(kept) < 2:
            continue
        groups = [g for g, _ in kept]
        levels = [lv for _, lv in kept]
        if len(groups) == 2:
            _, p = scipy_stats.mannwhitneyu(*groups, alternative="two-sided")
            test_name = "Mann-Whitney"
        else:
            _, p = scipy_stats.kruskal(*groups)
            test_name = "Kruskal-Wallis"
        panels.append((CATEGORICAL_LABELS[col], groups, levels, test_name, p))
    return panels


# =============================================================================
# MAIN
# =============================================================================
def generate_supplementary_figure() -> None:
    apply_lancet_style()

    print("=" * 70)
    print("  Supplementary Figure S10: Phenotype overview")
    print("=" * 70)

    df = load_data()
    box_panels = category_test_panels(df)
    n_rows_box = int(np.ceil(len(box_panels) / BOX_N_COLS))

    fig = plt.figure(
        figsize=lancet_figsize(FIGURE_HEIGHT_INCHES, columns="full"),
        constrained_layout=True,
    )
    row_top, row_bottom = fig.subfigures(2, 1, height_ratios=ROW_HEIGHT_RATIOS)

    ax_a, ax_b = row_top.subplots(1, 2, gridspec_kw={"width_ratios": [1.04, 0.96]})
    box_axes = np.array(row_bottom.subplots(n_rows_box, BOX_N_COLS)).reshape(-1)

    plot_mutual_information_panel(ax_a, df)
    add_panel_label(ax_a, "A")

    scatter_info = plot_gtaa1c_duration_by_insulin_regimen(ax_b, df)
    add_panel_label(ax_b, "B")
    row_top.legend(handles=scatter_info["handles"], title="Insulin regimen",
                   loc="outside lower right", ncol=3, frameon=False,
                   fontsize=FONT_BODY, title_fontsize=FONT_BODY,
                   handletextpad=0.4, columnspacing=1.2)

    letters = {}
    for i, (var_label, groups, levels, test_name, p) in enumerate(box_panels):
        ax = box_axes[i]
        plot_slh_by_category_panel(ax, groups, [lv.replace(" ", "\n") for lv in levels],
                                   test_name, p, var_label)
        # Three columns: heading on two lines (level names are wrapped above).
        ax.set_title(f"{var_label}\n{test_name} {format_p_value(p)}",
                     fontsize=FONT_BODY, loc="left", pad=4)
        letter = "CDEFGHIJKL"[i]
        add_panel_label(ax, letter, x=-0.22)
        letters[letter] = ax
    for ax in box_axes[len(box_panels):]:
        ax.set_visible(False)

    fixed = enforce_lancet_minimums(fig)
    if any(fixed.values()):
        print(f"  Minimums enforced on {fixed['text']} text object(s) "
              f"and {fixed['lines']} line(s)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    save_figure_with_panels(
        fig, output_path, OUTPUT_NAME,
        {"A": ax_a, "B": [ax_b, *row_top.legends], **letters},
        bbox_inches=None)
    print(f"  Saved: {output_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    generate_supplementary_figure()
