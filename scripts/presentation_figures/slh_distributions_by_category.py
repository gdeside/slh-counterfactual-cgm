"""
scripts/presentation_figures/slh_distributions_by_category.py
=============================================================

Presentation figure (16:9, 1 row × 3 columns) with the key SLH distributions, using
the SAME episode classification as manuscript Figure 2
(`scripts/paper_figures/generate_figure_2_classification_summary.py`):

  Panel A — SLH episodes per patient (daytime SLH only; histogram + median)
  Panel B — SLH episode duration (histogram + median, C1 bounds)
  Panel C — Onset time of day, one line per category
            (SLH / Not isolated / Night / LLH; 15-min bins)

Classification (C1–C6 from the episode-detection output; episodes failing C2 are dropped):
  C1 fails -> LLH · C4/C5/C6 not all pass -> Not isolated · C3 fails -> Night · else -> SLH.
Loading, categories and colours are imported from the Figure 2 script, so the
two figures cannot drift apart.

Usage
-----
    python3 -m scripts.presentation_figures.slh_distributions_by_category

Output (scripts/presentation_figures/output/slh_distributions_by_category/)
---------------------------------------------------------------------------
  slh_distributions_by_category.{pdf,svg,png}
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from src.config import EPISODES_PATH
from src.episodes.episode_config import D_MIN_MIN, D_MAX_MIN, NIGHT_START_H, NIGHT_END_H
from src.visualization.colors import COLOR_NIGHT
from scripts.paper_figures.generate_figure_2_classification_summary import (
    build_cohort_episodes,
    CATEGORY_ORDER,
    CATEGORY_COLORS,
    ONSET_SCATTER_BIN_WIDTH_MIN,
)

# =============================================================================
# OUTPUT
# =============================================================================
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output", "slh_distributions_by_category")
OUTPUT_NAME = "slh_distributions_by_category"

# =============================================================================
# PRESENTATION STYLE
# =============================================================================
FIG_W, FIG_H = 19.2, 7.0   # 16:9

FS_TITLE  = 17
FS_LABEL  = 14
FS_TICK   = 12
FS_LEGEND = 11

C_SLH    = CATEGORY_COLORS["SLH"]   # same orange as Figure 2
C_MEDIAN = "#CC3311"                # median line

plt.rcParams.update({
    "font.family":       "Arial",
    "font.size":         FS_LABEL,
    "axes.labelsize":    FS_LABEL,
    "axes.titlesize":    FS_TITLE,
    "xtick.labelsize":   FS_TICK,
    "ytick.labelsize":   FS_TICK,
    "legend.fontsize":   FS_LEGEND,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# =============================================================================
# PANEL DRAWERS
# =============================================================================
def _finish(ax, title, letter, xlabel, ylabel, handles):
    ax.set_xlabel(xlabel, fontsize=FS_LABEL, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL, labelpad=6)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=10, loc="left")
    ax.legend(handles=handles, fontsize=FS_LEGEND, loc="upper right",
              framealpha=0.92, edgecolor="#CCCCCC")
    ax.tick_params(labelsize=FS_TICK)
    ax.yaxis.grid(True, ls=":", lw=0.5, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    ax.text(-0.10, 1.04, letter, transform=ax.transAxes,
            fontsize=FS_TITLE + 2, fontweight="bold", va="bottom")


def panel_count(ax, cohort_df, cohort_pids):
    """Panel A: daytime SLH episodes per patient (same definition as Figure 2B)."""
    counts = (cohort_df.loc[cohort_df["category"] == "SLH", "patient_id"]
              .value_counts().reindex(cohort_pids, fill_value=0)).values
    med = np.median(counts)
    ax.hist(counts, bins=np.arange(0, counts.max() + 2) - 0.5, color=C_SLH, alpha=0.88,
            edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(med, color=C_MEDIAN, lw=2.5, ls="--", zorder=4)
    handles = [mpatches.Patch(color=C_SLH, alpha=0.88, label="Patient count"),
               Line2D([0], [0], color=C_MEDIAN, lw=2.5, ls="--", label=f"Median = {med:.0f}")]
    _finish(ax, "Episodes per patient", "A", "SLH episodes per patient (daytime)",
            "Number of patients", handles)


def panel_duration(ax, cohort_df):
    """Panel B: duration of SLH episodes."""
    data = cohort_df.loc[cohort_df["category"] == "SLH", "duration_min"].dropna().values
    med = np.median(data)
    bins = np.append(np.arange(D_MIN_MIN, D_MAX_MIN, 2.5), D_MAX_MIN + 0.001)
    ax.hist(data, bins=bins, color=C_SLH, alpha=0.88, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(med, color=C_MEDIAN, lw=2.5, ls="--", zorder=4)
    ax.axvline(D_MIN_MIN, color="#AAAAAA", lw=1.2, ls=":", zorder=5)
    ax.axvline(D_MAX_MIN, color="#AAAAAA", lw=1.2, ls=":", zorder=5)
    ax.set_xlim(D_MIN_MIN - 1, D_MAX_MIN + 1)
    handles = [mpatches.Patch(color=C_SLH, alpha=0.88, label="Episode count"),
               Line2D([0], [0], color=C_MEDIAN, lw=2.5, ls="--", label=f"Median = {med:.0f} min"),
               Line2D([0], [0], color="#AAAAAA", lw=1.2, ls=":",
                      label=f"C1: {D_MIN_MIN:.0f}–{D_MAX_MIN:.0f} min")]
    _finish(ax, "Episode duration", "B", "Episode duration (min)", "Number of episodes", handles)


def panel_onset(ax, cohort_df, bin_width_min=ONSET_SCATTER_BIN_WIDTH_MIN):
    """Panel C: onset time of day by category (same construction as Figure 2D)."""
    edges = np.arange(0, 24 * 60 + bin_width_min, bin_width_min)
    centers_h = (edges[:-1] + bin_width_min / 2) / 60

    ax.axvspan(NIGHT_START_H, 24, color=COLOR_NIGHT, alpha=0.5, zorder=0)
    ax.axvspan(0, NIGHT_END_H, color=COLOR_NIGHT, alpha=0.5, zorder=0)

    handles = []
    for c in CATEGORY_ORDER:
        sub = cohort_df.loc[cohort_df["category"] == c, "onset_hour"].values * 60
        if len(sub) == 0:
            continue
        counts, _ = np.histogram(sub, bins=edges)
        ax.scatter(centers_h, counts, s=22, color=CATEGORY_COLORS[c], zorder=3)
        ax.plot(centers_h, counts, color=CATEGORY_COLORS[c], lw=1.8, alpha=0.85, zorder=2)
        handles.append(Line2D([0], [0], marker="o", color=CATEGORY_COLORS[c], lw=1.8,
                              label=f"{c} (n={int(counts.sum()):,})"))
    handles.append(mpatches.Patch(color=COLOR_NIGHT, alpha=0.5, label="Night window"))

    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 4)], fontsize=FS_TICK)
    _finish(ax, "Onset time by category", "C", f"Onset time of day ({bin_width_min:.0f}-min bins)",
            f"Episodes per {bin_width_min:.0f}-min bin", handles)
    ax.legend(handles=handles, fontsize=FS_LEGEND, loc="upper left", framealpha=0.92,
              edgecolor="#CCCCCC")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("  SLH distributions by category — Presentation Figure  (16:9, 1×3)")
    print("=" * 70)
    print(f"  Episodes: {EPISODES_PATH}")
    print(f"  Output  : {OUTPUT_DIR}\n")

    cohort_df, cohort_pids = build_cohort_episodes()
    if len(cohort_df) == 0:
        print("[ERROR] No episode data found.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.13, wspace=0.30)
    panel_count(axes[0], cohort_df, cohort_pids)
    panel_duration(axes[1], cohort_df)
    panel_onset(axes[2], cohort_df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"\n  Saved → {OUTPUT_NAME}.pdf / .svg / .png")
    print(f"  Output: {OUTPUT_DIR}\n")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
