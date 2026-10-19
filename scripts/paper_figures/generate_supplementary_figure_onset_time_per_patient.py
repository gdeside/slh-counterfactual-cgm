"""
scripts/paper_figures/generate_supplementary_figure_onset_time_per_patient.py
=============================================================================
Supplementary figure: time of day at which hyperglycemia episodes start, shown
for every patient of the cohort and for each episode category.

Each hyperglycemia episode of the cohort belongs to one of four categories:

  SLH           short, isolated, sufficiently complete and starting in daytime
  Not isolated  short enough, but not isolated and/or insufficiently complete
  Night         would be SLH, but starts during the night window
  LLH           duration outside the short-episode bounds

The figure has three panels (A SLH + Night, B Not isolated, C LLH): SLH and Night
share panel A (a Night episode is an SLH that merely starts at night) but keep their
own colours: SLH orange in daytime hours, Night navy in night hours.
In every panel each row is one patient and each column is one hour of the day
(onset time); the colour intensity is the number of episodes of that group that started
in that hour, on a 4-level scale (plus white for none). The levels differ per
panel (PANEL_STEPS) because LLH counts are almost everywhere above 4, so they
need coarser steps than the other panels. Dashed vertical lines mark the edges
of the night window.

Rows are identical in all panels, so one patient can be followed across
panels: they are sorted by the circular mean onset hour of the patient's
SLH + Night episodes; patients without any come next, sorted the same way using
their first non-empty group in the order SLH + Night > Not isolated > LLH.

Console output: number of patients with at least one SLH / LLH, and number of
patients with at least one hour of the day that has SLH but no LLH.

Usage
-----
    python3 -m scripts.paper_figures.generate_supplementary_figure_onset_time_per_patient

Outputs (FIG_DIR / Manuscript_Figures)
---------------------------------------
  SuppFigure_Onset_Time_Per_Patient/SuppFigure_Onset_Time_Per_Patient.{pdf,svg}   (whole figure)
  SuppFigure_Onset_Time_Per_Patient/SuppFigure_Onset_Time_Per_Patient_panel_{A,B,C}.{pdf,svg}
  onset_time_per_patient_hourly_counts.csv   (patient x hour, one column per category, SLH and Night kept separate)

Authors: Guillaume Deside
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgb

from src.episodes.episode_config import NIGHT_START_H, NIGHT_END_H
from src.visualization.style import apply_publication_style, save_figure_with_panels, FONT_SIZE_BASE
from scripts.paper_figures.generate_figure_2_classification_summary import (
    build_cohort_episodes, CATEGORY_ORDER, CATEGORY_COLORS, OUTPUT_DIR,
)

OUTPUT_NAME = "SuppFigure_Onset_Time_Per_Patient"

BIN_WIDTH_H = 1          # one column per hour of onset

# Display panels: (title, categories drawn on the panel, each in its CATEGORY_COLORS colour).
PANELS = [
    ("SLH + Night",  ["SLH", "Night"]),
    ("Not isolated", ["Not isolated"]),
    ("LLH",          ["LLH"]),
]

# Per-panel colour steps: lower bound (episodes per cell) of each of the 4 colours;
# 0 is always white. LLH counts are almost always >= 4, hence its coarser steps.
PANEL_STEPS = {
    "SLH + Night":  (1, 2, 3, 4),
    "Not isolated": (1, 2, 3, 4),
    "LLH":          (1, 4, 8, 12),
}
SHADE_FRACTIONS = (0.3, 0.55, 0.78, 1.0)   # light -> full category colour


# =============================================================================
# DATA
# =============================================================================
def hourly_counts(episodes_df: pd.DataFrame, category: str, pids: list[str]) -> np.ndarray:
    """(n_patients, 24/BIN_WIDTH_H) matrix of `category` episode counts by onset hour, rows in `pids` order."""
    sub = episodes_df[episodes_df["category"] == category]
    n_bins = 24 // BIN_WIDTH_H
    hours = np.floor(sub["onset_hour"].values / BIN_WIDTH_H).astype(int) % n_bins
    mat = np.zeros((len(pids), n_bins), dtype=int)
    row = {p: i for i, p in enumerate(pids)}
    for pid, h in zip(sub["patient_id"].values, hours):
        mat[row[pid], h] += 1
    return mat


def circular_mean_hour(counts_row: np.ndarray) -> float:
    """Circular mean onset hour (0-24) from an hourly count vector; NaN when empty."""
    if counts_row.sum() == 0:
        return np.nan
    centers = (np.arange(len(counts_row)) + 0.5) * BIN_WIDTH_H
    ang = 2 * np.pi * centers / 24
    mean = np.arctan2((counts_row * np.sin(ang)).sum(), (counts_row * np.cos(ang)).sum())
    return (mean % (2 * np.pi)) * 24 / (2 * np.pi)


def order_patients(counts: dict[str, np.ndarray], order: list[str]) -> np.ndarray:
    """Row order: group = index of the patient's first non-empty panel in `order`
    (patients with no episode at all last); within a group, by circular-mean onset hour of
    that panel."""
    n = next(iter(counts.values())).shape[0]
    group = np.full(n, len(order))
    key = np.zeros(n)
    for i in range(n):
        for g, cat in enumerate(order):
            m = circular_mean_hour(counts[cat][i])
            if not np.isnan(m):
                group[i], key[i] = g, m
                break
    return np.lexsort((key, group))


# =============================================================================
# PLOTTING
# =============================================================================
def step_labels(steps: tuple[int, ...]) -> list[str]:
    """Colour-bar labels for the 4 levels given their lower bounds, e.g. (1,4,8,12) -> 1-3, 4-7, 8-11, >=12."""
    labels = []
    for lo, hi in zip(steps[:-1], steps[1:]):
        labels.append(str(lo) if hi - lo == 1 else f"{lo}–{hi - 1}")
    labels.append(f"≥{steps[-1]}")
    return labels


def plot_count_heatmap(fig, ax, mats: dict[str, np.ndarray], title: str, panel: str) -> None:
    """`mats`: category -> (n_patients, 24) counts, each drawn in its own colour on the same panel
    (SLH starts in daytime hours and Night in night hours, so their cells never overlap)."""
    steps = PANEL_STEPS[title]
    n = next(iter(mats.values())).shape[0]
    total = sum(mats.values())
    img = np.ones((*total.shape, 3))
    cmaps = {}
    for cat, mat in mats.items():
        base = np.array(to_rgb(CATEGORY_COLORS[cat]))
        shades = [tuple(1 - f * (1 - base)) for f in SHADE_FRACTIONS]
        cmaps[cat] = ListedColormap(["#ffffff"] + shades)
        level = np.digitize(mat, steps)            # 0 = none, 1..4 = colour level
        img[level > 0] = np.array([to_rgb(c) for c in cmaps[cat].colors])[level[level > 0]]
    ax.imshow(img, aspect="auto", interpolation="nearest", extent=(0, 24, n, 0), zorder=2)
    for h in (NIGHT_END_H, NIGHT_START_H):  # night window wraps midnight
        ax.axvline(h, color="#333333", lw=0.6, ls="--", zorder=5)
    ax.set_xlim(0, 24)
    ax.set_ylim(n, 0)
    ax.set_xticks(range(0, 25, 6))
    ax.tick_params(axis="x", labelsize=FONT_SIZE_BASE)
    ax.set_xlabel("Onset time of day (h)", fontsize=FONT_SIZE_BASE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.set_title(f"{title} (n={int(total.sum())})", fontweight="bold", loc="left", pad=6,
                 fontsize=FONT_SIZE_BASE)
    ax.text(-0.02 if panel == "A" else -0.08, 1.04, panel, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="right")

    # one colour bar per category, side by side; inset axes so they do not steal heatmap space
    norm = BoundaryNorm(np.arange(0, 6), 6)
    k = len(mats)
    gap, width = 0.08, (0.8 - 0.08 * (len(mats) - 1)) / len(mats)
    for i, (cat, mat) in enumerate(mats.items()):
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmaps[cat])
        cax = ax.inset_axes([0.1 + i * (width + gap), -0.15, width, 0.025])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_ticks(np.arange(5) + 0.5)
        cb.set_ticklabels(["0", *step_labels(steps)])
        cb.ax.tick_params(labelsize=FONT_SIZE_BASE, length=2)
        cb.set_label(f"{cat}: episodes per hour" if k > 1 else "Episodes per hour",
                     fontsize=FONT_SIZE_BASE)
        cb.outline.set_visible(False)


# =============================================================================
# MAIN
# =============================================================================
def generate_figure() -> None:
    apply_publication_style()
    print("\n" + "=" * 70)
    print("  GENERATING SUPPLEMENTARY FIGURE — ONSET TIME PER PATIENT (ALL CATEGORIES)")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    episodes_df, _ = build_cohort_episodes()
    pids = sorted(episodes_df["patient_id"].unique())

    counts = {c: hourly_counts(episodes_df, c, pids) for c in CATEGORY_ORDER}
    panel_counts = {t: sum(counts[c] for c in cats) for t, cats in PANELS}
    order = order_patients(panel_counts, [t for t, _ in PANELS])
    pids_sorted = [pids[i] for i in order]
    counts = {c: m[order] for c, m in counts.items()}
    panel_counts = {t: m[order] for t, m in panel_counts.items()}
    n = len(pids_sorted)

    # ---- numbers checkable against the Results sentence ----
    slh, llh = counts["SLH"], counts["LLH"]
    slh_not_llh = (slh > 0) & (llh == 0)
    print(f"\n  Patients with >=1 SLH: {int((slh.sum(1) > 0).sum())}/{n};  "
          f"with >=1 LLH: {int((llh.sum(1) > 0).sum())}/{n}")
    print(f"  Patients with >=1 hour of the day with SLH but no LLH: "
          f"{int((slh_not_llh.sum(1) > 0).sum())}/{n}")

    long = pd.DataFrame({
        "patient_id": np.repeat(pids_sorted, 24 // BIN_WIDTH_H),
        "hour": np.tile(np.arange(24 // BIN_WIDTH_H) * BIN_WIDTH_H, n),
        **{f"n_{c.replace(' ', '_')}": counts[c].ravel() for c in CATEGORY_ORDER},
    })
    csv_path = os.path.join(OUTPUT_DIR, "onset_time_per_patient_hourly_counts.csv")
    long.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # ---- layout ----
    fig_width = 10.5
    heat_h = max(4.0, min(0.045 * n, 7.0))   # ~0.045 in per patient row, bounded
    fig_height = heat_h + 1.4
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        1, len(PANELS), figure=fig, wspace=0.1,
        left=0.07, right=0.98, top=1 - 0.5 / fig_height, bottom=0.9 / fig_height,
    )
    axes = []
    for k, (title, cats) in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, k], sharey=axes[0] if axes else None)
        plot_count_heatmap(fig, ax, {c: counts[c] for c in cats}, title, "ABC"[k])
        if k:
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis="y", length=0)
        axes.append(ax)

    ax0 = axes[0]
    ax0.set_ylabel(f"Patients (n={n}, sorted by mean SLH + Night onset time)", fontsize=FONT_SIZE_BASE)
    ax0.set_yticks([0, n - 1] if n > 1 else [0])
    ax0.set_yticklabels(["1", str(n)] if n > 1 else ["1"], fontsize=FONT_SIZE_BASE)

    fig.text(0.98, 1 - 0.18 / fig_height,
             f"Dashed lines: night window ({NIGHT_START_H:02d}:00–{NIGHT_END_H:02d}:00)",
             fontsize=FONT_SIZE_BASE - 0.5, color="#555555", ha="right", va="top")

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    save_figure_with_panels(fig, output_path, OUTPUT_NAME, dict(zip("ABC", axes)))
    print("\n" + "=" * 70)
    print(f"  SAVED: {output_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    generate_figure()
