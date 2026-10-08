"""
scripts/paper_figures/generate_supplementary_figure_onset_time_per_patient.py
=============================================================================
Supplementary figure: per-patient time-of-day distribution of hyperglycemia
episodes, for all four categories of Figure 2.

Supports the sentence in the Results:

    "However, the time distribution of SLHs and LLHs varied among participants,
     with some patients experiencing SLHs but no LLHs at specific times of day."

Figure 2 panel D pools all participants into one curve per category, which hides
who contributes to which hour. This figure keeps the same categories, colours and
night window, but unfolds the cohort by patient: one row per participant, one
column per hour of onset, one panel per category:

  A — SLH           B — Not isolated           C — Night           D — LLH

Rows are identical in all panels (same patient on the same line) and sorted by
the circular mean onset hour of each patient's SLH episodes; patients without
any SLH follow, ordered the same way by their first non-empty category in the
order SLH > Not isolated > Night > LLH. A patient can therefore be followed
across panels: an orange cell with an empty blue cell at the same hour is an
SLH-without-LLH time of day.

Classification is exactly that of Figure 2 (imported from
`generate_figure_2_classification_summary`; episodes failing C2 are dropped).

Usage
-----
    python3 -m scripts.paper_figures.generate_supplementary_figure_onset_time_per_patient

Outputs (FIG_DIR / Manuscript_Figures)
---------------------------------------
  SuppFigure_Onset_Time_Per_Patient.{pdf,svg}
  onset_time_per_patient_hourly_counts.csv   (patient x hour, one column per category)

Authors: Guillaume Deside
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

from src.episodes.episode_config import NIGHT_START_H, NIGHT_END_H
from src.visualization.style import apply_publication_style, save_figure, FONT_SIZE_BASE
from scripts.paper_figures.generate_figure_2_classification_summary import (
    build_cohort_episodes, CATEGORY_ORDER, CATEGORY_COLORS, OUTPUT_DIR,
)

OUTPUT_NAME = "SuppFigure_Onset_Time_Per_Patient"

BIN_WIDTH_H = 1          # one column per hour of onset
MAX_COUNT_DISPLAYED = 4  # colour scale saturates at this many episodes per cell


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


def order_patients(counts: dict[str, np.ndarray]) -> np.ndarray:
    """Row order: group = index of the patient's first non-empty category in CATEGORY_ORDER
    (patients with no episode at all last); within a group, by circular-mean onset hour of
    that category."""
    n = next(iter(counts.values())).shape[0]
    group = np.full(n, len(CATEGORY_ORDER))
    key = np.zeros(n)
    for i in range(n):
        for g, cat in enumerate(CATEGORY_ORDER):
            m = circular_mean_hour(counts[cat][i])
            if not np.isnan(m):
                group[i], key[i] = g, m
                break
    return np.lexsort((key, group))


# =============================================================================
# PLOTTING
# =============================================================================
def plot_count_heatmap(fig, ax, mat: np.ndarray, category: str, panel: str) -> None:
    n = mat.shape[0]
    cmap = LinearSegmentedColormap.from_list(f"w_{panel}", ["#ffffff", CATEGORY_COLORS[category]],
                                             N=MAX_COUNT_DISPLAYED + 1)
    norm = BoundaryNorm(np.arange(0, MAX_COUNT_DISPLAYED + 2), cmap.N)
    ax.imshow(np.minimum(mat, MAX_COUNT_DISPLAYED), aspect="auto", cmap=cmap, norm=norm,
              interpolation="nearest", extent=(0, 24, n, 0), zorder=2)
    for h in (NIGHT_END_H, NIGHT_START_H):  # night window wraps midnight
        ax.axvline(h, color="#333333", lw=0.6, ls="--", zorder=5)
    ax.set_xlim(0, 24)
    ax.set_ylim(n, 0)
    ax.set_xticks(range(0, 25, 6))
    ax.tick_params(axis="x", labelsize=FONT_SIZE_BASE)
    ax.set_xlabel("Onset time of day (h)", fontsize=FONT_SIZE_BASE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.set_title(f"{category} (n={int(mat.sum())})", fontweight="bold", loc="left", pad=6,
                 fontsize=FONT_SIZE_BASE)
    ax.text(-0.02 if panel == "A" else -0.08, 1.04, panel, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="right")

    # inset axes (not `ax=`) so the colourbar does not steal height/width from the heatmap
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=ax.inset_axes([0.15, -0.15, 0.7, 0.025]), orientation="horizontal")
    ticks = np.arange(0, MAX_COUNT_DISPLAYED + 1)
    cb.set_ticks(ticks + 0.5)
    cb.set_ticklabels([str(t) if t < MAX_COUNT_DISPLAYED else f"≥{t}" for t in ticks])
    cb.ax.tick_params(labelsize=FONT_SIZE_BASE, length=2)
    cb.set_label("Episodes per hour", fontsize=FONT_SIZE_BASE)
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
    order = order_patients(counts)
    pids_sorted = [pids[i] for i in order]
    counts = {c: m[order] for c, m in counts.items()}
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
        1, len(CATEGORY_ORDER), figure=fig, wspace=0.1,
        left=0.07, right=0.98, top=1 - 0.5 / fig_height, bottom=0.9 / fig_height,
    )
    axes = []
    for k, cat in enumerate(CATEGORY_ORDER):
        ax = fig.add_subplot(gs[0, k], sharey=axes[0] if axes else None)
        plot_count_heatmap(fig, ax, counts[cat], cat, "ABCD"[k])
        if k:
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis="y", length=0)
        axes.append(ax)

    ax0 = axes[0]
    ax0.set_ylabel(f"Patients (n={n}, sorted by mean SLH onset time)", fontsize=FONT_SIZE_BASE)
    ax0.set_yticks([0, n - 1] if n > 1 else [0])
    ax0.set_yticklabels(["1", str(n)] if n > 1 else ["1"], fontsize=FONT_SIZE_BASE)

    fig.text(0.98, 1 - 0.18 / fig_height,
             f"Dashed lines: night window ({NIGHT_START_H:02d}:00–{NIGHT_END_H:02d}:00)",
             fontsize=FONT_SIZE_BASE - 0.5, color="#555555", ha="right", va="top")

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    save_figure(fig, output_path)
    print("\n" + "=" * 70)
    print(f"  SAVED: {output_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    generate_figure()
