"""
scripts/presentation_figures/slh_count_vs_counterfactual_deltas.py
==================================================================

Presentation figure: ΔCV / ΔTIR / Δmedian vs the number of SLH episodes per patient.

Three scatter plots (1 row × 3 columns, 16:9 slide format):
  Panel A — ΔCV (%)         vs n_SLH
  Panel B — ΔTIR 70–180 (%) vs n_SLH
  Panel C — ΔMedian glucose  vs n_SLH

All panels use daytime (06:00–22:00) delta values at the selected offset.
Linear regression line + Spearman ρ annotation included on each panel.
Representative patients (high- / low-SLH burden, chosen automatically or via CGM_PATIENT_HIGH / CGM_PATIENT_LOW) are marked with smaller markers than
in the manuscript figure.

Usage
-----
    python3 scripts/presentation_figures/slh_count_vs_counterfactual_deltas.py
    python3 scripts/presentation_figures/slh_count_vs_counterfactual_deltas.py --offset 30

Usage
-----
    python3 -m scripts.presentation_figures.slh_count_vs_counterfactual_deltas [--help for options]
    (or: make presentation_figures)

Output
------
    scripts/presentation_figures/output/slh_count_vs_counterfactual_deltas/  (git-ignored; --out_dir to change)
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats

from src.utils.patient_selection import resolve_patient_pair
from src.config import BASE_PATH, DATA_PATH
from src.visualization.style import apply_publication_style, save_figure, style_axes

# =============================================================================
# CONFIGURATION
# =============================================================================

SELECTED_OFFSET = 30

PATIENT_IDS = list(resolve_patient_pair())   # [high-SLH, low-SLH]

SENSITIVITY_RESULTS_DIR = os.path.join(BASE_PATH, "Results", "Sensitivity_Analysis")
DATA_DIR                = os.path.join(SENSITIVITY_RESULTS_DIR, "Data")
COHORT_CSV              = os.path.join(DATA_DIR, "cohort_sensitivity.csv")
EPISODES_SUMMARY_CSV    = os.path.join(DATA_PATH, "Episodes", "episodes_summary.csv")

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output", "slh_count_vs_counterfactual_deltas")
OUTPUT_NAME = "slh_count_vs_counterfactual_deltas"

# ── Presentation font sizes ──────────────────────────────────────────────────
FS_TITLE  = 16
FS_LABEL  = 13
FS_TICK   = 11
FS_ANNOT  = 11
FS_LEGEND = 11

# ── Colors ───────────────────────────────────────────────────────────────────
COLOR_POSITIVE    = "#009E73"    # Green  — Δ > 0
COLOR_NEGATIVE    = "#D55E00"    # Orange-red — Δ < 0
COLOR_REGRESSION  = "#0072B2"    # Blue   — regression line

PATIENT_COLORS = dict(zip(PATIENT_IDS, ["#E69F00", "#CC79A7"]))   # orange, purple
PATIENT_MARKERS = dict(zip(PATIENT_IDS, ["D", "s"]))             # diamond, square

# ── Scatter sizing ───────────────────────────────────────────────────────────
POINT_SIZE             = 30      # background cohort
POINT_ALPHA            = 0.60
POINT_EDGE_WIDTH       = 0.4

PATIENT_SCATTER_SIZE   = 60      # representative patients — deliberately smaller
PATIENT_SCATTER_ALPHA  = 0.95
PATIENT_SCATTER_EDGE_WIDTH = 1.2

# ── Reference lines ──────────────────────────────────────────────────────────
ZERO_COLOR = "#888888"
ZERO_LS    = ":"
ZERO_LW    = 0.9
ZERO_ALPHA = 0.70

# ── Figure geometry (16:9) ───────────────────────────────────────────────────
FIG_W, FIG_H = 19.2, 7.2   # inches — 16:9 at ~120 dpi = 2304×864 px


# =============================================================================
# HELPERS
# =============================================================================

def load_data(offset: int) -> pd.DataFrame:
    df = pd.read_csv(COHORT_CSV)
    if os.path.exists(EPISODES_SUMMARY_CSV):
        ep = pd.read_csv(EPISODES_SUMMARY_CSV)
        df = df.merge(ep[["patient_id", "n_SLH"]], on="patient_id", how="left")
    return df[df["offset"] == offset].copy()


def fisher_ci(r, n, level=0.95):
    """95 % CI for Spearman ρ via Fisher z-transform."""
    if n < 4 or abs(r) >= 1.0:
        return np.nan, np.nan
    z  = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    zc = stats.norm.ppf(1 - (1 - level) / 2)
    return np.tanh(z - zc * se), np.tanh(z + zc * se)


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# =============================================================================
# PANEL DRAWER
# =============================================================================

def plot_scatter(ax, df_offset, metric_col, y_label,
                 patient_ids, panel_label,
                 show_legend=False):
    """
    Draw one scatter panel: daytime Δmetric vs n_SLH.

    Points are coloured by sign of Δ (green = Δ > 0, orange = Δ < 0)
    without any assumption about which direction is desirable.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    df_offset : pd.DataFrame
        Cohort data already filtered to the selected offset.
    metric_col : str
        Column name for the daytime delta metric (e.g. 'daytime_delta_cv').
    y_label : str
        Y-axis label shown on the panel.
    patient_ids : list of str
    panel_label : str
        Letter label shown top-left ("A", "B", "C").
    show_legend : bool
    """
    valid = df_offset[["patient_id", "n_SLH", metric_col]].dropna()
    if len(valid) < 3:
        ax.text(0.5, 0.5, "Insufficient data",
                ha="center", va="center", transform=ax.transAxes, fontsize=FS_ANNOT)
        return

    x = valid["n_SLH"].values
    y = valid[metric_col].values

    # Colour by sign of Δ — no directional value judgement
    point_colors = [COLOR_POSITIVE if v > 0 else COLOR_NEGATIVE for v in y]

    ax.scatter(x, y, c=point_colors, s=POINT_SIZE, alpha=POINT_ALPHA,
               edgecolors="white", linewidths=POINT_EDGE_WIDTH, zorder=4)

    # Representative patients
    for pid in patient_ids:
        row = valid[valid["patient_id"] == pid]
        if len(row) == 0:
            continue
        xp = row["n_SLH"].iloc[0]
        yp = row[metric_col].iloc[0]
        ax.scatter([xp], [yp],
                   s=PATIENT_SCATTER_SIZE,
                   c=PATIENT_COLORS.get(pid, "#999999"),
                   alpha=PATIENT_SCATTER_ALPHA,
                   edgecolors="black",
                   linewidths=PATIENT_SCATTER_EDGE_WIDTH,
                   marker=PATIENT_MARKERS.get(pid, "o"),
                   zorder=10,
                   label=pid)

    # Regression line
    slope, intercept, r_lin, p_lin, _ = stats.linregress(x, y)
    x_fit = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_fit, slope * x_fit + intercept,
            color=COLOR_REGRESSION, lw=2.0, ls="--", alpha=0.80, zorder=3)

    # Spearman ρ + CI annotation
    sp_r, sp_p = stats.spearmanr(x, y)
    ci_lo, ci_hi = fisher_ci(sp_r, len(x))
    p_str = "< 0.001" if sp_p < 0.001 else f"= {sp_p:.3f}"
    stars = sig_stars(sp_p)
    annot = (f"ρ = {sp_r:.3f} {stars}\n"
             f"95% CI [{ci_lo:.2f}, {ci_hi:.2f}]\n"
             f"p {p_str}")
    ax.text(0.04, 0.97, annot,
            transform=ax.transAxes, fontsize=FS_ANNOT,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec="#BBBBBB", lw=0.9, alpha=0.95))

    # Zero line
    ax.axhline(0, color=ZERO_COLOR, ls=ZERO_LS, lw=ZERO_LW, alpha=ZERO_ALPHA, zorder=1)

    # Axes labels & style
    ax.set_xlabel("Number of SLH episodes", fontsize=FS_LABEL, labelpad=5)
    ax.set_ylabel(y_label, fontsize=FS_LABEL, labelpad=5)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, ls=":", lw=0.4, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)

    # Panel letter
    ax.text(-0.12, 1.04, panel_label,
            transform=ax.transAxes,
            fontsize=FS_TITLE, fontweight="bold", va="bottom")

    if show_legend:
        handles = [
            mpatches.Patch(color=COLOR_POSITIVE,  label="Δ > 0"),
            mpatches.Patch(color=COLOR_NEGATIVE,  label="Δ < 0"),
            Line2D([0], [0], color=COLOR_REGRESSION, lw=2.0, ls="--",
                   label="Regression"),
        ]
        for pid in patient_ids:
            handles.append(
                Line2D([0], [0], marker=PATIENT_MARKERS.get(pid, "o"),
                       color="w",
                       markerfacecolor=PATIENT_COLORS.get(pid, "#999999"),
                       markeredgecolor="black",
                       markersize=8, label=pid)
            )
        ax.legend(handles=handles, fontsize=FS_LEGEND,
                  loc="lower right", framealpha=0.93, edgecolor="#CCCCCC")


# =============================================================================
# FIGURE BUILDER
# =============================================================================

def build_figure(offset: int) -> plt.Figure:
    apply_publication_style()

    plt.rcParams.update({
        "font.size":        FS_LABEL,
        "axes.labelsize":   FS_LABEL,
        "axes.titlesize":   FS_TITLE,
        "xtick.labelsize":  FS_TICK,
        "ytick.labelsize":  FS_TICK,
        "legend.fontsize":  FS_LEGEND,
        "figure.dpi":       150,
    })

    df_offset = load_data(offset)
    n = df_offset["patient_id"].nunique()
    print(f"  Loaded {n} patients at offset δ = {offset} min")

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.14,
                        wspace=0.32)

    # Shared x-axis range across panels
    x_vals = df_offset[["patient_id", "n_SLH"]].dropna()["n_SLH"].values

    # ── Panel A: ΔCV ──────────────────────────────────────────────────────────
    print("  Panel A: ΔCV vs n_SLH")
    plot_scatter(axes[0], df_offset,
                 metric_col="daytime_delta_cv",
                 y_label="ΔCV (%)  [daytime]",
                 patient_ids=PATIENT_IDS,
                 panel_label="A",
                 show_legend=False)

    # ── Panel B: ΔTIR 70–180 ─────────────────────────────────────────────────
    print("  Panel B: ΔTIR 70–180 vs n_SLH")
    plot_scatter(axes[1], df_offset,
                 metric_col="daytime_delta_TIR70-180",
                 y_label="ΔTIR 70–180 (%)  [daytime]",
                 patient_ids=PATIENT_IDS,
                 panel_label="B",
                 show_legend=True)

    # ── Panel C: ΔMedian ─────────────────────────────────────────────────────
    print("  Panel C: ΔMedian glucose vs n_SLH")
    plot_scatter(axes[2], df_offset,
                 metric_col="daytime_delta_median",
                 y_label="ΔMedian glucose (mg/dL)  [daytime]",
                 patient_ids=PATIENT_IDS,
                 panel_label="C",
                 show_legend=False)

    fig.suptitle(
        f"Counterfactual CGM changes vs SLH episode count  "
        f"(δ = {offset} min, daytime 06:00–22:00,  n = {n})",
        fontsize=FS_TITLE, fontweight="bold", y=0.97,
    )

    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Presentation scatter figure: ΔCV / ΔTIR / ΔMedian vs n_SLH",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--offset", type=int, default=SELECTED_OFFSET,
                        help="Counterfactual offset in minutes")
    args = parser.parse_args()

    print("=" * 70)
    print("  Figure 4 bis — Scatter  (Presentation, 16:9)")
    print("=" * 70)
    print(f"  Offset : {args.offset} min")
    print(f"  Input  : {COHORT_CSV}")
    print(f"  Output : {OUTPUT_DIR}")
    print("=" * 70 + "\n")

    if not os.path.exists(COHORT_CSV):
        print(f"  [ERROR] Cohort data not found: {COHORT_CSV}")
        return

    fig = build_figure(args.offset)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}_delta{args.offset}min")
    save_figure(fig, stem)
    print(f"\n  Saved → {stem}.pdf / .png / .svg")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
