"""
Generate Figure 5: Counterfactual Distribution and Correlation Analysis
========================================================================
Renumbered from "Figure 4 bis" now that Figure 4 is the counterfactual
construction/sensitivity figure (generate_figure_4_counterfactual_construction.py).

Five-panel figure combining:
- Row 1: Histograms for ΔTIR and ΔMedian (full-day vs daytime with patient
  markers). The median panel (B) uses fixed 1 mg/dL bins (aligned so that 0 is
  a bin edge) rather than a bin count, so that each bar is one mg/dL of change
  in median glucose; the TIR panel (A) keeps a fixed number of bins.
  Bins stay 1 mg/dL wide when the figure is drawn in mmol/L (the same physical
  width, ~0.056 mmol/L), so the two versions show identical bars.
- Row 2: Cohort-wide distribution of the max |Delta median AGP(t)| between
  original and counterfactual signals (C), and of the time of day at which
  that maximum occurs (D) -- same statistic as
  scripts/counterfactual/sensitivity_analysis.py (compute_max_median_diff),
  replacing the previous ΔTIR/ΔMedian vs n_SLH scatter plots.
- Row 3: Spearman correlation matrix (CGM metrics, phenotype, biomarkers,
  n_SLH), reusing the data and computation from
  scripts/paper_figures/generate_figure_5_slh_determinants.py, with the
  upper triangle masked out (the matrix is symmetric, so showing both halves
  only repeats the same values).

ACCESSIBILITY FEATURES:
- Hatching patterns distinguish Full 24h (solid) vs Daytime (diagonal hatch)
- Different marker shapes for representative patients
- Enhanced legend showing patterns
- Wong colorblind-safe palette

Layout:
    A  B     (2 columns: ΔTIR / ΔMedian histograms)
    C  D     (2 columns: max |Δmedian| distribution / time-of-day distribution)
    E        (full width: masked Spearman correlation matrix)

Usage:
    python3 -m scripts.paper_figures.generate_figure_5_counterfactual_distribution

Authors: Guillaume Deside
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy import stats

# Import configuration
from src.utils.patient_selection import resolve_patient_id, resolve_patient_pair
from src.config import BASE_PATH, DATA_PATH, SENSITIVITY_PATH

# Import visualization
from src.visualization.style import apply_publication_style, save_figure, style_axes, FONT_SIZE_BASE
from src.visualization.colors import DIVERGING
from src.visualization.glucose_axis import apply_glucose_axis
from src.utils.unit_conversion import (
    format_glucose,
    glucose_unit_label,
    to_display,
)

# Reuse the correlation-matrix data/computation from Figure 5 (SLH determinants)
# rather than duplicating it -- only the plotting function is overridden below
# to mask the (redundant, symmetric) upper triangle.
from scripts.paper_figures.generate_figure_5_slh_determinants import (
    build_dataset,
    compute_correlation_stats,
    compute_mutual_information,
    ALL_COLS,
    CGM_COLS,
    PHENOTYPE_COLS,
    DOMAIN_OF,
    DOMAIN_COLORS,
    DOMAIN_TITLES,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# ── Selected offset ──────────────────────────────────────────────────────────
SELECTED_OFFSET = 30

# ── Representative patients ──────────────────────────────────────────────────
PATIENT_IDS = list(resolve_patient_pair())   # [high-SLH, low-SLH] (CGM_PATIENT_HIGH / CGM_PATIENT_LOW, else automatic)

# ── Input paths ──────────────────────────────────────────────────────────────
# NOTE: previously pointed at "Sensitivity_LinInterp_Modular", a directory
# that no longer exists -- scripts/counterfactual/sensitivity_analysis.py
# writes to SENSITIVITY_PATH ("Sensitivity_Analysis").
DATA_DIR = os.path.join(SENSITIVITY_PATH, "Data")
COHORT_CSV = os.path.join(DATA_DIR, "cohort_sensitivity.csv")

EPISODES_SUMMARY_CSV = os.path.join(DATA_PATH, "Episodes", "episodes_summary.csv")

# ── Output paths ─────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(BASE_PATH, "Figures", "Manuscript_Figures")
OUTPUT_NAME = "Figure5_Counterfactual_Distribution_Correlation"

# ── Colors (matching Figure 4) ───────────────────────────────────────────────
COLOR_FULLTIME = "#003865"      # Navy blue for full-time
COLOR_DAYTIME = "#C9A84C"       # Warm gold for daytime
COLOR_MEDIAN_REF = "#B5121B"    # Cohort-median reference line (row 2)

# Patient markers
PATIENT_COLORS = dict(zip(PATIENT_IDS, ["#E69F00", "#CC79A7"]))   # orange, purple

PATIENT_MARKERS = dict(zip(PATIENT_IDS, ["D", "s"]))              # diamond, square

# ── Histogram styling ────────────────────────────────────────────────────────
HIST_BINS = 30            # Row 1, panel A (TIR)
HIST_BIN_WIDTH_MEDIAN = 1.0   # Row 1, panel B (median) -- 1 mg/dL bins, see module docstring
                              # (always mg/dL: data and bins stay canonical)
HIST_BINS_MAX_DIFF = 20   # Row 2, panel C -- matches the notebook's Part F
HIST_ALPHA = 0.65
HIST_EDGE_WIDTH = 1.2
HIST_EDGE_COLOR_FULL = "#002244"
HIST_EDGE_COLOR_DAY = "#A08030"
HATCH_PATTERN = '///'           # Diagonal hatch for daytime

# Patient marker on histogram - REDUCED SIZE
PATIENT_VLINE_WIDTH = 1.8       # Reduced from 2.5
PATIENT_VLINE_ALPHA = 0.85

# ── Reference lines ──────────────────────────────────────────────────────────
ZERO_LINE_COLOR = "#888888"
ZERO_LINE_STYLE = ":"
ZERO_LINE_WIDTH = 0.9
ZERO_LINE_ALPHA = 0.7

# ── Correlation matrix (row 3) ───────────────────────────────────────────────
CORR_CELL_FONTSIZE = FONT_SIZE_BASE - 1   # rho values inside the cells
CORR_TICK_FONTSIZE = FONT_SIZE_BASE - 1   # variable labels on both axes

# ── Figure parameters ────────────────────────────────────────────────────────
# Widened relative to the previous 2-row version (7.24in) to give the
# correlation matrix (row 3) enough room for its 15x15 labels.
FIGURE_WIDTH = 8.6
FIGURE_HEIGHT = 12.5


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data():
    """Load all required datasets."""
    # Cohort sensitivity (includes max_median_diff_abs / _time_hour, added by
    # scripts/counterfactual/sensitivity_analysis.py)
    df_cohort = pd.read_csv(COHORT_CSV)

    # Episodes for scatter
    df_episodes = pd.read_csv(EPISODES_SUMMARY_CSV) if os.path.exists(EPISODES_SUMMARY_CSV) else None

    # Merge episode counts
    if df_episodes is not None:
        df_cohort = df_cohort.merge(
            df_episodes[["patient_id", "n_SLH"]],
            on="patient_id",
            how="left"
        )

    return df_cohort


# =============================================================================
# ROW 1: HISTOGRAMS WITH PATIENT MARKERS
# =============================================================================

def plot_distribution_histogram(ax, df, metric, offset, patient_ids, show_legend=False,
                                n_bins=HIST_BINS, bin_width=None, title=None):
    """
    Plot histogram comparing full-time and daytime distributions with patient markers.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis
    df : pd.DataFrame
        Cohort data
    metric : str
        Metric name ('TIR70-180' or 'median')
    offset : int
        Selected offset
    patient_ids : list
        Patient IDs to mark
    show_legend : bool
        Whether to show legend on this panel
    n_bins : int
        Number of histogram bins (see HIST_BINS). Ignored when bin_width is given.
    bin_width : float or None
        Fixed bin width in the metric's own stored unit (mg/dL for the median
        panel, see HIST_BIN_WIDTH_MEDIAN) — never in the display unit, since
        the data is binned before the axis is relabelled. Bins are aligned on
        multiples of the width so that 0 is a bin edge. Takes precedence over
        n_bins.
    title : str or None
        Panel subtitle drawn above the axes.
    """
    df_offset = df[df["offset"] == offset]

    # Get data
    col_full = f"delta_{metric}"
    col_day = f"daytime_delta_{metric}"

    vals_full = df_offset[col_full].dropna().values
    vals_day = df_offset[col_day].dropna().values

    # Determine common bin range
    all_vals = np.concatenate([vals_full, vals_day])
    bin_range = (all_vals.min(), all_vals.max())
    if bin_width is not None:
        # Fixed-width bins aligned on multiples of bin_width (0 is an edge)
        start = np.floor(bin_range[0] / bin_width) * bin_width
        stop = np.ceil(bin_range[1] / bin_width) * bin_width + 0.5 * bin_width
        bins = np.arange(start, stop, bin_width)
    else:
        bins = np.linspace(bin_range[0], bin_range[1], n_bins)

    # Plot full-time histogram (SOLID - no hatch)
    ax.hist(vals_full, bins=bins, alpha=HIST_ALPHA,
            color=COLOR_FULLTIME, edgecolor=HIST_EDGE_COLOR_FULL,
            linewidth=HIST_EDGE_WIDTH, label="Full 24h", zorder=2)

    # Plot daytime histogram (HATCHED)
    ax.hist(vals_day, bins=bins, alpha=HIST_ALPHA,
            color=COLOR_DAYTIME, edgecolor=HIST_EDGE_COLOR_DAY,
            linewidth=HIST_EDGE_WIDTH, hatch=HATCH_PATTERN,
            label="Daytime (06:00-22:00)", zorder=3)

    # Mark representative patients (using daytime data)
    for patient_id in patient_ids:
        patient_data = df_offset[df_offset["patient_id"] == patient_id]
        if len(patient_data) > 0:
            delta_val = patient_data[col_day].iloc[0]
            if not pd.isna(delta_val):
                color = PATIENT_COLORS.get(patient_id, "#999999")
                ax.axvline(delta_val, color=color, linewidth=PATIENT_VLINE_WIDTH,
                          alpha=PATIENT_VLINE_ALPHA, linestyle="--", zorder=10,
                          label=patient_id)

    # Zero reference line
    ax.axvline(0, color=ZERO_LINE_COLOR, linestyle=ZERO_LINE_STYLE,
              linewidth=ZERO_LINE_WIDTH, alpha=ZERO_LINE_ALPHA, zorder=1)

    # Formatting
    metric_labels = {
        "TIR70-180": "ΔTIR 70-180 (%)",
        "median": f"ΔMedian glucose ({glucose_unit_label()})"
    }
    ax.set_xlabel(metric_labels.get(metric, f"Δ{metric}"), fontsize=8, labelpad=3)

    # Δmedian is a glucose difference: relabel its ticks, leave the data alone
    if metric == "median":
        apply_glucose_axis(ax, axis="x", set_label=False)
    ax.set_ylabel("Number of patients", fontsize=8, labelpad=3)
    if title:
        ax.set_title(title, fontsize=FONT_SIZE_BASE, fontweight="bold",
                     loc="left", pad=6)

    # Grid and style
    ax.grid(True, axis='y', alpha=0.25, linewidth=0.5, zorder=0)
    style_axes(ax)
    ax.tick_params(length=3, width=0.8, labelsize=7)

    # Legend - only show if requested
    if show_legend:
        ax.legend(fontsize=6.5, loc='best', framealpha=0.95)


# =============================================================================
# ROW 2: MAX |DELTA MEDIAN AGP(t)| AND TIME-OF-DAY DISTRIBUTIONS
# (same statistic as compute_max_median_diff in
#  scripts/counterfactual/sensitivity_analysis.py)
# =============================================================================

def plot_max_diff_histogram(ax, df, offset, patient_ids, show_legend=False,
                            n_bins=HIST_BINS_MAX_DIFF, title=None):
    """
    Cohort-wide distribution of the largest |Delta median AGP(t)| between the
    original and counterfactual signals, at the given offset.
    """
    df_offset = df[df["offset"] == offset]
    values = df_offset["max_median_diff_abs"].dropna().values

    ax.hist(values, bins=n_bins, color=COLOR_DAYTIME, edgecolor=HIST_EDGE_COLOR_DAY,
            linewidth=HIST_EDGE_WIDTH, alpha=HIST_ALPHA, zorder=2)

    median_v = float(np.median(values))
    ax.axvline(median_v, color=COLOR_MEDIAN_REF, linewidth=1.4, linestyle="--",
              zorder=5, label=f"Median = {format_glucose(median_v)}")

    for patient_id in patient_ids:
        patient_data = df_offset[df_offset["patient_id"] == patient_id]
        if len(patient_data) > 0:
            val = patient_data["max_median_diff_abs"].iloc[0]
            if not pd.isna(val):
                color = PATIENT_COLORS.get(patient_id, "#999999")
                ax.axvline(val, color=color, linewidth=PATIENT_VLINE_WIDTH,
                          alpha=PATIENT_VLINE_ALPHA, linestyle=":", zorder=10,
                          label=patient_id)

    ax.set_xlabel(f"Max |Δ median AGP(t)| ({glucose_unit_label()})",
                  fontsize=8, labelpad=3)
    apply_glucose_axis(ax, axis="x", set_label=False)
    ax.set_ylabel("Number of patients", fontsize=8, labelpad=3)
    if title:
        ax.set_title(title, fontsize=FONT_SIZE_BASE, fontweight="bold",
                     loc="left", pad=6)

    ax.grid(True, axis='y', alpha=0.25, linewidth=0.5, zorder=0)
    style_axes(ax)
    ax.tick_params(length=3, width=0.8, labelsize=7)

    if show_legend:
        ax.legend(fontsize=6.5, loc='best', framealpha=0.95)


def plot_time_of_max_diff_histogram(ax, df, offset, patient_ids, show_legend=False,
                                    title=None):
    """
    Cohort-wide distribution of the time of day at which |Delta median AGP(t)|
    is largest, at the given offset. Hourly bins, matching the notebook.
    """
    df_offset = df[df["offset"] == offset]
    hours = df_offset["max_median_diff_time_hour"].dropna().values
    bins = np.arange(0, 24.5, 1.0)

    ax.hist(hours, bins=bins, color=COLOR_DAYTIME, edgecolor=HIST_EDGE_COLOR_DAY,
            linewidth=HIST_EDGE_WIDTH, alpha=HIST_ALPHA, zorder=2)

    for patient_id in patient_ids:
        patient_data = df_offset[df_offset["patient_id"] == patient_id]
        if len(patient_data) > 0:
            val = patient_data["max_median_diff_time_hour"].iloc[0]
            if not pd.isna(val):
                color = PATIENT_COLORS.get(patient_id, "#999999")
                ax.axvline(val, color=color, linewidth=PATIENT_VLINE_WIDTH,
                          alpha=PATIENT_VLINE_ALPHA, linestyle=":", zorder=10,
                          label=patient_id)

    ax.set_xlabel("Time of day (hour)", fontsize=8, labelpad=3)
    ax.set_ylabel("Number of patients", fontsize=8, labelpad=3)
    if title:
        ax.set_title(title, fontsize=FONT_SIZE_BASE, fontweight="bold",
                     loc="left", pad=6)
    ax.set_xticks(range(6, 25, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(6, 25, 3)], fontsize=6.5)
    ax.set_xlim(6, 24)

    ax.grid(True, axis='y', alpha=0.25, linewidth=0.5, zorder=0)
    style_axes(ax)
    ax.tick_params(length=3, width=0.8, labelsize=7)

    if show_legend:
        ax.legend(fontsize=6.5, loc='best', framealpha=0.95)


# =============================================================================
# ROW 3: SPEARMAN CORRELATION MATRIX (LOWER TRIANGLE ONLY)
# =============================================================================

def plot_correlation_matrix_masked(ax, rho_mat: np.ndarray, p_mat: np.ndarray,
                                   cols: list, labels: list) -> None:
    """
    Same panel as plot_correlation_matrix in generate_figure_5_slh_determinants.py,
    except the upper triangle is masked out: the matrix is symmetric (rho[i,j]
    == rho[j,i]), so showing both halves only repeats the same values.
    """
    n = len(cols)
    mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    rho_masked = np.ma.masked_where(mask_upper, rho_mat)

    im = ax.imshow(rho_masked, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, label="Spearman rho", shrink=0.75, pad=0.015)
    cbar.ax.tick_params(labelsize=FONT_SIZE_BASE - 2)
    cbar.set_label("Spearman rho", fontsize=FONT_SIZE_BASE - 1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=CORR_TICK_FONTSIZE)
    ax.set_yticklabels(labels, fontsize=CORR_TICK_FONTSIZE)

    tick_colors = [DOMAIN_COLORS[DOMAIN_OF[c]] for c in ALL_COLS] + ["black"]
    for tick, color in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(color)
        tick.set_fontweight("bold")
    for tick, color in zip(ax.get_yticklabels(), tick_colors):
        tick.set_color(color)
        tick.set_fontweight("bold")

    # Only annotate the visible (lower-triangle + diagonal) cells.
    for i in range(n):
        for j in range(n):
            if j > i:
                continue
            sig = "*" if p_mat[i, j] < 0.05 else ""
            txt = f"{rho_mat[i, j]:.2f}{sig}"
            color = "white" if abs(rho_mat[i, j]) > 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=CORR_CELL_FONTSIZE, fontweight="bold", color=color)

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
        "Spearman correlation matrix (lower triangle)\n"
        "(* p<0.05  |  dotted = domain boundary  |  dashed = SLH count)",
        fontsize=FONT_SIZE_BASE, fontweight="bold", loc="left", pad=6,
    )


# =============================================================================
# ROW 3, PANEL F: MUTUAL INFORMATION WITH THE SLH COUNT
# =============================================================================

def plot_mutual_information_panel(ax, mi_df: pd.DataFrame) -> None:
    """Mutual information between each baseline variable and the SLH count.

    Companion to the Spearman matrix in panel E: rho only captures monotonic
    association, so a variable can carry information about the SLH count while
    ranking low on rho. Bars are sorted by MI (largest at the top) rather than
    kept in the matrix's domain order, because the ranking is the point of the
    panel; the domain is carried by the bar colour instead, using the same
    three colours as panel E's axis labels and domain headers.

    Adapted from `plot_mutual_information` in
    generate_figure_5_slh_determinants.py — same bars and colours, but sorted
    descending, without the redundant domain legend (panel E already labels
    the three domains, in the same colours, immediately to the left) and with
    this figure's panel letter.
    """
    mi_sorted = mi_df.sort_values("MI", ascending=True)  # barh draws bottom-up
    bar_colors = [DOMAIN_COLORS[d] for d in mi_sorted["domain"]]

    bars = ax.barh(mi_sorted["label"], mi_sorted["MI"], color=bar_colors,
                   edgecolor="white", linewidth=0.4, alpha=0.88, zorder=2)

    x_max = float(mi_sorted["MI"].max())
    for bar, val in zip(bars, mi_sorted["MI"]):
        ax.text(val + x_max * 0.03, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=FONT_SIZE_BASE - 2.5,
                color="#333333")
    ax.set_xlim(0, x_max * 1.30)  # headroom for the value labels

    ax.tick_params(axis="y", labelsize=CORR_TICK_FONTSIZE, length=0)
    ax.tick_params(axis="x", labelsize=FONT_SIZE_BASE - 2)
    for tick, domain in zip(ax.get_yticklabels(), mi_sorted["domain"]):
        tick.set_color(DOMAIN_COLORS[domain])
        tick.set_fontweight("bold")

    ax.set_xlabel("Mutual information with\nnumber of SLH episodes",
                  fontsize=FONT_SIZE_BASE - 1)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.5, zorder=0)
    style_axes(ax)
    ax.set_title("Mutual information", fontweight="bold",
                 loc="left", pad=6, fontsize=FONT_SIZE_BASE)


# =============================================================================
# SHARED HELPERS
# =============================================================================

def add_panel_label(ax, label, x=-0.15, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold")


# =============================================================================
# MAIN FIGURE GENERATION
# =============================================================================

def generate_figure_5():
    """Generate Figure 5: distribution + correlation analysis."""
    apply_publication_style()

    # Load sensitivity/distribution data
    print("  Loading cohort sensitivity data...")
    df = load_all_data()
    n_patients = df["patient_id"].nunique()

    print(f"    Patients: {n_patients}")
    print(f"    Selected offset: {SELECTED_OFFSET} min")
    print(f"    Representative patients: {PATIENT_IDS}")

    # Load phenotype/biomarker data and compute the correlation matrix
    # (reused from generate_figure_5_slh_determinants.py)
    print("\n  Loading and merging CGM + phenotype + biomarker data...")
    df_corr = build_dataset(apply_filter=True)
    print("  Computing Spearman correlation matrix...")
    rho_mat, p_mat, cols, labels, _ = compute_correlation_stats(df_corr)

    print("  Computing mutual information...")
    mi_df = compute_mutual_information(df_corr)

    # Create figure with a 3-row grid: two 2-column rows, then a split row 3
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    gs = gridspec.GridSpec(3, 2, figure=fig,
                          height_ratios=[1.0, 1.0, 1.9],
                          hspace=0.55, wspace=0.32,
                          left=0.09, right=0.94, top=0.97, bottom=0.055)

    # Row 1: Histograms
    ax_a = fig.add_subplot(gs[0, 0])  # TIR histogram
    ax_b = fig.add_subplot(gs[0, 1])  # Median histogram

    # Row 2: Max |Delta median| and time-of-day distributions
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Row 3: correlation matrix (E) beside mutual information (F). The matrix
    # keeps most of the width -- it carries a 15x15 grid of labelled cells,
    # whereas F is 14 bars whose scale is short. wspace is generous because
    # E's colourbar and F's variable labels both sit in the gap between them.
    row3_gs = gs[2, :].subgridspec(1, 2, width_ratios=[2.35, 1.0], wspace=0.62)
    ax_e = fig.add_subplot(row3_gs[0, 0])
    ax_f = fig.add_subplot(row3_gs[0, 1])

    # ── Panel A: TIR histogram ───────────────────────────────────────────────
    print("  Plotting Panel A: TIR histogram...")
    plot_distribution_histogram(ax_a, df, "TIR70-180", SELECTED_OFFSET, PATIENT_IDS,
                               show_legend=True, n_bins=HIST_BINS,
                               title=f"Change in time in range "
                                     f"{format_glucose(70, with_unit=False)}-"
                                     f"{format_glucose(180)}")
    ax_a.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    add_panel_label(ax_a, "A")

    # ── Panel B: Median histogram (1 mg/dL bins, see module docstring) ──────
    print("  Plotting Panel B: Median histogram...")
    plot_distribution_histogram(ax_b, df, "median", SELECTED_OFFSET, PATIENT_IDS,
                               show_legend=False,
                               bin_width=HIST_BIN_WIDTH_MEDIAN,
                               title=f"Change in median glucose "
                                     f"({to_display(HIST_BIN_WIDTH_MEDIAN):.2g} "
                                     f"{glucose_unit_label()} bins)")
    add_panel_label(ax_b, "B")

    # ── Panel C: Max |Delta median AGP(t)| distribution ──────────────────────
    print("  Plotting Panel C: Max |Δ median| distribution...")
    plot_max_diff_histogram(ax_c, df, SELECTED_OFFSET, PATIENT_IDS, show_legend=True,
                            title="Largest deviation of the median AGP profile")
    add_panel_label(ax_c, "C")

    # ── Panel D: Time-of-day of the max difference ───────────────────────────
    print("  Plotting Panel D: Time-of-max-difference distribution...")
    plot_time_of_max_diff_histogram(ax_d, df, SELECTED_OFFSET, PATIENT_IDS, show_legend=False,
                                    title="Time of day of the largest deviation")
    add_panel_label(ax_d, "D")

    # ── Panel E: Correlation matrix (lower triangle only) ────────────────────
    print("  Plotting Panel E: Correlation matrix (lower triangle)...")
    plot_correlation_matrix_masked(ax_e, rho_mat, p_mat, cols, labels)
    add_panel_label(ax_e, "E", x=-0.02, y=1.20)

    # ── Panel F: Mutual information with the SLH count ───────────────────────
    print("  Plotting Panel F: Mutual information...")
    plot_mutual_information_panel(ax_f, mi_df)
    add_panel_label(ax_f, "F", x=-0.42, y=1.06)

    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution pipeline."""
    print("=" * 70)
    print("  Figure 5: Counterfactual Distribution and Correlation Analysis")
    print("=" * 70)
    print(f"  Input:  {COHORT_CSV}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Selected δ: {SELECTED_OFFSET} min")
    print(f"  Patients: {PATIENT_IDS}")
    print("=" * 70 + "\n")

    # Check inputs
    if not os.path.exists(COHORT_CSV):
        print(f"  [ERROR] Cohort data not found: {COHORT_CSV}")
        return

    # Generate figure
    fig = generate_figure_5()

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    save_figure(fig, output_path)

    print("\n" + "=" * 70)
    print(f"  Figure saved: {output_path}.pdf")
    print(f"                {output_path}.svg")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
