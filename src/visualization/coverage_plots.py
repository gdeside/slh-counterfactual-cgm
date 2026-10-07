"""
src/visualization/coverage_plots.py
=====================================

Panel-level plotting functions for CGM data-coverage figures.

Functions
---------
draw_histogram_panel
    Histogram of per-patient coverage percentages with median annotation
    and threshold line.
draw_strip_plot_panel
    Strip plot (jittered scatter + IQR box) for individual coverage values.

Color constants are imported from ``src.visualization.colors``; coverage
thresholds and sampling parameters from ``src.config``.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator, MaxNLocator

from src.config import MIN_COVERAGE_PCT, REQUIRED_SAMPLING_MIN, RANDOM_SEED

from src.visualization.colors import COLOR_PRIMARY, COLOR_THRESHOLD, COLOR_ANNOTATION, COLOR_BOX_BG
from src.visualization.style import style_axes

# Histogram parameters
HISTOGRAM_BIN_WIDTH = 2.0     # Bin width in percentage points
HISTOGRAM_START = 70.0        # Start of histogram bins
HISTOGRAM_END = 101.5         # End of histogram bins

STRIP_PLOT_JITTER = 0.14      # Horizontal jitter for strip plot points
STRIP_PLOT_POINT_SIZE = 8     # Point size for individual measurements
STRIP_PLOT_BOX_WIDTH = 0.22   # Width of IQR box

ALPHA_HISTOGRAM   = 0.65        # Histogram bar transparency
ALPHA_IQR_BAND    = 0.10        # IQR shading transparency
ALPHA_STRIP_POINT = 0.45        # Strip plot point transparency
ALPHA_IQR_BOX     = 0.25        # IQR box transparency

def draw_histogram_panel(ax, coverage_data, stats_dict):
    """
    Draw histogram panel showing coverage distribution.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on
    coverage_data : np.ndarray
        Coverage values
    stats_dict : dict
        Summary statistics from compute_summary_statistics()
    """
    # Create bins
    bins = np.arange(HISTOGRAM_START, HISTOGRAM_END, HISTOGRAM_BIN_WIDTH)
    
    # Draw histogram
    ax.hist(
        coverage_data, 
        bins=bins, 
        color=COLOR_PRIMARY, 
        alpha=ALPHA_HISTOGRAM,
        label=f"n = {stats_dict['n']}",
        zorder=3
    )
    
    # Median line
    ax.axvline(
        stats_dict["median"], 
        color=COLOR_PRIMARY, 
        lw=1.2, 
        ls="--", 
        zorder=4
    )
    
    # IQR shaded band
    ax.axvspan(
        stats_dict["q25"], 
        stats_dict["q75"], 
        color=COLOR_PRIMARY, 
        alpha=ALPHA_IQR_BAND, 
        zorder=2
    )
    
    # Threshold line
    ax.axvline(
        MIN_COVERAGE_PCT, 
        color=COLOR_THRESHOLD, 
        lw=0.9,
        ls=":", 
        zorder=5, 
        label=f"{MIN_COVERAGE_PCT:.0f} % threshold"
    )
    
    # Styling
    style_axes(
        ax,
        xlabel="Coverage (%)",
        ylabel="Number of participants",
        title="A  Coverage distribution"
    )
    
    # Axis limits and ticks
    ax.set_xlim(MIN_COVERAGE_PCT - 1, 101)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    
    # Legend
    ax.legend(
        fontsize=6.5, 
        frameon=False,
        loc="upper center",
        handlelength=1.2, 
        handleheight=0.85,
    )
    
    # Median annotation
    y_max = ax.get_ylim()[1]
    ax.text(
        stats_dict["median"] + 0.3, 
        y_max * 1.02,
        f"median {stats_dict['median']:.1f} %",
        color=COLOR_PRIMARY, 
        fontsize=6, 
        va="top",
        style="italic"
    )


# ─────────────────────────────────────────────────────────────────────────────
# STRIP PLOT PANEL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def draw_strip_plot_panel(ax, coverage_data, stats_dict):
    """
    Draw strip plot panel showing individual coverage values.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on
    coverage_data : np.ndarray
        Coverage values
    stats_dict : dict
        Summary statistics from compute_summary_statistics()
    """
    np.random.seed(RANDOM_SEED)
    
    x_position = 0  # Single column
    
    # Jittered scatter points
    jitter = np.random.uniform(
        -STRIP_PLOT_JITTER, 
        STRIP_PLOT_JITTER, 
        size=len(coverage_data)
    )
    ax.scatter(
        np.full(len(coverage_data), x_position) + jitter,
        coverage_data,
        color=COLOR_PRIMARY, 
        alpha=ALPHA_STRIP_POINT, 
        s=STRIP_PLOT_POINT_SIZE, 
        lw=0, 
        zorder=3,
    )
    
    # IQR box
    box_rect = plt.Rectangle(
        (x_position - STRIP_PLOT_BOX_WIDTH / 2, stats_dict["q25"]),
        STRIP_PLOT_BOX_WIDTH, 
        stats_dict["q75"] - stats_dict["q25"],
        facecolor=COLOR_PRIMARY, 
        alpha=ALPHA_IQR_BOX,
        edgecolor=COLOR_PRIMARY, 
        linewidth=0.8, 
        zorder=4,
    )
    ax.add_patch(box_rect)
    
    # Median line
    ax.plot(
        [x_position - STRIP_PLOT_BOX_WIDTH / 2, 
         x_position + STRIP_PLOT_BOX_WIDTH / 2],
        [stats_dict["median"], stats_dict["median"]],
        color=COLOR_PRIMARY, 
        lw=1.4, 
        zorder=5
    )
    
    # Whiskers (min/max)
    ax.plot(
        [x_position, x_position], 
        [stats_dict["min"], stats_dict["q25"]], 
        color=COLOR_PRIMARY, 
        lw=0.7, 
        ls="--", 
        zorder=4
    )
    ax.plot(
        [x_position, x_position], 
        [stats_dict["q75"], stats_dict["max"]], 
        color=COLOR_PRIMARY, 
        lw=0.7, 
        ls="--", 
        zorder=4
    )
    
    # Mean marker
    ax.scatter(
        [x_position], 
        [stats_dict["mean"]], 
        marker="D", 
        s=18,
        color=COLOR_BOX_BG, 
        edgecolors=COLOR_PRIMARY, 
        lw=0.9, 
        zorder=6
    )
    
    # Threshold line
    ax.axhline(
        MIN_COVERAGE_PCT, 
        color=COLOR_THRESHOLD, 
        lw=0.9, 
        ls=":", 
        zorder=2
    )
    
    # Styling
    ax.set_xticks([x_position])
    ax.set_xticklabels([f"{REQUIRED_SAMPLING_MIN:.0f} min"], fontsize=7.5)
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(MIN_COVERAGE_PCT - 2, 102)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    
    style_axes(
        ax,
        ylabel="Coverage (%)",
        title="B  Individual coverage"
    )
    
    # Legend
    median_line = mlines.Line2D(
        [], [], 
        color=COLOR_ANNOTATION, 
        lw=1.2,
        label="Median"
    )
    mean_marker = mlines.Line2D(
        [], [], 
        marker="D", 
        color=COLOR_BOX_BG,
        markeredgecolor=COLOR_ANNOTATION, 
        markersize=5,
        lw=0, 
        label="Mean"
    )
    ax.legend(
        handles=[median_line, mean_marker],
        fontsize=6, 
        frameon=False, 
        loc="center right",
        handlelength=1.0
    )


# # ─────────────────────────────────────────────────────────────────────────────
# # FIGURE GENERATION
# # ─────────────────────────────────────────────────────────────────────────────

# def generate_coverage_figure(df: pd.DataFrame, output_directory: str):
#     """
#     Generate two-panel coverage figure and save to disk.
    
#     Parameters
#     ----------
#     df : pd.DataFrame
#         Summary dataframe with coverage data
#     output_directory : str
#         Directory to save figure
#     """
#     apply_publication_style()
    
#     # Extract coverage data
#     coverage_data = df[COL_COVERAGE].values
#     stats = compute_summary_statistics(coverage_data)
    
#     # Create figure
#     fig, (ax_left, ax_right) = plt.subplots(
#         1, 2,
#         figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES),
#         gridspec_kw={"width_ratios": [2.2, 1.0], "wspace": 0.38},
#     )
#     fig.subplots_adjust(left=0.09, right=0.97, top=0.91, bottom=0.16)
    
#     # Draw panels
#     draw_histogram_panel(ax_left, coverage_data, stats)
#     draw_strip_plot_panel(ax_right, coverage_data, stats)
    
#     # Save figure
#     os.makedirs(output_directory, exist_ok=True)
#     for extension in ("pdf", "png"):
#         output_path = os.path.join(
#             output_directory, 
#             f"coverage_distribution.{extension}"
#         )
#         fig.savefig(output_path, dpi=DPI_SAVE, bbox_inches="tight")
#         print(f"Figure saved → {output_path}")
    
#     plt.close(fig)
    
