"""
src/visualization/episode_plots.py
===================================

Publication-ready figures for SLH/LLH episode classification results.

Functions
---------
plot_funnel_figure
    Two-panel figure: a horizontal waterfall funnel showing episode counts
    after each classification criterion (C1–C6 → SLH), plus a stacked bar
    breaking down SLH episodes by their OR-crossing pattern (before-only,
    after-only, both).

Episode-config constants (thresholds, window widths) are imported from
``src.episodes``; colours from ``src.visualization.colors``; font-size and
save helpers from ``src.visualization.style``.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from src.episodes import (
    H_CONTEXT,
    D_MIN_MIN,
    D_MAX_MIN,
    MIN_DATA_COMPLETENESS,
    CONTEXT_WIN_MIN,
)
from src.visualization.colors import CROSSING_COLORS, FUNNEL_COLORS
from src.visualization.style import FONT_SIZE_BASE, apply_publication_style, save_figure
from src.utils.unit_conversion import format_glucose

def plot_funnel_figure(funnel_stats: dict, patient_id: str, output_path: str):
    """
    Generate waterfall funnel figure showing episode filtering through criteria.
    
    Parameters
    ----------
    funnel_stats : dict
        Statistics from classify_episodes_sequential
    patient_id : str
        Patient identifier for title
    output_path : str
        Output file path (without extension)
    """
    apply_publication_style()
    
    # Calculate reference for percentages
    n_ref = max(funnel_stats["n_c1"], 1)
    
    # Build waterfall steps
    steps = [
        (f"C1  Duration [{D_MIN_MIN}–{D_MAX_MIN} min]", 
         funnel_stats["n_c1"], FUNNEL_COLORS["C1"]),
        (f"C2  + Data completeness ≥{MIN_DATA_COMPLETENESS*100:.0f}%", 
         funnel_stats["n_c2"], FUNNEL_COLORS["C2"]),
        ("C3  + Daytime (06:00–22:00)", 
         funnel_stats["n_c3"], FUNNEL_COLORS["C3"]),
        (f"C4  + No hyperglycemia {CONTEXT_WIN_MIN} min before", 
         funnel_stats["n_c4"], FUNNEL_COLORS["C4"]),
        (f"C5  + No hyperglycemia {CONTEXT_WIN_MIN} min after", 
         funnel_stats["n_c5"], FUNNEL_COLORS["C5"]),
        (f"SLH  + OR-crossing ≤{format_glucose(H_CONTEXT)}", 
         funnel_stats["n_slh"], FUNNEL_COLORS["SLH"]),
    ]
    
    labels = [s[0] for s in steps]
    values = [s[1] for s in steps]
    colors = [s[2] for s in steps]
    deltas = [None] + [values[i-1] - values[i] for i in range(1, len(values))]
    
    # Create figure layout
    fig, (ax_A, ax_B) = plt.subplots(
        1, 2,
        figsize=(7.24, 3.20),
        gridspec_kw={"width_ratios": [2.2, 1.0], "wspace": 0.50},
    )
    fig.subplots_adjust(left=0.35, right=0.97, top=0.91, bottom=0.13)
    
    # Panel A: Waterfall funnel
    y = np.arange(len(steps))[::-1]
    x_max = max(values) * 1.42 if max(values) > 0 else 10
    ax_A.set_xlim(0, x_max)
    
    bars = ax_A.barh(y, values, color=colors, height=0.52,
                     edgecolor="white", linewidth=0.4, zorder=3)
    
    # Annotate bars with counts and percentages
    for i, (bar, val, pct, delta) in enumerate(
        zip(bars, values, [v / n_ref * 100 for v in values], deltas)):
        
        # Count and percentage on right
        ax_A.text(bar.get_width() + x_max * 0.012,
                  bar.get_y() + bar.get_height() / 2,
                  f"n = {val}  ({pct:.0f}%)",
                  va="center", fontsize=7, color="#333333")
        
        # Exclusion count above bar
        if delta is not None and delta > 0:
            ax_A.text(bar.get_width() / 2,
                      bar.get_y() + bar.get_height() + 0.07,
                      f"−{delta}",
                      va="bottom", ha="center", fontsize=6,
                      color="#B5121B", fontweight="bold")
    
    # Styling
    ax_A.set_yticks(y)
    ax_A.set_yticklabels(labels, fontsize=7.5)
    ax_A.spines["top"].set_visible(False)
    ax_A.spines["right"].set_visible(False)
    ax_A.spines["left"].set_linewidth(0.8)
    ax_A.spines["bottom"].set_linewidth(0.8)
    ax_A.tick_params(length=3, width=0.8, labelsize=FONT_SIZE_BASE)
    ax_A.set_xlabel("Number of episodes", fontsize=FONT_SIZE_BASE + 1, labelpad=4)
    ax_A.set_title(f"A  Classification funnel — {patient_id}", 
                   fontsize=FONT_SIZE_BASE + 1, fontweight="bold",
                   pad=5, loc="left")
    
    # Panel B: SLH crossing breakdown
    n_before = funnel_stats["n_before_only"]
    n_after = funnel_stats["n_after_only"]
    n_both = funnel_stats["n_both"]
    
    components = [n_before, n_after, n_both]
    colors_comp = [
        CROSSING_COLORS["before_only"],
        CROSSING_COLORS["after_only"],
        CROSSING_COLORS["both"]
    ]
    labels_comp = [
        f"Before only  (n={n_before})",
        f"After only   (n={n_after})",
        f"Both         (n={n_both})"
    ]
    
    # Stacked bar
    bottom = 0.0
    for val, col, lbl in zip(components, colors_comp, labels_comp):
        ax_B.bar([0], [val], bottom=[bottom], color=col,
                 edgecolor="white", linewidth=0.4, width=0.52,
                 zorder=3, label=lbl)
        if val > 0:
            ax_B.text(0, bottom + val / 2, str(val),
                      ha="center", va="center",
                      fontsize=7.5, fontweight="bold", color="white")
        bottom += val
    
    # Styling
    ax_B.set_xticks([])
    ax_B.set_xlim(-0.55, 0.55)
    ax_B.spines["bottom"].set_visible(False)
    ax_B.spines["top"].set_visible(False)
    ax_B.spines["right"].set_visible(False)
    ax_B.spines["left"].set_linewidth(0.8)
    ax_B.tick_params(axis="y", length=3, width=0.8, labelsize=FONT_SIZE_BASE)
    ax_B.set_ylabel("SLH episodes", fontsize=FONT_SIZE_BASE + 1, labelpad=4)
    ax_B.set_title("B  OR-crossing\nbreakdown", 
                   fontsize=FONT_SIZE_BASE + 1, fontweight="bold",
                   pad=5, loc="left")
    ax_B.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax_B.legend(fontsize=6.5, frameon=False,
                loc="lower center", handlelength=1.0,
                bbox_to_anchor=(0.5, -0.28), ncol=1)
    
    # Save figure
    save_figure(fig, output_path)
    
    return fig

