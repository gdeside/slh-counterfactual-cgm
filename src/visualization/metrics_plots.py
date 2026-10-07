"""
src/visualization/metrics_plots.py
=====================================

Exploratory and publication-ready figures for the 19 CGM metric panel.

Functions
---------
view1_control_overview
    Scatter: mean glycemia vs TIR 70-180, coloured by CV.
view2_hypo_hyper_risk
    Quadrant scatter: TBR <70 vs TAR >180, coloured by TIR 70-180.
view3_diurnal_pattern
    Box-strip plot of median glucose across five time-of-day periods.
view4_variability
    Scatter: CV vs SD, coloured by mean glycemia.
view5_correlation_heatmap
    Annotated Pearson correlation heatmap for all 19 CGM metrics.
plot_distribution_metrics
    Six-panel boxplot summary (supplementary figure S1) organised by
    metric category (central tendency, variability, TIR, hypo, hyper,
    time-of-day).
add_panel_label
    Stamp a bold panel letter on an Axes.

Color constants are imported from ``src.visualization.colors``; output
directory from ``src.config``.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from src.config import FIG_METRICS_DIR
from src.visualization.colors import CV_GRADIENT, TIR_GRADIENT, DIVERGING, MEAN_GRADIENT, CATEGORICAL_5, WONG_COLORS

from src.visualization.style import apply_publication_style, save_figure
from src.visualization.glucose_axis import apply_glucose_axis, apply_glucose_colorbar
from src.utils.unit_conversion import (
    format_glucose,
    glucose_label,
    is_glucose_valued_column,
    to_display,
)

BOX_COLOR = WONG_COLORS['sky_blue']
MEDIAN_COLOR = WONG_COLORS['vermillion']
MEAN_COLOR = WONG_COLORS['orange']

FIG_SIZE    = (7, 5) 
FIG_SIZE_WIDE = (10, 5)         # For wider plots
FIG_SIZE_HEATMAP = (9, 8)       # For correlation matrix
FIG_WIDTH   = 7.0   # Single column width (inches)
FIG_HEIGHT  = 10.0  # Tall multi-panel figure

# ─────────────────────────────────────────────────────────────────────────────
# VIEW 1 — Glycaemic control overview
#   Mean_glycemia vs TIR70180, colour = CV
# ─────────────────────────────────────────────────────────────────────────────

def view1_control_overview(df: pd.DataFrame):
    """
    Mean glycemia (x) vs Time in Range 70-180 % (y), coloured by CV.
    Patients with good control should cluster: low mean, high TIR, low CV.
    """
    print("\n─── View 1: Glycaemic control overview ───")

    stem = os.path.join(FIG_METRICS_DIR, "view1_mean_vs_TIR70180_hue_CV")

    d = df[["patient_id", "Mean_glycemia", "TIR70180", "CV"]].dropna()
    d["CV_pct"] = (d["CV"] * 100).round(1)
    d["TIR70180_pct"] = (d["TIR70180"] * 100).round(1)


    apply_publication_style()

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    sc = ax.scatter(
        d["Mean_glycemia"], d["TIR70180"] * 100,
        c=d["CV_pct"], 
        cmap=CV_GRADIENT,
        s=25, 
        alpha=0.7, 
        linewidths=0.5,
        edgecolors='white',
        rasterized=True,  # For better PDF rendering with many points
    )
    
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("CV (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    apply_glucose_axis(ax, axis="x", quantity="Mean glycemia", fontsize=10)
    ax.set_ylabel("TIR 70–180 (%)", fontsize=10)
    ax.set_title("Glycaemic control: Mean glucose vs TIR", fontsize=11, pad=10)
    
    fig.tight_layout()
    save_figure(fig, stem)



# ─────────────────────────────────────────────────────────────────────────────
# VIEW 2 — Hypo / hyperglycaemia risk profile
#   TBR70 (x) vs TAR180 (y), colour = TIR70180
# ─────────────────────────────────────────────────────────────────────────────

def view2_hypo_hyper_risk(df: pd.DataFrame):
    """
    Quadrant plot: TBR70 (% time below 70) vs TAR180 (% time above 180).
    Patients in the bottom-left have both risks controlled.
    Colour = TIR70180 to confirm complementarity.
    """
    print("\n─── View 2: Hypo / hyperglycaemia risk profile ───")

    stem = os.path.join(FIG_METRICS_DIR, "view2_TBR70_vs_TAR180_hue_TIR70180")

    d = df[["patient_id", "TBR70", "TAR180", "TIR70180"]].dropna()
    d["TBR70_pct"]  = (d["TBR70"]  * 100).round(2)
    d["TAR180_pct"] = (d["TAR180"] * 100).round(2)
    d["TIR_pct"]    = (d["TIR70180"] * 100).round(1)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    sc = ax.scatter(
        d["TBR70_pct"], d["TAR180_pct"],
        c=d["TIR_pct"], 
        cmap=TIR_GRADIENT,
        s=25, 
        alpha=0.7,
        linewidths=0.5, 
        edgecolors='white',
        vmin=0, 
        vmax=100,
        rasterized=True,
    )
    
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("TIR 70–180 (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel(f"TBR <{format_glucose(70)} (%)", fontsize=10)
    ax.set_ylabel(f"TAR >{format_glucose(180)} (%)", fontsize=10)
    ax.set_title("Hypoglycaemia vs hyperglycaemia risk", fontsize=11, pad=10)
    
    fig.tight_layout()
    save_figure(fig, stem)


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 3 — Diurnal glycaemic pattern
#   Strip + box plot: median glucose per time-of-day period
# ─────────────────────────────────────────────────────────────────────────────

def view3_diurnal_pattern(df: pd.DataFrame):
    """
    Distribution of median glucose across the five time-of-day periods.
    Box plot with strip overlay ordered chronologically.
    """
    print("\n─── View 3: Diurnal glycaemic pattern ───")

    stem  = os.path.join(FIG_METRICS_DIR,"view3_diurnal_medians")

    tod_cols   = ["Median_morning", "Median_noon", "Median_afternoon",
                  "Median_evening", "Median_night"]
    period_map = {
        "Median_morning"  : "Morning",
        "Median_noon"     : "Noon",
        "Median_afternoon": "Afternoon",
        "Median_evening"  : "Evening",
        "Median_night"    : "Night",
    }

    # Wide → long
    long = (
        df[["patient_id"] + tod_cols]
        .melt(id_vars="patient_id", var_name="period_raw", value_name="median_glucose")
        .dropna()
    )
    long["period"] = long["period_raw"].map(period_map)
    
    # Preserve chronological order
    ordered = [period_map[c] for c in tod_cols]
    long["period"] = pd.Categorical(long["period"], categories=ordered, ordered=True)

    fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)

    # Box plot with Wong categorical palette
    bp = ax.boxplot(
        [long[long["period"] == p]["median_glucose"].values for p in ordered],
        positions=range(len(ordered)),
        widths=0.6,
        patch_artist=True,
        showfliers=False,  # We'll show points separately
        medianprops=dict(color='black', linewidth=1.5),
        boxprops=dict(linewidth=0.8),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    
    # Color boxes with Wong palette
    for patch, color in zip(bp['boxes'], CATEGORICAL_5):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Add strip plot overlay
    for i, period in enumerate(ordered):
        y_vals = long[long["period"] == period]["median_glucose"].values
        x_vals = np.random.normal(i, 0.08, size=len(y_vals))  # Add jitter
        ax.scatter(x_vals, y_vals, s=8, alpha=0.3, color='black', zorder=3)

    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered, fontsize=9)
    ax.set_xlabel("Time of day", fontsize=10)
    apply_glucose_axis(ax, quantity="Median glucose", fontsize=10)
    ax.set_title("Diurnal glycaemic pattern", fontsize=11, pad=10)
    
    fig.tight_layout()
    save_figure(fig, stem)


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 4 — Variability deep-dive
#   CV (x) vs SD (y), colour = Mean_glycemia, size = CGM_days_measured
# ─────────────────────────────────────────────────────────────────────────────

def view4_variability(df: pd.DataFrame):
    """
    CV (relative variability) vs SD (absolute variability).
    Size of marker encodes monitoring duration (CGM_days_measured).
    Helps separate patients whose high SD is driven by high mean glucose
    from those with genuine glycaemic instability (high CV).
    """
    print("\n─── View 4: Variability deep-dive ───")

    stem  = os.path.join(FIG_METRICS_DIR,"view4_CV_vs_SD_hue_mean_size_days")

    d = df[["patient_id", "CV", "SD", "Mean_glycemia", "CGM_days_measured"]].dropna()
    d["CV_pct"] = (d["CV"] * 100).round(1)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Scale marker sizes
    sizes = 15 + 50 * (d["CGM_days_measured"] - d["CGM_days_measured"].min()) / (
        d["CGM_days_measured"].max() - d["CGM_days_measured"].min() + 1e-9)

    sc = ax.scatter(
        d["CV_pct"], d["SD"],
        c=d["Mean_glycemia"], 
        cmap=MEAN_GRADIENT,
        # s=sizes.clip(15, 80),
        alpha=0.7, 
        linewidths=0.5, 
        edgecolors='white',
        rasterized=True,
    )
    
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    apply_glucose_colorbar(cbar, quantity="Mean glucose", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # # Add size legend
    # legend_days = [20, 35, 45]
    # legend_elements = []
    # for days in legend_days:
    #     sz = 15 + 50 * (days - d["CGM_days_measured"].min()) / (
    #         d["CGM_days_measured"].max() - d["CGM_days_measured"].min() + 1e-9)
    #     legend_elements.append(
    #         plt.scatter([], [], s=sz.clip(15, 80), c='grey', alpha=0.5, 
    #                    edgecolors='black', linewidths=0.5)
    #     )
    
    # ax.legend(legend_elements, [f"{d} days" for d in legend_days],
    #          title="Monitoring duration", fontsize=7, title_fontsize=8,
    #          loc='upper left', frameon=True, framealpha=0.9)

    ax.set_xlabel("CV (%)", fontsize=10)
    apply_glucose_axis(ax, quantity="SD", fontsize=10)
    ax.set_title("Glycaemic variability: CV vs SD", fontsize=11, pad=10)
    
    fig.tight_layout()
    save_figure(fig, stem)


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 5 — Correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────

def view5_correlation_heatmap(df: pd.DataFrame):
    """
    Pearson correlation matrix across all 19 CGM metrics.
    Annotated heatmap with clean styling.
    """
    print("\n─── View 5: Metric correlation heatmap ───")

    stem = os.path.join(FIG_METRICS_DIR,"view5_correlation_heatmap")

    metric_cols = [
        "Mean_glycemia", "Median_glycemia", "Minimum_glycemia", "Maximum_glycemia",
        "SD", "CV",
        "TIR70180", "TIR70140", "TIR60140",
        "TBR70", "TBR54", "TAR180", "TAR250",
        "CGM_days_measured",
        "Median_morning", "Median_noon", "Median_afternoon",
        "Median_evening", "Median_night",
    ]

    corr = df[metric_cols].dropna().corr(method="pearson")

    fig, ax = plt.subplots(figsize=FIG_SIZE_HEATMAP)
    
    # Create mask for upper triangle to reduce visual clutter (optional)
    # mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    sns.heatmap(
        corr,
        cmap=DIVERGING,
        center=0, 
        vmin=-1, 
        vmax=1,
        annot=True, 
        fmt=".2f", 
        annot_kws={"size": 6},
        linewidths=0.5, 
        linecolor='white',
        cbar_kws={
            "shrink": 0.8, 
            "label": "Pearson r",
            "ticks": [-1, -0.5, 0, 0.5, 1]
        },
        square=True,
        ax=ax,
    )
    
    ax.set_title("CGM metric correlations", fontsize=11, pad=12, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    fig.tight_layout()
    save_figure(fig, stem)


def add_panel_label(ax, label, x=-0.15, y=1.05):
    """Add panel label (A, B, C, etc.) to axis."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='right')


def plot_distribution_metrics(df: pd.DataFrame):
    """
    Multi-panel boxplot showing distribution of all 19 CGM metrics across
    the full cohort (before filtering to valid windows).
    
    Panels organized by metric category:
      A. Central tendency (mean, median, min, max)
      B. Variability (SD, CV)
      C. Time-in-range metrics (TIR70180, TIR70140, TIR60140)
      D. Hypoglycaemia (TBR70, TBR54)
      E. Hyperglycaemia (TAR180, TAR250)
      F. Time-of-day medians (5 periods)
    """
    print("\n─── Supplementary Figure S1: Metric distributions ───")
    
    stem = os.path.join(FIG_METRICS_DIR,"suppfig_s1_metric_distributions")
    
    # Define metric groups with panel labels
    metric_groups = [
        ('A', glucose_label('Central tendency'), [
            ('Mean_glycemia', 'Mean'),
            ('Median_glycemia', 'Median'),
            ('Minimum_glycemia', 'Minimum'),
            ('Maximum_glycemia', 'Maximum'),
        ]),
        ('B', 'Variability', [
            ('SD', glucose_label('SD', parentheses=True).replace(' (', '\n(')),
            ('CV', 'CV\n(%)'),
        ]),
        ('C', 'Time in range (%)', [
            ('TIR70180', 'TIR\n70–180'),
            ('TIR70140', 'TIR\n70–140'),
            ('TIR60140', 'TIR\n60–140'),
        ]),
        ('D', 'Hypoglycaemia (%)', [
            ('TBR70', 'TBR\n<70'),
            ('TBR54', 'TBR\n<54'),
        ]),
        ('E', 'Hyperglycaemia (%)', [
            ('TAR180', 'TAR\n>180'),
            ('TAR250', 'TAR\n>250'),
        ]),
        ('F', glucose_label('Time-of-day medians'), [
            ('Median_morning', 'Morning\n06:30–11:30'),
            ('Median_noon', 'Noon\n11:30–15:30'),
            ('Median_afternoon', 'Afternoon\n15:30–18:30'),
            ('Median_evening', 'Evening\n18:30–22:30'),
            ('Median_night', 'Night\n22:30–06:30'),
        ]),
    ]
    
    # Create figure with GridSpec for better control
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    gs = GridSpec(6, 1, figure=fig, hspace=0.6, 
                  height_ratios=[1.2, 0.8, 1.0, 0.8, 0.8, 1.3])
    
    for panel_idx, (panel_label, group_title, metrics) in enumerate(metric_groups):
        ax = fig.add_subplot(gs[panel_idx])
        
        # Prepare data for this group
        data_list = []
        labels = []
        
        for metric_col, label in metrics:
            if metric_col not in df.columns:
                continue
                
            values = df[metric_col].dropna()
            
            if len(values) == 0:
                continue
            
            # Convert proportions to percentages where appropriate
            if metric_col in ['CV', 'TIR70180', 'TIR70140', 'TIR60140', 
                             'TBR70', 'TBR54', 'TAR180', 'TAR250']:
                if values.max() <= 1.0:  # Stored as proportion
                    values = values * 100
            
            # Panel B puts SD (a concentration) next to CV (%), so the axis
            # cannot carry a single unit: convert the values themselves.
            elif is_glucose_valued_column(metric_col):
                values = to_display(values)
            
            data_list.append(values)
            labels.append(label)
        
        if len(data_list) == 0:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Create boxplot with styling
        bp = ax.boxplot(
            data_list,
            positions=range(len(data_list)),
            widths=0.5,
            patch_artist=True,
            showmeans=True,
            meanline=False,
            showfliers=True,
            flierprops=dict(marker='o', markerfacecolor='none', 
                          markeredgecolor=BOX_COLOR, markersize=2, 
                          alpha=0.4, linewidth=0.5),
            boxprops=dict(facecolor=BOX_COLOR, alpha=0.5, linewidth=0.8, 
                         edgecolor='black'),
            medianprops=dict(color=MEDIAN_COLOR, linewidth=1.5),
            whiskerprops=dict(color='black', linewidth=0.8),
            capprops=dict(color='black', linewidth=0.8),
            meanprops=dict(marker='D', markerfacecolor=MEAN_COLOR, 
                          markeredgecolor='black', markersize=3, 
                          linewidth=0.5),
        )
        
        # Add subtle horizontal grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.2, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        
        # # Add sample sizes below each box
        # for i, values in enumerate(data_list):
        #     n = len(values)
        #     # Position n below the axis
        #     ax.text(i, ax.get_ylim()[0], f'n={n}', 
        #            ha='center', va='top', fontsize=6, color='black',
        #            transform=ax.get_xaxis_transform(), 
        #            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
        #                     edgecolor='none', alpha=0.7))
        
        # Formatting
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7, ha='center')
        ax.set_ylabel(group_title, fontsize=9, fontweight='bold')
        ax.tick_params(axis='y', labelsize=7)
        ax.tick_params(axis='x', length=0)  # Remove x-axis ticks
        
        # Add panel label
        add_panel_label(ax, panel_label, x=-0.08, y=1.02)
    
    # Adjust layout
    plt.subplots_adjust(left=0.12, right=0.95, top=0.98, bottom=0.02)
    
    save_figure(fig, stem)
    