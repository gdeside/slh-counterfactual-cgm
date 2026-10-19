"""
src/visualization
=================

Publication-ready plotting library for the counterfactual CGM-SLH project.

Modules
-------
colors
    Centralised color palette (Wong color-blind-safe set + project extensions).
style
    Matplotlib rcParams setup (``apply_publication_style``), axis helper
    (``style_axes``), and figure-saving utility (``save_figure``).
agp_plot
    Ambulatory Glucose Profile panel and companion panels (recurrence
    heatmaps, 7-day trace, metrics comparison table).
coverage_plots
    Coverage distribution histogram and individual strip-plot panels.
episode_plots
    Episode classification funnel figure (waterfall + OR-crossing breakdown).
filtering_plots
    Patient-selection histogram and scatter figures.
metrics_plots
    Exploratory and supplementary CGM-metric figures (scatter views 1–5,
    distribution boxplot panel).
slh_schema_plots
    SLH episode schema overview and zoom panels (relative time, gap-safe).
glucose_axis
    Glucose axis / colorbar labelling in the current display unit
    (``apply_glucose_axis``), so a figure can be drawn in mmol/L without
    converting any plotted series.
visualization_helper
    Shared axis helpers: night shading, relative-day tick formatter.

Public API
----------
The most commonly used symbols are re-exported here for convenient
``from src.visualization import X`` imports.
"""

# Style helpers
from src.visualization.style import (
    apply_publication_style,
    style_axes,
    save_figure,
    save_figure_with_panels,
    FONT_SIZE_BASE,
)

# AGP figure panels
from src.visualization.agp_plot import (
    plot_agp_panel,
    plot_recurrence_classification_panel,
    plot_recurrence_duration_panel,
    plot_7day_trace,
    add_panel_label,
    plot_metrics_table,
)

# Coverage panels
from src.visualization.coverage_plots import (
    draw_histogram_panel,
    draw_strip_plot_panel,
)

# Episode classification figure
from src.visualization.episode_plots import plot_funnel_figure

# Patient-filtering figures
from src.visualization.filtering_plots import (
    compute_adaptive_bins,
    generate_histogram_figure,
    generate_scatter_figure,
)

# CGM-metric exploratory figures
from src.visualization.metrics_plots import (
    view1_control_overview,
    view2_hypo_hyper_risk,
    view3_diurnal_pattern,
    view4_variability,
    view5_correlation_heatmap,
    plot_distribution_metrics,
)

# SLH schema panels
from src.visualization.slh_schema_plots import (
    detect_sampling_interval,
    split_by_gaps,
    draw_overview_panel,
    draw_zoom_panel,
)

# Glucose axis / display-unit helpers
from src.visualization.glucose_axis import (
    apply_glucose_axis,
    apply_glucose_colorbar,
    glucose_cell,
)

# Shared axis helpers
from src.visualization.visualization_helper import (
    shade_nights_on_axis,
    setup_relative_time_axis,
    RelativeDayFormatter,
)

__all__ = [
    # style
    "apply_publication_style",
    "style_axes",
    "save_figure",
    "save_figure_with_panels",
    "FONT_SIZE_BASE",
    # agp_plot
    "plot_agp_panel",
    "plot_recurrence_classification_panel",
    "plot_recurrence_duration_panel",
    "plot_7day_trace",
    "add_panel_label",
    "plot_metrics_table",
    # coverage_plots
    "draw_histogram_panel",
    "draw_strip_plot_panel",
    # episode_plots
    "plot_funnel_figure",
    # filtering_plots
    "compute_adaptive_bins",
    "generate_histogram_figure",
    "generate_scatter_figure",
    # metrics_plots
    "view1_control_overview",
    "view2_hypo_hyper_risk",
    "view3_diurnal_pattern",
    "view4_variability",
    "view5_correlation_heatmap",
    "plot_distribution_metrics",
    # slh_schema_plots
    "detect_sampling_interval",
    "split_by_gaps",
    "draw_overview_panel",
    "draw_zoom_panel",
    # glucose_axis
    "apply_glucose_axis",
    "apply_glucose_colorbar",
    "glucose_cell",
    # visualization_helper
    "shade_nights_on_axis",
    "setup_relative_time_axis",
    "RelativeDayFormatter",
]
