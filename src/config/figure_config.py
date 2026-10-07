"""
src/config/figure_config.py
===========================

Figure configuration and output paths for the counterfactual CGM-SLH project.

All figure-related constants and directory paths are centralized here.
This ensures consistency across all analysis scripts and facilitates 
reproducibility for journal submission.
"""


# =============================================================================
# BASE FIGURE DIRECTORY
# =============================================================================
from src.config.paths import BASE_PATH

FIG_DIR = BASE_PATH / "Figures"

# =============================================================================
# SUB-DIRECTORIES FOR EACH ANALYSIS SECTION
# =============================================================================

FIG_PREPROCESSING_DIR = FIG_DIR / "Preprocessing"
FIG_FILTERING_DIR     = FIG_DIR / "Filtering"
FIG_METRICS_DIR       = FIG_DIR / "Metrics"

# You can add more as needed:
# FIG_EPISODES_DIR      = FIG_DIR / "Episodes"
# FIG_COVERAGE_DIR      = FIG_DIR / "Coverage"

# =============================================================================
# PUBLICATION-READY FIGURE SETTINGS 
# =============================================================================

DPI_SAVE = 600                      # High resolution required

FONT_SIZE_BASE = 7 

# double-column figure width (most common for main figures)
FIGURE_WIDTH_INCHES  = 7.24

# Single-column figure width (≈ 89 mm)
FIGURE_WIDTH_SINGLE_INCHES = 3.50

# Single-column / half-page height (adjust per figure)
FIGURE_HEIGHT_INCHES = 3.20         # Reasonable default for most panels

FIGURE_HEIGHT_RATIO = 0.90   # Height as fraction of width (alternative to fixed height)


# AGP plot settings
AGP_Y_MIN = 0                   # Y-axis minimum (mg/dL)
AGP_Y_MAX = 350                 # Y-axis maximum (mg/dL)

# =============================================================================
# THE LANCET / LANCET DIABETES & ENDOCRINOLOGY ARTWORK SPECIFICATION
# =============================================================================
# Lancet research pages are set on a three-column grid, and artwork is scaled
# at typesetting to exactly one of three widths.  A figure drawn at one of
# those widths is reproduced 1:1, so the point sizes and line weights it was
# drawn with are the ones that reach the printed page — which is what makes
# the minimum font size below meaningful.
#
#   1 column  ≈  55 mm      (narrow panels, single plots)
#   2 columns ≈ 115 mm      (half-page figures)
#   3 columns ≈ 175 mm      (full page width — used for the 45-day overviews)
#
# Depth: artwork plus its legend must fit the type area of one page; 230 mm
# is the practical ceiling for the artwork itself.
#
# Other house requirements enforced in src.visualization.style:
#   * sans-serif labelling, no text below 7 pt at final size;
#   * no rule thinner than 0.5 pt (thinner rules drop out in print);
#   * editable vector line art (PDF/SVG with TrueType fonts), ≥ 300 dpi for
#     any rasterised element;
#   * NO figure title inside the artwork.

MM_PER_INCH = 25.4

LANCET_WIDTH_1COL_MM = 55.0
LANCET_WIDTH_2COL_MM = 115.0
LANCET_WIDTH_FULL_MM = 175.0
LANCET_MAX_HEIGHT_MM = 230.0

LANCET_WIDTH_1COL_INCHES = LANCET_WIDTH_1COL_MM / MM_PER_INCH   # 2.17 in
LANCET_WIDTH_2COL_INCHES = LANCET_WIDTH_2COL_MM / MM_PER_INCH   # 4.53 in
LANCET_WIDTH_FULL_INCHES = LANCET_WIDTH_FULL_MM / MM_PER_INCH   # 6.89 in
LANCET_MAX_HEIGHT_INCHES = LANCET_MAX_HEIGHT_MM / MM_PER_INCH   # 9.06 in

#: Smallest type allowed anywhere in the artwork, in points.
LANCET_MIN_FONT_PT = 7.0

#: Thinnest rule allowed anywhere in the artwork, in points.
LANCET_MIN_LINEWIDTH_PT = 0.5

#: Minimum resolution for any rasterised element (DPI_SAVE already exceeds it).
LANCET_MIN_DPI = 300

#: Column keyword -> width in inches, for :func:`lancet_figsize`.
LANCET_COLUMN_WIDTHS_INCHES = {
    "single": LANCET_WIDTH_1COL_INCHES,
    "double": LANCET_WIDTH_2COL_INCHES,
    "full":   LANCET_WIDTH_FULL_INCHES,
}


def lancet_figsize(height_inches: float, columns: str = "full") -> tuple[float, float]:
    """
    Return a ``figsize`` that matches a Lancet column width exactly.

    Parameters
    ----------
    height_inches : float
        Desired figure height.  Clamped to ``LANCET_MAX_HEIGHT_INCHES`` so the
        artwork still fits one page together with its typeset legend.
    columns : {"full", "double", "single"}, default "full"
        Which of the three Lancet widths to draw at.

    Returns
    -------
    tuple of float
        ``(width_inches, height_inches)`` for ``plt.subplots(figsize=...)``.

    Raises
    ------
    ValueError
        If *columns* is not one of the three Lancet widths.
    """
    if columns not in LANCET_COLUMN_WIDTHS_INCHES:
        raise ValueError(
            f"columns must be one of {sorted(LANCET_COLUMN_WIDTHS_INCHES)}, "
            f"got {columns!r}"
        )

    return (
        LANCET_COLUMN_WIDTHS_INCHES[columns],
        min(float(height_inches), LANCET_MAX_HEIGHT_INCHES),
    )


# =============================================================================
# HELPER FUNCTION TO CREATE DIRECTORIES
# =============================================================================
def ensure_figure_dirs() -> None:
    """
    Create all figure sub-directories if they do not exist.
    Call this once at the beginning of main analysis scripts or in a Makefile.
    """
    directories = [
        FIG_DIR,
        FIG_PREPROCESSING_DIR,
        FIG_FILTERING_DIR,
        FIG_METRICS_DIR,
        # Add new directories here when created
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
