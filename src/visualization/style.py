"""
src/visualization/style.py
==========================

Matplotlib styling and figure saving for the counterfactual CGM-SLH project.

All figure constants are imported from src.config.figure_config.

Figures that show glucose are saved with the display unit in their filename
(``Figure4_..._mgdl.pdf`` / ``Figure4_..._mmoll.pdf``), so the two versions of
a figure sit side by side instead of overwriting one another.  See
:func:`save_figure` and ``docs/UNITS.md``.
"""

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

# =============================================================================
# IMPORT CENTRAL CONFIGURATION (no duplication)
# =============================================================================
from src.config.figure_config import (
    DPI_SAVE,
    LANCET_MIN_FONT_PT,
    LANCET_MIN_LINEWIDTH_PT,
)
from src.utils.unit_conversion import glucose_unit_label, unit_tag


FONT_SIZE_BASE = 7

#: Default policy for the per-unit filename suffix added by ``save_figure``.
#: ``"auto"`` tags only the figures that actually show a glucose unit; ``True``
#: tags every figure; ``False`` restores the plain, untagged filenames.
FIGURE_UNIT_SUFFIX = "auto"


#: Sans-serif fallback chain. Arial is the house font; on systems without it (Linux, CI) matplotlib falls back to a
#: metric-compatible Arial clone, then to DejaVu Sans, which ships with matplotlib — no "font not found" warnings.
FONT_SANS_SERIF = ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "Arimo", "DejaVu Sans"]

#: Same chain as a CSS / plotly font-family string.
FONT_CSS_FAMILY = "Arial, Helvetica, 'Liberation Sans', 'DejaVu Sans', sans-serif"


def apply_publication_style() -> None:
    """Apply consistent style matplotlib rcParams for all figures."""
    matplotlib.rcParams.update({
        # Font settings (Arial when available, otherwise a metric-compatible fallback — see FONT_SANS_SERIF)
        "font.family": "sans-serif",
        "font.sans-serif": FONT_SANS_SERIF,
        "font.size": FONT_SIZE_BASE,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "legend.title_fontsize": 7,

        # Line and tick appearance
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,

        # Spines & grid (clean look)
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.35,
        "grid.color": "#e0e0e0",
        "grid.linestyle": ":",
        "axes.axisbelow": True,

        # Output quality
        "pdf.fonttype": 42,      # TrueType fonts (editable in Illustrator)
        "ps.fonttype": 42,
        "figure.dpi": DPI_SAVE,

        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.88,
        "legend.edgecolor": "#cccccc",
        "legend.borderpad": 0.45,
    })


def style_axes(
    ax: plt.Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
) -> None:
    """Apply clean, consistent axis styling (complements rcParams)."""
    
    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    # Ticks
    ax.tick_params(which="major", length=3, width=0.8)
    ax.tick_params(which="minor", length=1.8, width=0.5)

    # Labels
    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=4)
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=4)
    if title is not None:
        ax.set_title(title, fontweight="bold", pad=6, loc="left")


def apply_lancet_style() -> None:
    """
    Apply the project style, then tighten it to *The Lancet*'s artwork rules.

    On top of :func:`apply_publication_style` this raises every type size to
    the 7 pt house minimum and every default rule to 0.5 pt, so that a figure
    drawn at one of the Lancet column widths (see
    ``src.config.figure_config``) is legible at 1:1 reproduction.

    One requirement is *not* expressible as an rcParam and stays with the
    caller: draw at a column width — use ``lancet_figsize``.

    See Also
    --------
    enforce_lancet_minimums : safety net for sizes hard-coded per artist.
    """
    apply_publication_style()

    matplotlib.rcParams.update({
        # Type: nothing below the 7 pt house minimum, sans-serif throughout.
        "font.family": "sans-serif",
        "font.sans-serif": FONT_SANS_SERIF,
        "font.size": LANCET_MIN_FONT_PT,
        "axes.titlesize": 8.0,
        "axes.labelsize": LANCET_MIN_FONT_PT,
        "xtick.labelsize": LANCET_MIN_FONT_PT,
        "ytick.labelsize": LANCET_MIN_FONT_PT,
        "legend.fontsize": LANCET_MIN_FONT_PT,
        "legend.title_fontsize": LANCET_MIN_FONT_PT,

        # Rules: nothing thinner than 0.5 pt, or it drops out in print.
        "axes.linewidth": 0.6,
        "grid.linewidth": LANCET_MIN_LINEWIDTH_PT,
        "lines.linewidth": 0.8,
        "patch.linewidth": LANCET_MIN_LINEWIDTH_PT,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": LANCET_MIN_LINEWIDTH_PT,
        "ytick.minor.width": LANCET_MIN_LINEWIDTH_PT,

        # Editable vector line art: TrueType fonts, no rasterisation.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": DPI_SAVE,

        # House style keeps legends unboxed.
        "legend.frameon": False,
    })


def enforce_lancet_minimums(
    fig: plt.Figure,
    min_font_pt: float = LANCET_MIN_FONT_PT,
    min_linewidth_pt: float = LANCET_MIN_LINEWIDTH_PT,
) -> dict[str, int]:
    """
    Raise any under-size type or under-weight rule in a finished figure.

    rcParams only cover what an artist inherits; sizes passed explicitly
    (``fontsize=6.5`` in a shared plotting helper, a hairline ``lw=0.35``)
    survive them.  Call this once on the assembled figure, immediately
    before saving, to catch those.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Fully drawn figure, modified in place.
    min_font_pt : float, default ``LANCET_MIN_FONT_PT``
        Smallest type size to allow, in points.
    min_linewidth_pt : float, default ``LANCET_MIN_LINEWIDTH_PT``
        Thinnest rule to allow, in points.

    Returns
    -------
    dict[str, int]
        Number of text objects and of lines that had to be corrected, under
        keys ``"text"`` and ``"lines"`` — a non-zero count is a hint that the
        hard-coded size is worth fixing at its source.
    """
    fixed = {"text": 0, "lines": 0}

    for text in fig.findobj(Text):
        if text.get_text() and text.get_fontsize() < min_font_pt:
            text.set_fontsize(min_font_pt)
            fixed["text"] += 1

    for line in fig.findobj(Line2D):
        width = line.get_linewidth()
        if width is not None and 0 < width < min_linewidth_pt:
            line.set_linewidth(min_linewidth_pt)
            fixed["lines"] += 1

    for ax in fig.get_axes():
        for spine in ax.spines.values():
            if spine.get_visible() and spine.get_linewidth() < min_linewidth_pt:
                spine.set_linewidth(min_linewidth_pt)
                fixed["lines"] += 1

    return fixed


def figure_shows_glucose_unit(fig: plt.Figure) -> bool:
    """
    Whether any text drawn on a figure carries the current glucose unit.

    Every glucose panel in this project labels its axis, colorbar, table
    header or threshold annotation with the unit, so this identifies the
    figures whose content depends on the display unit — and, by exclusion,
    leaves unit-free figures (coverage, funnels, sampling intervals) with
    their plain filenames.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to inspect.

    Returns
    -------
    bool
        True if the unit string appears anywhere in the figure's text.
    """
    unit = glucose_unit_label()

    return any(
        unit in text.get_text()
        for text in fig.findobj(Text)
    )


def save_figure(
    fig: plt.Figure,
    path_without_extension: str | Path,
    dpi: int | None = None,
    unit_suffix: bool | str | None = None,
    bbox_inches: str | None = "tight",
) -> None:
    """
    Save figure in both PDF (vector, preferred) and SVG.

    Glucose figures get the display unit appended to their filename, so the
    mg/dL and mmol/L versions coexist::

        Figure4_Counterfactual_Construction_<high>_vs_<low>_mgdl.pdf
        Figure4_Counterfactual_Construction_<high>_vs_<low>_mmoll.pdf

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    path_without_extension : str | Path
        Full path without file extension (e.g., "Figures/Preprocessing/myfigure").
    dpi : int, optional
        Override default DPI (uses DPI_SAVE from config).
    unit_suffix : bool or "auto", optional
        Whether to append the display unit to the filename.  ``"auto"``
        (the default, via ``FIGURE_UNIT_SUFFIX``) tags a figure only when it
        actually shows a glucose unit; ``True`` always tags; ``False`` never
        does.  Pass ``False`` for a figure that must keep a fixed name.
    bbox_inches : str or None, default "tight"
        Forwarded to ``savefig``.  ``"tight"`` crops to the drawn content,
        which changes the figure's final width; pass ``None`` for artwork
        that must come out at an exact submission width (a Lancet column
        width, say), and lay the figure out so nothing needs cropping.

    See Also
    --------
    figure_shows_glucose_unit : the test used by the ``"auto"`` policy.
    """
    if dpi is None:
        dpi = DPI_SAVE

    if unit_suffix is None:
        unit_suffix = FIGURE_UNIT_SUFFIX

    path = Path(path_without_extension)

    if unit_suffix == "auto":
        unit_suffix = figure_shows_glucose_unit(fig)

    if unit_suffix:
        path = path.with_name(f"{path.name}_{unit_tag()}")

    path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = {"dpi": dpi, "bbox_inches": bbox_inches, "transparent": False}

    # Save PDF
    fig.savefig(f"{path}.pdf", **save_kwargs)
    
    # Save PNG
    #fig.savefig(f"{path}.png", **save_kwargs)

    # Save SVG (optional, for web use or further editing)
    fig.savefig(f"{path}.svg", **save_kwargs)

    plt.close(fig)

    print(f"Saved: {path.name}.{{pdf,svg}}  (DPI: {dpi})")
