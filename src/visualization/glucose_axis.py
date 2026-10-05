"""
src/visualization/glucose_axis.py
=================================

Glucose axis helpers — draw figures in mg/dL or mmol/L without converting
any plotted data.

Rationale
---------
Every series this project plots comes out of the pipeline in mg/dL, and so
does every threshold (``THRESHOLD_HYPER``, ``H_CONTEXT`` ...).  Converting
the data before plotting would mean touching each ``ax.plot`` /
``fill_between`` / ``axhline`` / ``axhspan`` call, and any one that is
missed silently produces a figure mixing both units.

Instead, the data stays in mg/dL and only the *axis presentation* changes:
tick positions remain mg/dL data coordinates, while their labels — and the
axis label — are rendered in the display unit.  Adapting a panel is then a
one-line change::

    -   ax.set_ylabel("Glucose (mg/dL)", fontsize=9, labelpad=4)
    +   apply_glucose_axis(ax, fontsize=9, labelpad=4)

and the whole figure follows ``src.utils.unit_conversion``'s display unit.

Numbers that appear as *text* (threshold annotations, table cells) are not
axis-driven and must be formatted with ``format_glucose`` instead.

Functions
---------
apply_glucose_axis
    Label a glucose axis and, in mmol/L, relabel its ticks.
apply_glucose_colorbar
    Same treatment for a colorbar whose mappable is scaled in mg/dL.
glucose_cell
    Convert a mg/dL metric for a table cell.

Author
------
Guillaume Deside

Date
----
2026-09-10
"""

# =============================================================================
# PROJECT IMPORTS
# =============================================================================
from src.utils.unit_conversion import (
    CANONICAL_UNIT,
    get_display_unit,
    glucose_label,
    glucose_ticks,
    to_display,
)


# =============================================================================
# AXES
# =============================================================================
def apply_glucose_axis(
    ax,
    axis: str = "y",
    quantity: str = "Glucose",
    mgdl_ticks=None,
    mgdl_ticklabels=None,
    max_ticks: int = 7,
    set_label: bool = True,
    **label_kwargs,
):
    """
    Label a glucose axis in the current display unit.

    Call it *after* the axis limits are final: in mmol/L the tick positions
    are derived from the current limits.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes carrying a glucose axis.
    axis : {"y", "x"}, default "y"
        Which axis holds glucose.
    quantity : str, default "Glucose"
        Quantity name for the label, e.g. "Median glucose", "SD".
    mgdl_ticks : sequence of float, optional
        Tick positions to keep when displaying mg/dL (typically the hand-
        picked clinical ticks a panel already used).  Ignored in mmol/L,
        where round mmol/L ticks are computed instead.
    mgdl_ticklabels : sequence of str, optional
        Labels for ``mgdl_ticks``.  Defaults to the positions themselves.
    max_ticks : int, default 7
        Upper bound on the number of ticks generated in mmol/L.
    set_label : bool, default True
        Set the axis label.  Pass False to keep a caller-supplied label.
    **label_kwargs
        Forwarded to ``ax.set_ylabel`` / ``ax.set_xlabel`` (fontsize,
        labelpad, ...).

    Returns
    -------
    matplotlib.axes.Axes
        The same Axes, for chaining.

    Notes
    -----
    In mg/dL the tick locator is left untouched unless ``mgdl_ticks`` is
    given, so existing figures are reproduced exactly as before.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    if set_label:
        setter = ax.set_ylabel if axis == "y" else ax.set_xlabel
        setter(glucose_label(quantity), **label_kwargs)

    set_ticks = ax.set_yticks if axis == "y" else ax.set_xticks
    set_ticklabels = ax.set_yticklabels if axis == "y" else ax.set_xticklabels

    if get_display_unit() == CANONICAL_UNIT:
        if mgdl_ticks is not None:
            set_ticks(list(mgdl_ticks))
            set_ticklabels(
                list(mgdl_ticklabels)
                if mgdl_ticklabels is not None
                else [f"{t:g}" for t in mgdl_ticks]
            )
        return ax

    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    positions, labels = glucose_ticks(lo, hi, max_ticks=max_ticks)

    set_ticks(positions)
    set_ticklabels(labels)

    return ax


# =============================================================================
# COLORBARS
# =============================================================================
def apply_glucose_colorbar(
    cbar,
    quantity: str = "Glucose",
    max_ticks: int = 6,
    **label_kwargs,
):
    """
    Label a colorbar whose mappable is scaled in mg/dL.

    Parameters
    ----------
    cbar : matplotlib.colorbar.Colorbar
        Colorbar to relabel.
    quantity : str, default "Glucose"
        Quantity name for the label.
    max_ticks : int, default 6
        Upper bound on the number of ticks generated in mmol/L.
    **label_kwargs
        Forwarded to ``cbar.set_label`` (fontsize, labelpad, ...).

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The same colorbar, for chaining.
    """
    cbar.set_label(glucose_label(quantity), **label_kwargs)

    if get_display_unit() == CANONICAL_UNIT:
        return cbar

    lo, hi = cbar.mappable.get_clim()
    positions, labels = glucose_ticks(lo, hi, max_ticks=max_ticks)

    cbar.set_ticks(positions)
    cbar.set_ticklabels(labels)

    return cbar


# =============================================================================
# TABLES
# =============================================================================
def glucose_cell(value, decimals: int = 1):
    """
    Format a mg/dL metric for a table cell in the current display unit.

    Table cells are text, not axis-driven, so they need value conversion
    rather than :func:`apply_glucose_axis`.

    Parameters
    ----------
    value : float or str
        Metric value in mg/dL, or a placeholder such as ``"-"``.
    decimals : int, default 1
        Precision used when converting to mmol/L.

    Returns
    -------
    float or str
        Returned unchanged when the display unit is mg/dL (the pipeline
        already rounds these values) or when ``value`` is not numeric.
    """
    if get_display_unit() == CANONICAL_UNIT:
        return value

    try:
        return round(float(to_display(value)), decimals)
    except (TypeError, ValueError):
        return value
