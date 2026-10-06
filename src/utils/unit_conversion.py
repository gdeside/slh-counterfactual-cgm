"""
src/utils/unit_conversion.py
============================

Glucose unit conversion and display-unit handling (mg/dL <-> mmol/L).

The project computes exclusively in **mg/dL**: every threshold in
``src.config`` (``THRESHOLD_HYPO``, ``THRESHOLD_HYPER``, ``H_HYPER`` ...),
every stored data file, and every metric is mg/dL.  This module adds two
independent things on top of that invariant:

1. **Value conversion** — turn mg/dL numbers into mmol/L numbers (and back)
   for scalars, arrays, Series and whole DataFrames.  Used by
   ``scripts/preprocessing/convert_data_units.py`` to publish mmol/L copies
   of the data files.

2. **A display unit** — a process-wide setting that says which unit figures
   and printed values should *show*.  Analysis stays in mg/dL; only the
   presentation layer changes.  Set it with the ``CGM_GLUCOSE_UNIT``
   environment variable, with :func:`set_display_unit`, or temporarily with
   the :func:`display_unit` context manager.

The conversion uses the molar mass of glucose (180.156 g/mol):

    1 mmol/L = 18.0182 mg/dL

Functions
---------
mgdl_to_mmoll, mmoll_to_mgdl
    Convert glucose values between the two units.
normalize_unit
    Map a free-form unit string ("mmol/l", "mmolL", ...) to a canonical one.
convert_glucose_column
    Return a copy of a DataFrame with one glucose column converted.
convert_dataframe_units
    Convert every glucose-valued column of a DataFrame (leaves %, counts
    and durations untouched).
convert_metric_table
    Convert a long-format metric table (one row per metric, statistics in
    columns), converting only the glucose-valued rows.
is_glucose_valued_column
    Whether a column name holds a glucose concentration.
get_display_unit, set_display_unit, display_unit
    Read / set / temporarily override the unit used for display.
to_display, from_display
    Convert canonical mg/dL values to (and back from) the display unit.
glucose_unit_label, glucose_label, format_glucose
    Unit string, axis label, and formatted value in the display unit.
unit_tag
    Filename-safe token ("mgdl" / "mmoll") for naming per-unit outputs.
glucose_ticks
    Round tick positions (in mg/dL) with labels in the display unit.

Author
------
Guillaume Deside

Date
----
2026-09-08
"""

# =============================================================================
# STANDARD LIBRARY
# =============================================================================
import os
import re
from contextlib import contextmanager
from typing import Iterable, Optional, Sequence, Tuple, Union

# =============================================================================
# THIRD-PARTY LIBRARIES
# =============================================================================
import numpy as np
import pandas as pd

# =============================================================================
# CONVERSION FACTOR
# =============================================================================
# Molar mass of glucose: 180.156 g/mol  ->  1 mmol/L = 18.0182 mg/dL.
# Defined here (rather than in src.config.constants) so that this module has
# no import-time dependency on the config package, which re-exports it.
MGDL_PER_MMOLL = 18.0182

# Accepted input types for the scalar/array conversion helpers
GlucoseLike = Union[float, int, np.ndarray, pd.Series, pd.DataFrame]

# =============================================================================
# UNIT NAMES
# =============================================================================
UNIT_MGDL = "mg/dL"
UNIT_MMOLL = "mmol/L"

#: Unit every stored value, threshold and computation of the project uses.
CANONICAL_UNIT = UNIT_MGDL

#: Conventional number of decimals when printing a value in each unit.
DISPLAY_DECIMALS = {UNIT_MGDL: 0, UNIT_MMOLL: 1}

#: Environment variable read once at import to initialise the display unit.
ENV_DISPLAY_UNIT = "CGM_GLUCOSE_UNIT"

# Accepted spellings, compared after lower-casing and stripping "/", " ", "_".
_UNIT_ALIASES = {
    "mgdl": UNIT_MGDL,
    "mg": UNIT_MGDL,
    "mgperdl": UNIT_MGDL,
    "mmoll": UNIT_MMOLL,
    "mmol": UNIT_MMOLL,
    "mmolperl": UNIT_MMOLL,
    "mm": UNIT_MMOLL,
}


def normalize_unit(unit: str) -> str:
    """
    Map a free-form unit string onto ``UNIT_MGDL`` or ``UNIT_MMOLL``.

    Parameters
    ----------
    unit : str
        Unit spelling, e.g. "mmol/L", "mmol/l", "mmolL", "mg/dl", "MGDL".

    Returns
    -------
    str
        Either ``"mg/dL"`` or ``"mmol/L"``.

    Raises
    ------
    ValueError
        If the string does not name a supported glucose unit.

    Examples
    --------
    >>> normalize_unit("mmol/l")
    'mmol/L'
    >>> normalize_unit("MG/DL")
    'mg/dL'
    """
    key = str(unit).strip().lower().replace("/", "").replace(" ", "").replace("_", "")

    if key not in _UNIT_ALIASES:
        raise ValueError(
            f"Unknown glucose unit {unit!r}. Expected {UNIT_MGDL!r} or {UNIT_MMOLL!r}."
        )

    return _UNIT_ALIASES[key]


# =============================================================================
# SCALAR / ARRAY CONVERSIONS
# =============================================================================
def mgdl_to_mmoll(
    values: GlucoseLike,
    decimals: Optional[int] = None,
) -> GlucoseLike:
    """
    Convert glucose values from mg/dL to mmol/L.

    Parameters
    ----------
    values : float, int, np.ndarray, pd.Series or pd.DataFrame
        Glucose values expressed in mg/dL.  NaNs are propagated.
    decimals : int, optional
        If given, round the result to this number of decimals.  Clinical
        reporting of mmol/L conventionally uses one decimal.

    Returns
    -------
    Same type as ``values``
        Glucose values expressed in mmol/L.

    Examples
    --------
    >>> float(mgdl_to_mmoll(70, decimals=2))
    3.88
    >>> float(mgdl_to_mmoll(180, decimals=1))
    10.0
    """
    converted = np.divide(values, MGDL_PER_MMOLL)

    if decimals is not None:
        converted = np.round(converted, decimals)

    return converted


def mmoll_to_mgdl(
    values: GlucoseLike,
    decimals: Optional[int] = None,
) -> GlucoseLike:
    """
    Convert glucose values from mmol/L to mg/dL.

    Parameters
    ----------
    values : float, int, np.ndarray, pd.Series or pd.DataFrame
        Glucose values expressed in mmol/L.  NaNs are propagated.
    decimals : int, optional
        If given, round the result to this number of decimals.

    Returns
    -------
    Same type as ``values``
        Glucose values expressed in mg/dL.

    Examples
    --------
    >>> float(mmoll_to_mgdl(3.9, decimals=0))
    70.0
    """
    converted = np.multiply(values, MGDL_PER_MMOLL)

    if decimals is not None:
        converted = np.round(converted, decimals)

    return converted


# =============================================================================
# DATAFRAME CONVERSION
# =============================================================================
def convert_glucose_column(
    df: pd.DataFrame,
    to_unit: str = "mmol/L",
    column: Optional[str] = None,
    target_column: Optional[str] = None,
    decimals: Optional[int] = None,
) -> pd.DataFrame:
    """
    Convert the glucose column of a CGM DataFrame to the requested unit.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing ``column``.
    to_unit : {"mmol/L", "mg/dL"}, default "mmol/L"
        Unit to convert *to*.  The source unit is assumed to be the other
        one (the project stores glucose in mg/dL).  Any spelling accepted
        by :func:`normalize_unit` works.
    column : str, optional
        Name of the glucose column to convert.  Defaults to the project's
        ``GLUCOSE_COL``.
    target_column : str, optional
        Name of the column receiving the converted values.  Defaults to
        ``column`` (in-place conversion on the returned copy).
    decimals : int, optional
        If given, round the converted values to this number of decimals.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with the converted column.

    Raises
    ------
    KeyError
        If ``column`` is missing from ``df``.
    ValueError
        If ``to_unit`` is not a recognised glucose unit.

    Notes
    -----
    - The input DataFrame is never modified; a copy is returned.
    - No unit is inferred from the data: the caller is responsible for
      knowing the source unit.
    """
    if column is None:
        # Imported lazily: src.config imports this module at package init.
        from src.config.constants import GLUCOSE_COL
        column = GLUCOSE_COL

    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in DataFrame "
            f"(available: {list(df.columns)})"
        )

    converter = (
        mgdl_to_mmoll if normalize_unit(to_unit) == UNIT_MMOLL else mmoll_to_mgdl
    )

    df = df.copy()
    df[target_column or column] = converter(
        pd.to_numeric(df[column], errors="coerce"),
        decimals=decimals,
    )

    return df


# =============================================================================
# DISPLAY UNIT
# =============================================================================
# The unit that figures, tables and printed values should *show*.  It never
# affects computation: thresholds, episode detection and metrics stay in
# CANONICAL_UNIT (mg/dL).  Initialised from the environment so a whole run can
# be switched without touching the code:
#
#     CGM_GLUCOSE_UNIT="mmol/L" python3 -m scripts.paper_figures.generate_figure_3_patient_comparison
#
_display_unit = normalize_unit(os.environ.get(ENV_DISPLAY_UNIT, CANONICAL_UNIT))


def get_display_unit() -> str:
    """
    Return the unit currently used for display.

    Returns
    -------
    str
        ``"mg/dL"`` (default) or ``"mmol/L"``.
    """
    return _display_unit


def set_display_unit(unit: str) -> str:
    """
    Set the unit used for display, process-wide.

    Parameters
    ----------
    unit : str
        Target display unit; any spelling accepted by :func:`normalize_unit`.

    Returns
    -------
    str
        The normalised unit that was set.

    Notes
    -----
    Call this **before** building figures.  Computation is unaffected: the
    pipeline always works in ``CANONICAL_UNIT``.
    """
    global _display_unit
    _display_unit = normalize_unit(unit)
    return _display_unit


@contextmanager
def display_unit(unit: str):
    """
    Temporarily switch the display unit, restoring it on exit.

    Parameters
    ----------
    unit : str
        Display unit to use inside the ``with`` block.

    Examples
    --------
    Render the same figure twice, once per unit::

        for u in (UNIT_MGDL, UNIT_MMOLL):
            with display_unit(u):
                make_figure(out_name=f"figure3_{u.replace('/', '')}.pdf")
    """
    previous = get_display_unit()
    set_display_unit(unit)
    try:
        yield get_display_unit()
    finally:
        set_display_unit(previous)


def to_display(
    values: GlucoseLike,
    decimals: Optional[int] = None,
) -> GlucoseLike:
    """
    Convert canonical mg/dL values into the current display unit.

    A no-op (identity) while the display unit is mg/dL, so it is safe to
    wrap every glucose quantity that reaches a plot or a table.

    Parameters
    ----------
    values : float, int, np.ndarray, pd.Series or pd.DataFrame
        Glucose values in ``CANONICAL_UNIT`` (mg/dL).
    decimals : int, optional
        Round the result to this number of decimals.

    Returns
    -------
    Same type as ``values``
        Values expressed in :func:`get_display_unit`.
    """
    if get_display_unit() == CANONICAL_UNIT:
        return np.round(values, decimals) if decimals is not None else values

    return mgdl_to_mmoll(values, decimals=decimals)


def from_display(
    values: GlucoseLike,
    decimals: Optional[int] = None,
) -> GlucoseLike:
    """
    Convert values expressed in the display unit back to canonical mg/dL.

    Inverse of :func:`to_display`; use it when a user-supplied number (an
    axis limit, a threshold typed in mmol/L) must re-enter the pipeline.

    Parameters
    ----------
    values : float, int, np.ndarray, pd.Series or pd.DataFrame
        Glucose values in :func:`get_display_unit`.
    decimals : int, optional
        Round the result to this number of decimals.

    Returns
    -------
    Same type as ``values``
        Values expressed in ``CANONICAL_UNIT`` (mg/dL).
    """
    if get_display_unit() == CANONICAL_UNIT:
        return np.round(values, decimals) if decimals is not None else values

    return mmoll_to_mgdl(values, decimals=decimals)


def unit_tag(unit: Optional[str] = None) -> str:
    """
    Filename-safe token for a glucose unit.

    Used to keep the mg/dL and mmol/L versions of a figure or of a converted
    data tree side by side instead of overwriting one another.

    Parameters
    ----------
    unit : str, optional
        Unit to tag.  Defaults to the current display unit.

    Returns
    -------
    str
        ``"mgdl"`` or ``"mmoll"``.

    Examples
    --------
    >>> unit_tag("mmol/L")
    'mmoll'
    >>> unit_tag()
    'mgdl'
    """
    return normalize_unit(unit or get_display_unit()).replace("/", "").lower()


def glucose_unit_label() -> str:
    """
    Return the current display unit as it should be printed on a figure.

    Returns
    -------
    str
        ``"mg/dL"`` or ``"mmol/L"``.
    """
    return get_display_unit()


def glucose_label(quantity: str = "Glucose", parentheses: bool = True) -> str:
    """
    Build an axis / column label carrying the current display unit.

    Parameters
    ----------
    quantity : str, default "Glucose"
        Name of the plotted quantity, e.g. "Median glucose", "SD".
    parentheses : bool, default True
        Wrap the unit in parentheses.  Set to False for table headers that
        already provide their own brackets.

    Returns
    -------
    str
        e.g. ``"Glucose (mmol/L)"``.

    Examples
    --------
    >>> glucose_label()
    'Glucose (mg/dL)'
    >>> glucose_label("Median glucose")
    'Median glucose (mg/dL)'
    """
    unit = glucose_unit_label()

    if not quantity:
        return f"({unit})" if parentheses else unit

    return f"{quantity} ({unit})" if parentheses else f"{quantity} {unit}"


def format_glucose(
    value: float,
    with_unit: bool = True,
    decimals: Optional[int] = None,
) -> str:
    """
    Format a canonical mg/dL value for display, converting it if needed.

    Parameters
    ----------
    value : float
        Glucose value in ``CANONICAL_UNIT`` (mg/dL).
    with_unit : bool, default True
        Append the unit string.
    decimals : int, optional
        Override the conventional precision (0 for mg/dL, 1 for mmol/L).

    Returns
    -------
    str
        e.g. ``"180 mg/dL"`` or ``"10.0 mmol/L"``.

    Examples
    --------
    >>> format_glucose(180)
    '180 mg/dL'
    >>> with display_unit(UNIT_MMOLL):
    ...     format_glucose(180)
    '10.0 mmol/L'
    """
    unit = get_display_unit()

    if decimals is None:
        decimals = DISPLAY_DECIMALS[unit]

    text = f"{float(to_display(value)):.{decimals}f}"

    return f"{text} {unit}" if with_unit else text


# Candidate tick spacings, per unit, from finest to coarsest.
_TICK_STEPS = {
    UNIT_MGDL: (0.2, 0.5, 1, 2, 5, 10, 20, 25, 50, 100, 200),
    UNIT_MMOLL: (0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0),
}


def glucose_ticks(
    vmin: float,
    vmax: float,
    max_ticks: int = 7,
) -> Tuple[np.ndarray, list]:
    """
    Build round tick positions for a glucose axis.

    The returned positions are in ``CANONICAL_UNIT`` (mg/dL) — i.e. in the
    data coordinates of a plot drawn from unconverted pipeline data — while
    the labels are round numbers in the display unit.  This is what lets a
    figure switch to mmol/L without touching any plotted series.

    Parameters
    ----------
    vmin, vmax : float
        Axis limits in mg/dL.
    max_ticks : int, default 7
        Upper bound on the number of ticks produced.

    Returns
    -------
    positions : np.ndarray
        Tick positions in mg/dL.
    labels : list of str
        Matching labels in the display unit (no unit suffix).

    Examples
    --------
    >>> pos, lab = glucose_ticks(0, 350)
    >>> lab
    ['0', '50', '100', '150', '200', '250', '300', '350']
    """
    unit = get_display_unit()
    lo, hi = float(to_display(min(vmin, vmax))), float(to_display(max(vmin, vmax)))
    span = hi - lo

    step = _TICK_STEPS[unit][-1]
    for candidate in _TICK_STEPS[unit]:
        if span / candidate <= max_ticks:
            step = candidate
            break

    first = np.ceil(lo / step) * step
    display_positions = np.arange(first, hi + step * 1e-9, step)

    # Enough decimals to render the step exactly (0.25 needs two, 0.5 one).
    decimals = next(d for d in (0, 1, 2) if round(float(step), d) == float(step))
    # Snap the tick that lands on zero, so it never renders as "-0.0".
    display_positions = np.where(
        np.abs(display_positions) < step * 1e-6, 0.0, display_positions
    )
    labels = [f"{p:.{decimals}f}" for p in display_positions]

    return np.asarray(from_display(display_positions), dtype=float), labels


# =============================================================================
# COLUMN CLASSIFICATION
# =============================================================================
# Which columns of the project's data files hold a glucose *concentration*
# (and must therefore be converted) as opposed to a percentage, a count or a
# duration (which must not).  Matching is case-insensitive on the column name.

# Checked first: anything matching here is never treated as glucose.
_NON_GLUCOSE_PATTERNS = (
    r"(^|_)cv($|_)",          # coefficient of variation (unitless / %)
    r"(^|_)tir",              # time in range (%)
    r"(^|_)tar",              # time above range (%)
    r"(^|_)tbr",              # time below range (%)
    r"a1c",                   # HbA1c, IDAA1c, GTAA1c
    r"days",                  # CGM_days_measured  (but not daytime_* metrics)
    r"duration",
    r"_min$",                 # gap_before_min, dominant_step_min, duration_min
    r"_pct$",                 # coverage_pct
    r"coverage",
    r"step",
    r"^n(_|$)",               # n, n_points, n_missing
    r"_id$",
    r"(^|_)time($|_)",        # time, time_h, onset_time  (not daytime_*)
    r"flag",
    r"mmol",                  # already expressed in mmol/... — do not touch
)

# Then: a column is glucose-valued if it matches any of these.
_GLUCOSE_PATTERNS = (
    r"glycemi",               # glycemie, Mean_glycemia, Median_glycemia, ...
    r"glucose",               # glucose_original, glucose_ctf_delta_30, ...
    r"(^|_)sd($|_)",          # SD, orig_sd, ctf_sd, delta_sd
    r"(^|_)mean($|_)",        # orig_mean, ctf_mean, delta_mean
    r"(^|_)median($|_)",      # Median_morning, orig_median, delta_median
    r"(^|_)(minimum|maximum|min|max)($|_)",
    r"threshold",
)

_NON_GLUCOSE_RE = re.compile("|".join(_NON_GLUCOSE_PATTERNS), re.IGNORECASE)
_GLUCOSE_RE = re.compile("|".join(_GLUCOSE_PATTERNS), re.IGNORECASE)


def is_glucose_valued_column(name: str) -> bool:
    """
    Whether a column (or metric) name holds a glucose concentration.

    Percentages (CV, TIR/TAR/TBR), counts, durations, coverage, timestamps
    and HbA1c are rejected; glycemia / glucose / mean / median / SD / min /
    max columns are accepted.

    Parameters
    ----------
    name : str
        Column or metric name, e.g. ``"Mean_glycemia"``, ``"TIR70180"``.

    Returns
    -------
    bool
        True if the values are a glucose concentration in mg/dL.

    Examples
    --------
    >>> [is_glucose_valued_column(c) for c in ("Mean_glycemia", "CV", "TAR250")]
    [True, False, False]
    >>> [is_glucose_valued_column(c) for c in ("delta_sd", "CGM_days_measured")]
    [True, False]
    """
    name = str(name)

    if _NON_GLUCOSE_RE.search(name):
        return False

    return bool(_GLUCOSE_RE.search(name))


def rename_unit_suffix(name: str, to_unit: str) -> str:
    """
    Rewrite a unit suffix carried by a column name.

    ``glucose_mgdl`` becomes ``glucose_mmoll`` when converting, so that a
    published file never advertises the wrong unit in its header.

    Parameters
    ----------
    name : str
        Original column name.
    to_unit : str
        Unit the column now holds.

    Returns
    -------
    str
        Column name with any ``mgdl`` / ``mmoll`` suffix updated.

    Examples
    --------
    >>> rename_unit_suffix("glucose_mgdl", "mmol/L")
    'glucose_mmoll'
    >>> rename_unit_suffix("glycemie", "mmol/L")
    'glycemie'
    """
    tag = "mmoll" if normalize_unit(to_unit) == UNIT_MMOLL else "mgdl"

    return re.sub(r"mg_?dl|mmol_?l", tag, name, flags=re.IGNORECASE)


# =============================================================================
# WHOLE-TABLE CONVERSION
# =============================================================================
def convert_dataframe_units(
    df: pd.DataFrame,
    to_unit: str = UNIT_MMOLL,
    columns: Optional[Sequence[str]] = None,
    decimals: Optional[int] = 2,
    rename: bool = True,
) -> Tuple[pd.DataFrame, list]:
    """
    Convert every glucose-valued column of a wide table.

    Parameters
    ----------
    df : pd.DataFrame
        Table whose glucose columns are in the *other* unit than ``to_unit``.
    to_unit : str, default "mmol/L"
        Unit to convert to.
    columns : sequence of str, optional
        Explicit list of columns to convert.  By default the columns are
        selected with :func:`is_glucose_valued_column`, restricted to those
        holding numeric data.
    decimals : int or None, default 2
        Round converted values to this precision (None disables rounding).
    rename : bool, default True
        Update ``mgdl`` / ``mmoll`` suffixes in the converted column names.

    Returns
    -------
    df_converted : pd.DataFrame
        Copy of ``df`` with the glucose columns converted.
    converted : list of str
        Names of the columns that were converted (original names).

    Notes
    -----
    Non-numeric columns are never touched, so identifiers, timestamps and
    classification labels pass through unchanged.
    """
    to_unit = normalize_unit(to_unit)
    converter = mgdl_to_mmoll if to_unit == UNIT_MMOLL else mmoll_to_mgdl

    if columns is None:
        columns = [
            c for c in df.columns
            if is_glucose_valued_column(c)
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    out = df.copy()
    converted = []

    for column in columns:
        if column not in out.columns:
            continue
        out[column] = converter(
            pd.to_numeric(out[column], errors="coerce"), decimals=decimals
        )
        converted.append(column)

    if rename:
        out = out.rename(
            columns={c: rename_unit_suffix(c, to_unit) for c in converted}
        )

    return out, converted


def convert_metric_table(
    df: pd.DataFrame,
    metric_column: str = "metric",
    value_columns: Optional[Sequence[str]] = None,
    exclude_columns: Iterable[str] = ("n", "n_missing"),
    to_unit: str = UNIT_MMOLL,
    decimals: Optional[int] = 2,
) -> Tuple[pd.DataFrame, list]:
    """
    Convert a long-format metric table (one metric per row).

    Used for ``filtered_cohort_metrics_distribution.csv``, where the unit
    depends on the *row* (``Mean_glycemia`` is mg/dL, ``CV`` and ``TIR70180``
    are not) while the statistics (mean, std, quartiles) sit in columns.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format table.
    metric_column : str, default "metric"
        Column holding the metric names.
    value_columns : sequence of str, optional
        Statistic columns to convert.  Defaults to every numeric column
        except ``metric_column`` and ``exclude_columns``.
    exclude_columns : iterable of str, default ("n", "n_missing")
        Numeric columns that are counts, not statistics.
    to_unit : str, default "mmol/L"
        Unit to convert to.
    decimals : int or None, default 2
        Round converted values to this precision.

    Returns
    -------
    df_converted : pd.DataFrame
        Copy of ``df`` with glucose rows converted.
    converted_metrics : list of str
        Metric names whose rows were converted.

    Raises
    ------
    KeyError
        If ``metric_column`` is missing.
    """
    if metric_column not in df.columns:
        raise KeyError(
            f"Column '{metric_column}' not found in DataFrame "
            f"(available: {list(df.columns)})"
        )

    to_unit = normalize_unit(to_unit)
    converter = mgdl_to_mmoll if to_unit == UNIT_MMOLL else mmoll_to_mgdl

    if value_columns is None:
        excluded = set(exclude_columns) | {metric_column}
        value_columns = [
            c for c in df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
        ]

    out = df.copy()
    is_glucose_row = out[metric_column].map(is_glucose_valued_column)

    for column in value_columns:
        values = pd.to_numeric(out[column], errors="coerce")
        out.loc[is_glucose_row, column] = converter(
            values[is_glucose_row], decimals=decimals
        )

    return out, out.loc[is_glucose_row, metric_column].tolist()
