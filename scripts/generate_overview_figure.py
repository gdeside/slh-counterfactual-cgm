"""
scripts/generate_overview_figure.py
===================================
45-Day CGM Overview — Episode Classification by Criterion Level
===============================================================
Generate a publication-ready multi-row figure showing the raw CGM signal
for a 45-day window, with hyperglycaemic episodes colour-coded according
to the first SLH classification criterion they fail (C1–C6) or classified
as SLH when all criteria are satisfied.

Figure layout
-------------
Five rows of nine days each (configurable via --days_per_row / --n_rows).
Each row shares the same relative "Day N" x-axis.  Episodes that span a
row boundary are clipped to the row.

Episode colour legend (sequential filtering)
--------------------------------------------
  C1  Duration outside [D_MIN_MIN, D_MAX_MIN] min       → vermillion
  C2  Insufficient data coverage (< 75 %)                → light red
  C4  Pre-episode hyperglycaemia                          → orange
  C5  Post-episode hyperglycaemia                         → yellow
  C6  No crossing ≤ H_CONTEXT mg/dL                       → sky blue
  SLH daytime    all criteria satisfied, onset in daytime → bluish green
  SLH nighttime  all criteria satisfied, onset at night   → dark navy

C3 (daytime) is evaluated LAST, as a day/night split of the episodes that
already satisfy every other criterion — not a competing exclusion stage.

Artwork specification
---------------------
No title is drawn inside the artwork.

Output
------
PDF and SVG are written to *output_dir* via
``save_figure``:
  {patient_id}_overview.{pdf,svg}          — standard C1→C2→C3→C4→C5→C6
  {patient_id}_overview_grouped.{pdf,svg}  — grouped C1→C4+C5+C6→C2→C3

Usage
-----
  python3 -m scripts.generate_overview_figure [--patient_id ID] [--output Results/SLH_analysis] [--days_per_row 9] [--n_rows 5]

Authors
-------
Guillaume Deside

Date
----
2025
"""

# =============================================================================
# STANDARD LIBRARY
# =============================================================================
import os
import sys
import argparse

# =============================================================================
# THIRD-PARTY LIBRARIES
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# =============================================================================
# PROJECT — preprocessing
# =============================================================================
from src.utils.patient_selection import resolve_patient_id
from src.preprocessing.load_data import load_patient_window
from src.preprocessing.timestep import detect_dominant_sampling_interval

# =============================================================================
# PROJECT — episode detection & classification
# =============================================================================
from src.episodes.detection import detect_hyperglycemia_episodes
from src.episodes.criteria import (
    check_c1_duration_bounds,
    check_c2_data_completeness,
    check_c3_daytime_only,
    check_c4_no_pre_episode_hyperglycemia,
    check_c5_no_post_episode_hyperglycemia,
    check_c6_or_crossing_condition,
)
from src.episodes.episode_config import (
    H_HYPER,
    H_CONTEXT,
    D_MIN_MIN,
    D_MAX_MIN,
    MIN_DATA_COMPLETENESS,
    NIGHT_START_H,
    NIGHT_END_H,
    GAP_FACTOR,
    APPLY_C1,
    APPLY_C2,
    APPLY_C3,
    APPLY_C4,
    APPLY_C5,
    APPLY_C6,
)

# =============================================================================
# PROJECT — configuration
# =============================================================================
from src.config import TIME_COL, GLUCOSE_COL, FIG_DIR
from src.config.figure_config import LANCET_MIN_FONT_PT, lancet_figsize

# =============================================================================
# PROJECT — visualization helpers
# =============================================================================
from src.visualization.style import (
    apply_lancet_style,
    enforce_lancet_minimums,
    save_figure,
    style_axes,
)
from src.visualization.colors import (
    STAGE_COLORS,
    COLOR_NIGHT,
    COLOR_NIGHT_CATEGORY,
    COLOR_HYPER,
    COLOR_ORIGINAL,
)
from src.visualization.visualization_helper import (
    shade_nights_on_axis,
    setup_relative_time_axis,
)
from src.visualization.glucose_axis import apply_glucose_axis
from src.utils.unit_conversion import format_glucose

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_PATIENT_ID = None   # None = automatic: patient with the most SLH episodes
DEFAULT_OUTPUT_DIR = os.path.join(FIG_DIR, "SLH_analysis")

DAYS_PER_ROW = 9
N_ROWS       = 5

ALPHA_EPISODE = 0.38

# Human-readable labels for each classification stage (used in the legend).
STAGE_LABELS = {
    "C1_duration":    f"C1  Duration not in [{D_MIN_MIN}–{D_MAX_MIN} min]",
    "C2_incomplete":  f"C2  Incomplete data (< {MIN_DATA_COMPLETENESS * 100:.0f} %)",
    "C4_hyper_before": "C4  Pre-episode hyperglycaemia",
    "C5_hyper_after":  "C5  Post-episode hyperglycaemia",
    "C6_no_cross":    f"C6  No crossing ≤ {format_glucose(H_CONTEXT)}",
    "SLH_daytime":     "SLH daytime",
    "SLH_nighttime":   "SLH nighttime",
}

# Draw order: earlier stages are drawn first (lower z-order) so that the
# final SLH colour sits on top when episodes overlap. Night (C3) is applied
# LAST, splitting episodes that already survived every other criterion into
# daytime vs nighttime SLH, so "SLH_nighttime" is deliberately last here.
DRAW_ORDER = [
    "C1_duration",
    "C2_incomplete",
    "C4_hyper_before",
    "C5_hyper_after",
    "C6_no_cross",
    "SLH_daytime",
    "SLH_nighttime",
]

# Local colour additions for the two SLH sub-stages, kept out of the shared
# src.visualization.colors.STAGE_COLORS since other scripts may still rely on
# its original single "SLH" / "C3_nocturnal" keys.
OVERVIEW_STAGE_COLORS = {
    **STAGE_COLORS,
    "SLH_daytime":   STAGE_COLORS["SLH"],
    "SLH_nighttime": COLOR_NIGHT_CATEGORY,  # dark navy — same "Night" category color as Figure 2
}

# ---------------------------------------------------------------------------
# Grouped-context variant: C1 → C4+C5+C6 → C2 → C3 (C3 last, day/night split)
# C4, C5, C6 are merged into a single "Context_fail" stage.
# ---------------------------------------------------------------------------
GROUPED_DRAW_ORDER = [
    "C1_duration",
    "Context_fail",
    "C2_incomplete",
    "SLH_daytime",
    "SLH_nighttime",
]

GROUPED_STAGE_LABELS = {
    "C1_duration":   f"Duration not in [{D_MIN_MIN}–{D_MAX_MIN} min]",
    "Context_fail":  "Contextual analysis  Pre/post-hyper or no OR-crossing",
    "C2_incomplete": f"Incomplete data (< {MIN_DATA_COMPLETENESS * 100:.0f} %)",
    "SLH_daytime":   "SLH daytime",
    "SLH_nighttime": "SLH nighttime",
}

# "Context_fail" gets a single orange colour (C4 hue) to signal grouped context.
GROUPED_STAGE_COLORS = {
    "C1_duration":   OVERVIEW_STAGE_COLORS["C1_duration"],
    "Context_fail":  "#E69F00",
    "C2_incomplete": OVERVIEW_STAGE_COLORS["C2_incomplete"],
    "SLH_daytime":   OVERVIEW_STAGE_COLORS["SLH_daytime"],
    "SLH_nighttime": OVERVIEW_STAGE_COLORS["SLH_nighttime"],
}


# =============================================================================
# EPISODE CLASSIFICATION
# =============================================================================

def classify_episodes_for_visualization(
    df: pd.DataFrame,
    raw_episodes: list[dict],
) -> dict[str, list[dict]]:
    """
    Classify raw episodes sequentially through criteria C1–C6.

    Each episode is assigned to the stage corresponding to the first criterion
    it fails.  Episodes that pass all criteria are assigned to "SLH".

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed CGM data with TIME_COL and GLUCOSE_COL columns.
    raw_episodes : list of dict
        Raw detected episodes from ``detect_hyperglycemia_episodes``.

    Returns
    -------
    dict[str, list[dict]]
        Mapping from stage name (one of DRAW_ORDER) to a list of episode
        dicts with keys ``onset``, ``offset``, and ``duration_min``.
    """
    classified = {stage: [] for stage in DRAW_ORDER}

    if not raw_episodes:
        return classified

    n_episodes = len(raw_episodes)
    active = [True] * n_episodes  # True while the episode is still a candidate SLH

    def _make_ep(episode: dict) -> dict:
        return {
            "onset":        episode["onset_time"],
            "offset":       episode["offset_time"],
            "duration_min": episode["duration_min"],
        }

    def _apply(criterion_name: str, check_fn, episodes_subset: list) -> list[bool]:
        """Dispatch criterion check — pass df only when required."""
        if criterion_name == "C1":
            return check_fn(episodes_subset)
        return check_fn(df, episodes_subset)

    # Sequential criteria: each episode is tested only while still active.
    criteria = [
        ("C1", check_c1_duration_bounds,               APPLY_C1, "C1_duration"),
        ("C2", check_c2_data_completeness,              APPLY_C2, "C2_incomplete"),
        ("C4", check_c4_no_pre_episode_hyperglycemia,   APPLY_C4, "C4_hyper_before"),
        ("C5", check_c5_no_post_episode_hyperglycemia,  APPLY_C5, "C5_hyper_after"),
        ("C6", check_c6_or_crossing_condition,          APPLY_C6, "C6_no_cross"),
    ]

    for criterion_name, check_fn, apply_flag, fail_stage in criteria:
        if not apply_flag:
            continue

        active_eps  = [ep  for i, ep  in enumerate(raw_episodes) if active[i]]
        active_idxs = [i   for i      in range(n_episodes)        if active[i]]
        if not active_eps:
            break

        results = _apply(criterion_name, check_fn, active_eps)

        for idx, ep, passed in zip(active_idxs, active_eps, results):
            if not passed:
                active[idx] = False
                classified[fail_stage].append(_make_ep(ep))

    # C3 (daytime) applied LAST, only to episodes that survived every other
    # criterion: splits them into daytime vs nighttime SLH instead of
    # excluding night episodes into a separate stage.
    remaining = [ep for i, ep in enumerate(raw_episodes) if active[i]]
    if remaining and APPLY_C3:
        is_daytime = check_c3_daytime_only(remaining)
        for ep, daytime in zip(remaining, is_daytime):
            stage = "SLH_daytime" if daytime else "SLH_nighttime"
            classified[stage].append(_make_ep(ep))
    else:
        for ep in remaining:
            classified["SLH_daytime"].append(_make_ep(ep))

    return classified


def classify_episodes_grouped_context(
    df: pd.DataFrame,
    raw_episodes: list[dict],
) -> dict[str, list[dict]]:
    """
    Classify episodes with the reordered grouped-context pipeline.

    Criterion order: C1 → C4+C5+C6 (grouped as "Context") → C2 → C3.
    C4, C5, and C6 failures are merged into a single "Context_fail" bucket
    so the figure colour-codes all context-related exclusions with one colour.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed CGM data.
    raw_episodes : list of dict
        Raw detected episodes from ``detect_hyperglycemia_episodes``.

    Returns
    -------
    dict[str, list[dict]]
        Mapping from stage name (one of GROUPED_DRAW_ORDER) to episode dicts.
    """
    classified = {stage: [] for stage in GROUPED_DRAW_ORDER}

    if not raw_episodes:
        return classified

    n_episodes = len(raw_episodes)
    active = [True] * n_episodes

    def _make_ep(episode: dict) -> dict:
        return {
            "onset":        episode["onset_time"],
            "offset":       episode["offset_time"],
            "duration_min": episode["duration_min"],
        }

    def _apply(criterion_name: str, check_fn, episodes_subset: list) -> list[bool]:
        if criterion_name == "C1":
            return check_fn(episodes_subset)
        return check_fn(df, episodes_subset)

    def _filter_stage(criterion_name, check_fn, apply_flag, fail_stage):
        if not apply_flag:
            return
        active_eps  = [ep for i, ep in enumerate(raw_episodes) if active[i]]
        active_idxs = [i  for i     in range(n_episodes)        if active[i]]
        if not active_eps:
            return
        results = _apply(criterion_name, check_fn, active_eps)
        for idx, ep, passed in zip(active_idxs, active_eps, results):
            if not passed:
                active[idx] = False
                classified[fail_stage].append(_make_ep(ep))

    # C1 — duration bounds
    _filter_stage("C1", check_c1_duration_bounds, APPLY_C1, "C1_duration")

    # C4 + C5 + C6 — all grouped into "Context_fail"
    _filter_stage("C4", check_c4_no_pre_episode_hyperglycemia,  APPLY_C4, "Context_fail")
    _filter_stage("C5", check_c5_no_post_episode_hyperglycemia, APPLY_C5, "Context_fail")
    _filter_stage("C6", check_c6_or_crossing_condition,         APPLY_C6, "Context_fail")

    # C2 — data completeness
    _filter_stage("C2", check_c2_data_completeness, APPLY_C2, "C2_incomplete")

    # C3 (daytime) applied LAST, only to episodes that survived everything
    # else: splits them into daytime vs nighttime SLH instead of excluding
    # night episodes into a separate stage.
    remaining = [ep for i, ep in enumerate(raw_episodes) if active[i]]
    if remaining and APPLY_C3:
        is_daytime = check_c3_daytime_only(remaining)
        for ep, daytime in zip(remaining, is_daytime):
            stage = "SLH_daytime" if daytime else "SLH_nighttime"
            classified[stage].append(_make_ep(ep))
    else:
        for ep in remaining:
            classified["SLH_daytime"].append(_make_ep(ep))

    return classified


# =============================================================================
# OVERVIEW FIGURE
# =============================================================================

def plot_overview_figure(
    patient_id: str,
    df: pd.DataFrame,
    classified: dict[str, list[dict]],
    timestep_min: float,
    output_path: str,
    days_per_row: int = DAYS_PER_ROW,
    n_rows: int = N_ROWS,
    draw_order: list[str] | None = None,
    stage_labels: dict[str, str] | None = None,
    stage_colors_map: dict[str, str] | None = None,
) -> None:
    """
    Generate the 45-day episode-classification overview figure and save it.

    The figure is composed of *n_rows* horizontal strips, each covering
    *days_per_row* days.  Each strip shows the CGM trace with night periods
    shaded and hyperglycaemic episodes colour-coded by classification stage.

    Parameters
    ----------
    patient_id : str
        Patient identifier (used in the figure title).
    df : pd.DataFrame
        Preprocessed CGM data with TIME_COL and GLUCOSE_COL.
    classified : dict
        Output of :func:`classify_episodes_for_visualization` or
        :func:`classify_episodes_grouped_context`.
    timestep_min : float
        Dominant sampling interval in minutes (used for gap detection).
    output_path : str
        Destination path *without* file extension.  Both ``.pdf`` and
        ``.svg`` are saved via :func:`save_figure`.
    days_per_row : int, optional
        Number of days shown in each horizontal strip (default 9).
    n_rows : int, optional
        Number of strips (default 5).
    draw_order : list of str, optional
        Stage names in draw order (back-to-front). Defaults to DRAW_ORDER.
    stage_labels : dict, optional
        Human-readable label per stage name. Defaults to STAGE_LABELS.
    stage_colors_map : dict, optional
        Colour per stage name. Defaults to OVERVIEW_STAGE_COLORS.
    """
    _draw_order      = draw_order      or DRAW_ORDER
    _stage_labels    = stage_labels    or STAGE_LABELS
    _stage_colors    = stage_colors_map or OVERVIEW_STAGE_COLORS
    apply_lancet_style()

    row_span = pd.Timedelta(days=days_per_row)
    t_start  = df[TIME_COL].iloc[0]

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=lancet_figsize(1.40 * n_rows, columns="full"),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]

    # One glucose scale for every row: the rows are consecutive strips of the
    # same trace, so a per-row scale would make identical excursions look
    # different from one row to the next.
    g_all = df[GLUCOSE_COL].astype(float).dropna()
    ylo = max(40,  float(g_all.min()) - 12) if not g_all.empty else 40
    yhi = min(450, float(g_all.max()) + 18) if not g_all.empty else 450

    for row_i, ax in enumerate(axes):
        rs = t_start + row_i * row_span
        re = rs + row_span

        mask  = (df[TIME_COL] >= rs) & (df[TIME_COL] < re)
        t_row = df.loc[mask, TIME_COL]
        g_row = df.loc[mask, GLUCOSE_COL].astype(float)

        # Insert NaN at sensor gaps so the trace breaks visually.
        diffs  = t_row.diff().dt.total_seconds().fillna(0) / 60
        g_plot = g_row.copy()
        g_plot[diffs > timestep_min * GAP_FACTOR] = np.nan

        # Night shading — label only on the first strip.
        night_labeled = [] if row_i == 0 else [True]
        shade_nights_on_axis(ax, rs, re, night_labeled)

        # Episode highlights (drawn in draw order so SLH sits on top).
        for z_idx, stage in enumerate(_draw_order):
            color = _stage_colors[stage]
            for ep in classified.get(stage, []):
                if ep["offset"] < rs or ep["onset"] > re:
                    continue
                ax.axvspan(
                    max(ep["onset"],  rs),
                    min(ep["offset"], re),
                    color=color, alpha=ALPHA_EPISODE,
                    zorder=z_idx + 1,
                )

        # Hyperglycaemia threshold line.
        ax.axhline(
            H_HYPER, color=COLOR_HYPER, lw=0.8, ls="--", alpha=0.65, zorder=8,
            label=(f"Hyperglycaemia threshold ({format_glucose(H_HYPER)})"
                   if row_i == 0 else "_"),
        )

        # CGM trace.
        ax.plot(
            t_row, g_plot,
            color=COLOR_ORIGINAL, lw=0.65, alpha=0.90, zorder=9,
            label="CGM signal" if row_i == 0 else "_",
        )

        # Axis limits.
        ax.set_xlim(rs, re)
        ax.set_ylim(ylo, yhi)

        # Relative "Day N" x-axis ticks.
        setup_relative_time_axis(ax, t_start, re, days_per_row)
        plt.setp(ax.xaxis.get_majorticklabels(), fontsize=LANCET_MIN_FONT_PT)

        # Consistent axis styling from the project's style helper.
        style_axes(ax)
        apply_glucose_axis(ax, fontsize=LANCET_MIN_FONT_PT, labelpad=4)
        ax.tick_params(axis="y", labelsize=LANCET_MIN_FONT_PT)
        ax.grid(axis="y", ls=":", alpha=0.22, lw=0.6)

    # Shared legend below all strips.
    legend_handles = [
        mpatches.Patch(
            facecolor=_stage_colors[stage], alpha=0.60,
            label=_stage_labels[stage],
        )
        for stage in _draw_order
    ] + [
        Line2D([0], [0], color=COLOR_HYPER, ls="--", lw=1.0,
               label=f"Hyperglycaemia threshold ({format_glucose(H_HYPER)})"),
        mpatches.Patch(
            facecolor=COLOR_NIGHT, alpha=0.60,
            label=f"Night ({NIGHT_START_H:02d}:00–{NIGHT_END_H:02d}:00)",
        ),
    ]

    # "outside lower center" makes constrained_layout reserve room for the
    # legend inside the canvas, so the figure can be saved uncropped at
    # exactly the page width.
    fig.legend(
        handles=legend_handles,
        loc="outside lower center", ncol=3,
        fontsize=LANCET_MIN_FONT_PT, frameon=False,
    )

    fixed = enforce_lancet_minimums(fig)
    if any(fixed.values()):
        print(f"  Minimums enforced on {fixed['text']} text object(s) "
              f"and {fixed['lines']} line(s)")

    save_figure(fig, output_path, bbox_inches=None)


# =============================================================================
# PATIENT PIPELINE
# =============================================================================

def generate_overview_for_patient(
    patient_id: str,
    output_dir: str,
    days_per_row: int = DAYS_PER_ROW,
    n_rows: int = N_ROWS,
) -> dict[str, int] | None:
    """
    Run the full pipeline for a single patient and save the overview figure.

    Steps:

    1. Load the preprocessed 45-day window via :func:`load_patient_window`.
    2. Detect the dominant sampling interval.
    3. Detect raw hyperglycaemic episodes.
    4. Classify episodes by sequential criterion filtering.
    5. Plot and save the overview figure.

    Parameters
    ----------
    patient_id : str
        Patient identifier (must match a folder in the preprocessed data path).
    output_dir : str
        Directory where the figure files are written.
    days_per_row : int, optional
        Days per row in the figure (default 9).
    n_rows : int, optional
        Number of rows in the figure (default 5).

    Returns
    -------
    dict[str, int] or None
        Episode counts per stage if successful, ``None`` on failure.
    """
    print(f"\nProcessing patient {patient_id} ...")

    df = load_patient_window(patient_id)
    if df is None:
        print(f"  Failed to load data for {patient_id}")
        return None

    print(f"  Loaded {len(df):,} data points")

    timestep_min = detect_dominant_sampling_interval(df)
    print(f"  Sampling interval: {timestep_min:.0f} min")

    raw_episodes = detect_hyperglycemia_episodes(df)
    print(f"  Raw episodes detected: {len(raw_episodes)}")

    if not raw_episodes:
        print("  No episodes detected — skipping figure")
        return None

    os.makedirs(output_dir, exist_ok=True)

    # ── Standard figure: sequential C1 → C2 → C3 → C4 → C5 → C6 ─────────────
    classified = classify_episodes_for_visualization(df, raw_episodes)
    counts     = {stage: len(eps) for stage, eps in classified.items()}

    print("  Episode classification (standard):")
    for stage in DRAW_ORDER:
        if counts[stage] > 0:
            print(f"    {STAGE_LABELS[stage]}: {counts[stage]}")

    output_path = os.path.join(output_dir, f"{patient_id}_overview")
    plot_overview_figure(
        patient_id, df, classified, timestep_min,
        output_path, days_per_row, n_rows,
    )

    # ── Grouped-context figure: C1 → C4+C5+C6 → C2 → C3 ─────────────────────
    classified_grouped = classify_episodes_grouped_context(df, raw_episodes)
    counts_grouped     = {stage: len(eps) for stage, eps in classified_grouped.items()}

    print("  Episode classification (grouped context):")
    for stage in GROUPED_DRAW_ORDER:
        if counts_grouped[stage] > 0:
            print(f"    {GROUPED_STAGE_LABELS[stage]}: {counts_grouped[stage]}")

    output_path_grouped = os.path.join(output_dir, f"{patient_id}_overview_grouped")
    plot_overview_figure(
        patient_id, df, classified_grouped, timestep_min,
        output_path_grouped, days_per_row, n_rows,
        draw_order=GROUPED_DRAW_ORDER,
        stage_labels=GROUPED_STAGE_LABELS,
        stage_colors_map=GROUPED_STAGE_COLORS,
    )

    return counts


# =============================================================================
# CLI
# =============================================================================

def make_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate 45-day CGM overview with episode classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--patient_id",
        default=DEFAULT_PATIENT_ID,
        help="Patient ID (default: patient with the most SLH episodes)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the figure",
    )
    parser.add_argument(
        "--days_per_row",
        type=int,
        default=DAYS_PER_ROW,
        help="Number of days per horizontal strip",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=N_ROWS,
        help="Number of strips in the figure",
    )
    return parser


def main() -> int:
    """Entry point — parse arguments and run the patient pipeline."""
    args = make_parser().parse_args()
    args.patient_id = resolve_patient_id(args.patient_id)

    print("\n" + "=" * 70)
    print("  45-Day Overview — Episode Classification by Criterion Level")
    print("=" * 70)
    print(f"  Patient    : {args.patient_id}")
    print(f"  Output     : {args.output}  (standard + grouped-context variants)")
    print(f"  Layout     : {args.n_rows} rows × {args.days_per_row} days / row")
    print(f"  Total days : {args.n_rows * args.days_per_row}")
    print("=" * 70)

    counts = generate_overview_for_patient(
        args.patient_id, args.output,
        args.days_per_row, args.n_rows,
    )

    if counts is None:
        print("\nFailed to generate overview.")
        return 1

    print("\nDone.")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
