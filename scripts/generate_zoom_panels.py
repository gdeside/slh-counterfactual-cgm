"""
scripts/generate_zoom_panels.py
===============================

Zoom Panels — Representative Episodes by Classification Level
-------------------------------------------------------------
Generate detailed zoom panels showing one representative episode per
classification stage (C1, C2, C3, C4, C5, C6, SLH).

Each panel shows a ±ZOOM_WINDOW_H hour window around the episode core,
featuring:
- Context windows (±CONTEXT_WIN_MIN min) highlighted
- Episode core shaded with classification stage color
- Threshold lines (H_HYPER, H_CONTEXT)
- Peak glucose annotation with duration and data-completeness
- Night shading

The representative episode for each stage is selected by its peak
glucose value being closest to the median peak glucose of that stage.

Output
------
Both PDF and SVG are written to *output_dir*/{patient_id}_zoom_panels.{pdf,svg}
via ``save_figure`` from ``src.visualization.style``.

Usage
-----

python3 -m scripts.generate_zoom_panels     --patient_id ID     --output /path/to/output/dir     --n_panels 6

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
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# =============================================================================
# PROJECT — Preprocessing
# =============================================================================
from src.utils.patient_selection import resolve_patient_id
from src.preprocessing.load_data import load_patient_window
from src.preprocessing.timestep import detect_dominant_sampling_interval

# =============================================================================
# PROJECT — Episode detection & classification
# =============================================================================
from src.episodes.detection import detect_hyperglycemia_episodes
from src.episodes.criteria import (
    check_c1_duration_bounds,
    check_c2_data_completeness,
    check_c3_daytime_only,
    check_c4_no_pre_episode_hyperglycemia,
    check_c5_no_post_episode_hyperglycemia,
    check_c6_or_crossing_condition,
    calculate_data_completeness,
)
from src.episodes.episode_config import (
    H_HYPER,
    H_CONTEXT,
    D_MIN_MIN,
    D_MAX_MIN,
    MIN_DATA_COMPLETENESS,
    CONTEXT_WIN_MIN,
    NIGHT_START_H,
    NIGHT_END_H,
    GAP_FACTOR,
    APPLY_C1,
    APPLY_C2,
    APPLY_C3,
    APPLY_C4,
    APPLY_C5,
    APPLY_C6,
    REQUIRED_SAMPLING_MIN,
)

# =============================================================================
# PROJECT — Configuration
# =============================================================================
from src.config import TIME_COL, GLUCOSE_COL, FIG_DIR

# =============================================================================
# PROJECT — Visualization
# =============================================================================
from src.visualization.style import apply_publication_style, save_figure
from src.visualization.colors import (
    STAGE_COLORS,
    COLOR_HYPER,
    COLOR_NIGHT,
    COLOR_CONTEXT,
    COLOR_ORIGINAL,
)
from src.visualization.glucose_axis import apply_glucose_axis
from src.utils.unit_conversion import format_glucose

# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================

DEFAULT_PATIENT_ID = None   # None = automatic: patient with the most SLH episodes
DEFAULT_OUTPUT_DIR = os.path.join(FIG_DIR, "SLH_analysis")

# Visualization parameters
ZOOM_WINDOW_H = 3              # Hours of signal on each side of episode
N_PANELS_MAX = 6               # Maximum number of zoom panels

ALPHA_EPISODE = 0.38
ALPHA_NIGHT   = 0.15

STAGE_LABELS = {
    "C1_duration":    f"Duration not in [{D_MIN_MIN}–{D_MAX_MIN} min]",
    "C2_incomplete":  f"Incomplete data (<{MIN_DATA_COMPLETENESS * 100:.0f}%)",
    "C3_nocturnal":   f"Nocturnal ({NIGHT_START_H:02d}:00–{NIGHT_END_H:02d}:00)",
    "C4_hyper_before": "Pre-episode hyperglycemia",
    "C5_hyper_after":  "Post-episode hyperglycemia",
    "C6_no_cross":    f"No crossing ≤{format_glucose(H_CONTEXT)}",
    "SLH":             "SLH  Passed all criteria",
}

DRAW_ORDER = [
    "C1_duration",
    "C2_incomplete",
    "C3_nocturnal",
    "C4_hyper_before",
    "C5_hyper_after",
    "C6_no_cross",
    "SLH",
]


# =============================================================================
# EPISODE CLASSIFICATION
# =============================================================================

def classify_episodes_for_visualization(df: pd.DataFrame,
                                        raw_episodes: list[dict]) -> dict:
    """
    Classify raw episodes sequentially through criteria C1–C6.

    Episodes are routed into the stage at which they first fail a criterion.
    Those that survive all criteria are labelled SLH.  The returned dict
    can be used directly by the panel-selection and plotting helpers.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed CGM data (TIME_COL, GLUCOSE_COL).
    raw_episodes : list of dict
        Raw detected episodes from ``detect_hyperglycemia_episodes``.

    Returns
    -------
    dict
        Mapping from stage name (``DRAW_ORDER`` keys) to a list of episode
        dicts, each containing ``onset``, ``offset``, ``duration_min``,
        ``peak_glucose``, ``peak_time``, ``completeness_episode``, and
        ``idx``.
    """
    if len(raw_episodes) == 0:
        return {stage: [] for stage in DRAW_ORDER}

    classified = {stage: [] for stage in DRAW_ORDER}
    n_episodes  = len(raw_episodes)
    active      = [True] * n_episodes

    def apply_criterion(criterion_name, check_function, episodes_subset):
        if criterion_name in ("C1", "C3"):
            return check_function(episodes_subset)
        return check_function(df, episodes_subset)

    def make_episode_dict(episode):
        rows = df.loc[episode["idx"]]
        valid_glucose = rows[GLUCOSE_COL].dropna()
        if valid_glucose.empty:
            peak_glucose = 0.0
            peak_time    = episode["onset_time"]
        else:
            pk_idx       = valid_glucose.idxmax()
            peak_glucose = float(rows.loc[pk_idx, GLUCOSE_COL])
            peak_time    = rows.loc[pk_idx, TIME_COL]

        ep_completeness = calculate_data_completeness(
            df, episode["onset_time"], episode["offset_time"], REQUIRED_SAMPLING_MIN
        )

        return {
            "onset":                episode["onset_time"],
            "offset":               episode["offset_time"],
            "duration_min":         episode["duration_min"],
            "peak_glucose":         peak_glucose,
            "peak_time":            peak_time,
            "completeness_episode": ep_completeness,
            "idx":                  episode["idx"],
        }

    criteria = [
        ("C1", check_c1_duration_bounds,               APPLY_C1, "C1_duration"),
        ("C2", check_c2_data_completeness,              APPLY_C2, "C2_incomplete"),
        ("C3", check_c3_daytime_only,                   APPLY_C3, "C3_nocturnal"),
        ("C4", check_c4_no_pre_episode_hyperglycemia,   APPLY_C4, "C4_hyper_before"),
        ("C5", check_c5_no_post_episode_hyperglycemia,  APPLY_C5, "C5_hyper_after"),
        ("C6", check_c6_or_crossing_condition,          APPLY_C6, "C6_no_cross"),
    ]

    for criterion_name, check_function, apply_flag, exclusion_stage in criteria:
        if not apply_flag:
            continue

        active_episodes = [ep for i, ep in enumerate(raw_episodes) if active[i]]
        active_indices  = [i  for i in range(n_episodes)            if active[i]]

        if not active_episodes:
            break

        results = apply_criterion(criterion_name, check_function, active_episodes)

        for idx, ep, result in zip(active_indices, active_episodes, results):
            if not result:
                active[idx] = False
                classified[exclusion_stage].append(make_episode_dict(ep))

    for i, ep in enumerate(raw_episodes):
        if active[i]:
            classified["SLH"].append(make_episode_dict(ep))

    return classified


# =============================================================================
# NIGHT SHADING
# =============================================================================

def _get_night_spans(t_start: pd.Timestamp, t_end: pd.Timestamp) -> list:
    """Return list of (span_start, span_end) night intervals within [t_start, t_end]."""
    spans = []
    day   = t_start.normalize()

    while day <= t_end + pd.Timedelta(days=1):
        night_start = day + pd.Timedelta(hours=NIGHT_START_H)
        night_end   = day + pd.Timedelta(hours=24 + NIGHT_END_H)

        span_start = max(night_start, t_start)
        span_end   = min(night_end,   t_end)

        if span_start < span_end:
            spans.append((span_start, span_end))

        day += pd.Timedelta(days=1)

    return spans


def _shade_nights(ax, t_start: pd.Timestamp, t_end: pd.Timestamp) -> None:
    """Shade night windows on *ax* without adding a legend label."""
    for span_start, span_end in _get_night_spans(t_start, t_end):
        ax.axvspan(span_start, span_end,
                   color=COLOR_NIGHT, alpha=ALPHA_NIGHT, zorder=0, label="_")


# =============================================================================
# EPISODE SELECTION
# =============================================================================

def select_representative_episodes(classified: dict,
                                   n_max: int = N_PANELS_MAX) -> list:
    """
    Select one representative episode per classification stage.

    The representative is the episode whose peak glucose is closest to the
    median peak glucose of its stage.  If fewer than *n_max* stages have
    episodes, additional SLH episodes are appended to fill the grid.

    Parameters
    ----------
    classified : dict
        Output of ``classify_episodes_for_visualization``.
    n_max : int
        Maximum number of panels.

    Returns
    -------
    list of (episode_dict, stage_name) tuples
    """
    selected = []

    for stage in DRAW_ORDER:
        eps = classified.get(stage, [])
        if not eps:
            continue
        median_peak = np.median([e["peak_glucose"] for e in eps])
        ep = min(eps, key=lambda e: abs(e["peak_glucose"] - median_peak))
        selected.append((ep, stage))

    if len(selected) > n_max:
        selected = selected[-n_max:]

    if len(selected) < n_max:
        slh_pool   = classified.get("SLH", [])
        used_times = {e["onset"] for e, _ in selected}
        for ep in slh_pool:
            if len(selected) >= n_max:
                break
            if ep["onset"] not in used_times:
                selected.append((ep, "SLH"))
                used_times.add(ep["onset"])

    return selected


# =============================================================================
# ZOOM PANEL PLOTTING
# =============================================================================

def plot_single_zoom_panel(ax, df: pd.DataFrame, episode: dict, stage: str,
                           timestep_min: float,
                           window_hours: float = ZOOM_WINDOW_H) -> None:
    """
    Draw a single zoom panel for one episode.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    df : pd.DataFrame
        Full patient CGM data.
    episode : dict
        Episode dict produced by ``classify_episodes_for_visualization``.
    stage : str
        Classification stage key (used for colour and title).
    timestep_min : float
        Dominant sampling interval (minutes), used for gap-aware trace.
    window_hours : float
        Hours of context shown on each side of the episode core.
    """
    color  = STAGE_COLORS[stage]
    win_td = pd.Timedelta(hours=window_hours)
    ctx    = pd.Timedelta(minutes=CONTEXT_WIN_MIN)

    t_lo = max(episode["onset"] - win_td, df[TIME_COL].iloc[0])
    t_hi = min(episode["offset"] + win_td, df[TIME_COL].iloc[-1])

    mask  = (df[TIME_COL] >= t_lo) & (df[TIME_COL] <= t_hi)
    t_win = df.loc[mask, TIME_COL]
    g_win = df.loc[mask, GLUCOSE_COL].astype(float)

    # Break the trace at gaps larger than GAP_FACTOR × timestep
    diffs  = t_win.diff().dt.total_seconds().fillna(0) / 60
    g_plot = g_win.copy()
    g_plot[diffs > timestep_min * GAP_FACTOR] = np.nan

    _shade_nights(ax, t_lo, t_hi)

    # Context windows
    ax.axvspan(episode["onset"] - ctx, episode["onset"],
               color=COLOR_CONTEXT, alpha=0.09, zorder=1)
    ax.axvspan(episode["offset"], episode["offset"] + ctx,
               color=COLOR_CONTEXT, alpha=0.09, zorder=1)

    # Episode core
    ax.axvspan(episode["onset"], episode["offset"],
               color=color, alpha=ALPHA_EPISODE + 0.10, zorder=2)

    # Threshold lines
    ax.axhline(H_HYPER,   color=COLOR_HYPER,   ls="--", lw=1.0, zorder=4)
    ax.axhline(H_CONTEXT, color=COLOR_CONTEXT, ls=":",  lw=1.0, zorder=4)

    # Onset / offset markers
    for tm in (episode["onset"], episode["offset"]):
        ax.axvline(tm, color="#777777", lw=0.7, ls="-", alpha=0.45, zorder=5)

    # CGM trace
    ax.plot(t_win, g_plot, color=COLOR_ORIGINAL, lw=1.2, zorder=6)
    ax.scatter(t_win, g_win, color=COLOR_ORIGINAL, s=5, alpha=0.4, zorder=7)

    # Peak annotation
    pk_t, pk_v = episode["peak_time"], episode["peak_glucose"]
    ax.scatter([pk_t], [pk_v], marker="^", s=20, color=COLOR_HYPER, zorder=9)

    annot_text = f"{format_glucose(pk_v)}\n{episode['duration_min']:.0f} min"
    if "completeness_episode" in episode:
        annot_text += f"\nData: {episode['completeness_episode']:.0%}"

    ax.annotate(
        annot_text,
        xy=(pk_t, pk_v),
        xytext=(pk_t + pd.Timedelta(minutes=18), pk_v + 25),
        fontsize=6.5, color=COLOR_HYPER, ha="left", va="bottom",
        arrowprops=dict(arrowstyle="->", color=COLOR_HYPER, lw=0.7),
        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                  ec=COLOR_HYPER, lw=0.6, alpha=0.92),
        zorder=10,
    )

    # Axis formatting
    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(70, 350)

    span_min = (t_hi - t_lo).total_seconds() / 60
    interval = 1 if span_min <= 300 else 2
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=6.5, rotation=30, ha="right")

    apply_glucose_axis(ax, fontsize=7, labelpad=2)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=0.22, lw=0.5)

    ax.set_title(f"  {STAGE_LABELS[stage]}",
                 fontsize=7, fontweight="bold", pad=4, loc="left", color=color)


def plot_zoom_panels_figure(patient_id: str, df: pd.DataFrame,
                            classified: dict, timestep_min: float,
                            output_path: str,
                            n_panels: int = N_PANELS_MAX) -> None:
    """
    Compose and save the zoom-panels figure.

    Parameters
    ----------
    patient_id : str
        Patient identifier (used for the figure title).
    df : pd.DataFrame
        Full patient CGM data.
    classified : dict
        Output of ``classify_episodes_for_visualization``.
    timestep_min : float
        Dominant sampling interval (minutes).
    output_path : str
        Destination path without extension; PDF and SVG are written.
    n_panels : int
        Maximum number of panels.
    """
    apply_publication_style()

    selected = select_representative_episodes(classified, n_panels)
    if not selected:
        print("  [!] No episodes for zoom panels.")
        return

    n      = len(selected)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7.24, 2.80 * n_rows),
        constrained_layout=True,
    )

    if n_rows == 1:
        axes = np.array(axes).reshape(1, -1)
    axes_flat = np.array(axes).reshape(-1)

    for idx, (episode, stage) in enumerate(selected):
        plot_single_zoom_panel(axes_flat[idx], df, episode, stage, timestep_min)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    handles = [
        Line2D([0], [0], color=COLOR_ORIGINAL, lw=1.2,
               label="CGM signal"),
        Line2D([0], [0], color=COLOR_HYPER, ls="--", lw=1.0,
               label=f"H_HYPER = {format_glucose(H_HYPER)}"),
        Line2D([0], [0], color=COLOR_CONTEXT, ls=":", lw=1.0,
               label=f"H_CONTEXT = {format_glucose(H_CONTEXT)}"),
        mpatches.Patch(facecolor=COLOR_CONTEXT, alpha=0.12,
                       label=f"Context (±{CONTEXT_WIN_MIN} min)"),
        mpatches.Patch(facecolor=COLOR_NIGHT, alpha=ALPHA_NIGHT + 0.10,
                       label=f"Night ({NIGHT_START_H:02d}–{NIGHT_END_H:02d}h)"),
    ]

    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.05))

    save_figure(fig, output_path)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def generate_zoom_panels_for_patient(patient_id: str, output_dir: str,
                                     n_panels: int = N_PANELS_MAX) -> int:
    """
    Full pipeline: load data, classify episodes, generate figure for one patient.

    Parameters
    ----------
    patient_id : str
        Patient identifier.
    output_dir : str
        Output directory.
    n_panels : int
        Maximum number of panels.

    Returns
    -------
    int
        Number of panels generated, or 0 on failure.
    """
    print(f"\nProcessing patient {patient_id}...")

    df = load_patient_window(patient_id)
    if df is None:
        print(f"  Failed to load data for {patient_id}")
        return 0

    print(f"  Loaded {len(df):,} data points")

    timestep_min = detect_dominant_sampling_interval(df)
    print(f"  Detected sampling interval: {timestep_min:.0f} min")

    raw_episodes = detect_hyperglycemia_episodes(df)
    print(f"  Detected {len(raw_episodes)} raw hyperglycemia episodes")

    if not raw_episodes:
        print("  No episodes detected")
        return 0

    classified = classify_episodes_for_visualization(df, raw_episodes)

    print("\n  Episode classification:")
    for stage in DRAW_ORDER:
        count = len(classified[stage])
        if count > 0:
            print(f"    {stage}: {count}")

    selected = select_representative_episodes(classified, n_panels)
    print(f"\n  Selected {len(selected)} representative episodes for zoom panels:")
    for ep, stage in selected:
        print(f"    {stage}: {ep['onset'].strftime('%Y-%m-%d %H:%M')} "
              f"(peak {ep['peak_glucose']:.0f} mg/dL, {ep['duration_min']:.0f} min)")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{patient_id}_zoom_panels")

    plot_zoom_panels_figure(patient_id, df, classified, timestep_min,
                            output_path, n_panels)

    print(f"\n  Saved figure: {output_path}.pdf/.svg")
    return len(selected)


# =============================================================================
# CLI
# =============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate zoom panels for representative episodes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--patient_id", default=DEFAULT_PATIENT_ID,
                        help="Patient ID (default: patient with the most SLH episodes)")
    parser.add_argument("--output",     default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for figure")
    parser.add_argument("--n_panels",   type=int, default=N_PANELS_MAX,
                        help="Maximum number of zoom panels")
    return parser


def main() -> int:
    """Entry point."""
    args = parse_arguments().parse_args()
    args.patient_id = resolve_patient_id(args.patient_id)

    print("\n" + "=" * 70)
    print("  Zoom Panels — Representative Episodes")
    print("=" * 70)
    print(f"  Patient:      {args.patient_id}")
    print(f"  Output:       {args.output}")
    print(f"  Max panels:   {args.n_panels}")
    print("=" * 70)

    n_generated = generate_zoom_panels_for_patient(
        args.patient_id, args.output, args.n_panels
    )

    if n_generated == 0:
        print("\n  Failed to generate zoom panels")
        return 1

    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
