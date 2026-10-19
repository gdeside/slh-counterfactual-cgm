"""
scripts/paper_figures/generate_figure_3_patient_comparison.py
=============================================================
Figure 3 (main text): glucose profiles and SLH recurrence patterns in two
representative patients with similar CGM summary metrics but contrasting SLH
frequency (high vs low SLH frequency; patients chosen via CGM_PATIENT_HIGH / CGM_PATIENT_LOW,
or automatically: most / fewest SLH episodes).

Derived from `generate_classification_comparison_figure` in
scripts/patient_comparison.py, which produced this figure until now. The
panel structure is unchanged:

  A / B — Ambulatory glucose profile (AGP), patient A / patient B
  C / D — Episode recurrence by time of day, patient A / patient B
  E / F — Representative 7-day glucose trace, patient A / patient B
  G     — Summary table: standard CGM metrics and episode counts

What changed relative to patient_comparison.py
----------------------------------------------
Panels C/D used the BINARY `classification` column (SLH vs LLH) written by
the detection pipeline. Under that split, every episode that is not a
confirmed SLH — including nocturnal would-be SLH and episodes that fail the
isolation criteria — is drawn as LLH, so the night hours of the recurrence
panel showed nothing but LLH and the two most informative intermediate
categories were invisible.

Panels C/D (and the counts in panel G) now use the same FOUR-category
classification as Figure 2 — SLH / Not isolated / Night / LLH — imported from
generate_figure_2_classification_summary so that the two figures cannot drift
apart. "Night" is now visible as its own band in the 22:00–06:00 hours, and
"Not isolated" is separated from true long-lasting hyperglycemia.

Note that the fine classification first drops episodes failing C2 (data
completeness), exactly as Figure 2 does, so per-patient totals here match the
Figure 2 waterfall rather than the raw episode-file row count.

Panels E/F now shade episodes with the same fine 4-category CATEGORY_COLORS
as C/D and G (they call the shared `plot_7day_trace`, passing the fine
`category` column and CATEGORY_COLORS instead of the pipeline's binary
SLH/LLH pair it defaults to) — the binary pair used SLH=navy/LLH=rose,
clashing with C/D's SLH=orange/LLH=blue within the same figure. A legend on
panel F now spells out what each shaded colour means (previously unlabelled).

Usage
-----
    python3 -m scripts.paper_figures.generate_figure_3_patient_comparison

Outputs (FIG_DIR / Manuscript_Figures)
---------------------------------------
  Figure3_Patient_Comparison_{A}_vs_{B}/Figure3_Patient_Comparison_{A}_vs_{B}.{pdf,svg}   (whole figure)
  Figure3_Patient_Comparison_{A}_vs_{B}/..._panel_{A-G}.{pdf,svg}
  figure3_metrics_{A}_vs_{B}.csv

Authors: Guillaume Deside
"""
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.utils.patient_selection import resolve_patient_pair
from src.config.constants import TIME_COL, GLUCOSE_COL, COL_EPISODE_START, COL_EPISODE_END
from src.config.figure_config import FIG_DIR

from src.preprocessing.load_data import load_patient_window, prepare_valid_data
from src.preprocessing.timestep import detect_dominant_sampling_interval
from src.agp.compute_agp import compute_agp_profile
from src.visualization.style import apply_publication_style, save_figure_with_panels, style_axes
from src.visualization.colors import COLOR_RECURRENCE_THRESHOLD
from src.visualization.agp_plot import plot_agp_panel, plot_7day_trace, add_panel_label
from src.visualization.glucose_axis import glucose_cell
from src.utils.unit_conversion import glucose_unit_label

# The fine 4-category scheme and its colours are owned by the Figure 2 module:
# importing them (rather than restating them) is what keeps Figures 2 and 3 on
# one classification and one palette.
from scripts.paper_figures.generate_figure_2_classification_summary import (
    CATEGORY_ORDER,
    CATEGORY_COLORS,
    load_and_classify_fine,
)


# =============================================================================
# PARAMETERS
# =============================================================================
# High- and low-SLH-frequency patients. Automatic by default (most / fewest SLH episodes);
# to reproduce the manuscript use CGM_PATIENT_HIGH=<id> CGM_PATIENT_LOW=<id> (see README).
PATIENT_A, PATIENT_B = resolve_patient_pair()

OUTPUT_DIR = os.path.join(FIG_DIR, "Manuscript_Figures")
OUTPUT_NAME = f"Figure3_Patient_Comparison_{PATIENT_A}_vs_{PATIENT_B}"

FIGURE_WIDTH = 7.24
FIGURE_HEIGHT = 9.6

RECURRENCE_SYSTEMATIC_THRESHOLD = 50.0   # % of days — "systematic" reference line
RECURRENCE_X_TICK_INTERVAL_HOURS = 3
RECURRENCE_Y_MAX_BUFFER = 1.15
ALPHA_RECURRENCE_FILL = 0.85

# Panels E/F (7-day trace) shade episode spans by the same fine category as
# C/D/G, via CATEGORY_COLORS -- SLH gets the highest alpha since it is the
# category the figure is about, the other three stay lower so they read as
# context rather than competing with it.
TRACE_CATEGORY_ALPHAS = {"SLH": 0.32, "Not isolated": 0.20, "Night": 0.20, "LLH": 0.20}

# Night window, for the shaded band behind the recurrence panels. Kept in sync
# with the C3 criterion used by the classifier (see episode_config).
from src.episodes.episode_config import NIGHT_START_H, NIGHT_END_H
from src.visualization.colors import COLOR_NIGHT


# =============================================================================
# METRICS + COUNTS
# =============================================================================
def compute_cgm_metrics(df: pd.DataFrame) -> dict:
    """Standard CGM summary metrics (unchanged from patient_comparison.py)."""
    if len(df) < 10:
        return {}
    g = df[GLUCOSE_COL]
    n = len(df)
    mean_glucose = g.mean()
    return {
        "Mean":      round(mean_glucose, 1),
        "Median":    round(g.median(), 1),
        "SD":        round(g.std(), 1),
        "CV":        round(g.std() / mean_glucose * 100 if mean_glucose > 0 else 0, 1),
        "TIR70-180": round(((g >= 70) & (g <= 180)).sum() / n * 100, 1),
        "TAR>180":   round((g > 180).sum() / n * 100, 1),
        "TAR>250":   round((g > 250).sum() / n * 100, 1),
        "TBR<70":    round((g < 70).sum() / n * 100, 1),
    }


def count_episodes_by_fine_category(episodes_fine: pd.DataFrame) -> dict:
    """Episode counts for the 4 fine categories (was: SLH/LLH only)."""
    counts = episodes_fine["category"].value_counts().to_dict()
    return {cat: int(counts.get(cat, 0)) for cat in CATEGORY_ORDER}


# =============================================================================
# PANELS C/D: recurrence by FINE category
# =============================================================================
def compute_recurrence_by_fine_category(episodes_fine: pd.DataFrame,
                                        resolution_min: int,
                                        n_days: int,
                                        start_date) -> dict:
    """Per-time-of-day recurrence, split by the 4 fine categories.

    Same slot-occupancy construction as `compute_recurrence_by_classification`
    in src/agp/compute_agp.py — for every episode, each resolution-wide slot it
    covers is counted once per day — but keyed on the fine `category` column
    instead of the binary `classification` one, so "Night" and "Not isolated"
    get their own bands instead of being folded into LLH.
    """
    n_slots = int(24 * 60 / resolution_min)
    occurrence = {cat: np.zeros(n_slots) for cat in CATEGORY_ORDER}

    for _, episode in episodes_fine.iterrows():
        category = episode["category"]
        if category not in occurrence:
            continue

        current_time = episode[COL_EPISODE_START]
        seen_keys = set()
        while current_time <= episode[COL_EPISODE_END]:
            day_offset = (current_time.date() - start_date).days
            slot = min(int((current_time.hour * 60 + current_time.minute) / resolution_min),
                       n_slots - 1)
            key = (day_offset, slot)
            if key not in seen_keys:
                occurrence[category][slot] += 1
                seen_keys.add(key)
            current_time += timedelta(minutes=resolution_min)

    for cat in CATEGORY_ORDER:
        occurrence[cat] = occurrence[cat] / n_days * 100
    return occurrence


def plot_recurrence_fine_panel(ax, occurrence: dict, patient_id: str, n_slh: int,
                               resolution_min: int,
                               show_legend: bool = True,
                               show_threshold_annotation: bool = True,
                               y_max: float | None = None) -> None:
    """Stacked recurrence bands, one per fine category.

    Stacking order follows CATEGORY_ORDER (SLH first, at the bottom) so the
    SLH band always sits on the axis and can be read directly against it —
    it is the quantity the figure is about.
    """
    n_slots = int(24 * 60 / resolution_min)
    time_hours = np.arange(n_slots) * resolution_min / 60
    bottom = np.zeros(n_slots)

    # Night window behind the data, so the "Night" band is visibly confined to it.
    ax.axvspan(NIGHT_START_H, 24, color=COLOR_NIGHT, alpha=0.5, zorder=0)
    ax.axvspan(0, NIGHT_END_H, color=COLOR_NIGHT, alpha=0.5, zorder=0)

    for cat in CATEGORY_ORDER:
        ax.fill_between(time_hours, bottom, bottom + occurrence[cat],
                        color=CATEGORY_COLORS[cat], alpha=ALPHA_RECURRENCE_FILL,
                        label=cat, edgecolor="white", linewidth=0.4, zorder=2)
        bottom += occurrence[cat]

    ax.axhline(RECURRENCE_SYSTEMATIC_THRESHOLD, color=COLOR_RECURRENCE_THRESHOLD,
               linestyle="--", linewidth=1.2, zorder=5)

    # The shared helper this panel replaces floored the axis at 100%, which
    # left the stacked bands compressed into the bottom third and made the two
    # thin categories (Night, Not isolated) unreadable. Scale to the data
    # instead, and use ONE limit for both patients (passed in by the caller) so
    # panels C and D stay directly comparable — that comparison is the point of
    # the figure.
    if y_max is None:
        y_max = max(float(bottom.max()) * RECURRENCE_Y_MAX_BUFFER,
                    RECURRENCE_SYSTEMATIC_THRESHOLD * 1.05)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, y_max)
    ax.set_xticks(range(0, 25, RECURRENCE_X_TICK_INTERVAL_HOURS))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, RECURRENCE_X_TICK_INTERVAL_HOURS)])
    ax.set_xlabel("Time of day", fontsize=9, labelpad=4)
    ax.set_ylabel("% of days with hyperglycemia", fontsize=8, labelpad=4)
    ax.set_title(f"{patient_id} — {n_slh} SLH episodes", fontsize=8.5,
                 fontweight="bold", loc="left", pad=4)
    ax.tick_params(axis="both", which="major", labelsize=8)

    if show_threshold_annotation:
        ax.text(24, RECURRENCE_SYSTEMATIC_THRESHOLD,
                f"  {RECURRENCE_SYSTEMATIC_THRESHOLD:.0f}%",
                fontsize=7.5, va="center", ha="left",
                color=COLOR_RECURRENCE_THRESHOLD, fontweight="bold", clip_on=False)

    if show_legend:
        ax.legend(loc="upper right", fontsize=6.5, framealpha=0.88, ncol=2,
                  handlelength=1.4, columnspacing=1.0, labelspacing=0.35)

    style_axes(ax)


# =============================================================================
# PANEL G: summary table (fine counts)
# =============================================================================
def plot_metrics_table_fine(ax, metrics_a: dict, metrics_b: dict,
                            counts_a: dict, counts_b: dict,
                            patient_a: str, patient_b: str) -> None:
    """Metrics table with the 4 fine episode counts.

    Local to this figure rather than an edit to the shared
    `plot_metrics_table`, which other scripts still call with binary counts.
    Splitting "n non-SLH" into LLH / Not isolated / Night is what makes the
    per-patient totals in this table reconcilable with the Figure 2 waterfall.
    """
    ax.axis("off")

    n_total_a = sum(counts_a.values())
    n_total_b = sum(counts_b.values())

    col_labels = [
        "n\nepisodes", "n\nSLH", "n\nNot isol.", "n\nNight", "n\nLLH",
        f"Mean\n({glucose_unit_label()})", f"Median\n({glucose_unit_label()})",
        f"SD\n({glucose_unit_label()})", "CV\n(%)",
        "TIR 70-180\n(%)", "TAR >180\n(%)", "TAR >250\n(%)", "TBR <70\n(%)",
    ]

    def _row(counts, metrics, n_total):
        return [
            n_total, counts["SLH"], counts["Not isolated"], counts["Night"], counts["LLH"],
            glucose_cell(metrics.get("Mean", "-")),
            glucose_cell(metrics.get("Median", "-")),
            glucose_cell(metrics.get("SD", "-")),
            metrics.get("CV", "-"),
            metrics.get("TIR70-180", "-"), metrics.get("TAR>180", "-"),
            metrics.get("TAR>250", "-"), metrics.get("TBR<70", "-"),
        ]

    table_data = [_row(counts_a, metrics_a, n_total_a), _row(counts_b, metrics_b, n_total_b)]

    COLOR_HEADER_BG    = "#E8E8E8"
    COLOR_HEADER_TEXT  = "#1A1A1A"
    COLOR_ALT_ROW_BG   = "#F9F9F9"
    COLOR_DATA_TEXT    = "#2C2C2C"

    table = ax.table(cellText=table_data, rowLabels=[patient_a, patient_b],
                     colLabels=col_labels, cellLoc="center", loc="center",
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(7)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.5)
        if row == 0:                                  # header
            cell.set_facecolor(COLOR_HEADER_BG)
            cell.set_text_props(fontweight="bold", color=COLOR_HEADER_TEXT)
        elif col == -1:                               # row labels
            cell.set_facecolor(COLOR_HEADER_BG)
            cell.set_text_props(fontweight="bold", color=COLOR_HEADER_TEXT)
        else:
            cell.set_facecolor("white" if row == 1 else COLOR_ALT_ROW_BG)
            cell.set_text_props(color=COLOR_DATA_TEXT)

    # Tint the four episode-count headers with their category colour, so the
    # table reads as the same scheme as panels C/D.
    for col, cat in {1: "SLH", 2: "Not isolated", 3: "Night", 4: "LLH"}.items():
        header_cell = table[0, col]
        header_cell.set_facecolor(CATEGORY_COLORS[cat])
        header_cell.set_text_props(fontweight="bold", color="white")


# =============================================================================
# PATIENT DATA
# =============================================================================
def process_patient_data(patient_id: str) -> dict | None:
    window_df = load_patient_window(patient_id)
    if window_df is None:
        return None

    valid_data = prepare_valid_data(window_df)
    if len(valid_data) < 10:
        return None

    episodes_fine, n_dropped = load_and_classify_fine(patient_id)
    if episodes_fine is None or len(episodes_fine) == 0:
        return None

    resolution = detect_dominant_sampling_interval(valid_data)
    agp = compute_agp_profile(valid_data, resolution)
    metrics = compute_cgm_metrics(valid_data)
    counts = count_episodes_by_fine_category(episodes_fine)

    start_date = valid_data[TIME_COL].dt.date.min()
    end_date = valid_data[TIME_COL].dt.date.max()
    n_days = (end_date - start_date).days + 1

    recurrence = compute_recurrence_by_fine_category(
        episodes_fine, resolution, n_days, start_date)

    print(f"    {patient_id}: {sum(counts.values())} episodes "
          f"({', '.join(f'{c}={counts[c]}' for c in CATEGORY_ORDER)}; "
          f"{n_dropped} dropped for C2)")

    return {
        "agp": agp,
        "metrics": metrics,
        "counts": counts,
        "resolution": resolution,
        "recurrence": recurrence,
        "valid_data": valid_data,
        "episodes": episodes_fine,
        "n_dropped_c2": n_dropped,
    }


# =============================================================================
# FIGURE
# =============================================================================
def shared_recurrence_y_max(*occurrences: dict) -> float:
    """One y-limit for both recurrence panels: the tallest stacked total across
    the patients shown, with headroom, never below the systematic-recurrence
    reference line so that line stays on-scale in both."""
    stacked_max = max(
        float(sum(occ[cat] for cat in CATEGORY_ORDER).max()) for occ in occurrences
    )
    return max(stacked_max * RECURRENCE_Y_MAX_BUFFER,
               RECURRENCE_SYSTEMATIC_THRESHOLD * 1.05)


def build_figure(patient_a: str, patient_b: str, data_a: dict, data_b: dict):
    """Return (figure, {panel letter: axes})."""
    apply_publication_style()

    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 0.8, 1.2],
                  hspace=0.45, wspace=0.28,
                  left=0.09, right=0.95, top=0.95, bottom=0.05)

    # Row 1 — AGP
    ax_agp_a = fig.add_subplot(gs[0, 0])
    ax_agp_b = fig.add_subplot(gs[0, 1])
    plot_agp_panel(ax_agp_a, data_a["agp"], patient_a,
                   show_legend=False, show_threshold_annotations=False)
    plot_agp_panel(ax_agp_b, data_b["agp"], patient_b,
                   show_legend=True, show_threshold_annotations=True)
    add_panel_label(ax_agp_a, "A")
    add_panel_label(ax_agp_b, "B")

    # Row 2 — recurrence by FINE category
    ax_rec_a = fig.add_subplot(gs[1, 0])
    ax_rec_b = fig.add_subplot(gs[1, 1])
    rec_y_max = shared_recurrence_y_max(data_a["recurrence"], data_b["recurrence"])
    plot_recurrence_fine_panel(ax_rec_a, data_a["recurrence"], patient_a,
                               data_a["counts"]["SLH"], data_a["resolution"],
                               show_legend=False, show_threshold_annotation=False,
                               y_max=rec_y_max)
    plot_recurrence_fine_panel(ax_rec_b, data_b["recurrence"], patient_b,
                               data_b["counts"]["SLH"], data_b["resolution"],
                               show_legend=True, show_threshold_annotation=True,
                               y_max=rec_y_max)
    add_panel_label(ax_rec_a, "C")
    add_panel_label(ax_rec_b, "D")

    # Row 3 — 7-day traces, shaded by the same fine category scheme as C/D/G
    # (CATEGORY_COLORS) rather than the pipeline's binary SLH/LLH pair, so the
    # whole figure reads off one consistent palette; legend on the right panel
    # only, matching the row-1/row-2 convention above.
    ax_trace_a = fig.add_subplot(gs[2, 0])
    ax_trace_b = fig.add_subplot(gs[2, 1])
    plot_7day_trace(ax_trace_a, data_a["valid_data"], data_a["episodes"], patient_a,
                    show_threshold_annotations=False,
                    category_col="category", category_colors=CATEGORY_COLORS,
                    category_alphas=TRACE_CATEGORY_ALPHAS, category_order=CATEGORY_ORDER,
                    show_category_legend=False)
    plot_7day_trace(ax_trace_b, data_b["valid_data"], data_b["episodes"], patient_b,
                    show_threshold_annotations=True,
                    category_col="category", category_colors=CATEGORY_COLORS,
                    category_alphas=TRACE_CATEGORY_ALPHAS, category_order=CATEGORY_ORDER,
                    show_category_legend=True)
    add_panel_label(ax_trace_a, "E")
    add_panel_label(ax_trace_b, "F")

    # Row 4 — summary table
    ax_metrics = fig.add_subplot(gs[3, :])
    plot_metrics_table_fine(ax_metrics, data_a["metrics"], data_b["metrics"],
                            data_a["counts"], data_b["counts"], patient_a, patient_b)
    add_panel_label(ax_metrics, "G")

    panels = {"A": ax_agp_a, "B": ax_agp_b, "C": ax_rec_a, "D": ax_rec_b,
              "E": ax_trace_a, "F": ax_trace_b, "G": ax_metrics}
    return fig, panels


def generate_figure_3_patient_comparison() -> None:
    print("\n" + "=" * 70)
    print(f"  GENERATING FIGURE 3 — {PATIENT_A} vs {PATIENT_B}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n  Loading patients ...")
    data_a = process_patient_data(PATIENT_A)
    data_b = process_patient_data(PATIENT_B)
    if data_a is None or data_b is None:
        print("  FAILED: could not load one or both patients.")
        return

    rows = []
    for pid, data in [(PATIENT_A, data_a), (PATIENT_B, data_b)]:
        row = {"patient_id": pid}
        row.update(data["metrics"])
        for cat in CATEGORY_ORDER:
            row[f"n_{cat.replace(' ', '_')}"] = data["counts"][cat]
        row["n_total"] = sum(data["counts"].values())
        rows.append(row)
    csv_path = os.path.join(OUTPUT_DIR, f"figure3_metrics_{PATIENT_A}_vs_{PATIENT_B}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    fig, panels = build_figure(PATIENT_A, PATIENT_B, data_a, data_b)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    save_figure_with_panels(fig, output_path, OUTPUT_NAME, panels)

    print("\n" + "=" * 70)
    print(f"  SAVED: {output_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    generate_figure_3_patient_comparison()
