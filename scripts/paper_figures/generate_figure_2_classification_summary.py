"""
scripts/paper_figures/generate_figure_2_classification_summary.py
=================================================================
Figure 2 (main text): SLH/LLH episode classification summary.

Promoted from Supplementary Figure 2 to a main-text figure. Cohort-wide
support figure for the episode classification scheme (see `generate_supplementary_figure_3_slh_detection_schema.py`
for the schema itself):

  A — Simplified classification waterfall (adapted from panel A of
      `plot_simplified_funnel` in scripts/statistics/slh_llh_stats.py):
      All hyperglycemia episodes -> Duration -> Context (C4+C5+C6) ->
      Data completeness (C2) -> Daytime (C3) -> SLH. Bars for the
      intermediate (still-mixed) steps are neutral grey; the final SLH bar
      and the exclusion annotations use the same category colors as panels
      B-D below (LLH = duration exclusion, Not isolated = context/completeness
      exclusion, Night = daytime exclusion), so the funnel and the per-patient
      panels read as one consistent color scheme.
  B — Distribution of the number of SLH episodes per patient (daytime only —
      SLH requires C3 (onset outside the night window) to pass, so by
      construction every SLH episode already IS a daytime episode; the
      title makes this explicit rather than leaving it implicit)
  C — Per-patient category composition, as a proportion of each patient's
      own total episodes (stacked to 1.0), patients along the x-axis ordered
      by total episode count (ascending), with that total overlaid as a line
      on a secondary axis
  D — Onset-time scatter, cohort-wide: episode count per 15-min onset bin,
      one line per category (SLH / Not isolated / Night / LLH), connected
      point-to-point with no smoothing — same construction as
      `plot_onset_scatter_spline` in the notebook (§3.4)

Two panels of the previous six-panel version were removed as redundant:
  * the "exclusions by criterion" bar chart plotted exactly the five
    step-to-step deltas that panel A already prints as annotations;
  * the "category counts" panel was the panel-C composition before
    normalisation, i.e. the same data twice — the only thing it added was
    each patient's total episode count, now overlaid on panel C.
With those gone, the four surviving panels appear in the order the Results
text cites them (A -> B -> C -> D), so no sentence in the manuscript needs
to move.

Panel C's vertical-bar construction follows
`plot_cohort_category_composition_vertical` in the notebook (§3.5), except
no patient is highlighted (that notebook figure calls out its worked
examples; this one characterises the whole cohort, not individual patients).
Individual patient IDs are not shown on the x-axis (135 rotated ID labels are
illegible in a printed document) — the axis label states the ordering rule
instead; see `classification_summary_per_patient.csv` for the per-patient data.

Episode categorisation (C1–C6 already computed by the detection pipeline)
--------------------------------------------------------------------------
  1. C1 fails                                -> LLH            (duration outside [D_MIN_MIN, D_MAX_MIN])
  2. C1 passes, C2/C4/C5/C6 not all pass      -> Not isolated   (fails completeness and/or isolation)
  3. C1, C2, C4, C5, C6 all pass; C3 fails    -> Night           (would be SLH, but occurs at night)
  4. C1, C2, C3, C4, C5, C6 all pass          -> SLH             (short, isolated, complete, daytime)

Night (C3) is evaluated LAST, after isolation and completeness — matching
the paper's definition where day/night is a sub-classification of SLH, not
a competing criterion. (This corrects the previous Supplementary Figure 2,
which checked C3 before C4/C5/C6 and could misclassify a non-isolated
nighttime episode as "Night" instead of "Not isolated".)

Usage
-----
    python3 -m scripts.paper_figures.generate_figure_2_classification_summary

Outputs (FIG_DIR / Manuscript_Figures)
---------------------------------------
  Figure2_Classification_Summary.{pdf,svg}
  classification_summary_per_patient.csv

Authors: Guillaume Deside
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.config import FILTER_STATUS_CSV, PREPROCESSING_PATH, FIG_DIR
from src.config.paths import EPISODES_PATH
from src.preprocessing.load_data import discover_patients
from src.filtering.load_filtering import load_filtered_patient_list
from src.episodes.episode_config import D_MIN_MIN, D_MAX_MIN, NIGHT_START_H, NIGHT_END_H, MIN_DATA_COMPLETENESS
from src.visualization.style import apply_publication_style, save_figure, style_axes, FONT_SIZE_BASE
from src.visualization.colors import (WONG_COLORS, COLOR_NIGHT, COLOR_NIGHT_CATEGORY,
                                       COLOR_ANNOTATION)

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR  = os.path.join(FIG_DIR, "Manuscript_Figures")
OUTPUT_NAME = "Figure2_Classification_Summary"

ONSET_SCATTER_BIN_WIDTH_MIN = 15.0  # matches RESOLUTION_MIN used elsewhere in the project

CATEGORY_ORDER  = ["SLH", "Not isolated", "Night", "LLH"]
CATEGORY_COLORS = {
    "SLH":          WONG_COLORS["orange"],         # short, isolated, complete, daytime
    "Not isolated": WONG_COLORS["reddish_purple"],  # fails isolation and/or completeness
    "Night":        COLOR_NIGHT_CATEGORY,           # dark navy — would be SLH, but at night
    "LLH":          WONG_COLORS["blue"],            # duration bounds fail
}

# Neutral greys for the waterfall's intermediate (still-mixed) steps — these
# pools contain a mix of what later becomes SLH/Not isolated/Night, so they
# deliberately do NOT borrow a single category color.
WATERFALL_NEUTRAL_COLORS = ["#767676", "#8f8f8f", "#a8a8a8", "#c1c1c1"]

# Panel C's secondary-axis line (total episodes per patient) is an annotation
# of the x ordering, not a fifth episode category — so it reuses the palette's
# neutral annotation grey rather than introducing another one.
COLOR_TOTAL_LINE = COLOR_ANNOTATION


# =============================================================================
# EPISODE LOADING + CLASSIFICATION
# =============================================================================
def assign_fine_category(episodes_df: pd.DataFrame) -> pd.DataFrame:
    """Refine the binary SLH/LLH split into the 4 diagnostic categories in
    CATEGORY_ORDER, using the C1/C3/C4/C5/C6 columns already present in the
    episode-detection output. C2 (data completeness) is NOT used here — call
    `filter_data_quality` first to drop episodes that fail it.

    Isolation (C4/C5/C6) is checked BEFORE daytime (C3): "Night" must mean
    "would be SLH except for occurring at night", not merely "occurs at
    night" — an episode that fails both isolation and daytime is
    "Not isolated", matching the paper's definition and the funnel in
    panel A."""
    def _category(row):
        if not row["C1"]:
            return "LLH"
        if not (row["C4"] and row["C5"] and row["C6"]):
            return "Not isolated"
        if not row["C3"]:
            return "Night"
        return "SLH"

    out = episodes_df.copy()
    out["category"]   = out.apply(_category, axis=1)
    out["onset_hour"] = out["onset_time"].dt.hour + out["onset_time"].dt.minute / 60
    return out


def filter_data_quality(episodes_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop episodes failing C2 (insufficient data to judge C3-C6); return (kept_df, n_dropped)."""
    n_before = len(episodes_df)
    kept = episodes_df[episodes_df["C2"]].reset_index(drop=True)
    return kept, n_before - len(kept)


def load_episode_file(patient_id: str) -> pd.DataFrame | None:
    """Load the precomputed episode table for one patient (episode_detection.py output)."""
    path = EPISODES_PATH / f"{patient_id}_episodes.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["onset_time", "offset_time"])


def load_and_classify_fine(patient_id: str):
    """load_episode_file + C2 data-quality filter + fine-grained category assignment."""
    df = load_episode_file(patient_id)
    if df is None or len(df) == 0:
        return None, 0
    kept, n_dropped = filter_data_quality(df)
    if len(kept) == 0:
        return kept, n_dropped
    kept = assign_fine_category(kept)
    return kept, n_dropped


def build_cohort_episodes() -> tuple[pd.DataFrame, list[str]]:
    filtered_pids = load_filtered_patient_list(str(FILTER_STATUS_CSV))
    if filtered_pids is None:
        print("  Filter CSV not found — falling back to all available patients.")
        filtered_pids = set(discover_patients(str(PREPROCESSING_PATH)))
    cohort_pids = sorted(filtered_pids)
    print(f"  Patients to process: {len(cohort_pids)}")

    frames = []
    n_dropped_total = 0
    n_missing = 0
    for pid in cohort_pids:
        ep_df, n_dropped = load_and_classify_fine(pid)
        if ep_df is None:
            n_missing += 1
            continue
        n_dropped_total += n_dropped
        if len(ep_df) == 0:
            continue
        frames.append(ep_df)

    cohort_episodes_df = pd.concat(frames, ignore_index=True)
    print(f"  Cohort episodes: {len(cohort_episodes_df)}  "
          f"(from {cohort_episodes_df['patient_id'].nunique()} patients, "
          f"{n_missing} patient(s) with no episode file, "
          f"{n_dropped_total} episode(s) dropped for insufficient data — C2)")
    print(cohort_episodes_df["category"].value_counts().reindex(CATEGORY_ORDER).to_string())
    return cohort_episodes_df, cohort_pids


def build_raw_episode_pool(cohort_pids: list[str]) -> pd.DataFrame:
    """Load and concatenate the RAW per-patient episode tables (before the C2
    data-quality drop applied in `filter_data_quality`), needed for the
    waterfall in panel A: unlike panels B-D, the funnel must show data
    completeness (C2) as its own exclusion step rather than dropping those
    episodes silently."""
    frames = [df for pid in cohort_pids if (df := load_episode_file(pid)) is not None and len(df) > 0]
    return pd.concat(frames, ignore_index=True)


def compute_funnel_stats(raw_df: pd.DataFrame) -> dict:
    """
    Compute the 5-step simplified funnel: C1 (duration) -> C4+C5+C6 (context)
    -> C2 (completeness) -> C3 (daytime) -> SLH. Mirrors
    `plot_simplified_funnel` in scripts/statistics/slh_llh_stats.py.
    """
    c1, c2, c3 = raw_df["C1"], raw_df["C2"], raw_df["C3"]
    c456 = raw_df["C4"] & raw_df["C5"] & raw_df["C6"]

    n_raw        = len(raw_df)
    n_after_c1   = int(c1.sum())
    n_after_c456 = int((c1 & c456).sum())
    n_after_c2   = int((c1 & c456 & c2).sum())
    n_slh        = int((c1 & c456 & c2 & c3).sum())
    n_night_slh  = n_after_c2 - n_slh

    return {
        "n_raw": n_raw, "n_after_c1": n_after_c1, "n_after_c456": n_after_c456,
        "n_after_c2": n_after_c2, "n_slh": n_slh, "n_night_slh": n_night_slh,
    }


def add_panel_label(ax, label, x=-0.08, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="right")


def panel_label_y(axis_height_in: float, pad_in: float = 0.16) -> float:
    """Axes-fraction y position that keeps a constant PHYSICAL gap (`pad_in`,
    inches) above the axes top, regardless of the axes' own height — a fixed
    fractional offset (e.g. y=1.05) looks fine on a 4in-tall axes but leaves a
    huge floating gap on the ~20in-tall composition panels (C/D)."""
    return 1 + pad_in / axis_height_in


# =============================================================================
# PANEL A: Simplified classification waterfall
# =============================================================================
def plot_simplified_waterfall(ax, funnel_stats: dict, axis_height_in: float) -> None:
    """
    Single-panel waterfall funnel (adapted from `plot_simplified_funnel`'s
    panel A in scripts/statistics/slh_llh_stats.py): C1 (duration) ->
    C4+C5+C6 (context) -> C2 (completeness) -> C3 (daytime) -> SLH.

    Colors are chosen to match panels B-D rather than reusing slh_llh_stats's
    own (unrelated) FUNNEL_COLORS palette: intermediate, still-mixed pools
    are neutral grey, the final SLH bar uses CATEGORY_COLORS["SLH"], and each
    exclusion annotation is colored by the category those episodes join
    (LLH / Not isolated / Night).
    """
    n_raw, n_c1, n_c456, n_c2, n_slh, n_night = (
        funnel_stats["n_raw"], funnel_stats["n_after_c1"], funnel_stats["n_after_c456"],
        funnel_stats["n_after_c2"], funnel_stats["n_slh"], funnel_stats["n_night_slh"],
    )
    n_raw_safe = max(n_raw, 1)

    steps = [
        ("All hyperglycemia episodes",                          n_raw,  WATERFALL_NEUTRAL_COLORS[0]),
        (f"Duration [{D_MIN_MIN:.0f}–{D_MAX_MIN:.0f} min]", n_c1,   WATERFALL_NEUTRAL_COLORS[1]),
        ("Normoglycemia in surrounding hour", n_c456, WATERFALL_NEUTRAL_COLORS[2]),
        (f"Data completeness ≥{MIN_DATA_COMPLETENESS * 100:.0f}%", n_c2, WATERFALL_NEUTRAL_COLORS[3]),
        (f"Daytime ({NIGHT_END_H:02d}:00–{NIGHT_START_H:02d}:00)  SLH", n_slh, CATEGORY_COLORS["SLH"]),
    ]
    labels = [s[0] for s in steps]
    values = [s[1] for s in steps]
    colors = [s[2] for s in steps]
    deltas = [None] + [values[i - 1] - values[i] for i in range(1, len(values))]

    # Exclusion delta text colored by the category those episodes become.
    delta_colors = [None, CATEGORY_COLORS["LLH"], CATEGORY_COLORS["Not isolated"],
                     CATEGORY_COLORS["Not isolated"], CATEGORY_COLORS["Night"]]

    y = np.arange(len(steps))[::-1]
    x_max = max(values) * 1.5 if max(values) > 0 else 10
    ax.set_xlim(0, x_max)

    bars = ax.barh(y, values, color=colors, height=0.55, edgecolor="white",
                    linewidth=0.4, zorder=3)

    for i, (bar, val, delta) in enumerate(zip(bars, values, deltas)):
        pct = 100.0 * val / n_raw_safe
        ax.text(bar.get_width() + x_max * 0.012, bar.get_y() + bar.get_height() / 2,
                 f"n = {val:,}  ({pct:.0f}%)", va="center", fontsize=7, color="#333333")
        if delta is not None and delta > 0:
            label_txt = f"−{delta:,}"
            if i == len(steps) - 1 and n_night > 0:
                label_txt += f"\n(Night: {n_night:,})"
            ax.text(bar.get_width() / 2, bar.get_y() + bar.get_height() + 0.08, label_txt,
                     va="bottom", ha="center", fontsize=6.5, fontweight="bold",
                     color=delta_colors[i], linespacing=1.3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    style_axes(ax)
    ax.set_xlabel("Total number of hyperglycemic episodes across study "
                  "participants in their 45-day observation window",
                  fontsize=FONT_SIZE_BASE)
    add_panel_label(ax, "A", y=panel_label_y(axis_height_in))


# =============================================================================
# PANEL B: Distribution of the number of SLH episodes per patient (daytime)
# =============================================================================
def plot_slh_count_distribution(ax, cohort_episodes_df: pd.DataFrame, cohort_pids: list[str],
                                 axis_height_in: float) -> pd.Series:
    slh_counts = (cohort_episodes_df.loc[cohort_episodes_df["category"] == "SLH", "patient_id"]
                  .value_counts().reindex(cohort_pids, fill_value=0))

    y = slh_counts.values
    max_slh = int(y.max())
    bins = np.arange(0, max_slh + 2) - 0.5
    ax.hist(y, bins=bins, color=CATEGORY_COLORS["SLH"], edgecolor="white", alpha=0.88)

    ax.set_xlabel("Number of SLH episodes per patient (daytime)", fontsize=FONT_SIZE_BASE)
    ax.set_ylabel("Number of patients", fontsize=FONT_SIZE_BASE)
    style_axes(ax)
    ax.set_title(
        "SLH count distribution",
        fontweight="bold", loc="left", pad=6, fontsize=FONT_SIZE_BASE,
    )
    add_panel_label(ax, "B", y=panel_label_y(axis_height_in))
    return slh_counts


# =============================================================================
# PANEL D: Onset-time scatter, cohort-wide (connected by line, no smoothing)
# =============================================================================
def plot_onset_scatter(ax, episodes_df: pd.DataFrame, axis_height_in: float,
                        bin_width_min: float = ONSET_SCATTER_BIN_WIDTH_MIN) -> None:
    cats_present = [c for c in CATEGORY_ORDER if (episodes_df["category"] == c).any()]
    edges_min = np.arange(0, 24 * 60 + bin_width_min, bin_width_min)
    centers_h = (edges_min[:-1] + bin_width_min / 2) / 60

    ax.axvspan(NIGHT_START_H, 24, color=COLOR_NIGHT, alpha=0.5, zorder=0)
    ax.axvspan(0, NIGHT_END_H, color=COLOR_NIGHT, alpha=0.5, zorder=0, label="Night window")

    for c in cats_present:
        onset_min = episodes_df.loc[episodes_df["category"] == c, "onset_hour"].values * 60
        counts, _ = np.histogram(onset_min, bins=edges_min)
        ax.scatter(centers_h, counts, s=14, color=CATEGORY_COLORS[c], zorder=3,
                   label=f"{c} (n={int(counts.sum())})")
        ax.plot(centers_h, counts, color=CATEGORY_COLORS[c], lw=1.3, alpha=0.85, zorder=2)

    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel(f"Onset time of day (h) — {bin_width_min:.0f} min bins", fontsize=FONT_SIZE_BASE)
    ax.set_ylabel("Number of hyperglycemia episodes per 15-min bin", fontsize=FONT_SIZE_BASE)
    style_axes(ax)
    ax.legend(fontsize=FONT_SIZE_BASE + 1, frameon=False, loc="upper left", ncol=1,
              markerscale=1.8, handletextpad=0.5, columnspacing=1.2, labelspacing=0.5)
    add_panel_label(ax, "D", y=panel_label_y(axis_height_in))


# =============================================================================
# PANEL C: Per-patient category composition (proportions, with the per-patient
# total episode count overlaid on a secondary axis)
# =============================================================================
def plot_category_composition(ax, episodes_df: pd.DataFrame,
                               axis_height_in: float) -> pd.DataFrame:
    """Vertical stacked-bar panel (patients along the x-axis, ordered by total
    episode count, ascending, left to right) showing each patient's episode
    composition as proportions.

    The previous version of this figure drew the same data twice: proportions
    in one panel and raw counts in the one below it, on a shared x-axis. The
    counts panel carried exactly one thing the proportions do not — each
    patient's total episode count — so that total is now a line on a secondary
    axis here, and the duplicate panel is gone. It doubles as a legend for the
    x ordering, which the reader otherwise has to take on trust from the label.
    """
    counts = (episodes_df.groupby(["patient_id", "category"]).size()
              .unstack(fill_value=0).reindex(columns=CATEGORY_ORDER, fill_value=0))
    totals = counts.sum(axis=1)
    patient_order = totals.sort_values(ascending=True).index
    counts = counts.loc[patient_order]
    totals = totals.loc[patient_order]
    props = counts.div(totals, axis=0)

    n = len(patient_order)
    x = np.arange(n)

    bottom = np.zeros(n)
    for cat in CATEGORY_ORDER:
        vals = props[cat].values * 100
        ax.bar(x, vals, bottom=bottom, width=0.8, color=CATEGORY_COLORS[cat],
               edgecolor="white", linewidth=0.15, label=cat, zorder=2)
        bottom += vals

    ax.set_xlim(-0.6, n - 0.4)  # hug the bars — no default 5% categorical margin
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of episodes", fontsize=FONT_SIZE_BASE)
    style_axes(ax)
    ax.grid(True, axis="y", alpha=0.25, zorder=0)

    # Individual patient IDs are not shown: 135 rotated ID labels are illegible
    # in a printed document and carry no information (the ordering is purely
    # by total episode count, not patient identity) — a handful of rank ticks
    # is enough to read the axis as a scale. Full per-patient values are in
    # classification_summary_per_patient.csv.
    n_ticks = min(6, n)
    tick_idx = np.unique(np.linspace(0, n - 1, n_ticks).astype(int))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(i + 1) for i in tick_idx], fontsize=FONT_SIZE_BASE)
    ax.set_xlabel(f"Patients (n={n}, ordered by total episode count, ascending)",
                   fontsize=FONT_SIZE_BASE)

    # Stacked proportions fill the axes from 0 to 1 everywhere, so an in-axes
    # legend would sit on top of the data — place it above, opposite the
    # left-aligned title.
    ax.legend(fontsize=FONT_SIZE_BASE + 1, loc="lower right",
              bbox_to_anchor=(1.0, 1.005), ncol=len(CATEGORY_ORDER), frameon=False,
              handlelength=1.6, handleheight=1.4, columnspacing=1.4)

    ax_tot = ax.twinx()
    ax_tot.plot(x, totals.values, color=COLOR_TOTAL_LINE, lw=1.3, zorder=4)
    ax_tot.set_ylim(bottom=0)
    ax_tot.set_ylabel("Total hyperglycemia episodes\nper patient", fontsize=FONT_SIZE_BASE,
                      color=COLOR_TOTAL_LINE)
    ax_tot.tick_params(axis="y", labelsize=FONT_SIZE_BASE, colors=COLOR_TOTAL_LINE)
    ax_tot.spines["top"].set_visible(False)
    ax_tot.spines["right"].set_color(COLOR_TOTAL_LINE)

    add_panel_label(ax, "C", x=-0.045, y=panel_label_y(axis_height_in))

    out = counts.copy()
    out["total"] = totals
    out = out.join(props.add_suffix("_prop"))
    out.index.name = "patient_id"
    return out.reset_index()


# =============================================================================
# MAIN FIGURE GENERATION
# =============================================================================
def generate_figure_2() -> None:
    apply_publication_style()

    print("\n" + "=" * 70)
    print("  GENERATING FIGURE 2 — CLASSIFICATION SUMMARY")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n  Loading and classifying episodes ...")
    cohort_episodes_df, cohort_pids = build_cohort_episodes()

    print("  Loading raw episode pool for the classification waterfall (panel A) ...")
    raw_episode_pool = build_raw_episode_pool(cohort_pids)
    funnel_stats = compute_funnel_stats(raw_episode_pool)

    # Wider than the standard FIGURE_WIDTH_INCHES (7.24in): at that width the
    # long panel-A category labels and panel C's 135 stacked bars made the
    # whole figure feel compressed.
    fig_width = 10.5
    row_a_height_in = 2.7   # A (waterfall) + B (SLH count distribution)
    row_c_height_in = 3.0   # C — per-patient composition
    row_d_height_in = 3.4   # D — onset-time scatter

    # Fixed physical margins (inches) — kept constant regardless of cohort
    # size or fig_width, unlike a fractional left/top/bottom/hspace, which
    # would otherwise shrink to nothing (or balloon) as those change.
    LEFT_MARGIN_IN    = 1.5   # room for panel A's longest category label
    RIGHT_MARGIN_IN   = 0.85  # room for panel C's secondary-axis label
    TOP_MARGIN_IN     = 0.75  # room for the figure suptitle above panels A/B
    ROW_GAP_IN        = 0.95  # gap between rows (each row has an xlabel below
                               # it AND the next row's title/letter/legend
                               # above it sharing this gap)
    BOTTOM_MARGIN_IN  = 0.55  # room for the x-axis label below D

    row_heights = [row_a_height_in, row_c_height_in, row_d_height_in]
    fig_height = (TOP_MARGIN_IN + sum(row_heights) + 2 * ROW_GAP_IN + BOTTOM_MARGIN_IN)

    fig = plt.figure(figsize=(fig_width, fig_height))
    outer_gs = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=row_heights,
        hspace=ROW_GAP_IN / (sum(row_heights) / len(row_heights)),
        left=LEFT_MARGIN_IN / fig_width, right=1 - RIGHT_MARGIN_IN / fig_width,
        top=1 - TOP_MARGIN_IN / fig_height, bottom=BOTTOM_MARGIN_IN / fig_height,
    )

    # Row 1 is the only split row: the waterfall needs most of the width for
    # its bars and n= annotations, the count histogram reads fine narrow.
    top_gs = outer_gs[0].subgridspec(1, 2, width_ratios=[2.0, 1.15], wspace=0.28)

    ax_a = fig.add_subplot(top_gs[0, 0])
    plot_simplified_waterfall(ax_a, funnel_stats, row_a_height_in)

    ax_b = fig.add_subplot(top_gs[0, 1])
    slh_counts = plot_slh_count_distribution(ax_b, cohort_episodes_df, cohort_pids,
                                              row_a_height_in)

    ax_c = fig.add_subplot(outer_gs[1])
    summary_df = plot_category_composition(ax_c, cohort_episodes_df, row_c_height_in)

    ax_d = fig.add_subplot(outer_gs[2])
    plot_onset_scatter(ax_d, cohort_episodes_df, row_d_height_in)

    # fig.suptitle("SLH/LLH episode classification summary", fontsize=FONT_SIZE_BASE + 2,
    #              fontweight="bold", y=1 - (TOP_MARGIN_IN * 0.32) / fig_height)

    summary_path = os.path.join(OUTPUT_DIR, "classification_summary_per_patient.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Saved: {summary_path}")

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    save_figure(fig, output_path)

    print("\n" + "=" * 70)
    print(f"  SAVED: {output_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    generate_figure_2()
