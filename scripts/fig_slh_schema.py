"""
scripts/fig_slh_schema.py
=========================
Figure 1 — SLH Detection Schema.

Panel layout
------------
  A  Top panel: N_DAYS consecutive days of CGM trace.
       · Night hours shaded in pale grey
       · Raw glucose trace in neutral grey
       · Episodes colour-coded by final classification:
             Navy  = SLH  (all criteria C1–C6 passed)
             Pink  = LLH  (duration > D_MAX_MIN)
             Grey  = excluded (fails any criterion)
       · H_HYPER = 180 mg/dL: dashed orange line
       · H_CONTEXT = 140 mg/dL: dotted teal line
  B  Lower-left  ]  Zoomed view of confirmed SLH episode
  C  Lower-right ]  Zoomed view of second confirmed SLH episode

Inputs
------
  Data/Data_Preprocessed_45d/<patient_id>/<patient_id>_win1_*.xlsx

Outputs
-------
  Figures/SLH_Schema/<patient_id>_slh_schema.{pdf,png}

Usage
-----
    python3 -m scripts.fig_slh_schema [--patient_id A174] [--n_days 5]
                                     [--zoom_pad 80] [--out_dir <path>]

Authors
-------
Guillaume Deside
Date
----
2025
"""

import os
import argparse
from datetime import datetime, time as _dtime, timedelta

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.config import TIME_COL, GLUCOSE_COL, FIG_DIR, FIGURE_WIDTH_INCHES
from src.config.figure_config import FONT_SIZE_BASE

from src.preprocessing.load_data import load_patient_window, discover_patients
from src.preprocessing.timestep import detect_dominant_sampling_interval

from src.episodes.episode_config import (
    H_CONTEXT,
    USE_GAP_MERGING, MERGE_GAP_MIN, MERGE_INTERNAL_GAP_MIN,
    D_MAX_MIN,
    MIN_DATA_COMPLETENESS, CONTEXT_WIN_MIN,
    LABEL_SLH, LABEL_LLH,
)
from src.episodes.detection import detect_hyperglycemia_episodes
from src.episodes.criteria import classify_episodes as classify_episodes_core
from src.episodes.criteria import calculate_data_completeness

from src.visualization.slh_schema_plots import draw_zoom_panel, draw_overview_panel
from src.visualization.style import apply_publication_style, save_figure

# =============================================================================
# PARAMETERS
# =============================================================================
FIGURES_OUTPUT_PATH = os.path.join(FIG_DIR, "SLH_Schema")

DEFAULT_PATIENT_ID = "A174"
N_DAYS_OVERVIEW = 5
ZOOM_PAD_MIN = 80
ZOOM_MIN_DURATION_MIN = 15  # example SLH shown in zoom panels must be at least this long

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def add_panel_label(ax, label, x=-0.12, y=1.05):
    """Add panel letter to the top-left corner of an axis."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=FONT_SIZE_BASE + 3, fontweight="bold", va="top", ha="right")


def find_first_crossing(df: pd.DataFrame, t_start: pd.Timestamp,
                       t_end: pd.Timestamp, threshold: float = H_CONTEXT):
    """
    Return (timestamp, glucose) of the first point ≤ threshold in [t_start, t_end].
    Returns (None, None) if no crossing found.
    """
    mask = (df[TIME_COL] >= t_start) & (df[TIME_COL] <= t_end)
    subset = df.loc[mask].sort_values(TIME_COL)
    hits = subset[subset[GLUCOSE_COL] <= threshold]
    
    if hits.empty:
        return None, None
    
    return hits.iloc[0][TIME_COL], float(hits.iloc[0][GLUCOSE_COL])



def enrich_episodes_for_visualization(df: pd.DataFrame, episodes_df: pd.DataFrame, 
                                      raw_episodes: list, timestep_min: float) -> list[dict]:
    """
    Convert classified episodes DataFrame to list of dicts with visualization data.
    
    Adds crossing points and completeness percentages needed for schema figure.
    """
    ctx = pd.Timedelta(minutes=CONTEXT_WIN_MIN)
    
    # Create index mapping from onset_time to raw episode
    raw_ep_lookup = {ep["onset_time"]: ep for ep in raw_episodes}
    
    enriched = []
    
    for _, row in episodes_df.iterrows():
        onset_time = row["onset_time"]
        offset_time = row["offset_time"]
        
        # Get raw episode data
        raw_ep = raw_ep_lookup.get(onset_time, {})
        
        # Determine type based on classification and criteria
        if row["classification"] == LABEL_SLH:
            ep_type = "SLH"
        elif row["classification"] == LABEL_LLH:
            if row["duration_min"] > D_MAX_MIN:
                ep_type = "LLH"
            elif not row["C2"]:
                ep_type = "excl_incomplete"
            elif not row["C3"]:
                ep_type = "excl_night"
            elif not row["C4"]:
                ep_type = "excl_ctx_before"
            elif not row["C5"]:
                ep_type = "excl_ctx_after"
            elif not row["C6"]:
                ep_type = "excl_no_cross"
            else:
                ep_type = "excl_other"
        else:
            ep_type = "excl_other"
        
        # Find crossing points for SLH episodes
        cross_before_t, cross_before_g = None, None
        cross_after_t, cross_after_g = None, None
        
        if ep_type == "SLH":
            cross_before_t, cross_before_g = find_first_crossing(
                df, onset_time - ctx, onset_time
            )
            cross_after_t, cross_after_g = find_first_crossing(
                df, offset_time, offset_time + ctx
            )
        
        # Calculate completeness percentages
        ep_completeness = calculate_data_completeness(df, onset_time, offset_time, timestep_min)
        before_completeness = calculate_data_completeness(df, onset_time - ctx, onset_time, timestep_min)
        after_completeness = calculate_data_completeness(df, offset_time, offset_time + ctx, timestep_min)
        
        enriched.append({
            "onset_time": onset_time,
            "offset_time": offset_time,
            "duration_min": row["duration_min"],
            "idx": raw_ep.get("idx", []),
            "type": ep_type,
            "cross_before_t": cross_before_t,
            "cross_before_g": cross_before_g,
            "cross_after_t": cross_after_t,
            "cross_after_g": cross_after_g,
            "completeness_episode": ep_completeness,
            "completeness_before": before_completeness,
            "completeness_after": after_completeness,
        })
    
    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# EPISODE CLASSIFICATION 
# ─────────────────────────────────────────────────────────────────────────────
def classify_episodes(df: pd.DataFrame, timestep_min: float) -> list[dict]:
    """
    Classify episodes using src.episodes.criteria and enrich for visualization.
    
    Returns list of episode dicts with visualization-specific data:
        - type: 'SLH', 'LLH', 'excl_incomplete', 'excl_night', etc.
        - cross_before_t, cross_before_g, cross_after_t, cross_after_g
        - completeness_episode, completeness_before, completeness_after
    """
    # Get raw episodes for idx information
    raw_episodes = detect_hyperglycemia_episodes(df)
    
    if not raw_episodes:
        return []
    
    # Use core classification from criteria module
    episodes_df = classify_episodes_core(df, raw_episodes)
    
    # Enrich with visualization data
    enriched = enrich_episodes_for_visualization(df, episodes_df, raw_episodes, timestep_min)
    
    return enriched

# ─────────────────────────────────────────────────────────────────────────────
# WINDOW AND EPISODE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select_best_overview_window(df: pd.DataFrame, classified: list,
                                n_days: int = N_DAYS_OVERVIEW):
    """
    Select n_days consecutive days that maximize episode diversity.
    Score = 4 × n_SLH + 2 × n_LLH + 1 × n_excluded
    """
    t_first = df[TIME_COL].min()
    t_last = df[TIME_COL].max()
    d_first = t_first.date()
    d_last = t_last.date()
    n_available = (d_last - d_first).days + 1
    
    if n_available <= n_days:
        return (
            pd.Timestamp(datetime.combine(d_first, _dtime(0, 0))),
            pd.Timestamp(datetime.combine(d_last, _dtime(23, 59)))
        )
    
    best_score = -1
    best_start = None
    
    for i in range(n_available - n_days + 1):
        window_start = pd.Timestamp(datetime.combine(d_first + timedelta(days=i), _dtime(0, 0)))
        window_end = window_start + pd.Timedelta(days=n_days)
        
        score = sum(
            (4 if ep["type"] == "SLH" else
             2 if ep["type"] == "LLH" else 1)
            for ep in classified
            if window_start <= ep["onset_time"] < window_end
        )
        
        if score > best_score:
            best_score = score
            best_start = window_start
    
    return best_start, best_start + pd.Timedelta(days=n_days)


def select_zoom_episodes(classified: list, window_start, window_end, n: int = 2,
                         min_duration_min: float = ZOOM_MIN_DURATION_MIN):
    """
    Select up to n SLH episodes from the window for zoom panels.
    Prefer episodes with clear OR-crossing (deepest excursion below H_CONTEXT).

    Only SLH episodes with duration >= min_duration_min are considered, so the
    example shown is not a borderline ~10-min episode. Falls back to shorter
    SLH episodes if none in the window reach min_duration_min.
    """
    slh_episodes = [
        ep for ep in classified
        if ep["type"] == "SLH"
        and ep["onset_time"] >= window_start
        and ep["offset_time"] <= window_end
    ]

    if not slh_episodes:
        return []

    long_enough = [ep for ep in slh_episodes if ep["duration_min"] >= min_duration_min]
    if long_enough:
        slh_episodes = long_enough
    else:
        print(f"  [WARN] No SLH >= {min_duration_min} min in window; "
              f"falling back to shorter examples for zoom panels.")
    
    def crossing_score(ep):
        """Score episodes by crossing depth and number of sides."""
        depth = 0
        if ep["cross_before_g"] is not None:
            depth += max(0, H_CONTEXT - ep["cross_before_g"])
        if ep["cross_after_g"] is not None:
            depth += max(0, H_CONTEXT - ep["cross_after_g"])
        
        n_sides = (ep["cross_before_t"] is not None) + (ep["cross_after_t"] is not None)
        return n_sides * 200 + depth
    
    slh_sorted = sorted(slh_episodes, key=crossing_score, reverse=True)
    
    if len(slh_sorted) < 2:
        return slh_sorted
    
    # Try to pick one before-dominant and one after-dominant
    before_dominant = [ep for ep in slh_sorted if ep["cross_before_t"] is not None]
    after_dominant = [ep for ep in slh_sorted if ep["cross_after_t"] is not None]
    
    picks = []
    if before_dominant:
        picks.append(before_dominant[0])
    
    remaining = [ep for ep in slh_sorted if ep not in picks]
    if after_dominant:
        for ep in after_dominant:
            if ep not in picks:
                picks.append(ep)
                break
    
    while len(picks) < n and remaining:
        ep = remaining.pop(0)
        if ep not in picks:
            picks.append(ep)
    
    return picks[:n]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FIGURE ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────
def generate_schema_figure(patient_id, df, classified, window_start, window_end,
                           zoom_episodes, timestep_min, output_prefix):
    """Generate complete SLH schema figure."""
    apply_publication_style()
    
    # Count episodes
    n_slh = sum(1 for ep in classified if ep["type"] == "SLH")
    n_llh = sum(1 for ep in classified if ep["type"] == "LLH")
    n_incomplete = sum(1 for ep in classified if ep["type"] == "excl_incomplete")
    
    n_zoom = len(zoom_episodes)
    if n_zoom == 0:
        print(f"  [{patient_id}] No SLH episodes found — cannot draw zoom panels.")
        return
    
    # Layout
    fig_height = FIGURE_WIDTH_INCHES * (0.68 if n_zoom == 2 else 0.50)
    fig = plt.figure(figsize=(FIGURE_WIDTH_INCHES, fig_height))
    
    if n_zoom == 2:
        gs = gridspec.GridSpec(
            2, 2, figure=fig,
            height_ratios=[1.9, 1.0],
            hspace=0.42, wspace=0.30,
            left=0.09, right=0.93,
            top=0.93, bottom=0.10,
        )
        ax_overview = fig.add_subplot(gs[0, :])
        ax_zooms = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    else:
        gs = gridspec.GridSpec(
            2, 1, figure=fig,
            height_ratios=[1.9, 1.0],
            hspace=0.42,
            left=0.09, right=0.93,
            top=0.93, bottom=0.10,
        )
        ax_overview = fig.add_subplot(gs[0])
        ax_zooms = [fig.add_subplot(gs[1])]
    
    # Draw panels
    draw_overview_panel(ax_overview, df, classified, window_start, window_end)
    add_panel_label(ax_overview, "A")

    zoom_letters = ["B", "C"]
    for i, (episode, ax_zoom) in enumerate(zip(zoom_episodes, ax_zooms)):
        draw_zoom_panel(ax_zoom, df, episode, timestep_min, pad_min=ZOOM_PAD_MIN)
        add_panel_label(ax_zoom, zoom_letters[i])
        
        # Connect zoom to overview with bracket
        try:
            ax_overview.annotate(
                "", xy=(episode["onset_time"], ax_overview.get_ylim()[0]),
                xytext=(episode["onset_time"], ax_overview.get_ylim()[0] - 18),
                arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6),
                annotation_clip=True,
            )
        except Exception:
            pass
    
    # Super-title
    title_parts = [
        f"SLH episode detection schema  ·  Participant {patient_id}",
        f"SLH n = {n_slh}  ·  LLH n = {n_llh}"
    ]
    # if n_incomplete > 0:
    #     title_parts.append(f"Incomplete data n = {n_incomplete}")
    
    # fig.suptitle(
    #     "  ·  ".join(title_parts),
    #     fontsize=FONT_SIZE_BASE + 1, fontweight="bold", y=0.99,
    # )
    
    save_figure(fig, output_prefix)
    print(f"  [{patient_id}]  SLH={n_slh}  LLH={n_llh}  Incomplete={n_incomplete}  "
          f"Zoom episodes: {n_zoom}")


# ─────────────────────────────────────────────────────────────────────────────
# PER-PATIENT RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def process_patient(patient_id, n_days, output_dir):
    """Process single patient and generate schema figure."""
    df = load_patient_window(patient_id)
    if df is None:
        print(f"  [{patient_id}] SKIP — no valid data")
        return 0
    
    timestep = detect_dominant_sampling_interval(df)
    classified = classify_episodes(df, timestep)
    n_slh = sum(1 for ep in classified if ep["type"] == "SLH")
    
    window_start, window_end = select_best_overview_window(df, classified, n_days)
    zoom_episodes = select_zoom_episodes(classified, window_start, window_end, n=2)
    
    output_prefix = os.path.join(output_dir, f"{patient_id}_slh_schema")
    generate_schema_figure(patient_id, df, classified, window_start, window_end,
                          zoom_episodes, timestep, output_prefix)
    
    return n_slh


# ─────────────────────────────────────────────────────────────────────────────
# CLI & MAIN
# ─────────────────────────────────────────────────────────────────────────────
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Figure 1 — SLH detection schema (using criteria module)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--patient_id", default=DEFAULT_PATIENT_ID,
        help="Patient ID (e.g. A168) or ALL (auto-picks best patient)"
    )
    parser.add_argument(
        "--n_days", type=int, default=N_DAYS_OVERVIEW,
        help="Number of days shown in the top panel"
    )
    parser.add_argument(
        "--zoom_pad", type=int, default=ZOOM_PAD_MIN,
        help="Minutes of padding on each side of zoomed episodes"
    )
    parser.add_argument(
        "--out_dir", default=FIGURES_OUTPUT_PATH,
        help="Output directory for figures"
    )
    return parser


def main():
    """Main execution pipeline."""
    args = parse_arguments().parse_args()
    apply_publication_style()
    
    # Override global zoom padding if specified
    global ZOOM_PAD_MIN
    ZOOM_PAD_MIN = args.zoom_pad
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    merge_mode = "WITH gap merging" if USE_GAP_MERGING else "WITHOUT gap merging"
    
    print("\n" + "=" * 70)
    print("  Figure 1 — SLH Detection Schema (using criteria module)")
    print("=" * 70)
    print(f"  Episode detection: {merge_mode}")
    if USE_GAP_MERGING:
        print(f"    MERGE_GAP_MIN = {MERGE_GAP_MIN} min")
        print(f"    MERGE_INTERNAL_GAP_MIN = {MERGE_INTERNAL_GAP_MIN} min")
    print(f"  Output: {args.out_dir}")
    print(f"  Days: {args.n_days}  |  Zoom pad: {args.zoom_pad} min")
    print(f"  C2: Data completeness ≥ {MIN_DATA_COMPLETENESS*100:.0f}%")
    print("=" * 70)
    
    if args.patient_id.strip().upper() == "ALL":
        # Auto-select patient with most SLH episodes
        patients = discover_patients()
        print(f"\n  Scanning {len(patients)} patients for best illustration...")
        
        best_patient = None
        best_count = -1
        
        for pid in patients:
            df = load_patient_window(pid)
            if df is None:
                continue
            
            timestep = detect_dominant_sampling_interval(df)
            classified = classify_episodes(df, timestep)
            n_slh = sum(1 for ep in classified if ep["type"] == "SLH")
            
            if n_slh > best_count:
                best_count = n_slh
                best_patient = pid
        
        if best_patient is None:
            print("  No patient with SLH episodes found.")
            return
        
        print(f"  Best candidate: {best_patient}  (SLH n = {best_count})")
        process_patient(best_patient, args.n_days, args.out_dir)
    else:
        process_patient(args.patient_id.strip(), args.n_days, args.out_dir)
    
    print("\n" + "=" * 70)
    print(f"Output: {os.path.abspath(args.out_dir)}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()