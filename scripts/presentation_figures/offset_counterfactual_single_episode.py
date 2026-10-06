"""
scripts/presentation_figures/offset_counterfactual_single_episode.py
====================================================================

Presentation figure (single SLH episode, 16:9): how different zone-expansion offsets
(δ = 0, 15, 30, 45, 60 minutes) affect the counterfactual signal construction.

Shows:
  · Original signal G(t) — bold dark line
  · Episode span — orange background highlight
  · Multiple counterfactual curves — one per offset value, color-coded
  · 180 mg/dL hyperglycaemia threshold (dashed)
  · Night shading for context
  · 3-hour window around episode (±1.5h from onset/offset)
  · Clean legend with offset values
  · Minimalist axes (no top/right spines)

Usage
-----
    python3 -m scripts.presentation_figures.offset_counterfactual_single_episode         [--patient_id ID]         [--out_dir DIR]

Output (default scripts/presentation_figures/output/offset_counterfactual_single_episode/)
  offset_counterfactual_single_episode_<patient_id>.{pdf,svg}
"""

import os
import sys
import argparse
from datetime import datetime, time as _dtime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.patient_selection import resolve_patient_id
from src.preprocessing.load_data import load_patient_window
from src.preprocessing.timestep import detect_dominant_sampling_interval
from src.episodes.detection import detect_hyperglycemia_episodes
from src.episodes.criteria import classify_episodes
from src.episodes.episode_config import LABEL_SLH, NIGHT_START_H, NIGHT_END_H
from src.config import TIME_COL, GLUCOSE_COL, COL_FLAG_TIME_ERROR, COL_FLAG_OUTLIER_GLUCOSE
from src.counterfactual.build_counterfactual import build_full_counterfactual_signal
from src.visualization.style import apply_publication_style, save_figure
from src.visualization.colors import OFFSET_COLORS, COLOR_NIGHT

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "output", "offset_counterfactual_single_episode"
)

# ---------------------------------------------------------------------------
# Figure geometry — exactly 16:9
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 13.0, 7.3125      # 16:9 ratio

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
COLOR_ORIGINAL = "#333333"          # dark grey — original signal
COLOR_THRESHOLD = "#D55E00"         # vermilion — 180 mg/dL
COLOR_EPISODE = "#E69F00"           # orange — episode fill

# ---------------------------------------------------------------------------
# Font sizes
# ---------------------------------------------------------------------------
FS_LABEL = 14
FS_TICK = 12
FS_LEGEND = 11
FS_TITLE = 15

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
OFFSET_VALUES = [0, 15, 30, 45, 60]  # Zone-expansion offsets to compare
H_HYPER = 180
PLOT_WINDOW_H = 3                   # Hours around episode (±1.5h)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_valid_data(df):
    """Filter out flagged bad data points."""
    mask = pd.Series(True, index=df.index)
    if COL_FLAG_TIME_ERROR in df.columns:
        mask &= ~df[COL_FLAG_TIME_ERROR].fillna(False).astype(bool)
    if COL_FLAG_OUTLIER_GLUCOSE in df.columns:
        mask &= ~df[COL_FLAG_OUTLIER_GLUCOSE].fillna(False).astype(bool)
    return df[mask][[TIME_COL, GLUCOSE_COL]].dropna().sort_values(TIME_COL).reset_index(drop=True)


def load_and_prepare_data(patient_id: str):
    """Load patient data and extract SLH episodes."""
    print(f"  Loading patient {patient_id}...")
    df_raw = load_patient_window(patient_id)

    if df_raw is None:
        raise FileNotFoundError(f"Cannot load patient {patient_id}")

    df = prepare_valid_data(df_raw)
    timestep = detect_dominant_sampling_interval(df)

    print(f"    Dominant sampling interval: {timestep:.1f} min")
    print(f"    Data points: {len(df)}")

    # Detect episodes and classify
    raw_episodes = detect_hyperglycemia_episodes(df)
    classification_df = classify_episodes(df, raw_episodes)

    # Extract SLH episodes only
    slh_df = classification_df[classification_df["classification"] == LABEL_SLH]
    slh_episodes = [
        {"onset_time": row["onset_time"], "offset_time": row["offset_time"]}
        for _, row in slh_df.iterrows()
    ]

    print(f"    SLH episodes: {len(slh_episodes)}")

    return df, timestep, slh_episodes


def select_episode(df, slh_episodes):
    """Select a representative episode (middle one by time, with margin)."""
    window_delta = pd.Timedelta(hours=PLOT_WINDOW_H / 2)
    eligible = []

    for ep in slh_episodes:
        if (ep["onset_time"] - window_delta >= df[TIME_COL].iloc[0] and
            ep["offset_time"] + window_delta <= df[TIME_COL].iloc[-1]):
            eligible.append(ep)

    if not eligible:
        raise ValueError("No eligible episodes with sufficient margin for plotting")

    # Select middle episode
    selected = eligible[len(eligible) // 2]
    print(f"    Selected episode: {selected['onset_time']} → {selected['offset_time']}")

    return selected


# =============================================================================
# FIGURE
# =============================================================================

def build_figure(patient_id: str, df, episode, timestep_min: float) -> plt.Figure:
    """Compose and return the episode comparison figure."""

    apply_publication_style()
    matplotlib.rcParams.update({
        "font.size": FS_LABEL,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "axes.grid": False,
    })

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=0.09, right=0.93, top=0.90, bottom=0.12)

    # Define plot window
    window_delta = pd.Timedelta(hours=PLOT_WINDOW_H / 2)
    t_start = episode["onset_time"] - window_delta
    t_end = episode["offset_time"] + window_delta

    # Extract window
    mask = (df[TIME_COL] >= t_start) & (df[TIME_COL] <= t_end)
    t_window = df.loc[mask, TIME_COL]
    g_window = df.loc[mask, GLUCOSE_COL].astype(float).values

    # ── 1. Night shading ──────────────────────────────────────────────────────
    current_date = t_start.date()
    end_date = t_end.date() + timedelta(days=1)
    night_labeled = False

    while current_date <= end_date:
        for hour_start, hour_end in [(0, NIGHT_END_H), (NIGHT_START_H, 24)]:
            night_start = pd.Timestamp(datetime.combine(current_date, _dtime(hour_start % 24, 0)))
            night_end = pd.Timestamp(datetime.combine(current_date, _dtime(0, 0))) + pd.Timedelta(hours=hour_end)

            if night_end > t_start and night_start < t_end:
                label = f"Night ({NIGHT_START_H:02d}–{NIGHT_END_H:02d}h)" if not night_labeled else "_"
                ax.axvspan(
                    max(night_start, t_start), min(night_end, t_end),
                    color=COLOR_NIGHT, alpha=0.25, zorder=0, linewidth=0, label=label
                )
                night_labeled = True

        current_date += timedelta(days=1)

    # ── 2. Episode span ───────────────────────────────────────────────────────
    ax.axvspan(episode["onset_time"], episode["offset_time"],
              color=COLOR_EPISODE, alpha=0.25, zorder=1, linewidth=0, label="SLH episode")

    # ── 3. Threshold line ─────────────────────────────────────────────────────
    ax.axhline(H_HYPER, color=COLOR_THRESHOLD, ls="--", lw=1.5, alpha=0.75, zorder=2)

    # ── 4. Original signal (G(t)) – bold ──────────────────────────────────────
    ax.plot(t_window, g_window, color=COLOR_ORIGINAL, lw=2.0, alpha=0.90,
            zorder=4, label="G(t)")

    # ── 5. Counterfactual signals (C(t) for each offset) ──────────────────────
    for offset in OFFSET_VALUES:
        cfact = build_full_counterfactual_signal(df, [episode], offset, timestep_min)[mask.values]
        color = OFFSET_COLORS.get(offset, "#333333")
        ax.plot(t_window, cfact, color=color, lw=1.3, alpha=0.80, zorder=5,
               label=f"δ = {offset} min")

    # ── 6. Axes configuration ─────────────────────────────────────────────────
    ax.set_xlim(t_start, t_end)

    y_min = max(50, float(np.nanmin(g_window)) - 15)
    y_max = min(420, float(np.nanmax(g_window)) + 25)
    ax.set_ylim(y_min, y_max)

    # X-axis: time of day (HH:MM)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=FS_TICK, rotation=30, ha="right")

    ax.set_xlabel("Time of day", fontsize=FS_LABEL, labelpad=8)
    ax.set_ylabel("Glucose (mg/dL)", fontsize=FS_LABEL, labelpad=8)

    # Title with patient ID
    ax.set_title(
        f"Episode Counterfactual Comparison — Participant {patient_id}",
        fontsize=FS_TITLE, fontweight='bold', pad=14
    )

    # ── 7. Clean spines ───────────────────────────────────────────────────────
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)

    # Subtle grid
    ax.yaxis.grid(True, ls=":", lw=0.4, color="#EEEEEE", zorder=0)
    ax.set_axisbelow(True)

    # ── 8. Legend ─────────────────────────────────────────────────────────────
    ax.legend(
        fontsize=FS_LEGEND,
        loc="lower right",
        framealpha=0.90,
        edgecolor="black",
        fancybox=False,
        shadow=False,
        ncol=2
    )

    return fig


# =============================================================================
# PIPELINE
# =============================================================================

def generate(patient_id: str, output_dir: str) -> None:
    """Generate and save the episode comparison figure."""
    df, timestep, slh_episodes = load_and_prepare_data(patient_id)

    print(f"\n  Selecting representative episode...")
    episode = select_episode(df, slh_episodes)

    # Build and save figure
    fig = build_figure(patient_id, df, episode, timestep)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"offset_counterfactual_single_episode_{patient_id}")
    save_figure(fig, out)
    print(f"\n  Saved → {out}.pdf / .svg\n")


# =============================================================================
# CLI
# =============================================================================

def parse_arguments():
    p = argparse.ArgumentParser(
        description="Episode counterfactual comparison with multiple offsets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--patient_id",
        type=str,
        default=None,
        help="Patient identifier (default: patient with the most SLH episodes)"
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory"
    )
    return p


def main():
    args = parse_arguments().parse_args()
    args.patient_id = resolve_patient_id(args.patient_id)

    print("\n" + "=" * 65)
    print("  Episode Counterfactual Comparison")
    print("=" * 65)
    print(f"  Patient: {args.patient_id}")
    print(f"  Offsets: {OFFSET_VALUES} min")
    print(f"  Output: {args.out_dir}")
    print("=" * 65)

    generate(args.patient_id, args.out_dir)

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
