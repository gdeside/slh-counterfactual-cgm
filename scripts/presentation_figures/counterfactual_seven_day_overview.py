"""
scripts/presentation_figures/counterfactual_seven_day_overview.py
=================================================================

7-day CGM overview figure comparing original and counterfactual glucose signals
for a specific patient and zone-expansion offset (δ).

Shows:
  · Original signal G(t) — dashed line, lighter
  · Counterfactual signal C(t) — solid line, darker
  · 180 mg/dL hyperglycaemia threshold (dashed)
  · SLH episode spans highlighted in background
  · Night periods (22:00–06:00) shaded
  · Day separators on x-axis
  · Relative time axis (Day 1, Day 2, etc.)
  · Clean legend with signal labels and offset value
  · Minimalist axes (no top/right spines)

Usage
-----
    python3 -m scripts.presentation_figures.counterfactual_seven_day_overview \
        [--patient_id ID] \
        [--offset_min 45] \
        [--out_dir DIR]
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
from src.episodes.episode_config import (
    LABEL_SLH, NIGHT_START_H, NIGHT_END_H, GAP_FACTOR
)
from src.config import TIME_COL, GLUCOSE_COL, COL_FLAG_TIME_ERROR, COL_FLAG_OUTLIER_GLUCOSE
from src.counterfactual.build_counterfactual import build_full_counterfactual_signal
from src.visualization.style import apply_publication_style, save_figure
from src.visualization.colors import OFFSET_COLORS, COLOR_NIGHT

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "output", "counterfactual_seven_day_overview"
)

# ---------------------------------------------------------------------------
# Figure geometry — exactly 16:9
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 13.0, 7.3125      # 16:9 ratio

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
COLOR_ORIGINAL = "#666666"          # medium grey — original signal
COLOR_THRESHOLD = "#D55E00"         # vermilion — 180 mg/dL
COLOR_EPISODE = "#E69F00"           # orange — SLH episode fill

# ---------------------------------------------------------------------------
# Font sizes
# ---------------------------------------------------------------------------
FS_LABEL = 14
FS_TICK = 12
FS_LEGEND = 11
FS_TITLE = 15

# ---------------------------------------------------------------------------
# Y-axis limits
# ---------------------------------------------------------------------------
Y_MIN, Y_MAX = 40, 300

# ---------------------------------------------------------------------------
# Thresholds and parameters
# ---------------------------------------------------------------------------
H_HYPER = 180
DAYS_TO_PLOT = 7


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


# =============================================================================
# TIME-AXIS FORMATTING
# =============================================================================

class RelativeDayFormatter(mdates.DateFormatter):
    """Format x-axis as relative day numbers (Day 1, Day 2, etc.)."""

    def __init__(self, t_start):
        super().__init__("")
        if hasattr(t_start, 'tz') and t_start.tz is not None:
            self.day_start = t_start.tz_localize(None).normalize()
        else:
            self.day_start = t_start.normalize()

    def __call__(self, x, pos=None):
        dt = mdates.num2date(x)
        if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
            dt_pd = pd.Timestamp(dt).tz_localize(None)
        else:
            dt_pd = pd.Timestamp(dt)
        dt_normalized = dt_pd.normalize()
        day_num = (dt_normalized - self.day_start).days + 1
        return f"Day {day_num}"


# =============================================================================
# FIGURE
# =============================================================================

def build_figure(patient_id: str, t_data, g_original, g_ctf, slh_episodes,
                 offset_min: float, t_start: pd.Timestamp, t_end: pd.Timestamp) -> plt.Figure:
    """Compose and return the 7-day CGM comparison figure."""

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

    # ── 2. SLH episode spans ──────────────────────────────────────────────────
    episode_labeled = False

    for ep in slh_episodes:
        if ep["offset_time"] < t_start or ep["onset_time"] > t_end:
            continue

        label = "SLH episode" if not episode_labeled else "_"
        ax.axvspan(
            max(ep["onset_time"], t_start), min(ep["offset_time"], t_end),
            color=COLOR_EPISODE, alpha=0.20, zorder=1, linewidth=0, label=label
        )
        episode_labeled = True

    # ── 3. Threshold line ─────────────────────────────────────────────────────
    ax.axhline(H_HYPER, color=COLOR_THRESHOLD, ls="--", lw=1.5, alpha=0.75, zorder=2)

    # ── 4. Original signal (G(t)) – dashed, lighter ───────────────────────────
    ax.plot(t_data, g_original, color=COLOR_ORIGINAL, lw=0.8, ls='--', alpha=0.65,
            zorder=4, label="G(t)")

    # ── 5. Counterfactual signal (C(t)) – solid, darker ───────────────────────
    color_ctf = OFFSET_COLORS.get(offset_min, "#0072B2")
    ax.plot(t_data, g_ctf, color=color_ctf, lw=1.0, ls='-', alpha=0.85,
            zorder=5, label=f"C(t) δ={offset_min} min")

    # ── 6. Axes configuration ─────────────────────────────────────────────────
    ax.set_xlim(t_start, t_end)
    ax.set_ylim(Y_MIN, Y_MAX)

    # X-axis: relative day numbers
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(RelativeDayFormatter(t_start))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=FS_TICK, rotation=0, ha="center")

    ax.set_xlabel("Time (days)", fontsize=FS_LABEL, labelpad=8)
    ax.set_ylabel("Glucose (mg/dL)", fontsize=FS_LABEL, labelpad=8)

    # Title with patient ID
    ax.set_title(
        f"7-Day CGM Overview — Participant {patient_id}",
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
        loc="upper right",
        framealpha=0.90,
        edgecolor="black",
        fancybox=False,
        shadow=False
    )

    return fig


# =============================================================================
# PIPELINE
# =============================================================================

def generate(patient_id: str, offset_min: float, output_dir: str) -> None:
    """Generate and save the 7-day CGM counterfactual comparison figure."""
    df, timestep, slh_episodes = load_and_prepare_data(patient_id)

    # Extract 7-day window
    t_start = df[TIME_COL].iloc[0]
    t_end = t_start + pd.Timedelta(days=DAYS_TO_PLOT)

    print(f"\n  Extracting 7-day window...")
    mask = (df[TIME_COL] >= t_start) & (df[TIME_COL] < t_end)
    t_data = df.loc[mask, TIME_COL]
    g_original = df.loc[mask, GLUCOSE_COL].astype(float).copy()

    # Build counterfactual signal
    print(f"  Building counterfactual signal...")
    c_signal = build_full_counterfactual_signal(df, slh_episodes, offset_min, timestep)
    g_ctf = c_signal[mask.values].copy()

    # Handle data gaps (don't interpolate across large gaps)
    diffs = t_data.diff().dt.total_seconds().fillna(0) / 60
    gap_mask = diffs > timestep * GAP_FACTOR
    g_original[gap_mask] = np.nan
    g_ctf[gap_mask] = np.nan

    # Compute summary statistics
    orig_valid = g_original.dropna().values
    ctf_valid = g_ctf[~np.isnan(g_ctf)]

    if len(orig_valid) > 0 and len(ctf_valid) > 0:
        orig_median = np.nanmedian(orig_valid)
        ctf_median = np.nanmedian(ctf_valid)
        delta_median = ctf_median - orig_median

        print(f"    Original median: {orig_median:.1f} mg/dL")
        print(f"    Counterfactual median: {ctf_median:.1f} mg/dL")
        print(f"    Delta: {delta_median:+.1f} mg/dL")

    # Build and save figure
    fig = build_figure(patient_id, t_data, g_original, g_ctf, slh_episodes,
                      offset_min, t_start, t_end)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"counterfactual_seven_day_overview_{patient_id}_delta{offset_min}")
    save_figure(fig, out)
    print(f"\n  Saved → {out}.pdf / .svg\n")


# =============================================================================
# CLI
# =============================================================================

def parse_arguments():
    p = argparse.ArgumentParser(
        description="7-day CGM counterfactual comparison for a single patient",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--patient_id",
        type=str,
        default=None,
        help="Patient identifier (default: chosen automatically — most / fewest SLH episodes)"
    )
    p.add_argument(
        "--offset_min",
        type=int,
        default=45,
        help="Zone-expansion offset δ in minutes"
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
    print("  7-Day CGM Counterfactual Comparison")
    print("=" * 65)
    print(f"  Patient: {args.patient_id}")
    print(f"  Offset δ: {args.offset_min} min")
    print(f"  Output: {args.out_dir}")
    print("=" * 65)

    generate(args.patient_id, args.offset_min, args.out_dir)

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
