"""
scripts/presentation_figures/offset_sensitivity_single_patient.py
=================================================================

Presentation figure (single patient, 16:9): sensitivity of the counterfactual to the
zone-expansion offset δ. Shows how different offsets affect two key metrics: Time-in-Range (TIR 70-180) and median glucose.

Shows:
  · Left axis: ΔTIR 70-180 (%) vs offset — solid line with triangle markers
  · Right axis: ΔMedian glucose (mg/dL) vs offset — dashed line with circle markers
  · Horizontal zero-line (no change)
  · Data points color-coded by offset value
  · Daytime-only metrics (06:00–22:00)
  · Clean twin-axis design
  · Minimalist axes (no top/right spines)

Usage
-----
    python3 -m scripts.presentation_figures.offset_sensitivity_single_patient         [--patient_id ID]         [--out_dir DIR]

Output (default scripts/presentation_figures/output/offset_sensitivity_single_patient/)
  offset_sensitivity_single_patient_<patient_id>.{pdf,svg}
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from src.visualization.colors import OFFSET_COLORS

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "output", "offset_sensitivity_single_patient"
)

# ---------------------------------------------------------------------------
# Figure geometry — exactly 16:9
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 13.0, 7.3125      # 16:9 ratio

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
COLOR_TIR_LINE = "#333333"          # dark grey — TIR line
COLOR_MEDIAN_LINE = "#666666"       # medium grey — median line
COLOR_ZERO = "#888888"              # light grey — zero baseline

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
H_HYPO = 70
MARKER_TIR = "^"                    # Triangle for TIR
MARKER_MEDIAN = "o"                 # Circle for median


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
# METRICS COMPUTATION
# =============================================================================

def is_night(timestamp):
    """Check if timestamp is during night hours."""
    return timestamp.hour >= NIGHT_START_H or timestamp.hour < NIGHT_END_H


def compute_metrics(glucose, time_series=None, daytime_only=False):
    """Compute TIR and median glucose."""
    if daytime_only and time_series is not None:
        daytime_mask = ~time_series.apply(is_night).values
        glucose = glucose[daytime_mask]

    g = glucose[~np.isnan(glucose)]
    if len(g) < 10:
        return {"TIR70-180": np.nan, "median": np.nan}

    n = len(g)
    return {
        "TIR70-180": round(100 * ((g >= H_HYPO) & (g <= H_HYPER)).sum() / n, 4),
        "median": round(np.median(g), 2)
    }


# =============================================================================
# FIGURE
# =============================================================================

def build_figure(patient_id: str, deltas_tir, deltas_median) -> plt.Figure:
    """Compose and return the sensitivity analysis figure."""

    apply_publication_style()
    matplotlib.rcParams.update({
        "font.size": FS_LABEL,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "axes.grid": False,
    })

    fig, ax1 = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=0.11, right=0.88, top=0.90, bottom=0.12)

    # Create twin axis for median
    ax2 = ax1.twinx()

    # ── 1. Left axis (TIR) ────────────────────────────────────────────────────
    ax1.axhline(0, color=COLOR_ZERO, linestyle="--", linewidth=0.9, alpha=0.7, zorder=1)

    ax1.plot(OFFSET_VALUES, deltas_tir, color=COLOR_TIR_LINE, lw=2.0, ls='-',
            alpha=0.85, zorder=3, label="ΔTIR 70-180")

    for i, offset in enumerate(OFFSET_VALUES):
        color = OFFSET_COLORS.get(offset, "#999999")
        ax1.scatter([offset], [deltas_tir[i]], s=80, marker=MARKER_TIR,
                   c=color, edgecolors=COLOR_TIR_LINE, linewidths=1.2, alpha=0.90, zorder=5)

    # ── 2. Right axis (Median) ────────────────────────────────────────────────
    ax2.axhline(0, color=COLOR_ZERO, linestyle="--", linewidth=0.9, alpha=0.7, zorder=1)

    ax2.plot(OFFSET_VALUES, deltas_median, color=COLOR_MEDIAN_LINE, lw=2.0, ls='--',
            alpha=0.85, zorder=4, label="ΔMedian glucose")

    for i, offset in enumerate(OFFSET_VALUES):
        color = OFFSET_COLORS.get(offset, "#999999")
        ax2.scatter([offset], [deltas_median[i]], s=70, marker=MARKER_MEDIAN,
                   c=color, edgecolors=COLOR_MEDIAN_LINE, linewidths=1.2, alpha=0.80, zorder=6)

    # ── 3. Configure left axis (TIR) ──────────────────────────────────────────
    ax1.set_xlabel("Zone-expansion offset δ (min)", fontsize=FS_LABEL, labelpad=10)
    ax1.set_ylabel("ΔTIR 70-180 (%, daytime)", fontsize=FS_LABEL, labelpad=10, color=COLOR_TIR_LINE)
    ax1.tick_params(axis='y', labelcolor=COLOR_TIR_LINE)
    ax1.set_xticks(OFFSET_VALUES)
    ax1.set_xticklabels([str(d) for d in OFFSET_VALUES], fontsize=FS_TICK)

    # ── 4. Configure right axis (Median) ──────────────────────────────────────
    ax2.set_ylabel("ΔMedian glucose (mg/dL, daytime)", fontsize=FS_LABEL, labelpad=10, color=COLOR_MEDIAN_LINE)
    ax2.tick_params(axis='y', labelcolor=COLOR_MEDIAN_LINE)

    # ── 5. Title ──────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Sensitivity Analysis — Participant {patient_id}",
        fontsize=FS_TITLE, fontweight='bold', y=0.97
    )

    # ── 6. Clean spines ───────────────────────────────────────────────────────
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_linewidth(0.8)
    ax1.spines["left"].set_linewidth(0.8)
    ax1.spines["left"].set_color(COLOR_TIR_LINE)

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_linewidth(0.8)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color(COLOR_MEDIAN_LINE)

    # Grid on left axis only
    ax1.grid(True, alpha=0.25, linewidth=0.5, linestyle=':', zorder=0)
    ax1.set_axisbelow(True)

    # ── 7. Combined legend ────────────────────────────────────────────────────
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
              fontsize=FS_LEGEND, loc='upper left', framealpha=0.92,
              edgecolor="black", fancybox=False, shadow=False)

    return fig


# =============================================================================
# PIPELINE
# =============================================================================

def generate(patient_id: str, output_dir: str) -> None:
    """Generate and save the sensitivity analysis figure."""
    df, timestep, slh_episodes = load_and_prepare_data(patient_id)

    print(f"\n  Computing sensitivity metrics...")

    # Original metrics (daytime only)
    orig_glucose = df[GLUCOSE_COL].values
    time_series = df[TIME_COL]
    orig_metrics = compute_metrics(orig_glucose, time_series, daytime_only=True)

    print(f"    Original TIR 70-180 (daytime): {orig_metrics['TIR70-180']:.2f}%")
    print(f"    Original median (daytime): {orig_metrics['median']:.1f} mg/dL")

    # Counterfactual metrics for each offset
    deltas_tir = []
    deltas_median = []

    for offset in OFFSET_VALUES:
        c_signal = build_full_counterfactual_signal(df, slh_episodes, offset, timestep)
        ctf_metrics = compute_metrics(c_signal, time_series, daytime_only=True)

        delta_tir = ctf_metrics["TIR70-180"] - orig_metrics["TIR70-180"]
        delta_median = ctf_metrics["median"] - orig_metrics["median"]

        deltas_tir.append(delta_tir)
        deltas_median.append(delta_median)

        print(f"    δ={offset:2d} min: ΔTIR={delta_tir:+6.2f}%, ΔMedian={delta_median:+6.1f} mg/dL")

    # Build and save figure
    fig = build_figure(patient_id, deltas_tir, deltas_median)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"offset_sensitivity_single_patient_{patient_id}")
    save_figure(fig, out)
    print(f"\n  Saved → {out}.pdf / .svg\n")


# =============================================================================
# CLI
# =============================================================================

def parse_arguments():
    p = argparse.ArgumentParser(
        description="Sensitivity analysis: TIR and median glucose vs offset",
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
    print("  Sensitivity Analysis")
    print("=" * 65)
    print(f"  Patient: {args.patient_id}")
    print(f"  Offsets: {OFFSET_VALUES} min")
    print(f"  Metrics: TIR 70-180 (%), Median glucose (mg/dL)")
    print(f"  Analysis: Daytime only (06:00–22:00)")
    print(f"  Output: {args.out_dir}")
    print("=" * 65)

    generate(args.patient_id, args.out_dir)

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
