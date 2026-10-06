"""
scripts/presentation_figures/agp_counterfactual_recurrence.py
=============================================================

Two-panel AGP figure comparing original and counterfactual glucose signals
with classification-based recurrence analysis for a specific patient.

Panels:
  · TOP: AGP comparison
    - Original signal G(t) — dashed line with light IQR band
    - Counterfactual signal C(t) — solid line with hatched IQR band
    - 70 & 180 mg/dL thresholds (dashed)
    - Target zone (70–180 mg/dL) background
    - Legend with signal labels and offset value

  · BOTTOM: Classification-based recurrence heatmap
    - SLH episodes (short-lasting hyperglycaemia) — one color
    - LLH episodes (long-lasting hyperglycaemia, plus other non-SLH episodes) — another color
    - 50% systematic threshold line
    - Time-of-day on x-axis (0–24h)

  · Minimalist axes (no top/right spines)

Usage
-----
    python3 -m scripts.presentation_figures.agp_counterfactual_recurrence         [--patient_id ID]         [--offset_min 45]         [--out_dir DIR]

Output
------
    scripts/presentation_figures/output/agp_counterfactual_recurrence/  (git-ignored; --out_dir to change)
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, time as _dtime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.patient_selection import resolve_patient_id
from src.preprocessing.load_data import load_patient_window
from src.preprocessing.timestep import detect_dominant_sampling_interval
from src.episodes.detection import detect_hyperglycemia_episodes
from src.episodes.criteria import classify_episodes
from src.episodes.episode_config import LABEL_SLH
from src.episodes.load_episodes import load_patient_episodes
from src.config import TIME_COL, GLUCOSE_COL, COL_FLAG_TIME_ERROR, COL_FLAG_OUTLIER_GLUCOSE
from src.counterfactual.build_counterfactual import build_full_counterfactual_signal
from src.agp.compute_agp import compute_agp_profile, compute_recurrence_by_classification
from src.visualization.style import apply_publication_style, save_figure, style_axes
from src.visualization.colors import OFFSET_COLORS
from src.visualization.agp_plot import plot_recurrence_classification_panel

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "agp_counterfactual_recurrence")

# ---------------------------------------------------------------------------
# Figure geometry — 2-panel layout, 16:9 overall
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 13.0, 9.0         # 16:9 ratio with 2 panels
FIG_W_TWO, FIG_H_TWO = 22.0, 9.0  # side-by-side two patients
HEIGHT_RATIOS = [2, 1]            # AGP (top, 2x) vs Recurrence (bottom, 1x)

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
COLOR_ORIGINAL = "#333333"          # dark grey — original signal
COLOR_THRESHOLD_HYPER = "#D55E00"   # vermilion — 180 mg/dL
COLOR_THRESHOLD_HYPO = "#CC79A7"    # mauve — 70 mg/dL
COLOR_TARGET_ZONE = "#009E73"       # green — 70-180 target zone

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
Y_MIN, Y_MAX = 50, 350

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
H_HYPO = 70
H_HYPER = 180


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
# AGP PROFILE COMPUTATION
# =============================================================================

def compute_agp_with_counterfactual(df, slh_episodes, offset_min, timestep_min):
    """Compute AGP for both original and counterfactual signals."""
    # Original signal AGP
    agp_original = compute_agp_profile(df, timestep_min)

    # Counterfactual signal
    c_signal = build_full_counterfactual_signal(df, slh_episodes, offset_min, timestep_min)
    df_ctf = df.copy()
    df_ctf[GLUCOSE_COL] = c_signal
    agp_ctf = compute_agp_profile(df_ctf, timestep_min)

    return agp_original, agp_ctf


# =============================================================================
# FIGURE
# =============================================================================

def build_figure(patient_id: str, agp_original, agp_ctf, offset_min: float,
                 episodes_df, valid_data, timestep_min) -> plt.Figure:
    """Compose and return the 2-panel AGP comparison + recurrence figure."""
    apply_publication_style()
    matplotlib.rcParams.update({
        "font.size": FS_LABEL,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "axes.grid": False,
    })

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, FIG_H), height_ratios=HEIGHT_RATIOS)
    fig.subplots_adjust(left=0.09, right=0.93, top=0.94, bottom=0.10, hspace=0.35)

    _draw_patient_column(axes, patient_id, agp_original, agp_ctf, offset_min,
                         episodes_df, valid_data, timestep_min)
    return fig


# =============================================================================
# SINGLE-COLUMN HELPER  (reusable for both single and two-patient figures)
# =============================================================================

def _draw_patient_column(axes_col, patient_id, agp_original, agp_ctf, offset_min,
                         episodes_df, valid_data, timestep_min):
    """Draw AGP + recurrence panels into a (ax_agp, ax_rec) column pair."""
    ax = axes_col[0]

    def get_agp_x_axis(agp):
        today = datetime.today().date()
        return [
            datetime.combine(today, datetime.strptime(t, "%H:%M").time())
            for t in agp["time_str"]
        ]

    x_orig = get_agp_x_axis(agp_original)
    x_ctf = get_agp_x_axis(agp_ctf)

    ax.axhspan(H_HYPO, H_HYPER, alpha=0.08, color=COLOR_TARGET_ZONE, zorder=0)
    ax.axhline(H_HYPER, color=COLOR_THRESHOLD_HYPER, ls="--", lw=1.5, alpha=0.75, zorder=2)
    ax.axhline(H_HYPO, color=COLOR_THRESHOLD_HYPO, ls="--", lw=1.5, alpha=0.75, zorder=2)

    ax.fill_between(x_orig, agp_original["q25"], agp_original["q75"],
                    alpha=0.15, color=COLOR_ORIGINAL, zorder=1, linewidth=0)
    ax.plot(x_orig, agp_original["q25"], color=COLOR_ORIGINAL, lw=0.9, ls=':', alpha=0.55, zorder=3)
    ax.plot(x_orig, agp_original["q75"], color=COLOR_ORIGINAL, lw=0.9, ls=':', alpha=0.55, zorder=3)
    ax.plot(x_orig, agp_original["median"], color=COLOR_ORIGINAL, lw=2.0, ls='--',
            alpha=0.80, label="G(t)", zorder=5)

    color_ctf = OFFSET_COLORS.get(offset_min, "#0072B2")
    ax.fill_between(x_ctf, agp_ctf["q25"], agp_ctf["q75"],
                    alpha=0.28, color=color_ctf, zorder=2, linewidth=0,
                    hatch='///', edgecolor=color_ctf)
    ax.plot(x_ctf, agp_ctf["q25"], color=color_ctf, lw=1.0, ls=':', alpha=0.70, zorder=4)
    ax.plot(x_ctf, agp_ctf["q75"], color=color_ctf, lw=1.0, ls=':', alpha=0.70, zorder=4)
    ax.plot(x_ctf, agp_ctf["median"], color=color_ctf, lw=2.0, ls='-',
            alpha=0.85, label=f"C(t) δ={offset_min} min", zorder=6)

    today = datetime.today().date()
    ax.set_xlim(
        datetime.combine(today, _dtime(0, 0)),
        datetime.combine(today, _dtime(23, 59))
    )
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=FS_TICK, rotation=0, ha="center")
    ax.set_xlabel("Time of day", fontsize=FS_LABEL, labelpad=8)
    ax.set_ylabel("Glucose (mg/dL)", fontsize=FS_LABEL, labelpad=8)
    ax.set_yticks([70, 100, 140, 180, 250, 350])
    ax.set_title(
        f"Ambulatory Glucose Profile — Participant {patient_id}",
        fontsize=FS_TITLE, fontweight='bold', pad=14
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)
    ax.yaxis.grid(True, ls=":", lw=0.4, color="#EEEEEE", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=FS_LEGEND, loc="upper right", framealpha=0.90,
              edgecolor="black", fancybox=False, shadow=False)

    ax_recurrence = axes_col[1]
    start_date = valid_data[TIME_COL].dt.date.min()
    end_date = valid_data[TIME_COL].dt.date.max()
    n_days = (end_date - start_date).days + 1
    recurrence = compute_recurrence_by_classification(
        episodes_df, timestep_min, n_days, start_date
    )
    plot_recurrence_classification_panel(ax_recurrence, recurrence, patient_id, timestep_min)


# =============================================================================
# TWO-PATIENT FIGURE
# =============================================================================

def build_figure_two_patients(
    patient_id1, agp_orig1, agp_ctf1, episodes_df1, valid_data1, timestep1,
    patient_id2, agp_orig2, agp_ctf2, episodes_df2, valid_data2, timestep2,
    offset_min
) -> plt.Figure:
    """1 row × 2 columns, each column = AGP (top) + recurrence (bottom)."""
    apply_publication_style()
    matplotlib.rcParams.update({
        "font.size": FS_LABEL,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "axes.grid": False,
    })

    fig, axes = plt.subplots(
        2, 2,
        figsize=(FIG_W_TWO, FIG_H_TWO),
        height_ratios=HEIGHT_RATIOS,
    )
    fig.subplots_adjust(left=0.06, right=0.97, top=0.94, bottom=0.10,
                        hspace=0.38, wspace=0.30)

    _draw_patient_column(axes[:, 0], patient_id1, agp_orig1, agp_ctf1, offset_min,
                         episodes_df1, valid_data1, timestep1)
    _draw_patient_column(axes[:, 1], patient_id2, agp_orig2, agp_ctf2, offset_min,
                         episodes_df2, valid_data2, timestep2)

    return fig


# =============================================================================
# PIPELINE
# =============================================================================

def _load_patient_all(patient_id: str, offset_min: float):
    """Load all data needed for one patient column."""
    df, timestep, slh_episodes = load_and_prepare_data(patient_id)
    print(f"  Computing AGP profiles for {patient_id}...")
    agp_original, agp_ctf = compute_agp_with_counterfactual(df, slh_episodes, offset_min, timestep)
    print(f"  Loading episodes for {patient_id}...")
    episodes_df = load_patient_episodes(patient_id)
    if episodes_df is None or len(episodes_df) == 0:
        episodes_df = pd.DataFrame()
    return df, timestep, agp_original, agp_ctf, episodes_df


def generate(patient_id: str, offset_min: float, output_dir: str) -> None:
    """Generate and save the AGP counterfactual comparison figure for one patient."""
    df, timestep, agp_original, agp_ctf, episodes_df = _load_patient_all(patient_id, offset_min)

    fig = build_figure(patient_id, agp_original, agp_ctf, offset_min,
                       episodes_df, df, timestep)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"agp_counterfactual_{patient_id}_delta{offset_min}")
    save_figure(fig, out)
    print(f"\n  Saved → {out}.pdf / .svg\n")


def generate_two_patients(patient_id1: str, patient_id2: str,
                          offset_min: float, output_dir: str) -> None:
    """Generate and save a side-by-side two-patient AGP counterfactual figure."""
    df1, ts1, agp_orig1, agp_ctf1, ep1 = _load_patient_all(patient_id1, offset_min)
    df2, ts2, agp_orig2, agp_ctf2, ep2 = _load_patient_all(patient_id2, offset_min)

    fig = build_figure_two_patients(
        patient_id1, agp_orig1, agp_ctf1, ep1, df1, ts1,
        patient_id2, agp_orig2, agp_ctf2, ep2, df2, ts2,
        offset_min,
    )

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(
        output_dir,
        f"agp_counterfactual_{patient_id1}_{patient_id2}_delta{offset_min}"
    )
    save_figure(fig, out)
    print(f"\n  Saved → {out}.pdf / .svg\n")


# =============================================================================
# CLI
# =============================================================================

def parse_arguments():
    p = argparse.ArgumentParser(
        description="AGP counterfactual comparison (one or two patients)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--patient_id", type=str, default=None,
                   help="First (or only) patient identifier")
    p.add_argument("--patient_id2", type=str, default=None,
                   help="Second patient identifier — triggers side-by-side layout")
    p.add_argument("--offset_min", type=int, default=45,
                   help="Zone-expansion offset δ in minutes")
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                   help="Output directory")
    return p


def main():
    args = parse_arguments().parse_args()
    args.patient_id = resolve_patient_id(args.patient_id)

    print("\n" + "=" * 65)
    print("  AGP Counterfactual Comparison")
    print("=" * 65)
    print(f"  Patient 1: {args.patient_id}")
    if args.patient_id2:
        print(f"  Patient 2: {args.patient_id2}  (side-by-side)")
    print(f"  Offset δ: {args.offset_min} min")
    print(f"  Output: {args.out_dir}")
    print("=" * 65)

    if args.patient_id2:
        generate_two_patients(args.patient_id, args.patient_id2,
                              args.offset_min, args.out_dir)
    else:
        generate(args.patient_id, args.offset_min, args.out_dir)

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
