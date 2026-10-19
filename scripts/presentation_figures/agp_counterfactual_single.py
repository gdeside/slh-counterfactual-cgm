"""
scripts/presentation_figures/agp_counterfactual_single.py
=========================================================

AGP of one patient, observed vs counterfactual — presentation version of the
AGP panel of paper Figure 4 (row 3,
scripts/paper_figures/generate_figure_4_counterfactual_construction.py).

One panel:
  · Original G(t)        median (dashed grey) + IQR band (grey, dotted edges)
  · Counterfactual C(t)  median (solid) + IQR band (hatched, dotted edges)
  · Time-of-day slot where the two medians differ the most:
    vertical bracket between them + "max |Δ| = … @ hh:mm" annotation
    (same formula as the cohort-level sensitivity analysis:
     Δmedian(t) = median_G(t) − median_C(t))
  · Hypo / hyper thresholds and target-range shading

The counterfactual shifts the SLH episodes of the patient by --offset_min
minutes (default 30, the offset of the main analysis).  No patient ID is
printed on the figure.

Usage
-----
    python3 -m scripts.presentation_figures.agp_counterfactual_single
            [--patient_id ID] [--offset_min 30] [--out_dir DIR]

    # Same figure in mmol/L (default is mg/dL; output filename gets a _mmoll suffix)
    CGM_GLUCOSE_UNIT="mmol/L" python3 -m scripts.presentation_figures.agp_counterfactual_single

Output
------
    scripts/presentation_figures/output/agp_counterfactual_single/  (git-ignored; --out_dir to change)
        agp_counterfactual_single_<patient_id>_offset<δ>.{pdf,svg}
"""

import os
import sys
import argparse
from datetime import datetime, time as _dtime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.patient_selection import resolve_patient_id
from src.agp.compute_agp import compute_agp_profile
from src.config import GLUCOSE_COL
from src.episodes.episode_config import H_HYPER, H_HYPO
from src.counterfactual.build_counterfactual import build_full_counterfactual_signal
from src.visualization.style import apply_publication_style, save_figure
from src.visualization.colors import OFFSET_COLORS
from src.visualization.glucose_axis import apply_glucose_axis
from src.utils.unit_conversion import format_glucose
from scripts.presentation_figures.counterfactual_seven_day_overview import load_and_prepare_data

DEFAULT_OFFSET_MIN = 30
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "agp_counterfactual_single")

FIG_W, FIG_H = 10.0, 5.6
FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT = 14, 12, 12, 12

COLOR_ORIGINAL = "#333333"
COLOR_THRESHOLD = "#D55E00"
COLOR_HYPO = "#E69F00"
COLOR_TARGET_ZONE = "#009E73"
COLOR_MAX_DIFF = "#B5121B"
Y_MIN, Y_MAX = 50, 350          # mg/dL, widest allowed range; tightened to the data


def max_median_diff(agp_original, agp_ctf):
    """Time slot where |median_G − median_C| is largest (None if no valid slot)."""
    diff = agp_original["median"].to_numpy() - agp_ctf["median"].to_numpy()
    valid = ~np.isnan(diff)
    if not valid.any():
        return None
    idx = int(np.argmax(np.where(valid, np.abs(diff), -np.inf)))
    return {
        "slot_idx": idx,
        "max_abs_diff": float(abs(diff[idx])),
        "time_of_max": agp_original["time_str"].iloc[idx],
        "median_original": float(agp_original["median"].iloc[idx]),
        "median_ctf": float(agp_ctf["median"].iloc[idx]),
    }


def plot_agp_counterfactual(df, slh_episodes, timestep_min, offset_min) -> plt.Figure:
    agp_orig = compute_agp_profile(df, timestep_min)
    df_ctf = df.copy()
    df_ctf[GLUCOSE_COL] = build_full_counterfactual_signal(df, slh_episodes, offset_min, timestep_min)
    agp_ctf = compute_agp_profile(df_ctf, timestep_min)

    today = datetime.today().date()
    x_orig = [datetime.combine(today, datetime.strptime(t, "%H:%M").time()) for t in agp_orig["time_str"]]
    x_ctf = [datetime.combine(today, datetime.strptime(t, "%H:%M").time()) for t in agp_ctf["time_str"]]
    color_ctf = OFFSET_COLORS.get(offset_min, "#0072B2")

    apply_publication_style()
    matplotlib.rcParams.update({"font.size": FS_TICK, "axes.labelsize": FS_LABEL,
                                "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK})
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # y-limits tightened to the IQR bands of both profiles
    vals = np.concatenate([agp_orig["q25"], agp_orig["q75"], agp_ctf["q25"], agp_ctf["q75"]])
    y_lo = max(Y_MIN, np.floor(np.nanmin(vals) / 10) * 10 - 10)
    y_hi = min(Y_MAX, np.ceil(np.nanmax(vals) / 10) * 10 + 10)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(datetime.combine(today, _dtime(0, 0)), datetime.combine(today, _dtime(23, 59)))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax.axhline(H_HYPER, color=COLOR_THRESHOLD, ls="--", lw=1.6, alpha=0.8, zorder=2)
    ax.axhline(H_HYPO, color=COLOR_HYPO, ls="--", lw=1.6, alpha=0.7, zorder=2)
    ax.axhspan(H_HYPO, H_HYPER, alpha=0.04, color=COLOR_TARGET_ZONE, zorder=0)

    # Original: grey IQR; counterfactual: hatched IQR
    ax.fill_between(x_orig, agp_orig["q25"], agp_orig["q75"],
                    alpha=0.12, color=COLOR_ORIGINAL, zorder=1, linewidth=0)
    for q in ("q25", "q75"):
        ax.plot(x_orig, agp_orig[q], color=COLOR_ORIGINAL, lw=1.0, ls=":", alpha=0.5, zorder=3)
    ax.fill_between(x_ctf, agp_ctf["q25"], agp_ctf["q75"], alpha=0.30, color=color_ctf,
                    zorder=2, linewidth=0, hatch="///", edgecolor=color_ctf)
    for q in ("q25", "q75"):
        ax.plot(x_ctf, agp_ctf[q], color=color_ctf, lw=1.2, ls=":", alpha=0.7, zorder=4)

    ax.plot(x_orig, agp_orig["median"], color=COLOR_ORIGINAL, lw=2.4, ls="--",
            label="Observed G(t)", zorder=5)
    ax.plot(x_ctf, agp_ctf["median"], color=color_ctf, lw=2.4, ls="-",
            label=f"Counterfactual C(t)  (δ = {offset_min} min)", zorder=6)

    # Slot where the medians differ the most
    info = max_median_diff(agp_orig, agp_ctf)
    if info is not None:
        xm = x_orig[info["slot_idx"]]
        lo, hi = sorted((info["median_original"], info["median_ctf"]))
        ax.axvline(xm, color=COLOR_MAX_DIFF, lw=1.2, ls=":", alpha=0.6, zorder=7)
        ax.plot([xm, xm], [lo, hi], color=COLOR_MAX_DIFF, lw=3.2, alpha=0.9,
                solid_capstyle="butt", zorder=8)
        ax.scatter([xm, xm], [lo, hi], color=COLOR_MAX_DIFF, s=36, zorder=9,
                   edgecolor="white", linewidth=0.6)
        ax.annotate(
            f"max |Δ| = {format_glucose(info['max_abs_diff'])} @ {info['time_of_max']}",
            xy=(xm, hi), xycoords="data", xytext=(0.03, 0.05), textcoords="axes fraction",
            fontsize=FS_ANNOT, color=COLOR_MAX_DIFF, fontweight="bold", ha="left", va="bottom",
            arrowprops=dict(arrowstyle="-", color=COLOR_MAX_DIFF, lw=0.9, alpha=0.55,
                            shrinkA=0, shrinkB=4),
            zorder=10,
        )

    apply_glucose_axis(ax, fontsize=FS_LABEL, labelpad=6)   # after ylim is final
    ax.set_xlabel("Time of day", fontsize=FS_LABEL, labelpad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, ls=":", lw=0.5, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=FS_LEGEND, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    return fig


def parse_arguments():
    p = argparse.ArgumentParser(
        description="AGP observed vs counterfactual — presentation figure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--patient_id", default=None,
                   help="Patient ID (default: chosen automatically — most SLH episodes)")
    p.add_argument("--offset_min", type=int, default=DEFAULT_OFFSET_MIN,
                   help="Counterfactual offset δ in minutes")
    p.add_argument("--out_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    return p


def main():
    args = parse_arguments().parse_args()
    patient_id = resolve_patient_id(args.patient_id)

    df, timestep_min, slh_episodes = load_and_prepare_data(patient_id)
    fig = plot_agp_counterfactual(df, slh_episodes, timestep_min, args.offset_min)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir,
                       f"agp_counterfactual_single_{patient_id}_offset{args.offset_min}")
    save_figure(fig, out)
    print(f"  Saved → {out}_<unit>.pdf / .svg")


if __name__ == "__main__":
    main()
