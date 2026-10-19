"""
scripts/presentation_figures/cgm_meal_spikes.py
===============================================

Synthetic two-panel figure: why a short post-meal spike (SLH) is a
*behavioural* signal, not a reason to give insulin.

Six "days" of the same person, each with a slightly different post-meal
excursion (height, width, timing), one colour per day.  This is an
introduction figure: days are *not* classified here (SLH or not) — the point
is only that the same meal behaviour gives slightly different excursions from
one day to the next, some crossing 180 mg/dL briefly, some not, some lasting
a little longer.

Left panel  — "Without insulin"
    The six days superimposed.  Every excursion resolves on its own through
    endogenous regulation.

Right panel — "Correcting with insulin"
    Same six days (thin dashed = what would have happened anyway).  A
    rapid-acting bolus acts slowly (onset ~20 min, peak action ~1-2 h, tail
    ~4-5 h).  On the days where the excursion is already over, its whole
    effect lands on a glucose back at baseline and pushes it below 70 mg/dL
    (pink); on the longest excursion the bolus meets glucose that is still
    high.  The insulin-action curve is drawn under the traces.

Everything is synthetic and illustrative (no patient data); the model is
    glucose(t) = spontaneous(t) - A * cumulative_insulin_action(t)
with a gamma-shaped action profile and a leaky (counter-regulated) integral.

Usage
-----
    python3 -m scripts.presentation_figures.cgm_meal_spikes [--out_dir DIR]

    # Same figure in mmol/L (default is mg/dL; output filename gets a _mmoll suffix)
    CGM_GLUCOSE_UNIT="mmol/L" python3 -m scripts.presentation_figures.cgm_meal_spikes

Output
------
    scripts/presentation_figures/output/cgm_meal_spikes/  (git-ignored; --out_dir to change)
"""


import os
import sys
import argparse
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gamma

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.episodes.episode_config import H_HYPER, H_HYPO
from src.visualization.style import apply_publication_style, save_figure
from src.visualization.glucose_axis import apply_glucose_axis
from src.utils.unit_conversion import format_glucose

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "cgm_meal_spikes")

FIG_W, FIG_H = 15.0, 7.3125

# Day colours: one gradient, darker = higher peak (assigned after DAYS below);
# kept clear of the orange / pink / blue used for thresholds, hypo and insulin.
COLOR_INS   = "#0072B2"     # insulin action curve
COLOR_H180  = "#D55E00"
COLOR_H70   = "#CC79A7"
COLOR_MEAL  = "#777777"
COLOR_HYPO  = "#CC79A7"

FS_LABEL, FS_TICK, FS_TITLE = 15, 13, 16
Y_MIN, Y_MAX = 35, 245          # mg/dL (data stay in mg/dL; only labels change)

# Insulin model (illustrative)
T_MEAL       = 0.0
ACTION_SHAPE = 3.2          # gamma shape  -> peak action ~ 1.5 h after bolus
ACTION_SCALE = 0.5
TAU_COUNTERREG = 1.6        # h, counter-regulatory decay of the effect
T_START, T_END = -0.75, 6.0


@dataclass
class Day:
    """One synthetic post-meal excursion (asymmetric bump over a baseline)."""
    base: float      # baseline glucose (mg/dL)
    peak: float      # peak glucose (mg/dL)
    t_peak: float    # time of the peak (h after meal)
    s_rise: float    # rise width (h)
    s_fall: float    # fall width (h)
    t_bolus: float   # insulin bolus time (h after meal; < 0 = taken before the meal)
    dose: float      # glucose-lowering scale of the bolus (mg/dL)


# Slightly different days: height, width and timing all vary (some cross 180
# mg/dL briefly, one never does, one stays above it a little longer).  The
# bolus differs too: taken before the meal it flattens the excursion (and may
# prevent the hyperglycaemia altogether); taken at or after the peak it lands
# on a glucose that is already coming back down.
DAYS = [
    Day(88, 202, 0.70, 0.17, 0.30,  0.75,  62),  # bolus at the peak        -> hypo
    Day(92, 216, 0.60, 0.15, 0.36, -0.50,  70),  # bolus before the meal    -> spike shortened
    Day(86, 192, 0.80, 0.20, 0.27,  0.75,  62),  # bolus at the peak        -> hypo
    Day(90, 168, 0.70, 0.20, 0.35, -0.50,  40),  # bolus before the meal    -> stays in range
    Day(90, 208, 0.85, 0.28, 1.05, -0.50, 100),  # bolus before the meal    -> hyper avoided
    Day(94, 232, 0.90, 0.30, 1.30, -0.25,  90),  # bolus just before        -> hyper halved
]


def _day_colors(days) -> list[str]:
    """Viridis gradient ranked by peak glucose: highest peak darkest."""
    ranks = np.argsort(np.argsort([d.peak for d in days]))     # 0 = lowest peak
    return [matplotlib.colors.to_hex(plt.cm.viridis(0.85 * (1 - r / (len(days) - 1))))
            for r in ranks]


DAY_COLORS = _day_colors(DAYS)


def spontaneous_curve(t: np.ndarray, day: Day) -> np.ndarray:
    """Self-limited post-meal excursion returning to baseline."""
    sigma = np.where(t < day.t_peak, day.s_rise, day.s_fall)
    return day.base + (day.peak - day.base) * np.exp(-0.5 * ((t - day.t_peak) / sigma) ** 2)


def minutes_above(t: np.ndarray, g: np.ndarray, threshold: float = H_HYPER) -> float:
    """Total time (min) the curve spends above *threshold*."""
    return float((g > threshold).sum() * (t[1] - t[0]) * 60)


def insulin_action(t: np.ndarray, t_bolus: float) -> np.ndarray:
    """Insulin action rate (unit area), zero before the bolus."""
    return gamma.pdf(np.clip(t - t_bolus, 0, None), a=ACTION_SHAPE, scale=ACTION_SCALE) * (t >= t_bolus)


def make_curves():
    t = np.linspace(T_START, T_END, 3000)
    dt = t[1] - t[0]
    # glucose-lowering effect of a bolus, attenuated by counter-regulation
    # (leaky integration of the action rate)
    kernel = np.exp(-np.arange(0, T_END - T_START, dt) / TAU_COUNTERREG)
    g_spont, g_ins, acts = [], [], []
    for d in DAYS:
        act = insulin_action(t, d.t_bolus)
        effect = np.convolve(act, kernel)[: len(t)] * dt
        g0 = spontaneous_curve(t, d)
        g_spont.append(g0)
        g_ins.append(g0 - d.dose * effect)
        acts.append(act)
    return t, g_spont, g_ins, acts


def _style_axis(ax, ylabel: bool):
    ax.axhline(H_HYPER, color=COLOR_H180, ls="--", lw=2.0, alpha=0.85, zorder=2)
    ax.axhline(H_HYPO,  color=COLOR_H70,  ls="--", lw=2.0, alpha=0.85, zorder=2)
    ax.axvline(T_MEAL, color=COLOR_MEAL, ls="--", lw=1.4, alpha=0.55, zorder=2)
    ax.set_xlim(T_START, T_END)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels(["meal", "1 h", "2 h", "3 h", "4 h", "5 h"], fontsize=FS_TICK)
    ax.set_xlabel("Time after meal", fontsize=FS_LABEL, labelpad=6)
    apply_glucose_axis(ax, mgdl_ticks=[70, 100, 140, 180, 220], set_label=ylabel,
                       fontsize=FS_LABEL, labelpad=6)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    if not ylabel:
        ax.tick_params(labelleft=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True, ls=":", lw=0.4, color="#EEEEEE", zorder=0)
    ax.set_axisbelow(True)


def build_figure(t, g_spont, g_ins, acts) -> plt.Figure:
    apply_publication_style()
    matplotlib.rcParams.update({"font.size": FS_LABEL, "axes.grid": False})

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.14, wspace=0.06)

    # ── Left: no insulin ─────────────────────────────────────────────────────
    _style_axis(axL, ylabel=True)
    for g, color in zip(g_spont, DAY_COLORS):
        axL.plot(t, g, color=color, lw=2.8, zorder=4)
    axL.set_title("Excursions resolve on their own", fontsize=FS_TITLE,
                  fontweight="bold", pad=12)

    # ── Right: insulin correction ────────────────────────────────────────────
    _style_axis(axR, ylabel=False)
    for g0, g, color in zip(g_spont, g_ins, DAY_COLORS):
        axR.plot(t, g0, color=color, lw=1.1, ls="--", alpha=0.45, zorder=2)
        axR.plot(t, g, color=color, lw=2.8, zorder=4)
        axR.fill_between(t, g, H_HYPO, where=g < H_HYPO, color=COLOR_HYPO,
                         alpha=0.35, zorder=3, linewidth=0)

    # Insulin action of each day's bolus along the bottom, in the day's colour,
    # with a marker at the bolus time
    for d, act, color in zip(DAYS, acts, DAY_COLORS):
        act_n = 24 * act / act.max()
        axR.fill_between(t, Y_MIN, Y_MIN + act_n, color=color, alpha=0.10, zorder=1, linewidth=0)
        axR.plot(t, Y_MIN + act_n, color=color, lw=1.3, alpha=0.8, zorder=1)
        axR.plot([d.t_bolus], [Y_MIN], marker="v", ms=10, color=color,
                 markeredgecolor="white", zorder=5, clip_on=False)
    axR.set_title("With an insulin bolus: it depends on its timing", fontsize=FS_TITLE,
                  fontweight="bold", pad=12)

    axL.legend(
        handles=[
            Line2D([0], [0], color="#555555", lw=2.8, label="One curve per day (darker = higher peak)"),
            Line2D([0], [0], color=COLOR_H180, lw=2.0, ls="--",
                   label=f"Hyper threshold ({format_glucose(H_HYPER)})"),
            Line2D([0], [0], color=COLOR_H70, lw=2.0, ls="--",
                   label=f"Hypo threshold ({format_glucose(H_HYPO)})"),
        ],
        loc="upper right", fontsize=FS_TICK - 1, framealpha=0.88,
    )
    axR.legend(
        handles=[
            Line2D([0], [0], color="#555555", lw=2.8, label="Same days, with insulin bolus"),
            Line2D([0], [0], color="#555555", lw=0, marker="v", ms=9,
                   label="Bolus time (before / at / after the meal)"),
            Line2D([0], [0], color="#555555", lw=1.4, ls="--", alpha=0.5, label="Without insulin"),
            Patch(facecolor=COLOR_HYPO, alpha=0.45,
                  label=f"Hypoglycaemia  (< {format_glucose(H_HYPO)})"),
            Patch(facecolor="#555555", alpha=0.25, label="Insulin action (same colour as the day)"),
        ],
        loc="upper right", fontsize=FS_TICK - 1, framealpha=0.88,
    )
    return fig


def generate(output_dir: str) -> None:
    t, g_spont, g_ins, acts = make_curves()
    for i, (g0, g) in enumerate(zip(g_spont, g_ins), 1):
        hypo = (g < H_HYPO).sum() * (t[1] - t[0]) * 60
        print(f"  Day {i}: bolus at {DAYS[i-1].t_bolus:+.2f} h | peak {format_glucose(g0.max())}, "
              f"above {format_glucose(H_HYPER, False)}: {minutes_above(t, g0):4.0f} min  |  "
              f"with insulin peak {format_glucose(g[t >= 0].max(), False)}, above: {minutes_above(t, g):3.0f} min, nadir {format_glucose(g.min())}, below hypo {hypo:3.0f} min")

    fig = build_figure(t, g_spont, g_ins, acts)
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "cgm_meal_spikes")
    save_figure(fig, out)
    print(f"  Saved → {out}_<unit>.pdf / .svg")


def main():
    p = argparse.ArgumentParser(
        description="Synthetic meal excursions: SLH vs not, and insulin-induced hypoglycaemia",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out_dir", default=DEFAULT_OUTPUT_DIR)
    generate(p.parse_args().out_dir)


if __name__ == "__main__":
    main()
