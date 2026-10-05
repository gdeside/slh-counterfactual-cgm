"""
scripts/presentation_figures/cgm_meal_spikes.py
===============================================

Synthetic two-panel figure: why a short post-meal spike (SLH) is a
*behavioural* signal, not a reason to give insulin.

Left panel  — "Without insulin"
    A meal produces a brief excursion just above 180 mg/dL.  Endogenous
    regulation brings glucose back to baseline within ~2 h: the same
    control seen in people without diabetes.  Nothing to correct.

Right panel — "Correcting the spike with insulin"
    Same meal, same spike (grey dashed = what would have happened anyway).
    A rapid-acting bolus given at the peak acts slowly (onset ~20 min, peak
    action ~1-2 h, tail ~4-5 h).  By the time it acts the spike is already
    over, so its whole effect lands on a glucose that is back at baseline:
    glucose is pushed below 70 mg/dL.  The insulin-action curve is drawn
    under the trace to show the timing mismatch.

Everything is synthetic and illustrative (no patient data); the model is
    glucose(t) = spontaneous(t) - A * cumulative_insulin_action(t)
with a gamma-shaped action profile and a leaky (counter-regulated) integral.

Usage
-----
    python3 -m scripts.presentation_figures.cgm_meal_spikes [--out_dir DIR]
"""

import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
from scipy.stats import gamma

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.visualization.style import apply_publication_style, save_figure

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "cgm_meal_spikes")

FIG_W, FIG_H = 15.0, 7.3125

COLOR_TRACE = "#E69F00"     # CGM, no insulin
COLOR_INS   = "#0072B2"     # insulin-treated trace / action curve
COLOR_H180  = "#D55E00"
COLOR_H70   = "#CC79A7"
COLOR_MEAL  = "#777777"
COLOR_HYPO  = "#CC79A7"

FS_LABEL, FS_TICK, FS_TITLE = 15, 13, 16
Y_MIN, Y_MAX = 35, 235

# Insulin model (illustrative)
T_MEAL       = 0.0
T_BOLUS      = 0.75         # h after meal: bolus given at the spike peak
ACTION_SHAPE = 3.2          # gamma shape  -> peak action ~ 1.5 h after bolus
ACTION_SCALE = 0.5
EFFECT_MG_DL = 68.0         # glucose-lowering scale of the bolus (mg/dL)
TAU_COUNTERREG = 1.6        # h, counter-regulatory decay of the effect
T_END        = 6.0


def spontaneous_curve(t: np.ndarray) -> np.ndarray:
    """Short post-meal spike, self-limited return to baseline (hours after meal)."""
    t_ctrl = np.array([-1.0, -0.3, 0.0, 0.2, 0.5, 0.75, 1.0, 1.4, 2.0, 2.8, 4.0, 6.0])
    g_ctrl = np.array([ 88,   88,  87,  115, 178,  195,  170, 122,  98,  90,  88,  88])
    return CubicSpline(t_ctrl, g_ctrl, bc_type="natural")(t)


def insulin_action(t: np.ndarray) -> np.ndarray:
    """Insulin action rate (unit area), zero before the bolus."""
    return gamma.pdf(np.clip(t - T_BOLUS, 0, None), a=ACTION_SHAPE, scale=ACTION_SCALE) * (t >= T_BOLUS)


def make_curves():
    t = np.linspace(-0.5, T_END, 3000)
    g_spont = spontaneous_curve(t)
    act = insulin_action(t)
    dt = t[1] - t[0]
    # glucose-lowering effect of the bolus, attenuated by counter-regulation
    # (leaky integration of the action rate)
    kernel = np.exp(-np.arange(0, T_END, dt) / TAU_COUNTERREG)
    effect = np.convolve(act, kernel)[: len(t)] * dt
    g_ins = g_spont - EFFECT_MG_DL * effect
    return t, g_spont, g_ins, act


def _style_axis(ax, ylabel: bool):
    ax.axhline(180, color=COLOR_H180, ls="--", lw=2.0, alpha=0.85, zorder=2)
    ax.axhline(70,  color=COLOR_H70,  ls="--", lw=2.0, alpha=0.85, zorder=2)
    ax.axvline(T_MEAL, color=COLOR_MEAL, ls="--", lw=1.4, alpha=0.55, zorder=2)
    ax.set_xlim(-0.5, T_END)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels(["meal", "1 h", "2 h", "3 h", "4 h", "5 h"], fontsize=FS_TICK)
    ax.set_xlabel("Time after meal", fontsize=FS_LABEL, labelpad=6)
    ax.set_yticks([70, 100, 140, 180])
    ax.tick_params(axis="y", labelsize=FS_TICK)
    if ylabel:
        ax.set_ylabel("Glucose (mg/dL)", fontsize=FS_LABEL, labelpad=6)
    else:
        ax.tick_params(labelleft=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, ls=":", lw=0.4, color="#EEEEEE", zorder=0)
    ax.set_axisbelow(True)


def build_figure(t, g_spont, g_ins, act) -> plt.Figure:
    apply_publication_style()
    matplotlib.rcParams.update({"font.size": FS_LABEL, "axes.grid": False})

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.14, wspace=0.06)

    # ── Left: no insulin ─────────────────────────────────────────────────────
    _style_axis(axL, ylabel=True)
    axL.plot(t, g_spont, color=COLOR_TRACE, lw=3.0, zorder=4)
    axL.set_title("Spike resolves on its own", fontsize=FS_TITLE, fontweight="bold", pad=12)

    # ── Right: insulin correction ────────────────────────────────────────────
    _style_axis(axR, ylabel=False)
    axR.plot(t, g_spont, color="#999999", lw=2.0, ls="--", zorder=3)
    axR.plot(t, g_ins, color=COLOR_INS, lw=3.0, zorder=4)
    axR.fill_between(t, g_ins, 70, where=g_ins < 70, color=COLOR_HYPO, alpha=0.45, zorder=3)

    # Insulin action curve, drawn as a filled profile along the bottom
    act_n = act / act.max()
    axR.fill_between(t, Y_MIN, Y_MIN + 28 * act_n, color=COLOR_INS, alpha=0.25, zorder=1)
    axR.plot(t, Y_MIN + 28 * act_n, color=COLOR_INS, lw=1.2, alpha=0.7, zorder=1)
    axR.plot([T_BOLUS], [Y_MIN], marker="v", ms=11, color=COLOR_INS, zorder=5, clip_on=False)
    axR.set_title("Correcting it with insulin", fontsize=FS_TITLE, fontweight="bold", pad=12)

    axL.legend(
        handles=[
            Line2D([0], [0], color=COLOR_TRACE, lw=3.0, label="CGM signal"),
            Line2D([0], [0], color=COLOR_H180, lw=2.0, ls="--", label="Hyper threshold (180 mg/dL)"),
            Line2D([0], [0], color=COLOR_H70,  lw=2.0, ls="--", label="Hypo threshold (70 mg/dL)"),
        ],
        loc="upper right", fontsize=FS_TICK, framealpha=0.88,
    )
    axR.legend(
        handles=[
            Line2D([0], [0], color=COLOR_INS, lw=3.0, label="CGM with insulin bolus"),
            Line2D([0], [0], color="#999999", lw=2.0, ls="--", label="Without insulin"),
            Line2D([0], [0], color=COLOR_INS, lw=8, alpha=0.25, label="Insulin action"),
        ],
        loc="upper right", fontsize=FS_TICK, framealpha=0.88,
    )
    return fig


def generate(output_dir: str) -> None:
    t, g_spont, g_ins, act = make_curves()
    over = t[g_spont > 180]
    print(f"  Spike above 180 mg/dL : {over.max() - over.min():.2f} h  (peak {g_spont.max():.0f} mg/dL)")
    print(f"  With insulin — nadir  : {g_ins.min():.0f} mg/dL at t = {t[np.argmin(g_ins)]:.2f} h")
    print(f"  Time below 70 mg/dL   : {(g_ins < 70).sum() * (t[1] - t[0]):.2f} h")

    fig = build_figure(t, g_spont, g_ins, act)
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "cgm_meal_spikes")
    save_figure(fig, out)
    print(f"  Saved → {out}.pdf / .svg")


def main():
    p = argparse.ArgumentParser(
        description="Synthetic meal spike: spontaneous resolution vs insulin-induced hypoglycaemia",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out_dir", default=DEFAULT_OUTPUT_DIR)
    generate(p.parse_args().out_dir)


if __name__ == "__main__":
    main()
