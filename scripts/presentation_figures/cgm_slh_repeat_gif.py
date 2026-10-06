"""
scripts/presentation_figures/cgm_slh_repeat_gif.py
==================================================

Synthetic GIF: what happens to the median (AGP) profile and to the usual CGM
metrics when the same short-lasting hyperglycaemia (SLH) recurs day after day
around 10:30.

The record has a fixed number of days, all displayed from the start.  Frame 0
has no SLH at all; each following frame injects the 10:30 spike into one more
day (day 1, then day 2, …) until every day has one.  Two versions of the
*same* days are simulated with identical noise and identical meals:

  · "no SLH"   — the well-controlled days alone
  · "with SLH" — the same days plus a post-meal spike near 10:30

so the median profile and the metrics evolve only because SLH are added.

Left panel  — 24 h view: individual days (faint), median profile with IQR band
              (bold), and the median of the no-SLH days (grey dashed).
Right panel — TIR / TAR / mean / CV / GMI for both versions, with the delta.

Everything is synthetic and illustrative (no patient data).

Usage
-----
    python3 -m scripts.presentation_figures.cgm_slh_repeat_gif
            [--n_days 45] [--n_slh 25] [--slh_hour 10.5] [--fps 2] [--seed 3] [--out_dir DIR]

Output
------
    scripts/presentation_figures/output/cgm_slh_repeat_gif/  (git-ignored; --out_dir to change)
"""

import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.visualization.style import apply_publication_style

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "cgm_slh_repeat_gif")

FIG_W, FIG_H = 15.0, 7.3125
STEP_MIN = 5
T = np.arange(0, 24, STEP_MIN / 60)            # hours of day

COLOR_SLH   = "#E69F00"     # with SLH
COLOR_BASE  = "#666666"     # without SLH
COLOR_H180  = "#D55E00"
COLOR_H70   = "#CC79A7"
COLOR_BAND  = "#E69F00"

FS_LABEL, FS_TICK, FS_TITLE = 15, 13, 16
Y_MIN, Y_MAX = 50, 250


# =============================================================================
# SYNTHETIC DAYS
# =============================================================================

def _bump(t, centre, width, height):
    return height * np.exp(-0.5 * ((t - centre) / width) ** 2)


def _asym_spike(t, t0, height, rise=0.22, fall=0.45):
    """Fast rise / slower fall excursion starting at t0 (hours)."""
    x = t - t0
    out = np.where(x < 0, 0.0,
                   np.where(x < rise * 3, (1 - np.exp(-x / rise)),
                            (1 - np.exp(-3)) * np.exp(-(x - rise * 3) / fall)))
    return height * out


def simulate_days(n_days, slh_hour, seed):
    """Return (no_slh, with_slh) arrays of shape (n_days, len(T))."""
    rng = np.random.default_rng(seed)
    base, slh = [], []
    for _ in range(n_days):
        # slow AR(1) noise, shared by both versions
        noise = np.zeros_like(T)
        for i in range(1, len(T)):
            noise[i] = 0.97 * noise[i - 1] + rng.normal(0, 2.2)
        level = 98 + rng.normal(0, 6) + _bump(T, 4, 3.0, -8)       # slight night dip
        g = level + noise
        # ordinary, well-handled meals (never cross 180)
        for h, a in [(7.5, 28), (12.8, 32), (19.5, 34)]:
            g += _asym_spike(T, h + rng.normal(0, 0.25), a * rng.uniform(0.8, 1.15))
        g = np.clip(g, 60, None)
        # recurring SLH: fast excursion above 180 mg/dL lasting < 1 h
        # onset time, amplitude and duration all vary from day to day: some
        # spikes are brief, some long enough to fall outside the 10-60 min SLH
        # window (so they are not counted as SLH)
        t0 = slh_hour - 0.35 + rng.normal(0, 0.6)
        h_spike = rng.uniform(70, 130)
        g_slh = g + _asym_spike(T, t0, h_spike, rise=rng.uniform(0.12, 0.30),
                                fall=rng.uniform(0.15, 0.75))
        base.append(g)
        slh.append(g_slh)
    return np.array(base), np.array(slh)


# =============================================================================
# METRICS
# =============================================================================

def cgm_metrics(days: np.ndarray) -> dict:
    x = days.ravel()
    mean = x.mean()
    return {
        "TIR (70–180)": 100 * np.mean((x >= 70) & (x <= 180)),
        "TAR (>180)":   100 * np.mean(x > 180),
        "Mean (mg/dL)": mean,
        "CV (%)":       100 * x.std() / mean,
        "GMI (%)":      3.31 + 0.02392 * mean,
    }


def slh_episodes(days: np.ndarray) -> int:
    """Number of excursions above 180 mg/dL lasting 10–60 min."""
    n = 0
    for d in days:
        above = np.r_[False, d > 180, False]
        edges = np.flatnonzero(np.diff(above.astype(int)))
        for s, e in zip(edges[::2], edges[1::2]):
            if 10 <= (e - s) * STEP_MIN <= 60:
                n += 1
    return n


# =============================================================================
# ANIMATION
# =============================================================================

def build_animation(base, slh, slh_hour, fps, n_slh):
    apply_publication_style()
    matplotlib.rcParams.update({"font.size": FS_LABEL, "axes.grid": False})

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.1, 1], height_ratios=[3, 1.1],
                          left=0.06, right=0.98, top=0.88, bottom=0.11,
                          wspace=0.08, hspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    axd = fig.add_subplot(gs[1, 0], sharex=ax)
    axm = fig.add_subplot(gs[:, 1])
    n_days = len(base)
    # fixed y-range for the delta panel, from the final (all SLH added) median
    final = np.vstack([slh[:n_slh], base[n_slh:]])
    d_max = np.abs(np.median(final, axis=0) - np.median(base, axis=0)).max()
    d_lim = max(10.0, 1.4 * d_max)

    def draw(k):
        n = n_days                      # all days always shown
        cur = np.vstack([slh[:k], base[k:]])   # first k days carry an SLH
        ax.clear()
        axd.clear()
        axm.clear()

        # ── left: profiles ────────────────────────────────────────────────
        ax.axhline(180, color=COLOR_H180, ls="--", lw=2.0, alpha=0.85, zorder=2)
        ax.axhline(70,  color=COLOR_H70,  ls="--", lw=2.0, alpha=0.85, zorder=2)
        ax.axvline(slh_hour, color="#777777", ls=":", lw=1.4, alpha=0.6, zorder=1)
        for i, d in enumerate(cur):
            has = i < k
            ax.plot(T, d, color=COLOR_SLH if has else "#999999", lw=0.8,
                    alpha=0.35 if has else 0.22, zorder=2)
        q25, med, q75 = np.percentile(cur, [25, 50, 75], axis=0)
        med_base = np.median(base, axis=0)
        ax.fill_between(T, q25, q75, color=COLOR_BAND, alpha=0.28, zorder=3, lw=0)
        ax.plot(T, med_base, color=COLOR_BASE, lw=2.2, ls="--", zorder=4)
        ax.plot(T, med, color=COLOR_SLH, lw=3.4, zorder=5)

        ax.set_xlim(0, 24)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_xticks(range(0, 25, 3))
        ax.tick_params(axis="x", labelbottom=False)
        ax.set_yticks([70, 100, 140, 180, 220])
        ax.tick_params(axis="y", labelsize=FS_TICK)

        # ── left-bottom: delta of the median ──────────────────────────────
        delta = med - med_base
        axd.axhline(0, color=COLOR_BASE, lw=1.2, zorder=2)
        axd.axvline(slh_hour, color="#777777", ls=":", lw=1.4, alpha=0.6, zorder=1)
        axd.fill_between(T, 0, delta, where=delta >= 0, color=COLOR_H180, alpha=0.35, lw=0)
        axd.fill_between(T, 0, delta, where=delta < 0, color=COLOR_H70, alpha=0.35, lw=0)
        axd.plot(T, delta, color=COLOR_H180, lw=2.4, zorder=4)
        axd.set_xlim(0, 24)
        axd.set_ylim(-0.35 * d_lim, d_lim)
        axd.set_xticks(range(0, 25, 3))
        axd.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)], fontsize=FS_TICK)
        axd.tick_params(axis="y", labelsize=FS_TICK)
        axd.set_xlabel("Time of day", fontsize=FS_LABEL, labelpad=6)
        axd.set_ylabel("Δ median\n(mg/dL)", fontsize=FS_TICK, labelpad=6)
        for sp in ("top", "right"):
            axd.spines[sp].set_visible(False)
        ax.set_ylabel("Glucose (mg/dL)", fontsize=FS_LABEL, labelpad=6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.yaxis.grid(True, ls=":", lw=0.4, color="#EEEEEE", zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(f"Median profile  ·  SLH added to {k} of {n_days} days"
                     f"  ·  {slh_episodes(cur)} SLH detected",
                     fontsize=FS_TITLE, fontweight="bold", pad=12)
        ax.legend(handles=[
            Line2D([0], [0], color=COLOR_SLH, lw=3.4, label="Median profile"),
            Line2D([0], [0], color=COLOR_BAND, lw=9, alpha=0.3, label="IQR"),
            Line2D([0], [0], color=COLOR_BASE, lw=2.2, ls="--", label="Median profile without any SLH"),
        ], loc="upper left", fontsize=FS_TICK, framealpha=0.88)

        # ── right: metrics ────────────────────────────────────────────────
        m_b, m_s = cgm_metrics(base), cgm_metrics(cur)
        axm.set_xlim(0, 1)
        axm.set_ylim(0, 1)
        axm.axis("off")
        axm.set_title("CGM metrics", fontsize=FS_TITLE, fontweight="bold", pad=12)
        axm.text(0.53, 0.93, "no SLH", ha="center", fontsize=FS_TICK, color=COLOR_BASE, fontweight="bold")
        axm.text(0.75, 0.93, "with SLH", ha="center", fontsize=FS_TICK, color=COLOR_SLH, fontweight="bold")
        axm.text(0.95, 0.93, "Δ", ha="center", fontsize=FS_TICK, fontweight="bold")
        for i, key in enumerate(m_b):
            y = 0.81 - i * 0.15
            fmt = "{:.0f}" if "Mean" in key else "{:.1f}"
            d = m_s[key] - m_b[key]
            axm.text(0.0, y, key, fontsize=FS_TICK, va="center")
            axm.text(0.53, y, fmt.format(m_b[key]), ha="center", va="center", fontsize=FS_TICK, color=COLOR_BASE)
            axm.text(0.75, y, fmt.format(m_s[key]), ha="center", va="center", fontsize=FS_TICK, color="#222222")
            axm.text(0.95, y, ("+" if d >= 0 else "") + fmt.format(d), ha="center", va="center",
                     fontsize=FS_TICK, color=COLOR_H180 if abs(d) > 0.05 else "#888888")
            axm.plot([0, 1], [y - 0.075] * 2, color="#DDDDDD", lw=0.8)

    anim = FuncAnimation(fig, draw, frames=n_slh + 1, interval=1000 / fps)
    return fig, anim, draw


def generate(out_dir, n_days, n_slh, slh_hour, fps, seed):
    base, slh = simulate_days(n_days, slh_hour, seed)
    final = np.vstack([slh[:n_slh], base[n_slh:]])
    mb, ms = cgm_metrics(base), cgm_metrics(final)
    for k in mb:
        print(f"  {k:14s} {mb[k]:7.1f} → {ms[k]:7.1f}   (Δ {ms[k] - mb[k]:+.1f})")
    print(f"  SLH episodes: {slh_episodes(base)} → {slh_episodes(final)}")

    fig, anim, draw = build_animation(base, slh, slh_hour, fps, n_slh)
    os.makedirs(out_dir, exist_ok=True)
    gif = os.path.join(out_dir, "cgm_slh_repeat.gif")
    anim.save(gif, writer=PillowWriter(fps=fps), dpi=90)
    draw(n_slh)
    fig.savefig(os.path.join(out_dir, "cgm_slh_repeat_last_frame.png"), dpi=200)
    print(f"  Saved → {gif}")


def main():
    p = argparse.ArgumentParser(
        description="GIF: recurring SLH at ~10:30 → median profile and CGM metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--n_days", type=int, default=45)
    p.add_argument("--n_slh", type=int, default=25, help="days (from the first) that receive an SLH")
    p.add_argument("--slh_hour", type=float, default=10.5)
    p.add_argument("--fps", type=float, default=2)
    p.add_argument("--seed", type=int, default=3)
    a = p.parse_args()
    generate(a.out_dir, a.n_days, a.n_slh, a.slh_hour, a.fps, a.seed)


if __name__ == "__main__":
    main()
