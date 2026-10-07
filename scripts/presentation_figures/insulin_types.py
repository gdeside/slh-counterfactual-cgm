"""
scripts/presentation_figures/insulin_types.py
=============================================

Recreates the "types of insulin" intensity-vs-duration figure.
Sized for one column of a two-column 16:9 presentation slide
(≈ 6.5 × 6.0 inches).

White background, clean axes, annotation arrows for each curve.

Usage
-----
    python3 -m scripts.presentation_figures.insulin_types

Output: scripts/presentation_figures/output/insulin_types/insulin_types_presentation.{pdf,svg,png}
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# OUTPUT
# =============================================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "insulin_types")
OUTPUT_NAME = "insulin_types_presentation"

# =============================================================================
# COLORS  (close to original)
# =============================================================================
C_RAPID   = "#E8472A"   # red-orange
C_SHORT   = "#F0A500"   # amber/gold
C_INTER   = "#3BBFAD"   # teal
C_LONG    = "#2979BE"   # medium blue
FILL_ALPHA = 0.18

# =============================================================================
# FIGURE GEOMETRY
# One column of a 2-column 16:9 slide  →  ~6.5 × 6.0 inches
# =============================================================================
FIG_W, FIG_H = 6.5, 6.0

# Presentation font sizes
FS_TITLE  = 13
FS_LABEL  = 11
FS_TICK   = 9
FS_ANNOT  = 8
FS_LEGEND = 9.5

# =============================================================================
# CURVE HELPERS
# Each insulin type is modelled as a log-normal density, scaled to a
# desired peak intensity and peak time.
# =============================================================================

def lognorm_curve(t, peak_t, spread, amplitude):
    """
    Log-normal curve peaking at peak_t.
    spread  — shape parameter σ (larger = broader, right-skewed)
    amplitude — scale to desired peak height
    """
    sigma = spread
    mu    = np.log(peak_t)
    pdf   = (1 / (t * sigma * np.sqrt(2 * np.pi))) * np.exp(
                -((np.log(t) - mu) ** 2) / (2 * sigma ** 2))
    return amplitude * pdf / pdf.max()


def flat_lognorm(t, start, end_tau, amplitude):
    """
    'Does not peak' shape: rises slowly, stays elevated, then falls.
    Modelled as a very broad log-normal (large σ).
    """
    centre = (start + end_tau) / 2.5
    sigma  = 1.05
    mu     = np.log(centre)
    pdf    = (1 / (t * sigma * np.sqrt(2 * np.pi))) * np.exp(
                 -((np.log(t) - mu) ** 2) / (2 * sigma ** 2))
    curve  = amplitude * pdf / pdf.max()
    # suppress values before start
    curve[t < start * 0.6] = 0
    return curve


# =============================================================================
# BUILD CURVES ON COMMON TIME GRID
# =============================================================================
t = np.linspace(0.01, 24, 1200)   # hours

# Rapid-acting:  starts 5-10 min, peaks 30-90 min, lasts ~3-4 h
rapid   = lognorm_curve(t, peak_t=0.75, spread=0.38, amplitude=1.00)

# Regular/short: starts 30 min-1 h, peaks 2-5 h, lasts ~6-8 h
short   = lognorm_curve(t, peak_t=2.80, spread=0.40, amplitude=0.82)

# Intermediate:  starts 1-2 h, peaks 4-12 h, lasts ~14-16 h
inter   = lognorm_curve(t, peak_t=7.50, spread=0.55, amplitude=0.72)

# Long-acting:   starts 1-2 h, does not peak, lasts ~24 h
lng     = flat_lognorm(t, start=1.5, end_tau=22, amplitude=0.38)

# Clip to zero below a small threshold (tidy curve tails)
for arr in (rapid, short, inter, lng):
    arr[arr < 0.004] = 0

# =============================================================================
# PLOT
# =============================================================================
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.14)

# Y-axis extended to 1.80 to leave a clear annotation band above the curves
# (all curves peak at 1.0; annotations live in the 1.05–1.75 band)
Y_TOP = 1.80

# ── Filled areas ─────────────────────────────────────────────────────────────
ax.fill_between(t, rapid, color=C_RAPID, alpha=FILL_ALPHA)
ax.fill_between(t, short, color=C_SHORT, alpha=FILL_ALPHA)
ax.fill_between(t, inter,  color=C_INTER,  alpha=FILL_ALPHA)
ax.fill_between(t, lng,    color=C_LONG,   alpha=FILL_ALPHA)

# ── Curve lines ───────────────────────────────────────────────────────────────
ax.plot(t, rapid, color=C_RAPID,  lw=2.4, zorder=5)
ax.plot(t, short, color=C_SHORT,  lw=2.4, zorder=4)
ax.plot(t, inter,  color=C_INTER,  lw=2.4, zorder=3)
ax.plot(t, lng,    color=C_LONG,   lw=2.4, zorder=2)

# =============================================================================
# ANNOTATIONS — all in data coordinates, inside the extended y-axis space
# No arrows; text staggered so nothing overlaps
# =============================================================================

# Rapid-acting — directly above its narrow peak (x ≈ 0.75)
ax.text(0.12, 1.72,
        "starts to work\n5–10 minutes\n\npeaks\n30–90 minutes",
        fontsize=FS_ANNOT, color=C_RAPID, style="italic",
        va="top", ha="left")

# Regular/short-acting — right of rapid, slightly lower row
ax.text(1.5, 1.42,
        "starts to work\n30 mins – 1 hour\n\npeaks\n2–5 hours",
        fontsize=FS_ANNOT, color=C_SHORT, style="italic",
        va="top", ha="left")

# Intermediate-acting — above its broad peak (x ≈ 7.5), top row
ax.text(4.8, 1.72,
        "starts to work\n1–2 hours\n\npeaks\n4–12 hours",
        fontsize=FS_ANNOT, color=C_INTER, style="italic",
        va="top", ha="left")

# Long-acting — above the flat plateau, right side
ax.text(15.5, 1.10,
        "starts to work\n1–2 hours\n\ndoes not peak",
        fontsize=FS_ANNOT, color=C_LONG, style="italic",
        va="top", ha="left")

# =============================================================================
# LEGEND (top-right, circles + labels)
# =============================================================================
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_RAPID,
           markersize=9, label="rapid-acting"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_SHORT,
           markersize=9, label="regular-/short-acting"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_INTER,
           markersize=9, label="intermediate-acting"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_LONG,
           markersize=9, label="long-acting"),
]
ax.legend(
    handles=legend_elements,
    fontsize=FS_LEGEND,
    loc="upper right",
    bbox_to_anchor=(0.99, 0.99),
    frameon=True,
    framealpha=0.95,
    edgecolor="#DDDDDD",
    handletextpad=0.6,
    labelspacing=0.55,
)

# =============================================================================
# AXES & TITLE
# =============================================================================
ax.set_xlim(0, 24)
ax.set_ylim(0, Y_TOP)

ax.set_xticks(range(0, 25, 2))
ax.set_xticklabels([str(v) for v in range(0, 25, 2)], fontsize=FS_TICK)

ax.set_yticks([])   # intensity axis — no numeric ticks

ax.set_xlabel("duration (hours)", fontsize=FS_LABEL + 1, labelpad=8,
              fontweight="bold")
ax.set_ylabel("intensity", fontsize=FS_LABEL + 1, labelpad=10,
              fontweight="bold")

ax.set_title("types of insulin", fontsize=FS_TITLE + 1,
             fontweight="bold", loc="left", pad=10)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)
ax.tick_params(axis="x", length=4, width=0.9)

# =============================================================================
# SAVE
# =============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
stem = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
for ext in ("pdf", "svg", "png"):
    fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=300,
                facecolor="white")
    print(f"  Saved → {OUTPUT_NAME}.{ext}")

plt.close(fig)
print(f"\n  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    pass
