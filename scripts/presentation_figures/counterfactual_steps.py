"""
scripts/presentation_figures/counterfactual_steps.py
====================================================

Four standalone 16:9 figures, one per step of the counterfactual
construction, for a single SLH episode of a chosen patient.

  Figure 1  step1_original    — Original CGM signal  G(t)
  Figure 2  step2_anchors     — Zone expansion  +  local-minima anchors
  Figure 3  step3_lininterp   — Linear interpolation  L(t)
  Figure 4  step4_counterfact — Final counterfactual  C(t)  +  Δ annotation

All four figures share the same time window, y-axis range, and background
elements (night shading, episode fill, H threshold) so they can be shown
as successive slides with a clean visual progression.

Usage
-----
    python3 -m scripts.presentation_figures.counterfactual_steps [--patient_id ID]
                                                  [--episode_idx 0]
                                                  [--offset_min 45]
                                                  [--window_h 3]
                                                  [--out_dir DIR]

Output
------
    scripts/presentation_figures/output/counterfactual_steps/  (git-ignored; --out_dir to change)
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
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.patient_selection import resolve_patient_id
from src.preprocessing.load_data import load_patient_window, prepare_valid_data
from src.preprocessing.timestep import detect_dominant_sampling_interval
from src.episodes.detection import detect_hyperglycemia_episodes
from src.episodes.criteria import classify_episodes
from src.episodes.episode_config import H_HYPER, NIGHT_START_H, NIGHT_END_H, LABEL_SLH
from src.config import TIME_COL, GLUCOSE_COL
from src.counterfactual.build_counterfactual import (
    build_linear_interpolation,
    build_counterfactual,
)
from src.counterfactual.counterfactual_config import (
    APPLY_KERNEL, KERNEL_MIN, SELECTED_OFFSET_MIN,
)
from src.visualization.style import apply_publication_style, save_figure
from src.visualization.colors import (
    COLOR_ORIGINAL,       # "#333333"  dark grey  — G(t)
    COLOR_THRESHOLD,      # "#B5121B"  red        — H line
    COLOR_ZONE,           # "#56B4E9"  sky blue   — shoulder zones
    COLOR_LININTERP,      # "#E69F00"  orange     — L(t)
    COLOR_COUNTERFACTUAL, # "#0072B2"  blue       — C(t)
    COLOR_ANCHOR,         # "#D55E00"  vermilion  — anchor markers
    COLOR_EPISODE,        # "#E69F00"  orange     — episode fill
    COLOR_NIGHT,          # "#DBD0D0"  pale grey  — night bands
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PATIENT_ID  = None   # None = automatic: patient with the most SLH episodes
DEFAULT_EPISODE_IDX = 0            # which eligible episode to illustrate
DEFAULT_WINDOW_H    = 3            # hours shown on each side of episode
DEFAULT_OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output", "counterfactual_steps")

# ---------------------------------------------------------------------------
# Figure geometry — 16:9
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 12.0, 5.5

# Presentation font sizes
FS_LABEL  = 14
FS_TICK   = 12
FS_LEGEND = 11
FS_ANNOT  = 11
FS_TITLE  = 15

# Alpha for the dimmed G(t) trace in steps 2-4
ALPHA_G_STEP = {1: 0.92, 2: 0.50, 3: 0.28, 4: 0.18}


# =============================================================================
# DATA HELPERS
# =============================================================================

def _enrich_slh_episodes(df: pd.DataFrame,
                         classification_df: pd.DataFrame,
                         raw_episodes: list) -> list[dict]:
    """Return SLH episodes enriched with peak_glucose / peak_time."""
    slh_rows = classification_df[classification_df["classification"] == LABEL_SLH]
    raw_lut  = {ep["onset_time"]: ep for ep in raw_episodes}
    result   = []
    for _, row in slh_rows.iterrows():
        raw_ep = raw_lut.get(row["onset_time"], {})
        idx    = raw_ep.get("idx", [])
        ep_df  = df.loc[idx] if idx else df[
            (df[TIME_COL] >= row["onset_time"]) &
            (df[TIME_COL] <= row["offset_time"])
        ]
        valid  = ep_df[GLUCOSE_COL].dropna()
        if valid.empty:
            continue
        pk_idx = valid.idxmax()
        result.append({
            "onset_time":    row["onset_time"],
            "offset_time":   row["offset_time"],
            "duration_min":  row["duration_min"],
            "peak_glucose":  float(df.loc[pk_idx, GLUCOSE_COL]),
            "peak_time":     df.loc[pk_idx, TIME_COL],
        })
    return result


# =============================================================================
# SHARED AXIS UTILITIES
# =============================================================================

def _night_shade(ax, t_start, t_end) -> None:
    """Add night bands (absolute timestamp x-axis)."""
    from datetime import datetime, time as _dtime
    cur = t_start.normalize()
    end = t_end.normalize() + pd.Timedelta(days=1)
    while cur <= end:
        for h0, h1 in [(0, NIGHT_END_H), (NIGHT_START_H, 24)]:
            ns = pd.Timestamp(datetime.combine(cur.date(), _dtime(h0 % 24, 0)))
            ne = pd.Timestamp(datetime.combine(cur.date(), _dtime(0, 0))) + \
                 pd.Timedelta(hours=h1)
            if ne > t_start and ns < t_end:
                ax.axvspan(max(ns, t_start), min(ne, t_end),
                           color=COLOR_NIGHT, alpha=0.40, zorder=0, linewidth=0)
        cur += pd.Timedelta(days=1)


def _base_ax(ax, t_start, t_end, t_onset, t_offset,
             delta_td, y_min, y_max, show_zones: bool) -> None:
    """
    Draw the common background elements onto *ax*:
      night shading, episode fill, optional shoulder zones,
      H threshold, onset/offset markers, axis limits and ticks.
    """
    # Night shading
    _night_shade(ax, t_start, t_end)

    # Shoulder zones (step 2-4)
    if show_zones:
        for span_s, span_e in [
            (t_onset - delta_td, t_onset),
            (t_offset, t_offset + delta_td),
        ]:
            ax.axvspan(
                max(span_s, t_start), min(span_e, t_end),
                color=COLOR_ZONE, alpha=0.18, zorder=1, linewidth=0,
            )

    # Episode core fill
    ax.axvspan(
        max(t_onset, t_start), min(t_offset, t_end),
        color=COLOR_EPISODE, alpha=0.22, zorder=1, linewidth=0,
    )

    # H = 180 threshold
    ax.axhline(H_HYPER, color=COLOR_THRESHOLD, ls="--", lw=1.8,
               alpha=0.85, zorder=3)
    ax.annotate(
        f"H = {H_HYPER} mg/dL",
        xy=(t_end, H_HYPER), xytext=(5, 0),
        textcoords="offset points",
        fontsize=FS_ANNOT, color=COLOR_THRESHOLD,
        fontweight="bold", va="center", annotation_clip=False,
    )

    # Onset / offset markers
    for tm in (t_onset, t_offset):
        ax.axvline(tm, color="#777777", lw=1.0, ls="--", alpha=0.45, zorder=3)

    # Axes limits & ticks
    ax.set_xlim(t_start, t_end)
    ax.set_ylim(y_min, y_max)

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[30]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=FS_TICK)

    yticks = sorted({int(y_min) + 10, H_HYPER, 250, int(y_max) - 10})
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(v) for v in yticks], fontsize=FS_TICK)

    ax.set_xlabel("Time of day", fontsize=FS_LABEL, labelpad=5)
    ax.set_ylabel("Glucose (mg/dL)", fontsize=FS_LABEL, labelpad=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, ls=":", lw=0.4, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)


def _new_fig():
    """Create a blank 16:9 figure with one axes."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=0.08, right=0.88, top=0.88, bottom=0.13)
    return fig, ax


def _add_title(ax, text: str) -> None:
    ax.set_title(text, fontsize=FS_TITLE, fontweight="bold",
                 loc="left", pad=8, color="#222222")


# =============================================================================
# STEP 1 — Original signal G(t)
# =============================================================================

def draw_step1(ax, t_window, g_window, episode,
               t_start, t_end, delta_td, y_min, y_max) -> None:

    _base_ax(ax, t_start, t_end,
             episode["onset_time"], episode["offset_time"],
             delta_td, y_min, y_max, show_zones=False)

    ax.plot(t_window, g_window, color=COLOR_ORIGINAL,
            lw=2.0, zorder=4, label="G(t)")

    # Peak marker
    ax.scatter([episode["peak_time"]], [episode["peak_glucose"]],
               color=COLOR_THRESHOLD, s=80, marker="^", zorder=8, label="_")
    y_range = y_max - y_min
    ax.annotate(
        f"Peak  {episode['peak_glucose']:.0f} mg/dL\n"
        f"Duration  {episode['duration_min']:.0f} min",
        xy=(episode["peak_time"], episode["peak_glucose"]),
        xytext=(episode["peak_time"] + pd.Timedelta(minutes=22),
                episode["peak_glucose"] + y_range * 0.08),
        fontsize=FS_ANNOT, color=COLOR_THRESHOLD, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLOR_THRESHOLD, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                  ec=COLOR_THRESHOLD, lw=0.8, alpha=0.92),
        zorder=9,
    )

    ax.legend(
        handles=[
            Line2D([0], [0], color=COLOR_ORIGINAL, lw=2.0, label="G(t) — original signal"),
            Line2D([0], [0], color=COLOR_THRESHOLD, lw=1.8, ls="--",
                   label=f"H = {H_HYPER} mg/dL"),
            mpatches.Patch(facecolor=COLOR_EPISODE, alpha=0.30,
                           label="SLH episode"),
        ],
        loc="upper left", fontsize=FS_LEGEND, framealpha=0.92,
        handlelength=1.8,
    )
    _add_title(ax, "Step 1 — Original CGM signal  G(t)")


# =============================================================================
# STEP 2 — Zone expansion + anchors
# =============================================================================

def draw_step2(ax, t_window, g_window, episode, anchor_info,
               t_start, t_end, delta_td, offset_min, y_min, y_max) -> None:

    _base_ax(ax, t_start, t_end,
             episode["onset_time"], episode["offset_time"],
             delta_td, y_min, y_max, show_zones=True)

    ax.plot(t_window, g_window, color=COLOR_ORIGINAL,
            lw=1.4, alpha=ALPHA_G_STEP[2], zorder=4)

    y_range = y_max - y_min

    if anchor_info:
        ax.scatter(
            [anchor_info["t_a"], anchor_info["t_b"]],
            [anchor_info["g_a"], anchor_info["g_b"]],
            color=COLOR_ANCHOR, s=120, marker="v",
            zorder=9, label="Local-min anchor",
        )
        for t_anc, g_anc, tag in [
            (anchor_info["t_a"], anchor_info["g_a"], "Lₐ"),
            (anchor_info["t_b"], anchor_info["g_b"], "Lₔ"),
        ]:
            ax.annotate(
                f"{tag} = {g_anc:.0f} mg/dL",
                xy=(t_anc, g_anc),
                xytext=(t_anc + pd.Timedelta(minutes=18),
                        g_anc + y_range * 0.13),
                fontsize=FS_ANNOT, color=COLOR_ANCHOR, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLOR_ANCHOR, lw=1.0),
                zorder=10,
            )

    ax.legend(
        handles=[
            Line2D([0], [0], color=COLOR_ORIGINAL, lw=1.4,
                   alpha=0.6, label="G(t) — original signal"),
            mpatches.Patch(facecolor=COLOR_ZONE, alpha=0.35,
                           label=f"Shoulder zone  (δ = {offset_min} min)"),
            Line2D([0], [0], color=COLOR_ANCHOR, ls="none",
                   marker="v", markersize=10, label="Local-min anchor"),
        ],
        loc="upper left", fontsize=FS_LEGEND, framealpha=0.92,
        handlelength=1.8,
    )
    _add_title(ax, f"Step 2 — Zone expansion  (δ = {offset_min} min)  +  local-minima anchors")


# =============================================================================
# STEP 3 — Linear interpolation L(t)
# =============================================================================

def draw_step3(ax, t_window, g_window, l_window, episode, anchor_info,
               t_start, t_end, delta_td, y_min, y_max) -> None:

    _base_ax(ax, t_start, t_end,
             episode["onset_time"], episode["offset_time"],
             delta_td, y_min, y_max, show_zones=True)

    # G(t) very faded
    ax.plot(t_window, g_window, color=COLOR_ORIGINAL,
            lw=1.2, alpha=ALPHA_G_STEP[3], zorder=4)

    # L(t) — dashed orange
    ax.plot(t_window, l_window, color=COLOR_LININTERP,
            lw=2.4, ls="--", zorder=6, label="L(t)")

    if anchor_info:
        ax.scatter(
            [anchor_info["t_a"], anchor_info["t_b"]],
            [anchor_info["g_a"], anchor_info["g_b"]],
            color=COLOR_ANCHOR, s=100, marker="v", zorder=9,
        )

    ax.legend(
        handles=[
            Line2D([0], [0], color=COLOR_LININTERP, lw=2.4, ls="--",
                   label="L(t) — linear interpolation"),
            Line2D([0], [0], color=COLOR_ORIGINAL, lw=1.2,
                   alpha=0.45, label="G(t) — original signal"),
            Line2D([0], [0], color=COLOR_ANCHOR, ls="none",
                   marker="v", markersize=9, label="Anchor points"),
        ],
        loc="upper left", fontsize=FS_LEGEND, framealpha=0.92,
        handlelength=1.8,
    )
    _add_title(ax, "Step 3 — Linear interpolation baseline  L(t)")


# =============================================================================
# STEP 4 — Final counterfactual C(t) + Δ annotation
# =============================================================================

def draw_step4(ax, t_window, g_window, l_window, c_window,
               episode, anchor_info,
               t_start, t_end, delta_td, y_min, y_max) -> None:

    _base_ax(ax, t_start, t_end,
             episode["onset_time"], episode["offset_time"],
             delta_td, y_min, y_max, show_zones=True)

    y_range = y_max - y_min

    # G(t) — barely visible
    ax.plot(t_window, g_window, color=COLOR_ORIGINAL,
            lw=1.0, alpha=ALPHA_G_STEP[4], zorder=4)

    # L(t) — faint dashed
    ax.plot(t_window, l_window, color=COLOR_LININTERP,
            lw=1.2, ls="--", alpha=0.40, zorder=5)

    # C(t) — bold blue
    ax.plot(t_window, c_window, color=COLOR_COUNTERFACTUAL,
            lw=2.6, zorder=7, label="C(t)")

    # ── Δ annotation at peak ──────────────────────────────────────────────────
    ts_float = t_window.values.astype("datetime64[s]").astype(np.float64)
    c_at_peak = float(np.interp(
        np.float64(episode["peak_time"].timestamp()),
        ts_float, c_window,
    ))
    delta_g = episode["peak_glucose"] - c_at_peak

    # Vertical double-headed arrow between G(peak) and C(peak)
    x_arrow = episode["peak_time"] + pd.Timedelta(minutes=28)
    ax.annotate(
        "",
        xy=(x_arrow, c_at_peak),
        xytext=(x_arrow, episode["peak_glucose"]),
        arrowprops=dict(
            arrowstyle="<->",
            color="#444444", lw=1.4,
            shrinkA=0, shrinkB=0,
        ),
        zorder=10,
    )
    ax.text(
        x_arrow + pd.Timedelta(minutes=4),
        (episode["peak_glucose"] + c_at_peak) / 2,
        f"ΔG = {delta_g:.0f} mg/dL",
        fontsize=FS_ANNOT, color="#222222", va="center",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc="white",
                  ec="#888888", lw=0.7, alpha=0.92),
        zorder=11,
    )

    # G and C labels at peak
    ax.scatter([episode["peak_time"]], [episode["peak_glucose"]],
               color=COLOR_ORIGINAL, s=60, marker="o", zorder=9)
    ax.scatter([episode["peak_time"]], [c_at_peak],
               color=COLOR_COUNTERFACTUAL, s=60, marker="o", zorder=9)

    kernel_lbl = (f" + smoothing  (k = {KERNEL_MIN} min)" if APPLY_KERNEL else "")
    ax.legend(
        handles=[
            Line2D([0], [0], color=COLOR_COUNTERFACTUAL, lw=2.6,
                   label="C(t) — counterfactual" + kernel_lbl),
            Line2D([0], [0], color=COLOR_LININTERP, lw=1.6, ls="--",
                   alpha=0.55, label="L(t) — linear interpolation"),
            Line2D([0], [0], color=COLOR_ORIGINAL, lw=1.2,
                   alpha=0.35, label="G(t) — original signal"),
        ],
        loc="upper left", fontsize=FS_LEGEND, framealpha=0.92,
        handlelength=1.8,
    )
    title = "Step 4 — Final counterfactual  C(t)"
    if APPLY_KERNEL:
        title += f"  [rolling mean  k = {KERNEL_MIN} min]"
    _add_title(ax, title)


# =============================================================================
# PIPELINE
# =============================================================================

def generate(patient_id: str, episode_idx: int, offset_min: float,
             window_h: float, output_dir: str) -> None:
    """Full pipeline: load → detect → select episode → draw 4 figures → save."""

    # ── Load & classify ───────────────────────────────────────────────────────
    raw = load_patient_window(patient_id)
    if raw is None:
        print(f"[!] No data for {patient_id}")
        return
    df = prepare_valid_data(raw)
    print(f"  Loaded {len(df):,} valid points for {patient_id}")

    timestep = detect_dominant_sampling_interval(df)
    raw_eps   = detect_hyperglycemia_episodes(df)
    class_df  = classify_episodes(df, raw_eps)
    slh_eps   = _enrich_slh_episodes(df, class_df, raw_eps)
    print(f"  SLH episodes : {len(slh_eps)}")

    # ── Select episode ────────────────────────────────────────────────────────
    win_td   = pd.Timedelta(hours=window_h)
    zone_td  = pd.Timedelta(minutes=offset_min)
    eligible = [
        ep for ep in slh_eps
        if (ep["onset_time"]  - win_td  >= df[TIME_COL].iloc[0] and
            ep["offset_time"] + win_td  <= df[TIME_COL].iloc[-1] and
            ep["onset_time"]  - zone_td >= df[TIME_COL].iloc[0] and
            ep["offset_time"] + zone_td <= df[TIME_COL].iloc[-1])
    ]
    if not eligible:
        print("[!] No eligible SLH episodes with sufficient margins.")
        return

    # Sort by peak glucose descending for the best visual default
    eligible.sort(key=lambda e: e["peak_glucose"], reverse=True)
    idx     = min(episode_idx, len(eligible) - 1)
    episode = eligible[idx]
    print(f"  Episode {idx}  |  "
          f"onset {episode['onset_time'].strftime('%d %b %H:%M')}  |  "
          f"{episode['duration_min']:.0f} min  |  "
          f"peak {episode['peak_glucose']:.0f} mg/dL")

    # ── Build signals ─────────────────────────────────────────────────────────
    signal_lin, anchor_info = build_linear_interpolation(
        df, episode, offset_min, timestep)
    signal_ctf, _           = build_counterfactual(
        df, episode, offset_min, timestep)

    t_start = episode["onset_time"]  - win_td
    t_end   = episode["offset_time"] + win_td
    mask    = (df[TIME_COL] >= t_start) & (df[TIME_COL] <= t_end)

    t_window = df.loc[mask, TIME_COL]
    g_window = df.loc[mask, GLUCOSE_COL].astype(float).values
    l_window = signal_lin[mask.values]
    c_window = signal_ctf[mask.values]

    delta_td = pd.Timedelta(minutes=offset_min)
    y_min    = max(50,  float(np.nanmin(g_window)) - 20)
    y_max    = min(420, float(np.nanmax(g_window)) + 30)

    # ── Apply style ───────────────────────────────────────────────────────────
    apply_publication_style()
    matplotlib.rcParams.update({
        "font.size":        FS_TICK,
        "axes.labelsize":   FS_LABEL,
        "xtick.labelsize":  FS_TICK,
        "ytick.labelsize":  FS_TICK,
        "legend.fontsize":  FS_LEGEND,
    })

    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, f"{patient_id}_ep{idx:02d}")

    # ── Figure 1 ──────────────────────────────────────────────────────────────
    fig, ax = _new_fig()
    draw_step1(ax, t_window, g_window, episode,
               t_start, t_end, delta_td, y_min, y_max)
    save_figure(fig, f"{prefix}_step1_original")
    print(f"  Saved step 1")

    # ── Figure 2 ──────────────────────────────────────────────────────────────
    fig, ax = _new_fig()
    draw_step2(ax, t_window, g_window, episode, anchor_info,
               t_start, t_end, delta_td, offset_min, y_min, y_max)
    save_figure(fig, f"{prefix}_step2_anchors")
    print(f"  Saved step 2")

    # ── Figure 3 ──────────────────────────────────────────────────────────────
    fig, ax = _new_fig()
    draw_step3(ax, t_window, g_window, l_window, episode, anchor_info,
               t_start, t_end, delta_td, y_min, y_max)
    save_figure(fig, f"{prefix}_step3_lininterp")
    print(f"  Saved step 3")

    # ── Figure 4 ──────────────────────────────────────────────────────────────
    fig, ax = _new_fig()
    draw_step4(ax, t_window, g_window, l_window, c_window,
               episode, anchor_info,
               t_start, t_end, delta_td, y_min, y_max)
    save_figure(fig, f"{prefix}_step4_counterfactual")
    print(f"  Saved step 4")


# =============================================================================
# CLI
# =============================================================================

def parse_arguments():
    p = argparse.ArgumentParser(
        description="Counterfactual — 4 step-by-step presentation figures (16:9)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--patient_id",   default=DEFAULT_PATIENT_ID)
    p.add_argument("--episode_idx",  type=int, default=DEFAULT_EPISODE_IDX,
                   help="Index into eligible SLH episodes sorted by peak glucose (0 = highest)")
    p.add_argument("--offset_min",   type=float, default=SELECTED_OFFSET_MIN,
                   help="Zone-expansion δ in minutes")
    p.add_argument("--window_h",     type=float, default=DEFAULT_WINDOW_H,
                   help="Hours of signal shown on each side of the episode")
    p.add_argument("--out_dir",      default=DEFAULT_OUTPUT_DIR)
    return p


def main():
    args = parse_arguments().parse_args()
    args.patient_id = resolve_patient_id(args.patient_id)

    print("\n" + "=" * 60)
    print("  Counterfactual — Step-by-Step Presentation Figures")
    print("=" * 60)
    print(f"  Patient      : {args.patient_id}")
    print(f"  Episode idx  : {args.episode_idx}  (0 = highest peak)")
    print(f"  δ (offset)   : {args.offset_min} min")
    print(f"  Window       : ±{args.window_h} h")
    print(f"  Kernel       : {'k=' + str(KERNEL_MIN) + ' min' if APPLY_KERNEL else 'off'}")
    print(f"  Output       : {args.out_dir}")
    print("=" * 60 + "\n")

    generate(args.patient_id, args.episode_idx, args.offset_min,
             args.window_h, args.out_dir)

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
