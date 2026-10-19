"""
scripts/paper_figures/generate_figure_4_counterfactual_construction.py
======================================================================
Figure 4: Counterfactual Signal Construction and Sensitivity Analysis
====================================================================
Renumbered from Figure 3 (bis) now that Figure 3 is the high-vs-low-SLH patient
comparison (scripts/patient_comparison.py).

Multi-panel figure combining:
- Row 1: Single episode with different offset values (full width)
- Row 2: 7-day overview with counterfactual (single patient, full width)
- Row 3: AGP median comparison (2 patients, patient ID in title), with the
  time-of-day slot of largest |Delta median(t)| between G(t) and C(t) marked
  visually (vertical bracket + annotation) on each patient's panel
- Row 4: TIR and Median sensitivity to offset (E: ΔTIR, F: ΔMedian; one line per
  participant, δ=30 min offset highlighted; the participant legend is shown on
  panel F only)

Exports sensitivity delta values to CSV for both patients.

Usage:
-----
# Automatic pair (most / fewest SLH episodes, or CGM_PATIENT_HIGH / CGM_PATIENT_LOW)
python3 -m scripts.paper_figures.generate_figure_4_counterfactual_construction

# Explicit pair: --high_id = high-SLH patient (panels B, C), --low_id = low-SLH patient (panels A, D)
python3 -m scripts.paper_figures.generate_figure_4_counterfactual_construction \\
    --high_id <ID_HIGH> --low_id <ID_LOW>

# Same figure in mmol/L (file name gets the _mmoll suffix; default is mg/dL)
CGM_GLUCOSE_UNIT="mmol/L" python3 -m scripts.paper_figures.generate_figure_4_counterfactual_construction \\
    --high_id <ID_HIGH> --low_id <ID_LOW>

Authors: Guillaume Deside

Output
------
    <CGM_DATA_ROOT>/Figures/Manuscript_Figures/Figure4_Counterfactual_Construction_<high>_vs_<low>.{pdf,svg}
    and the underlying delta tables in Figures/Manuscript_Figures/<high>_vs_<low>/*.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from datetime import datetime, time as _dtime

# Import existing functions
from src.utils.patient_selection import resolve_patient_pair
from src.preprocessing.load_data import load_patient_window
from src.preprocessing.timestep import detect_dominant_sampling_interval
from src.episodes.detection import detect_hyperglycemia_episodes
from src.episodes.criteria import classify_episodes
from src.episodes.episode_config import (
    H_HYPER, NIGHT_START_H, NIGHT_END_H, GAP_FACTOR,
    LABEL_SLH,
)
from src.config import TIME_COL, GLUCOSE_COL, FIG_DIR, COL_FLAG_TIME_ERROR, COL_FLAG_OUTLIER_GLUCOSE
from src.counterfactual.build_counterfactual import build_full_counterfactual_signal
from src.agp.compute_agp import compute_agp_profile
from src.visualization.style import apply_publication_style, save_figure_with_panels, style_axes
from src.visualization.colors import OFFSET_COLORS, COLOR_NIGHT, CATEGORICAL_5, COLOR_ZERO_LINE
from src.visualization.glucose_axis import apply_glucose_axis
from src.utils.unit_conversion import format_glucose, glucose_unit_label

# =============================================================================
# CONFIGURATION
# =============================================================================
PATIENT_IDS = list(resolve_patient_pair())  # [high-SLH, low-SLH] patients (CGM_PATIENT_HIGH / CGM_PATIENT_LOW, else automatic)
SELECTED_OFFSET_MIN = 30        # Main offset value for 7-day and AGP panels; used in main analysis
OFFSET_VALUES_COMPARE = [0, 15, 30, 45, 60, 75, 90]  # Offsets for episode comparison
APPLY_KERNEL = True
KERNEL_WIDTH_MIN = 60
PLOT_WINDOW_H = 3               # Hours around episode
DAYS_PER_ROW = 7                # Days in overview

OUTPUT_DIR = os.path.join(FIG_DIR, "Manuscript_Figures")

# Per-participant styling for sensitivity panels (row 4): legend separates
# participants, not metrics — each metric now has its own panel.
PATIENT_STYLES = {
    PATIENT_IDS[0]: {"color": CATEGORICAL_5[0], "marker": "o", "ls": "-"},
    PATIENT_IDS[1]: {"color": CATEGORICAL_5[1], "marker": "s", "ls": "--"},
}

# Colors
COLOR_ORIGINAL = "#333333"
COLOR_THRESHOLD = "#D55E00"
COLOR_EPISODE = "#E69F00"
COLOR_ZONE = "#56B4E9"
COLOR_HYPO = "#E69F00"
COLOR_TARGET_ZONE = "#009E73"
COLOR_MAX_DIFF = "#B5121B"   # max |Delta median| marker on the AGP panels (C/D) -- kept
                             # distinct from COLOR_THRESHOLD (used for the H_HYPER line)

# AGP parameters
AGP_Y_MIN, AGP_Y_MAX = 50, 350
H_HYPO = 70

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def prepare_valid_data(df):
    mask = pd.Series(True, index=df.index)
    if COL_FLAG_TIME_ERROR in df.columns:
        mask &= ~df[COL_FLAG_TIME_ERROR].fillna(False).astype(bool)
    if COL_FLAG_OUTLIER_GLUCOSE in df.columns:
        mask &= ~df[COL_FLAG_OUTLIER_GLUCOSE].fillna(False).astype(bool)
    return df[mask][[TIME_COL, GLUCOSE_COL]].dropna().sort_values(TIME_COL).reset_index(drop=True)

def shade_nights_on_axis(ax, t_start, t_end, labeled_list):
    from datetime import timedelta
    current_date = t_start.date()
    end_date = t_end.date() + timedelta(days=1)
    while current_date <= end_date:
        for hour_start, hour_end in [(0, NIGHT_END_H), (NIGHT_START_H, 24)]:
            night_start = pd.Timestamp(datetime.combine(current_date, _dtime(hour_start % 24, 0)))
            night_end = pd.Timestamp(datetime.combine(current_date, _dtime(0, 0))) + pd.Timedelta(hours=hour_end)
            if night_end > t_start and night_start < t_end:
                label = f"Night ({NIGHT_START_H:02d}–{NIGHT_END_H:02d}h)" if not labeled_list else "_"
                ax.axvspan(max(night_start, t_start), min(night_end, t_end),
                          color=COLOR_NIGHT, alpha=0.35, zorder=0, linewidth=0, label=label)
                labeled_list.append(True)
        current_date += timedelta(days=1)

def is_night(timestamp):
    return timestamp.hour >= NIGHT_START_H or timestamp.hour < NIGHT_END_H

def compute_metrics(glucose, time_series=None, daytime_only=False):
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

def compute_max_median_diff(agp_original, agp_ctf):
    """
    Time-slot-wise Delta-median = median_original(t) - median_ctf(t), and the
    slot where |Delta| is largest -- same formula used in
    scripts/counterfactual/sensitivity_analysis.py, so the panel C/D
    annotation matches the cohort-level metric exactly.

    Returns None if no time slot has valid data in both profiles.
    """
    diff = agp_original["median"].to_numpy() - agp_ctf["median"].to_numpy()
    valid = ~np.isnan(diff)
    if not valid.any():
        return None

    abs_diff = np.where(valid, np.abs(diff), -np.inf)
    slot_idx = int(np.argmax(abs_diff))

    return {
        "slot_idx": slot_idx,
        "max_abs_diff": float(abs(diff[slot_idx])),
        "time_of_max": agp_original["time_str"].iloc[slot_idx],
        "median_original_at_max": float(agp_original["median"].iloc[slot_idx]),
        "median_ctf_at_max": float(agp_ctf["median"].iloc[slot_idx]),
    }

def add_panel_label(ax, label, x=-0.12, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="right")

def add_panel_title(ax, title, fontsize=7.5, pad=6):
    """Short descriptive subtitle placed above a panel, next to its letter label."""
    ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=pad)

def compute_sensitivity_records(patient_id, df, slh_episodes, timestep_min, offsets=OFFSET_VALUES_COMPARE):
    """Delta TIR/median (daytime) vs. offset for one patient, computed once and reused
    across the y-limit pass, the row-4 plots, and the CSV export."""
    orig_glucose = df[GLUCOSE_COL].values
    time_series = df[TIME_COL]
    orig_dt = compute_metrics(orig_glucose, time_series, daytime_only=True)

    records = []
    for offset in offsets:
        c_signal = build_full_counterfactual_signal(df, slh_episodes, offset, timestep_min)
        ctf_dt = compute_metrics(c_signal, time_series, daytime_only=True)
        records.append({
            "patient_id": patient_id,
            "offset_min": offset,
            "delta_TIR_70_180_pct": round(ctf_dt["TIR70-180"] - orig_dt["TIR70-180"], 4),
            "delta_median_mg_dL": round(ctf_dt["median"] - orig_dt["median"], 2),
            "original_TIR_70_180_pct": round(orig_dt["TIR70-180"], 4),
            "original_median_mg_dL": round(orig_dt["median"], 2),
            "counterfactual_TIR_70_180_pct": round(ctf_dt["TIR70-180"], 4),
            "counterfactual_median_mg_dL": round(ctf_dt["median"], 2),
        })
    return records

class RelativeDayFormatter(mdates.DateFormatter):
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

def setup_relative_time_axis(ax, t_start, t_end, days_in_row):
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(RelativeDayFormatter(t_start))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=6.5, rotation=0, ha="center")

# =============================================================================
# PANEL A: EPISODE COMPARISON
# =============================================================================
def plot_episode_comparison(ax, patient_id, df, episode, offsets, timestep_min):
    # Define line styles for black and white readability
    LINE_STYLES = {
        0: '-',      # Solid
        15: '-',    # Dashed
        30: '-',    # Dash-dot
        45: '-',     # Dotted
        60: '-',  # Dash-dot-dot
        75: '-',
        90: '-',
    }
    
    window_delta = pd.Timedelta(hours=PLOT_WINDOW_H)
    t_start = episode["onset_time"] - window_delta
    t_end = episode["offset_time"] + window_delta
    
    mask = (df[TIME_COL] >= t_start) & (df[TIME_COL] <= t_end)
    t_window = df.loc[mask, TIME_COL]
    g_window = df.loc[mask, GLUCOSE_COL].astype(float).values
    
    labeled_list = []
    shade_nights_on_axis(ax, t_start, t_end, labeled_list)
    
    ax.axvspan(episode["onset_time"], episode["offset_time"],
              color=COLOR_EPISODE, alpha=0.22, zorder=1, linewidth=0)
    ax.axhline(H_HYPER, color=COLOR_THRESHOLD, ls="--", lw=0.8, alpha=0.65, zorder=2)
    ax.plot(t_window, g_window, color=COLOR_ORIGINAL, lw=1.0, alpha=0.9, zorder=6,
           label="G(t)")
    
    for offset in offsets:
        cfacts = build_full_counterfactual_signal(df, [episode], offset, timestep_min)[mask.values]
        color = OFFSET_COLORS.get(offset, "#333333")
        linestyle = LINE_STYLES.get(offset, '-')
        ax.plot(t_window, cfacts, color=color, lw=1.3, ls=linestyle, zorder=5,
               label=f"δ = {offset} min")
    
    y_min = max(50, float(np.nanmin(g_window)) - 15)
    y_max = min(420, float(np.nanmax(g_window)) + 25)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(t_start, t_end)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=6.5, rotation=30, ha="right")
    apply_glucose_axis(ax, fontsize=7, labelpad=2)
    style_axes(ax)
    ax.legend(fontsize=6, loc="lower right", framealpha=0.90, ncol=3)
    add_panel_label(ax, "A")

# =============================================================================
# PANEL B: 7-DAY OVERVIEW (SINGLE PATIENT, FULL WIDTH)
# =============================================================================
def plot_7day_overview(ax, patient_id, df, slh_episodes, offset_min, timestep_min, y_limits=None):
    t_start = df[TIME_COL].iloc[0]
    t_end = t_start + pd.Timedelta(days=DAYS_PER_ROW)
    
    mask = (df[TIME_COL] >= t_start) & (df[TIME_COL] < t_end)
    t_row = df.loc[mask, TIME_COL]
    g_row = df.loc[mask, GLUCOSE_COL].astype(float).copy()
    c_row = build_full_counterfactual_signal(df, slh_episodes, offset_min, timestep_min)[mask.values]
    
    diffs = t_row.diff().dt.total_seconds().fillna(0) / 60
    gap_mask = diffs > timestep_min * GAP_FACTOR
    g_row[gap_mask] = np.nan
    c_row[gap_mask] = np.nan
    
    labeled_list = []
    shade_nights_on_axis(ax, t_start, t_end, labeled_list)
    
    slh_labeled = False
    for ep in slh_episodes:
        if ep["offset_time"] < t_start or ep["onset_time"] > t_end:
            continue
        label = "SLH episode" if not slh_labeled else "_"
        ax.axvspan(max(ep["onset_time"], t_start), min(ep["offset_time"], t_end),
                  color=COLOR_EPISODE, alpha=0.28, zorder=2, linewidth=0, label=label)
        slh_labeled = True
    
    ax.axhline(H_HYPER, color=COLOR_THRESHOLD, ls="--", lw=0.8, alpha=0.65, zorder=3)
    
    # Original: dashed, thinner
    ax.plot(t_row, g_row, color=COLOR_ORIGINAL, lw=0.6, ls='--', alpha=0.60, 
            zorder=4, label="G(t)")
    
    # Counterfactual: solid, slightly thicker
    color_ctf = OFFSET_COLORS.get(offset_min, "#0072B2")
    ax.plot(t_row, c_row, color=color_ctf, lw=0.8, ls='-', alpha=0.85,
            zorder=5, label=f"C(t) δ={offset_min}")
    
    ax.set_xlim(t_start, t_end)
    
    # Use provided y-limits if available
    if y_limits is not None:
        ax.set_ylim(y_limits[0], y_limits[1])
    else:
        g_valid = g_row.dropna()
        y_min = max(40, float(g_valid.min()) - 12) if not g_valid.empty else 40
        y_max = min(450, float(g_valid.max()) + 18) if not g_valid.empty else 400
        ax.set_ylim(y_min, y_max)
    
    setup_relative_time_axis(ax, t_start, t_end, DAYS_PER_ROW)
    apply_glucose_axis(ax, fontsize=7, labelpad=2)
    
    style_axes(ax)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.90)

# =============================================================================
# PANELS C/D: AGP COMPARISON (WITH PATIENT ID IN TITLE)
# =============================================================================
def plot_agp_comparison(ax, patient_id, df, slh_episodes, offset_min, timestep_min, y_limits=None, show_legend=True):
    def get_agp_x_axis(agp):
        today = datetime.today().date()
        return [datetime.combine(today, datetime.strptime(t, "%H:%M").time())
                for t in agp["time_str"]]
    
    agp_original = compute_agp_profile(df, timestep_min)
    
    c_signal = build_full_counterfactual_signal(df, slh_episodes, offset_min, timestep_min)
    df_ctf = df.copy()
    df_ctf[GLUCOSE_COL] = c_signal
    agp_ctf = compute_agp_profile(df_ctf, timestep_min)
    
    today = datetime.today().date()
    ax.set_xlim(datetime.combine(today, _dtime(0, 0)),
                datetime.combine(today, _dtime(23, 59)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=6.5, rotation=0)
    
    # Use provided y-limits if available
    if y_limits is not None:
        ax.set_ylim(y_limits[0], y_limits[1])
    else:
        ax.set_ylim(AGP_Y_MIN, AGP_Y_MAX)
    
    ax.axhline(H_HYPER, color=COLOR_THRESHOLD, ls="--", lw=0.9, alpha=0.75, zorder=2)
    ax.axhline(H_HYPO, color=COLOR_HYPO, ls="--", lw=0.9, alpha=0.65, zorder=2)
    ax.axhspan(H_HYPO, H_HYPER, alpha=0.03, color=COLOR_TARGET_ZONE, zorder=0)
    
    x_orig = get_agp_x_axis(agp_original)
    x_ctf = get_agp_x_axis(agp_ctf)
    
    # Plot IQR bands with dotted boundaries
    # Original: no hatch, lighter alpha
    ax.fill_between(x_orig, agp_original["q25"], agp_original["q75"],
                    alpha=0.12, color=COLOR_ORIGINAL, zorder=1, linewidth=0)
    
    # Original IQR boundaries with dotted lines
    ax.plot(x_orig, agp_original["q25"], color=COLOR_ORIGINAL, lw=0.8, 
           ls=':', alpha=0.5, zorder=3)
    ax.plot(x_orig, agp_original["q75"], color=COLOR_ORIGINAL, lw=0.8, 
           ls=':', alpha=0.5, zorder=3)
    
    # Counterfactual: diagonal hatch for b/w readability
    color_ctf = OFFSET_COLORS.get(offset_min, "#0072B2")
    ax.fill_between(x_ctf, agp_ctf["q25"], agp_ctf["q75"],
                    alpha=0.30, color=color_ctf, zorder=2, linewidth=0,
                    hatch='///', edgecolor=color_ctf)
    
    # Counterfactual IQR boundaries with dotted lines
    ax.plot(x_ctf, agp_ctf["q25"], color=color_ctf, lw=1.0, 
           ls=':', alpha=0.7, zorder=4)
    ax.plot(x_ctf, agp_ctf["q75"], color=color_ctf, lw=1.0, 
           ls=':', alpha=0.7, zorder=4)
    
    # Plot median lines with different styles
    # Original: dashed
    ax.plot(x_orig, agp_original["median"], color=COLOR_ORIGINAL, lw=1.5, ls='--',
           label="G(t)" if show_legend else None, zorder=5)
    
    # Counterfactual: solid
    ax.plot(x_ctf, agp_ctf["median"], color=color_ctf, lw=1.5, ls='-',
           label=f"C(t) δ={offset_min}" if show_legend else None, zorder=6)

    # Mark the time-of-day slot where the median curves differ the most.
    diff_info = compute_max_median_diff(agp_original, agp_ctf)
    if diff_info is not None:
        x_max = x_orig[diff_info["slot_idx"]]
        y_lo = min(diff_info["median_original_at_max"], diff_info["median_ctf_at_max"])
        y_hi = max(diff_info["median_original_at_max"], diff_info["median_ctf_at_max"])

        ax.axvline(x_max, color=COLOR_MAX_DIFF, lw=0.8, ls=":", alpha=0.6, zorder=7)
        ax.plot([x_max, x_max], [y_lo, y_hi], color=COLOR_MAX_DIFF, lw=2.0,
               alpha=0.85, solid_capstyle="butt", zorder=8)
        ax.scatter([x_max, x_max], [y_lo, y_hi], color=COLOR_MAX_DIFF, s=16,
                  zorder=9, edgecolor="white", linewidth=0.4)
        ax.annotate(
            f"max |Δ| = {format_glucose(diff_info['max_abs_diff'])} "
            f"@ {diff_info['time_of_max']}",
            xy=(x_max, y_hi), xycoords="data",
            xytext=(0.03, 0.04), textcoords="axes fraction",
            fontsize=6, color=COLOR_MAX_DIFF, ha="left", va="bottom",
            arrowprops=dict(arrowstyle="-", color=COLOR_MAX_DIFF, lw=0.6,
                            alpha=0.55, shrinkA=0, shrinkB=3),
            zorder=10,
        )

    apply_glucose_axis(ax, fontsize=7, labelpad=2)
    ax.set_xlabel("Time of day", fontsize=7, labelpad=2)
    
    # Set title with patient ID + panel subtitle
    add_panel_title(ax, f"Participant {patient_id} — AGP: observed vs. counterfactual")
    
    style_axes(ax)
    
    if show_legend:
        ax.legend(fontsize=6, loc="upper right", framealpha=0.90)
    
    return agp_original, agp_ctf

# =============================================================================
# PANELS E/F: SENSITIVITY LINE PLOTS (E = TIR, F = MEDIAN; ONE LINE PER PARTICIPANT)
# =============================================================================
def plot_sensitivity_metric(ax, ylabel, series, offsets=OFFSET_VALUES_COMPARE,
                            y_limits=None, show_legend=False,
                            highlight_offset=SELECTED_OFFSET_MIN,
                            glucose_axis=False):
    """Plot one sensitivity metric (ΔTIR or ΔMedian) vs. offset, one line per participant.
    The legend (shown on panel F only) distinguishes participants; `highlight_offset` is
    called out with a shaded band and an open ring around each series' point.
    `series` is an iterable of (patient_id, values) pairs, values aligned with `offsets`.
    `glucose_axis` marks the y values as a glucose difference (mg/dL), so the tick
    labels follow the display unit; `ylabel` is supplied by the caller either way.
    """
    ax.axhline(0, color=COLOR_ZERO_LINE, linestyle="--", linewidth=0.7, zorder=1)

    if highlight_offset in offsets:
        step = min(np.diff(sorted(offsets))) if len(offsets) > 1 else 10
        band_halfwidth = step / 3
        ax.axvspan(highlight_offset - band_halfwidth, highlight_offset + band_halfwidth,
                  color="#F0C808", alpha=0.18, zorder=0, linewidth=0)
        ax.axvline(highlight_offset, color="#9A7B00", linestyle=":", linewidth=1.0,
                  alpha=0.8, zorder=2)

    for patient_id, values in series:
        style = PATIENT_STYLES[patient_id]
        ax.plot(offsets, values, color=style["color"], lw=1.3, ls=style["ls"],
               alpha=0.85, zorder=3, label=f"Participant {patient_id}")
        ax.scatter(offsets, values, s=38, marker=style["marker"], c=style["color"],
                  edgecolors=style["color"], linewidths=0.8, alpha=0.9, zorder=5)

        if highlight_offset in offsets:
            idx = offsets.index(highlight_offset)
            ax.scatter([highlight_offset], [values[idx]], s=95, marker=style["marker"],
                      facecolors='none', edgecolors=style["color"], linewidths=1.6, zorder=6)

    ax.set_xlabel("δ (min)", fontsize=7, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=7, labelpad=2)
    ax.set_xticks(offsets)
    ax.set_xticklabels([str(d) for d in offsets], fontsize=6.5)

    if y_limits is not None:
        ax.set_ylim(y_limits[0], y_limits[1])

    if glucose_axis:
        apply_glucose_axis(ax, set_label=False)

    style_axes(ax)
    ax.tick_params(length=3, width=0.8, labelsize=6.5)
    ax.grid(True, alpha=0.3, linewidth=0.5, zorder=0)

    if show_legend:
        ax.legend(fontsize=6, loc='best', framealpha=0.95)

# =============================================================================
# MAIN FIGURE GENERATION
# =============================================================================
def set_patient_ids(high=None, low=None):
    """Select the (high-SLH, low-SLH) pair; None falls back to env vars / automatic selection."""
    global PATIENT_IDS, PATIENT_STYLES
    if high or low:
        PATIENT_IDS = list(resolve_patient_pair(high, low))
        PATIENT_STYLES = {
            PATIENT_IDS[0]: {"color": CATEGORICAL_5[0], "marker": "o", "ls": "-"},
            PATIENT_IDS[1]: {"color": CATEGORICAL_5[1], "marker": "s", "ls": "--"},
        }


def generate_figure_4(high=None, low=None):
    set_patient_ids(high, low)
    apply_publication_style()
    
    print("\n" + "=" * 70)
    print("  GENERATING FIGURE 4 (WITH CSV EXPORT)")
    print("=" * 70)
    print(f"  Patients: {PATIENT_IDS}")
    print(f"  7-day overview: {PATIENT_IDS[0]} only (full width)")
    print(f"  AGP comparison: both patients (with patient ID in title)")
    print(f"  Selected offset: {SELECTED_OFFSET_MIN} min")
    print("=" * 70)
    
    # Load data for both patients
    data = {}
    for pid in PATIENT_IDS:
        print(f"\n  Loading {pid}...")
        df_raw = load_patient_window(pid)
        if df_raw is None:
            raise FileNotFoundError(f"Cannot load {pid}")
        
        df = prepare_valid_data(df_raw)
        timestep = detect_dominant_sampling_interval(df)
        raw_episodes = detect_hyperglycemia_episodes(df)
        classification_df = classify_episodes(df, raw_episodes)
        
        slh_df = classification_df[classification_df["classification"] == LABEL_SLH]
        slh_episodes = [{"onset_time": row["onset_time"], "offset_time": row["offset_time"]}
                       for _, row in slh_df.iterrows()]
        
        data[pid] = {
            "df": df,
            "timestep": timestep,
            "slh_episodes": slh_episodes,
            "raw_episodes": raw_episodes,
        }
        print(f"    SLH episodes: {len(slh_episodes)}")
    
    # Select episode for panel A (from second patient) - MUST BE SLH
    pid0 = PATIENT_IDS[1]
    eligible_eps = []
    window_delta = pd.Timedelta(hours=PLOT_WINDOW_H)
    
    for ep_dict in data[pid0]["slh_episodes"]:
        if ((ep_dict["onset_time"] - window_delta >= data[pid0]["df"][TIME_COL].iloc[0]) and
            (ep_dict["offset_time"] + window_delta <= data[pid0]["df"][TIME_COL].iloc[-1])):
            eligible_eps.append(ep_dict)
    
    if not eligible_eps:
        raise ValueError("No eligible SLH episodes for panel A")
    
    episode_for_panel_a = eligible_eps[len(eligible_eps) // 2]
    
    # Compute y-limits for 7-day overview (first patient only)
    print("\n  Computing y-limits for 7-day overview...")
    pid_overview = PATIENT_IDS[0]
    t_start = data[pid_overview]["df"][TIME_COL].iloc[0]
    t_end = t_start + pd.Timedelta(days=DAYS_PER_ROW)
    mask = (data[pid_overview]["df"][TIME_COL] >= t_start) & (data[pid_overview]["df"][TIME_COL] < t_end)
    
    g_row = data[pid_overview]["df"].loc[mask, GLUCOSE_COL].values
    c_row = build_full_counterfactual_signal(data[pid_overview]["df"], 
                                             data[pid_overview]["slh_episodes"],
                                             SELECTED_OFFSET_MIN,
                                             data[pid_overview]["timestep"])[mask.values]
    
    all_overview_values = list(g_row[~np.isnan(g_row)]) + list(c_row[~np.isnan(c_row)])
    overview_y_min = max(40, np.floor(np.nanmin(all_overview_values) / 10) * 10 - 12)
    overview_y_max = min(450, np.ceil(np.nanmax(all_overview_values) / 10) * 10 + 18)
    overview_y_limits = (overview_y_min, overview_y_max)
    print(f"  Overview y-limits: [{overview_y_min}, {overview_y_max}]")
    
    # Compute AGP y-limits (both patients)
    print("\n  Computing AGP y-limits...")
    all_agp_values = []
    
    for pid in PATIENT_IDS:
        agp_orig = compute_agp_profile(data[pid]["df"], data[pid]["timestep"])
        c_signal = build_full_counterfactual_signal(data[pid]["df"], 
                                                    data[pid]["slh_episodes"],
                                                    SELECTED_OFFSET_MIN,
                                                    data[pid]["timestep"])
        df_ctf_temp = data[pid]["df"].copy()
        df_ctf_temp[GLUCOSE_COL] = c_signal
        agp_ctf = compute_agp_profile(df_ctf_temp, data[pid]["timestep"])
        
        all_agp_values.extend(agp_orig["q25"].values)
        all_agp_values.extend(agp_orig["q75"].values)
        all_agp_values.extend(agp_ctf["q25"].values)
        all_agp_values.extend(agp_ctf["q75"].values)
    
    agp_y_min = max(AGP_Y_MIN, np.floor(np.nanmin(all_agp_values) / 10) * 10 - 10)
    agp_y_max = min(AGP_Y_MAX, np.ceil(np.nanmax(all_agp_values) / 10) * 10 + 10)
    agp_y_limits = (agp_y_min, agp_y_max)
    print(f"  AGP y-limits: [{agp_y_min}, {agp_y_max}]")
    
    # Compute sensitivity records once per patient (reused for y-limits, row-4 plots, and CSV export)
    print("\n  Computing sensitivity records...")
    all_delta_tir = []
    all_delta_median = []

    for pid in PATIENT_IDS:
        records = compute_sensitivity_records(pid, data[pid]["df"], data[pid]["slh_episodes"],
                                              data[pid]["timestep"], OFFSET_VALUES_COMPARE)
        data[pid]["sensitivity_records"] = records
        data[pid]["delta_tir"] = [r["delta_TIR_70_180_pct"] for r in records]
        data[pid]["delta_median"] = [r["delta_median_mg_dL"] for r in records]

        all_delta_tir.extend(data[pid]["delta_tir"])
        all_delta_median.extend(data[pid]["delta_median"])

    tir_y_min = np.floor(min(all_delta_tir) * 10) / 10 - 0.5
    tir_y_max = np.ceil(max(all_delta_tir) * 10) / 10 + 0.5
    tir_y_limits = (tir_y_min, tir_y_max)
    
    median_y_min = np.floor(min(all_delta_median)) - 1
    median_y_max = np.ceil(max(all_delta_median)) + 1
    median_y_limits = (median_y_min, median_y_max)
    
    print(f"  TIR y-limits: [{tir_y_min}, {tir_y_max}]")
    print(f"  Median y-limits: [{median_y_min}, {median_y_max}]")
    
    # Create figure
    fig = plt.figure(figsize=(7.24, 9.5))
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1.2, 1.2, 1.2, 1.0],
                          hspace=0.40, wspace=0.30,
                          left=0.08, right=0.93, top=0.94, bottom=0.06)
    
    # Panel A: Episode comparison (full width)
    ax_a = fig.add_subplot(gs[0, :])
    plot_episode_comparison(ax_a, pid0, data[pid0]["df"], episode_for_panel_a,
                           OFFSET_VALUES_COMPARE, data[pid0]["timestep"])
    add_panel_title(ax_a, "Single SLH episode: observed G(t) vs. counterfactual C(t) across δ")
    
    # Panel B: 7-day overview (full width, first patient only)
    ax_b = fig.add_subplot(gs[1, :])
    plot_7day_overview(ax_b, PATIENT_IDS[0], data[PATIENT_IDS[0]]["df"],
                      data[PATIENT_IDS[0]]["slh_episodes"], SELECTED_OFFSET_MIN,
                      data[PATIENT_IDS[0]]["timestep"], y_limits=overview_y_limits)
    add_panel_title(ax_b, f"7-day overview, Participant {PATIENT_IDS[0]}: "
                          f"G(t) vs. C(t) (δ = {SELECTED_OFFSET_MIN} min)")
    add_panel_label(ax_b, "B", x=-0.05)
    
    # Panels C/D: AGP comparison (with patient ID in title)
    ax_c = fig.add_subplot(gs[2, 0])
    plot_agp_comparison(ax_c, PATIENT_IDS[0], data[PATIENT_IDS[0]]["df"],
                       data[PATIENT_IDS[0]]["slh_episodes"], SELECTED_OFFSET_MIN,
                       data[PATIENT_IDS[0]]["timestep"], y_limits=agp_y_limits,
                       show_legend=False)
    add_panel_label(ax_c, "C", x=-0.10)
    
    ax_d = fig.add_subplot(gs[2, 1])
    plot_agp_comparison(ax_d, PATIENT_IDS[1], data[PATIENT_IDS[1]]["df"],
                       data[PATIENT_IDS[1]]["slh_episodes"], SELECTED_OFFSET_MIN,
                       data[PATIENT_IDS[1]]["timestep"], y_limits=agp_y_limits,
                       show_legend=True)
    add_panel_label(ax_d, "D", x=-0.10)
    
    # Panels E/F: Sensitivity — E = ΔTIR, F = ΔMedian, one line per participant.
    # δ=30 min (SELECTED_OFFSET_MIN, the main-analysis offset) is highlighted in both.
    tir_series = [(pid, data[pid]["delta_tir"]) for pid in PATIENT_IDS]
    median_series = [(pid, data[pid]["delta_median"]) for pid in PATIENT_IDS]

    ax_e = fig.add_subplot(gs[3, 0])
    plot_sensitivity_metric(ax_e, "ΔTIR 70-180 (%, daytime)", tir_series,
                           OFFSET_VALUES_COMPARE, y_limits=tir_y_limits, show_legend=False)
    add_panel_title(ax_e, "Sensitivity of ΔTIR to δ")
    add_panel_label(ax_e, "E", x=-0.10)

    ax_f = fig.add_subplot(gs[3, 1])
    plot_sensitivity_metric(ax_f, f"ΔMedian ({glucose_unit_label()}, daytime)", median_series,
                           OFFSET_VALUES_COMPARE, y_limits=median_y_limits, show_legend=True,
                           glucose_axis=True)
    add_panel_title(ax_f, "Sensitivity of ΔMedian to δ")
    add_panel_label(ax_f, "F", x=-0.10)
    
    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"Figure4_Counterfactual_Construction_{PATIENT_IDS[0]}_vs_{PATIENT_IDS[1]}")
    save_figure_with_panels(fig, output_path, os.path.basename(output_path),
                            {"A": ax_a, "B": ax_b, "C": ax_c, "D": ax_d, "E": ax_e, "F": ax_f})
    
    print("\n" + "=" * 70)
    print(f"  SAVED: {output_path}")
    print("=" * 70 + "\n")
    
    # =============================================================================
    # EXPORT DELTA VALUES TO CSV
    # =============================================================================
    print("  Exporting delta values to CSV...")
    csv_records = [record for pid in PATIENT_IDS for record in data[pid]["sensitivity_records"]]

    df_deltas = pd.DataFrame(csv_records)
    
    # Save to subfolder with patient IDs
    csv_subfolder = os.path.join(OUTPUT_DIR, f"{PATIENT_IDS[0]}_vs_{PATIENT_IDS[1]}")
    os.makedirs(csv_subfolder, exist_ok=True)
    csv_path = os.path.join(csv_subfolder, "sensitivity_deltas_daytime.csv")
    
    df_deltas.to_csv(csv_path, index=False)
    print(f"  CSV SAVED: {csv_path}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Figure 4: counterfactual construction.")
    parser.add_argument("--high_id", default=None, help="High-SLH patient ID (default: automatic)")
    parser.add_argument("--low_id", default=None, help="Low-SLH patient ID (default: automatic)")
    args = parser.parse_args()
    generate_figure_4(args.high_id, args.low_id)