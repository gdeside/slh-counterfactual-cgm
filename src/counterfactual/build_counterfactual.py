"""
src/counterfactual/build_counterfactual.py
==========================================

Counterfactual CGM signal construction for sustained-low hyperglycaemia (SLH)
episodes.

For each SLH episode the module replaces the measured glucose trace with a
smooth baseline that represents what glucose would have looked like in the
absence of the episode.  Construction proceeds in two steps:

1. **Linear interpolation** – local-minimum anchors are found immediately
   before episode onset and immediately after episode offset; a straight line
   is drawn between them (``build_linear_interpolation``).

2. **Rolling-mean smoothing** – the interpolated segment is convolved with a
   centred rolling mean of width ``KERNEL_MIN`` minutes to eliminate sharp
   transitions at the anchor points (``apply_rolling_mean_smoothing``).

The top-level entry points are:

* ``build_counterfactual`` – process a single episode.
* ``build_full_counterfactual_signal`` – apply the procedure across all SLH
  episodes in a patient record and return the composite counterfactual array.

Configuration (kernel width, offset range, default offset) is centralised in
``counterfactual_config.py``.
"""

import pandas as pd
import numpy as np

from src.config import TIME_COL, GLUCOSE_COL
from src.episodes.episode_config import H_HYPER
from src.counterfactual.counterfactual_config import APPLY_KERNEL, KERNEL_MIN


def find_local_minimum(df: pd.DataFrame, t_start: pd.Timestamp,
                       t_end: pd.Timestamp) -> tuple:
    """
    Find local minimum glucose value and time in window [t_start, t_end].

    Returns
    -------
    (t_min, g_min) : tuple of (pd.Timestamp, float) or (None, None)
    """
    mask = (df[TIME_COL] >= t_start) & (df[TIME_COL] <= t_end)
    subset = df.loc[mask].dropna(subset=[GLUCOSE_COL])

    if subset.empty:
        return None, None

    idx_min = subset[GLUCOSE_COL].idxmin()
    return subset.loc[idx_min, TIME_COL], float(subset.loc[idx_min, GLUCOSE_COL])


def build_linear_interpolation(df: pd.DataFrame, episode: dict,
                               offset_min: float, timestep_min: float):
    """
    Build linear interpolation baseline L(t) between anchor points.

    Parameters
    ----------
    df : pd.DataFrame
        Full patient data.
    episode : dict
        Episode dict with ``onset_time`` and ``offset_time`` keys.
    offset_min : float
        Zone expansion offset δ in minutes.
    timestep_min : float
        Sampling interval in minutes.

    Returns
    -------
    signal : np.ndarray
        Signal array with linear interpolation applied.
    anchor_info : dict
        Information about anchors and interpolation.
    """
    delta_td = pd.Timedelta(minutes=offset_min)
    zone_start = episode["onset_time"] - delta_td
    zone_end = episode["offset_time"] + delta_td

    # Find local-minima anchors at zone boundaries
    t_a, g_a = find_local_minimum(df, zone_start, episode["onset_time"])
    if t_a is None:
        t_a, g_a = zone_start, H_HYPER - 20.0

    t_b, g_b = find_local_minimum(df, episode["offset_time"], zone_end)
    if t_b is None:
        t_b, g_b = zone_end, H_HYPER - 20.0

    # Find indices between anchors (inclusive)
    anchor_mask = (df[TIME_COL] >= t_a) & (df[TIME_COL] <= t_b)
    anchor_idx = df.index[anchor_mask].tolist()

    signal = df[GLUCOSE_COL].values.astype(float).copy()

    if not anchor_idx:
        return signal, {}

    # Linear interpolation between anchors
    ta_s = t_a.timestamp()
    tb_s = t_b.timestamp()
    span = tb_s - ta_s

    for i in anchor_idx:
        ti_s = df.loc[i, TIME_COL].timestamp()
        if span > 0:
            signal[i] = g_a + (g_b - g_a) * (ti_s - ta_s) / span
        else:
            signal[i] = (g_a + g_b) / 2.0

    anchor_info = {
        "zone_start": zone_start,
        "zone_end": zone_end,
        "t_a": t_a,
        "g_a": g_a,
        "t_b": t_b,
        "g_b": g_b,
        "anchor_idx": anchor_idx,
    }

    return signal, anchor_info


def apply_rolling_mean_smoothing(signal: np.ndarray, anchor_idx: list,
                                 timestep_min: float) -> np.ndarray:
    """
    Apply centered rolling-mean smoothing to signal at anchor indices.

    Parameters
    ----------
    signal : np.ndarray
        Signal array.
    anchor_idx : list
        Indices to apply smoothing.
    timestep_min : float
        Sampling interval in minutes.

    Returns
    -------
    np.ndarray
        Smoothed signal.
    """
    if not APPLY_KERNEL:
        return signal

    window_width = max(3, round(KERNEL_MIN / timestep_min))
    half_width = (window_width - 1) // 2
    snapshot = signal.copy()
    n_total = len(signal)

    for i in anchor_idx:
        lo = max(0, i - half_width)
        hi = min(n_total, i + half_width + 1)
        vals = snapshot[lo:hi]
        valid_vals = vals[~np.isnan(vals)]
        if valid_vals.size > 0:
            signal[i] = float(np.mean(valid_vals))

    return signal


def build_counterfactual(df: pd.DataFrame, episode: dict,
                         offset_min: float, timestep_min: float):
    """
    Build complete counterfactual C(t) = linear interpolation + smoothing.

    Parameters
    ----------
    df : pd.DataFrame
        Full patient data.
    episode : dict
        Episode to process (must have ``onset_time`` and ``offset_time``).
    offset_min : float
        Zone expansion offset δ in minutes.
    timestep_min : float
        Sampling interval in minutes.

    Returns
    -------
    signal : np.ndarray
        Counterfactual signal.
    anchor_info : dict
        Anchor information.
    """
    signal, anchor_info = build_linear_interpolation(df, episode, offset_min, timestep_min)

    if anchor_info:
        signal = apply_rolling_mean_smoothing(signal, anchor_info["anchor_idx"], timestep_min)

    return signal, anchor_info


def build_full_counterfactual_signal(df: pd.DataFrame, slh_episodes: list,
                                     offset_min: float, timestep_min: float) -> np.ndarray:
    """
    Build complete counterfactual signal C(t) across all SLH episodes.

    Starts with original signal and progressively applies counterfactual
    construction to each SLH episode region.

    Parameters
    ----------
    df : pd.DataFrame
        Full patient data.
    slh_episodes : list of dict
        SLH episodes to process (each must have ``onset_time`` and ``offset_time``).
    offset_min : float
        Zone expansion offset δ in minutes.
    timestep_min : float
        Sampling interval in minutes.

    Returns
    -------
    np.ndarray
        Counterfactual signal array.
    """
    signal = df[GLUCOSE_COL].values.astype(float).copy()

    for ep in sorted(slh_episodes, key=lambda e: e["onset_time"]):
        signal_ep, anchor_info = build_counterfactual(df, ep, offset_min, timestep_min)

        if anchor_info:
            anchor_idx = anchor_info["anchor_idx"]
            signal[anchor_idx] = signal_ep[anchor_idx]

    return signal
