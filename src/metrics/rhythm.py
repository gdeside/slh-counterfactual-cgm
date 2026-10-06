"""
src/metrics/rhythm.py
=====================

Clock-lock index Λ — a permutation-*sensitive* CGM metric.

Every consensus CGM metric (TIR, TAR, TBR, CV, SD, GMI) is a functional of the
marginal distribution of glucose, and is therefore invariant under permutation
of the readings: shuffle a 45-day record into random order and not one of them
changes.  They encode *how much* time is spent out of range, never *in what
shape*, and therefore never *when*.

Λ measures the one thing permutation destroys first: whether the out-of-range
frequency falls at the **same clock times every day**.  It answers a question no
consensus metric can — *is this patient's hyperglycaemia a schedule or a
scatter?* — and it separates two therapeutic situations that share a CGM
report: a reproducible daily pattern (a regimen problem, fixable at a known
hour) and a diffuse one (a behavioural or variability problem).

Construction
------------
1. **Day × clock matrix.**  ``H[d, b]`` is the fraction of readings beyond the
   threshold on day ``d`` in clock bin ``b`` (``N_CLOCK_BINS`` bins per 24 h).
2. **Observed concentration.**  ``F_obs`` is the share of the total frequency
   carried by the worst ``TOP_FRACTION`` of clock bins — by default the worst
   6 h of the day, not necessarily contiguous, so that a regular three-meal
   pattern is not penalised for being multimodal.
3. **Chance floor.**  Every record shows some clustering by chance.  ``F_null``
   is ``F_obs`` re-computed over ``N_NULL_DRAWS`` draws in which each day is
   circularly shifted by an independent random offset.  That null preserves the
   marginal distribution exactly, every excursion duration exactly, and the
   within-day autocorrelation — it destroys *only* alignment to clock time.
4. **Intrinsic ceiling.**  A patient with TAR 60 % cannot fit their frequency into
   6 h of the day.  ``F_max`` is the concentration reached if every day stacked
   its whole frequency into the same bins, capped cell-by-cell at 1.0.  Without
   this term the index is an artefact of TAR (ρ = −0·48 in the SLH cohort);
   with it, it is independent of every consensus metric.

   .. math:: \\Lambda = \\frac{F_{obs} - F_{null}}{F_{max} - F_{null}}

Λ = 0 means the frequency is no more clock-locked than chance; Λ = 1 means it is
as clock-locked as it could possibly be given how much of it there is.  Both
normalisers are computed from the patient's own record, so Λ is portable to any
cohort without re-fitting.

The threshold is a parameter: applied above 180 mg/dL it scores reproducible
hyperglycaemia, applied below 70 mg/dL it scores **recurrent hypoglycaemia at a
fixed hour**, which is the more immediately actionable of the two.

Authors
-------
Guillaume Deside

Date
----
2026
"""

import numpy as np
import pandas as pd

from src.config import TIME_COL, GLUCOSE_COL, THRESHOLD_HYPER, THRESHOLD_HYPO

# =============================================================================
# CONFIGURATION
# =============================================================================

#: Number of clock bins per 24 h (48 → 30-minute resolution).  Coarse enough to
#: absorb day-to-day jitter in meal timing, fine enough to separate meals.
N_CLOCK_BINS = 48

#: Share of the day defining "the worst hours" (0.25 → the worst 6 h).
TOP_FRACTION = 0.25

#: Draws of the circular-shift null used to estimate the chance floor.
N_NULL_DRAWS = 500

#: Seed for the null, so Λ is reproducible run to run.
RANDOM_SEED = 42

#: Minimum number of days with data required to report Λ.  Below this the
#: chance floor is too wide for the index to mean anything.
MIN_DAYS_FOR_RHYTHM = 10

# =============================================================================
# METRIC NAMES
# =============================================================================

METRIC_CLOCK_LOCK_HYPER = "Lambda_hyper"
METRIC_CLOCK_LOCK_HYPO = "Lambda_hypo"
METRIC_CLOCK_LOCK_P_HYPER = "Lambda_hyper_p"
METRIC_CLOCK_LOCK_P_HYPO = "Lambda_hypo_p"

RHYTHM_METRIC_NAMES = [
    METRIC_CLOCK_LOCK_HYPER,
    METRIC_CLOCK_LOCK_P_HYPER,
    METRIC_CLOCK_LOCK_HYPO,
    METRIC_CLOCK_LOCK_P_HYPO,
]


# =============================================================================
# DAY × CLOCK MATRIX
# =============================================================================

def build_day_clock_matrix(times, glucose, threshold: float, above: bool = True,
                           n_bins: int = N_CLOCK_BINS) -> np.ndarray:
    """
    Build the day × clock-bin frequency matrix H.

    Parameters
    ----------
    times : pd.Series of datetime64
        Reading timestamps.
    glucose : array-like
        Glucose values, same length as ``times``.  NaNs are dropped.
    threshold : float
        Glucose threshold in mg/dL.
    above : bool, default True
        ``True`` scores readings above the threshold (hyperglycaemia),
        ``False`` scores readings below it (hypoglycaemia).
    n_bins : int
        Clock bins per 24 h.

    Returns
    -------
    np.ndarray, shape (n_days, n_bins)
        ``H[d, b]`` is the fraction of valid readings beyond the threshold in
        that day and clock bin; ``NaN`` where the cell holds no reading.
    """
    times = pd.Series(pd.to_datetime(times)).reset_index(drop=True)
    glucose = np.asarray(glucose, dtype=float)

    valid = np.isfinite(glucose)
    times, glucose = times[valid].reset_index(drop=True), glucose[valid]

    if len(glucose) == 0:
        return np.empty((0, n_bins))

    day = times.dt.normalize().values.astype("datetime64[D]").astype(int)
    day = day - day.min()
    minutes_per_bin = 1440 // n_bins
    clock_bin = ((times.dt.hour * 60 + times.dt.minute) // minutes_per_bin).values

    beyond = (glucose > threshold) if above else (glucose < threshold)

    n_days = int(day.max()) + 1
    total = np.zeros((n_days, n_bins))
    hits = np.zeros((n_days, n_bins))
    np.add.at(total, (day, clock_bin), 1.0)
    np.add.at(hits, (day, clock_bin), beyond.astype(float))

    with np.errstate(invalid="ignore"):
        return np.where(total > 0, hits / np.maximum(total, 1.0), np.nan)


# =============================================================================
# CONCENTRATION, FLOOR AND CEILING
# =============================================================================

def frequency_concentration(clock_profile: np.ndarray,
                         top_fraction: float = TOP_FRACTION) -> float:
    """
    Share of the frequency carried by the worst ``top_fraction`` of clock bins.

    The bins are selected by rank, not by contiguity, so a regular three-meal
    pattern scores as concentrated even though it is multimodal.
    """
    total = np.nansum(clock_profile)
    if not np.isfinite(total) or total <= 0:
        return np.nan

    k = max(1, int(round(top_fraction * len(clock_profile))))
    return float(np.sort(np.nan_to_num(clock_profile))[-k:].sum() / total)


def ceiling_concentration(H: np.ndarray,
                          top_fraction: float = TOP_FRACTION) -> float:
    """
    Largest concentration reachable given how much frequency the patient carries.

    If every day stacked its whole frequency into the same ``k`` bins, day ``d``
    could place at most ``min(mass_d, k)`` there, because a single (day, bin)
    cell cannot exceed 1.0.  This is what makes Λ independent of TAR: it
    divides by what the patient *could* have achieved, not by 1.
    """
    if H.size == 0:
        return np.nan

    k = max(1, int(round(top_fraction * H.shape[1])))
    mass = np.nansum(H, axis=1)
    total = mass.sum()

    if total <= 0:
        return np.nan

    return float(np.minimum(mass, k).sum() / total)


def null_concentration(H: np.ndarray, n_draws: int = N_NULL_DRAWS,
                       top_fraction: float = TOP_FRACTION,
                       seed: int = RANDOM_SEED) -> np.ndarray:
    """
    Concentration under the per-day circular-shift null.

    Each day is rotated in clock time by its own random offset.  The marginal
    distribution, every excursion duration and the within-day autocorrelation
    survive intact; only alignment across days is destroyed.

    Returns
    -------
    np.ndarray, shape (n_draws,)
    """
    rng = np.random.default_rng(seed)
    n_days, n_bins = H.shape
    bins = np.arange(n_bins)

    draws = np.empty(n_draws)
    for i in range(n_draws):
        offsets = rng.integers(0, n_bins, size=n_days)
        rotated = np.take_along_axis(H, (bins[None, :] - offsets[:, None]) % n_bins, axis=1)
        draws[i] = frequency_concentration(np.nanmean(rotated, axis=0), top_fraction)

    return draws


# =============================================================================
# THE METRIC
# =============================================================================

def compute_clock_lock(H: np.ndarray, n_draws: int = N_NULL_DRAWS,
                       top_fraction: float = TOP_FRACTION,
                       seed: int = RANDOM_SEED) -> dict:
    """
    Compute the clock-lock index Λ from a day × clock matrix.

    Returns
    -------
    dict with keys
        ``lambda_`` : float
            The index.  0 = no more clock-locked than chance, 1 = maximally
            clock-locked given the frequency.  Slightly negative values are
            sampling noise around 0.
        ``p_value`` : float
            Permutation p-value — the share of null draws at least as
            concentrated as observed.  Answers "is this patient clock-locked
            at all?"
        ``f_observed``, ``f_null``, ``f_max`` : float
            The three concentrations, for diagnostics and plotting.
        ``clock_profile`` : np.ndarray
            Mean frequency per clock bin — the curve Λ summarises.
        ``n_days`` : int
    """
    empty = dict(lambda_=np.nan, p_value=np.nan, f_observed=np.nan,
                 f_null=np.nan, f_max=np.nan, clock_profile=None, n_days=0)

    if H.size == 0 or H.shape[0] < MIN_DAYS_FOR_RHYTHM:
        return {**empty, "n_days": int(H.shape[0]) if H.size else 0}

    clock_profile = np.nanmean(H, axis=0)
    f_observed = frequency_concentration(clock_profile, top_fraction)
    f_max = ceiling_concentration(H, top_fraction)

    if not np.isfinite(f_observed) or not np.isfinite(f_max):
        return {**empty, "n_days": int(H.shape[0])}

    draws = null_concentration(H, n_draws, top_fraction, seed)
    f_null = float(np.mean(draws))

    span = f_max - f_null
    lambda_ = float((f_observed - f_null) / span) if span > 1e-9 else np.nan
    p_value = float((np.sum(draws >= f_observed) + 1) / (n_draws + 1))

    return dict(lambda_=lambda_, p_value=p_value, f_observed=f_observed,
                f_null=f_null, f_max=f_max, clock_profile=clock_profile,
                n_days=int(H.shape[0]))


def compute_rhythm_metrics(df: pd.DataFrame, time_col: str = TIME_COL,
                           glucose_col: str = GLUCOSE_COL,
                           n_draws: int = N_NULL_DRAWS) -> dict:
    """
    Compute Λ above the hyperglycaemia threshold and below the hypoglycaemia
    threshold for one patient record.

    Parameters
    ----------
    df : pd.DataFrame
        Patient record with a timestamp column and a glucose column (mg/dL).

    Returns
    -------
    dict
        Keyed by :data:`RHYTHM_METRIC_NAMES`, ready to merge into the standard
        metrics row.
    """
    times, glucose = df[time_col], df[glucose_col].values

    hyper = compute_clock_lock(
        build_day_clock_matrix(times, glucose, THRESHOLD_HYPER, above=True),
        n_draws=n_draws)
    hypo = compute_clock_lock(
        build_day_clock_matrix(times, glucose, THRESHOLD_HYPO, above=False),
        n_draws=n_draws)

    return {
        METRIC_CLOCK_LOCK_HYPER: hyper["lambda_"],
        METRIC_CLOCK_LOCK_P_HYPER: hyper["p_value"],
        METRIC_CLOCK_LOCK_HYPO: hypo["lambda_"],
        METRIC_CLOCK_LOCK_P_HYPO: hypo["p_value"],
    }


def peak_clock_hours(clock_profile: np.ndarray,
                     top_fraction: float = TOP_FRACTION) -> np.ndarray:
    """
    Clock hours making up the worst ``top_fraction`` of the day.

    This is the clinically operative output: Λ says *whether* there is a
    schedule, this says *what time it is*.
    """
    if clock_profile is None:
        return np.array([])

    n_bins = len(clock_profile)
    k = max(1, int(round(top_fraction * n_bins)))
    worst = np.argsort(np.nan_to_num(clock_profile))[-k:]
    return np.sort(worst) * (24.0 / n_bins)
