"""Tests for src/metrics/rhythm.py — the clock-lock index Λ."""

import numpy as np
import pandas as pd
import pytest

from src.metrics.rhythm import (
    build_day_clock_matrix,
    frequency_concentration,
    ceiling_concentration,
    compute_clock_lock,
    compute_rhythm_metrics,
    peak_clock_hours,
    N_CLOCK_BINS,
    MIN_DAYS_FOR_RHYTHM,
    RHYTHM_METRIC_NAMES,
)

TIME_COL = "time"
GLUCOSE_COL = "glycemie"


def make_record(n_days: int, peak_hours=None, baseline=120.0, peak=250.0,
                jitter_min: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """
    Synthetic 5-minute record.

    ``peak_hours`` is a list of clock hours that go hyperglycaemic every day.
    ``jitter_min`` shifts each day's peaks by a random offset, which is what
    turns a schedule into a scatter.
    """
    rng = np.random.default_rng(seed)
    times = pd.date_range("2025-01-01", periods=n_days * 288, freq="5min")
    glucose = np.full(len(times), baseline)

    if peak_hours:
        hours = times.hour + times.minute / 60.0
        day_index = (times - times[0]).days
        offsets = rng.uniform(-jitter_min, jitter_min, size=n_days) / 60.0
        for peak_hour in peak_hours:
            centre = peak_hour + offsets[day_index]
            glucose = np.where(np.abs(hours - centre) < 0.5, peak, glucose)

    return pd.DataFrame({TIME_COL: times, GLUCOSE_COL: glucose})


class TestDayClockMatrix:
    def test_shape_is_days_by_clock_bins(self):
        df = make_record(14, peak_hours=[8])
        H = build_day_clock_matrix(df[TIME_COL], df[GLUCOSE_COL].values, 180.0)
        assert H.shape == (14, N_CLOCK_BINS)

    def test_cells_are_fractions_in_unit_interval(self):
        df = make_record(14, peak_hours=[8, 13])
        H = build_day_clock_matrix(df[TIME_COL], df[GLUCOSE_COL].values, 180.0)
        finite = H[np.isfinite(H)]
        assert finite.min() >= 0.0 and finite.max() <= 1.0

    def test_below_threshold_direction(self):
        df = make_record(14, peak_hours=[8], peak=40.0)
        H = build_day_clock_matrix(df[TIME_COL], df[GLUCOSE_COL].values, 70.0,
                                   above=False)
        assert np.nansum(H) > 0

    def test_empty_record_returns_empty_matrix(self):
        df = make_record(2)
        df[GLUCOSE_COL] = np.nan
        H = build_day_clock_matrix(df[TIME_COL], df[GLUCOSE_COL].values, 180.0)
        assert H.shape[0] == 0


class TestConcentration:
    def test_uniform_profile_gives_the_top_fraction(self):
        assert frequency_concentration(np.ones(48), top_fraction=0.25) == pytest.approx(0.25)

    def test_single_spike_gives_one(self):
        profile = np.zeros(48)
        profile[10] = 1.0
        assert frequency_concentration(profile, top_fraction=0.25) == pytest.approx(1.0)

    def test_ceiling_is_one_when_frequency_fits_in_the_top_bins(self):
        H = np.zeros((10, 48))
        H[:, :4] = 1.0                      # 4 bins of frequency, ceiling holds 12
        assert ceiling_concentration(H, top_fraction=0.25) == pytest.approx(1.0)

    def test_ceiling_falls_below_one_when_frequency_overflows(self):
        H = np.ones((10, 48))               # frequency everywhere: can only ever be 25 %
        assert ceiling_concentration(H, top_fraction=0.25) == pytest.approx(0.25)


class TestClockLock:
    def test_scheduled_peaks_score_high_and_significant(self):
        df = make_record(45, peak_hours=[8, 13, 19])
        result = compute_clock_lock(build_day_clock_matrix(
            df[TIME_COL], df[GLUCOSE_COL].values, 180.0), n_draws=200)
        assert result["lambda_"] > 0.8
        assert result["p_value"] < 0.01

    def test_scattered_peaks_score_near_zero(self):
        df = make_record(45, peak_hours=[8, 13, 19], jitter_min=600, seed=3)
        result = compute_clock_lock(build_day_clock_matrix(
            df[TIME_COL], df[GLUCOSE_COL].values, 180.0), n_draws=200)
        assert result["lambda_"] < 0.4

    def test_permuting_the_record_collapses_lambda(self):
        """The defining property: Λ is not a functional of the marginal."""
        df = make_record(45, peak_hours=[8, 13, 19])
        glucose = df[GLUCOSE_COL].values
        shuffled = np.random.default_rng(0).permutation(glucose)

        observed = compute_clock_lock(build_day_clock_matrix(
            df[TIME_COL], glucose, 180.0), n_draws=200)["lambda_"]
        permuted = compute_clock_lock(build_day_clock_matrix(
            df[TIME_COL], shuffled, 180.0), n_draws=200)["lambda_"]

        # the marginal — and therefore every consensus metric — is untouched
        assert np.sort(glucose) == pytest.approx(np.sort(shuffled))
        assert observed > 0.8
        assert abs(permuted) < 0.15

    def test_short_records_are_refused(self):
        df = make_record(MIN_DAYS_FOR_RHYTHM - 1, peak_hours=[8])
        result = compute_clock_lock(build_day_clock_matrix(
            df[TIME_COL], df[GLUCOSE_COL].values, 180.0), n_draws=50)
        assert np.isnan(result["lambda_"])

    def test_no_frequency_returns_nan(self):
        df = make_record(20)                # never crosses 180
        result = compute_clock_lock(build_day_clock_matrix(
            df[TIME_COL], df[GLUCOSE_COL].values, 180.0), n_draws=50)
        assert np.isnan(result["lambda_"])

    def test_is_reproducible_across_runs(self):
        df = make_record(30, peak_hours=[9, 14])
        H = build_day_clock_matrix(df[TIME_COL], df[GLUCOSE_COL].values, 180.0)
        assert compute_clock_lock(H, n_draws=100)["lambda_"] == \
               compute_clock_lock(H, n_draws=100)["lambda_"]


class TestRhythmMetrics:
    def test_returns_every_declared_metric(self):
        df = make_record(30, peak_hours=[8, 13])
        metrics = compute_rhythm_metrics(df, n_draws=100)
        assert set(metrics) == set(RHYTHM_METRIC_NAMES)

    def test_peak_hours_recover_the_scheduled_peaks(self):
        df = make_record(45, peak_hours=[8, 13, 19])
        result = compute_clock_lock(build_day_clock_matrix(
            df[TIME_COL], df[GLUCOSE_COL].values, 180.0), n_draws=100)
        hours = peak_clock_hours(result["clock_profile"])
        for scheduled in (8, 13, 19):
            assert np.any(np.abs(hours - scheduled) <= 1.0)
