"""
Tests for src/preprocessing/coverage.py

Tests coverage calculation functions for CGM data quality assessment.
"""

import pytest
import numpy as np

# Import the functions to test
from src.preprocessing.coverage import (
    compute_expected_measurements,
    compute_coverage_percentage
)


# ==============================================================================
# Tests for compute_expected_measurements
# ==============================================================================

class TestComputeExpectedMeasurements:
    """Test suite for compute_expected_measurements function."""
    
    def test_standard_45day_5min_sampling(self):
        """Test standard project configuration: 45 days, 5-min sampling."""
        result = compute_expected_measurements(window_days=45, sampling_min=5.0)
        expected = 45 * 24 * 60 / 5  # = 12,960
        assert result == 12960
        assert isinstance(result, int)
    
    def test_standard_45day_15min_sampling(self):
        """Test 45-day window with 15-min sampling (alternative CGM device)."""
        result = compute_expected_measurements(window_days=45, sampling_min=15.0)
        expected = 45 * 24 * 60 / 15  # = 4,320
        assert result == 4320
        assert isinstance(result, int)
    
    @pytest.mark.parametrize("window_days,sampling_min,expected", [
        (1, 5, 288),        # 1 day, 5-min sampling
        (1, 15, 96),        # 1 day, 15-min sampling
        (7, 5, 2016),       # 1 week, 5-min sampling
        (7, 15, 672),       # 1 week, 15-min sampling
        (30, 5, 8640),      # 1 month, 5-min sampling
        (30, 15, 2880),     # 1 month, 15-min sampling
        (90, 5, 25920),     # 3 months, 5-min sampling
        (45, 1, 64800),     # 45 days, 1-min sampling (high-frequency)
        (45, 10, 6480),     # 45 days, 10-min sampling
    ])
    def test_various_window_sampling_combinations(self, window_days, sampling_min, expected):
        """Test different combinations of window durations and sampling intervals."""
        result = compute_expected_measurements(window_days, sampling_min)
        assert result == expected
        assert isinstance(result, int)
    
    def test_return_type_is_integer(self):
        """Ensure function always returns an integer."""
        result = compute_expected_measurements(window_days=45, sampling_min=5.0)
        assert isinstance(result, int)
        assert type(result) is int  # Strict type check
    
    def test_single_day_single_minute(self):
        """Edge case: 1 day with 1-minute sampling."""
        result = compute_expected_measurements(window_days=1, sampling_min=1.0)
        assert result == 1440  # 24 * 60
    
    def test_fractional_sampling_interval(self):
        """Test handling of fractional sampling intervals."""
        # 2.5 minute sampling (rare but possible)
        result = compute_expected_measurements(window_days=1, sampling_min=2.5)
        expected = int(1 * 24 * 60 / 2.5)
        assert result == expected
        assert result == 576
    
    def test_large_window(self):
        """Test with very large window (e.g., 1 year)."""
        result = compute_expected_measurements(window_days=365, sampling_min=5.0)
        expected = int(365 * 24 * 60 / 5)
        assert result == expected
        assert result == 105120


# ==============================================================================
# Tests for compute_coverage_percentage
# ==============================================================================

class TestComputeCoveragePercentage:
    """Test suite for compute_coverage_percentage function."""
    
    def test_perfect_coverage_100_percent(self):
        """Test 100% coverage (all expected measurements present)."""
        window_days = 45
        sampling_min = 5.0
        expected = compute_expected_measurements(window_days, sampling_min)
        n_valid = expected  # All measurements present
        
        result = compute_coverage_percentage(n_valid, window_days, sampling_min)
        assert result == 100.0
        assert isinstance(result, float)
    
    def test_partial_coverage_70_percent(self):
        """Test 70% coverage (project minimum threshold)."""
        window_days = 45
        sampling_min = 5.0
        expected = compute_expected_measurements(window_days, sampling_min)
        n_valid = int(expected * 0.7)  # 70% coverage
        
        result = compute_coverage_percentage(n_valid, window_days, sampling_min)
        assert 69.9 <= result <= 70.1  # Allow small floating-point tolerance
    
    def test_partial_coverage_50_percent(self):
        """Test 50% coverage."""
        window_days = 45
        sampling_min = 5.0
        expected = compute_expected_measurements(window_days, sampling_min)
        n_valid = expected // 2  # 50% coverage
        
        result = compute_coverage_percentage(n_valid, window_days, sampling_min)
        assert 49.9 <= result <= 50.1
    
    def test_zero_coverage(self):
        """Test 0% coverage (no valid measurements)."""
        result = compute_coverage_percentage(
            n_valid=0, 
            window_days=45, 
            sampling_min=5.0
        )
        assert result == 0.0
    
    def test_overcoverage_capped_at_100(self):
        """Test that coverage >100% is capped at exactly 100%."""
        window_days = 45
        sampling_min = 5.0
        expected = compute_expected_measurements(window_days, sampling_min)
        n_valid = int(expected * 1.5)  # 150% of expected
        
        result = compute_coverage_percentage(n_valid, window_days, sampling_min)
        assert result == 100.0  # Must be capped
    
    def test_slightly_over_100_percent(self):
        """Test edge case: slightly over 100% is capped."""
        window_days = 45
        sampling_min = 5.0
        expected = compute_expected_measurements(window_days, sampling_min)
        n_valid = expected + 10  # Just slightly over
        
        result = compute_coverage_percentage(n_valid, window_days, sampling_min)
        assert result == 100.0
    
    @pytest.mark.parametrize("coverage_pct", [
        10.0, 25.0, 33.3, 50.0, 70.0, 75.0, 85.0, 90.0, 95.0, 99.0
    ])
    def test_various_coverage_percentages(self, coverage_pct):
        """Test different coverage percentages."""
        window_days = 45
        sampling_min = 5.0
        expected = compute_expected_measurements(window_days, sampling_min)
        n_valid = int(expected * coverage_pct / 100)
        
        result = compute_coverage_percentage(n_valid, window_days, sampling_min)
        # Allow 1% tolerance for rounding
        assert coverage_pct - 1.0 <= result <= coverage_pct + 1.0
    
    def test_return_type_is_float(self):
        """Ensure function always returns a float."""
        result = compute_coverage_percentage(
            n_valid=9000, 
            window_days=45, 
            sampling_min=5.0
        )
        assert isinstance(result, float)
    
    def test_different_sampling_intervals(self):
        """Test coverage calculation with different sampling intervals."""
        window_days = 45
        n_valid = 6000
        
        # 5-min sampling
        cov_5min = compute_coverage_percentage(n_valid, window_days, 5.0)
        
        # 15-min sampling  
        cov_15min = compute_coverage_percentage(n_valid, window_days, 15.0)
        
        # Same n_valid but different expected -> different coverage %
        assert cov_5min < cov_15min  # Lower expected means higher coverage %
    
    def test_consistency_with_project_config(self):
        """Test with actual project configuration values."""
        # From config.py: WINDOW_DAYS = 45, MIN_COVERAGE_PCT = 70.0
        window_days = 45
        sampling_min = 5.0
        expected = compute_expected_measurements(window_days, sampling_min)
        
        # Minimum acceptable measurements for 70% coverage
        min_valid = int(expected * 0.7)
        
        result = compute_coverage_percentage(min_valid, window_days, sampling_min)
        assert result >= 70.0  # Should meet or exceed threshold
    
    def test_single_measurement(self):
        """Edge case: single valid measurement."""
        result = compute_coverage_percentage(
            n_valid=1,
            window_days=1,
            sampling_min=1440.0  # 1 measurement per day
        )
        assert result == 100.0
    
    def test_very_low_coverage(self):
        """Test very low coverage (<1%)."""
        window_days = 45
        sampling_min = 5.0
        n_valid = 100  # Very few measurements
        
        result = compute_coverage_percentage(n_valid, window_days, sampling_min)
        assert 0.0 < result < 1.0


# ==============================================================================
# Edge Cases and Error Conditions
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_exact_boundary_values(self):
        """Test exact boundary at 100% coverage."""
        window_days = 1
        sampling_min = 5.0
        expected = compute_expected_measurements(window_days, sampling_min)
        
        # Exactly at boundary
        result = compute_coverage_percentage(expected, window_days, sampling_min)
        assert result == 100.0
        
        # Just below boundary
        result = compute_coverage_percentage(expected - 1, window_days, sampling_min)
        assert result < 100.0
        
        # Just above boundary
        result = compute_coverage_percentage(expected + 1, window_days, sampling_min)
        assert result == 100.0  # Capped
    
    def test_rounding_behavior(self):
        """Test that rounding doesn't cause unexpected behavior."""
        # Choose values that might cause rounding issues
        window_days = 7
        sampling_min = 3.0  # Non-standard interval
        expected = compute_expected_measurements(window_days, sampling_min)
        
        # 99.999% coverage
        n_valid = expected - 1
        result = compute_coverage_percentage(n_valid, window_days, sampling_min)
        assert result < 100.0
    
    def test_integer_division_consistency(self):
        """Ensure integer conversion in expected_measurements doesn't cause issues."""
        # Test cases where division might not be exact
        result1 = compute_expected_measurements(window_days=1, sampling_min=7.0)
        result2 = compute_expected_measurements(window_days=7, sampling_min=1.0)
        
        # Both should be integers
        assert isinstance(result1, int)
        assert isinstance(result2, int)
        
        # Mathematical relationship should hold
        assert result1 * 7 <= result2 * 1  # Account for integer rounding


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestIntegration:
    """Integration tests combining both functions."""
    
    def test_round_trip_calculation(self):
        """Test that coverage calculation is consistent with expected measurements."""
        window_days = 45
        sampling_min = 5.0
        
        expected = compute_expected_measurements(window_days, sampling_min)
        
        # Full coverage
        coverage = compute_coverage_percentage(expected, window_days, sampling_min)
        assert coverage == 100.0
        
        # Half coverage
        coverage = compute_coverage_percentage(expected // 2, window_days, sampling_min)
        assert 49.0 <= coverage <= 51.0
    
    def test_realistic_cgm_scenario(self):
        """Test realistic CGM data scenario."""
        # Scenario: 45-day window, 5-min sampling, 75% data completeness
        window_days = 45
        sampling_min = 5.0
        target_coverage = 75.0
        
        expected = compute_expected_measurements(window_days, sampling_min)
        n_valid = int(expected * 0.75)
        
        actual_coverage = compute_coverage_percentage(n_valid, window_days, sampling_min)
        
        # Should be close to target (within 1%)
        assert abs(actual_coverage - target_coverage) < 1.0
        
        # Actual values for this scenario
        assert expected == 12960
        assert n_valid == 9720
        assert 74.0 <= actual_coverage <= 76.0
    
    def test_multiple_window_sizes_consistency(self):
        """Test that coverage scales consistently across window sizes."""
        sampling_min = 5.0
        n_valid = 1000
        
        # Coverage should decrease as window size increases (for fixed n_valid)
        cov_1day = compute_coverage_percentage(n_valid, 1, sampling_min)
        cov_7day = compute_coverage_percentage(n_valid, 7, sampling_min)
        cov_45day = compute_coverage_percentage(n_valid, 45, sampling_min)
        
        assert cov_1day > cov_7day > cov_45day


# ==============================================================================
# Documentation Tests
# ==============================================================================

class TestDocumentation:
    """Test that functions behave as documented."""
    
    def test_expected_measurements_docstring_example(self):
        """Verify examples from docstring work correctly."""
        # Standard 45-day, 5-min sampling window
        result = compute_expected_measurements(45, 5.0)
        assert result == 12960
    
    def test_coverage_percentage_docstring_example(self):
        """Verify coverage calculation examples."""
        # 100% coverage
        result = compute_coverage_percentage(12960, 45, 5.0)
        assert result == 100.0
        
        # Overcoverage capped at 100%
        result = compute_coverage_percentage(20000, 45, 5.0)
        assert result == 100.0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
