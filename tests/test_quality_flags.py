"""Tests for src/preprocessing/quality_flags.py - attach_quality_flags function."""

import pytest
import pandas as pd

from src.preprocessing.quality import attach_quality_flags


# Constants come from the project config so the tests cannot drift from the code
from src.config.constants import (
    TIME_COL, GLUCOSE_COL, GLUCOSE_MIN, GLUCOSE_MAX,
)


@pytest.fixture
def sample_data():
    """Create sample CGM data with 5-min sampling."""
    times = pd.date_range("2024-09-10 12:00:00", periods=10, freq="5min")
    glucose = [100, 120, 140, 160, 180, 200, 220, 150, 130, 110]
    return pd.DataFrame({TIME_COL: times, GLUCOSE_COL: glucose})


class TestAttachQualityFlags:
    """Test suite for attach_quality_flags function."""
    
    def test_adds_all_flag_columns(self, sample_data):
        """Test that all flag columns are added."""
        result = attach_quality_flags(sample_data, dominant_step_min=5.0)
        
        assert "flag_time_error" in result.columns
        assert "flag_gap_before" in result.columns
        assert "gap_before_min" in result.columns
        assert "flag_outlier_glucose" in result.columns
    
    def test_original_columns_preserved(self, sample_data):
        """Test that original columns are not modified."""
        result = attach_quality_flags(sample_data, dominant_step_min=5.0)
        
        assert TIME_COL in result.columns
        assert GLUCOSE_COL in result.columns
        pd.testing.assert_series_equal(result[TIME_COL], sample_data[TIME_COL])
        pd.testing.assert_series_equal(result[GLUCOSE_COL], sample_data[GLUCOSE_COL])
    
    def test_does_not_modify_input(self, sample_data):
        """Test that input DataFrame is not modified."""
        original_cols = sample_data.columns.tolist()
        attach_quality_flags(sample_data, dominant_step_min=5.0)
        
        assert sample_data.columns.tolist() == original_cols
    
    def test_no_rows_removed(self, sample_data):
        """Test that no rows are dropped."""
        result = attach_quality_flags(sample_data, dominant_step_min=5.0)
        assert len(result) == len(sample_data)
    
    def test_normal_sampling_no_flags(self, sample_data):
        """Test that normal 5-min sampling produces no flags."""
        result = attach_quality_flags(sample_data, dominant_step_min=5.0)
        
        # All time differences should be 5 min (except first row)
        assert result["flag_time_error"].sum() == 0
        assert result["flag_gap_before"].sum() == 0
        assert result["flag_outlier_glucose"].sum() == 0
    
    def test_time_error_detection(self):
        """Test detection of time errors (duplicate/reversed timestamps)."""
        # TIME_ERROR_RATIO = 0.5, so with 5-min sampling, errors are < 2.5 min
        data = pd.DataFrame({
            TIME_COL: [
                pd.Timestamp("2024-09-10 12:00:00"),
                pd.Timestamp("2024-09-10 12:01:00"),  # 1 min gap < 2.5 min
                pd.Timestamp("2024-09-10 12:06:00"),  # 5 min gap (normal)
            ],
            GLUCOSE_COL: [100, 110, 120]
        })
        
        result = attach_quality_flags(data, dominant_step_min=5.0)
        
        assert result["flag_time_error"].iloc[0] == False  # First row always False
        assert result["flag_time_error"].iloc[1] == True   # 1 min < 2.5 min
        assert result["flag_time_error"].iloc[2] == False  # 5 min is normal
    
    def test_gap_detection(self):
        """Test detection of gaps (missing data)."""
        # GAP_MULTIPLE = 2.0, so with 5-min sampling, gaps are > 10 min
        data = pd.DataFrame({
            TIME_COL: [
                pd.Timestamp("2024-09-10 12:00:00"),
                pd.Timestamp("2024-09-10 12:05:00"),  # 5 min (normal)
                pd.Timestamp("2024-09-10 12:20:00"),  # 15 min gap > 10 min
            ],
            GLUCOSE_COL: [100, 110, 120]
        })
        
        result = attach_quality_flags(data, dominant_step_min=5.0)
        
        assert result["flag_gap_before"].iloc[0] == False
        assert result["flag_gap_before"].iloc[1] == False  # 5 min normal
        assert result["flag_gap_before"].iloc[2] == True   # 15 min is gap
    
    def test_gap_before_min_values(self):
        """Test that gap_before_min contains correct gap durations."""
        data = pd.DataFrame({
            TIME_COL: [
                pd.Timestamp("2024-09-10 12:00:00"),
                pd.Timestamp("2024-09-10 12:05:00"),
                pd.Timestamp("2024-09-10 12:25:00"),  # 20 min gap
            ],
            GLUCOSE_COL: [100, 110, 120]
        })
        
        result = attach_quality_flags(data, dominant_step_min=5.0)
        
        assert pd.isna(result["gap_before_min"].iloc[0])
        assert pd.isna(result["gap_before_min"].iloc[1])
        assert result["gap_before_min"].iloc[2] == 20.0
    
    def test_glucose_outlier_low(self):
        """Test detection of glucose values below minimum."""
        data = pd.DataFrame({
            TIME_COL: pd.date_range("2024-09-10", periods=3, freq="5min"),
            GLUCOSE_COL: [15, 100, 50]  # 15 < GLUCOSE_MIN (20)
        })
        
        result = attach_quality_flags(data, dominant_step_min=5.0)
        
        assert result["flag_outlier_glucose"].iloc[0] == True   # 15 < 20
        assert result["flag_outlier_glucose"].iloc[1] == False  # 100 is normal
        assert result["flag_outlier_glucose"].iloc[2] == False  # 50 is normal
    
    def test_glucose_outlier_high(self):
        """Test detection of glucose values above maximum."""
        data = pd.DataFrame({
            TIME_COL: pd.date_range("2024-09-10", periods=3, freq="5min"),
            GLUCOSE_COL: [100, 650, 300]  # 650 > GLUCOSE_MAX
        })
        
        result = attach_quality_flags(data, dominant_step_min=5.0)
        
        assert result["flag_outlier_glucose"].iloc[0] == False  # 100 normal
        assert result["flag_outlier_glucose"].iloc[1] == True   # 650 > GLUCOSE_MAX
        assert result["flag_outlier_glucose"].iloc[2] == False  # 300 normal
    
    def test_glucose_outlier_boundary(self):
        """Test glucose outlier detection at exact boundaries."""
        data = pd.DataFrame({
            TIME_COL: pd.date_range("2024-09-10", periods=4, freq="5min"),
            GLUCOSE_COL: [GLUCOSE_MIN, GLUCOSE_MIN - 1, GLUCOSE_MAX, GLUCOSE_MAX + 1]  # Boundary values
        })
        
        result = attach_quality_flags(data, dominant_step_min=5.0)
        
        assert result["flag_outlier_glucose"].iloc[0] == False  # GLUCOSE_MIN is OK
        assert result["flag_outlier_glucose"].iloc[1] == True   # below GLUCOSE_MIN
        assert result["flag_outlier_glucose"].iloc[2] == False  # GLUCOSE_MAX is OK
        assert result["flag_outlier_glucose"].iloc[3] == True   # above GLUCOSE_MAX
    
    def test_multiple_flags_same_row(self):
        """Test that multiple flags can be set on same row."""
        data = pd.DataFrame({
            TIME_COL: [
                pd.Timestamp("2024-09-10 12:00:00"),
                pd.Timestamp("2024-09-10 12:25:00"),  # Gap
            ],
            GLUCOSE_COL: [100, 650]  # Outlier
        })
        
        result = attach_quality_flags(data, dominant_step_min=5.0)
        
        # Second row has both gap and outlier flags
        assert result["flag_gap_before"].iloc[1] == True
        assert result["flag_outlier_glucose"].iloc[1] == True
    
    def test_first_row_never_has_time_flags(self, sample_data):
        """Test that first row never has time-based flags."""
        result = attach_quality_flags(sample_data, dominant_step_min=5.0)
        
        assert result["flag_time_error"].iloc[0] == False
        assert result["flag_gap_before"].iloc[0] == False
        assert pd.isna(result["gap_before_min"].iloc[0])
    
    def test_different_sampling_intervals(self):
        """Test with different sampling intervals (15 min)."""
        data = pd.DataFrame({
            TIME_COL: pd.date_range("2024-09-10", periods=5, freq="15min"),
            GLUCOSE_COL: [100, 120, 140, 160, 180]
        })
        
        result = attach_quality_flags(data, dominant_step_min=15.0)
        
        # Normal 15-min sampling should produce no flags
        assert result["flag_time_error"].sum() == 0
        assert result["flag_gap_before"].sum() == 0
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        data = pd.DataFrame({TIME_COL: [], GLUCOSE_COL: []})
        data[TIME_COL] = pd.to_datetime(data[TIME_COL])
        
        result = attach_quality_flags(data, dominant_step_min=5.0)
        
        assert len(result) == 0
        assert "flag_time_error" in result.columns
    
    def test_single_row(self):
        """Test with single row."""
        data = pd.DataFrame({
            TIME_COL: [pd.Timestamp("2024-09-10 12:00:00")],
            GLUCOSE_COL: [100]
        })
        
        result = attach_quality_flags(data, dominant_step_min=5.0)
        
        assert len(result) == 1
        assert result["flag_time_error"].iloc[0] == False
        assert result["flag_gap_before"].iloc[0] == False
    
    def test_return_type(self, sample_data):
        """Test that function returns a DataFrame."""
        result = attach_quality_flags(sample_data, dominant_step_min=5.0)
        assert isinstance(result, pd.DataFrame)
    
    def test_flag_dtypes(self, sample_data):
        """Test that flags have correct data types."""
        result = attach_quality_flags(sample_data, dominant_step_min=5.0)
        
        assert result["flag_time_error"].dtype == bool
        assert result["flag_gap_before"].dtype == bool
        assert result["flag_outlier_glucose"].dtype == bool
        assert pd.api.types.is_float_dtype(result["gap_before_min"])


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_all_outliers(self):
        """Test when all glucose values are outliers."""
        data = pd.DataFrame({
            TIME_COL: pd.date_range("2024-09-10", periods=3, freq="5min"),
            GLUCOSE_COL: [10, 5, 15]  # All < 20
        })
        
        result = attach_quality_flags(data, dominant_step_min=5.0)
        assert result["flag_outlier_glucose"].all()
    
    def test_all_gaps(self):
        """Test when all intervals are gaps."""
        data = pd.DataFrame({
            TIME_COL: [
                pd.Timestamp("2024-09-10 12:00:00"),
                pd.Timestamp("2024-09-10 12:30:00"),  # 30 min gap
                pd.Timestamp("2024-09-10 13:00:00"),  # 30 min gap
            ],
            GLUCOSE_COL: [100, 110, 120]
        })
        
        result = attach_quality_flags(data, dominant_step_min=5.0)
        
        assert result["flag_gap_before"].iloc[0] == False  # First always False
        assert result["flag_gap_before"].iloc[1] == True
        assert result["flag_gap_before"].iloc[2] == True
    
    def test_very_small_dominant_step(self):
        """Test with 1-min dominant step."""
        data = pd.DataFrame({
            TIME_COL: pd.date_range("2024-09-10", periods=5, freq="1min"),
            GLUCOSE_COL: [100, 110, 120, 130, 140]
        })
        
        result = attach_quality_flags(data, dominant_step_min=1.0)
        assert result["flag_time_error"].sum() == 0
        assert result["flag_gap_before"].sum() == 0
    
    def test_realistic_mixed_scenario(self):
        """Test realistic scenario with mixed flag conditions."""
        data = pd.DataFrame({
            TIME_COL: [
                pd.Timestamp("2024-09-10 12:00:00"),
                pd.Timestamp("2024-09-10 12:05:00"),  # Normal
                pd.Timestamp("2024-09-10 12:06:00"),  # Time error (1 min)
                pd.Timestamp("2024-09-10 12:11:00"),  # Normal
                pd.Timestamp("2024-09-10 12:35:00"),  # Gap (24 min)
            ],
            GLUCOSE_COL: [100, 15, 120, 650, 200]  # Row 1: low, Row 3: high
        })
        
        result = attach_quality_flags(data, dominant_step_min=5.0)
        
        # Check time errors
        assert result["flag_time_error"].iloc[2] == True
        
        # Check gaps
        assert result["flag_gap_before"].iloc[4] == True
        assert result["gap_before_min"].iloc[4] == 24.0
        
        # Check glucose outliers
        assert result["flag_outlier_glucose"].iloc[1] == True  # 15 < 20
        assert result["flag_outlier_glucose"].iloc[3] == True  # 650 > GLUCOSE_MAX


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
