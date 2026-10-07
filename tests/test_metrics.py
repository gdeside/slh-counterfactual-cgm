"""Tests for src/metrics/metrics.py - CGM metrics computation functions."""

import pytest
import pandas as pd

from src.metrics.metrics import (
    classify_time_of_day,
    create_empty_metrics,
    compute_basic_statistics,
    compute_period_medians,
    compute_time_based_metrics,
    compute_time_in_range_metrics,
    ALL_METRIC_NAMES,
)


TIME_COL = "time"
GLUCOSE_COL = "glycemie"


@pytest.fixture
def sample_glucose_values():
    """Sample glucose values for testing."""
    return pd.Series([100, 120, 140, 160, 180, 200, 220, 150, 130, 110])


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame with time and glucose."""
    times = pd.date_range("2024-09-10 08:00:00", periods=24, freq="1h")
    glucose = [100, 120, 140, 160, 180, 200, 220, 180, 160, 140, 
               120, 100, 110, 130, 150, 170, 190, 180, 160, 140, 
               120, 100, 90, 80]
    return pd.DataFrame({TIME_COL: times, GLUCOSE_COL: glucose})


class TestClassifyTimeOfDay:
    """Test time-of-day classification."""
    
    def test_morning_period(self):
        """Test morning classification (06:30 - 11:30)."""
        assert classify_time_of_day(pd.Timestamp("2024-09-10 07:00:00")) == "morning"
        assert classify_time_of_day(pd.Timestamp("2024-09-10 10:00:00")) == "morning"
    
    def test_noon_period(self):
        """Test noon classification (11:30 - 15:30)."""
        assert classify_time_of_day(pd.Timestamp("2024-09-10 12:00:00")) == "noon"
        assert classify_time_of_day(pd.Timestamp("2024-09-10 14:00:00")) == "noon"
    
    def test_afternoon_period(self):
        """Test afternoon classification (15:30 - 18:30)."""
        assert classify_time_of_day(pd.Timestamp("2024-09-10 16:00:00")) == "afternoon"
        assert classify_time_of_day(pd.Timestamp("2024-09-10 18:00:00")) == "afternoon"
    
    def test_evening_period(self):
        """Test evening classification (18:30 - 22:30)."""
        assert classify_time_of_day(pd.Timestamp("2024-09-10 19:00:00")) == "evening"
        assert classify_time_of_day(pd.Timestamp("2024-09-10 22:00:00")) == "evening"
    
    def test_night_period(self):
        """Test night classification (22:30 - 06:30)."""
        assert classify_time_of_day(pd.Timestamp("2024-09-10 23:00:00")) == "night"
        assert classify_time_of_day(pd.Timestamp("2024-09-10 02:00:00")) == "night"
        assert classify_time_of_day(pd.Timestamp("2024-09-10 05:00:00")) == "night"
    
    def test_boundary_times(self):
        """Test exact boundary times."""
        assert classify_time_of_day(pd.Timestamp("2024-09-10 06:30:00")) == "morning"
        assert classify_time_of_day(pd.Timestamp("2024-09-10 11:30:00")) == "noon"
        assert classify_time_of_day(pd.Timestamp("2024-09-10 15:30:00")) == "afternoon"
        assert classify_time_of_day(pd.Timestamp("2024-09-10 18:30:00")) == "evening"
        assert classify_time_of_day(pd.Timestamp("2024-09-10 22:30:00")) == "night"


class TestCreateEmptyMetrics:
    """Test empty metrics creation."""
    
    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        result = create_empty_metrics()
        assert isinstance(result, dict)
    
    def test_has_all_metrics(self):
        """Test that all metric names are present."""
        result = create_empty_metrics()
        assert len(result) == len(ALL_METRIC_NAMES)
        for metric_name in ALL_METRIC_NAMES:
            assert metric_name in result
    
    def test_all_values_nan(self):
        """Test that all values are NaN."""
        result = create_empty_metrics()
        for value in result.values():
            assert pd.isna(value)


class TestComputeBasicStatistics:
    """Test basic statistics computation."""
    
    def test_mean_glucose(self, sample_glucose_values):
        """Test mean glucose calculation."""
        result = compute_basic_statistics(sample_glucose_values)
        expected_mean = sample_glucose_values.mean()
        assert result["Mean_glycemia"] == round(expected_mean, 1)
    
    def test_median_glucose(self, sample_glucose_values):
        """Test median glucose calculation."""
        result = compute_basic_statistics(sample_glucose_values)
        expected_median = sample_glucose_values.median()
        assert result["Median_glycemia"] == round(expected_median, 1)
    
    def test_min_max_glucose(self, sample_glucose_values):
        """Test min and max glucose."""
        result = compute_basic_statistics(sample_glucose_values)
        assert result["Minimum_glycemia"] == 100.0
        assert result["Maximum_glycemia"] == 220.0
    
    def test_standard_deviation(self, sample_glucose_values):
        """Test SD calculation."""
        result = compute_basic_statistics(sample_glucose_values)
        expected_sd = sample_glucose_values.std()
        assert result["SD"] == round(expected_sd, 1)
    
    def test_coefficient_of_variation(self, sample_glucose_values):
        """Test CV calculation."""
        result = compute_basic_statistics(sample_glucose_values)
        mean = sample_glucose_values.mean()
        sd = sample_glucose_values.std()
        expected_cv = sd / mean
        assert result["CV"] == round(expected_cv, 3)
    
    def test_all_metrics_present(self, sample_glucose_values):
        """Test that all basic statistics are returned."""
        result = compute_basic_statistics(sample_glucose_values)
        assert "Mean_glycemia" in result
        assert "Median_glycemia" in result
        assert "Minimum_glycemia" in result
        assert "Maximum_glycemia" in result
        assert "SD" in result
        assert "CV" in result
    
    def test_cv_zero_mean(self):
        """Test CV when mean is zero."""
        glucose = pd.Series([0, 0, 0])
        result = compute_basic_statistics(glucose)
        assert pd.isna(result["CV"])


class TestComputePeriodMedians:
    """Test period median computation."""
    
    def test_all_periods_present(self, sample_dataframe):
        """Test that all time periods are in result."""
        result = compute_period_medians(sample_dataframe)
        assert "Median_morning" in result
        assert "Median_noon" in result
        assert "Median_afternoon" in result
        assert "Median_evening" in result
        assert "Median_night" in result
    
    def test_median_calculation(self):
        """Test median calculation for specific period."""
        # Create data with known values in morning
        times = [
            pd.Timestamp("2024-09-10 07:00:00"),
            pd.Timestamp("2024-09-10 08:00:00"),
            pd.Timestamp("2024-09-10 09:00:00"),
        ]
        glucose = [100, 120, 140]
        df = pd.DataFrame({TIME_COL: times, GLUCOSE_COL: glucose})
        
        result = compute_period_medians(df)
        assert result["Median_morning"] == 120.0  # median of [100, 120, 140]
    
    def test_missing_period_returns_nan(self):
        """Test that missing periods return NaN."""
        # Only morning data
        times = [pd.Timestamp("2024-09-10 07:00:00")]
        glucose = [100]
        df = pd.DataFrame({TIME_COL: times, GLUCOSE_COL: glucose})
        
        result = compute_period_medians(df)
        assert pd.isna(result["Median_noon"])
    
    def test_original_dataframe_not_modified(self, sample_dataframe):
        """Test that original DataFrame is not modified."""
        original_cols = sample_dataframe.columns.tolist()
        compute_period_medians(sample_dataframe)
        assert sample_dataframe.columns.tolist() == original_cols


class TestComputeTimeBasedMetrics:
    """Test time-based metrics computation."""
    
    def test_single_day(self):
        """Test CGM days with single day."""
        times = pd.date_range("2024-09-10 00:00:00", periods=24, freq="1h")
        glucose = [100] * 24
        df = pd.DataFrame({TIME_COL: times, GLUCOSE_COL: glucose})
        
        result = compute_time_based_metrics(df)
        assert result["CGM_days_measured"] == 1
    
    def test_multiple_days(self):
        """Test CGM days with multiple days."""
        times = pd.date_range("2024-09-10", periods=72, freq="1h")  # 3 days
        glucose = [100] * 72
        df = pd.DataFrame({TIME_COL: times, GLUCOSE_COL: glucose})
        
        result = compute_time_based_metrics(df)
        assert result["CGM_days_measured"] == 3
    
    def test_partial_days(self):
        """Test CGM days with partial days."""
        times = [
            pd.Timestamp("2024-09-10 08:00:00"),
            pd.Timestamp("2024-09-10 12:00:00"),
            pd.Timestamp("2024-09-11 10:00:00"),
        ]
        glucose = [100, 120, 140]
        df = pd.DataFrame({TIME_COL: times, GLUCOSE_COL: glucose})
        
        result = compute_time_based_metrics(df)
        assert result["CGM_days_measured"] == 2  # 2 unique dates
    
    def test_return_type_is_int(self, sample_dataframe):
        """Test that result is integer."""
        result = compute_time_based_metrics(sample_dataframe)
        assert isinstance(result["CGM_days_measured"], int)


class TestComputeTimeInRangeMetrics:
    """Test TIR/TBR/TAR metrics computation."""
    
    def test_tir_70_180_all_in_range(self):
        """Test TIR when all values in range."""
        glucose = pd.Series([100, 120, 140, 160, 180])
        result = compute_time_in_range_metrics(glucose)
        assert result["TIR70180"] == 1.0
    
    def test_tir_70_180_none_in_range(self):
        """Test TIR when no values in range."""
        glucose = pd.Series([50, 60, 65, 200, 250])
        result = compute_time_in_range_metrics(glucose)
        assert result["TIR70180"] == 0.0
    
    def test_tir_70_180_partial(self):
        """Test TIR with partial values in range."""
        glucose = pd.Series([50, 100, 150, 200, 250])  # 2/5 in range
        result = compute_time_in_range_metrics(glucose)
        assert result["TIR70180"] == 0.4
    
    def test_tbr_70_detection(self):
        """Test TBR <70 detection."""
        glucose = pd.Series([50, 60, 100, 150, 200])  # 2/5 below 70
        result = compute_time_in_range_metrics(glucose)
        assert result["TBR70"] == 0.4
    
    def test_tbr_54_detection(self):
        """Test TBR <54 detection."""
        glucose = pd.Series([40, 50, 100, 150, 200])  # 2/5 below 54 (40 and 50)
        result = compute_time_in_range_metrics(glucose)
        assert result["TBR54"] == 0.4
    
    def test_tar_180_detection(self):
        """Test TAR >180 detection."""
        glucose = pd.Series([100, 150, 200, 250, 300])  # 3/5 above 180
        result = compute_time_in_range_metrics(glucose)
        assert result["TAR180"] == 0.6
    
    def test_tar_250_detection(self):
        """Test TAR >250 detection."""
        glucose = pd.Series([100, 200, 260, 300, 350])  # 3/5 above 250
        result = compute_time_in_range_metrics(glucose)
        assert result["TAR250"] == 0.6
    
    def test_tir_70_140_tight_range(self):
        """Test tight TIR 70-140."""
        glucose = pd.Series([60, 100, 120, 160, 200])  # 2/5 in 70-140
        result = compute_time_in_range_metrics(glucose)
        assert result["TIR70140"] == 0.4
    
    def test_tir_60_140_relaxed_range(self):
        """Test relaxed TIR 60-140."""
        glucose = pd.Series([50, 80, 120, 160, 200])  # 2/5 in 60-140
        result = compute_time_in_range_metrics(glucose)
        assert result["TIR60140"] == 0.4
    
    def test_all_metrics_present(self, sample_glucose_values):
        """Test that all TIR metrics are returned."""
        result = compute_time_in_range_metrics(sample_glucose_values)
        assert "TIR70180" in result
        assert "TIR70140" in result
        assert "TIR60140" in result
        assert "TBR70" in result
        assert "TBR54" in result
        assert "TAR180" in result
        assert "TAR250" in result
    
    def test_fractions_rounded_to_3_decimals(self):
        """Test that fractions are rounded to 3 decimals."""
        glucose = pd.Series([100, 120, 140])  # 3/3 = 1.0
        result = compute_time_in_range_metrics(glucose)
        for metric in ["TIR70180", "TIR70140", "TIR60140", "TBR70", "TBR54", "TAR180", "TAR250"]:
            # Check that value has at most 3 decimal places
            value = result[metric]
            assert value == round(value, 3)
    
    def test_boundary_values_70_180(self):
        """Test boundary values for TIR 70-180."""
        glucose = pd.Series([70, 180])  # Both boundaries included
        result = compute_time_in_range_metrics(glucose)
        assert result["TIR70180"] == 1.0
    
    def test_realistic_scenario(self):
        """Test with realistic glucose distribution."""
        # 10 values: 2 low, 5 normal, 3 high
        glucose = pd.Series([60, 65, 100, 120, 140, 160, 170, 200, 220, 250])
        result = compute_time_in_range_metrics(glucose)
        
        assert result["TIR70180"] == 0.5  # 5/10 in range
        assert result["TBR70"] == 0.2     # 2/10 below 70
        assert result["TAR180"] == 0.3    # 3/10 above 180


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_complete_metrics_pipeline(self):
        """Test computing all metrics together."""
        times = pd.date_range("2024-09-10 00:00:00", periods=48, freq="30min")
        glucose = [100 + i * 2 for i in range(48)]
        df = pd.DataFrame({TIME_COL: times, GLUCOSE_COL: glucose})
        
        # Compute all metrics
        basic_stats = compute_basic_statistics(df[GLUCOSE_COL])
        tir_metrics = compute_time_in_range_metrics(df[GLUCOSE_COL])
        time_metrics = compute_time_based_metrics(df)
        period_metrics = compute_period_medians(df)
        
        # Verify all metrics computed
        assert len(basic_stats) == 6
        assert len(tir_metrics) == 7
        assert len(time_metrics) == 1
        assert len(period_metrics) == 5
        
        # Total should be 19 metrics
        total_metrics = {**basic_stats, **tir_metrics, **time_metrics, **period_metrics}
        assert len(total_metrics) == 19


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_value(self):
        """Test with single glucose value."""
        glucose = pd.Series([100])
        result = compute_basic_statistics(glucose)
        assert result["Mean_glycemia"] == 100.0
        assert pd.isna(result["SD"])  # SD is NaN for single value (ddof=1)
    
    def test_all_same_values(self):
        """Test with all identical values."""
        glucose = pd.Series([150] * 10)
        result = compute_basic_statistics(glucose)
        assert result["Mean_glycemia"] == 150.0
        assert result["SD"] == 0.0
        assert result["CV"] == 0.0
    
    def test_extreme_values(self):
        """Test with extreme glucose values."""
        glucose = pd.Series([20, 600])
        result = compute_basic_statistics(glucose)
        assert result["Minimum_glycemia"] == 20.0
        assert result["Maximum_glycemia"] == 600.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])