"""
Tests for src/preprocessing/load_data.py

Tests data loading functions for raw and preprocessed CGM files.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Import functions to test
from src.preprocessing.load_data import (
    load_raw_time_column,
    load_patient_raw_data,
    load_window_data
)

# Import config constants
from src.config import RAW_PATH, PREPROCESSING_PATH


# ==============================================================================
# Test Configuration
# ==============================================================================

# Column names (match your config.py)
TIME_COL = "time"
GLUCOSE_COL = "glycemie"


# ==============================================================================
# Fixtures - Test Data Creation
# ==============================================================================

@pytest.fixture
def temp_raw_dir(tmp_path):
    """Create temporary directory for raw data files."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    return raw_dir


@pytest.fixture
def temp_prep_dir(tmp_path):
    """Create temporary directory for preprocessed data files."""
    prep_dir = tmp_path / "preprocessed"
    prep_dir.mkdir()
    return prep_dir


@pytest.fixture
def sample_raw_data():
    """Create sample raw CGM data matching A001 format."""
    # Based on the user's actual A001 data
    times = [
        "10/09/2024 23:54",
        "10/09/2024 23:59",
        "11/09/2024 00:04",
        "11/09/2024 00:09",
        "11/09/2024 00:14",
        "11/09/2024 00:19",
        "11/09/2024 00:24",
        "11/09/2024 00:29",
        "11/09/2024 00:34",
        "11/09/2024 00:39",
        "11/09/2024 00:44",
        "11/09/2024 00:49",
        "11/09/2024 00:54",
        "11/09/2024 00:59",
        "11/09/2024 01:04",
        "11/09/2024 01:09",
    ]
    glucose = [253, 248, 249, 237, 225, 221, 210, 198, 191, 185, 182, 180, 179, 181, 182, 184]
    
    return pd.DataFrame({
        "time": times,
        "glycemie": glucose
    })


@pytest.fixture
def sample_preprocessed_data():
    """Create sample preprocessed CGM data with quality flags."""
    times = pd.date_range("2024-09-10 23:54:00", periods=16, freq="5min")
    glucose = [253, 248, 249, 237, 225, 221, 210, 198, 191, 185, 182, 180, 179, 181, 182, 184]
    
    return pd.DataFrame({
        TIME_COL: times,
        GLUCOSE_COL: glucose,
        "flag_time_error": [False] * 16,
        "flag_gap_before": [False] * 16,
        "gap_before_min": [np.nan] * 16,
        "flag_outlier_glucose": [False] * 16,
    })


@pytest.fixture
def create_raw_excel(temp_raw_dir, sample_raw_data):
    """Factory fixture to create raw Excel files."""
    def _create_file(patient_id, data=None, columns=None):
        if data is None:
            data = sample_raw_data.copy()
        else:
            data = data.copy()
        if columns is not None:
            data.columns = columns
        
        filepath = temp_raw_dir / f"{patient_id}.xlsx"
        data.to_excel(filepath, index=False)
        return filepath
    
    return _create_file


@pytest.fixture
def create_prep_excel(temp_prep_dir, sample_preprocessed_data):
    """Factory fixture to create preprocessed Excel files."""
    def _create_file(filename, data=None, columns=None):
        if data is None:
            data = sample_preprocessed_data.copy()
        else:
            data = data.copy()
        if columns is not None:
            data.columns = columns
        
        filepath = temp_prep_dir / filename
        data.to_excel(filepath, index=False)
        return filepath
    
    return _create_file


# ==============================================================================
# Tests for load_raw_time_column
# ==============================================================================

class TestLoadRawTimeColumn:
    """Test suite for load_raw_time_column function."""
    
    def test_successful_load_standard_case(self, create_raw_excel, temp_raw_dir, monkeypatch):
        """Test loading time column from standard raw file."""
        create_raw_excel("A001")
        
        # Patch RAW_PATH in the module where it's used
        import src.preprocessing.load_data
        monkeypatch.setattr(src.preprocessing.load_data, "RAW_PATH", temp_raw_dir)
        
        result = load_raw_time_column("A001")
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert TIME_COL in result.columns
        assert len(result.columns) == 1
        assert len(result) == 16
        assert pd.api.types.is_datetime64_any_dtype(result[TIME_COL])
    
    def test_time_column_parsed_correctly(self, create_raw_excel, temp_raw_dir, monkeypatch):
        """Test that DD/MM/YYYY format is parsed correctly."""
        create_raw_excel("A001")
        
        import src.preprocessing.load_data
        monkeypatch.setattr(src.preprocessing.load_data, "RAW_PATH", temp_raw_dir)
        
        result = load_raw_time_column("A001")
        
        # First timestamp should be 10/09/2024 23:54
        expected_first = pd.Timestamp("2024-09-10 23:54:00")
        assert result[TIME_COL].iloc[0] == expected_first
    
    def test_missing_file_returns_none(self, temp_raw_dir, monkeypatch):
        """Test that missing file returns None."""
        import src.preprocessing.load_data
        monkeypatch.setattr(src.preprocessing.load_data, "RAW_PATH", temp_raw_dir)
        
        result = load_raw_time_column("A246")  # Non-existent patient
        
        assert result is None
    
    def test_case_insensitive_time_column(self, create_raw_excel, temp_raw_dir, monkeypatch, sample_raw_data):
        """Test that 'Time', 'TIME', 'time' all work."""
        import src.preprocessing.load_data
        monkeypatch.setattr(src.preprocessing.load_data, "RAW_PATH", temp_raw_dir)
        
        for col_name in ["Time", "TIME", "time", "TiMe"]:
            create_raw_excel("A100", columns=[col_name, "glycemie"])
            result = load_raw_time_column("A100")
            
            assert result is not None
            assert TIME_COL in result.columns
    
    def test_missing_time_column_returns_none(self, create_raw_excel, temp_raw_dir, monkeypatch, sample_raw_data):
        """Test file without 'time' column returns None."""
        data = sample_raw_data.copy()
        data.columns = ["timestamp", "glycemie"]  # Wrong column name
        
        create_raw_excel("A999", data=data, columns=["timestamp", "glycemie"])
        
        import src.preprocessing.load_data
        monkeypatch.setattr(src.preprocessing.load_data, "RAW_PATH", temp_raw_dir)
        
        result = load_raw_time_column("A999")
        
        assert result is None
    
    def test_invalid_timestamps_dropped(self, create_raw_excel, temp_raw_dir, monkeypatch):
        """Test that invalid timestamps are dropped."""
        data = pd.DataFrame({
            "time": ["10/09/2024 23:54", "invalid_date", "11/09/2024 00:04"],
            "glycemie": [253, 248, 249]
        })
        
        create_raw_excel("A050", data=data)
        
        import src.preprocessing.load_data
        monkeypatch.setattr(src.preprocessing.load_data, "RAW_PATH", temp_raw_dir)
        
        result = load_raw_time_column("A050")
        
        assert result is not None
        assert len(result) == 2  # Invalid row dropped
    
    def test_sorted_by_time(self, create_raw_excel, temp_raw_dir, monkeypatch):
        """Test that returned data is sorted by time."""
        # Create unsorted data
        data = pd.DataFrame({
            "time": [
                "11/09/2024 01:00",
                "10/09/2024 23:00",
                "11/09/2024 00:30",
            ],
            "glycemie": [100, 200, 150]
        })
        
        create_raw_excel("A075", data=data)
        
        import src.preprocessing.load_data
        monkeypatch.setattr(src.preprocessing.load_data, "RAW_PATH", temp_raw_dir)
        
        result = load_raw_time_column("A075")
        
        assert result is not None
        assert result[TIME_COL].is_monotonic_increasing
    
    def test_corrupted_file_returns_none(self, temp_raw_dir, monkeypatch):
        """Test that corrupted Excel file returns None."""
        # Create invalid Excel file
        corrupted_file = temp_raw_dir / "A888.xlsx"
        corrupted_file.write_text("This is not an Excel file")
        
        import src.preprocessing.load_data
        monkeypatch.setattr(src.preprocessing.load_data, "RAW_PATH", temp_raw_dir)
        
        result = load_raw_time_column("A888")
        
        assert result is None


# ==============================================================================
# Tests for load_patient_raw_data
# ==============================================================================

class TestLoadPatientRawData:
    """Test suite for load_patient_raw_data function."""
    
    def test_successful_load_complete_data(self, create_raw_excel, temp_raw_dir):
        """Test successful loading of time and glucose columns."""
        create_raw_excel("A001")
        
        result = load_patient_raw_data("A001", str(temp_raw_dir))
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == [TIME_COL, GLUCOSE_COL]
        assert len(result) == 16
        assert pd.api.types.is_datetime64_any_dtype(result[TIME_COL])
        assert pd.api.types.is_numeric_dtype(result[GLUCOSE_COL])
    
    def test_glucose_values_correct(self, create_raw_excel, temp_raw_dir):
        """Test that glucose values are loaded correctly."""
        create_raw_excel("A001")
        
        result = load_patient_raw_data("A001", str(temp_raw_dir))
        
        # First few glucose values from sample data
        expected_glucose = [253, 248, 249, 237, 225]
        assert list(result[GLUCOSE_COL].head(5)) == expected_glucose
    
    def test_missing_file_returns_none(self, temp_raw_dir):
        """Test that non-existent patient returns None."""
        result = load_patient_raw_data("A246", str(temp_raw_dir))
        assert result is None
    
    def test_case_insensitive_columns(self, create_raw_excel, temp_raw_dir, sample_raw_data):
        """Test case-insensitive column detection."""
        test_cases = [
            (["Time", "Glycemie"], "A100"),
            (["TIME", "GLYCEMIE"], "A101"),
            (["time", "glucose"], "A102"),
            (["Time", "Glucose"], "A103"),
        ]
        
        for columns, patient_id in test_cases:
            create_raw_excel(patient_id, columns=columns)
            result = load_patient_raw_data(patient_id, str(temp_raw_dir))
            
            assert result is not None, f"Failed for columns: {columns}"
            assert TIME_COL in result.columns
            assert GLUCOSE_COL in result.columns
    
    def test_missing_time_column_returns_none(self, create_raw_excel, temp_raw_dir, sample_raw_data):
        """Test file missing time column returns None."""
        data = sample_raw_data.copy()
        data.columns = ["timestamp", "glycemie"]  # Wrong time column name
        
        create_raw_excel("A200", data=data, columns=["timestamp", "glycemie"])
        result = load_patient_raw_data("A200", str(temp_raw_dir))
        
        assert result is None
    
    def test_missing_glucose_column_returns_none(self, create_raw_excel, temp_raw_dir, sample_raw_data):
        """Test file missing glucose column returns None."""
        data = sample_raw_data.copy()
        data.columns = ["time", "blood_sugar"]  # Wrong glucose column name
        
        create_raw_excel("A201", data=data, columns=["time", "blood_sugar"])
        result = load_patient_raw_data("A201", str(temp_raw_dir))
        
        assert result is None
    
    def test_invalid_glucose_values_dropped(self, create_raw_excel, temp_raw_dir):
        """Test that non-numeric glucose values are dropped."""
        data = pd.DataFrame({
            "time": ["10/09/2024 23:54", "10/09/2024 23:59", "11/09/2024 00:04"],
            "glycemie": [253, "invalid", 249]
        })
        
        create_raw_excel("A150", data=data)
        result = load_patient_raw_data("A150", str(temp_raw_dir))
        
        assert result is not None
        assert len(result) == 2  # Invalid row dropped
    
    def test_rows_with_nan_dropped(self, create_raw_excel, temp_raw_dir):
        """Test that rows with NaN in time or glucose are dropped."""
        data = pd.DataFrame({
            "time": ["10/09/2024 23:54", None, "11/09/2024 00:04", "11/09/2024 00:09"],
            "glycemie": [253, 248, None, 237]
        })
        
        create_raw_excel("A151", data=data)
        result = load_patient_raw_data("A151", str(temp_raw_dir))
        
        assert result is not None
        assert len(result) == 2  # Only rows without NaN
        assert result[GLUCOSE_COL].isna().sum() == 0
        assert result[TIME_COL].isna().sum() == 0
    
    def test_sorted_by_time(self, create_raw_excel, temp_raw_dir):
        """Test output is sorted by timestamp."""
        data = pd.DataFrame({
            "time": ["11/09/2024 02:00", "10/09/2024 22:00", "11/09/2024 00:00"],
            "glycemie": [100, 200, 150]
        })
        
        create_raw_excel("A160", data=data)
        result = load_patient_raw_data("A160", str(temp_raw_dir))
        
        assert result is not None
        assert result[TIME_COL].is_monotonic_increasing
    
    def test_index_reset(self, create_raw_excel, temp_raw_dir):
        """Test that index is reset to 0, 1, 2, ..."""
        create_raw_excel("A001")
        result = load_patient_raw_data("A001", str(temp_raw_dir))
        
        assert result is not None
        assert list(result.index) == list(range(len(result)))
    
    def test_only_time_and_glucose_columns(self, create_raw_excel, temp_raw_dir):
        """Test that only time and glucose columns are returned."""
        # Create file with extra columns
        data = pd.DataFrame({
            "time": ["10/09/2024 23:54", "10/09/2024 23:59"],
            "glycemie": [253, 248],
            "extra_col1": [1, 2],
            "extra_col2": ["a", "b"]
        })
        
        create_raw_excel("A170", data=data)
        result = load_patient_raw_data("A170", str(temp_raw_dir))
        
        assert result is not None
        assert list(result.columns) == [TIME_COL, GLUCOSE_COL]
    
    def test_corrupted_file_returns_none(self, temp_raw_dir):
        """Test that corrupted file returns None."""
        corrupted_file = temp_raw_dir / "A999.xlsx"
        corrupted_file.write_text("Not a valid Excel file")
        
        result = load_patient_raw_data("A999", str(temp_raw_dir))
        assert result is None


# ==============================================================================
# Tests for load_window_data
# ==============================================================================

class TestLoadWindowData:
    """Test suite for load_window_data function."""
    
    def test_successful_load_preprocessed(self, create_prep_excel):
        """Test successful loading of preprocessed window data."""
        filepath = create_prep_excel("A001_win1_test.xlsx")
        
        result = load_window_data(str(filepath))
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert TIME_COL in result.columns
        assert GLUCOSE_COL in result.columns
    
    def test_all_columns_preserved(self, create_prep_excel, sample_preprocessed_data):
        """Test that all columns including flags are preserved."""
        filepath = create_prep_excel("A001_win1_test.xlsx")
        
        result = load_window_data(str(filepath))
        
        expected_cols = [
            TIME_COL, GLUCOSE_COL, 
            "flag_time_error", "flag_gap_before", 
            "gap_before_min", "flag_outlier_glucose"
        ]
        for col in expected_cols:
            assert col in result.columns
    
    def test_case_insensitive_column_names(self, create_prep_excel, sample_preprocessed_data):
        """Test case-insensitive column normalization."""
        data = sample_preprocessed_data.copy()
        
        # Test different case variations
        test_cases = [
            ["Time", "Glycemie", "flag_time_error", "flag_gap_before", "gap_before_min", "flag_outlier_glucose"],
            ["TIME", "GLYCEMIE", "flag_time_error", "flag_gap_before", "gap_before_min", "flag_outlier_glucose"],
            ["time", "glucose", "flag_time_error", "flag_gap_before", "gap_before_min", "flag_outlier_glucose"],
        ]
        
        for i, columns in enumerate(test_cases):
            filepath = create_prep_excel(f"test_{i}.xlsx", columns=columns)
            result = load_window_data(str(filepath))
            
            assert result is not None
            assert TIME_COL in result.columns
            assert GLUCOSE_COL in result.columns
    
    def test_alternative_glucose_column_names(self, create_prep_excel, sample_preprocessed_data):
        """Test that 'glycemia' and 'glucose' are recognized."""
        for gluc_col in ["glycemia", "glycemie", "glucose"]:
            data = sample_preprocessed_data.copy()
            columns = [TIME_COL, gluc_col, "flag_time_error", "flag_gap_before", 
                      "gap_before_min", "flag_outlier_glucose"]
            
            filepath = create_prep_excel(f"test_{gluc_col}.xlsx", columns=columns)
            result = load_window_data(str(filepath))
            
            assert result is not None
            assert GLUCOSE_COL in result.columns
    
    def test_missing_time_column_returns_none(self, create_prep_excel, sample_preprocessed_data):
        """Test file without time column returns None."""
        data = sample_preprocessed_data.copy()
        data.columns = ["timestamp", GLUCOSE_COL, "flag_time_error", "flag_gap_before", 
                       "gap_before_min", "flag_outlier_glucose"]
        
        filepath = create_prep_excel("missing_time.xlsx", data=data, 
                                     columns=["timestamp", GLUCOSE_COL, "flag_time_error", 
                                            "flag_gap_before", "gap_before_min", "flag_outlier_glucose"])
        
        result = load_window_data(str(filepath))
        assert result is None
    
    def test_missing_glucose_column_returns_none(self, create_prep_excel, sample_preprocessed_data):
        """Test file without glucose column returns None."""
        data = sample_preprocessed_data.copy()
        data.columns = [TIME_COL, "blood_sugar", "flag_time_error", "flag_gap_before", 
                       "gap_before_min", "flag_outlier_glucose"]
        
        filepath = create_prep_excel("missing_glucose.xlsx", data=data,
                                     columns=[TIME_COL, "blood_sugar", "flag_time_error", 
                                            "flag_gap_before", "gap_before_min", "flag_outlier_glucose"])
        
        result = load_window_data(str(filepath))
        assert result is None
    
    def test_missing_file_returns_none(self, temp_prep_dir):
        """Test non-existent file returns None."""
        fake_path = temp_prep_dir / "nonexistent.xlsx"
        result = load_window_data(str(fake_path))
        assert result is None
    
    def test_corrupted_file_returns_none(self, temp_prep_dir):
        """Test corrupted Excel file returns None."""
        corrupted = temp_prep_dir / "corrupted.xlsx"
        corrupted.write_text("Invalid Excel data")
        
        result = load_window_data(str(corrupted))
        assert result is None
    
    def test_minimal_valid_file(self, temp_prep_dir):
        """Test loading minimal file with just time and glucose."""
        data = pd.DataFrame({
            TIME_COL: pd.date_range("2024-09-10", periods=5, freq="5min"),
            GLUCOSE_COL: [100, 110, 120, 130, 140]
        })
        
        filepath = temp_prep_dir / "minimal.xlsx"
        data.to_excel(filepath, index=False)
        
        result = load_window_data(str(filepath))
        
        assert result is not None
        assert len(result) == 5
        assert TIME_COL in result.columns
        assert GLUCOSE_COL in result.columns


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestIntegration:
    """Integration tests across multiple functions."""
    
    def test_raw_to_preprocessed_column_consistency(self, create_raw_excel, 
                                                      create_prep_excel, 
                                                      temp_raw_dir):
        """Test that raw and preprocessed loading use consistent column names."""
        create_raw_excel("A001")
        prep_file = create_prep_excel("A001_win1.xlsx")
        
        raw_result = load_patient_raw_data("A001", str(temp_raw_dir))
        prep_result = load_window_data(str(prep_file))
        
        assert raw_result is not None
        assert prep_result is not None
        
        # Both should have same core column names
        assert TIME_COL in raw_result.columns
        assert TIME_COL in prep_result.columns
        assert GLUCOSE_COL in raw_result.columns
        assert GLUCOSE_COL in prep_result.columns
    
    def test_realistic_workflow(self, create_raw_excel, create_prep_excel, temp_raw_dir, monkeypatch):
        """Test realistic workflow: load raw, then load preprocessed."""
        # Create test files
        create_raw_excel("A001")
        prep_file = create_prep_excel("A001_win1.xlsx")
        
        # Patch RAW_PATH for load_raw_time_column
        import src.preprocessing.load_data
        monkeypatch.setattr(src.preprocessing.load_data, "RAW_PATH", temp_raw_dir)
        
        # Step 1: Load raw data
        raw_data = load_patient_raw_data("A001", str(temp_raw_dir))
        assert raw_data is not None
        assert len(raw_data) == 16
        
        # Step 2: Load time column only
        time_data = load_raw_time_column("A001")
        assert time_data is not None
        assert len(time_data) == 16
        
        # Step 3: Load preprocessed data
        prep_data = load_window_data(str(prep_file))
        assert prep_data is not None
        assert len(prep_data) == 16
        
        # Verify time consistency
        assert raw_data[TIME_COL].iloc[0] == prep_data[TIME_COL].iloc[0]


# ==============================================================================
# Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_excel_file(self, temp_raw_dir):
        """Test completely empty Excel file."""
        empty_file = temp_raw_dir / "A500.xlsx"
        pd.DataFrame().to_excel(empty_file, index=False)
        
        result = load_patient_raw_data("A500", str(temp_raw_dir))
        assert result is None
    
    def test_single_row_file(self, create_raw_excel, temp_raw_dir):
        """Test file with only one data row."""
        data = pd.DataFrame({
            "time": ["10/09/2024 23:54"],
            "glycemie": [253]
        })
        
        create_raw_excel("A501", data=data)
        result = load_patient_raw_data("A501", str(temp_raw_dir))
        
        assert result is not None
        assert len(result) == 1
        assert result[GLUCOSE_COL].iloc[0] == 253
    
    def test_very_large_file(self, temp_raw_dir):
        """Test loading file with many rows."""
        # Create 10,000 rows (~ 1 month of 5-min data)
        times = pd.date_range("2024-09-01", periods=10000, freq="5min")
        data = pd.DataFrame({
            "time": times.strftime("%d/%m/%Y %H:%M"),
            "glycemie": np.random.randint(70, 250, size=10000)
        })
        
        filepath = temp_raw_dir / "A600.xlsx"
        data.to_excel(filepath, index=False)
        
        result = load_patient_raw_data("A600", str(temp_raw_dir))
        
        assert result is not None
        assert len(result) == 10000
        assert result[TIME_COL].is_monotonic_increasing


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])