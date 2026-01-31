"""Tests for data loading and cleaning functions."""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os

from src.data_loader import load_data, clean_data


class TestLoadData:
    def test_load_data_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_data("non_existent_file.csv")
    
    def test_load_data_invalid_extension(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as f:
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="must be a CSV"):
                load_data(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_data_success(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\n1,2\n3,4\n")
            temp_path = f.name
        
        try:
            df = load_data(temp_path)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert list(df.columns) == ['col1', 'col2']
        finally:
            os.unlink(temp_path)


class TestCleanData:
    def test_clean_data_fills_nan(self):
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [10, np.nan, 30, 40]
        })
        cleaned = clean_data(df)
        assert cleaned.isna().sum().sum() == 0
    
    def test_clean_data_preserves_complete_data(self):
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        cleaned = clean_data(df)
        pd.testing.assert_frame_equal(df, cleaned)
    
    def test_clean_data_fills_with_mean(self):
        df = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0]  # mean = 7/3 ≈ 2.33
        })
        cleaned = clean_data(df)
        expected_mean = (1 + 2 + 4) / 3
        assert cleaned['A'].iloc[2] == pytest.approx(expected_mean)
