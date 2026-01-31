"""Tests for utility functions."""
import pytest
import pandas as pd
from datetime import datetime, timedelta

from src.utils import format_currency, format_percentage, get_status_color, generate_borrower_history


class TestFormatters:
    def test_format_currency(self):
        assert format_currency(1000) == "$1,000"
        assert format_currency(1234567) == "$1,234,567"
        assert format_currency(0) == "$0"
    
    def test_format_percentage(self):
        assert format_percentage(50.0) == "50.0%"
        assert format_percentage(99.9) == "99.9%"
        assert format_percentage(0.0) == "0.0%"


class TestStatusColor:
    def test_fully_recovered_green(self):
        assert get_status_color('Fully Recovered') == 'green'
    
    def test_partially_recovered_orange(self):
        assert get_status_color('Partially Recovered') == 'orange'
    
    def test_written_off_red(self):
        assert get_status_color('Written Off') == 'red'
    
    def test_unknown_status_red(self):
        assert get_status_color('Unknown') == 'red'


class TestGenerateBorrowerHistory:
    @pytest.fixture
    def sample_borrower(self):
        return {
            'Borrower_ID': 'BRW_TEST_001',
            'Risk_Score': 0.75,
            'Recovery_Status': 'Partially Recovered',
            'Legal_Action_Taken': 'No',
            'Collection_Attempts': 2,
            'Collection_Method': 'Calls',
            'Num_Missed_Payments': 3,
            'Days_Past_Due': 45
        }
    
    def test_returns_dataframe(self, sample_borrower):
        result = generate_borrower_history(sample_borrower)
        assert isinstance(result, pd.DataFrame)
    
    def test_has_required_columns(self, sample_borrower):
        result = generate_borrower_history(sample_borrower)
        assert 'Date' in result.columns
        assert 'Action' in result.columns
        assert 'Agent' in result.columns
        assert 'Result' in result.columns
    
    def test_reproducible_for_same_borrower(self, sample_borrower):
        result1 = generate_borrower_history(sample_borrower)
        result2 = generate_borrower_history(sample_borrower)
        
        # Should produce identical results
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_different_for_different_borrowers(self, sample_borrower):
        borrower2 = sample_borrower.copy()
        borrower2['Borrower_ID'] = 'BRW_TEST_002'
        
        result1 = generate_borrower_history(sample_borrower)
        result2 = generate_borrower_history(borrower2)
        
        # Should produce different results (different random seed)
        assert not result1.equals(result2) or len(result1) != len(result2)
    
    def test_includes_system_action(self, sample_borrower):
        result = generate_borrower_history(sample_borrower)
        actions = result['Action'].tolist()
        assert 'AI Risk Model Evaluation' in actions
    
    def test_includes_emi_bounces_for_missed_payments(self, sample_borrower):
        result = generate_borrower_history(sample_borrower)
        emi_actions = result[result['Action'].str.contains('EMI', na=False)]
        assert len(emi_actions) > 0
    
    def test_sorted_descending_by_date(self, sample_borrower):
        result = generate_borrower_history(sample_borrower)
        if len(result) > 1:
            dates = pd.to_datetime(result['Date'])
            assert dates.is_monotonic_decreasing
