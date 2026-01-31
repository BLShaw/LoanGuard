"""Tests for feature engineering."""
import pytest
import pandas as pd
import numpy as np

from src.features import FeatureEngineer


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Age': [30, 45, 55],
        'Monthly_Income': [5000, 8000, 12000],
        'Loan_Amount': [50000, 100000, 150000],
        'Loan_Tenure': [24, 36, 48],
        'Interest_Rate': [10.0, 12.0, 15.0],
        'Collateral_Value': [30000, 60000, 90000],
        'Outstanding_Loan_Amount': [40000, 80000, 120000],
        'Monthly_EMI': [2500, 3500, 4500],
        'Num_Missed_Payments': [0, 2, 5],
        'Days_Past_Due': [0, 30, 90],
        'Recovery_Status': ['Fully Recovered', 'Partially Recovered', 'Written Off']
    })


class TestFeatureEngineer:
    def test_preprocess_for_training_returns_correct_types(self, sample_data):
        fe = FeatureEngineer()
        X, y = fe.preprocess_for_training(sample_data)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(sample_data)
        assert len(y) == len(sample_data)
    
    def test_preprocess_for_training_creates_target(self, sample_data):
        fe = FeatureEngineer()
        X, y = fe.preprocess_for_training(sample_data)
        
        # Fully Recovered = 0 (Low Risk), others = 1 (High Risk)
        expected = [0, 1, 1]
        assert list(y) == expected
    
    def test_preprocess_for_training_scales_features(self, sample_data):
        fe = FeatureEngineer()
        X, y = fe.preprocess_for_training(sample_data)
        
        # Scaled data should have mean ~0 and std ~1
        for col in X.columns:
            assert X[col].mean() == pytest.approx(0, abs=0.1)
            assert X[col].std() == pytest.approx(1, abs=0.3)
    
    def test_preprocess_for_inference_requires_fitted_scaler(self, sample_data):
        fe = FeatureEngineer()
        
        with pytest.raises(RuntimeError, match="Scaler not fitted"):
            fe.preprocess_for_inference(sample_data)
    
    def test_preprocess_for_inference_after_training(self, sample_data):
        fe = FeatureEngineer()
        fe.preprocess_for_training(sample_data)
        
        X_inf = fe.preprocess_for_inference(sample_data)
        assert isinstance(X_inf, pd.DataFrame)
        assert len(X_inf) == len(sample_data)
    
    def test_numeric_features_list(self):
        fe = FeatureEngineer()
        expected = [
            "Age", "Monthly_Income", "Loan_Amount", "Loan_Tenure",
            "Interest_Rate", "Collateral_Value", "Outstanding_Loan_Amount",
            "Monthly_EMI", "Num_Missed_Payments", "Days_Past_Due"
        ]
        assert fe.numeric_features == expected
