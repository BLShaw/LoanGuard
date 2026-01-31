"""Tests for ML models."""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os

from src.model import RiskModel, SegmentModel


@pytest.fixture
def training_data():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'feature3': np.random.randn(n)
    })


@pytest.fixture
def training_labels():
    np.random.seed(42)
    return pd.Series(np.random.randint(0, 2, 100))


class TestRiskModel:
    def test_train_and_predict(self, training_data, training_labels):
        model = RiskModel()
        model.train(training_data, training_labels)
        
        predictions = model.predict(training_data)
        assert len(predictions) == len(training_data)
        assert all(p in [0, 1] for p in predictions)
    
    def test_predict_proba_shape(self, training_data, training_labels):
        model = RiskModel()
        model.train(training_data, training_labels)
        
        proba = model.predict_proba(training_data)
        assert proba.shape == (len(training_data), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_predict_proba_with_ci(self, training_data, training_labels):
        model = RiskModel()
        model.train(training_data, training_labels)
        
        mean, lower, upper = model.predict_proba_with_ci(training_data, confidence=0.95)
        
        assert len(mean) == len(training_data)
        assert len(lower) == len(training_data)
        assert len(upper) == len(training_data)
        
        # Lower <= Mean <= Upper
        assert all(lower <= mean)
        assert all(mean <= upper)
        
        # All values in [0, 1]
        assert all(0 <= v <= 1 for v in lower)
        assert all(0 <= v <= 1 for v in upper)
    
    def test_save_and_load(self, training_data, training_labels):
        model = RiskModel()
        model.train(training_data, training_labels)
        
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_path = f.name
        
        try:
            model.save(temp_path)
            
            loaded = RiskModel()
            loaded.load(temp_path)
            
            # Predictions should be identical
            orig_pred = model.predict(training_data)
            load_pred = loaded.predict(training_data)
            np.testing.assert_array_equal(orig_pred, load_pred)
        finally:
            os.unlink(temp_path)


class TestSegmentModel:
    def test_train_and_predict(self, training_data):
        model = SegmentModel(n_clusters=3)
        model.train(training_data)
        
        segments = model.predict(training_data)
        assert len(segments) == len(training_data)
        assert all(s in [0, 1, 2] for s in segments)
    
    def test_cluster_count(self, training_data):
        for n in [2, 3, 4]:
            model = SegmentModel(n_clusters=n)
            model.train(training_data)
            segments = model.predict(training_data)
            assert len(set(segments)) <= n
    
    def test_save_and_load(self, training_data):
        model = SegmentModel(n_clusters=3)
        model.train(training_data)
        
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_path = f.name
        
        try:
            model.save(temp_path)
            
            loaded = SegmentModel()
            loaded.load(temp_path)
            
            orig_pred = model.predict(training_data)
            load_pred = loaded.predict(training_data)
            np.testing.assert_array_equal(orig_pred, load_pred)
        finally:
            os.unlink(temp_path)
