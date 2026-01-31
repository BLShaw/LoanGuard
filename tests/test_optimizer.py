"""Tests for optimization engine (multi-arm bandit)."""
import pytest
import numpy as np

from src.optimizer import ContextualBandit, CausalEstimator, ActionRecommendation


class TestContextualBandit:
    def test_initialization(self):
        bandit = ContextualBandit()
        assert len(bandit.actions) == 3
        assert "Standard Monitoring" in bandit.actions
        assert "Settlement Offer" in bandit.actions
        assert "Legal Action" in bandit.actions
    
    def test_custom_actions(self):
        bandit = ContextualBandit(actions=["A", "B"])
        assert bandit.actions == ["A", "B"]
        assert bandit.n_actions == 2
    
    def test_recommend_action_returns_valid(self):
        bandit = ContextualBandit()
        rec = bandit.recommend_action(risk_score=0.5)
        
        assert isinstance(rec, ActionRecommendation)
        assert rec.action in bandit.actions
        assert 0 <= rec.confidence <= 1
        assert 0 <= rec.expected_recovery_rate <= 1
        assert isinstance(rec.reasoning, str)
    
    def test_recommend_action_exploit_only(self):
        bandit = ContextualBandit()
        
        # Train the bandit to prefer "Legal Action"
        for _ in range(50):
            bandit.update("Legal Action", success=True, risk_score=0.8)
            bandit.update("Standard Monitoring", success=False, risk_score=0.8)
        
        rec = bandit.recommend_action(risk_score=0.8, explore=False)
        assert rec.action == "Legal Action"
    
    def test_update_changes_parameters(self):
        bandit = ContextualBandit()
        
        initial_params = bandit._global_params["Legal Action"]
        bandit.update("Legal Action", success=True)
        updated_params = bandit._global_params["Legal Action"]
        
        assert updated_params[0] == initial_params[0] + 1  # Alpha increased
        assert updated_params[1] == initial_params[1]  # Beta unchanged
    
    def test_context_bucketing(self):
        bandit = ContextualBandit()
        
        assert bandit._get_context_bucket(0.2) == "low_risk"
        assert bandit._get_context_bucket(0.5) == "medium_risk"
        assert bandit._get_context_bucket(0.8) == "high_risk"
        
        assert bandit._get_context_bucket(0.5, "Segment_A") == "medium_risk_Segment_A"
    
    def test_get_action_stats(self):
        bandit = ContextualBandit()
        
        bandit.update("Legal Action", True)
        bandit.update("Legal Action", True)
        bandit.update("Legal Action", False)
        
        stats = bandit.get_action_stats()
        
        assert "Legal Action" in stats
        assert stats["Legal Action"]["observations"] == 3
        assert stats["Legal Action"]["successes"] == 2
        assert stats["Legal Action"]["failures"] == 1
    
    def test_credible_interval(self):
        bandit = ContextualBandit()
        
        for _ in range(100):
            bandit.update("Settlement Offer", success=True)
        
        stats = bandit.get_action_stats()
        ci = stats["Settlement Offer"]["confidence_interval"]
        
        assert ci[0] < ci[1]  # Lower < Upper
        assert 0 <= ci[0] <= 1
        assert 0 <= ci[1] <= 1
    
    def test_simulate_outcomes(self):
        bandit = ContextualBandit()
        
        simulation = bandit.simulate_outcomes(n_simulations=100)
        
        for action in bandit.actions:
            assert action in simulation
            assert "mean" in simulation[action]
            assert "p10" in simulation[action]
            assert "p90" in simulation[action]
            assert simulation[action]["p10"] <= simulation[action]["p50"]
            assert simulation[action]["p50"] <= simulation[action]["p90"]


class TestCausalEstimator:
    def test_add_observation(self):
        estimator = CausalEstimator()
        
        estimator.add_observation("Treatment", 1.0, {"risk": 0.5})
        estimator.add_observation("Control", 0.0, {"risk": 0.5})
        
        assert len(estimator._observations) == 2
    
    def test_estimate_insufficient_data(self):
        estimator = CausalEstimator()
        
        estimator.add_observation("Treatment", 1.0, {})
        estimator.add_observation("Control", 0.0, {})
        
        result = estimator.estimate_treatment_effect("Treatment", "Control")
        
        assert result['effect'] is None
        assert "Insufficient" in result['message']
    
    def test_estimate_with_clear_effect(self):
        estimator = CausalEstimator()
        
        # Treatment always succeeds
        for _ in range(20):
            estimator.add_observation("Treatment", 1.0, {})
        
        # Control always fails
        for _ in range(20):
            estimator.add_observation("Control", 0.0, {})
        
        result = estimator.estimate_treatment_effect("Treatment", "Control")
        
        assert result['effect'] == pytest.approx(1.0)
        assert result['significant'] == True
        assert result['p_value'] < 0.001
    
    def test_estimate_no_effect(self):
        estimator = CausalEstimator()
        np.random.seed(42)
        
        # Both have 50% success rate
        for _ in range(50):
            estimator.add_observation("Treatment", float(np.random.random() > 0.5), {})
            estimator.add_observation("Control", float(np.random.random() > 0.5), {})
        
        result = estimator.estimate_treatment_effect("Treatment", "Control")
        
        # Effect should be small and not significant
        assert abs(result['effect']) < 0.2
    
    def test_interpretation_positive_effect(self):
        estimator = CausalEstimator()
        
        for _ in range(20):
            estimator.add_observation("Better", 0.9, {})
            estimator.add_observation("Worse", 0.1, {})
        
        result = estimator.estimate_treatment_effect("Better", "Worse")
        
        assert "higher" in result['interpretation']
        assert "statistically significant" in result['interpretation']
