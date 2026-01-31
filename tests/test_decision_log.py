"""Tests for decision logging."""
import pytest
import tempfile
import os
import json
from datetime import datetime

from src.decision_log import DecisionLogger, DecisionType


@pytest.fixture
def temp_log_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def logger(temp_log_dir):
    return DecisionLogger(log_dir=temp_log_dir)


class TestDecisionLogger:
    def test_log_prediction(self, logger):
        logger.log_prediction(
            borrower_id="BRW_001",
            risk_score=0.75,
            recommended_strategy="Legal Action",
            segment="High Risk"
        )
        decisions = logger.get_all_decisions()
        assert len(decisions) == 1
        entry = decisions[0]
        assert entry['decision_type'] == 'prediction'
        assert entry['borrower_id'] == 'BRW_001'
        assert entry['data']['risk_score'] == 0.75

    def test_log_what_if_simulation(self, logger):
        logger.log_what_if_simulation(
            borrower_id="BRW_002",
            original_risk_score=0.5,
            simulated_risk_score=0.3,
            parameters_changed={'Monthly_Income': 10000}
        )
        decisions = logger.get_all_decisions()
        assert len(decisions) == 1
        entry = decisions[0]
        assert entry['decision_type'] == 'what_if_simulation'

    def test_record_outcome(self, logger):
        logger.record_outcome(
            borrower_id="BRW_003",
            predicted_risk_score=0.8,
            predicted_strategy="Legal Action",
            actual_outcome="Fully Recovered",
            days_to_resolution=30
        )
        outcomes = logger.get_outcomes()
        assert len(outcomes) == 1
        entry = outcomes[0]
        assert entry['borrower_id'] == 'BRW_003'
        assert entry['data']['actual_outcome'] == 'Fully Recovered'
    
    def test_calculate_model_accuracy_empty(self, logger):
        result = logger.calculate_model_accuracy()
        assert result['total_outcomes'] == 0
        assert result['accuracy'] is None
    
    def test_calculate_model_accuracy_with_data(self, logger):
        logger.record_outcome("B1", 0.8, "Legal", "Written Off")  # Correct
        logger.record_outcome("B2", 0.2, "Standard", "Fully Recovered")  # Correct
        logger.record_outcome("B3", 0.8, "Legal", "Fully Recovered")  # Incorrect
        
        result = logger.calculate_model_accuracy()
        assert result['total_outcomes'] == 3
        # Ensure correct predictions calculation matches logic: 
        # B1 (High Risk -> Bad Outcome) = Correct
        # B2 (Low Risk -> Good Outcome) = Correct
        # B3 (High Risk -> Good Outcome) = Incorrect
        assert result['correct_predictions'] == 2
        assert result['accuracy'] == pytest.approx(2/3, rel=0.01)

    def test_multiple_logs_append(self, logger):
        logger.log_prediction("B1", 0.1, recommended_strategy="Standard")
        logger.log_prediction("B2", 0.5, recommended_strategy="Settlement")
        decisions = logger.get_all_decisions()
        assert len(decisions) == 2
