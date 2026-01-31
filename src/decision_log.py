"""
Decision Logger: Persistent audit trail for all predictions and outcomes.

Features:
- Log every prediction with timestamp, borrower ID, risk score, recommended strategy
- Track user decisions and overrides
- Record actual outcomes for feedback loop
- Calculate model accuracy over time
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class DecisionType(Enum):
    """Types of decisions that can be logged."""
    PREDICTION = "prediction"
    STRATEGY_RECOMMENDATION = "strategy_recommendation"
    USER_OVERRIDE = "user_override"
    OUTCOME_RECORDED = "outcome_recorded"
    WHAT_IF_SIMULATION = "what_if_simulation"


@dataclass
class DecisionLogEntry:
    """A single decision log entry."""
    timestamp: str
    decision_type: str
    borrower_id: str
    data: Dict[str, Any]
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DecisionLogEntry':
        return cls(**data)


class DecisionLogger:
    """
    Persistent decision logging system.
    Stores logs as JSON Lines (.jsonl) for efficient append operations.
    """
    
    def __init__(self, log_dir: str = None):
        """Initialize the decision logger."""
        if log_dir is None:
            # Default to project's data directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_dir = os.path.join(base_dir, 'data', 'logs')
        
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, 'decisions.jsonl')
        self.outcomes_file = os.path.join(log_dir, 'outcomes.jsonl')
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
    
    def _append_log(self, filepath: str, entry: Dict) -> None:
        """Append a log entry to a JSONL file."""
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
    
    def _read_logs(self, filepath: str) -> List[Dict]:
        """Read all logs from a JSONL file."""
        if not os.path.exists(filepath):
            return []
        
        logs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines
        return logs
    
    def log_prediction(
        self,
        borrower_id: str,
        risk_score: float,
        confidence_lower: float = None,
        confidence_upper: float = None,
        recommended_strategy: str = None,
        segment: str = None,
        user_id: str = None
    ) -> None:
        """Log a risk prediction."""
        entry = DecisionLogEntry(
            timestamp=datetime.now().isoformat(),
            decision_type=DecisionType.PREDICTION.value,
            borrower_id=borrower_id,
            user_id=user_id,
            data={
                'risk_score': round(risk_score, 4),
                'confidence_lower': round(confidence_lower, 4) if confidence_lower else None,
                'confidence_upper': round(confidence_upper, 4) if confidence_upper else None,
                'recommended_strategy': recommended_strategy,
                'segment': segment
            }
        )
        self._append_log(self.log_file, entry.to_dict())
    
    def log_strategy_recommendation(
        self,
        borrower_id: str,
        recommended_strategy: str,
        risk_score: float,
        ab_group: str = None,
        user_id: str = None
    ) -> None:
        """Log a strategy recommendation."""
        entry = DecisionLogEntry(
            timestamp=datetime.now().isoformat(),
            decision_type=DecisionType.STRATEGY_RECOMMENDATION.value,
            borrower_id=borrower_id,
            user_id=user_id,
            data={
                'recommended_strategy': recommended_strategy,
                'risk_score': round(risk_score, 4),
                'ab_group': ab_group
            }
        )
        self._append_log(self.log_file, entry.to_dict())
    
    def log_user_override(
        self,
        borrower_id: str,
        original_strategy: str,
        override_strategy: str,
        reason: str = None,
        user_id: str = None
    ) -> None:
        """Log when a user overrides the recommended strategy."""
        entry = DecisionLogEntry(
            timestamp=datetime.now().isoformat(),
            decision_type=DecisionType.USER_OVERRIDE.value,
            borrower_id=borrower_id,
            user_id=user_id,
            data={
                'original_strategy': original_strategy,
                'override_strategy': override_strategy,
                'reason': reason
            }
        )
        self._append_log(self.log_file, entry.to_dict())
    
    def log_what_if_simulation(
        self,
        borrower_id: str,
        original_risk_score: float,
        simulated_risk_score: float,
        parameters_changed: Dict[str, Any],
        user_id: str = None
    ) -> None:
        """Log a What-If simulation."""
        entry = DecisionLogEntry(
            timestamp=datetime.now().isoformat(),
            decision_type=DecisionType.WHAT_IF_SIMULATION.value,
            borrower_id=borrower_id,
            user_id=user_id,
            data={
                'original_risk_score': round(original_risk_score, 4),
                'simulated_risk_score': round(simulated_risk_score, 4),
                'risk_delta': round(simulated_risk_score - original_risk_score, 4),
                'parameters_changed': parameters_changed
            }
        )
        self._append_log(self.log_file, entry.to_dict())
    
    def record_outcome(
        self,
        borrower_id: str,
        predicted_risk_score: float,
        predicted_strategy: str,
        actual_outcome: str,
        actual_recovery_amount: float = None,
        days_to_resolution: int = None,
        user_id: str = None
    ) -> None:
        """Record the actual outcome for a borrower (feedback loop)."""
        entry = DecisionLogEntry(
            timestamp=datetime.now().isoformat(),
            decision_type=DecisionType.OUTCOME_RECORDED.value,
            borrower_id=borrower_id,
            user_id=user_id,
            data={
                'predicted_risk_score': round(float(predicted_risk_score), 4),
                'predicted_strategy': predicted_strategy,
                'actual_outcome': actual_outcome,
                'actual_recovery_amount': actual_recovery_amount,
                'days_to_resolution': int(days_to_resolution) if days_to_resolution is not None else None,
                # Compute if prediction was correct
                'prediction_correct': self._evaluate_prediction(
                    float(predicted_risk_score), actual_outcome
                )
            }
        )
        self._append_log(self.outcomes_file, entry.to_dict())
    
    def _evaluate_prediction(self, predicted_risk_score: float, actual_outcome: str) -> bool:
        """Evaluate if prediction was correct based on outcome."""
        # High risk (>0.5) should correlate with non-recovery
        predicted_high_risk = predicted_risk_score > 0.5
        actual_high_risk = actual_outcome != 'Fully Recovered'
        # Convert to Python bool to ensure JSON serializable
        return bool(predicted_high_risk == actual_high_risk)
    
    def get_decision_history(self, borrower_id: str) -> List[Dict]:
        """Get all decisions for a specific borrower."""
        all_logs = self._read_logs(self.log_file)
        return [log for log in all_logs if log.get('borrower_id') == borrower_id]
    
    def get_outcomes(self) -> List[Dict]:
        """Get all recorded outcomes."""
        return self._read_logs(self.outcomes_file)
    
    def get_all_decisions(self) -> List[Dict]:
        """Get all decision logs."""
        return self._read_logs(self.log_file)
    
    def calculate_model_accuracy(self) -> Dict[str, Any]:
        """Calculate model accuracy from recorded outcomes."""
        outcomes = self.get_outcomes()
        
        if not outcomes:
            return {
                'total_outcomes': 0,
                'accuracy': None,
                'message': 'No outcomes recorded yet'
            }
        
        correct = sum(1 for o in outcomes if o.get('data', {}).get('prediction_correct', False))
        total = len(outcomes)
        
        # Calculate by outcome type
        outcome_breakdown = {}
        for outcome in outcomes:
            actual = outcome.get('data', {}).get('actual_outcome', 'Unknown')
            if actual not in outcome_breakdown:
                outcome_breakdown[actual] = {'total': 0, 'correct': 0}
            outcome_breakdown[actual]['total'] += 1
            if outcome.get('data', {}).get('prediction_correct', False):
                outcome_breakdown[actual]['correct'] += 1
        
        return {
            'total_outcomes': total,
            'correct_predictions': correct,
            'accuracy': round(correct / total, 4) if total > 0 else None,
            'outcome_breakdown': outcome_breakdown
        }
    
    def get_strategy_performance(self) -> Dict[str, Dict]:
        """Get performance metrics by strategy."""
        outcomes = self.get_outcomes()
        
        strategy_stats = {}
        for outcome in outcomes:
            data = outcome.get('data', {})
            strategy = data.get('predicted_strategy', 'Unknown')
            
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'total': 0,
                    'fully_recovered': 0,
                    'partially_recovered': 0,
                    'written_off': 0,
                    'avg_days_to_resolution': []
                }
            
            strategy_stats[strategy]['total'] += 1
            actual_outcome = data.get('actual_outcome', '')
            
            if actual_outcome == 'Fully Recovered':
                strategy_stats[strategy]['fully_recovered'] += 1
            elif actual_outcome == 'Partially Recovered':
                strategy_stats[strategy]['partially_recovered'] += 1
            elif actual_outcome == 'Written Off':
                strategy_stats[strategy]['written_off'] += 1
            
            days = data.get('days_to_resolution')
            if days is not None:
                strategy_stats[strategy]['avg_days_to_resolution'].append(days)
        
        # Calculate averages
        for strategy in strategy_stats:
            days_list = strategy_stats[strategy]['avg_days_to_resolution']
            if days_list:
                strategy_stats[strategy]['avg_days_to_resolution'] = round(
                    sum(days_list) / len(days_list), 1
                )
            else:
                strategy_stats[strategy]['avg_days_to_resolution'] = None
            
            # Calculate recovery rate
            total = strategy_stats[strategy]['total']
            recovered = strategy_stats[strategy]['fully_recovered']
            strategy_stats[strategy]['recovery_rate'] = round(recovered / total, 4) if total > 0 else None
        
        return strategy_stats


# Singleton instance for app-wide use
_logger_instance = None

def get_logger(log_dir: str = None) -> DecisionLogger:
    """Get the singleton decision logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = DecisionLogger(log_dir)
    return _logger_instance
