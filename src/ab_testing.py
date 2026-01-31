"""
A/B Strategy Testing Framework.

Features:
- Randomly assign borrowers to strategy groups
- Track outcomes per strategy group
- Calculate statistical significance (chi-square test)
- Thompson Sampling for adaptive allocation
"""

import json
import os
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from scipy import stats


@dataclass
class ABTestConfig:
    """Configuration for an A/B test."""
    test_id: str
    test_name: str
    strategies: List[str]
    allocation_weights: List[float]  # Must sum to 1.0
    start_date: str
    end_date: Optional[str] = None
    is_active: bool = True
    use_thompson_sampling: bool = False


@dataclass
class ABTestResult:
    """Results for a single strategy in an A/B test."""
    strategy: str
    total_assigned: int
    fully_recovered: int
    partially_recovered: int
    written_off: int
    recovery_rate: float
    avg_recovery_amount: Optional[float]
    
    
class ABTestingFramework:
    """
    A/B Testing framework for comparing recovery strategies.
    """
    
    def __init__(self, data_dir: str = None):
        """Initialize the A/B testing framework."""
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, 'data', 'ab_tests')
        
        self.data_dir = data_dir
        self.config_file = os.path.join(data_dir, 'ab_config.json')
        self.assignments_file = os.path.join(data_dir, 'assignments.jsonl')
        self.results_file = os.path.join(data_dir, 'results.jsonl')
        
        os.makedirs(data_dir, exist_ok=True)
        
        # Thompson Sampling priors (alpha, beta for Beta distribution)
        # Key: (test_id, strategy) -> (successes + 1, failures + 1)
        self._thompson_params: Dict[Tuple[str, str], Tuple[int, int]] = {}
    
    def _load_config(self) -> Dict[str, ABTestConfig]:
        """Load all test configurations."""
        if not os.path.exists(self.config_file):
            return {}
        
        with open(self.config_file, 'r') as f:
            data = json.load(f)
        
        return {k: ABTestConfig(**v) for k, v in data.items()}
    
    def _save_config(self, configs: Dict[str, ABTestConfig]) -> None:
        """Save test configurations."""
        data = {k: asdict(v) for k, v in configs.items()}
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_test(
        self,
        test_name: str,
        strategies: List[str],
        allocation_weights: List[float] = None,
        use_thompson_sampling: bool = False
    ) -> str:
        """Create a new A/B test."""
        if allocation_weights is None:
            # Equal allocation by default
            allocation_weights = [1.0 / len(strategies)] * len(strategies)
        
        # Validate weights sum to 1
        weight_sum = sum(allocation_weights)
        if abs(weight_sum - 1.0) > 0.001:
            # Normalize weights
            allocation_weights = [w / weight_sum for w in allocation_weights]
        
        test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        config = ABTestConfig(
            test_id=test_id,
            test_name=test_name,
            strategies=strategies,
            allocation_weights=allocation_weights,
            start_date=datetime.now().isoformat(),
            use_thompson_sampling=use_thompson_sampling
        )
        
        configs = self._load_config()
        configs[test_id] = config
        self._save_config(configs)
        
        # Initialize Thompson Sampling parameters
        for strategy in strategies:
            self._thompson_params[(test_id, strategy)] = (1, 1)  # Prior: Beta(1,1)
        
        return test_id
    
    def get_active_test(self) -> Optional[ABTestConfig]:
        """Get the currently active A/B test."""
        configs = self._load_config()
        for config in configs.values():
            if config.is_active:
                return config
        return None
    
    def assign_strategy(
        self,
        borrower_id: str,
        test_id: str = None,
        risk_score: float = None
    ) -> str:
        """
        Assign a borrower to a strategy group.
        Uses deterministic hashing for consistency (same borrower always gets same group).
        """
        # Get active test if not specified
        if test_id is None:
            active_test = self.get_active_test()
            if active_test is None:
                # No active test, return default strategy based on risk
                return self._default_strategy(risk_score)
            test_id = active_test.test_id
        
        configs = self._load_config()
        if test_id not in configs:
            return self._default_strategy(risk_score)
        
        config = configs[test_id]
        
        if config.use_thompson_sampling:
            strategy = self._thompson_sampling_select(test_id, config.strategies)
        else:
            # Deterministic assignment based on borrower ID hash
            hash_value = int(hashlib.sha256(borrower_id.encode()).hexdigest(), 16)
            rand_value = (hash_value % 10000) / 10000.0
            
            cumulative = 0.0
            strategy = config.strategies[-1]  # Default to last
            for i, weight in enumerate(config.allocation_weights):
                cumulative += weight
                if rand_value < cumulative:
                    strategy = config.strategies[i]
                    break
        
        # Log assignment
        self._log_assignment(test_id, borrower_id, strategy, risk_score)
        
        return strategy
    
    def _thompson_sampling_select(self, test_id: str, strategies: List[str]) -> str:
        """Select strategy using Thompson Sampling."""
        samples = []
        for strategy in strategies:
            key = (test_id, strategy)
            if key not in self._thompson_params:
                self._thompson_params[key] = (1, 1)
            alpha, beta = self._thompson_params[key]
            # Sample from Beta distribution
            sample = np.random.beta(alpha, beta)
            samples.append((sample, strategy))
        
        # Return strategy with highest sample
        samples.sort(reverse=True)
        return samples[0][1]
    
    def update_thompson_params(
        self,
        test_id: str,
        strategy: str,
        success: bool
    ) -> None:
        """Update Thompson Sampling parameters based on outcome."""
        key = (test_id, strategy)
        if key not in self._thompson_params:
            self._thompson_params[key] = (1, 1)
        
        alpha, beta = self._thompson_params[key]
        if success:
            self._thompson_params[key] = (alpha + 1, beta)
        else:
            self._thompson_params[key] = (alpha, beta + 1)
    
    def _default_strategy(self, risk_score: float = None) -> str:
        """Return default strategy based on risk score."""
        if risk_score is None:
            return "Standard Monitoring"
        if risk_score > 0.75:
            return "Legal Action"
        elif risk_score > 0.50:
            return "Settlement Offer"
        return "Standard Monitoring"
    
    def _log_assignment(
        self,
        test_id: str,
        borrower_id: str,
        strategy: str,
        risk_score: float = None
    ) -> None:
        """Log a strategy assignment."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'test_id': test_id,
            'borrower_id': borrower_id,
            'assigned_strategy': strategy,
            'risk_score': round(risk_score, 4) if risk_score else None
        }
        with open(self.assignments_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def record_outcome(
        self,
        test_id: str,
        borrower_id: str,
        strategy: str,
        outcome: str,
        recovery_amount: float = None
    ) -> None:
        """Record the outcome for a borrower in an A/B test."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'test_id': test_id,
            'borrower_id': borrower_id,
            'strategy': strategy,
            'outcome': outcome,
            'recovery_amount': recovery_amount
        }
        with open(self.results_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        # Update Thompson Sampling if applicable
        configs = self._load_config()
        if test_id in configs and configs[test_id].use_thompson_sampling:
            success = outcome == 'Fully Recovered'
            self.update_thompson_params(test_id, strategy, success)
    
    def get_test_results(self, test_id: str) -> Dict[str, ABTestResult]:
        """Get aggregated results for an A/B test."""
        if not os.path.exists(self.results_file):
            return {}
        
        # Read all results
        results_by_strategy: Dict[str, Dict] = {}
        
        with open(self.results_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get('test_id') != test_id:
                        continue
                    
                    strategy = entry.get('strategy', 'Unknown')
                    if strategy not in results_by_strategy:
                        results_by_strategy[strategy] = {
                            'total': 0,
                            'fully_recovered': 0,
                            'partially_recovered': 0,
                            'written_off': 0,
                            'recovery_amounts': []
                        }
                    
                    results_by_strategy[strategy]['total'] += 1
                    outcome = entry.get('outcome', '')
                    
                    if outcome == 'Fully Recovered':
                        results_by_strategy[strategy]['fully_recovered'] += 1
                    elif outcome == 'Partially Recovered':
                        results_by_strategy[strategy]['partially_recovered'] += 1
                    elif outcome == 'Written Off':
                        results_by_strategy[strategy]['written_off'] += 1
                    
                    if entry.get('recovery_amount') is not None:
                        results_by_strategy[strategy]['recovery_amounts'].append(
                            entry['recovery_amount']
                        )
                except json.JSONDecodeError:
                    continue
        
        # Convert to ABTestResult objects
        results = {}
        for strategy, data in results_by_strategy.items():
            total = data['total']
            recovery_rate = data['fully_recovered'] / total if total > 0 else 0.0
            
            amounts = data['recovery_amounts']
            avg_amount = sum(amounts) / len(amounts) if amounts else None
            
            results[strategy] = ABTestResult(
                strategy=strategy,
                total_assigned=total,
                fully_recovered=data['fully_recovered'],
                partially_recovered=data['partially_recovered'],
                written_off=data['written_off'],
                recovery_rate=round(recovery_rate, 4),
                avg_recovery_amount=round(avg_amount, 2) if avg_amount else None
            )
        
        return results
    
    def calculate_significance(
        self,
        test_id: str,
        metric: str = 'recovery_rate'
    ) -> Dict[str, Any]:
        """
        Calculate statistical significance between strategies using chi-square test.
        Returns p-value and whether the difference is significant at alpha=0.05.
        """
        results = self.get_test_results(test_id)
        
        if len(results) < 2:
            return {
                'significant': False,
                'p_value': None,
                'message': 'Insufficient data: Need at least 2 strategies to compare'
            }
        
        # Build contingency table for chi-square test
        # Rows: strategies, Columns: [recovered, not_recovered]
        strategies = list(results.keys())
        contingency_table = []
        
        for strategy in strategies:
            r = results[strategy]
            recovered = r.fully_recovered
            not_recovered = r.total_assigned - r.fully_recovered
            contingency_table.append([recovered, not_recovered])
        
        contingency_table = np.array(contingency_table)
        
        # Check if we have enough data
        if contingency_table.sum() < 20:
            return {
                'significant': False,
                'p_value': None,
                'message': 'Insufficient data for significance test (need at least 20 outcomes)'
            }
        
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            return {
                'significant': p_value < 0.05,
                'p_value': round(p_value, 4),
                'chi2_statistic': round(chi2, 4),
                'degrees_of_freedom': dof,
                'strategies_compared': strategies,
                'message': 'Statistically significant difference' if p_value < 0.05 
                          else 'No significant difference detected'
            }
        except Exception as e:
            return {
                'significant': False,
                'p_value': None,
                'message': f'Could not calculate significance: {str(e)}'
            }
    
    def get_best_strategy(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get the best performing strategy based on recovery rate."""
        results = self.get_test_results(test_id)
        
        if not results:
            return None
        
        best = max(results.values(), key=lambda r: r.recovery_rate)
        return {
            'strategy': best.strategy,
            'recovery_rate': best.recovery_rate,
            'total_assigned': best.total_assigned
        }


# Singleton instance
_ab_framework = None

def get_ab_framework(data_dir: str = None) -> ABTestingFramework:
    """Get the singleton A/B testing framework instance."""
    global _ab_framework
    if _ab_framework is None:
        _ab_framework = ABTestingFramework(data_dir)
    return _ab_framework
