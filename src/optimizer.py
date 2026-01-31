"""
Optimization Engine: ML-based optimal action recommendation.

Features:
- Multi-arm bandit for strategy selection
- Thompson Sampling for exploration/exploitation balance
- Contextual bandits considering borrower features
- Learns optimal action from historical outcomes
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ActionRecommendation:
    """A recommended action with confidence."""
    action: str
    confidence: float  # 0-1, how confident we are
    expected_recovery_rate: float
    reasoning: str
    alternative_actions: List[Tuple[str, float]]  # (action, expected_rate)


class ContextualBandit:
    """
    Contextual Multi-Arm Bandit for strategy optimization.
    
    Uses Thompson Sampling with context (borrower features) to learn
    which strategies work best for different borrower profiles.
    """
    
    def __init__(self, actions: List[str] = None):
        if actions is None:
            actions = ["Standard Monitoring", "Settlement Offer", "Legal Action"]
        
        self.actions = actions
        self.n_actions = len(actions)
        
        # Thompson Sampling parameters for each action
        # (successes, failures) for Beta distribution
        self._global_params: Dict[str, Tuple[int, int]] = {
            action: (1, 1) for action in actions  # Uniform prior
        }
        
        # Context-specific parameters
        # Key: (context_bucket, action) -> (successes, failures)
        self._context_params: Dict[Tuple[str, str], Tuple[int, int]] = defaultdict(
            lambda: (1, 1)
        )
        
        # Track total observations
        self._total_observations = 0
    
    def _get_context_bucket(
        self, 
        risk_score: float, 
        segment: str = None
    ) -> str:
        """Convert continuous features into discrete context buckets."""
        # Discretize risk score into buckets
        if risk_score < 0.33:
            risk_bucket = "low_risk"
        elif risk_score < 0.66:
            risk_bucket = "medium_risk"
        else:
            risk_bucket = "high_risk"
        
        if segment:
            return f"{risk_bucket}_{segment}"
        return risk_bucket
    
    def recommend_action(
        self,
        risk_score: float,
        segment: str = None,
        explore: bool = True
    ) -> ActionRecommendation:
        """
        Recommend an action for a borrower.
        
        Args:
            risk_score: The borrower's risk score (0-1)
            segment: The borrower's segment (optional)
            explore: Whether to explore (True) or exploit only (False)
            
        Returns:
            ActionRecommendation with action, confidence, and reasoning
        """
        context = self._get_context_bucket(risk_score, segment)
        
        # Calculate expected values for each action
        action_scores = []
        
        for action in self.actions:
            # Get parameters (combine global and context-specific)
            global_alpha, global_beta = self._global_params[action]
            ctx_alpha, ctx_beta = self._context_params.get(
                (context, action), (1, 1)
            )
            
            # Weighted combination: context weight increases with observations
            # Max weight 0.7 for context (reached after ~20 observations)
            # Remaining 0.3+ always from global to prevent overfitting to small samples
            ctx_weight = min(0.7, (ctx_alpha + ctx_beta - 2) / 20)
            global_weight = 1 - ctx_weight
            
            combined_alpha = global_weight * global_alpha + ctx_weight * ctx_alpha
            combined_beta = global_weight * global_beta + ctx_weight * ctx_beta
            
            if explore:
                # Thompson Sampling: sample from posterior
                score = np.random.beta(combined_alpha, combined_beta)
            else:
                # Exploitation only: use mean
                score = combined_alpha / (combined_alpha + combined_beta)
            
            expected_rate = combined_alpha / (combined_alpha + combined_beta)
            action_scores.append((action, score, expected_rate))
        
        # Sort by Thompson sample (or mean if not exploring)
        action_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_action, best_score, best_rate = action_scores[0]
        
        # Calculate confidence based on observations
        alpha, beta = self._global_params[best_action]
        total_obs = alpha + beta - 2  # Subtract prior
        confidence = min(0.95, 0.5 + 0.4 * (total_obs / 100))  # Asymptotic to 0.9
        
        # Calculate margin over second-best
        if len(action_scores) > 1:
            margin = best_rate - action_scores[1][2]
            if margin > 0.1:
                confidence = min(confidence + 0.1, 0.95)
        
        # Build alternatives list
        alternatives = [(a, round(r, 3)) for a, _, r in action_scores[1:]]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            best_action, best_rate, risk_score, segment, total_obs
        )
        
        return ActionRecommendation(
            action=best_action,
            confidence=round(confidence, 3),
            expected_recovery_rate=round(best_rate, 3),
            reasoning=reasoning,
            alternative_actions=alternatives
        )
    
    def _generate_reasoning(
        self,
        action: str,
        expected_rate: float,
        risk_score: float,
        segment: str,
        observations: int
    ) -> str:
        """Generate human-readable reasoning for the recommendation."""
        parts = []
        
        # Action-specific reasoning
        if action == "Legal Action":
            parts.append(f"Legal action recommended due to high risk score ({risk_score:.2f}).")
        elif action == "Settlement Offer":
            parts.append("Settlement offer balances recovery probability with cost.")
        else:
            parts.append("Standard monitoring is appropriate for this risk profile.")
        
        # Expected outcome
        parts.append(f"Expected recovery rate: {expected_rate:.1%}.")
        
        # Confidence context
        if observations < 10:
            parts.append("(Limited historical data - recommendation may change with more observations)")
        elif observations > 50:
            parts.append(f"(Based on {observations} historical observations)")
        
        return " ".join(parts)
    
    def update(
        self,
        action: str,
        success: bool,
        risk_score: float = None,
        segment: str = None
    ) -> None:
        """
        Update the model with an observed outcome.
        
        Args:
            action: The action that was taken
            success: Whether the outcome was successful (e.g., full recovery)
            risk_score: The borrower's risk score (for context)
            segment: The borrower's segment (for context)
        """
        # Update global parameters
        alpha, beta = self._global_params[action]
        if success:
            self._global_params[action] = (alpha + 1, beta)
        else:
            self._global_params[action] = (alpha, beta + 1)
        
        # Update context-specific parameters if context provided
        if risk_score is not None:
            context = self._get_context_bucket(risk_score, segment)
            ctx_alpha, ctx_beta = self._context_params[(context, action)]
            if success:
                self._context_params[(context, action)] = (ctx_alpha + 1, ctx_beta)
            else:
                self._context_params[(context, action)] = (ctx_alpha, ctx_beta + 1)
        
        self._total_observations += 1
    
    def get_action_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for each action."""
        stats = {}
        for action in self.actions:
            alpha, beta = self._global_params[action]
            observations = alpha + beta - 2
            success_rate = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
            
            stats[action] = {
                'observations': observations,
                'successes': alpha - 1,  # Subtract prior
                'failures': beta - 1,
                'success_rate': round(success_rate, 4),
                'confidence_interval': self._get_credible_interval(alpha, beta)
            }
        
        return stats
    
    def _get_credible_interval(
        self, 
        alpha: int, 
        beta: int, 
        level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate Bayesian credible interval for success rate."""
        from scipy import stats as sp_stats
        
        lower = sp_stats.beta.ppf((1 - level) / 2, alpha, beta)
        upper = sp_stats.beta.ppf((1 + level) / 2, alpha, beta)
        
        return (round(lower, 4), round(upper, 4))
    
    def simulate_outcomes(
        self,
        n_simulations: int = 1000
    ) -> Dict[str, Dict[str, float]]:
        """
        Simulate future outcomes using Monte Carlo.
        Useful for understanding uncertainty in recommendations.
        """
        results = {action: [] for action in self.actions}
        
        for _ in range(n_simulations):
            for action in self.actions:
                alpha, beta = self._global_params[action]
                # Sample success rate from posterior
                rate = np.random.beta(alpha, beta)
                results[action].append(rate)
        
        summary = {}
        for action in self.actions:
            rates = results[action]
            summary[action] = {
                'mean': round(np.mean(rates), 4),
                'std': round(np.std(rates), 4),
                'p10': round(np.percentile(rates, 10), 4),
                'p50': round(np.percentile(rates, 50), 4),
                'p90': round(np.percentile(rates, 90), 4)
            }
        
        return summary


class CausalEstimator:
    """
    Simple causal effect estimation.
    
    Uses propensity score weighting to estimate causal effects
    of different strategies on recovery outcomes.
    """
    
    def __init__(self):
        self._observations: List[Dict] = []
    
    def add_observation(
        self,
        treatment: str,  # Strategy used
        outcome: float,  # 1 for recovered, 0 for not
        covariates: Dict[str, float]  # Borrower features
    ) -> None:
        """Add an observation for causal analysis."""
        self._observations.append({
            'treatment': treatment,
            'outcome': outcome,
            'covariates': covariates
        })
    
    def estimate_treatment_effect(
        self,
        treatment: str,
        control: str = "Standard Monitoring"
    ) -> Dict[str, Any]:
        """
        Estimate the causal effect of a treatment vs control.
        
        Uses simple difference-in-means with optional covariate adjustment.
        For production use, consider more sophisticated methods (IPW, AIPW, etc.)
        """
        treatment_outcomes = [
            o['outcome'] for o in self._observations 
            if o['treatment'] == treatment
        ]
        control_outcomes = [
            o['outcome'] for o in self._observations 
            if o['treatment'] == control
        ]
        
        if len(treatment_outcomes) < 5 or len(control_outcomes) < 5:
            return {
                'effect': None,
                'significant': False,
                'message': 'Insufficient data for causal estimation'
            }
        
        # Simple difference in means
        treatment_mean = np.mean(treatment_outcomes)
        control_mean = np.mean(control_outcomes)
        effect = treatment_mean - control_mean
        
        # Standard error (assuming independence)
        treatment_se = np.std(treatment_outcomes) / np.sqrt(len(treatment_outcomes))
        control_se = np.std(control_outcomes) / np.sqrt(len(control_outcomes))
        combined_se = np.sqrt(treatment_se**2 + control_se**2)
        
        # Z-score and p-value
        if combined_se > 0:
            z_score = effect / combined_se
            from scipy import stats
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        elif abs(effect) > 0:
            # Perfect separation - effect with no variance = significant
            z_score = float('inf') if effect > 0 else float('-inf')
            p_value = 0.0
        else:
            z_score = 0
            p_value = 1.0
        
        return {
            'treatment': treatment,
            'control': control,
            'effect': round(effect, 4),
            'standard_error': round(combined_se, 4),
            'z_score': round(z_score, 2),
            'p_value': round(p_value, 4),
            'significant': p_value < 0.05,
            'treatment_n': len(treatment_outcomes),
            'control_n': len(control_outcomes),
            'interpretation': self._interpret_effect(effect, p_value, treatment, control)
        }
    
    def _interpret_effect(
        self, 
        effect: float, 
        p_value: float,
        treatment: str,
        control: str
    ) -> str:
        """Generate human-readable interpretation of causal effect."""
        if p_value >= 0.05:
            return f"No statistically significant difference between {treatment} and {control}."
        
        direction = "higher" if effect > 0 else "lower"
        magnitude = abs(effect) * 100
        
        return (
            f"{treatment} shows {magnitude:.1f}% {direction} recovery rate "
            f"compared to {control} (statistically significant)."
        )


# Singleton instances
_optimizer = None
_causal_estimator = None

def get_optimizer() -> ContextualBandit:
    """Get the singleton optimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = ContextualBandit()
    return _optimizer

def get_causal_estimator() -> CausalEstimator:
    """Get the singleton causal estimator instance."""
    global _causal_estimator
    if _causal_estimator is None:
        _causal_estimator = CausalEstimator()
    return _causal_estimator
