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
    
# Strategy Operational Parameters
STRATEGY_PARAMS = {
    "Standard Monitoring": {"cost": 1.0},
    "Settlement Offer": {"cost": 50.0},
    "Legal Action": {"cost": 2500.0}
}


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
        # Set realistic banking baseline priors
        self._global_priors = {
            "Standard Monitoring": (4.0, 6.0),  # ~40% expected recovery
            "Settlement Offer": (6.0, 4.0),     # ~60% expected recovery
            "Legal Action": (8.0, 2.0)          # ~80% expected recovery
        }
        
        self._global_params: Dict[str, Tuple[float, float]] = {
            action: self._global_priors.get(action, (1.0, 1.0)) for action in actions
        }
        
        # Set context-specific baseline priors
        self._context_priors = {
            # Low Risk: Standard monitoring is very effective, legal is not needed
            ("low_risk", "Standard Monitoring"): (9.0, 1.0),
            ("low_risk", "Settlement Offer"): (8.0, 2.0),
            ("low_risk", "Legal Action"): (9.5, 0.5),
            # Medium Risk: Settlement is highly effective
            ("medium_risk", "Standard Monitoring"): (5.0, 5.0),
            ("medium_risk", "Settlement Offer"): (7.0, 3.0),
            ("medium_risk", "Legal Action"): (8.5, 1.5),
            # High Risk: Legal action is highly effective, standard is useless
            ("high_risk", "Standard Monitoring"): (1.5, 8.5),
            ("high_risk", "Settlement Offer"): (5.0, 5.0),
            ("high_risk", "Legal Action"): (7.5, 2.5)
        }
        
        self._context_params: Dict[Tuple[str, str], Tuple[float, float]] = defaultdict(
            lambda: (1.0, 1.0)
        )
        
        # Populate context params with defaults
        for key, val in self._context_priors.items():
            self._context_params[key] = val
        
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
        outstanding_amount: float = 1000.0,
        segment: str = None,
        is_scra_active: bool = False,
        is_bankrupt: bool = False,
        explore: bool = True
    ) -> ActionRecommendation:
        """
        Recommend an action for a borrower by maximizing Expected Net Recovery Value (ENRV)
        and applying compliance guardrails.
        
        Args:
            risk_score: The borrower's risk score (0-1)
            outstanding_amount: The outstanding loan balance ($)
            segment: The borrower's segment (optional)
            is_scra_active: Whether the borrower is active duty military
            is_bankrupt: Whether the borrower has active bankruptcy
            explore: Whether to explore (True) or exploit only (False)
            
        Returns:
            ActionRecommendation with action, confidence, and reasoning
        """
        # Compliance Veto: Bankruptcy Automatic Stay
        if is_bankrupt:
            return ActionRecommendation(
                action="Standard Monitoring",
                confidence=1.0,
                expected_recovery_rate=0.0,
                reasoning="Compliance Override: Borrower has active bankruptcy filing. Automatic stay is in effect; all active recovery actions are suspended by law.",
                alternative_actions=[]
            )
            
        context = self._get_context_bucket(risk_score, segment)
        
        # Calculate expected values for each action
        action_scores = []
        
        # Compliance Veto: SCRA Protection
        available_actions = self.actions.copy()
        if is_scra_active:
            available_actions = [a for a in available_actions if a != "Legal Action"]
            
        for action in available_actions:
            # Get parameters (combine global and context-specific)
            global_alpha, global_beta = self._global_params[action]
            
            # Lookup context params. If not present in context_params, try context_priors fallback
            if (context, action) in self._context_params:
                ctx_alpha, ctx_beta = self._context_params[(context, action)]
            else:
                risk_bucket = "low_risk" if "low" in context else ("medium_risk" if "medium" in context else "high_risk")
                ctx_alpha, ctx_beta = self._context_priors.get((risk_bucket, action), (1.0, 1.0))
            
            # Weighted combination: context weight increases with observations
            # Max weight 0.7 for context (reached after ~20 observations)
            # Remaining 0.3+ always from global to prevent overfitting to small samples
            ctx_weight = min(0.7, (ctx_alpha + ctx_beta - 2.0) / 20.0)
            global_weight = 1.0 - ctx_weight
            
            combined_alpha = global_weight * global_alpha + ctx_weight * ctx_alpha
            combined_beta = global_weight * global_beta + ctx_weight * ctx_beta
            
            if explore:
                # Thompson Sampling: sample from posterior
                rate = np.random.beta(combined_alpha, combined_beta)
            else:
                # Exploitation only: use mean
                rate = combined_alpha / (combined_alpha + combined_beta)
            
            # Incorporate cost structure to compute dollar score
            cost = STRATEGY_PARAMS.get(action, {}).get("cost", 0.0)
            score = (rate * outstanding_amount) - cost
            
            expected_rate = combined_alpha / (combined_alpha + combined_beta)
            action_scores.append((action, score, expected_rate))
        
        # Sort by expected ENRV score
        action_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_action, best_score, best_rate = action_scores[0]
        
        # Calculate confidence based on observations
        alpha, beta = self._global_params[best_action]
        prior_alpha, prior_beta = self._global_priors.get(best_action, (1.0, 1.0))
        total_obs = max(0.0, alpha + beta - (prior_alpha + prior_beta))
        confidence = min(0.95, 0.5 + 0.4 * (total_obs / 100.0))  # Asymptotic to 0.9
        
        # Calculate margin over second-best
        if len(action_scores) > 1:
            margin = best_rate - action_scores[1][2]
            if margin > 0.1:
                confidence = min(confidence + 0.1, 0.95)
        
        # Build alternatives list
        alternatives = [(a, round(r, 3)) for a, _, r in action_scores[1:]]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            best_action, best_rate, outstanding_amount, risk_score, segment, int(total_obs)
        )
        if is_scra_active:
            reasoning += " (Compliance Veto: SCRA active-duty litigation check applied)."
        
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
        outstanding_amount: float,
        risk_score: float,
        segment: str,
        observations: int
    ) -> str:
        """Generate human-readable reasoning for the recommendation based on ENRV."""
        cost = STRATEGY_PARAMS.get(action, {}).get("cost", 0.0)
        expected_enrv = (expected_rate * outstanding_amount) - cost
        
        parts = []
        
        # Action-specific reasoning
        if action == "Legal Action":
            parts.append(f"Legal action recommended for high exposure (${outstanding_amount:,.2f}).")
        elif action == "Settlement Offer":
            parts.append(f"Settlement offer balances recovery probability with cost.")
        else:
            parts.append("Standard monitoring is appropriate for low-balance or low-risk accounts.")
        
        # Expected outcomes
        parts.append(f"Expected Net Recovery Value (ENRV): ${expected_enrv:,.2f}.")
        parts.append(f"Expected recovery rate: {expected_rate:.1%} | Strategy cost: ${cost:,.2f}.")
        
        # Confidence context
        if observations < 10:
            parts.append("(Limited historical data)")
        elif observations > 50:
            parts.append(f"(Based on {observations} observations)")
        
        return " ".join(parts)
    
    def update(
        self,
        action: str,
        success: Any,
        risk_score: float = None,
        segment: str = None
    ) -> None:
        """
        Update the model with an observed outcome.
        
        Args:
            action: The action that was taken
            success: Recovery rate between 0.0 and 1.0 (or boolean success/failure)
            risk_score: The borrower's risk score (for context)
            segment: The borrower's segment (for context)
        """
        # Convert boolean success to float
        if isinstance(success, bool):
            success = 1.0 if success else 0.0
        else:
            success = float(success)
            success = min(1.0, max(0.0, success))
            
        # Update global parameters using fractional updates
        alpha, beta = self._global_params[action]
        self._global_params[action] = (alpha + success, beta + (1.0 - success))
        
        # Update context-specific parameters if context provided
        if risk_score is not None:
            context = self._get_context_bucket(risk_score, segment)
            ctx_alpha, ctx_beta = self._context_params[(context, action)]
            self._context_params[(context, action)] = (ctx_alpha + success, ctx_beta + (1.0 - success))
        
        self._total_observations += 1

    def load_from_outcomes(self, outcomes_file: str) -> None:
        """Load bandit parameters by playing back historically recorded outcomes."""
        import os
        import json
        if not os.path.exists(outcomes_file):
            return
        
        try:
            with open(outcomes_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        data = entry.get('data', {})
                        action = data.get('predicted_strategy')
                        actual_outcome = data.get('actual_outcome')
                        
                        if not action:
                            continue
                            
                        # Determine recovery rate / success
                        outstanding = data.get('outstanding_amount')
                        recovery_amount = data.get('actual_recovery_amount')
                        
                        if outstanding is not None and recovery_amount is not None and outstanding > 0:
                            success = min(1.0, max(0.0, recovery_amount / outstanding))
                        else:
                            if actual_outcome == 'Fully Recovered':
                                success = 1.0
                            elif actual_outcome == 'Partially Recovered':
                                success = 0.5
                            else:
                                success = 0.0
                                
                        # Get risk score and segment for context
                        risk_score = data.get('predicted_risk_score')
                        segment = data.get('segment')
                        
                        # Update parameters
                        self.update(
                            action=action,
                            success=success,
                            risk_score=risk_score,
                            segment=segment
                        )
                    except Exception:
                        continue
        except Exception:
            pass
    
    def get_action_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for each action."""
        stats = {}
        for action in self.actions:
            alpha, beta = self._global_params[action]
            prior_alpha, prior_beta = self._global_priors.get(action, (1.0, 1.0))
            successes = max(0.0, alpha - prior_alpha)
            failures = max(0.0, beta - prior_beta)
            observations = successes + failures
            success_rate = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
            
            stats[action] = {
                'observations': observations,
                'successes': successes,
                'failures': failures,
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
        Estimate the causal effect of a treatment vs control using Risk Stratification.
        This eliminates confounding bias caused by strategy assignment rules (e.g.,
        assigning Legal Action only to high-risk borrowers).
        """
        treatment_outcomes = [o for o in self._observations if o['treatment'] == treatment]
        control_outcomes = [o for o in self._observations if o['treatment'] == control]
        
        if len(treatment_outcomes) < 5 or len(control_outcomes) < 5:
            return {
                'effect': None,
                'significant': False,
                'message': 'Insufficient data for causal estimation'
            }
        
        # Risk Stratification
        # We'll create strata based on risk scores (0-0.33, 0.33-0.66, 0.66-1.0)
        def get_stratum(obs):
            risk = obs['covariates'].get('Risk_Score', 0.5)
            if risk < 0.33: return 'Low'
            elif risk < 0.66: return 'Medium'
            else: return 'High'
            
        strata_effects = []
        strata_weights = []
        
        strata_keys = ['Low', 'Medium', 'High']
        for key in strata_keys:
            t_strat = [o['outcome'] for o in treatment_outcomes if get_stratum(o) == key]
            c_strat = [o['outcome'] for o in control_outcomes if get_stratum(o) == key]
            
            # We need at least some data in the stratum to calculate effect
            if len(t_strat) > 0 and len(c_strat) > 0:
                effect = np.mean(t_strat) - np.mean(c_strat)
                weight = len(t_strat) + len(c_strat)
                strata_effects.append(effect)
                strata_weights.append(weight)
        
        if not strata_effects:
            # Fallback to simple difference in means if stratification fails due to lack of overlap
            treatment_mean = np.mean([o['outcome'] for o in treatment_outcomes])
            control_mean = np.mean([o['outcome'] for o in control_outcomes])
            overall_effect = treatment_mean - control_mean
        else:
            # Weighted average of effects across strata
            overall_effect = np.average(strata_effects, weights=strata_weights)
            
        # Simplified standard error for demonstration
        treatment_vals = [o['outcome'] for o in treatment_outcomes]
        control_vals = [o['outcome'] for o in control_outcomes]
        treatment_se = np.std(treatment_vals) / np.sqrt(len(treatment_vals))
        control_se = np.std(control_vals) / np.sqrt(len(control_vals))
        combined_se = np.sqrt(treatment_se**2 + control_se**2)
        
        if combined_se > 0:
            z_score = overall_effect / combined_se
            from scipy import stats
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        elif abs(overall_effect) > 0:
            z_score = float('inf') if overall_effect > 0 else float('-inf')
            p_value = 0.0
        else:
            z_score = 0
            p_value = 1.0
            
        return {
            'treatment': treatment,
            'control': control,
            'effect': round(overall_effect, 4),
            'standard_error': round(combined_se, 4),
            'z_score': round(z_score, 2),
            'p_value': round(p_value, 4),
            'significant': p_value < 0.05,
            'treatment_n': len(treatment_outcomes),
            'control_n': len(control_outcomes),
            'interpretation': self._interpret_effect(overall_effect, p_value, treatment, control)
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
        # Load from outcomes log
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        outcomes_file = os.path.join(base_dir, 'data', 'logs', 'outcomes.jsonl')
        _optimizer.load_from_outcomes(outcomes_file)
    return _optimizer

def get_causal_estimator() -> CausalEstimator:
    """Get the singleton causal estimator instance."""
    global _causal_estimator
    if _causal_estimator is None:
        _causal_estimator = CausalEstimator()
    return _causal_estimator
