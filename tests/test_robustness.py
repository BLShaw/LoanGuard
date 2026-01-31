"""Robustness and stress tests for core components."""
import pytest
import numpy as np
import pandas as pd
import random
import string
from src.features import FeatureEngineer
from src.model import RiskModel
from src.optimizer import ContextualBandit

def generate_random_string(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

class TestRobustness:
    def test_feature_engineering_extreme_values(self):
        """Test feature engineering with extreme numeric values."""
        extreme_df = pd.DataFrame({
            'Age': [1e9, -1e9, 0, np.nan],
            'Monthly_Income': [1e15, -500, np.inf, -np.inf],
            'Loan_Amount': [0, 1e20, np.nan, 5000],
            'Loan_Tenure': [1000, -10, 0, 12],
            'Interest_Rate': [100.0, 0.0, -5.0, 5000.0],
            'Collateral_Value': [1e20, -1e10, 0, np.nan],
            'Outstanding_Loan_Amount': [1e20, -1e10, 0, 5000],
            'Monthly_EMI': [1e10, -5000, 0, 100],
            'Num_Missed_Payments': [1000, -5, 0, np.nan],
            'Days_Past_Due': [10000, -50, 0, np.nan],
            'Recovery_Status': ['Unknown', 'Fully Recovered', None, generate_random_string()]
        })
        
        fe = FeatureEngineer()
        try:
            X, y = fe.preprocess_for_training(extreme_df.replace([np.inf, -np.inf], np.nan))
            
            # Verify output structure
            assert not X.empty
            assert len(y) == len(X)
            pass
        except Exception as e:
             pytest.fail(f"Feature engineering crashed on extreme values: {e}")

    def test_risk_model_garbage_input(self):
        """Test model prediction with random noise input."""
        model = RiskModel()
        # Train on dummy data first to avoid NotFittedError
        X_train = np.random.rand(10, 10)
        y_train = np.random.randint(0, 2, 10)
        model.train(X_train, y_train)
        
        # Create random input array (100, 10 features)
        X_garbage = np.random.rand(100, 10) * 1e6
        
        # Should output valid probabilities [0, 1]
        try:
            probas = model.predict_proba(X_garbage)
            assert probas.shape == (100, 2)
            assert np.all((probas >= 0) & (probas <= 1))
        except Exception as e:
            pytest.fail(f"Model crashed on garbage input: {e}")

    def test_bandit_convergence_check(self):
        """Test bandit convergence over many iterations."""
        bandit = ContextualBandit(actions=["Arm0", "Arm1", "Arm2"])
        
        # True success rates for context "high_risk" (risk_score=0.8)
        # Arm 1 is best (80%), Arm 0 (20%), Arm 2 (20%)
        true_rates = {"Arm0": 0.2, "Arm1": 0.8, "Arm2": 0.2}
        
        n_steps = 200
        chosen_best = 0
        risk_score = 0.8 # high_risk context
        
        for _ in range(n_steps):
            rec = bandit.recommend_action(risk_score=risk_score)
            action = rec.action
            
            # Simulate outcome
            # Check correctness relative to best arm
            if action == "Arm1":
                chosen_best += 1
            
            # Generate reward
            success = random.random() < true_rates[action]
            
            # Update bandit
            bandit.update(action=action, success=success, risk_score=risk_score)
            
        # Expect to choose best arm > 40% of time eventually
        assert chosen_best > (n_steps * 0.4)

    def test_loading_empty_dataframe(self):
        """Test pipeline with empty dataframe."""
        empty_df = pd.DataFrame(columns=[
            'Age', 'Monthly_Income', 'Loan_Amount', 'Loan_Tenure',
            'Interest_Rate', 'Collateral_Value', 'Outstanding_Loan_Amount', 
            'Monthly_EMI', 'Num_Missed_Payments', 'Days_Past_Due', 'Recovery_Status'
        ])
        
        fe = FeatureEngineer()
        try:
            X, y = fe.preprocess_for_training(empty_df)
            assert len(X) == 0
        except (ValueError, IndexError):
            pass # Acceptable behavior
