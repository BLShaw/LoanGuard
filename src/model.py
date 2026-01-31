"""
Machine Learning Models for LoanGuard.

Includes:
- RiskModel: Random Forest classifier with confidence intervals
- SegmentModel: K-Means clustering for borrower segmentation
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from typing import Tuple, Dict, List, Any


class RiskModel:
    """
    Risk prediction model using Random Forest.
    Supports confidence intervals via tree variance estimation.
    """
    
    def __init__(self, n_estimators: int = 100):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=42,
            oob_score=True  # Enable out-of-bag score for validation
        )
        self._is_fitted = False

    def train(self, X, y):
        """Train the risk model."""
        self.model.fit(X, y)
        self._is_fitted = True

    def predict_proba(self, X):
        """Get probability predictions."""
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """Get class predictions (0 or 1)."""
        return self.model.predict(X)
    
    def predict_proba_with_ci(
        self, 
        X, 
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get probability predictions with confidence intervals.
        
        Uses individual tree predictions to estimate uncertainty.
        This is a conservative estimate based on tree variance.
        
        Args:
            X: Feature matrix
            confidence: Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            Tuple of (mean_proba, lower_bound, upper_bound)
        """
        # Get predictions from each tree
        tree_predictions = np.array([
            tree.predict_proba(X)[:, 1] for tree in self.model.estimators_
        ])  # Shape: (n_trees, n_samples)
        
        # Calculate mean and std across trees
        mean_proba = tree_predictions.mean(axis=0)
        std_proba = tree_predictions.std(axis=0)
        
        # Calculate confidence interval using normal approximation
        # z-score for 95% CI is approximately 1.96
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        lower = np.clip(mean_proba - z_score * std_proba, 0, 1)
        upper = np.clip(mean_proba + z_score * std_proba, 0, 1)
        
        return mean_proba, lower, upper
    
    def get_feature_importance(self, feature_names: List[str] = None) -> Dict[str, float]:
        """Get feature importance scores."""
        importance = self.model.feature_importances_
        
        if feature_names is None:
            return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}
        
        return {name: float(imp) for name, imp in zip(feature_names, importance)}
    
    def get_prediction_variance(self, X) -> np.ndarray:
        """
        Get prediction variance for uncertainty quantification.
        Higher variance indicates less confident predictions.
        """
        tree_predictions = np.array([
            tree.predict_proba(X)[:, 1] for tree in self.model.estimators_
        ])
        return tree_predictions.var(axis=0)

    def save(self, filepath):
        """Save model to file."""
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        """Load model from file."""
        self.model = joblib.load(filepath)
        self._is_fitted = True


class SegmentModel:
    """K-Means clustering for borrower segmentation."""
    
    def __init__(self, n_clusters: int = 4):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.n_clusters = n_clusters
    
    def train(self, X):
        """Train the segmentation model."""
        self.kmeans.fit(X)
    
    def predict(self, X):
        """Predict cluster assignments."""
        return self.kmeans.predict(X)
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centroids."""
        return self.kmeans.cluster_centers_
    
    def get_cluster_distances(self, X) -> np.ndarray:
        """Get distances to each cluster center."""
        return self.kmeans.transform(X)
    
    def save(self, filepath):
        """Save model to file."""
        joblib.dump(self.kmeans, filepath)
    
    def load(self, filepath):
        """Load model from file."""
        self.kmeans = joblib.load(filepath)

