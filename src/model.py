import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import os

class RiskModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, filepath):
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        self.model = joblib.load(filepath)

class SegmentModel:
    def __init__(self, n_clusters=4):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    def train(self, X):
        self.kmeans.fit(X)
        
    def predict(self, X):
        return self.kmeans.predict(X)
    
    def save(self, filepath):
        joblib.dump(self.kmeans, filepath)
    
    def load(self, filepath):
        self.kmeans = joblib.load(filepath)
