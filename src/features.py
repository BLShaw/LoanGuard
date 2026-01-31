import pandas as pd
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.numeric_features = [
            "Age", "Monthly_Income", "Loan_Amount", "Loan_Tenure",
            "Interest_Rate", "Collateral_Value", "Outstanding_Loan_Amount",
            "Monthly_EMI", "Num_Missed_Payments", "Days_Past_Due"
        ] 

    def preprocess_for_training(self, df):
        """
        Prepares data for training:
        1. Encodes targets
        2. Scales numeric features
        """
        df = df.copy()
        
        # Target Definition (Ground Truth)
        # 1 = High Risk (Not Fully Recovered), 0 = Low Risk (Fully Recovered)
        df["High_Risk_Flag"] = df["Recovery_Status"].apply(
            lambda x: 1 if x != "Fully Recovered" else 0
        )

        X = df[self.numeric_features]
        y = df["High_Risk_Flag"]
        
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)
        X_processed = pd.DataFrame(X_scaled, columns=self.numeric_features, index=df.index)
        
        return X_processed, y

    def preprocess_for_inference(self, df):
        """Prepares data for inference using fitted scaler."""
        if not hasattr(self.scaler, 'mean_'):
            raise RuntimeError("Scaler not fitted. Run preprocess_for_training first or load a fitted scaler.")
        X = df[self.numeric_features]
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=self.numeric_features, index=df.index)