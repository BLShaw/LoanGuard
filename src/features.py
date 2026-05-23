import pandas as pd
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.numeric_features = [
            "Monthly_Income", "Loan_Amount", "Loan_Tenure", "Months_On_Book",
            "Interest_Rate", "Collateral_Value", "Outstanding_Loan_Amount",
            "Monthly_EMI", "Num_Missed_Payments", "Days_Past_Due",
            "Debt_to_Income_Ratio", "Loan_to_Value_Ratio"
        ] 

    def _add_engineered_features(self, df):
        df = df.copy()
        
        # Debt-to-Income (DTI) Ratio = Monthly EMI / Monthly Income
        if "Monthly_EMI" in df.columns and "Monthly_Income" in df.columns:
            df["Debt_to_Income_Ratio"] = df["Monthly_EMI"] / df["Monthly_Income"].replace(0, 1)
            df["Debt_to_Income_Ratio"] = df["Debt_to_Income_Ratio"].clip(0, 5.0)
        else:
            df["Debt_to_Income_Ratio"] = pd.Series(0.0, index=df.index)
            
        # Loan-to-Value (LTV) Ratio = Outstanding Amount / Collateral Value
        if "Outstanding_Loan_Amount" in df.columns and "Collateral_Value" in df.columns:
            df["Loan_to_Value_Ratio"] = df["Outstanding_Loan_Amount"] / df["Collateral_Value"].replace(0, 1)
            df["Loan_to_Value_Ratio"] = df["Loan_to_Value_Ratio"].clip(0, 10.0)
        else:
            df["Loan_to_Value_Ratio"] = pd.Series(0.0, index=df.index)
            
        return df

    def preprocess_for_training(self, df):
        """
        Prepares data for training:
        1. Encodes targets
        2. Scales numeric features
        """
        df = df.copy()
        
        if 'Recovery_Status' in df.columns:
            df = df[df['Recovery_Status'] != 'Pending Resolution'].copy()
        
        # Target Definition (Ground Truth)
        # 1 = High Risk (Not Fully Recovered), 0 = Low Risk (Fully Recovered)
        df["High_Risk_Flag"] = df["Recovery_Status"].apply(
            lambda x: 1 if x != "Fully Recovered" else 0
        )

        df = self._add_engineered_features(df)
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
        df = self._add_engineered_features(df)
        X = df[self.numeric_features]
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=self.numeric_features, index=df.index)