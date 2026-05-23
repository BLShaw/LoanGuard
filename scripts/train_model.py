import sys
import os
import joblib

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import DATA_PATH, MODEL_DIR, RISK_MODEL_PATH, SEGMENT_MODEL_PATH, SCALER_PATH
from src.data_loader import load_data, clean_data
from src.features import FeatureEngineer
from src.model import RiskModel, SegmentModel

def calculate_ks_statistic(y_true, y_prob):
    """Calculate Kolmogorov-Smirnov statistic for default vs non-default separation."""
    from scipy.stats import ks_2samp
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    if len(pos_probs) == 0 or len(neg_probs) == 0:
        return 0.0
    return ks_2samp(pos_probs, neg_probs).statistic

def main():
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = load_data(DATA_PATH)
    except FileNotFoundError:
        print("Error: Data file not found. Ensure loan-recovery.csv is in data/ folder.")
        sys.exit(1)

    df = clean_data(df)

    print("Feature Engineering...")
    fe = FeatureEngineer()
    X, y = fe.preprocess_for_training(df)
    
    # Stratified Train/Test Split (80/20) for compliance evaluation
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Data split - Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")

    print("Training Risk Model...")
    risk_model = RiskModel()
    risk_model.train(X_train, y_train)
    
    # Calculate Validation Metrics
    test_probs = risk_model.predict_proba(X_test)[:, 1]
    test_preds = risk_model.predict(X_test)
    
    test_accuracy = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs)
    test_gini = 2.0 * test_auc - 1.0
    test_ks = calculate_ks_statistic(y_test, test_probs)
    
    print("\n=== Model Validation Metrics (Held-out Test Split) ===")
    print(f"ROC-AUC Score:     {test_auc:.4f}")
    print(f"Gini Coefficient:   {test_gini:.4f}")
    print(f"KS-Statistic:       {test_ks:.4f}")
    print(f"F1-Score:           {test_f1:.4f}")
    print(f"Accuracy:           {test_accuracy:.4f}\n")
    
    # Model Promotion Gate: Validate baseline quality
    if test_auc < 0.70:
        print("[REJECTED] MODEL PROMOTION REJECTED! Test AUC falls below regulatory threshold (0.70). Existing model preserved.")
        sys.exit(1)
        
    print("[APPROVED] Model promotion approved! Saving artifacts...")
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    joblib.dump(fe, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")
    
    risk_model.save(RISK_MODEL_PATH)
    print(f"Risk model saved to {RISK_MODEL_PATH}")

    print("Training Segmentation Model...")
    segment_model = SegmentModel()
    segment_model.train(X_train)  # Train cluster centroids on same train split
    segment_model.save(SEGMENT_MODEL_PATH)
    print(f"Segmentation model saved to {SEGMENT_MODEL_PATH}")

    print("Training Complete.")

if __name__ == "__main__":
    main()