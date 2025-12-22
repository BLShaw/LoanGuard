import sys
import os
import joblib

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import DATA_PATH, MODEL_DIR, RISK_MODEL_PATH, SEGMENT_MODEL_PATH, SCALER_PATH
from src.data_loader import load_data, clean_data
from src.features import FeatureEngineer
from src.model import RiskModel, SegmentModel

def main():
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = load_data(DATA_PATH)
    except FileNotFoundError:
        print("Error: Data file not found. Ensure loan-recovery.csv is in data/ folder.")
        return

    df = clean_data(df)

    print("Feature Engineering...")
    fe = FeatureEngineer()
    X, y = fe.preprocess_for_training(df)
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    joblib.dump(fe, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")

    print("Training Risk Model...")
    risk_model = RiskModel()
    risk_model.train(X, y)
    risk_model.save(RISK_MODEL_PATH)
    print(f"Risk model saved to {RISK_MODEL_PATH}")

    print("Training Segmentation Model...")
    segment_model = SegmentModel()
    segment_model.train(X)
    segment_model.save(SEGMENT_MODEL_PATH)
    print(f"Segmentation model saved to {SEGMENT_MODEL_PATH}")

    print("Training Complete.")

if __name__ == "__main__":
    main()