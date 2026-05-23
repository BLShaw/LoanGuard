import streamlit as st
import pandas as pd
import joblib
import shap
import os
from datetime import datetime

# Config & Utils
from config.settings import (
    APP_TITLE, APP_ICON, APP_VERSION, 
    DATA_PATH, RISK_MODEL_PATH, SEGMENT_MODEL_PATH, SCALER_PATH
)
from src.data_loader import load_data, clean_data
from src.features import FeatureEngineer
from src.model import RiskModel, SegmentModel

# UI Modules
from src.ui import dashboard, portfolio, risk_engine, customer_360, what_if, analytics

# Decision Intelligence
from src.decision_log import get_logger

# --- Configuration ---
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Resource Loading ---
@st.cache_resource
def load_system():
    if not os.path.exists(RISK_MODEL_PATH):
        return None, None, None, None
    try:
        risk_model = RiskModel()
        risk_model.load(RISK_MODEL_PATH)
        segment_model = SegmentModel()
        segment_model.load(SEGMENT_MODEL_PATH)
        fe = joblib.load(SCALER_PATH)
        
        # SHAP Explainer
        explainer = shap.TreeExplainer(risk_model.model)
        
        return risk_model, segment_model, fe, explainer
    except Exception as e:
        return None, None, None, None

@st.cache_data
def load_and_process_data():
    try:
        df = load_data(DATA_PATH)
        df = clean_data(df)
        return df
    except Exception:
        return pd.DataFrame()

def check_and_initialize_system():
    """Ensure data is generated and models are trained on first run."""
    import os
    
    # 1. Check if dataset exists
    if not os.path.exists(DATA_PATH):
        with st.spinner("First-time setup: Generating synthetic loan recovery dataset..."):
            try:
                from scripts.generate_data import generate_data
                df = generate_data()
                os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
                df.to_csv(DATA_PATH, index=False)
                st.success("Dataset generated successfully!")
            except Exception as e:
                st.error(f"Failed to generate dataset: {str(e)}")
                st.stop()
                
    # 2. Check if models exist
    if not os.path.exists(RISK_MODEL_PATH) or not os.path.exists(SEGMENT_MODEL_PATH) or not os.path.exists(SCALER_PATH):
        with st.spinner("First-time setup: Training risk and segmentation models..."):
            try:
                from scripts.train_model import main as train_main
                train_main()
                st.success("Models trained and saved successfully!")
            except SystemExit as se:
                if se.code != 0:
                    st.error("Model promotion gate failed! Model test AUC was below 0.70.")
                    st.stop()
            except Exception as e:
                st.error(f"Failed to train models: {str(e)}")
                st.stop()

# --- Main App Logic ---
def main():
    # Ensure system files and models are initialized
    check_and_initialize_system()
    
    # Load Resources
    risk_model, segment_model, fe, explainer = load_system()
    df = load_and_process_data()

    if risk_model is None or df.empty:
        st.error("Critical Error: Unable to load system resources. Please check your logs.")
        st.stop()

    # Inference Pipeline
    X_inference = fe.preprocess_for_inference(df)
    df["Borrower_Segment"] = segment_model.predict(X_inference)
    import json
    segment_mapping_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "segment_mapping.json")
    try:
        with open(segment_mapping_path, "r") as f:
            segment_mapping = {int(k): v for k, v in json.load(f).items()}
    except FileNotFoundError:
        segment_mapping = {0: "Segment A", 1: "Segment B", 2: "Segment C", 3: "Segment D"}
    df["Segment_Name"] = df["Borrower_Segment"].map(segment_mapping)
    risk_probs = risk_model.predict_proba(X_inference)
    df["Risk_Score"] = risk_probs[:, 1]
    
    from src.optimizer import get_optimizer
    optimizer = get_optimizer()
    # Generate compliance attributes (deterministic based on borrower ID)
    scra_flags = []
    bankruptcy_flags = []
    import random
    for bid in df['Borrower_ID']:
        borrower_seed = hash(bid) % (2**32)
        rng = random.Random(borrower_seed)
        scra_flags.append(rng.random() < 0.05)       # 5% active duty military
        bankruptcy_flags.append(rng.random() < 0.03)  # 3% active bankruptcy
    df["Is_SCRA"] = scra_flags
    df["Is_Bankrupt"] = bankruptcy_flags

    strategies = []
    for score, segment, balance, scra, bankrupt, status in zip(
        df["Risk_Score"], df["Segment_Name"], df["Outstanding_Loan_Amount"], df["Is_SCRA"], df["Is_Bankrupt"], df["Recovery_Status"]
    ):
        if status == "Fully Recovered":
            strategies.append("No Action Required")
        else:
            rec = optimizer.recommend_action(
                score, float(balance), segment, is_scra_active=scra, is_bankrupt=bankrupt, explore=False
            )
            strategies.append(rec.action)
    df["Recovery_Strategy"] = strategies
    df["Risk_Label"] = df["Risk_Score"].apply(lambda x: "High" if x > 0.5 else "Low")

    # Log predictions (only on first load, not every rerun)
    if 'predictions_logged' not in st.session_state:
        logger = get_logger()
        for _, row in df.head(10).iterrows():  # Sample to avoid log bloat
            logger.log_prediction(
                borrower_id=row['Borrower_ID'],
                risk_score=row['Risk_Score'],
                recommended_strategy=row['Recovery_Strategy'],
                segment=row['Segment_Name']
            )
        st.session_state.predictions_logged = True

    # Sidebar Navigation
    with st.sidebar:
        st.title("🏦 LoanGuard")
        st.markdown("### Intelligent Recovery")
        st.markdown("---")
        
        page = st.radio("Navigation", [
            "Dashboard Overview", 
            "Portfolio Management",
            "Risk Analysis Engine",
            "Customer 360",
            "What-If Simulator",
            "Analytics Hub"
        ])
        
        st.markdown("---")
        st.caption(f"System Status: Online")
        st.caption(f"Last Sync: {datetime.now().strftime('%H:%M:%S')}")
        st.caption(APP_VERSION)

    # Router
    if page == "Dashboard Overview":
        dashboard.render(df)
    elif page == "Portfolio Management":
        portfolio.render(df)
    elif page == "Risk Analysis Engine":
        risk_engine.render(df)
    elif page == "Customer 360":
        customer_360.render(df, explainer, fe, X_inference)
    elif page == "What-If Simulator":
        what_if.render(df, risk_model, fe, explainer)
    elif page == "Analytics Hub":
        analytics.render(df, risk_model, fe)

if __name__ == "__main__":
    main()