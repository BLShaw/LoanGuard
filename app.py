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

# --- Main App Logic ---
def main():
    # Load Resources
    risk_model, segment_model, fe, explainer = load_system()
    df = load_and_process_data()

    if risk_model is None or df.empty:
        st.error("System is initializing or data is missing. Please run the training script first.")
        st.stop()

    # Inference Pipeline
    X_inference = fe.preprocess_for_inference(df)
    df["Borrower_Segment"] = segment_model.predict(X_inference)
    # NOTE: Cluster IDs are arbitrary. If model is retrained, verify these labels
    # by analyzing cluster centroids (e.g., mean income, loan amount per cluster).
    segment_mapping = {
        0: "Moderate Income, High Burden",
        1: "High Income, Low Risk",
        2: "Moderate Income, Medium Risk",
        3: "High Loan, High Risk",
    }
    df["Segment_Name"] = df["Borrower_Segment"].map(segment_mapping)
    risk_probs = risk_model.predict_proba(X_inference)
    df["Risk_Score"] = risk_probs[:, 1]
    
    def strategy_rule(score):
        if score > 0.75: return "Legal Action"
        elif score > 0.50: return "Settlement Offer"
        else: return "Standard Monitoring"
        
    df["Recovery_Strategy"] = df["Risk_Score"].apply(strategy_rule)
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