"""
What-If Simulator: Decision Intelligence module for scenario analysis.
Allows users to adjust borrower parameters and see real-time risk predictions.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from src.decision_log import get_logger


def render(df, risk_model, fe, explainer):
    """Render the What-If Simulator interface."""
    st.header("🔮 What-If Simulator")
    st.markdown("Explore how changes to borrower attributes affect risk scores and recovery strategies.")
    
    # --- Borrower Selection ---
    col_select, col_info = st.columns([1, 2])
    
    with col_select:
        selected_id = st.selectbox(
            "Select Baseline Borrower",
            df['Borrower_ID'].unique(),
            help="Choose a borrower to use as the starting point for simulation"
        )
    
    if not selected_id:
        st.info("Please select a borrower to begin simulation.")
        return
    
    # Get baseline borrower data
    baseline = df[df['Borrower_ID'] == selected_id].iloc[0]
    
    with col_info:
        st.info(f"**Baseline:** {baseline['Employment_Type']} | Age {int(baseline['Age'])} | "
                f"Segment: {baseline['Segment_Name']}")
    
    st.markdown("---")
    
    # --- Interactive Sliders ---
    st.subheader("📊 Adjust Risk Factors")
    
    # Helper function to ensure valid slider ranges
    def safe_range(value, factor=0.5, min_bound=1000):
        """Return min/max ensuring valid slider range."""
        low = max(int(value * (1 - factor)), 0)
        high = max(int(value * (1 + factor)), min_bound)
        if low >= high:
            low = 0
            high = max(int(value), min_bound)
        return low, high
    
    col1, col2 = st.columns(2)
    
    with col1:
        income_min, income_max = safe_range(baseline['Monthly_Income'], 0.5, 5000)
        sim_income = st.slider(
            "Monthly Income ($)",
            min_value=income_min,
            max_value=income_max,
            value=int(baseline['Monthly_Income']),
            step=500,
            help="Adjust monthly income (±50% of baseline)"
        )
        
        sim_outstanding = st.slider(
            "Outstanding Loan Amount ($)",
            min_value=0,
            max_value=max(int(baseline['Loan_Amount']), 1000),
            value=int(baseline['Outstanding_Loan_Amount']),
            step=1000,
            help="Amount still owed on the loan"
        )
        
        collateral_min, collateral_max = safe_range(baseline['Collateral_Value'], 0.5, 10000)
        sim_collateral = st.slider(
            "Collateral Value ($)",
            min_value=collateral_min,
            max_value=collateral_max,
            value=min(max(int(baseline['Collateral_Value']), collateral_min), collateral_max),
            step=1000,
            help="Value of assets securing the loan"
        )
    
    with col2:
        sim_missed = st.slider(
            "Number of Missed Payments",
            min_value=0,
            max_value=12,
            value=int(baseline['Num_Missed_Payments']),
            help="Total EMI payments missed"
        )
        
        sim_dpd = st.slider(
            "Days Past Due",
            min_value=0,
            max_value=365,
            value=int(baseline['Days_Past_Due']),
            help="Days since last payment was due"
        )
        
        emi_min, emi_max = safe_range(baseline['Monthly_EMI'], 0.5, 500)
        sim_emi = st.slider(
            "Monthly EMI ($)",
            min_value=emi_min,
            max_value=emi_max,
            value=min(max(int(baseline['Monthly_EMI']), emi_min), emi_max),
            step=100,
            help="Monthly installment amount"
        )
    
    # --- Build Simulated Feature Vector ---
    sim_data = {
        'Age': baseline['Age'],
        'Monthly_Income': sim_income,
        'Loan_Amount': baseline['Loan_Amount'],
        'Loan_Tenure': baseline['Loan_Tenure'],
        'Interest_Rate': baseline['Interest_Rate'],
        'Collateral_Value': sim_collateral,
        'Outstanding_Loan_Amount': sim_outstanding,
        'Monthly_EMI': sim_emi,
        'Num_Missed_Payments': sim_missed,
        'Days_Past_Due': sim_dpd
    }
    
    sim_df = pd.DataFrame([sim_data])
    sim_scaled = fe.scaler.transform(sim_df[fe.numeric_features])
    sim_X = pd.DataFrame(sim_scaled, columns=fe.numeric_features)
    
    # Get predictions with confidence intervals
    sim_proba = risk_model.predict_proba(sim_X)
    sim_risk_score = sim_proba[0, 1]
    
    # Get confidence intervals if available
    try:
        mean_proba, ci_lower, ci_upper = risk_model.predict_proba_with_ci(sim_X, confidence=0.95)
        has_ci = True
        sim_ci_lower = ci_lower[0]
        sim_ci_upper = ci_upper[0]
    except Exception:
        has_ci = False
        sim_ci_lower = sim_ci_upper = None
    
    # Strategy mapping
    def get_strategy(score):
        if score > 0.75:
            return "Legal Action"
        elif score > 0.50:
            return "Settlement Offer"
        else:
            return "Standard Monitoring"
    
    sim_strategy = get_strategy(sim_risk_score)
    original_strategy = baseline['Recovery_Strategy']
    
    # Log the simulation
    logger = get_logger()
    logger.log_what_if_simulation(
        borrower_id=selected_id,
        original_risk_score=float(baseline['Risk_Score']),
        simulated_risk_score=float(sim_risk_score),
        parameters_changed={
            'Monthly_Income': sim_income,
            'Outstanding_Loan_Amount': sim_outstanding,
            'Collateral_Value': sim_collateral,
            'Num_Missed_Payments': sim_missed,
            'Days_Past_Due': sim_dpd
        }
    )
    
    st.markdown("---")
    
    # --- Results Comparison ---
    st.subheader("📈 Simulation Results")
    
    res1, res2, res3 = st.columns(3)
    
    with res1:
        st.markdown("#### Original")
        st.metric("Risk Score", f"{baseline['Risk_Score']:.3f}")
        st.metric("Strategy", original_strategy)
    
    with res2:
        st.markdown("#### Simulated")
        delta = sim_risk_score - baseline['Risk_Score']
        delta_str = f"{delta:+.3f}"
        st.metric("Risk Score", f"{sim_risk_score:.3f}", delta=delta_str, delta_color="inverse")
        
        # Show confidence interval if available
        if has_ci:
            st.caption(f"95% CI: [{sim_ci_lower:.3f}, {sim_ci_upper:.3f}]")
        
        strategy_changed = sim_strategy != original_strategy
        if strategy_changed:
            st.metric("Strategy", sim_strategy, delta="Changed!", delta_color="off")
        else:
            st.metric("Strategy", sim_strategy)
    
    with res3:
        st.markdown("#### Impact")
        pct_change = ((sim_risk_score - baseline['Risk_Score']) / baseline['Risk_Score']) * 100 if baseline['Risk_Score'] > 0 else 0
        
        if delta > 0.05:
            st.error(f"⬆️ Risk increased by {pct_change:.1f}%")
        elif delta < -0.05:
            st.success(f"⬇️ Risk decreased by {abs(pct_change):.1f}%")
        else:
            st.info("↔️ Minimal impact on risk")
    
    # --- Dual Gauge Visualization ---
    st.markdown("---")
    st.subheader("🎯 Risk Score Comparison")
    
    fig = go.Figure()
    
    # Original gauge (left)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=baseline['Risk_Score'],
        title={'text': "Original", 'font': {'size': 16}},
        domain={'x': [0, 0.45], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "#6c757d"},
            'steps': [
                {'range': [0, 0.5], 'color': "#d1e7dd"},
                {'range': [0.5, 0.75], 'color': "#fff3cd"},
                {'range': [0.75, 1], 'color': "#f8d7da"}
            ]
        }
    ))
    
    # Simulated gauge (right)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=sim_risk_score,
        title={'text': "Simulated", 'font': {'size': 16}},
        domain={'x': [0.55, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "#1a237e"},
            'steps': [
                {'range': [0, 0.5], 'color': "#d1e7dd"},
                {'range': [0.5, 0.75], 'color': "#fff3cd"},
                {'range': [0.75, 1], 'color': "#f8d7da"}
            ]
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Factor Changes Table ---
    st.markdown("---")
    st.subheader("🔍 Parameter Changes")
    
    changes = []
    params = [
        ('Monthly Income', baseline['Monthly_Income'], sim_income, '$'),
        ('Outstanding Amount', baseline['Outstanding_Loan_Amount'], sim_outstanding, '$'),
        ('Collateral Value', baseline['Collateral_Value'], sim_collateral, '$'),
        ('Monthly EMI', baseline['Monthly_EMI'], sim_emi, '$'),
        ('Missed Payments', baseline['Num_Missed_Payments'], sim_missed, ''),
        ('Days Past Due', baseline['Days_Past_Due'], sim_dpd, ' days'),
    ]
    
    for name, orig, sim, unit in params:
        if orig != sim:
            pct = ((sim - orig) / orig * 100) if orig != 0 else 0
            changes.append({
                'Parameter': name,
                'Original': f"{unit}{orig:,.0f}" if unit == '$' else f"{int(orig)}{unit}",
                'Simulated': f"{unit}{sim:,.0f}" if unit == '$' else f"{int(sim)}{unit}",
                'Change': f"{pct:+.1f}%"
            })
    
    if changes:
        st.dataframe(pd.DataFrame(changes), use_container_width=True, hide_index=True)
    else:
        st.info("No parameters have been modified from baseline.")
    
    # --- SHAP Comparison (if explainer available) ---
    if explainer:
        st.markdown("---")
        st.subheader("🧠 Explainability: What Drove the Change?")
        
        try:
            # Get SHAP values for simulated scenario
            shap_values = explainer.shap_values(sim_X)
            
            if isinstance(shap_values, list):
                impact = shap_values[1][0]
            elif len(shap_values.shape) == 3:
                impact = shap_values[0, :, 1]
            else:
                impact = shap_values[0]
            
            impact = np.array(impact).flatten()
            
            shap_df = pd.DataFrame({
                'Feature': fe.numeric_features,
                'Impact': impact
            })
            shap_df['AbsImpact'] = shap_df['Impact'].abs()
            shap_df = shap_df.sort_values('AbsImpact', ascending=True).tail(6)
            shap_df['Color'] = shap_df['Impact'].apply(lambda x: '#dc3545' if x > 0 else '#28a745')
            
            fig_shap = px.bar(
                shap_df,
                x='Impact',
                y='Feature',
                orientation='h',
                title="SHAP Feature Contributions (Simulated Scenario)",
                text_auto='.3f'
            )
            fig_shap.update_traces(marker_color=shap_df['Color'])
            fig_shap.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_shap, use_container_width=True)
            
            st.caption("🔴 Red = increases risk | 🟢 Green = decreases risk")
            
        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {e}")
