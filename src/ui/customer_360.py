import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from src.utils import format_currency, get_status_color, generate_borrower_history

def render(df, explainer, fe, X_inference):
    st.header("Customer 360 View")
    
    # Search
    c_search_col, _ = st.columns([1, 2])
    with c_search_col:
        selected_id = st.selectbox("Search Borrower ID", df['Borrower_ID'].unique())
    
    if selected_id:
        cust = df[df['Borrower_ID'] == selected_id].iloc[0]
        
        # Profile Header
        status_msg = f"{selected_id} | {cust['Employment_Type']} | {cust['Age']} Years Old"
        status_color = get_status_color(cust['Recovery_Status'])
        
        if status_color == 'green':
            st.success(status_msg, icon="✅")
        elif status_color == 'orange':
            st.warning(status_msg, icon="⚠️")
        else:
            st.error(status_msg, icon="🚨")
        
        st.write("") # Spacer
        
        # 3-Column Detail Layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 💰 Financial Profile")
            st.write(f"**Monthly Income:** {format_currency(cust['Monthly_Income'])}")
            st.write(f"**Loan Amount:** {format_currency(cust['Loan_Amount'])}")
            st.write(f"**Outstanding:** {format_currency(cust['Outstanding_Loan_Amount'])}")
            st.write(f"**Monthly EMI:** {format_currency(cust['Monthly_EMI'])}")
            
        with col2:
            st.markdown("#### ⚠️ Risk Profile")
            
            # Risk Score Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = cust['Risk_Score'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "AI Risk Score"},
                gauge = {
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "#d1e7dd"},
                        {'range': [0.5, 0.75], 'color': "#fff3cd"},
                        {'range': [0.75, 1], 'color': "#f8d7da"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': cust['Risk_Score']
                    }
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.write(f"**Segment:** {cust['Segment_Name']}")
            st.write(f"**Missed Payments:** {int(cust['Num_Missed_Payments'])}")
            
            # SHAP Analysis
            if explainer:
                try:
                    st.markdown("##### 🕵️ Why this Score?")
                    
                    borrower_idx = df[df['Borrower_ID'] == selected_id].index[0]
                    X_target = X_inference.iloc[[borrower_idx]]
                    
                    shap_values = explainer.shap_values(X_target)
                    
                    if isinstance(shap_values, list):
                        raw_impact = shap_values[1][0]
                    elif len(shap_values.shape) == 3:
                        raw_impact = shap_values[0, :, 1]
                    else:
                        raw_impact = shap_values[0]

                    impact = np.array(raw_impact).flatten()

                    shap_df = pd.DataFrame({
                        'Feature': fe.numeric_features,
                        'Impact': impact
                    })
                    
                    shap_df['AbsImpact'] = shap_df['Impact'].abs()
                    shap_df = shap_df.sort_values('AbsImpact', ascending=True).tail(5)
                    
                    shap_df['Color'] = shap_df['Impact'].apply(lambda x: '#dc3545' if x > 0 else '#28a745')
                    
                    fig_shap = px.bar(
                        shap_df,
                        x='Impact',
                        y='Feature',
                        orientation='h',
                        title="Top Factors Influencing Risk",
                        text_auto='.3f'
                    )
                    fig_shap.update_traces(marker_color=shap_df['Color'])
                    fig_shap.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig_shap, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Could not generate explanation: {e}")

        with col3:
            st.markdown("#### 📋 Action Plan")
            st.info(f"Recommended Strategy:\n\n**{cust['Recovery_Strategy']}**")
            st.write(f"**Current Status:** {cust['Recovery_Status']}")
            st.write(f"**Legal Action:** {cust['Legal_Action_Taken']}")

        # Activity Log
        st.markdown("---")
        st.subheader("Collection Activity Log")
        
        activity_data = generate_borrower_history(cust)
        
        st.dataframe(
            activity_data, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Date": st.column_config.TextColumn("Date", width="medium"),
                "Action": st.column_config.TextColumn("Event Type", width="large"),
                "Agent": st.column_config.TextColumn("Executed By", width="medium"),
                "Result": st.column_config.TextColumn("Outcome/Status", width="large"),
            }
        )