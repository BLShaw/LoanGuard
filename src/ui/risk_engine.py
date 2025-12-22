import streamlit as st
import plotly.express as px
import pandas as pd

def render(df):
    st.header("Risk Analysis Engine")
    
    # Correlation Matrix
    st.subheader("Risk Factor Correlations")
    corr_cols = ['Risk_Score', 'Loan_Amount', 'Monthly_Income', 'Age', 'Num_Missed_Payments', 'Days_Past_Due', 'Collateral_Value']
    corr = df[corr_cols].corr()
    
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Heatmap"
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    r1, r2 = st.columns([1, 1])
    
    with r1:
        st.subheader("Risk Distribution")
        fig_risk_dist = px.histogram(
            df, 
            x="Risk_Score", 
            nbins=50, 
            title="Distribution of Risk Scores",
            color="Recovery_Status",
            color_discrete_map={
                "Fully Recovered": "#28a745",
                "Partially Recovered": "#ffc107",
                "Written Off": "#dc3545"
            },
            marginal="box"
        )
        fig_risk_dist.update_layout(height=400)
        st.plotly_chart(fig_risk_dist, use_container_width=True)

    with r2:
        st.subheader("Income vs Risk Clusters")
        fig_seg = px.scatter(
            df,
            x="Monthly_Income",
            y="Loan_Amount",
            color="Risk_Score",
            size="Outstanding_Loan_Amount",
            hover_data=["Borrower_ID", "Segment_Name"],
            title="Risk Intensity by Loan Size & Income",
            color_continuous_scale="RdYlGn_r"
        )
        fig_seg.update_layout(height=400)
        st.plotly_chart(fig_seg, use_container_width=True)