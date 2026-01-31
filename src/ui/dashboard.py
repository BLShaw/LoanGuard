import streamlit as st
import plotly.express as px
from src.utils import format_currency, format_percentage

def render(df):
    st.header("Executive Dashboard")
    st.markdown("Real-time overview of portfolio health and recovery performance.")
    
    # Top Level Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Portfolio Value", format_currency(df['Loan_Amount'].sum()))
    with m2:
        rec_rate = (df['Recovery_Status'] == 'Fully Recovered').mean() * 100
        st.metric("Recovery Rate", format_percentage(rec_rate))
    with m3:
        high_risk_vol = df[df['Risk_Label'] == 'High']['Outstanding_Loan_Amount'].sum()
        st.metric("At-Risk Amount", format_currency(high_risk_vol))
    with m4:
        st.metric("Active Borrowers", len(df))

    st.markdown("---")

    # Main Charts
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Recovery Performance Trend")
        fig_trend = px.bar(
            df.groupby(['Payment_History', 'Recovery_Status'])['Loan_Amount'].sum().reset_index(),
            x='Payment_History',
            y='Loan_Amount',
            color='Recovery_Status',
            title="Loan Volume by Payment Behavior & Status",
            color_discrete_map={
                "Fully Recovered": "#28a745",
                "Partially Recovered": "#ffc107",
                "Written Off": "#dc3545"
            },
            barmode='stack'
        )
        fig_trend.update_layout(height=400)
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with c2:
        st.subheader("Portfolio Composition")
        fig_sun = px.sunburst(
            df,
            path=['Segment_Name', 'Recovery_Status'],
            values='Outstanding_Loan_Amount',
            color='Recovery_Status',
            color_discrete_map={
                "Fully Recovered": "#28a745",
                "Partially Recovered": "#ffc107",
                "Written Off": "#dc3545",
                "(?)": "#dddddd"  # Fallback for unmapped segments
            },
            title="Exposure by Segment & Status"
        )
        fig_sun.update_layout(height=400)
        st.plotly_chart(fig_sun, use_container_width=True)