import streamlit as st
import plotly.express as px

def render(df):
    st.header("Portfolio Management")
    
    # Filters
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    with col_filter1:
        f_segment = st.multiselect("Filter by Segment", df['Segment_Name'].unique())
    with col_filter2:
        f_status = st.multiselect("Filter by Status", df['Recovery_Status'].unique())
    with col_filter3:
        f_strategy = st.multiselect("Filter by Strategy", df['Recovery_Strategy'].unique())
        
    # Apply Filters
    df_filtered = df.copy()
    if f_segment: df_filtered = df_filtered[df_filtered['Segment_Name'].isin(f_segment)]
    if f_status: df_filtered = df_filtered[df_filtered['Recovery_Status'].isin(f_status)]
    if f_strategy: df_filtered = df_filtered[df_filtered['Recovery_Strategy'].isin(f_strategy)]
    
    # Borrower Journey Flow
    if not df_filtered.empty:
        st.subheader("Borrower Journey Flow")
        st.markdown("Trace how Employment Type and Payment History impact Recovery Status.")
        
        plot_df = df_filtered.sample(min(1000, len(df_filtered)), random_state=42)
        
        fig_flow = px.parallel_categories(
            plot_df,
            dimensions=['Employment_Type', 'Payment_History', 'Recovery_Status', 'Recovery_Strategy'],
            color="Risk_Score",
            color_continuous_scale=px.colors.diverging.RdYlGn[::-1],
            labels={'Employment_Type': 'Employment', 'Payment_History': 'History', 'Recovery_Status': 'Status'}
        )
        fig_flow.update_layout(height=500)
        st.plotly_chart(fig_flow, use_container_width=True)

    # Detailed Register
    st.subheader("Detailed Loan Register")
    st.dataframe(
        df_filtered[[
            "Borrower_ID", "Monthly_Income", "Loan_Amount", "Outstanding_Loan_Amount", 
            "Recovery_Status", "Risk_Score", "Recovery_Strategy"
        ]],
        column_config={
            "Risk_Score": st.column_config.ProgressColumn(
                "Risk Score",
                help="Probability of default (0-1)",
                min_value=0,
                max_value=1,
                format="%.2f",
            ),
            "Monthly_Income": st.column_config.NumberColumn(format="$%d"),
            "Loan_Amount": st.column_config.NumberColumn(format="$%d"),
            "Outstanding_Loan_Amount": st.column_config.NumberColumn(format="$%d"),
            "Recovery_Status": st.column_config.TextColumn("Status"),
        },
        use_container_width=True,
        hide_index=True,
        height=500
    )