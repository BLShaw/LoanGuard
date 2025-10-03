import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import re

# Set page config
st.set_page_config(
    page_title="LoanGuard: Smart Loan Recovery System Dashboard",
    page_icon="💳",
    layout="wide"
)

# Define recovery strategy function
def assign_recovery_strategy(risk_score):
    if risk_score > 0.75:
        return "Immediate legal notices & aggressive recovery attempts"
    elif 0.50 <= risk_score <= 0.75:
        return "Settlement offers & repayment plans"
    else:
        return "Automated reminders & monitoring"

# Title
st.title("💳 LoanGuard Recovery System Dashboard")
st.markdown("""
This dashboard analyzes loan data to identify high-risk borrowers and recommend recovery strategies.
""")

@st.cache_data
def load_data():
    import os  # Import here to avoid caching issues
    
    # Validate file path to prevent path traversal
    file_path = 'loan-recovery.csv'
    if '..' in file_path or file_path.startswith('/'):
        raise ValueError("Invalid file path")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file {file_path} not found")
    
    # Add file size validation to prevent loading excessively large files
    file_size = os.path.getsize(file_path)
    max_size = 100 * 1024 * 1024  # 100MB limit
    if file_size > max_size:
        raise ValueError(f"Data file is too large: {file_size} bytes")
    
    df = pd.read_csv(file_path)
    return df

df = load_data()

# Data preprocessing (always run, needed for metrics)
df["Gender_encoded"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Employment_Type_encoded"] = df["Employment_Type"].map(
    {"Salaried": 0, "Self-Employed": 1, "Business Owner": 2}
)
df["High_Risk_Flag"] = df["Recovery_Status"].apply(
    lambda x: 1 if x != "Fully Recovered" else 0
)
df = df.fillna(df.mean(numeric_only=True))

# Borrower Segmentation (always run to ensure Segment_Name exists)
features = [
    "Age",
    "Monthly_Income",
    "Loan_Amount",
    "Loan_Tenure",
    "Interest_Rate",
    "Collateral_Value",
    "Outstanding_Loan_Amount",
    "Monthly_EMI",
    "Num_Missed_Payments",
    "Days_Past_Due",
]

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Borrower_Segment"] = kmeans.fit_predict(df_scaled)

# Define segment names
segment_mapping = {
    0: "Moderate Income, High Loan Burden",
    1: "High Income, Low Default Risk",
    2: "Moderate Income, Medium Risk",
    3: "High Loan, Higher Default Risk",
}

df["Segment_Name"] = df["Borrower_Segment"].map(segment_mapping)

# Sidebar
st.sidebar.header("Dashboard Settings")
show_visualizations = st.sidebar.checkbox("Show data visualizations", value=True)
show_segments = st.sidebar.checkbox("Show borrower segments", value=True)
show_risk_model = st.sidebar.checkbox("Show risk predictions", value=True)
show_raw_data = st.sidebar.checkbox("Show raw data", value=False)

# KPIs - Always shown at the top
st.subheader("📈 Key Performance Indicators")
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4, gap="small")

with kpi_col1:
    total_borrowers = len(df)
    st.metric(label="Total Borrowers", value=total_borrowers)

with kpi_col2:
    high_risk_count = df['High_Risk_Flag'].sum()
    st.metric(label="High Risk Borrowers", value=high_risk_count)

with kpi_col3:
    recovery_rate = (df[df['Recovery_Status'] == 'Fully Recovered'].shape[0] / total_borrowers) * 100
    st.metric(label="Recovery Rate", value=f"{recovery_rate:.1f}%")

with kpi_col4:
    avg_missed_payments = df['Num_Missed_Payments'].mean()
    st.metric(label="Avg. Missed Payments", value=f"{avg_missed_payments:.1f}")

# Additional KPIs
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4, gap="small")

with kpi_col1:
    avg_loan_amount = df['Loan_Amount'].mean()
    st.metric(label="Avg. Loan Amount", value=f"${avg_loan_amount:,.0f}")

with kpi_col2:
    avg_monthly_income = df['Monthly_Income'].mean()
    st.metric(label="Avg. Monthly Income", value=f"${avg_monthly_income:,.0f}")

with kpi_col3:
    fully_recovered_count = (df['Recovery_Status'] == 'Fully Recovered').sum()
    st.metric(label="Fully Recovered", value=fully_recovered_count)

with kpi_col4:
    written_off_count = (df['Recovery_Status'] == 'Written Off').sum()
    st.metric(label="Written Off", value=written_off_count)

# Segment distribution
st.subheader("👤 Borrower Segments Distribution")
seg_col1, seg_col2 = st.columns([1, 1], gap="large")
with seg_col1:
    segment_dist = df["Segment_Name"].value_counts()
    st.dataframe(segment_dist, use_container_width=True)
with seg_col2:
    fig = px.pie(
        values=segment_dist.values,
        names=segment_dist.index,
        title="Borrower Segment Distribution"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Risk Prediction Model
if show_risk_model:
    st.subheader("⚖️ Risk Prediction Model")
    
    # Update High Risk Flag based on segments
    df["High_Risk_Flag"] = df["Segment_Name"].apply(
        lambda x: 1
        if x in ["High Loan, Higher Default Risk", "Moderate Income, High Loan Burden"]
        else 0
    )
    
    features = [
        "Age",
        "Monthly_Income",
        "Loan_Amount",
        "Loan_Tenure",
        "Interest_Rate",
        "Collateral_Value",
        "Outstanding_Loan_Amount",
        "Monthly_EMI",
        "Num_Missed_Payments",
        "Days_Past_Due",
    ]
    
    X = df[features]
    y = df["High_Risk_Flag"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    risk_scores = rf_model.predict_proba(X_test)[:, 1]
    
    # Show model performance
    perf_col1, perf_col2 = st.columns(2, gap="large")
    with perf_col1:
        st.metric("Training Accuracy", f"{rf_model.score(X_train, y_train):.3f}")
    with perf_col2:
        st.metric("Test Accuracy", f"{rf_model.score(X_test, y_test):.3f}")
    
    # Create test dataframe with risk scores
    df_test = X_test.copy()
    df_test["Risk_Score"] = risk_scores
    df_test["Predicted_High_Risk"] = (df_test["Risk_Score"] > 0.5).astype(int)
    df_test = df_test.merge(
        df[
            [
                "Borrower_ID",
                "Segment_Name",
                "Recovery_Status",
                "Collection_Method",
                "Collection_Attempts",
                "Legal_Action_Taken",
            ]
        ],
        left_index=True,
        right_index=True,
    )
    
    df_test["Recovery_Strategy"] = df_test["Risk_Score"].apply(assign_recovery_strategy)
    
    # Arrange strategy distribution and sample predictions side by side
    strat_col1, strat_col2 = st.columns(2, gap="large")
    
    with strat_col1:
        st.write("**Recovery Strategy Distribution:**")
        st.dataframe(df_test["Recovery_Strategy"].value_counts(), use_container_width=True)
    
    with strat_col2:
        st.write("**Sample Risk Predictions:**")
        sample_predictions = df_test[["Borrower_ID", "Risk_Score", "Predicted_High_Risk", "Recovery_Strategy"]].head(10)
        st.dataframe(sample_predictions, use_container_width=True)
else:
    # Even if not showing the model, we still need to train it for individual borrower lookups
    # Update High Risk Flag based on segments
    df["High_Risk_Flag"] = df["Segment_Name"].apply(
        lambda x: 1
        if x in ["High Loan, Higher Default Risk", "Moderate Income, High Loan Burden"]
        else 0
    )
    
    features = [
        "Age",
        "Monthly_Income",
        "Loan_Amount",
        "Loan_Tenure",
        "Interest_Rate",
        "Collateral_Value",
        "Outstanding_Loan_Amount",
        "Monthly_EMI",
        "Num_Missed_Payments",
        "Days_Past_Due",
    ]
    
    X = df[features]
    y = df["High_Risk_Flag"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

# Visualizations
if show_visualizations:
    st.subheader("📊 Data Visualizations")
    
    vis_col1, vis_col2 = st.columns(2, gap="large")
    
    with vis_col1:
        fig1 = px.histogram(df, x="Loan_Amount", nbins=30, 
                           title="Loan Amount Distribution", 
                           color_discrete_sequence=["#636EFA"])
        fig1.update_layout(height=500)
        st.plotly_chart(fig1, use_container_width=True)
    
    with vis_col2:
        fig2 = px.histogram(df, x="Payment_History", color="Recovery_Status", 
                           barmode="group", title="Payment History vs Recovery Status",
                           color_discrete_map={
                               "Fully Recovered": "#00CC96",
                               "Partially Recovered": "#FFA15A",
                               "Written Off": "#EF553B"
                           })
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)

# Borrower Segmentation visualization
if show_segments:
    st.subheader("👤 Borrower Segmentation Visualization")
    
    fig3 = px.scatter(
        df,
        x="Monthly_Income",
        y="Loan_Amount",
        color="Segment_Name",  # Use actual segment names instead of numbers
        size="Loan_Amount",
        hover_data={
            "Monthly_Income": True, 
            "Loan_Amount": True, 
            "Borrower_Segment": True,
            "Borrower_ID": True  # Show Borrower ID on hover
        },
        title="Borrower Segments Based on Monthly Income and Loan Amount",
        labels={
            "Monthly_Income": "Monthly Income ($)",
            "Loan_Amount": "Loan Amount ($)",
            "Segment_Name": "Segment",  # Use actual segment names in legend
        },
        color_discrete_map={
            "Moderate Income, High Loan Burden": "#FF9999",
            "High Income, Low Default Risk": "#66B2FF",
            "Moderate Income, Medium Risk": "#99FF99",
            "High Loan, Higher Default Risk": "#FF9966"
        }
    )
    
    fig3.update_layout(height=600)
    st.plotly_chart(fig3, use_container_width=True)

# Individual borrower lookup
st.subheader("🔍 Individual Borrower Lookup")

# Validate that df is not empty before creating selectbox
if df.empty:
    st.warning("No data available to display")
else:
    borrower_id = st.selectbox("Select Borrower ID", df['Borrower_ID'].unique())
    
    # Validate the selected borrower_id exists in the dataframe
    if borrower_id and df[df['Borrower_ID'] == borrower_id].empty:
        st.error("Invalid borrower ID selected")
    else:
        borrower_data = df[df['Borrower_ID'] == borrower_id].iloc[0]

# Create two columns for better layout
if not df.empty and borrower_id and not df[df['Borrower_ID'] == borrower_id].empty:
    detail_col1, detail_col2 = st.columns([1, 2], gap="large")

    with detail_col1:
        st.write(f"**Borrower Details for {borrower_id}:**")
        
        # Build borrower info based on available data
        borrower_attributes = []
        borrower_values = []

        # Always available attributes
        borrower_attributes.extend([
            'Age', 'Gender', 'Employment Type', 'Monthly Income', 
            'Loan Amount', 'Recovery Status', 'Segment'
        ])
        borrower_values.extend([
            borrower_data['Age'], 
            borrower_data['Gender'], 
            borrower_data['Employment_Type'], 
            f"${borrower_data['Monthly_Income']:,}", 
            f"${borrower_data['Loan_Amount']:,}", 
            borrower_data['Recovery_Status'], 
            borrower_data['Segment_Name']
        ])

        # Add risk-related attributes - compute risk score for the specific borrower if the model exists
        if show_risk_model:
            try:
                # Prepare features for this specific borrower
                features = [
                    "Age",
                    "Monthly_Income",
                    "Loan_Amount",
                    "Loan_Tenure",
                    "Interest_Rate",
                    "Collateral_Value",
                    "Outstanding_Loan_Amount",
                    "Monthly_EMI",
                    "Num_Missed_Payments",
                    "Days_Past_Due",
                ]
                
                # Create feature array for this borrower using the same features as the model
                single_borrower_features = pd.DataFrame([{
                    feat: borrower_data[feat] for feat in features
                }])
                
                # Calculate risk score for this specific borrower using the trained model
                try:
                    single_borrower_risk_score = rf_model.predict_proba(single_borrower_features)[0, 1]
                    borrower_attributes.extend(['Risk Score', 'Recovery Strategy'])
                    borrower_values.extend([
                        f"{single_borrower_risk_score:.3f}",
                        assign_recovery_strategy(single_borrower_risk_score)
                    ])
                except Exception:
                    borrower_attributes.extend(['Risk Score', 'Recovery Strategy'])
                    borrower_values.extend(["N/A", "N/A"])
            except (NameError, KeyError, ValueError):
                borrower_attributes.extend(['Risk Score', 'Recovery Strategy'])
                borrower_values.extend(["N/A", "N/A"])
        else:
            borrower_attributes.extend(['Risk Score', 'Recovery Strategy'])
            borrower_values.extend(["N/A", "N/A"])

        borrower_info = pd.DataFrame({
            'Attribute': borrower_attributes,
            'Value': [str(v) for v in borrower_values]  # Convert all values to strings to ensure compatibility
        })
        st.dataframe(borrower_info, hide_index=True, use_container_width=True)

    with detail_col2:
        # Show a summary chart for this borrower
        st.write("**Borrower Profile Summary**")
        
        # Create a summary chart of key metrics for this borrower
        summary_data = {
            'Metric': ['Loan Amount', 'Monthly Income', 'EMI', 'Missed Payments', 'Days Past Due'],
            'Value': [
                borrower_data['Loan_Amount'],
                borrower_data['Monthly_Income'],
                borrower_data['Monthly_EMI'],
                borrower_data['Num_Missed_Payments'],
                borrower_data['Days_Past_Due']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create a bar chart for the borrower's metrics
        fig = px.bar(
            summary_df, 
            x='Value', 
            y='Metric', 
            orientation='h',
            title=f"Key Metrics for {borrower_id}",
            color='Metric',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# Show raw data at the bottom if selected
if show_raw_data:
    st.subheader("📋 Raw Loan Data")
    if not df.empty:
        st.dataframe(df, use_container_width=True, height=600)
        st.write(f"Dataset contains {df.shape[0]} records and {df.shape[1]} columns")
    else:
        st.warning("No data available to display")

# Footer
st.markdown("---")