"""
Analytics Hub: Outcome tracking, A/B test results, and causal analysis.

Features:
- Model accuracy over time (predicted vs actual)
- A/B test results with statistical significance
- Strategy performance comparison
- Causal effect estimates
- Decision audit trail
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.decision_log import get_logger
from src.ab_testing import get_ab_framework
from src.optimizer import get_optimizer, get_causal_estimator


def render(df, risk_model=None, fe=None):
    """Render the Analytics Hub interface."""
    st.header("📊 Analytics Hub")
    st.markdown("Track outcomes, analyze strategy performance, and understand model behavior.")
    
    # Initialize services
    logger = get_logger()
    ab_framework = get_ab_framework()
    optimizer = get_optimizer()
    
    # Tabs for different analytics
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Outcome Tracking",
        "🧪 A/B Testing", 
        "🎯 Optimizer Stats",
        "⚖️ Causal Analysis",
        "📋 Audit Trail"
    ])
    
    with tab1:
        render_outcome_tracking(logger, df)
    
    with tab2:
        render_ab_testing(ab_framework, df)
    
    with tab3:
        render_optimizer_stats(optimizer)
    
    with tab4:
        render_causal_analysis(df)
    
    with tab5:
        render_audit_trail(logger)


def render_outcome_tracking(logger, df):
    """Render outcome tracking section."""
    st.subheader("Model Performance Tracking")
    
    # Get accuracy metrics
    accuracy_data = logger.calculate_model_accuracy()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if accuracy_data['accuracy'] is not None:
            st.metric(
                "Overall Accuracy", 
                f"{accuracy_data['accuracy']:.1%}",
                help="Percentage of predictions that matched actual outcomes"
            )
        else:
            st.metric("Overall Accuracy", "N/A")
    
    with col2:
        st.metric(
            "Total Outcomes Recorded",
            accuracy_data['total_outcomes'],
            help="Number of loans with recorded actual outcomes"
        )
    
    with col3:
        if accuracy_data.get('correct_predictions'):
            st.metric(
                "Correct Predictions",
                accuracy_data['correct_predictions']
            )
        else:
            st.metric("Correct Predictions", "0")
    
    # Outcome breakdown
    if accuracy_data.get('outcome_breakdown'):
        st.markdown("---")
        st.subheader("Accuracy by Outcome Type")
        
        breakdown = accuracy_data['outcome_breakdown']
        breakdown_df = pd.DataFrame([
            {
                'Outcome': outcome,
                'Total': data['total'],
                'Correct': data['correct'],
                'Accuracy': data['correct'] / data['total'] if data['total'] > 0 else 0
            }
            for outcome, data in breakdown.items()
        ])
        
        if not breakdown_df.empty:
            fig = px.bar(
                breakdown_df,
                x='Outcome',
                y='Accuracy',
                color='Outcome',
                title="Prediction Accuracy by Recovery Status",
                color_discrete_map={
                    'Fully Recovered': '#28a745',
                    'Partially Recovered': '#ffc107',
                    'Written Off': '#dc3545'
                }
            )
            fig.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No outcomes recorded yet. Use the 'Record Outcome' feature in Customer 360 to start tracking.")
    
    # Strategy performance
    st.markdown("---")
    st.subheader("Strategy Performance")
    
    strategy_stats = logger.get_strategy_performance()
    
    if strategy_stats:
        strategy_df = pd.DataFrame([
            {
                'Strategy': strategy,
                'Total Cases': data['total'],
                'Fully Recovered': data['fully_recovered'],
                'Recovery Rate': data.get('recovery_rate', 0),
                'Avg Days to Resolution': data.get('avg_days_to_resolution', 'N/A')
            }
            for strategy, data in strategy_stats.items()
        ])
        
        st.dataframe(
            strategy_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Recovery Rate': st.column_config.ProgressColumn(
                    'Recovery Rate',
                    min_value=0,
                    max_value=1,
                    format="%.1%"
                )
            }
        )
    else:
        st.info("Record outcomes to see strategy performance metrics.")
    
    # Record outcome form
    st.markdown("---")
    st.subheader("🎯 Record New Outcome")
    
    with st.form("record_outcome_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            borrower_id = st.selectbox(
                "Select Borrower",
                df['Borrower_ID'].unique()
            )
            actual_outcome = st.selectbox(
                "Actual Outcome",
                ["Fully Recovered", "Partially Recovered", "Written Off"]
            )
        
        with col2:
            recovery_amount = st.number_input(
                "Recovery Amount ($)",
                min_value=0,
                value=0,
                step=1000
            )
            days_to_resolution = st.number_input(
                "Days to Resolution",
                min_value=0,
                value=30
            )
        
        submitted = st.form_submit_button("Record Outcome", type="primary")
        
        if submitted:
            borrower = df[df['Borrower_ID'] == borrower_id].iloc[0]
            logger.record_outcome(
                borrower_id=borrower_id,
                predicted_risk_score=borrower['Risk_Score'],
                predicted_strategy=borrower['Recovery_Strategy'],
                actual_outcome=actual_outcome,
                actual_recovery_amount=recovery_amount if recovery_amount > 0 else None,
                days_to_resolution=days_to_resolution
            )
            st.success(f"✅ Outcome recorded for {borrower_id}")
            st.rerun()


def render_ab_testing(ab_framework, df):
    """Render A/B testing section."""
    st.subheader("A/B Strategy Testing")
    
    # Active test status
    active_test = ab_framework.get_active_test()
    
    if active_test:
        st.success(f"Active Test: **{active_test.test_name}** (ID: {active_test.test_id})")
        st.write(f"Strategies: {', '.join(active_test.strategies)}")
        st.write(f"Started: {active_test.start_date[:10]}")
        
        # Get results
        results = ab_framework.get_test_results(active_test.test_id)
        
        if results:
            st.markdown("---")
            st.subheader("Test Results")
            
            # Results table
            results_df = pd.DataFrame([
                {
                    'Strategy': r.strategy,
                    'Total Assigned': r.total_assigned,
                    'Fully Recovered': r.fully_recovered,
                    'Recovery Rate': r.recovery_rate,
                    'Avg Recovery ($)': r.avg_recovery_amount or 'N/A'
                }
                for r in results.values()
            ])
            
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Significance test
            significance = ab_framework.calculate_significance(active_test.test_id)
            
            st.markdown("---")
            st.subheader("Statistical Significance")
            
            if significance['p_value'] is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("P-Value", f"{significance['p_value']:.4f}")
                with col2:
                    if significance['significant']:
                        st.success("✅ Statistically Significant (p < 0.05)")
                    else:
                        st.warning("⏳ Not Yet Significant")
                
                st.caption(significance['message'])
            else:
                st.info(significance['message'])
            
            # Bar chart comparison
            if len(results) >= 2:
                fig = px.bar(
                    results_df,
                    x='Strategy',
                    y='Recovery Rate',
                    color='Strategy',
                    title="Recovery Rate by Strategy"
                )
                fig.update_layout(yaxis_tickformat='.0%')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No outcomes recorded for this test yet.")
    else:
        st.info("No active A/B test. Create one below.")
    
    # Create new test
    st.markdown("---")
    st.subheader("🧪 Create New A/B Test")
    
    with st.form("create_ab_test"):
        test_name = st.text_input("Test Name", value="Strategy Comparison Test")
        
        strategies = st.multiselect(
            "Strategies to Compare",
            ["Standard Monitoring", "Settlement Offer", "Legal Action", "Proactive Outreach"],
            default=["Standard Monitoring", "Settlement Offer"]
        )
        
        use_thompson = st.checkbox(
            "Use Thompson Sampling (Adaptive Allocation)",
            help="Automatically allocate more traffic to better-performing strategies"
        )
        
        submitted = st.form_submit_button("Create Test", type="primary")
        
        if submitted and len(strategies) >= 2:
            test_id = ab_framework.create_test(
                test_name=test_name,
                strategies=strategies,
                use_thompson_sampling=use_thompson
            )
            st.success(f"✅ Created test: {test_id}")
            st.rerun()
        elif submitted:
            st.error("Select at least 2 strategies to compare")


def render_optimizer_stats(optimizer):
    """Render optimization engine statistics."""
    st.subheader("Multi-Arm Bandit Optimizer")
    
    st.markdown("""
    The optimizer uses **Thompson Sampling** to learn which strategies work best for different borrower profiles.
    It balances exploration (trying different strategies) with exploitation (using known good strategies).
    """)
    
    # Action statistics
    stats = optimizer.get_action_stats()
    
    if any(s['observations'] > 0 for s in stats.values()):
        st.subheader("Strategy Performance (Learned)")
        
        stats_df = pd.DataFrame([
            {
                'Strategy': action,
                'Observations': data['observations'],
                'Success Rate': data['success_rate'],
                'CI Lower': data['confidence_interval'][0],
                'CI Upper': data['confidence_interval'][1]
            }
            for action, data in stats.items()
        ])
        
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Confidence interval visualization
        fig = go.Figure()
        
        for _, row in stats_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['Strategy']],
                y=[row['Success Rate']],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[row['CI Upper'] - row['Success Rate']],
                    arrayminus=[row['Success Rate'] - row['CI Lower']]
                ),
                mode='markers',
                marker=dict(size=15),
                name=row['Strategy']
            ))
        
        fig.update_layout(
            title="Success Rate with 95% Credible Intervals",
            yaxis_title="Success Rate",
            yaxis_tickformat='.0%',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Monte Carlo simulation
        st.markdown("---")
        st.subheader("Monte Carlo Simulation")
        
        simulation = optimizer.simulate_outcomes(n_simulations=1000)
        
        sim_df = pd.DataFrame([
            {
                'Strategy': action,
                'Expected (Mean)': data['mean'],
                '10th Percentile': data['p10'],
                '50th Percentile': data['p50'],
                '90th Percentile': data['p90']
            }
            for action, data in simulation.items()
        ])
        
        st.dataframe(sim_df, use_container_width=True, hide_index=True)
    else:
        st.info("The optimizer hasn't collected enough data yet. Record outcomes to train it.")
    
    # Demo recommendation
    st.markdown("---")
    st.subheader("🎯 Get Recommendation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        demo_risk = st.slider("Risk Score", 0.0, 1.0, 0.5, 0.05)
    
    with col2:
        demo_segment = st.selectbox("Segment", [
            "High Income, Low Risk",
            "Moderate Income, Medium Risk", 
            "Moderate Income, High Burden",
            "High Loan, High Risk"
        ])
    
    if st.button("Get ML Recommendation"):
        recommendation = optimizer.recommend_action(
            risk_score=demo_risk,
            segment=demo_segment,
            explore=False  # Use pure exploitation for demo
        )
        
        st.success(f"**Recommended Action:** {recommendation.action}")
        st.write(f"**Confidence:** {recommendation.confidence:.1%}")
        st.write(f"**Expected Recovery Rate:** {recommendation.expected_recovery_rate:.1%}")
        st.caption(recommendation.reasoning)
        
        if recommendation.alternative_actions:
            st.write("**Alternatives:**")
            for action, rate in recommendation.alternative_actions:
                st.write(f"  • {action}: {rate:.1%} expected")


def render_causal_analysis(df):
    """Render causal analysis section."""
    st.subheader("Causal Effect Analysis")
    
    st.markdown("""
    > ⚠️ **Important:** SHAP values show feature *correlations*, not *causation*. 
    > This section attempts to estimate causal effects using statistical methods.
    """)
    
    causal = get_causal_estimator()
    
    # Explanation of difference
    with st.expander("📚 Correlation vs Causation"):
        st.markdown("""
        **Correlation (SHAP):** "Borrowers with more missed payments tend to have higher risk scores."
        
        **Causation (Causal Inference):** "Reducing missed payments would *cause* a reduction in risk."
        
        Causal inference requires either:
        1. **Randomized experiments** (A/B tests)
        2. **Observational methods** (propensity scoring, instrumental variables)
        
        We use A/B test data when available, falling back to observational estimates with appropriate caveats.
        """)
    
    st.markdown("---")
    st.subheader("Strategy Causal Effects")
    
    # Get A/B test data for causal estimates
    ab_framework = get_ab_framework()
    active_test = ab_framework.get_active_test()
    
    if active_test:
        results = ab_framework.get_test_results(active_test.test_id)
        
        if len(results) >= 2:
            # Since we have randomized data, we can make causal claims
            strategies = list(results.keys())
            control = strategies[0]
            
            st.info(f"Using A/B test data (randomized assignment) with **{control}** as control.")
            
            for strategy in strategies[1:]:
                effect = results[strategy].recovery_rate - results[control].recovery_rate
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**{strategy}** vs **{control}**")
                with col2:
                    if effect > 0:
                        st.success(f"+{effect:.1%} recovery rate")
                    else:
                        st.error(f"{effect:.1%} recovery rate")
        else:
            st.info("Need outcomes for multiple strategies to estimate causal effects.")
    else:
        st.warning("""
        No randomized A/B test available. Observational causal estimates may be biased.
        
        To enable reliable causal inference:
        1. Create an A/B test in the A/B Testing tab
        2. Record outcomes for multiple strategies
        3. Return here for causal effect estimates
        """)
    
    # Feature causal effects (simplified)
    st.markdown("---")
    st.subheader("Feature Impact (Observational)")
    st.caption("⚠️ These are observational correlations, not proven causal effects.")
    
    # Calculate correlations
    numeric_cols = ['Monthly_Income', 'Collateral_Value', 'Num_Missed_Payments', 'Days_Past_Due']
    correlations = []
    
    for col in numeric_cols:
        if col in df.columns:
            corr = df[col].corr(df['Risk_Score'])
            direction = "increases" if corr > 0 else "decreases"
            correlations.append({
                'Feature': col,
                'Correlation': corr,
                'Direction': direction,
                'Interpretation': f"Higher {col} → {direction} risk (correlation, not causation)"
            })
    
    corr_df = pd.DataFrame(correlations)
    
    fig = px.bar(
        corr_df,
        x='Feature',
        y='Correlation',
        color='Correlation',
        color_continuous_scale='RdBu_r',
        title="Feature Correlations with Risk Score"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(
        corr_df[['Feature', 'Interpretation']],
        use_container_width=True,
        hide_index=True
    )


def render_audit_trail(logger):
    """Render decision audit trail."""
    st.subheader("Decision Audit Trail")
    
    st.markdown("Complete log of all predictions, recommendations, and user actions.")
    
    # Get all decisions
    decisions = logger.get_all_decisions()
    
    if decisions:
        # Convert to DataFrame
        decisions_df = pd.DataFrame(decisions)
        
        # Add data fields as columns
        for key in ['risk_score', 'recommended_strategy', 'simulated_risk_score']:
            decisions_df[key] = decisions_df['data'].apply(
                lambda x: x.get(key) if isinstance(x, dict) else None
            )
        
        # Filter controls
        col1, col2 = st.columns(2)
        
        with col1:
            type_filter = st.multiselect(
                "Filter by Type",
                decisions_df['decision_type'].unique(),
                default=list(decisions_df['decision_type'].unique())
            )
        
        with col2:
            borrower_filter = st.multiselect(
                "Filter by Borrower",
                decisions_df['borrower_id'].unique()
            )
        
        # Apply filters
        filtered = decisions_df[decisions_df['decision_type'].isin(type_filter)]
        if borrower_filter:
            filtered = filtered[filtered['borrower_id'].isin(borrower_filter)]
        
        # Display - only include columns that exist
        all_display_cols = ['timestamp', 'decision_type', 'borrower_id', 'risk_score', 'recommended_strategy']
        display_cols = [c for c in all_display_cols if c in filtered.columns]
        display_df = filtered[display_cols].sort_values('timestamp', ascending=False)
        
        st.dataframe(
            display_df.head(100),
            use_container_width=True,
            hide_index=True
        )
        
        st.caption(f"Showing {min(100, len(display_df))} of {len(display_df)} records")
        
        # Export option
        if st.button("📥 Export Full Audit Log"):
            csv = decisions_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"audit_log_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No decisions logged yet. The system will automatically log predictions as you use the app.")
