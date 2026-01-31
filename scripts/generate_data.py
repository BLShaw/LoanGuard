"""
Generate realistic loan recovery dataset with correct risk correlations.

Risk Logic:
- Higher Num_Missed_Payments → Higher risk (positive correlation)
- Higher Days_Past_Due → Higher risk (positive correlation)  
- Higher Outstanding_Loan_Amount → Higher risk (positive correlation)
- Higher Collateral_Value → Lower risk (negative correlation)
- Higher Monthly_Income → Lower risk (negative correlation)
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

# Configuration
N_SAMPLES = 500
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'loan-recovery.csv')

def generate_data():
    """Generate synthetic loan data with logical risk correlations."""
    
    # Base borrower attributes
    data = {
        'Borrower_ID': [f'BRW_{i+1}' for i in range(N_SAMPLES)],
        'Age': np.random.randint(22, 65, N_SAMPLES),
        'Gender': np.random.choice(['Male', 'Female'], N_SAMPLES, p=[0.6, 0.4]),
        'Num_Dependents': np.random.randint(0, 5, N_SAMPLES),
        'Employment_Type': np.random.choice(
            ['Salaried', 'Self-Employed', 'Business Owner'], 
            N_SAMPLES, 
            p=[0.5, 0.3, 0.2]
        ),
    }
    
    # Financial attributes - these are the base drivers
    data['Monthly_Income'] = np.random.randint(3000, 25000, N_SAMPLES)
    data['Loan_Amount'] = np.random.randint(10000, 200000, N_SAMPLES)
    data['Loan_Tenure'] = np.random.choice([12, 24, 36, 48, 60], N_SAMPLES)
    data['Interest_Rate'] = np.round(np.random.uniform(8, 18, N_SAMPLES), 2)
    
    # Calculate realistic EMI (handle zero interest edge case)
    monthly_rate = data['Interest_Rate'] / 100 / 12
    emi = np.where(
        monthly_rate == 0,
        data['Loan_Amount'] / data['Loan_Tenure'],  # Zero interest: simple division
        data['Loan_Amount'] * monthly_rate * (1 + monthly_rate)**data['Loan_Tenure'] / 
        ((1 + monthly_rate)**data['Loan_Tenure'] - 1)
    )
    data['Monthly_EMI'] = np.round(emi, 0).astype(int)
    
    # Debt-to-Income ratio drives risk
    dti_ratio = data['Monthly_EMI'] / data['Monthly_Income']
    
    # Generate a base risk score (0-1) that makes logical sense
    # High DTI → High risk, Low Income → High risk
    base_risk = np.clip(
        0.3 + 
        0.4 * dti_ratio +  # DTI increases risk
        0.2 * (1 - (data['Monthly_Income'] - 3000) / 22000) +  # Low income increases risk
        np.random.normal(0, 0.1, N_SAMPLES),  # Random noise
        0.05, 0.95
    )
    
    # Collateral - higher income/loan gets more collateral, reduces risk
    data['Collateral_Value'] = np.where(
        base_risk < 0.4,
        np.random.randint(50000, 150000, N_SAMPLES),  # Low risk → good collateral
        np.where(
            base_risk < 0.7,
            np.random.randint(10000, 50000, N_SAMPLES),  # Medium risk → some collateral
            np.random.randint(0, 10000, N_SAMPLES)  # High risk → little/no collateral
        )
    )
    
    # Missed payments - STRONGLY correlated with risk
    data['Num_Missed_Payments'] = np.where(
        base_risk < 0.3, 0,
        np.where(
            base_risk < 0.5, np.random.randint(0, 2, N_SAMPLES),
            np.where(
                base_risk < 0.7, np.random.randint(1, 5, N_SAMPLES),
                np.random.randint(3, 10, N_SAMPLES)
            )
        )
    )
    
    # Days past due - correlated with missed payments
    data['Days_Past_Due'] = np.where(
        data['Num_Missed_Payments'] == 0, 0,
        data['Num_Missed_Payments'] * np.random.randint(15, 35, N_SAMPLES)
    )
    
    # Outstanding amount depends on payment behavior
    repayment_progress = np.clip(1 - base_risk + np.random.normal(0, 0.1, N_SAMPLES), 0.1, 0.9)
    data['Outstanding_Loan_Amount'] = np.round(
        data['Loan_Amount'] * (1 - repayment_progress * 0.5),
        0
    ).astype(int)
    
    # Payment History categorical
    data['Payment_History'] = np.where(
        data['Num_Missed_Payments'] == 0, 'Good',
        np.where(data['Num_Missed_Payments'] <= 2, 'Fair', 'Poor')
    )
    
    # Collection attempts - more for riskier borrowers
    data['Collection_Attempts'] = np.where(
        base_risk < 0.4, 0,
        np.where(base_risk < 0.6, np.random.randint(1, 3, N_SAMPLES),
        np.random.randint(2, 8, N_SAMPLES))
    )
    
    data['Collection_Method'] = np.where(
        data['Collection_Attempts'] == 0, 'None',
        np.random.choice(['Calls', 'Settlement Offer', 'Legal Notice', 'Field Visit'], N_SAMPLES)
    )
    
    # Recovery Status - THE TARGET VARIABLE
    # Directly driven by risk score
    data['Recovery_Status'] = np.where(
        base_risk < 0.35, 'Fully Recovered',
        np.where(base_risk < 0.70, 'Partially Recovered', 'Written Off')
    )
    
    # Legal action for high risk cases
    data['Legal_Action_Taken'] = np.where(
        (base_risk > 0.6) & (data['Days_Past_Due'] > 90),
        'Yes', 'No'
    )
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Reorder columns to match original
    column_order = [
        'Borrower_ID', 'Age', 'Gender', 'Num_Dependents', 'Employment_Type',
        'Monthly_Income', 'Loan_Amount', 'Loan_Tenure', 'Interest_Rate',
        'Collateral_Value', 'Outstanding_Loan_Amount', 'Monthly_EMI',
        'Num_Missed_Payments', 'Days_Past_Due', 'Payment_History',
        'Collection_Attempts', 'Collection_Method', 'Recovery_Status',
        'Legal_Action_Taken'
    ]
    
    # Keep only columns that exist
    df = df[[c for c in column_order if c in df.columns]]
    
    return df


def validate_correlations(df):
    """Verify the data has correct risk correlations."""
    df = df.copy()
    df['Risk_Flag'] = df['Recovery_Status'].apply(lambda x: 1 if x != 'Fully Recovered' else 0)
    
    expected_positive = ['Num_Missed_Payments', 'Days_Past_Due', 'Outstanding_Loan_Amount']
    expected_negative = ['Monthly_Income', 'Collateral_Value']
    
    print("\n=== Correlation Validation ===")
    
    all_correct = True
    for col in expected_positive:
        corr = df[col].corr(df['Risk_Flag'])
        status = "✅" if corr > 0 else "❌"
        if corr <= 0:
            all_correct = False
        print(f"{status} {col}: {corr:.4f} (expected positive)")
    
    for col in expected_negative:
        corr = df[col].corr(df['Risk_Flag'])
        status = "✅" if corr < 0 else "❌"
        if corr >= 0:
            all_correct = False
        print(f"{status} {col}: {corr:.4f} (expected negative)")
    
    print(f"\n{'✅ All correlations correct!' if all_correct else '❌ Some correlations incorrect'}")
    return all_correct


if __name__ == '__main__':
    print("Generating loan recovery dataset with correct risk correlations...")
    
    df = generate_data()
    
    # Validate before saving
    if validate_correlations(df):
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"\n✅ Dataset saved to {OUTPUT_PATH}")
        print(f"   Shape: {df.shape}")
        print(f"\nRecovery Status Distribution:")
        print(df['Recovery_Status'].value_counts())
    else:
        print("\n❌ Data generation failed validation. Please check the logic.")
