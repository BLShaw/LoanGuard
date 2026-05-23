"""
Generate realistic loan recovery dataset with stochastic risk logic and loan maturity.
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
    
    data['Monthly_Income'] = np.random.randint(3000, 25000, N_SAMPLES)
    data['Loan_Amount'] = np.random.randint(10000, 200000, N_SAMPLES)
    data['Loan_Tenure'] = np.random.choice([12, 24, 36, 48, 60], N_SAMPLES)
    
    # Introduce Loan Maturity (Months on Book)
    data['Months_On_Book'] = np.array([np.random.randint(1, t + 1) for t in data['Loan_Tenure']])
    
    data['Interest_Rate'] = np.round(np.random.uniform(8, 18, N_SAMPLES), 2)
    
    monthly_rate = data['Interest_Rate'] / 100 / 12
    emi = np.where(
        monthly_rate == 0,
        data['Loan_Amount'] / data['Loan_Tenure'],
        data['Loan_Amount'] * monthly_rate * (1 + monthly_rate)**data['Loan_Tenure'] / 
        ((1 + monthly_rate)**data['Loan_Tenure'] - 1)
    )
    data['Monthly_EMI'] = np.round(emi, 0).astype(int)
    
    dti_ratio = data['Monthly_EMI'] / data['Monthly_Income']
    capped_dti = np.clip(dti_ratio, 0, 1.0)
    
    base_risk = np.clip(
        0.1 + 
        0.4 * capped_dti + 
        0.3 * (1 - (data['Monthly_Income'] - 3000) / 22000) + 
        np.random.normal(0, 0.05, N_SAMPLES),
        0.05, 0.95
    )
    
    data['Collateral_Value'] = np.where(
        base_risk < 0.4,
        np.random.randint(50000, 150000, N_SAMPLES),
        np.where(
            base_risk < 0.7,
            np.random.randint(10000, 50000, N_SAMPLES),
            np.random.randint(0, 10000, N_SAMPLES)
        )
    )
    
    raw_missed = np.where(
        base_risk < 0.3, 0,
        np.where(
            base_risk < 0.5, np.random.randint(0, 2, N_SAMPLES),
            np.where(
                base_risk < 0.7, np.random.randint(1, 5, N_SAMPLES),
                np.random.randint(3, 10, N_SAMPLES)
            )
        )
    )
    # Cannot miss more payments than months on book
    data['Num_Missed_Payments'] = np.minimum(raw_missed, data['Months_On_Book'])
    
    data['Days_Past_Due'] = np.where(
        data['Num_Missed_Payments'] == 0, 0,
        data['Num_Missed_Payments'] * np.random.randint(15, 35, N_SAMPLES)
    )
    
    # Calculate realistic outstanding balance
    expected_paid_fraction = data['Months_On_Book'] / data['Loan_Tenure']
    expected_balance = data['Loan_Amount'] * (1 - expected_paid_fraction)
    actual_balance = expected_balance + (data['Num_Missed_Payments'] * data['Monthly_EMI'])
    # Add a risk penalty to balance so it correlates properly
    actual_balance += base_risk * data['Loan_Amount'] * 0.1 
    data['Outstanding_Loan_Amount'] = np.clip(actual_balance, 0, data['Loan_Amount'] * 1.5).astype(int)
    
    data['Payment_History'] = np.where(
        data['Num_Missed_Payments'] == 0, 'Good',
        np.where(data['Num_Missed_Payments'] <= 2, 'Fair', 'Poor')
    )
    
    data['Collection_Attempts'] = np.where(
        base_risk < 0.4, 0,
        np.where(base_risk < 0.6, np.random.randint(1, 3, N_SAMPLES),
        np.random.randint(2, 8, N_SAMPLES))
    )
    
    data['Collection_Method'] = np.where(
        data['Collection_Attempts'] == 0, 'None',
        np.random.choice(['Calls', 'Settlement Offer', 'Legal Notice', 'Field Visit'], N_SAMPLES)
    )
    
    # Stochastic Recovery Status with Pending State
    random_factor = np.random.uniform(0, 1, N_SAMPLES)
    is_pending = np.random.uniform(0, 1, N_SAMPLES) < 0.20
    
    status = []
    for i in range(N_SAMPLES):
        if data['Num_Missed_Payments'][i] == 0:
            status.append('Fully Recovered')
        elif is_pending[i]:
            status.append('Pending Resolution')
        elif random_factor[i] > base_risk[i] + 0.1: 
            status.append('Fully Recovered')
        elif random_factor[i] > base_risk[i] - 0.2:
            status.append('Partially Recovered')
        else:
            status.append('Written Off')
            
    data['Recovery_Status'] = status
    
    data['Legal_Action_Taken'] = np.where(
        (base_risk > 0.6) & (data['Days_Past_Due'] > 90),
        'Yes', 'No'
    )
    
    df = pd.DataFrame(data)
    
    column_order = [
        'Borrower_ID', 'Age', 'Gender', 'Num_Dependents', 'Employment_Type',
        'Monthly_Income', 'Loan_Amount', 'Loan_Tenure', 'Months_On_Book', 'Interest_Rate',
        'Collateral_Value', 'Outstanding_Loan_Amount', 'Monthly_EMI',
        'Num_Missed_Payments', 'Days_Past_Due', 'Payment_History',
        'Collection_Attempts', 'Collection_Method', 'Recovery_Status',
        'Legal_Action_Taken'
    ]
    
    df = df[[c for c in column_order if c in df.columns]]
    return df

def validate_correlations(df):
    """Verify the data has correct risk correlations."""
    df_eval = df[df['Recovery_Status'] != 'Pending Resolution'].copy()
    df_eval['Risk_Flag'] = df_eval['Recovery_Status'].apply(lambda x: 1 if x != 'Fully Recovered' else 0)
    
    expected_positive = ['Num_Missed_Payments', 'Days_Past_Due', 'Outstanding_Loan_Amount']
    expected_negative = ['Monthly_Income', 'Collateral_Value']
    
    print("\n=== Correlation Validation ===")
    
    all_correct = True
    for col in expected_positive:
        corr = df_eval[col].corr(df_eval['Risk_Flag'])
        status = "✅" if corr > 0 else "❌"
        if corr <= 0:
            all_correct = False
        print(f"{status} {col}: {corr:.4f} (expected positive)")
    
    for col in expected_negative:
        corr = df_eval[col].corr(df_eval['Risk_Flag'])
        status = "✅" if corr < 0 else "❌"
        if corr >= 0:
            all_correct = False
        print(f"{status} {col}: {corr:.4f} (expected negative)")
    
    print(f"\n{'✅ All correlations correct!' if all_correct else '❌ Some correlations incorrect'}")
    return all_correct

if __name__ == '__main__':
    print("Generating realistic loan recovery dataset...")
    df = generate_data()
    
    if validate_correlations(df):
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"\n✅ Dataset saved to {OUTPUT_PATH}")
        print(f"   Shape: {df.shape}")
        print(f"\nRecovery Status Distribution:")
        print(df['Recovery_Status'].value_counts())
    else:
        print("\n❌ Data generation failed validation.")
