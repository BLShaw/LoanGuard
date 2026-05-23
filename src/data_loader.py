import pandas as pd
import os
import warnings

def load_data(file_path):
    """Loads loan data from a CSV file with validation."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file {file_path} not found")
    
    if not file_path.endswith('.csv'):
        raise ValueError("File must be a CSV")

    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    """Performs basic data cleaning."""
    df = df.copy()
    
    # Fill numeric columns with column means, and warn only if numeric columns contain NaNs
    numeric_cols = df.select_dtypes(include=['number']).columns
    numeric_nan_count = df[numeric_cols].isna().sum().sum()
    if numeric_nan_count > 0:
        warnings.warn(f"Data contains {numeric_nan_count} numeric NaN values, filling with column means")
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Fill non-numeric columns with 'None'
    object_cols = df.select_dtypes(exclude=['number']).columns
    df[object_cols] = df[object_cols].fillna('None')
    
    return df
