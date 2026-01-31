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
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        warnings.warn(f"Data contains {nan_count} NaN values, filling with column means")
    
    df = df.fillna(df.mean(numeric_only=True))
    return df
