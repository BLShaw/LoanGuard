import pandas as pd
import os

def load_data(file_path):
    """
    Loads loan data from a CSV file with validation.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file {file_path} not found")
    
    # Simple check for file extension
    if not file_path.endswith('.csv'):
        raise ValueError("File must be a CSV")

    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    """
    Performs basic data cleaning.
    """
    # Fill missing values for numeric columns with mean
    df = df.fillna(df.mean(numeric_only=True))
    
    # Ensure categorical consistency if needed (omitted for now as data seems clean enough from exploration)
    return df
