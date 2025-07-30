import pandas as pd

def load_data(file_path):
    """
    Load insurance claims data from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data(df):
    """
    Preprocess the insurance claims data for fraud detection.
    Args:
        df (pd.DataFrame): Raw insurance claims data.
    Returns:
        pd.DataFrame: Preprocessed data ready for model input.
    """
    df = df.copy()
    
    # Fill missing numeric values with median
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    
    # Fill missing categorical values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
    
    # Convert categorical columns to dummy variables
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df
