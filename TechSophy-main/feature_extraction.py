import pandas as pd

def extract_features(df):
    """
    Perform advanced feature engineering on insurance claims data.
    Args:
        df (pd.DataFrame): Preprocessed insurance claims data.
    Returns:
        pd.DataFrame: Data with additional engineered features.
    """
    df = df.copy()
    
    # Example feature: claim amount to policy amount ratio (if columns exist)
    if 'claim_amount' in df.columns and 'policy_amount' in df.columns:
        df['claim_policy_ratio'] = df['claim_amount'] / (df['policy_amount'] + 1e-5)
    
    # Placeholder for network analysis features
    # For example, count of claims per claimant or linked entities
    if 'claimant_id' in df.columns:
        claim_counts = df.groupby('claimant_id').size()
        df['claimant_claim_count'] = df['claimant_id'].map(claim_counts)
    
    # Fill any new missing values
    df.fillna(0, inplace=True)
    
    return df
