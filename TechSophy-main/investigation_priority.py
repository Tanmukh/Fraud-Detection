import pandas as pd

def prioritize_investigations(df, confidence_col='fraud_confidence', risk_factors=None):
    """
    Score and prioritize flagged claims for investigation.
    Args:
        df (pd.DataFrame): DataFrame containing flagged claims with confidence scores.
        confidence_col (str): Column name for fraud confidence scores.
        risk_factors (dict): Optional dictionary of additional risk factors and their weights.
    Returns:
        pd.DataFrame: DataFrame sorted by investigation priority score (descending).
    """
    df = df.copy()
    
    # Base score is the confidence score
    df['investigation_score'] = df[confidence_col]
    
    # Add weighted risk factors if provided
    if risk_factors:
        for factor, weight in risk_factors.items():
            if factor in df.columns:
                df['investigation_score'] += weight * df[factor]
    
    
    df_sorted = df.sort_values(by='investigation_score', ascending=False)
    
    return df_sorted
