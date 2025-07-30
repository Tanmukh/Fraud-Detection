import pandas as pd
import numpy as np
from data_processing import load_data, preprocess_data
from feature_extraction import extract_features
from anomaly_detection import AnomalyDetection
from fraud_detection import FraudDetectionModel
from investigation_priority import prioritize_investigations

def test_missing_values():
    print("Testing missing values handling...")
    # Create data with missing values
    data = pd.DataFrame({
        'claim_id': [1, 2],
        'claimant_id': [1001, 1002],
        'claim_amount': [5000, None],
        'policy_amount': [10000, 15000],
        'claim_type': ['Theft', None],
        'incident_type': ['Robbery', 'Collision'],
        'fraud_reported': ['Y', 'N']
    })
    processed = preprocess_data(data)
    assert not processed.isnull().any().any(), "Missing values not handled properly"
    print("Missing values test passed.")

def test_unexpected_feature_columns():
    print("Testing feature extraction with unexpected columns...")
    data = pd.DataFrame({
        'claim_id': [1],
        'claimant_id': [1001],
        'claim_amount': [5000],
        'policy_amount': [10000],
        'claim_type': ['Theft'],
        'incident_type': ['Robbery'],
        'fraud_reported': ['Y']
    })
    processed = preprocess_data(data)
    # Remove 'policy_amount' to simulate missing column
    processed = processed.drop(columns=['policy_amount'], errors='ignore')
    features = extract_features(processed)
    assert 'claim_policy_ratio' not in features.columns or features['claim_policy_ratio'].isnull().all() == False, "Feature extraction failed on missing columns"
    print("Feature extraction test passed.")

def test_model_prediction_on_edge_cases():
    print("Testing model prediction on edge cases...")
    data = pd.DataFrame({
        'claim_id': [1, 2],
        'claimant_id': [1001, 1002],
        'claim_amount': [0, 1000000],  # Edge values
        'policy_amount': [10000, 10000],
        'claim_type': ['Theft', 'Fire'],
        'incident_type': ['Robbery', 'House Fire'],
        'fraud_reported': ['N', 'Y']
    })
    processed = preprocess_data(data)
    features = extract_features(processed)
    X = features.drop(columns=['fraud_reported_Y', 'fraud_reported_N'], errors='ignore')
    y = data['fraud_reported'].map({'Y': 1, 'N': 0})
    model = FraudDetectionModel()
    model.train(X, y)
    preds, conf = model.predict(X)
    assert len(preds) == len(data), "Prediction length mismatch"
    print("Model prediction edge case test passed.")

def test_prioritization_with_edge_scores():
    print("Testing prioritization with edge case scores...")
    data = pd.DataFrame({
        'claim_id': [1, 2],
        'fraud_confidence': [0.0, 1.0],
        'fraud_predicted': [1, 1]
    })
    prioritized = prioritize_investigations(data, confidence_col='fraud_confidence')
    assert prioritized.iloc[0]['fraud_confidence'] == 1.0, "Prioritization sorting failed"
    print("Prioritization test passed.")

if __name__ == "__main__":
    test_missing_values()
    test_unexpected_feature_columns()
    test_model_prediction_on_edge_cases()
    test_prioritization_with_edge_scores()
    print("All edge case tests completed successfully.")
