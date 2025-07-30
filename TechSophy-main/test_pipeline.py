import pandas as pd
from data_processing import load_data, preprocess_data
from feature_extraction import extract_features
from anomaly_detection import AnomalyDetection
from fraud_detection import FraudDetectionModel
from investigation_priority import prioritize_investigations

def test_pipeline(file_path):
    print("Starting pipeline test...")
    # Load data
    data = load_data(file_path)
    print(f"Loaded data shape: {data.shape}")
    
    # Preprocess data
    data_processed = preprocess_data(data)
    print(f"Data after preprocessing shape: {data_processed.shape}")
    
    # Feature extraction
    data_features = extract_features(data_processed)
    print(f"Data after feature extraction shape: {data_features.shape}")
    
    # Separate features and target
    X = data_features.drop(columns=['fraud_reported_Y', 'fraud_reported_N'], errors='ignore')
    y = data['fraud_reported'].map({'Y': 1, 'N': 0})
    
    # Anomaly detection
    anomaly_detector = AnomalyDetection()
    anomaly_detector.fit(X)
    anomaly_preds = anomaly_detector.predict(X)
    anomaly_flags = (anomaly_preds == -1).astype(int)
    import pandas as pd
    print(f"Anomaly flags distribution:\n{pd.Series(anomaly_flags).value_counts()}")
    
    # Fraud detection model
    model = FraudDetectionModel()
    model.train(X, y)
    fraud_preds, fraud_confidence = model.predict(X)
    print(f"Fraud predictions distribution:\n{pd.Series(fraud_preds).value_counts()}")
    
    # Combine results
    combined_flag = ((fraud_preds == 1) | (anomaly_flags == 1)).astype(int)
    data['fraud_predicted'] = combined_flag
    data['fraud_confidence'] = fraud_confidence
    flagged = data[data['fraud_predicted'] == 1]
    print(f"Number of flagged claims: {len(flagged)}")
    
    # Prioritize investigations
    if 'fraud_confidence' in flagged.columns:
        prioritized = prioritize_investigations(flagged, confidence_col='fraud_confidence')
    else:
        # If fraud_confidence column is missing, prioritize without confidence column
        prioritized = prioritize_investigations(flagged)
    print("Top prioritized claims:")
    print(prioritized.head(5))
    
    print("Pipeline test completed successfully.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python test_pipeline.py <path_to_insurance_claims_csv>")
    else:
        test_pipeline(sys.argv[1])
