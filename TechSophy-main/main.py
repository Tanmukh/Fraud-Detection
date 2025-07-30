import sys
import pandas as pd
from data_processing import load_data, preprocess_data
from feature_extraction import extract_features
from anomaly_detection import AnomalyDetection
from fraud_detection import FraudDetectionModel
from investigation_priority import prioritize_investigations

def process_claims(file_path):
    # Load data
    data = load_data(file_path)
    
    # Check for target column
    if 'fraud_reported' not in data.columns:
        raise ValueError("The dataset must contain a 'fraud_reported' column as the target.")
    
    data_processed = preprocess_data(data)
    
    data_features = extract_features(data_processed)
    
    # Separate features and target
    X = data_features.drop(columns=['fraud_reported_Y', 'fraud_reported_N'], errors='ignore')
    y = data['fraud_reported'].map({'Y': 1, 'N': 0})
    
    anomaly_detector = AnomalyDetection()
    anomaly_detector.fit(X)
    anomaly_preds = anomaly_detector.predict(X)
    # Convert anomaly predictions: -1 (anomaly) to 1, 1 (normal) to 0
    anomaly_flags = (anomaly_preds == -1).astype(int)
    
    model = FraudDetectionModel()
    model.train(X, y)
    fraud_preds, fraud_confidence = model.predict(X)
    
    from sklearn.metrics import classification_report
    report = classification_report(y, fraud_preds, output_dict=True)
    
    # Combine results: flag if either model flags fraud
    combined_flag = ((fraud_preds == 1) | (anomaly_flags == 1)).astype(int)
    
    data['fraud_predicted'] = combined_flag
    data['fraud_confidence'] = fraud_confidence
    
    flagged = data[data['fraud_predicted'] == 1]
    
    num_flagged = len(flagged)
    
    prioritized = prioritize_investigations(flagged, confidence_col='fraud_confidence')
    
    return data, prioritized, report, num_flagged

def main(file_path):
    try:
        data, prioritized, report, num_flagged = process_claims(file_path)
        print(f"\\nFlagged {num_flagged} potentially fraudulent claims after combining models.")
        print("\\nTop claims prioritized for investigation:")
        print(prioritized.head(10))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_insurance_claims_csv>")
    else:
        main(sys.argv[1])
