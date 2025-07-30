# Insurance Claims Fraud Detection AI System

## Overview
This project implements a comprehensive AI system designed to analyze insurance claims data and identify potentially fraudulent submissions for investigation. It combines anomaly detection techniques with supervised machine learning models to flag suspicious claims effectively.

## Project Structure and Components

- **data_processing.py**: Contains functions to load, clean, and preprocess raw insurance claims data, preparing it for analysis.
- **feature_extraction.py**: Extracts relevant features from the preprocessed data to be used as input for the detection models.
- **anomaly_detection.py**: Implements anomaly detection algorithms (e.g., Isolation Forest) to identify unusual claims that deviate from normal patterns.
- **fraud_detection.py**: Implements a supervised fraud detection model (e.g., Random Forest classifier) trained on labeled data with known fraud cases.
- **investigation_priority.py**: Scores and prioritizes flagged claims based on fraud confidence scores and optional risk factors, helping investigators focus on the most critical cases.
- **main.py**: Orchestrates the entire analysis pipeline, including data loading, preprocessing, feature extraction, model predictions, combining results, and prioritization.
- **web_app.py**: A Flask-based web application that provides a user-friendly interface for uploading insurance claims CSV files, running the analysis pipeline, and displaying results. It includes:
  - Display of original data with fraud predictions.
  - Flagged claims prioritized for investigation.
  - Model evaluation metrics visualized with charts.
  - A textbox showing the number of flagged claims out of total claims.
- **test_pipeline.py** and **test_edge_cases.py**: Contain tests for the pipeline and edge cases to ensure robustness.
- **requirements.txt**: Lists Python dependencies required to run the project.

## Usage

1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

2. **Run the analysis from the command line:**
   ```
   python main.py <path_to_insurance_claims_csv>
   ```
   The CSV file must contain a `fraud_reported` column with 'Y' or 'N' values indicating known fraud cases.

3. **Or run the web application:**
   ```
   python web_app.py
   ```
   Then open the browser at `http://localhost:5000` to upload a CSV file and view results.

## Features

- Combines anomaly detection and supervised learning for robust fraud detection.
- Prioritizes flagged claims for efficient investigation.
- Provides detailed evaluation metrics and visualizations.
- User-friendly web interface for easy data upload and result viewing.
- Displays the number of flagged claims out of total claims prominently.

## Data Requirements

- Input CSV files must include a `fraud_reported` column with 'Y' or 'N' values.
- Data should be clean and properly formatted for accurate analysis.

## Testing

- Includes test scripts for pipeline and edge cases.
- Recommended to perform critical-path or thorough testing after any changes.

## Notes

- Extend feature extraction and model tuning for improved accuracy.
- Add additional unit tests and error handling as needed.


