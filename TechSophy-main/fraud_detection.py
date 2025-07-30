import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class FraudDetectionModel:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)

    def train(self, X, y):
        """
        Train the fraud detection model.
        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Target labels (fraud or not).
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print("Model Evaluation:")
        print(classification_report(y_test, y_pred))

    def predict(self, X):
        """
        Predict fraud on new data.
        Args:
            X (pd.DataFrame): Features.
        Returns:
            tuple: (predictions, confidence_scores)
        """
        preds = self.model.predict(X)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            if proba.shape[1] > 1:
                confidence = proba[:, 1]
            else:
                confidence = proba[:, 0]
        else:
            confidence = np.zeros(len(preds))
        return preds, confidence
