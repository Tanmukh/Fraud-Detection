from sklearn.ensemble import IsolationForest

class AnomalyDetection:
    def __init__(self):
        self.model = IsolationForest(contamination=0.01, random_state=42)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)
