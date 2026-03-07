import numpy as np
from sklearn.linear_model import LogisticRegression
import sentence-transformers


class WinPredictor:
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray):
        return self.model.predict_proba(X)[:, 1]
    


def main():


