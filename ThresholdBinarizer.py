from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class ThresholdBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._threshold = 0.0

    def _gini(self, labels):
        p_0 = np.sum(labels == 1) / max(float(len(labels)), 1)
        p_1 = 1 - p_0
        return 1 - p_0 ** 2 - p_1 ** 2

    def _calculate_split(self, predicted_probs, labels):
        gini = 1
        for p in predicted_probs:
            p_gini = self._gini(labels[predicted_probs > p]) + self._gini(labels[p >= predicted_probs])
            if p_gini < gini:
                self._threshold = p
                gini = p_gini

    def fit(self, X, y=None):
        self._calculate_split(X, y)
        return self

    def transform(self, X):
        return (X > self._threshold).astype(np.int)
