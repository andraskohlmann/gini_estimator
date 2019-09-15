from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from ThresholdBinarizer import ThresholdBinarizer


class custom_estimator(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self._model = LogisticRegression(solver='lbfgs')
        self._threshold_binarizer = ThresholdBinarizer()

    def fit(self, X, y=None):
        self._model.fit(X, y)
        probs = self._model.predict_proba(X)[..., 1]
        self._threshold_binarizer.fit(probs, y)
        return self

    def predict(self, X):
        return self._threshold_binarizer.transform(self._model.predict_proba(X)[..., 1])

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y_pred=self.predict(X), y_true=y, sample_weight=sample_weight)
