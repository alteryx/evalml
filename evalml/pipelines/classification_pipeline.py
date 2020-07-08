from collections import OrderedDict

import pandas as pd

from evalml.objectives import get_objective
from evalml.pipelines import PipelineBase
from sklearn.preprocessing import LabelEncoder 

class ClassificationPipeline(PipelineBase):
    """Pipeline subclass for all classification pipelines."""

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        y = self._encode_targets(y)
        self._fit(X, y)
        return self


    def _encode_targets(self, y):
        self.encoder = LabelEncoder()
        self.encoder.fit(y)
        return self.encoder.transform(y)

    def _decode_targets(self, y):
        return pd.Series(self.encoder.inverse_transform(y))

    def predict(self, X, objective=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_t = self._transform(X)
        predictions = self.estimator.predict(X_t)
        return self._decode_targets(predictions)

    def predict_proba(self, X):
        """Make probability estimates for labels.

        Arguments:
            X (pd.DataFrame or np.array) : data of shape [n_samples, n_features]

        Returns:
            pd.DataFrame : probability estimates
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = self._transform(X)
        proba = self.estimator.predict_proba(X)

        # todo: separate case for series? do under binary / multiclass
        if not isinstance(proba, pd.DataFrame):
            proba = pd.DataFrame(proba)
        proba.columns = self.encoder.inverse_transform(proba.columns)
        return proba

    def score(self, X, y, objectives):
        """Evaluate model performance on objectives

        Arguments:
            X (pd.DataFrame or np.array) : data of shape [n_samples, n_features]
            y (pd.Series) : true labels of length [n_samples]
            objectives (list): list of objectives to score

        Returns:
            dict: ordered dictionary of objective scores
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        objectives = [get_objective(o) for o in objectives]
        y_predicted, y_predicted_proba = self._compute_predictions(X, objectives)
        scores = OrderedDict()
        for objective in objectives:
            score = self._score(X, y, y_predicted_proba if objective.score_needs_proba else y_predicted, objective)
            scores.update({objective.name: score})
        return scores

    def _compute_predictions(self, X, objectives):
        """Scan through the objectives list and precompute"""
        y_predicted = None
        y_predicted_proba = None
        for objective in objectives:
            if objective.score_needs_proba and y_predicted_proba is None:
                y_predicted_proba = self.predict_proba(X)
            if not objective.score_needs_proba and y_predicted is None:
                y_predicted = self.predict(X)
        return y_predicted, y_predicted_proba
