from collections import OrderedDict

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from evalml.objectives import get_objective
from evalml.pipelines import PipelineBase


class ClassificationPipeline(PipelineBase):
    """Pipeline subclass for all classification pipelines."""

    def __init__(self, parameters, random_state=0):
        """Machine learning classification pipeline made out of transformers and a classifier.

        Required Class Variables:
            component_graph (list): List of components in order. Accepts strings or ComponentBase subclasses in the list

        Arguments:
            parameters (dict): dictionary with component names as keys and dictionary of that component's parameters as values.
                 An empty dictionary {} implies using all default values for component parameters.
            random_state (int, np.random.RandomState): The random seed/state. Defaults to 0.
        """
        self._encoder = LabelEncoder()
        super().__init__(parameters, random_state)

    def fit(self, X, y):
        """Build a model. For string and categorical targets, classes are sorted
            by sorted(set(y)) and then are mapped to values between 0 and n_classes-1.

        Arguments:
            X (pd.DataFrame or np.array): the input training data of shape [n_samples, n_features]

            y (pd.Series): the target training labels of length [n_samples]

        Returns:
            self

        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        self._encoder.fit(y)
        y = self._encode_targets(y)
        self._fit(X, y)
        return self

    def _encode_targets(self, y):
        """Converts target values from their original values to integer values that can be processed."""
        try:
            return pd.Series(self._encoder.transform(y))
        except ValueError as e:
            raise ValueError(str(e))

    def _decode_targets(self, y):
        """Converts encoded numerical values to their original target values.
            Note: we cast y as ints first to address boolean values that may be returned from
            calculating predictions which we would not be able to otherwise transform if we
            originally had integer targets."""
        return self._encoder.inverse_transform(y.astype(int))


    def _predict(self, X, objective=None):
        """Make predictions using selected features.

        Arguments:
            X (pd.DataFrame or np.array): data of shape [n_samples, n_features]
            objective (Object or string): the objective to use to make predictions

        Returns:
            pd.Series : estimated labels
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_t = self._transform(X)
        return self.estimator.predict(X_t)

    def predict(self, X, objective=None):
        """Make predictions using selected features.

        Arguments:
            X (pd.DataFrame or np.array): data of shape [n_samples, n_features]
            objective (Object or string): the objective to use to make predictions

        Returns:
            pd.Series : estimated labels
        """
        predictions = self._predict(X, objective)
        return self._decode_targets(predictions)

    def predict_proba(self, X):
        """Make probability estimates for labels.

        Arguments:
            X (pd.DataFrame or np.array): data of shape [n_samples, n_features]

        Returns:
            pd.DataFrame : probability estimates
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = self._transform(X)
        proba = self.estimator.predict_proba(X)
        return proba

    def score(self, X, y, objectives):
        """Evaluate model performance on objectives

        Arguments:
            X (pd.DataFrame or np.array): data of shape [n_samples, n_features]
            y (pd.Series): true labels of length [n_samples]
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
        y = self._encode_targets(y)
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
                y_predicted = self._predict(X)
        return y_predicted, y_predicted_proba
