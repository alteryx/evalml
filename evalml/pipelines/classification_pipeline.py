
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from evalml.objectives import get_objective
from evalml.pipelines import PipelineBase
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper
)


class ClassificationPipeline(PipelineBase):
    """Pipeline subclass for all classification pipelines."""

    def __init__(self, parameters, random_state=0):
        """Machine learning classification pipeline made out of transformers and a classifier.

        Required Class Variables:
            component_graph (list): List of components in order. Accepts strings or ComponentBase subclasses in the list

        Arguments:
            parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
                 An empty dictionary {} implies using all default values for component parameters.
            random_state (int, np.random.RandomState): The random seed/state. Defaults to 0.
        """
        self._encoder = LabelEncoder()
        super().__init__(parameters, random_state)

    def fit(self, X, y):
        """Build a classification model. For string and categorical targets, classes are sorted
            by sorted(set(y)) and then are mapped to values between 0 and n_classes-1.

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, np.ndarray): The target training labels of length [n_samples]

        Returns:
            self

        """
        X = _convert_to_woodwork_structure(X)
        y = _convert_to_woodwork_structure(y)
        y = _convert_woodwork_types_wrapper(y.to_series())
        self._encoder.fit(y)
        y = self._encode_targets(y)
        self._fit(X, y)
        return self

    def _encode_targets(self, y):
        """Converts target values from their original values to integer values that can be processed."""
        try:
            return pd.Series(self._encoder.transform(y), index=y.index)
        except ValueError as e:
            raise ValueError(str(e))

    def _decode_targets(self, y):
        """Converts encoded numerical values to their original target values.
            Note: we cast y as ints first to address boolean values that may be returned from
            calculating predictions which we would not be able to otherwise transform if we
            originally had integer targets."""
        return self._encoder.inverse_transform(y.astype(int))

    @property
    def classes_(self):
        """Gets the class names for the problem."""
        if not hasattr(self._encoder, "classes_"):
            raise AttributeError("Cannot access class names before fitting the pipeline.")
        return self._encoder.classes_

    def _predict(self, X, objective=None):
        """Make predictions using selected features.

        Arguments:
            X (pd.DataFrame): Data of shape [n_samples, n_features]
            objective (Object or string): The objective to use to make predictions

        Returns:
            pd.Series: Estimated labels
        """
        return self._component_graph.predict(X)

    def predict(self, X, objective=None):
        """Make predictions using selected features.

        Arguments:
            X (ww.DataTable, pd.DataFrame, or np.ndarray): Data of shape [n_samples, n_features]
            objective (Object or string): The objective to use to make predictions

        Returns:
            pd.Series : Estimated labels
        """
        predictions = self._predict(X, objective)
        return pd.Series(self._decode_targets(predictions))

    def predict_proba(self, X):
        """Make probability estimates for labels.

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): Data of shape [n_samples, n_features]

        Returns:
            pd.DataFrame: Probability estimates
        """
        X = self.compute_estimator_features(X, y=None)
        proba = self.estimator.predict_proba(X)
        proba.columns = self._encoder.classes_
        return proba

    def score(self, X, y, objectives):
        """Evaluate model performance on objectives

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): Data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, or np.ndarray): True labels of length [n_samples]
            objectives (list): List of objectives to score

        Returns:
            dict: Ordered dictionary of objective scores
        """
        y = _convert_to_woodwork_structure(y)
        y = _convert_woodwork_types_wrapper(y.to_series())
        objectives = [get_objective(o, return_instance=True) for o in objectives]
        y = self._encode_targets(y)
        y_predicted, y_predicted_proba = self._compute_predictions(X, objectives)

        return self._score_all_objectives(X, y, y_predicted, y_predicted_proba, objectives)

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
