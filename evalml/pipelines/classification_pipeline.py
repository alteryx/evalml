
import pandas as pd

from evalml.pipelines import PipelineBase
from evalml.utils import _convert_woodwork_types_wrapper, infer_feature_types


class ClassificationPipeline(PipelineBase):
    """Pipeline subclass for all classification pipelines."""

    def __init__(self, parameters, random_seed=0):
        """Machine learning classification pipeline made out of transformers and a classifier.

        Required Class Variables:
            component_graph (list): List of components in order. Accepts strings or ComponentBase subclasses in the list

        Arguments:
            parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
                 An empty dictionary {} implies using all default values for component parameters.
            random_state (int): Deprecated - use random_seed instead.
            random_seed (int): Seed for the random number generator. Defaults to 0.
        """
        super().__init__(parameters, random_seed=random_seed)

    def fit(self, X, y):
        """Build a classification model. For string and categorical targets, classes are sorted
            by sorted(set(y)) and then are mapped to values between 0 and n_classes-1.

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, np.ndarray): The target training labels of length [n_samples]

        Returns:
            self

        """
        X = infer_feature_types(X)
        y = infer_feature_types(y)
        y = _convert_woodwork_types_wrapper(y.to_series())
        self._fit(X, y)
        return self

    @property
    def classes_(self):
        """Gets the class names for the problem."""
        encoder = self._component_graph.get_component('Label Encoder')._encoder
        if not hasattr(encoder, "classes_"):
            raise AttributeError("Cannot access class names before fitting the pipeline.")
        return encoder.classes_

    def _predict(self, X, objective=None):
        """Make predictions using selected features.

        Arguments:
            X (ww.DataTable, pd.DataFrame): Data of shape [n_samples, n_features]
            objective (Object or string): The objective to use to make predictions

        Returns:
            ww.DataColumn: Estimated labels
        """
        return self._component_graph.predict(X)

    def predict(self, X, objective=None):
        """Make predictions using selected features.

        Arguments:
            X (ww.DataTable, pd.DataFrame, or np.ndarray): Data of shape [n_samples, n_features]
            objective (Object or string): The objective to use to make predictions

        Returns:
            ww.DataColumn: Estimated labels
        """
        predictions = self._predict(X, objective=objective).to_series()
        predictions = pd.Series(predictions, name=self.input_target_name)
        return infer_feature_types(predictions)

    def predict_proba(self, X):
        """Make probability estimates for labels.

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): Data of shape [n_samples, n_features]

        Returns:
            ww.DataTable: Probability estimates
        """
        X = self.compute_estimator_features(X, y=None)
        proba = self.estimator.predict_proba(X).to_dataframe()
        try:
            classes = self.classes_
            proba.columns = classes
        except Exception:
            pass
        return infer_feature_types(proba)

    def score(self, X, y, objectives):
        """Evaluate model performance on objectives

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): Data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, or np.ndarray): True labels of length [n_samples]
            objectives (list): List of objectives to score

        Returns:
            dict: Ordered dictionary of objective scores
        """
        y = infer_feature_types(y)
        y = _convert_woodwork_types_wrapper(y.to_series())
        objectives = self.create_objectives(objectives)
        y_predicted, y_predicted_proba = self._compute_predictions(X, y, objectives)
        if y_predicted is not None:
            y_predicted = _convert_woodwork_types_wrapper(y_predicted.to_series())
        if y_predicted_proba is not None:
            y_predicted_proba = _convert_woodwork_types_wrapper(y_predicted_proba.to_dataframe())
        return self._score_all_objectives(X, y, y_predicted, y_predicted_proba, objectives)

    def _compute_predictions(self, X, y, objectives, time_series=False):
        """Compute predictions/probabilities based on objectives."""
        y_predicted = None
        y_predicted_proba = None
        if any(o.score_needs_proba for o in objectives):
            y_predicted_proba = self.predict_proba(X, y) if time_series else self.predict_proba(X)
        if any(not o.score_needs_proba for o in objectives):
            y_predicted = self._predict(X, y, pad=True) if time_series else self._predict(X)
        return y_predicted, y_predicted_proba
