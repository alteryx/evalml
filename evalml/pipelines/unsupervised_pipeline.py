"""Pipeline subclass for all unsupervised learning pipelines."""
import numpy as np
import pandas as pd
import woodwork as ww

from evalml.pipelines import PipelineBase
from evalml.problem_types import ProblemTypes
from evalml.utils import infer_feature_types


class UnsupervisedPipeline(PipelineBase):

    """Pipeline subclass for all unsupervised learning pipelines.

    Args:
        component_graph (list or dict): List of components in order. Accepts strings or ComponentBase subclasses in the list.
            Note that when duplicate components are specified in a list, the duplicate component names will be modified with the
            component's index in the list. For example, the component graph
            [Imputer, One Hot Encoder, Imputer, DBSCAN Clusterer] will have names
            ["Imputer", "One Hot Encoder", "Imputer_2", "DBSCAN Clusterer"]
        parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
             An empty dictionary or None implies using all default values for component parameters. Defaults to None.
        custom_name (str): Custom name for the pipeline. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """
    
    problem_type = ProblemTypes.CLUSTERING
    """ProblemTypes.CLUSTERING"""

    def __init__(
        self,
        component_graph,
        parameters=None,
        custom_name=None,
        random_seed=0,
    ):
        self._classes_ = None
        super().__init__(
            component_graph,
            custom_name=custom_name,
            parameters=parameters,
            random_seed=random_seed,
        )

    def fit(self, X, y):
        """Build an unsupervised model.

        Args:
            X (pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]

        Returns:
            self
        """
        X = infer_feature_types(X)

        self._fit(X, y)
        return self

    def _predict(self, X, objective=None):
        """Make predictions using selected features.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].
            objective (Object or string): The objective to use to make predictions.

        Returns:
            pd.Series: Estimated labels
        """
        return self.component_graph.predict(X)

    def predict(self, X, objective=None, X_train=None, y_train=None):
        """Make predictions using selected features.

        Note: we cast y as ints first to address boolean values that may be returned from
        calculating predictions which we would not be able to otherwise transform if we
        originally had integer targets.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].
            objective (Object or string): The objective to use to make predictions.
            X_train (pd.DataFrame): Training data. Ignored. Only used for time series.
            y_train (pd.Series): Training labels. Ignored. Only used for time series.

        Returns:
            pd.Series: Estimated labels.
        """
        _predictions = self._predict(X, objective=objective)
        predictions = self.inverse_transform(_predictions.astype(int))
        predictions = pd.Series(
            predictions, name=self.input_target_name, index=_predictions.index
        )
        return infer_feature_types(predictions)

    def score(self, X, objectives, X_train=None, y_train=None):
        """Evaluate model performance on objectives.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features]
            objectives (list): List of objectives to score
            X_train (pd.DataFrame): Training data. Ignored. Only used for time series.
            y_train (pd.Series): Training labels. Ignored. Only used for time series.

        Returns:
            dict: Ordered dictionary of objective scores.
        """
        objectives = self.create_objectives(objectives)
        y_predicted = self._compute_predictions(X, y=None, objectives=objectives)
        return self._score_all_objectives(
            X, y, y_predicted, y_predicted_proba, objectives
        )

    def _compute_predictions(self, X, y, objectives):
        """Compute predictions/probabilities based on objectives."""
        return self._predict(X)
