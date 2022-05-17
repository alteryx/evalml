"""Baseline classifier."""
import numpy as np
import pandas as pd

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Unsupervised
from evalml.problem_types import ProblemTypes
from evalml.utils import get_random_state, infer_feature_types


class BaselineClusterer(Unsupervised):
    """Clusterer that sorts data using the specified strategy.

    This is useful as a simple baseline clusterer to compare with other clusterers.

    Args:
        strategy (str): Method used to predict. Valid options are "random" and "random_weighted". Defaults to "random".
        n_clusters (int): Number of clusters that the baseline should predict. Defaults to 8.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Baseline Clusterer"
    hyperparameter_ranges = {}
    """{}"""
    model_family = ModelFamily.BASELINE
    """ModelFamily.BASELINE"""
    supported_problem_types = [ProblemTypes.CLUSTERING]
    """[ProblemTypes.CLUSTERING]"""

    def __init__(self, n_clusters=8, random_seed=0, **kwargs):
        if not isinstance(n_clusters, int) or n_clusters < 2:
            raise ValueError("The number of clusters must be a whole number greater than 1.")

        parameters = {"n_clusters": n_clusters}
        parameters.update(kwargs)
        super().__init__(
            parameters=parameters, component_obj=None, random_seed=random_seed
        )

    def fit(self, X, y=None):
        """Fits baseline clusterer component to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series): The target values, which may not exist and are not used in unsupervised learning. Ignored.

        Returns:
            self
        """
        self._num_features = X.shape[1]
        return self

    def predict(self, X):
        """Make predictions using the baseline clustering strategy.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].

        Returns:
            pd.Series: Predicted values.
        """
        X = infer_feature_types(X)
        n_clusters = self.parameters["n_clusters"]
        predictions = get_random_state(self.random_seed).choice(n_clusters, len(X))
        return infer_feature_types(predictions)

    @property
    def feature_importance(self):
        """Returns importance associated with each feature. Since baseline clusterers do not use input features to calculate predictions, returns an array of zeroes.

        Returns:
            pd.Series: An array of zeroes
        """
        return pd.Series(np.zeros(self._num_features))
