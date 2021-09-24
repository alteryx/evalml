"""Baseline clusterer."""
import numpy as np
import pandas as pd

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import get_random_state, infer_feature_types


class BaselineClusterer(Estimator):
    """Clusterer that predicts using the specified strategy.

    This is useful as a simple baseline clusterer to compare with other clusterers.

    Args:
        strategy (str): Method used to predict. Valid options are "all". Defaults to "all".
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Baseline Clusterer"
    hyperparameter_ranges = {}
    """{}"""
    model_family = ModelFamily.BASELINE
    """ModelFamily.BASELINE"""
    supported_problem_types = [ProblemTypes.CLUSTERING]
    """[ProblemTypes.CLUSTERING]"""

    def __init__(self, strategy="all", random_seed=0, **kwargs):
        if strategy not in ["all"]:
            raise ValueError(
                "'strategy' parameter must equal 'all'"
            )
        parameters = {"strategy": strategy}
        parameters.update(kwargs)
        self._classes = None
        self._percentage_freq = None
        self._num_features = None
        self._num_unique = None
        self._mode = None
        super().__init__(
            parameters=parameters, component_obj=None, random_seed=random_seed
        )

    def fit(self, X, y=None):
        """Fits baseline clusterer component to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series): The target training data of length [n_samples]. Ignored for clustering problems.

        Returns:
            self
        """
        return self

    def predict(self, X):
        """Make predictions using the baseline classification strategy.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].

        Returns:
            pd.Series: Predicted values.
        """
        X = infer_feature_types(X)
        strategy = self.parameters["strategy"]
        if strategy == "all":
            predictions = pd.Series(i%2 for i in range(len(X)))
        return infer_feature_types(predictions)

    @property
    def feature_importance(self):
        """Returns importance associated with each feature. Since baseline clusterers do not use input features to calculate predictions, returns an array of zeroes.

        Returns:
            np.ndarray (float): An array of zeroes
        """
        return np.zeros(self._num_features)
