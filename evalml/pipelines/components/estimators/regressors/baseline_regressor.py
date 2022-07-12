"""Baseline regressor that uses a simple strategy to make predictions. This is useful as a simple baseline regressor to compare with other regressors."""
import numpy as np
import pandas as pd

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import infer_feature_types


class BaselineRegressor(Estimator):
    """Baseline regressor that uses a simple strategy to make predictions. This is useful as a simple baseline regressor to compare with other regressors.

    Args:
        strategy (str): Method used to predict. Valid options are "mean", "median". Defaults to "mean".
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Baseline Regressor"
    hyperparameter_ranges = {}
    """{}"""
    model_family = ModelFamily.BASELINE
    """ModelFamily.BASELINE"""
    supported_problem_types = [
        ProblemTypes.REGRESSION,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ]
    """[
        ProblemTypes.REGRESSION,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ]"""

    def __init__(self, strategy="mean", random_seed=0, **kwargs):
        if strategy not in ["mean", "median"]:
            raise ValueError(
                "'strategy' parameter must equal either 'mean' or 'median'",
            )
        parameters = {"strategy": strategy}
        parameters.update(kwargs)

        self._prediction_value = None
        self._num_features = None
        super().__init__(
            parameters=parameters,
            component_obj=None,
            random_seed=random_seed,
        )

    def fit(self, X, y=None):
        """Fits baseline regression component to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series): The target training data of length [n_samples].

        Returns:
            self

        Raises:
            ValueError: If input y is None.
        """
        if y is None:
            raise ValueError("Cannot fit Baseline regressor if y is None")
        X = infer_feature_types(X)
        y = infer_feature_types(y)

        if self.parameters["strategy"] == "mean":
            self._prediction_value = y.mean()
        elif self.parameters["strategy"] == "median":
            self._prediction_value = y.median()
        self._num_features = X.shape[1]
        return self

    def predict(self, X):
        """Make predictions using the baseline regression strategy.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].

        Returns:
            pd.Series: Predicted values.
        """
        X = infer_feature_types(X)
        predictions = pd.Series([self._prediction_value] * len(X))
        return infer_feature_types(predictions)

    @property
    def feature_importance(self):
        """Returns importance associated with each feature. Since baseline regressors do not use input features to calculate predictions, returns an array of zeroes.

        Returns:
            np.ndarray (float): An array of zeroes.
        """
        return np.zeros(self._num_features)
