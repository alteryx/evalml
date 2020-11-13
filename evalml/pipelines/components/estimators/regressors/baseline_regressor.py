import numpy as np
import pandas as pd

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper
)


class BaselineRegressor(Estimator):
    """Regressor that predicts using the specified strategy.

    This is useful as a simple baseline regressor to compare with other regressors.
    """
    name = "Baseline Regressor"
    hyperparameter_ranges = {}
    model_family = ModelFamily.BASELINE
    supported_problem_types = [ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION]

    def __init__(self, strategy="mean", random_state=0, **kwargs):
        """Baseline regressor that uses a simple strategy to make predictions.

        Arguments:
            strategy (str): method used to predict. Valid options are "mean", "median". Defaults to "mean".
            random_state (int, np.random.RandomState): seed for the random number generator

        """
        if strategy not in ["mean", "median"]:
            raise ValueError("'strategy' parameter must equal either 'mean' or 'median'")
        parameters = {"strategy": strategy}
        parameters.update(kwargs)

        self._prediction_value = None
        self._num_features = None
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Cannot fit Baseline regressor if y is None")
        X = _convert_to_woodwork_structure(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        y = _convert_to_woodwork_structure(y)
        y = _convert_woodwork_types_wrapper(y.to_series())

        if self.parameters["strategy"] == "mean":
            self._prediction_value = y.mean()
        elif self.parameters["strategy"] == "median":
            self._prediction_value = y.median()
        self._num_features = X.shape[1]
        return self

    def predict(self, X):
        X = _convert_to_woodwork_structure(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        return pd.Series([self._prediction_value] * len(X))

    @property
    def feature_importance(self):
        """Returns importance associated with each feature. Since baseline regressors do not use input features to calculate predictions, returns an array of zeroes.

        Returns:
            np.ndarray (float): an array of zeroes

        """
        return np.zeros(self._num_features)
