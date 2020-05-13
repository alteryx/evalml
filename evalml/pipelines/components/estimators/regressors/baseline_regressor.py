import numpy as np
import pandas as pd

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class BaselineRegressor(Estimator):
    """Regressor that predicts using the specified strategy.

    This is useful as a simple baseline regressor to compare with other regressors.
    """
    name = "Baseline Regressor"
    hyperparameter_ranges = {}
    model_family = ModelFamily.BASELINE
    supported_problem_types = [ProblemTypes.REGRESSION]

    def __init__(self, strategy="mean", random_state=0):
        """Baseline regressor that uses a simple strategy to make predictions.

        Arguments:
            strategy (str): method used to predict. Valid options are "mean", "median". Defaults to "mean".
            random_state (int, np.random.RandomState): seed for the random number generator

        """
        if strategy not in ["mean", "median"]:
            raise ValueError("'strategy' parameter must equal either 'mean' or 'median'")
        parameters = {"strategy": strategy}
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Cannot fit Baseline regressor if y is None")

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        if self.parameters["strategy"] == "mean":
            self.val = y.mean()
        elif self.parameters["strategy"] == "median":
            self.val = y.median()
        self.num_features = X.shape[1]
        return self

    def predict(self, X):
        try:
            val = self.val
        except AttributeError:
            raise RuntimeError("You must fit Baseline regressor before calling predict!")
        return pd.Series([val] * len(X))

    @property
    def feature_importances(self):
        """Returns feature importances. Since baseline regressors do not use input features to calculate predictions, returns an array of zeroes.

        Returns:
            np.array (float) : an array of zeroes

        """
        try:
            num_features = self.num_features
            return np.array([0.0] * num_features)
        except AttributeError:
            raise RuntimeError("You must fit Baseline regressor before accessing feature_importances!")
