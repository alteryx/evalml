import numpy as np
import pandas as pd

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class ZeroRRegressor(Estimator):
    """Regressor that predicts using the specified strategy.

    This is useful as a simple baseline regressor to compare with other regressor.
"""
    name = "ZeroR Regressor"
    hyperparameter_ranges = {}
    model_family = ModelFamily.NONE
    supported_problem_types = [ProblemTypes.REGRESSION]

    def __init__(self, strategy="mean", random_state=0):
        """TODO"""
        if strategy not in ["mean", "median"]:
            raise ValueError("'strategy' parameter must equal either 'mean' or 'median'")
        parameters = {"strategy": strategy}
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def fit(self, X, y=None):
        """Fits component to data
        TODO
        Arguments:
            X (pd.DataFrame or np.array): the input training data of shape [n_samples, n_features]
            y (pd.Series, optional): the target training labels of length [n_samples]

        Returns:
            self
        """
        if y is None:
            raise ValueError("Cannot fit ZeroR classifier if y is None")

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        if self.parameters["strategy"] == "mean":
            self.val = y.mean()
        elif self.parameters["strategy"] == "median":
            self.val = y.median()
        self.num_features = len(X)
        return self

    def predict(self, X):
        """Make predictions using selected features.
        TODO

        Args:
            X (pd.DataFrame) : features

        Returns:
            pd.Series : estimated labels
        """
        try:
            val = self.val
        except AttributeError:
            raise RuntimeError("You must fit ZeroR classifier before calling predict!")
        return pd.Series([val] * len(X))

    @property
    def feature_importances(self):
        """Returns feature importances.
        TODO

        Returns:
            np.array (float) : importance associated with each feature

        """
        try:
            num_features = self.num_features
        except AttributeError:
            raise RuntimeError("You must fit ZeroR classifier before gettong feature_importances!")
        return np.array([0.0] * num_features)
