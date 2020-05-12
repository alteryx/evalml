import numpy as np
import pandas as pd

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class BaselineClassifier(Estimator):
    """TODO
    Classifier that predicts using the mode. In the case where there is no single mode, the lowest value is used.

    This is useful as a simple baseline classifier to compare with other classifiers.
    """
    name = "Baseline Classifier"
    hyperparameter_ranges = {}
    model_family = ModelFamily.NONE
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def __init__(self, strategy="mode", random_state=0):
        """TODO"""
        if strategy not in ["mode", "random"]:
            raise ValueError("'strategy' parameter must equal either 'mode' or 'random'")
        parameters = {"strategy": strategy}
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def fit(self, X, y=None):
        """Fits component to data

        Arguments:
            X (pd.DataFrame or np.array): the input training data of shape [n_samples, n_features]
            y (pd.Series, optional): the target training labels of length [n_samples]

        Returns:
            self
        """
        if y is None:
            raise ValueError("Cannot fit Baseline classifier if y is None")

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        self.unique_vals = np.unique(y)
        self.num_unique = len(self.unique_vals)
        self.num_rows = X.shape[0]
        self.num_features = X.shape[1]

        if self.parameters["strategy"] == "mode":
            self.mode = y.mode()[0]
        return self

    def predict(self, X):
        """Make predictions using selected features.

        Args:
            X (pd.DataFrame) : features

        Returns:
            pd.Series : estimated labels
        """
        try:
            if self.parameters["strategy"] == "mode":
                mode = self.mode
                return pd.Series([mode] * len(X))
            else:
                unique_vals = self.unique_vals
                return self.random_state.choice(unique_vals, self.num_rows)
        except AttributeError:
            raise RuntimeError("You must fit Baseline classifier before calling predict!")

    def predict_proba(self, X):
        """Make probability estimates for labels.

        Args:
            X (pd.DataFrame) : features

        Returns:
            np.array : probability estimates
        """
        try:
            if self.parameters["strategy"] == "mode":
                mode = self.mode
                num_unique = self.num_unique
                return np.array([[1.0 if i == mode else 0.0 for i in range(num_unique)]] * len(X))
            else:
                num_unique = self.num_unique
                return np.array([[1.0 / self.num_unique for i in range(num_unique)]] * len(X))
        except AttributeError:
            raise RuntimeError("You must fit Baseline classifier before calling predict_proba!")

    @property
    def feature_importances(self):
        """Returns feature importances.

        Returns:
            np.array (float) : importance associated with each feature

        """
        try:
            num_features = self.num_features
            return np.array([0.0] * num_features)
        except AttributeError:
            raise RuntimeError("You must fit Baseline classifier before gettong feature_importances!")
