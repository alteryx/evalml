import numpy as np
import pandas as pd

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class BaselineClassifier(Estimator):
    """Classifier that predicts using the specified strategy.

    This is useful as a simple baseline classifier to compare with other classifiers.
    """
    name = "Baseline Classifier"
    hyperparameter_ranges = {}
    model_family = ModelFamily.BASELINE
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def __init__(self, strategy="mode", random_state=0):
        """Baseline classifier that uses a simple strategy to make predictions.

        Arguments:
            strategy (str): method used to predict. Valid options are "mode", "random" and "random_weighted". Defaults to "mode".
            random_state (int, np.random.RandomState): seed for the random number generator

        """
        if strategy not in ["mode", "random", "random_weighted"]:
            raise ValueError("'strategy' parameter must equal either 'mode', 'random', or 'random_weighted'")
        parameters = {"strategy": strategy}
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Cannot fit Baseline classifier if y is None")

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        vals, counts = np.unique(y, return_counts=True)
        self.unique_vals = vals
        self.percentage_freq = counts.astype(float) / len(y)
        self.num_unique = len(self.unique_vals)
        self.num_features = X.shape[1]

        if self.parameters["strategy"] == "mode":
            self.mode = y.mode()[0]
        return self

    def predict(self, X):
        strategy = self.parameters["strategy"]
        try:
            if strategy == "mode":
                mode = self.mode
                return pd.Series([mode] * len(X))
            elif strategy == "random":
                unique_vals = self.unique_vals
                return self.random_state.choice(unique_vals, len(X))
            else:
                unique_vals = self.unique_vals
                return self.random_state.choice(unique_vals, len(X), p=self.percentage_freq)
        except AttributeError:
            raise RuntimeError("You must fit Baseline classifier before calling predict!")

    def predict_proba(self, X):
        strategy = self.parameters["strategy"]
        try:
            if strategy == "mode":
                mode = self.mode
                num_unique = self.num_unique
                return np.array([[1.0 if i == mode else 0.0 for i in range(num_unique)]] * len(X))
            elif strategy == "random":
                num_unique = self.num_unique
                return np.array([[1.0 / self.num_unique for i in range(num_unique)]] * len(X))
            else:
                num_unique = self.num_unique
                return np.array([[self.percentage_freq[i] for i in range(num_unique)]] * len(X))
        except AttributeError:
            raise RuntimeError("You must fit Baseline classifier before calling predict_proba!")

    @property
    def feature_importances(self):
        """Returns feature importances. Since baseline classifiers do not use input features to calculate predictions, returns an array of zeroes.

        Returns:
            np.array (float) : an array of zeroes

        """
        try:
            num_features = self.num_features
            return np.array([0.0] * num_features)
        except AttributeError:
            raise RuntimeError("You must fit Baseline classifier before getting feature_importances!")
