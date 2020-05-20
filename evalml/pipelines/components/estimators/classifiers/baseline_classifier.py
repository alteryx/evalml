
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
        self._unique_vals = None
        self._percentage_freq = None
        self._num_features = None
        self._num_unique = None
        self._mode = None
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Cannot fit Baseline classifier if y is None")

        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        vals, counts = np.unique(y, return_counts=True)
        self._unique_vals = vals
        self._percentage_freq = counts.astype(float) / len(y)
        self._num_unique = len(self._unique_vals)
        self._num_features = X.shape[1]

        if self.parameters["strategy"] == "mode":
            self._mode = y.mode()[0]
        return self

    def predict(self, X):
        strategy = self.parameters["strategy"]
        if strategy == "mode":
            if self._mode is None:
                raise RuntimeError("You must fit Baseline classifier before calling predict!")
            return pd.Series([self._mode] * len(X))
        elif strategy == "random":
            if self._unique_vals is None:
                raise RuntimeError("You must fit Baseline classifier before calling predict!")
            return self.random_state.choice(self._unique_vals, len(X))
        else:
            if self._unique_vals is None:
                raise RuntimeError("You must fit Baseline classifier before calling predict!")
            return self.random_state.choice(self._unique_vals, len(X), p=self._percentage_freq)

    def predict_proba(self, X):
        strategy = self.parameters["strategy"]
        if strategy == "mode":
            if self._mode is None or self._num_unique is None:
                raise RuntimeError("You must fit Baseline classifier before calling predict!")
            return np.array([[1.0 if i == self._mode else 0.0 for i in range(self._num_unique)]] * len(X))
        elif strategy == "random":
            if self._unique_vals is None:
                raise RuntimeError("You must fit Baseline classifier before calling predict!")
            return np.array([[1.0 / self._num_unique for i in range(self._num_unique)]] * len(X))
        else:
            if self._unique_vals is None or self._percentage_freq is None:
                raise RuntimeError("You must fit Baseline classifier before calling predict!")
            return np.array([[self._percentage_freq[i] for i in range(self._num_unique)]] * len(X))

    @property
    def feature_importances(self):
        """Returns feature importances. Since baseline classifiers do not use input features to calculate predictions, returns an array of zeroes.

        Returns:
            np.array (float) : an array of zeroes

        """
        if self._num_unique is None:
            raise RuntimeError("You must fit Baseline classifier before getting feature_importances!")
        return np.zeros(self._num_features)
