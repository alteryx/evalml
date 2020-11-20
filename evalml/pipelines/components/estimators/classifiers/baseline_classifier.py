
import numpy as np
import pandas as pd

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper
)


class BaselineClassifier(Estimator):
    """Classifier that predicts using the specified strategy.

    This is useful as a simple baseline classifier to compare with other classifiers.
    """
    name = "Baseline Classifier"
    hyperparameter_ranges = {}
    model_family = ModelFamily.BASELINE
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def __init__(self, strategy="mode", random_state=0, **kwargs):
        """Baseline classifier that uses a simple strategy to make predictions.

        Arguments:
            strategy (str): Method used to predict. Valid options are "mode", "random" and "random_weighted". Defaults to "mode".
            random_state (int, np.random.RandomState): Seed for the random number generator
        """
        if strategy not in ["mode", "random", "random_weighted"]:
            raise ValueError("'strategy' parameter must equal either 'mode', 'random', or 'random_weighted'")
        parameters = {"strategy": strategy}
        parameters.update(kwargs)
        self._classes = None
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
        X = _convert_to_woodwork_structure(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        y = _convert_to_woodwork_structure(y)
        y = _convert_woodwork_types_wrapper(y.to_series())

        vals, counts = np.unique(y, return_counts=True)
        self._classes = list(vals)
        self._percentage_freq = counts.astype(float) / len(y)
        self._num_unique = len(self._classes)
        self._num_features = X.shape[1]

        if self.parameters["strategy"] == "mode":
            self._mode = y.mode()[0]
        return self

    def predict(self, X):
        X = _convert_to_woodwork_structure(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        strategy = self.parameters["strategy"]
        if strategy == "mode":
            return pd.Series([self._mode] * len(X))
        elif strategy == "random":
            return self.random_state.choice(self._classes, len(X))
        else:
            return self.random_state.choice(self._classes, len(X), p=self._percentage_freq)

    def predict_proba(self, X):
        X = _convert_to_woodwork_structure(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        strategy = self.parameters["strategy"]
        if strategy == "mode":
            mode_index = self._classes.index(self._mode)
            proba_arr = np.array([[1.0 if i == mode_index else 0.0 for i in range(self._num_unique)]] * len(X))
            return pd.DataFrame(proba_arr, columns=self._classes)
        elif strategy == "random":
            proba_arr = np.array([[1.0 / self._num_unique for i in range(self._num_unique)]] * len(X))
            return pd.DataFrame(proba_arr, columns=self._classes)
        else:
            proba_arr = np.array([[self._percentage_freq[i] for i in range(self._num_unique)]] * len(X))
            return pd.DataFrame(proba_arr, columns=self._classes)

    @property
    def feature_importance(self):
        """Returns importance associated with each feature. Since baseline classifiers do not use input features to calculate predictions, returns an array of zeroes.

        Returns:
            np.ndarray (float): An array of zeroes
        """
        return np.zeros(self._num_features)

    @property
    def classes_(self):
        """Returns class labels. Will return None before fitting.

        Returns:
            list(str) or list(float) : Class names
        """
        return self._classes
