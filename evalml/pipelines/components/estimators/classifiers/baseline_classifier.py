"""Baseline classifier."""
import numpy as np
import pandas as pd

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import get_random_state, infer_feature_types


class BaselineClassifier(Estimator):
    """Classifier that predicts using the specified strategy.

    This is useful as a simple baseline classifier to compare with other classifiers.

    Args:
        strategy (str): Method used to predict. Valid options are "mode", "random" and "random_weighted". Defaults to "mode".
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Baseline Classifier"
    hyperparameter_ranges = {}
    """{}"""
    model_family = ModelFamily.BASELINE
    """ModelFamily.BASELINE"""
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
    """[ProblemTypes.BINARY, ProblemTypes.MULTICLASS]"""

    def __init__(self, strategy="mode", random_seed=0, **kwargs):
        if strategy not in ["mode", "random", "random_weighted"]:
            raise ValueError(
                "'strategy' parameter must equal either 'mode', 'random', or 'random_weighted'"
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
        """Fits baseline classifier component to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series): The target training data of length [n_samples].

        Returns:
            self

        Raises:
            ValueError: If y is None.
        """
        if y is None:
            raise ValueError("Cannot fit Baseline classifier if y is None")
        X = infer_feature_types(X)
        y = infer_feature_types(y)

        vals, counts = np.unique(y, return_counts=True)
        self._classes = list(vals)
        self._percentage_freq = counts.astype(float) / len(y)
        self._num_unique = len(self._classes)
        self._num_features = X.shape[1]

        if self.parameters["strategy"] == "mode":
            self._mode = y.mode()[0]
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
        if strategy == "mode":
            predictions = pd.Series([self._mode] * len(X))
        elif strategy == "random":
            predictions = get_random_state(self.random_seed).choice(
                self._classes, len(X)
            )
        else:
            predictions = get_random_state(self.random_seed).choice(
                self._classes, len(X), p=self._percentage_freq
            )
        return infer_feature_types(predictions)

    def predict_proba(self, X):
        """Make prediction probabilities using the baseline classification strategy.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].

        Returns:
            pd.DataFrame: Predicted probability values.
        """
        X = infer_feature_types(X)
        strategy = self.parameters["strategy"]
        if strategy == "mode":
            mode_index = self._classes.index(self._mode)
            proba_arr = np.array(
                [[1.0 if i == mode_index else 0.0 for i in range(self._num_unique)]]
                * len(X)
            )
        elif strategy == "random":
            proba_arr = np.array(
                [[1.0 / self._num_unique for i in range(self._num_unique)]] * len(X)
            )
        else:
            proba_arr = np.array(
                [[self._percentage_freq[i] for i in range(self._num_unique)]] * len(X)
            )
        predictions = pd.DataFrame(proba_arr, columns=self._classes)
        return infer_feature_types(predictions)

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
            list[str] or list(float) : Class names
        """
        return self._classes
