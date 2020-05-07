import numpy as np
import pandas as pd

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class ZeroRClassifier(Estimator):
    """Classifier that predicts using the mode. In the case where there is no single mode, the lowest value is used.

    This is useful as a simple baseline classifier to compare with other classifiers.
    """
    name = "ZeroR Classifier"
    hyperparameter_ranges = {}
    model_family = ModelFamily.NONE
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def __init__(self, random_state=0):
        """TODO"""
        parameters = {}
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
            raise ValueError("Cannot fit ZeroR classifier if y is None")

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        self.mode = y.mode()[0]
        self.num_unique = len(y.value_counts())
        self.num_features = len(X)
        return self

    def predict(self, X):
        """Make predictions using selected features.

        Args:
            X (pd.DataFrame) : features

        Returns:
            pd.Series : estimated labels
        """
        try:
            mode = self.mode
        except AttributeError:
            raise RuntimeError("You must fit ZeroR classifier before calling predict!")
        return pd.Series([mode] * len(X))

    def predict_proba(self, X):
        """Make probability estimates for labels.

        Args:
            X (pd.DataFrame) : features

        Returns:
            np.array : probability estimates
        """
        try:
            mode = self.mode
            num_unique = self.num_unique
        except AttributeError:
            raise RuntimeError("You must fit ZeroR classifier before calling predict_proba!")
        return np.array([[1.0 if i == mode else 0.0 for i in range(num_unique)]] * len(X))

    @property
    def feature_importances(self):
        """Returns feature importances.

        Returns:
            np.array (float) : importance associated with each feature

        """
        try:
            num_features = self.num_features
        except AttributeError:
            raise RuntimeError("You must fit ZeroR classifier before gettong feature_importances!")
        return np.array([0.0] * num_features)
