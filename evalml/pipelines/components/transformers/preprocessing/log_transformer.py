import numpy as np

from evalml.pipelines.components.transformers.transformer import (
    TargetTransformer,
)
from evalml.utils import infer_feature_types


class LogTransformer(TargetTransformer):
    """Applies a log transformation to the target data."""

    name = "Log Transformer"

    hyperparameter_ranges = {}

    def __init__(self, random_seed=0):
        super().__init__(parameters={}, component_obj=None, random_seed=random_seed)

    def fit(self, X, y=None):
        """Fits the LogTransform.

        Arguments:
            X (pd.DataFrame or np.ndarray): Ignored.
            y (pd.Series, optional): Ignored.

        Returns:
            self
        """
        return self

    def transform(self, X, y=None):
        """Log transforms the target variable.

        Arguments:
            X (pd.DataFrame, optional): Ignored.
            y (pd.Series): Target data to log transform.

        Returns:
            tuple of pd.DataFrame, pd.Series: The input features are returned without modification. The target
                variable y is log transformed.
        """
        if y is None:
            return X, y
        self.min = y.min()
        if self.min <= 0:
            y = y + abs(self.min) + 1
        y = infer_feature_types(np.log(y))
        return X, y

    def fit_transform(self, X, y=None):
        """Log transforms the target variable.

        Arguments:
            X (pd.DataFrame, optional): Ignored.
            y (pd.Series): Target variable to log transform.

        Returns:
            tuple of pd.DataFrame, pd.Series: The input features are returned without modification. The target
                variable y is log transformed.
        """
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, y):
        y = np.exp(y)
        if self.min <= 0:
            y = y - abs(self.min) - 1
        y = infer_feature_types(y)
        return y