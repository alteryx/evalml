import numpy as np

from evalml.pipelines.components.transformers.transformer import (
    TargetTransformer,
)
from evalml.utils import infer_feature_types


class LogTransformer(TargetTransformer):
    """Applies a log transformation to the target data."""

    name = "Log Transformer"

    hyperparameter_ranges = {}
    """{}"""

    def __init__(self, random_seed=0):
        super().__init__(parameters={}, component_obj=None, random_seed=random_seed)

    def fit(self, X, y=None):
        """Fits the LogTransformer.

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
        y_ww = infer_feature_types(y)
        self.min = y_ww.min()
        if self.min <= 0:
            y_ww = y_ww.apply(lambda x: x + abs(self.min) + 1)
        y_t = infer_feature_types(y_ww.apply(np.log))
        return X, y_t

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
        y_ww_inv = infer_feature_types(y)
        y_inv = y_ww_inv.apply(np.exp)
        if self.min <= 0:
            y_inv = y_inv.apply(lambda x: x - abs(self.min) - 1)
        return infer_feature_types(y_inv)
