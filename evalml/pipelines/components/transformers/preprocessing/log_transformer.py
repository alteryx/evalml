"""Component that applies a log transformation to the target data."""
import numpy as np
import pandas as pd

from evalml.pipelines.components.transformers.transformer import Transformer
from evalml.utils import infer_feature_types


class LogTransformer(Transformer):
    """Applies a log transformation to the target data."""

    name = "Log Transformer"

    hyperparameter_ranges = {}
    """{}"""
    modifies_features = False
    modifies_target = True

    def __init__(self, random_seed=0):
        super().__init__(parameters={}, component_obj=None, random_seed=random_seed)

    def fit(self, X, y=None):
        """Fits the LogTransformer.

        Args:
            X (pd.DataFrame or np.ndarray): Ignored.
            y (pd.Series, optional): Ignored.

        Returns:
            self
        """
        return self

    def transform(self, X, y=None):
        """Log transforms the target variable.

        Args:
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
        y_t = y_ww.apply(np.log)
        y_t.ww.init(logical_type="double")
        return X, y_t

    def fit_transform(self, X, y=None):
        """Log transforms the target variable.

        Args:
            X (pd.DataFrame, optional): Ignored.
            y (pd.Series): Target variable to log transform.

        Returns:
            tuple of pd.DataFrame, pd.Series: The input features are returned without modification. The target
                variable y is log transformed.
        """
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, y):
        """Apply exponential to target data.

        Args:
            y (pd.Series): Target variable.

        Returns:
            pd.Series: Target with exponential applied.

        """
        y_ww = infer_feature_types(y)
        y_inv = y_ww.apply(np.exp)
        if self.min <= 0:
            y_inv = y_inv.apply(lambda x: x - abs(self.min) - 1)

        y_inv = infer_feature_types(pd.Series(y_inv, index=y_ww.index))
        return y_inv
