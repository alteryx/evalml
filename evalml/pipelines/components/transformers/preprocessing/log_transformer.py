import numpy as np

from evalml.pipelines.components.transformers.transformer import (
    TargetTransformer,
)
from evalml.utils import infer_feature_types

class LogTransform(TargetTransformer):
    """Applies a log transformation to the data."""

    name = "Log Transformer"

    def __init__(self, random_seed=0):
        super().__init__(parameters={}, component_obj=None, random_seed=random_seed)

    def fit(self, X, y):
        """Fits the LogTransform.

        Arguments:
            X (pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (pd.Series, optional): The target training data of length [n_samples]

        Returns:
            self
        """
        return self

    def transform(self, X, y=None):
        """Removes fitted trend from target variable.

        Arguments:
            X (pd.DataFrame, optional): Ignored.
            y (pd.Series): Target variable to log transform.

        Returns:
            tuple of pd.DataFrame, pd.Series: The input features are returned without modification. The target
                variable y is detrended
        """
        if y is None:
            return X, y
        y = infer_feature_types(y)
        return X, infer_feature_types(np.log(y))

    def inverse_transform(self, y):
        y = infer_feature_types(y)
        return infer_feature_types(np.exp(y))