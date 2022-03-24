"""A transformer that standardizes input features to a given range."""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as SkScaler

from evalml.pipelines.components.transformers import Transformer
from evalml.utils import infer_feature_types


class MinMaxScaler(Transformer):
    """A transformer that standardizes input features to a given range.

    Args:
        feature_range (tuple): (min, max) range for the data transformation. Defaults to (0, 1).
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "MinMax Scaler"
    hyperparameter_ranges = {}
    """{}"""

    def __init__(self, feature_range=(0, 1), random_seed=0, **kwargs):
        parameters = {"feature_range": feature_range}
        parameters.update(kwargs)

        scaler = SkScaler(**parameters)
        super().__init__(
            parameters=parameters, component_obj=scaler, random_seed=random_seed
        )

    def fit(self, X, y=None):
        """Fit the minmax scaler component to the given data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            pd.DataFrame: Transformed data.
        """
        X = infer_feature_types(X)
        X_numeric = X.ww.select("numeric")
        if len(X_numeric.columns) == 0:
            return self

        self._component_obj.fit(X_numeric)
        return self

    def transform(self, X, y=None):
        """Transform data using the fitted minmax scaler.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            pd.DataFrame: Transformed data.
        """
        X = infer_feature_types(X)
        X_numeric = X.ww.select("numeric")
        X_numeric_cols = list(X_numeric.columns)
        if len(X_numeric_cols) == 0:
            return X

        X_t = self._component_obj.transform(X_numeric)
        X_t_df = pd.DataFrame(X_t, columns=X_numeric_cols, index=X.index)
        X[X_numeric_cols] = X_t_df

        return X
