"""A transformer that standardizes input features by removing the mean and scaling to unit variance."""
import pandas as pd
from sklearn.preprocessing import StandardScaler as SkScaler
from woodwork.logical_types import Boolean, Categorical, Integer

from evalml.pipelines.components.transformers import Transformer
from evalml.utils import infer_feature_types


class StandardScaler(Transformer):
    """A transformer that standardizes input features by removing the mean and scaling to unit variance.

    Args:
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Standard Scaler"
    hyperparameter_ranges = {}
    """{}"""

    def __init__(self, random_seed=0, **kwargs):
        parameters = {}
        parameters.update(kwargs)

        scaler = SkScaler(**parameters)
        super().__init__(
            parameters=parameters, component_obj=scaler, random_seed=random_seed
        )

    def transform(self, X, y=None):
        """Transform data using the fitted standard scaler.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            pd.DataFrame: Transformed data.
        """
        X = infer_feature_types(X)
        X = X.ww.select_dtypes(exclude=["datetime"])
        X_t = self._component_obj.transform(X)
        X_t_df = pd.DataFrame(X_t, columns=X.columns, index=X.index)

        schema = X.ww.select(
            exclude=[Integer, Categorical, Boolean], return_schema=True
        )
        X_t_df.ww.init(schema=schema)
        return X_t_df

    def fit_transform(self, X, y=None):
        """Fit and transform data using the standard scaler component.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            pd.DataFrame: Transformed data.
        """
        X = infer_feature_types(X)
        X = X.select_dtypes(exclude=["datetime"])
        return self.fit(X, y).transform(X, y)
