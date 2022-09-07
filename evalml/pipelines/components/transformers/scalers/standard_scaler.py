"""A transformer that standardizes input features by removing the mean and scaling to unit variance."""
import pandas as pd
from sklearn.preprocessing import StandardScaler as SkScaler

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
        self._supported_types = [
            "Age",
            "AgeNullable",
            "Double",
            "Integer",
            "IntegerNullable",
        ]
        scaler = SkScaler(**parameters)
        super().__init__(
            parameters=parameters,
            component_obj=scaler,
            random_seed=random_seed,
        )

    def fit(self, X, y=None):
        """Fits the standard scalar on the given data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            self
        """
        X = infer_feature_types(X)
        X_scalable = X.ww.select(self._supported_types)
        self.scaled_columns = list(X_scalable.columns)
        if X_scalable.empty:
            return self
        self._component_obj.fit(X_scalable)
        return self

    def transform(self, X, y=None):
        """Transform data using the fitted standard scaler.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            pd.DataFrame: Transformed data.
        """
        X = infer_feature_types(X)

        X = X.ww.select(exclude=["datetime"])
        if not self.scaled_columns:
            return X
        X_scaled_columns = X.ww[self.scaled_columns]
        scaled = self._component_obj.transform(X_scaled_columns)
        X[self.scaled_columns] = scaled
        X.ww.set_types(logical_types={col: "Double" for col in self.scaled_columns})
        return X

    def fit_transform(self, X, y=None):
        """Fit and transform data using the standard scaler component.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            pd.DataFrame: Transformed data.
        """
        if not isinstance(X, pd.DataFrame):
            X = infer_feature_types(X)
        X = X.select_dtypes(exclude=["datetime"])
        return self.fit(X, y).transform(X, y)
