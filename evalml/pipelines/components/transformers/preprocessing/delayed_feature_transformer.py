"""Transformer that delays input features and target variable for time series problems."""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from woodwork import logical_types

from evalml.pipelines.components.transformers.transformer import Transformer
from evalml.utils import infer_feature_types


class DelayedFeatureTransformer(Transformer):
    """Transformer that delays input features and target variable for time series problems.

    Args:
        date_index (str): Name of the column containing the datetime information used to order the data. Ignored.
        max_delay (int): Maximum number of time units to delay each feature. Defaults to 2.
        forecast_horizon (int): The number of time periods the pipeline is expected to forecast.
        delay_features (bool): Whether to delay the input features. Defaults to True.
        delay_target (bool): Whether to delay the target. Defaults to True.
        gap (int): The number of time units between when the features are collected and
            when the target is collected. For example, if you are predicting the next time step's target, gap=1.
            This is only needed because when gap=0, we need to be sure to start the lagging of the target variable
            at 1. Defaults to 1.
        random_seed (int): Seed for the random number generator. This transformer performs the same regardless of the random seed provided.
    """

    name = "Delayed Feature Transformer"
    hyperparameter_ranges = {}
    """{}"""
    needs_fitting = False
    target_colname_prefix = "target_delay_{}"
    """target_delay_{}"""

    def __init__(
        self,
        date_index=None,
        max_delay=2,
        gap=0,
        forecast_horizon=1,
        delay_features=True,
        delay_target=True,
        random_seed=0,
        **kwargs,
    ):
        self.date_index = date_index
        self.max_delay = max_delay
        self.delay_features = delay_features
        self.delay_target = delay_target
        self.forecast_horizon = forecast_horizon
        self.gap = gap

        self.start_delay = self.forecast_horizon + self.gap

        parameters = {
            "date_index": date_index,
            "max_delay": max_delay,
            "delay_target": delay_target,
            "delay_features": delay_features,
            "forecast_horizon": forecast_horizon,
            "gap": gap,
        }
        parameters.update(kwargs)
        super().__init__(parameters=parameters, random_seed=random_seed)

    def fit(self, X, y=None):
        """Fits the DelayFeatureTransformer.

        Args:
            X (pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (pd.Series, optional): The target training data of length [n_samples]

        Returns:
            self
        """
        return self

    @staticmethod
    def _encode_y_while_preserving_index(y):
        y_encoded = LabelEncoder().fit_transform(y)
        y = pd.Series(y_encoded, index=y.index)
        return y

    @staticmethod
    def _get_categorical_columns(X):
        return list(X.ww.select(["categorical"], return_schema=True).columns)

    @staticmethod
    def _encode_X_while_preserving_index(X_categorical):
        return pd.DataFrame(
            OrdinalEncoder().fit_transform(X_categorical),
            columns=X_categorical.columns,
            index=X_categorical.index,
        )

    def transform(self, X, y=None):
        """Computes the delayed features for all features in X and y.

        For each feature in X, it will add a column to the output dataframe for each
        delay in the (inclusive) range [1, max_delay]. The values of each delayed feature are simply the original
        feature shifted forward in time by the delay amount. For example, a delay of 3 units means that the feature
        value at row n will be taken from the n-3rd row of that feature

        If y is not None, it will also compute the delayed values for the target variable.

        Args:
            X (pd.DataFrame or None): Data to transform. None is expected when only the target variable is being used.
            y (pd.Series, or None): Target.

        Returns:
            pd.DataFrame: Transformed X.
        """
        if X is None:
            X = pd.DataFrame()
        # Normalize the data into pandas objects
        X_ww = infer_feature_types(X)
        X_ww = X_ww.ww.copy()
        categorical_columns = self._get_categorical_columns(X_ww)
        original_features = list(X_ww.columns)
        if self.delay_features and len(X) > 0:
            X_categorical = self._encode_X_while_preserving_index(
                X_ww[categorical_columns]
            )
            for col_name in X_ww:
                col = X_ww[col_name]
                if col_name in categorical_columns:
                    col = X_categorical[col_name]
                for t in range(self.start_delay, self.start_delay + self.max_delay + 1):
                    X_ww.ww[f"{col_name}_delay_{t}"] = col.shift(t)
        # Handle cases where the target was passed in
        if self.delay_target and y is not None:
            y = infer_feature_types(y)
            if type(y.ww.logical_type) == logical_types.Categorical:
                y = self._encode_y_while_preserving_index(y)
            for t in range(self.start_delay, self.start_delay + self.max_delay + 1):
                X_ww.ww[self.target_colname_prefix.format(t)] = y.shift(t)
        return X_ww.ww.drop(original_features)

    def fit_transform(self, X, y):
        """Fit the component and transform the input data.

        Args:
            X (pd.DataFrame or None): Data to transform. None is expected when only the target variable is being used.
            y (pd.Series, or None): Target.

        Returns:
            pd.DataFrame: Transformed X.
        """
        return self.fit(X, y).transform(X, y)
