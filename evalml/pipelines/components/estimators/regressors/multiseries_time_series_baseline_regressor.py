"""Time series estimator that predicts using the naive forecasting approach."""
import numpy as np
import pandas as pd

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.pipelines.components.transformers import TimeSeriesFeaturizer
from evalml.problem_types import ProblemTypes
from evalml.utils import infer_feature_types


class MultiseriesTimeSeriesBaselineRegressor(Estimator):
    """Multiseries time series regressor that predicts using the naive forecasting approach.

    This is useful as a simple baseline estimator for multiseries time series problems.

    Args:
        gap (int): Gap between prediction date and target date and must be a positive integer. If gap is 0, target date will be shifted ahead by 1 time period. Defaults to 1.
        forecast_horizon (int): Number of time steps the model is expected to predict.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Multiseries Time Series Baseline Regressor"
    hyperparameter_ranges = {}
    """{}"""
    model_family = ModelFamily.BASELINE
    """ModelFamily.BASELINE"""
    is_multiseries = True
    supported_problem_types = [
        ProblemTypes.TIME_SERIES_REGRESSION,
    ]
    """[
        ProblemTypes.TIME_SERIES_REGRESSION,
    ]"""

    def __init__(self, gap=1, forecast_horizon=1, random_seed=0, **kwargs):
        self._prediction_value = None
        self.start_delay = forecast_horizon + gap
        self._num_features = None

        if gap < 0:
            raise ValueError(
                f"gap value must be a positive integer. {gap} was provided.",
            )

        parameters = {"gap": gap, "forecast_horizon": forecast_horizon}
        parameters.update(kwargs)
        super().__init__(
            parameters=parameters,
            component_obj=None,
            random_seed=random_seed,
        )

    def fit(self, X, y=None):
        """Fits multiseries time series baseline regressor to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features * n_series].
            y (pd.DataFrame): The target training data of shape [n_samples, n_features * n_series].

        Returns:
            self

        Raises:
            ValueError: If input y is None or if y is not a DataFrame with multiple columns.
        """
        if y is None:
            raise ValueError(
                "Cannot train Multiseries Time Series Baseline Regressor if y is None",
            )
        if isinstance(y, pd.Series):
            raise ValueError(
                "y must be a DataFrame with multiple columns for Multiseries Time Series Baseline Regressor",
            )
        self._target_column_names = list(y.columns)
        self._num_features = X.shape[1]

        return self

    def predict(self, X):
        """Make predictions using fitted multiseries time series baseline regressor.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].

        Returns:
            pd.DataFrame: Predicted values.

        Raises:
            ValueError: If the lagged columns are not present in X.
        """
        X = infer_feature_types(X)
        feature_names = [
            TimeSeriesFeaturizer.df_colname_prefix.format(col, self.start_delay)
            for col in self._target_column_names
        ]
        if not set(feature_names).issubset(set(X.columns)):
            raise ValueError(
                "Multiseries Time Series Baseline Regressor is meant to be used in a pipeline with "
                "a Time Series Featurizer",
            )
        delayed_features = X.ww[feature_names]

        # Get the original column names, rather than the lagged column names
        new_column_names = {
            col_name: col_name.split("_delay_")[0] for col_name in feature_names
        }

        return delayed_features.ww.rename(columns=new_column_names)

    @property
    def feature_importance(self):
        """Returns importance associated with each feature.

        Since baseline estimators do not use input features to calculate predictions, returns an array of zeroes.

        Returns:
            np.ndarray (float): An array of zeroes.
        """
        importance = np.array([0] * self._num_features)
        return importance
