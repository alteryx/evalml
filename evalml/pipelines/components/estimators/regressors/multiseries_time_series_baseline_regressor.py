"""Time series estimator that predicts using the naive forecasting approach."""
import numpy as np
import pandas as pd

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
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

    name = "Time Series Baseline Regressor"
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
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series): The target training data of length [n_samples].

        Returns:
            self

        Raises:
            ValueError: If input y is None.
        """
        if y is None:
            raise ValueError(
                "Cannot train Multiseries Time Series Baseline Regressor if y is None",
            )
        self._series_names = y.columns

        delay_columns = pd.DataFrame(
            np.zeros((self.start_delay, y.shape[1])),
            columns=self._series_names,
            index=range(y.index[-1], self.start_delay + y.index[-1]),
        )
        y = pd.concat([y, delay_columns])

        self._delayed_target = y.shift(self.start_delay, fill_value=0)

        return self

    def predict(self, X):
        """Make predictions using fitted multiseries time series baseline regressor.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].

        Returns:
            pd.Series: Predicted values.

        Raises:
            ValueError: If input y is None.
        """
        X = infer_feature_types(X)
        self._num_features = X.shape[1]

        in_sample_delay = self._delayed_target[self._delayed_target.index.isin(X.index)]

        out_of_sample_delay = pd.DataFrame(columns=self._series_names)
        out_of_sample_offset = X.index[-1] - self._delayed_target.index[-1]
        if out_of_sample_offset > 0:
            out_of_sample_delay = pd.DataFrame(
                np.zeros((out_of_sample_offset, len(self._series_names))),
                columns=self._series_names,
                index=range(self._delayed_target.index[-1] + 1, X.index[-1] + 1),
            )

        y_pred = pd.concat([in_sample_delay, out_of_sample_delay])
        return y_pred

    @property
    def feature_importance(self):
        """Returns importance associated with each feature.

        Since baseline estimators do not use input features to calculate predictions, returns an array of zeroes.

        Returns:
            np.ndarray (float): An array of zeroes.
        """
        importance = np.array([0] * self._num_features)
        return importance
