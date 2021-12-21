"""Holt-Winters Exponential Smoothing Forecaster."""
import numpy as np
import pandas as pd
from skopt.space import Integer

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import import_or_raise, infer_feature_types


class ExponentialSmoothingRegressor(Estimator):
    """Holt-Winters Exponential Smoothing Forecaster.

    Currently ExponentialSmoothingRegressor isn't supported via conda install. It's recommended that it be installed via PyPI.

    Args:
        n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines. Defaults to -1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Exponential Smoothing Regressor"
    hyperparameter_ranges = {
        "trend": [None, "additive", "multiplicative"],
        "damped_trend": [True, False],
        "seasonal": [None, "additive", "multiplicative"],
        "sp": Integer(1, 12),
        "use_boxcox": [True, False, "log"],
    }
    """{
        "trend": ["additive", "multiplicative"],
        "damped_trend": [True, False],
        "seasonal": ["additive", "multiplicative"],
        "sp": Integer(1, 12),
        "use_boxcox": [True, False, "log"],
    }"""
    model_family = ModelFamily.EXPONENTIAL_SMOOTHING
    """ModelFamily.EXPONENTIAL_SMOOTHING"""
    supported_problem_types = [ProblemTypes.TIME_SERIES_REGRESSION]
    """[ProblemTypes.TIME_SERIES_REGRESSION]"""

    def __init__(
        self,
        trend=None,
        damped_trend=False,
        seasonal=None,
        sp=1,
        use_boxcox=False,
        n_jobs=-1,
        random_seed=0,
        **kwargs,
    ):
        parameters = {
            "trend": trend,
            "damped_trend": damped_trend,
            "seasonal": seasonal,
            "sp": sp,
            "use_boxcox": use_boxcox,
        }
        parameters.update(kwargs)
        smoothing_model_msg = (
            "sktime is not installed. Please install using `pip install sktime.`"
        )
        sktime_smoothing = import_or_raise(
            "sktime.forecasting.exp_smoothing", error_msg=smoothing_model_msg
        )
        smoothing_model = sktime_smoothing.ExponentialSmoothing(**parameters)

        super().__init__(
            parameters=parameters,
            component_obj=smoothing_model,
            random_seed=random_seed,
        )

    def _remove_datetime(self, data, features=False):
        if data is None:
            return None
        data_no_dt = data.copy()
        if isinstance(
            data_no_dt.index, (pd.DatetimeIndex, pd.PeriodIndex, pd.IntervalIndex)
        ):
            data_no_dt = data_no_dt.reset_index(drop=True)
        if features:
            data_no_dt = data_no_dt.select_dtypes(exclude=["datetime64"])

        return data_no_dt

    def _match_indices(self, X, y):
        if X is not None:
            y.index = X.index
        return X, y

    def _set_forecast(self, X):
        from sktime.forecasting.base import ForecastingHorizon

        fh_ = ForecastingHorizon([i + 1 for i in range(len(X))], is_relative=True)
        return fh_

    def fit(self, X, y=None):
        """Fits Exponential Smoothing Regressor to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series): The target training data of length [n_samples].

        Returns:
            self

        Raises:
            ValueError: If y was not passed in.
        """
        X, y = self._manage_woodwork(X, y)
        if y is None:
            raise ValueError("Exponential Smoothing Regressor requires y as input.")

        X = self._remove_datetime(X, features=True)
        y = self._remove_datetime(y)
        X, y = self._match_indices(X, y)

        if X is not None and not X.empty:
            self._component_obj.fit(y=y, X=X)
        else:
            self._component_obj.fit(y=y)
        return self

    def predict(self, X, y=None):
        """Make predictions using fitted Exponential Smoothing regressor.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].
            y (pd.Series): Target data.

        Returns:
            pd.Series: Predicted values.
        """
        X, y = self._manage_woodwork(X, y)
        fh_ = self._set_forecast(X)
        X = X.select_dtypes(exclude=["datetime64"])

        if not X.empty:
            y_pred = self._component_obj.predict(fh=fh_, X=X)
        else:
            y_pred = self._component_obj.predict(fh=fh_)
        y_pred.index = X.index

        return infer_feature_types(y_pred)

    @property
    def feature_importance(self):
        """Returns array of 0's with a length of 1 as feature_importance is not defined for Exponential Smoothing regressor."""
        return np.zeros(1)
