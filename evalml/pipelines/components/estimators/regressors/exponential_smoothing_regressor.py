"""Holt-Winters Exponential Smoothing Forecaster."""

from typing import Dict, List, Optional, Union

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
        trend (str): Type of trend component. Defaults to None.
        damped_trend (bool): If the trend component should be damped. Defaults to False.
        seasonal (str): Type of seasonal component. Takes one of {“additive”, None}. Can also be multiplicative if
        none of the target data is 0, but AutoMLSearch wiill not tune for this. Defaults to None.
        sp (int): The number of seasonal periods to consider. Defaults to 2.
        n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines. Defaults to -1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Exponential Smoothing Regressor"
    hyperparameter_ranges = {
        "trend": [None, "additive"],
        "damped_trend": [True, False],
        "seasonal": [None, "additive"],
        "sp": Integer(2, 8),
    }
    """{
        "trend": [None, "additive"],
        "damped_trend": [True, False],
        "seasonal": [None, "additive"],
        "sp": Integer(2, 8),
    }"""
    model_family = ModelFamily.EXPONENTIAL_SMOOTHING
    """ModelFamily.EXPONENTIAL_SMOOTHING"""
    supported_problem_types = [ProblemTypes.TIME_SERIES_REGRESSION]
    """[ProblemTypes.TIME_SERIES_REGRESSION]"""

    def __init__(
        self,
        trend: Optional[str] = None,
        damped_trend: bool = False,
        seasonal: Optional[str] = None,
        sp: int = 2,
        n_jobs: int = -1,
        random_seed: Union[int, float] = 0,
        **kwargs,
    ):
        if trend is None:
            damped_trend = False

        parameters = {
            "trend": trend,
            "damped_trend": damped_trend,
            "seasonal": seasonal,
            "sp": sp,
            "random_state": random_seed,
        }
        parameters.update(kwargs)
        smoothing_model_msg = (
            "sktime is not installed. Please install using `pip install sktime.`"
        )
        sktime_smoothing = import_or_raise(
            "sktime.forecasting.exp_smoothing",
            error_msg=smoothing_model_msg,
        )
        smoothing_model = sktime_smoothing.ExponentialSmoothing(**parameters)

        super().__init__(
            parameters=parameters,
            component_obj=smoothing_model,
            random_seed=random_seed,
        )

    def _remove_datetime(self, data: pd.DataFrame) -> pd.DataFrame:
        data_no_dt = data.copy()
        if isinstance(
            data_no_dt.index,
            (pd.DatetimeIndex, pd.PeriodIndex, pd.IntervalIndex),
        ):
            data_no_dt = data_no_dt.reset_index(drop=True)

        return data_no_dt

    def _set_forecast(self, X: pd.DataFrame):
        from sktime.forecasting.base import ForecastingHorizon

        fh_ = ForecastingHorizon([i + 1 for i in range(len(X))], is_relative=True)
        return fh_

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fits Exponential Smoothing Regressor to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features]. Ignored.
            y (pd.Series): The target training data of length [n_samples].

        Returns:
            self

        Raises:
            ValueError: If y was not passed in.
        """
        X, y = self._manage_woodwork(X, y)
        if y is None:
            raise ValueError("Exponential Smoothing Regressor requires y as input.")

        y = self._remove_datetime(y)

        self._component_obj.fit(y=y)
        return self

    def predict(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.Series:
        """Make predictions using fitted Exponential Smoothing regressor.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features]. Ignored except to set forecast horizon.
            y (pd.Series): Target data.

        Returns:
            pd.Series: Predicted values.
        """
        X, y = self._manage_woodwork(X, y)
        fh_ = self._set_forecast(X)

        y_pred = self._component_obj.predict(fh=fh_)
        y_pred.index = X.index
        y_pred.name = None
        return infer_feature_types(y_pred)

    def get_prediction_intervals(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        coverage: List[float] = None,
        predictions: pd.Series = None,
    ) -> Dict[str, pd.Series]:
        """Find the prediction intervals using the fitted ExponentialSmoothingRegressor.

        Calculates the prediction intervals by using a simulation of the time series following a specified state space model.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].
            y (pd.Series): Target data. Optional.
            coverage (List[float]): A list of floats between the values 0 and 1 that the upper and lower bounds of the
                prediction interval should be calculated for.
            predictions (pd.Series): Not used for Exponential Smoothing regressor.

        Returns:
            dict: Prediction intervals, keys are in the format {coverage}_lower or {coverage}_upper.
        """
        if coverage is None:
            coverage = [0.95]
        X, y = self._manage_woodwork(X, y)
        # Accesses the fitted statsmodels model within sktime
        # nsimulations represents how many steps should be simulated
        # repetitions represents the number of simulations that should be run (confusing, I know)
        # anchor represents where the simulations should start from (forecasting is done from the "end")
        y_pred = self._component_obj._fitted_forecaster.simulate(
            nsimulations=X.shape[0],
            repetitions=400,
            anchor="end",
            random_state=self.parameters["random_state"],
        )
        prediction_interval_result = {}
        for conf_int in coverage:
            prediction_interval_lower = y_pred.quantile(
                q=round((1 - conf_int) / 2, 3),
                axis="columns",
            )
            prediction_interval_upper = y_pred.quantile(
                q=round((1 + conf_int) / 2, 3),
                axis="columns",
            )
            prediction_interval_lower.index = X.index
            prediction_interval_upper.index = X.index
            prediction_interval_result[f"{conf_int}_lower"] = prediction_interval_lower
            prediction_interval_result[f"{conf_int}_upper"] = prediction_interval_upper
        return prediction_interval_result

    @property
    def feature_importance(self) -> pd.Series:
        """Returns array of 0's with a length of 1 as feature_importance is not defined for Exponential Smoothing regressor."""
        return pd.Series(np.zeros(1))
