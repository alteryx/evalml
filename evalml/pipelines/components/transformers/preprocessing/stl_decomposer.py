"""Component that removes trends and seasonality from time series using STL."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.seasonal import STL as STL

from evalml.pipelines.components.transformers.preprocessing.decomposer import Decomposer
from evalml.pipelines.components.transformers.preprocessing.stl import STL as github_STL
from evalml.utils import import_or_raise, infer_feature_types


class STLDecomposer(Decomposer):

    name = "STL Decomposer"
    hyperparameter_ranges = {}

    modifies_features = False
    modifies_target = True

    def __init__(
        self,
        time_index: str = None,
        degree: int = 1,
        seasonal_period: int = 367,
        random_seed: int = 0,
        **kwargs,
    ):
        self.logger = logging.getLogger(__name__)

        if seasonal_period % 2 == 0:
            self.logger.warning(
                f"STLDecomposer provided with an even period of {seasonal_period}"
                f"Changing seasonal period to {seasonal_period+1}",
            )
            seasonal_period += 1

        super().__init__(
            component_obj=None,
            random_seed=random_seed,
            degree=degree,
            seasonal_period=seasonal_period,
            time_index=time_index,
            **kwargs,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> STLDecomposer:
        if y is None:
            raise ValueError("y cannot be None for STLDecomposer!")

        if not isinstance(y.index, pd.DatetimeIndex):
            y = self._set_time_index(X, y)

        # Warn for poor decomposition use with higher periods
        if self.seasonal_period > 14:
            str_dict = {"D": "daily", "M": "monthly"}
            data_str = ""
            if y.index.freqstr in str_dict:
                data_str = str_dict[y.index.freqstr]
            self.logger.warning(
                f"STLDecomposer may perform poorly on {data_str} data with a high seasonal period ({self.seasonal_period}).",
            )

        self._component_obj = STL(y, seasonal=self.seasonal_period)
        res = self._component_obj.fit()
        self.seasonal = res.seasonal
        self.trend = res.trend
        self.residual = res.resid
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        return X, self.residual

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, y_t):
        if y_t is None:
            raise ValueError("y_t cannot be None for STLDecomposer!")
        y_t = infer_feature_types(y_t)

        if all(y_t.index == self.trend.index):
            y = y_t + self.trend + self.seasonal
        else:
            # Determine how many units forward to forecast
            units_forward = 25

            # Model the trend and project it forward
            stlf = STLForecast(
                self.trend,
                ARIMA,
                model_kwargs=dict(order=(1, 1, 0), trend="t"),
            )
            stlf_res = stlf.fit()
            projected_trend = stlf_res.forecast(units_forward)

            # Reseasonalize
            projected_seasonal = self._build_seasonal_signal(
                y_t,
                self.seasonality,
                self.periodicity,
                self.frequency,
            )
            y = infer_feature_types(
                pd.Series(y_t + projected_trend + projected_seasonal, index=y_t.index),
            )
        return y

    def get_trend_dataframe(self, y):
        pass
