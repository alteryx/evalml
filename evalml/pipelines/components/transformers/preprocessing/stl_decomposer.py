"""Component that removes trends and seasonality from time series using STL."""
from __future__ import annotations

import logging

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.seasonal import STL as STL

from evalml.pipelines.components.transformers.preprocessing.decomposer import Decomposer
from evalml.utils import infer_feature_types


class STLDecomposer(Decomposer):

    name = "STL Decomposer"
    hyperparameter_ranges = {}

    modifies_features = False
    modifies_target = True

    def __init__(
        self,
        time_index: str = None,
        degree: int = 1,
        seasonal_period: int = 7,
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
        self.seasonality = self.seasonal[: self.seasonal_period]
        self.trend = res.trend
        self.residual = res.resid

        # Save the frequency of the fitted series for checking against transform data.
        self.frequency = y.index.freqstr
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

        if len(y_t.index) == len(self.trend.index) and all(
            y_t.index == self.trend.index,
        ):
            y = y_t + self.trend + self.seasonal
        else:
            # Determine how many units forward to forecast
            right_delta = y_t.index[-1] - self.trend.index[-1]
            if y_t.index[-1] < self.trend.index[0] or (
                self.trend.index[-1] > y_t.index[0]
                and self.trend.index[-1] < y_t.index[-1]
            ):
                raise ValueError(
                    f"STLDecomposer cannot recompose/inverse transform data out of sample and before the data used"
                    f"to fit the decomposer, or partially in and out of sample."
                    f"\nRequested date range: {str(y_t.index[0])}:{str(y_t.index[-1])}."
                    f"\nSample date range: {str(self.trend.index[0])}:{str(self.trend.index[-1])}.",
                )
            delta = pd.to_timedelta(1, self.frequency)

            # Model the trend and project it forward
            stlf = STLForecast(
                self.trend,
                ARIMA,
                model_kwargs=dict(order=(1, 1, 0), trend="t"),
            )
            stlf_res = stlf.fit()
            forecast = stlf_res.forecast(int(right_delta / delta))
            overlapping_ind = [ind for ind in y_t.index if ind in forecast.index]
            right_projected_trend = forecast[overlapping_ind]

            # Reseasonalize
            projected_seasonal = self._build_seasonal_signal(
                y_t,
                self.seasonality,
                self.seasonal_period,
                self.frequency,
            )

            y = infer_feature_types(
                pd.Series(
                    y_t + right_projected_trend + projected_seasonal,
                    index=y_t.index,
                ),
            )
        return y

    def get_trend_dataframe(self, X, y):
        """Return a list of dataframes with 3 columns: trend, seasonality, residual.

        Scikit-learn's PolynomialForecaster is used to generate the trend portion of the target data. statsmodel's
        seasonal_decompose is used to generate the seasonality of the data.

        Args:
            X (pd.DataFrame): Input data with time series data in index.
            y (pd.Series or pd.DataFrame): Target variable data provided as a Series for univariate problems or
                a DataFrame for multivariate problems.

        Returns:
            list of pd.DataFrame: Each DataFrame contains the columns "signal", "trend", "seasonality" and "residual,"
                with the latter 3 column values being the decomposed elements of the target data.  The "signal" column
                is simply the input target signal but reindexed with a datetime index to match the input features.

        Raises:
            TypeError: If X does not have time-series data in the index.
            ValueError: If time series index of X does not have an inferred frequency.
            ValueError: If the forecaster associated with the detrender has not been fit yet.
            TypeError: If y is not provided as a pandas Series or DataFrame.

        """
        X = infer_feature_types(X)
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("Provided X should have datetimes in the index.")
        if X.index.freq is None:
            raise ValueError(
                "Provided DatetimeIndex of X should have an inferred frequency.",
            )
        # Change the y index to a matching datetimeindex or else we get a failure
        # in ForecastingHorizon during decomposition.
        if not isinstance(y.index, pd.DatetimeIndex):
            y = self._set_time_index(X, y)

        return pd.DataFrame(
            {
                "signal": y,
                "trend": self.trend,
                "seasonality": self.seasonal,
                "residual": self.residual,
            },
        )
