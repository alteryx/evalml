"""Component that removes trends and seasonality from time series using STL."""
from __future__ import annotations

import logging

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.seasonal import STL as STL

from evalml.pipelines.components.transformers.preprocessing.decomposer import Decomposer
from evalml.utils import infer_feature_types

# def fit_check(method):
#     def inner(ref):
#         if not ref.is_fit:
#             raise ValueError(
#                 "STLDecomposer has not been fit yet.  Please fit it and then build the decomposed dataframe.",
#             )
#         else:
#             return method(ref)
#     return inner


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

    def _check_oos_past(self, y):
        if y.index[-1] < self.trend.index[0]:
            raise ValueError(
                f"STLDecomposer cannot transform/inverse transform data out of sample and before the data used"
                f"to fit the decomposer."
                f"\nRequested date range: {str(y.index[0])}:{str(y.index[-1])}."
                f"\nSample date range: {str(self.trend.index[0])}:{str(self.trend.index[-1])}.",
            )

    def _project_trend(self, y):
        right_delta = y.index[-1] - self.trend.index[-1]
        self._check_oos_past(y)
        delta = pd.to_timedelta(1, self.frequency)

        # Model the trend and project it forward
        stlf = STLForecast(
            self.trend,
            ARIMA,
            model_kwargs=dict(order=(1, 1, 0), trend="t"),
        )
        stlf_res = stlf.fit()
        forecast = stlf_res.forecast(int(right_delta / delta))
        overlapping_ind = [ind for ind in y.index if ind in forecast.index]
        return forecast[overlapping_ind]

    def _project_trend_and_seasonality(self, X, y):
        # Determine how many units forward to forecast
        projected_trend = self._project_trend(y)

        # Reseasonalize
        projected_seasonality = self._build_seasonal_signal(
            y,
            self.seasonality,
            self.seasonal_period,
            self.frequency,
        )
        return projected_trend, projected_seasonality

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
        self.is_fit = True
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        if y is None:
            return X, y
        if not isinstance(y.index, pd.DatetimeIndex):
            y = self._set_time_index(X, y)

        if not self.is_fit:
            raise ValueError(
                "STLDecomposer has not been fit yet.  Please fit it and then build the decomposed dataframe.",
            )

        self._check_oos_past(y)

        y_in_sample = pd.Series([])
        y_out_of_sample = pd.Series([])

        # For wholly in-sample data, retrieve stored results.
        if y.index[0] <= self.trend.index[-1] and y.index[0] >= self.trend.index[0]:
            y_in_sample = self.residual[y.index[0] : y.index[-1]]

        # For out of sample data....
        if y.index[-1] > self.trend.index[-1]:
            try:
                # ...that is partially out of sample and partially in sample.
                truncated_y = y[y.index.get_loc(self.trend.index[-1]) + 1 :]
            except KeyError:
                # ...that is entirely out of sample.
                truncated_y = y

            (
                projected_trend,
                projected_seasonality,
            ) = self._project_trend_and_seasonality(X, truncated_y)

            y_out_of_sample = infer_feature_types(
                pd.Series(
                    truncated_y - projected_trend - projected_seasonality,
                    index=truncated_y.index,
                ),
            )

        return X, y_in_sample.append(y_out_of_sample)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)

    # @fit_check
    def inverse_transform(self, y_t):
        if y_t is None:
            raise ValueError("y_t cannot be None for STLDecomposer!")
        if not self.is_fit:
            raise ValueError(
                "STLDecomposer has not been fit yet.  Please fit it and then build the decomposed dataframe.",
            )

        y_t = infer_feature_types(y_t)
        self._check_oos_past(y_t)

        y_in_sample = pd.Series([])
        y_out_of_sample = pd.Series([])

        # For wholly in-sample data, retrieve stored results.
        if y_t.index[0] <= self.trend.index[-1] and y_t.index[0] >= self.trend.index[0]:
            left_index = y_t.index[0]
            right_index = y_t.index[-1]
            y_in_sample = (
                y_t
                + self.trend[left_index:right_index]
                + self.seasonal[left_index:right_index]
            )
            y_in_sample = y_in_sample.dropna()

        # For out of sample data....
        if y_t.index[-1] > self.trend.index[-1]:
            try:
                # ...that is partially out of sample and partially in sample.
                truncated_y_t = y_t[y_t.index.get_loc(self.trend.index[-1]) + 1 :]
            except KeyError:
                # ...that is entirely out of sample.
                truncated_y_t = y_t
            (
                projected_trend,
                projected_seasonality,
            ) = self._project_trend_and_seasonality(None, truncated_y_t)

            y_out_of_sample = infer_feature_types(
                pd.Series(
                    truncated_y_t + projected_trend + projected_seasonality,
                    index=truncated_y_t.index,
                ),
            )
        return y_in_sample.append(y_out_of_sample)

    # @fit_check
    def get_trend_dataframe(self, X, y):
        """Return a list of dataframes with 4 columns: signal, trend, seasonality, residual.

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
        if not self.is_fit:
            raise ValueError(
                "STLDecomposer has not been fit yet.  Please fit it and then build the decomposed dataframe.",
            )
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

        self._check_oos_past(y)

        result_dfs = []

        def _decompose_target(X, y, fh):
            """Function to generate a single DataFrame with trend, seasonality and residual components."""
            if len(y.index) == len(self.trend.index) and all(
                y.index == self.trend.index,
            ):
                trend = self.trend
                seasonality = self.seasonality
                residual = self.residual
            else:
                # TODO: Do a better job cloning.
                decomposer = STLDecomposer(seasonal_period=self.seasonal_period)
                decomposer.fit(X, y)
                trend = decomposer.trend
                seasonality = decomposer.seasonality
                residual = decomposer.residual
            return pd.DataFrame(
                {
                    "signal": y,
                    "trend": trend,
                    "seasonality": seasonality,
                    "residual": residual,
                },
            )

        if isinstance(y, pd.Series):
            result_dfs.append(_decompose_target(X, y, None))
        elif isinstance(y, pd.DataFrame):
            for colname in y.columns:
                result_dfs.append(_decompose_target(X, y[colname], None))
        return result_dfs
