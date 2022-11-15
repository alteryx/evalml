"""Component that removes trends and seasonality from time series using STL."""
from __future__ import annotations

import logging

import pandas as pd
from pandas.core.index import Int64Index, RangeIndex
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.seasonal import STL

from evalml.pipelines.components.transformers.preprocessing.decomposer import Decomposer
from evalml.utils import infer_feature_types


class STLDecomposer(Decomposer):
    """Removes trends and seasonality from time series using the STL algorithm.

    https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.STL.html

    Args:
        time_index (str): Specifies the name of the column in X that provides the datetime objects. Defaults to None.
        degree (int): Not currently used.  STL 3x "degree-like" values.  None are able to be set at
            this time. Defaults to 1.
        seasonal_period (int): The number of entries in the time series data that corresponds to one period of a
            cyclic signal.  For instance, if data is known to possess a weekly seasonal signal, and if the data
            is daily data, seasonal_period should be 7.  For daily data with a yearly seasonal signal, seasonal_period
            should be 365.  For compatibility with the underlying STL algorithm, must be odd. If an even number
            is provided, the next, highest odd number will be used. Defaults to 7.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "STL Decomposer"
    hyperparameter_ranges = {}

    modifies_features = False
    modifies_target = True
    invalid_frequencies = [
        "SM",
        "BM",
        "SMS",
        "BMS",
        "BQ",
        "BQS",
        "T",
        "S",
        "L",
        "U",
        "N",
        "A",
        "BA",
        "AS",
        "BAS",
        "BH",
    ]

    def __init__(
        self,
        time_index: str = None,
        degree: int = 1,  # Currently unused.
        seasonal_period: int = 7,
        random_seed: int = 0,
        **kwargs,
    ):
        self.logger = logging.getLogger(__name__)

        # Programmatically adjust seasonal_period to fit underlying STL requirements,
        # that seasonal_period must be odd.
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

    def _project_trend(self, y):
        """Function to project the in-sample trend into the future."""
        self._check_oos_past(y)

        index = self._choose_proper_index(y)

        # Determine how many units forward to project by finding the difference,
        # in index values, between the requested target and the fit data.
        if isinstance(y.index, pd.DatetimeIndex):
            units_forward = (
                len(
                    pd.date_range(
                        start=self.trend.index[-1],
                        end=y.index[-1],
                        freq=self.frequency,
                    ),
                )
                - 1
            )
        elif isinstance(y.index, (RangeIndex, Int64Index)):
            units_forward = int(y.index[-1] - index[-1])

        # Model the trend and project it forward
        stlf = STLForecast(
            self.trend,
            ARIMA,
            model_kwargs=dict(order=(1, 1, 0), trend="t"),
        )
        stlf_res = stlf.fit()
        forecast = stlf_res.forecast(units_forward)

        # Handle out-of-sample forecasts.  The forecast will have additional data
        # between the end of the in-sample data and the beginning of the
        # requested out-of-sample data to inverse transform.
        overlapping_ind = [ind for ind in y.index if ind in forecast.index]
        if len(overlapping_ind) > 0:
            return forecast[overlapping_ind]
        # This branch handles the cross-validation cases where the indices are
        # integer indices and we know the forecast length will match the requested
        # transform data length.
        else:
            fore = forecast[-len(y) :]
            fore.index = y.index
            return fore

    def _project_trend_and_seasonality(self, y):
        """Function to project both trend and seasonality forward into the future."""
        projected_trend = self._project_trend(y)

        projected_seasonality = self._project_seasonal(
            y,
            self.seasonality,
            self.seasonal_period,
            self.frequency,
        )
        return projected_trend, projected_seasonality

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> STLDecomposer:
        """Fits the STLDecomposer and determine the seasonal signal.

        Instantiates a statsmodels STL decompose object with the component's stored
        parameters and fits it.  Since the statsmodels object does not fit the sklearn
        api, it is not saved during __init__() in _component_obj and will be re-instantiated
        each time fit is called.

        To emulate the sklearn API, when the STL decomposer is fit, the full seasonal
        component, a single period sample of the seasonal component, the full
        trend-cycle component and the residual are saved.

        y(t) = S(t) + T(t) + R(t)

        Args:
            X (pd.DataFrame, optional): Conditionally used to build datetime index.
            y (pd.Series): Target variable to detrend and deseasonalize.

        Returns:
            self

        Raises:
            ValueError: If y is None.
            ValueError: If target data doesn't have DatetimeIndex AND no Datetime features in features data
        """
        self.original_index = y.index if y is not None else None
        X, y = self._check_target(X, y)
        self._map_dt_to_integer(self.original_index, y.index)

        # Warn for poor decomposition use with higher periods
        if self.seasonal_period > 14:
            str_dict = {"D": "daily", "M": "monthly"}
            data_str = ""
            if y.index.freqstr in str_dict:
                data_str = str_dict[y.index.freqstr]
            self.logger.warning(
                f"STLDecomposer may perform poorly on {data_str} data with a high seasonal period ({self.seasonal_period}).",
            )

        # Save the frequency of the fitted series for checking against transform data.
        self.frequency = y.index.freqstr or pd.infer_freq(y.index)

        stl = STL(y, seasonal=self.seasonal_period)
        res = stl.fit()
        self.seasonal = res.seasonal
        self.seasonal_period = stl.period
        dist = len(y) % self.seasonal_period
        self.seasonality = (
            self.seasonal[-(dist + self.seasonal_period) : -dist]
            if dist > 0
            else self.seasonal[-self.seasonal_period :]
        )
        self.trend = res.trend
        self.residual = res.resid

        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Transforms the target data by removing the STL trend and seasonality.

        Uses an ARIMA model to project forward the addititve trend and removes it. Then, utilizes the first period's
        worth of seasonal data determined in the .fit() function to extrapolate the seasonal signal of the data to be
        transformed.  This seasonal signal is also assumed to be additive and is removed.

        Args:
            X (pd.DataFrame, optional): Conditionally used to build datetime index.
            y (pd.Series): Target variable to detrend and deseasonalize.

        Returns:
            tuple of pd.DataFrame, pd.Series: The input features are returned without modification. The target
                variable y is detrended and deseasonalized.

        Raises:
            ValueError: If target data doesn't have DatetimeIndex AND no Datetime features in features data
        """
        if y is None:
            return X, y
        original_index = y.index
        X, y = self._check_target(X, y)

        self._check_oos_past(y)

        y_in_sample = pd.Series([])
        y_out_of_sample = pd.Series([])

        # For partially and wholly in-sample data, retrieve stored results.
        if self.trend.index[0] <= y.index[0] <= self.trend.index[-1]:
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
            ) = self._project_trend_and_seasonality(truncated_y)

            y_out_of_sample = infer_feature_types(
                pd.Series(
                    truncated_y - projected_trend - projected_seasonality,
                    index=truncated_y.index,
                ),
            )
        y_t = y_in_sample.append(y_out_of_sample)
        y_t.index = original_index
        return X, y_t

    def inverse_transform(self, y_t: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Adds back fitted trend and seasonality to target variable.

        The STL trend is projected to cover the entire requested target range, then added back into the signal. Then,
        the seasonality is projected forward to and added back into the signal.

        Args:
            y_t (pd.Series): Target variable.

        Returns:
            tuple of pd.DataFrame, pd.Series: The first element are the input features returned without modification.
                The second element is the target variable y with the trend and seasonality added back in.

        Raises:
            ValueError: If y is None.
        """
        if y_t is None:
            raise ValueError("y_t cannot be None for Decomposer!")
        original_index = y_t.index

        y_t = infer_feature_types(y_t)
        self._check_oos_past(y_t)

        index = self._choose_proper_index(y_t)

        y_in_sample = pd.Series([])
        y_out_of_sample = pd.Series([])

        # For partially and wholly in-sample data, retrieve stored results.
        if index[0] <= y_t.index[0] <= index[-1]:
            left_index = y_t.index[0]
            right_index = (
                y_t.index[-1] + 1
                if isinstance(y_t.index, (Int64Index, pd.RangeIndex))
                else y_t.index[-1] + 1 * y_t.index.freq
            )
            trend = (
                self.trend.reset_index(drop=True)[left_index:right_index]
                if isinstance(y_t.index, (Int64Index, pd.RangeIndex))
                else self.trend[left_index:right_index]
            )
            seasonal = (
                self.seasonal.reset_index(drop=True)[left_index:right_index]
                if isinstance(y_t.index, (Int64Index, pd.RangeIndex))
                else self.seasonal[left_index:right_index]
            )
            y_in_sample = y_t + trend + seasonal
            y_in_sample = y_in_sample.dropna()

        # For out of sample data....
        if y_t.index[-1] > index[-1]:
            try:
                # ...that is partially out of sample and partially in sample.
                truncated_y_t = y_t[y_t.index.get_loc(index[-1]) + 1 :]
            except KeyError:
                # ...that is entirely out of sample.
                truncated_y_t = y_t
            (
                projected_trend,
                projected_seasonality,
            ) = self._project_trend_and_seasonality(truncated_y_t)

            y_out_of_sample = infer_feature_types(
                pd.Series(
                    truncated_y_t + projected_trend + projected_seasonality,
                    index=truncated_y_t.index,
                ),
            )
        y = y_in_sample.append(y_out_of_sample)
        y.index = original_index
        return y

    def get_trend_dataframe(self, X, y):
        """Return a list of dataframes with 4 columns: signal, trend, seasonality, residual.

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

        self._check_oos_past(y)

        result_dfs = []

        def _decompose_target(X, y, fh):
            """Function to generate a single DataFrame with trend, seasonality and residual components."""
            if len(y.index) == len(self.trend.index) and all(
                y.index == self.trend.index,
            ):
                trend = self.trend
                seasonal = self.seasonal
                residual = self.residual
            else:
                # TODO: Do a better job cloning.
                decomposer = STLDecomposer(seasonal_period=self.seasonal_period)
                decomposer.fit(X, y)
                trend = decomposer.trend
                seasonal = decomposer.seasonal
                residual = decomposer.residual
            return pd.DataFrame(
                {
                    "signal": y,
                    "trend": trend,
                    "seasonality": seasonal,
                    "residual": residual,
                },
            )

        if isinstance(y, pd.Series):
            result_dfs.append(_decompose_target(X, y, None))
        elif isinstance(y, pd.DataFrame):
            for colname in y.columns:
                result_dfs.append(_decompose_target(X, y[colname], None))
        return result_dfs
