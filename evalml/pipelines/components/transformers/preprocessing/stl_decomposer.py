"""Component that removes trends and seasonality from time series using STL."""
from __future__ import annotations

import logging
from typing import Union

import pandas as pd
from pandas import RangeIndex
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
        series_id (str): Specifies the name of the column in X that provides the series_id objects for multiseries. Defaults to None.
        degree (int): Not currently used.  STL 3x "degree-like" values.  None are able to be set at
            this time. Defaults to 1.
        period (int): The number of entries in the time series data that corresponds to one period of a
            cyclic signal.  For instance, if data is known to possess a weekly seasonal signal, and if the data
            is daily data, the period should likely be 7.  For daily data with a yearly seasonal signal, the period
            should likely be 365. If None, statsmodels will infer the period based on the frequency. Defaults to None.
        seasonal_smoother (int): The length of the seasonal smoother used by the underlying STL algorithm. For compatibility,
            must be odd. If an even number is provided, the next, highest odd number will be used. Defaults to 7.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "STL Decomposer"
    hyperparameter_ranges = {}

    modifies_features = False
    modifies_target = True

    def __init__(
        self,
        time_index: str = None,
        series_id: str = None,
        degree: int = 1,  # Currently unused.
        period: int = None,
        periods: dict = None,
        seasonal_smoother: int = 7,
        random_seed: int = 0,
        **kwargs,
    ):
        self.logger = logging.getLogger(__name__)
        self.series_id = series_id
        self.periods = periods
        # Programmatically adjust seasonal_smoother to fit underlying STL requirements,
        # that seasonal_smoother must be odd.
        if seasonal_smoother % 2 == 0:
            self.logger.warning(
                f"STLDecomposer provided with an even period of {seasonal_smoother}"
                f"Changing seasonal period to {seasonal_smoother+1}",
            )
            seasonal_smoother += 1

        self.forecast_summary = None
        parameters = {
            "degree": degree,
            "period": period,
            "periods": periods,
            "seasonal_smoother": seasonal_smoother,
            "time_index": time_index,
            "series_id": series_id,
        }
        parameters.update(kwargs)

        super().__init__(
            component_obj=None,
            random_seed=random_seed,
            **parameters,
            **kwargs,
        )

    def _project_trend(self, y, trend, period):
        """Function to project the in-sample trend into the future."""
        self._check_oos_past(y)

        index = self._choose_proper_index(y)

        # Determine how many units forward to project by finding the difference,
        # in index values, between the requested target and the fit data.
        if isinstance(y.index, pd.DatetimeIndex):
            units_forward = (
                len(
                    pd.date_range(
                        start=trend.index[-1],
                        end=y.index[-1],
                        freq=self.frequency,
                    ),
                )
                - 1
            )
        elif isinstance(y.index, RangeIndex) or y.index.is_numeric():
            units_forward = int(y.index[-1] - index[-1])

        # Model the trend and project it forward
        stlf = STLForecast(
            trend,
            ARIMA,
            model_kwargs=dict(order=(1, 1, 0), trend="t"),
            period=period,
        )
        stlf = stlf.fit()
        forecast = stlf.forecast(units_forward)

        # Store forecast summary for use in calculating trend prediction intervals.
        self.forecast_summary = stlf.get_prediction(
            len(trend),
            len(trend) + units_forward - 1,
        )

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

    def _project_trend_and_seasonality(self, y, trend, seasonality, periodicity):
        """Function to project both trend and seasonality forward into the future."""
        projected_trend = self._project_trend(y, trend, periodicity)

        projected_seasonality = self._project_seasonal(
            y,
            seasonality,
            periodicity,
            self.frequency,
        )
        return projected_trend, projected_seasonality

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame] = None,
    ) -> STLDecomposer:
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
            y (pd.Series or pd.DataFrame): Target variable to detrend and deseasonalize.

        Returns:
            self

        Raises:
            ValueError: If y is None.
            ValueError: If target data doesn't have DatetimeIndex AND no Datetime features in features data
        """
        from evalml.pipelines.utils import unstack_multiseries

        # Warn for poor decomposition use with higher seasonal smoothers
        if self.seasonal_smoother > 14:
            self.logger.warning(
                f"STLDecomposer may perform poorly on data with a high seasonal smoother ({self.seasonal_smoother}).",
            )

        # If y is a stacked pd.Series, unstack it
        if self.series_id is not None and isinstance(y, pd.Series):
            X, y = unstack_multiseries(X, y, self.series_id, self.time_index, y.name)

        if isinstance(y, pd.Series):
            y = y.to_frame()

        self.original_index = y.index if y is not None else None

        X, y = self._check_target(X, y)

        self._map_dt_to_integer(self.original_index, y.index)

        # Save the frequency of the fitted series for checking against transform data.
        self.frequency = y.index.freqstr or pd.infer_freq(y.index)
        # Iterate through each id group
        self.seasonals = {}
        self.seasonalities = {}
        self.trends = {}
        self.residuals = {}
        if self.periods is None:
            self.periods = {}

        for id in y.columns:
            series_y = y[id]

            # Determine the period of the seasonal component
            if id not in self.periods:
                # If the user provides a period for single series, use that
                period = (
                    self.period
                    if len(y.columns) == 1 and self.period is not None
                    else self.determine_periodicity(X, series_y)
                )
                self.periods[id] = period

            stl = STL(
                series_y,
                seasonal=self.seasonal_smoother,
                period=self.periods[id],
            )
            res = stl.fit()
            self.seasonals[id] = res.seasonal
            self.periods[id] = stl.period
            dist = len(series_y) % stl.period
            seasonality = (
                res.seasonal[-(dist + stl.period) : -dist]
                if dist > 0
                else res.seasonal[-stl.period :]
            )
            self.seasonalities[id] = seasonality
            self.trends[id] = res.trend
            self.residuals[id] = res.resid
        self.update_parameters({"periods": self.periods})

        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame] = None,
    ) -> Union[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.DataFrame]]:
        """Transforms the target data by removing the STL trend and seasonality.

        Uses an ARIMA model to project forward the addititve trend and removes it. Then, utilizes the first period's
        worth of seasonal data determined in the .fit() function to extrapolate the seasonal signal of the data to be
        transformed.  This seasonal signal is also assumed to be additive and is removed.

        Args:
            X (pd.DataFrame, optional): Conditionally used to build datetime index.
            y (pd.Series or pd.DataFrame): Target variable to detrend and deseasonalize.

        Returns:
            (Single series) pd.DataFrame, pd.Series: The list of input features are returned without modification. The target
                variable y is detrended and deseasonalized.
            (Multi series) pd.DataFrame, pd.DataFrame: The list of input features are returned without modification. The target
                variable y is detrended and deseasonalized.

        Raises:
            ValueError: If target data doesn't have DatetimeIndex AND no Datetime features in features data
        """
        from evalml.pipelines.utils import unstack_multiseries

        if y is None:
            return X, y

        # If y is a stacked pd.Series, unstack it
        if self.series_id is not None and isinstance(y, pd.Series):
            X, y = unstack_multiseries(X, y, self.series_id, self.time_index, y.name)

        if isinstance(y, pd.Series):
            y = y.to_frame()

        original_index = y.index
        X, y = self._check_target(X, y)
        self._check_oos_past(y)

        detrending_list = []
        # Iterate through each id group
        for id in y.columns:
            series_y = y[id]

            if len(y.columns) > 1:
                seasonality = self.seasonalities[id]
                trend = self.trends[id]
                residual = self.residuals[id]
                period = self.periods[id]
            else:
                seasonality = list(self.seasonalities.values())[0]
                trend = list(self.trends.values())[0]
                residual = list(self.residuals.values())[0]
                period = list(self.periods.values())[0]

            y_in_sample = pd.Series([])
            y_out_of_sample = pd.Series([])

            # For partially and wholly in-sample data, retrieve stored results.
            if trend.index[0] <= series_y.index[0] <= trend.index[-1]:
                y_in_sample = residual[series_y.index[0] : series_y.index[-1]]

            # For out of sample data....
            if series_y.index[-1] > trend.index[-1]:
                try:
                    # ...that is partially out of sample and partially in sample.
                    truncated_y = series_y[
                        series_y.index.get_loc(trend.index[-1]) + 1 :
                    ]
                except KeyError:
                    # ...that is entirely out of sample.
                    truncated_y = series_y

                (
                    projected_trend,
                    projected_seasonality,
                ) = self._project_trend_and_seasonality(
                    truncated_y,
                    trend,
                    seasonality,
                    period,
                )

                y_out_of_sample = infer_feature_types(
                    pd.Series(
                        truncated_y - projected_trend - projected_seasonality,
                        index=truncated_y.index,
                    ),
                )
            y_t = pd.concat([y_in_sample, y_out_of_sample])
            y_t.index = original_index

            # If it is a single series time series, return tuple[pd.DataFrame, pd.Series]
            if len(y.columns) <= 1:
                return X, y_t

            detrending_list.append(y_t)

        # Convert the list to a DataFrame
        # For multiseries, return tuple[pd.DataFrame, pd.Dataframe] where each column is a series_id
        detrending_df = pd.DataFrame(detrending_list).T
        return X, detrending_df

    def inverse_transform(
        self,
        y_t: Union[pd.Series, pd.DataFrame],
    ) -> Union[pd.Series, pd.DataFrame]:
        """Adds back fitted trend and seasonality to target variable.

        The STL trend is projected to cover the entire requested target range, then added back into the signal. Then,
        the seasonality is projected forward to and added back into the signal.

        Args:
            y_t (pd.Series or pd.DataFrame): Target variable.

        Returns:
            pd.Series or pd.DataFrame: The target variable y with the trend and seasonality added back in.

        Raises:
            ValueError: If y is None.
        """
        if y_t is None:
            raise ValueError("y_t cannot be None for Decomposer!")
        original_index = y_t.index

        y_t = infer_feature_types(y_t)
        self._check_oos_past(y_t)

        if isinstance(y_t, pd.Series):
            y_t = y_t.to_frame()

        index = self._choose_proper_index(y_t)
        y = []
        for id in y_t.columns:
            y_in_sample = pd.Series([])
            y_out_of_sample = pd.Series([])
            series_y = y_t[id]

            if len(y_t.columns) > 1:
                old_trend = self.trends[id]
                old_seasonal = self.seasonals[id]
                old_seasonality = self.seasonalities[id]
                period = self.periods[id]
            else:
                old_trend = list(self.trends.values())[0]
                old_seasonal = list(self.seasonals.values())[0]
                old_seasonality = list(self.seasonalities.values())[0]
                period = list(self.periods.values())[0]
            # For partially and wholly in-sample data, retrieve stored results.
            if index[0] <= series_y.index[0] <= index[-1]:
                left_index = series_y.index[0]
                right_index = (
                    series_y.index[-1] + 1
                    if isinstance(series_y.index, pd.RangeIndex)
                    or series_y.index.is_numeric()
                    else series_y.index[-1] + 1 * series_y.index.freq
                )
                trend = (
                    old_trend.reset_index(drop=True)[left_index:right_index]
                    if isinstance(series_y.index, pd.RangeIndex)
                    or series_y.index.is_numeric()
                    else old_trend[left_index:right_index]
                )
                seasonal = (
                    old_seasonal.reset_index(drop=True)[left_index:right_index]
                    if isinstance(series_y.index, pd.RangeIndex)
                    or series_y.index.is_numeric()
                    else old_seasonal[left_index:right_index]
                )
                y_in_sample = series_y + trend + seasonal
                y_in_sample = y_in_sample.dropna()

            # For out of sample data....
            if series_y.index[-1] > index[-1]:
                try:
                    # ...that is partially out of sample and partially in sample.
                    truncated_y_t = series_y[series_y.index.get_loc(index[-1]) + 1 :]
                except KeyError:
                    # ...that is entirely out of sample.
                    truncated_y_t = series_y
                (
                    projected_trend,
                    projected_seasonality,
                ) = self._project_trend_and_seasonality(
                    truncated_y_t,
                    old_trend,
                    old_seasonality,
                    period,
                )

                y_out_of_sample = infer_feature_types(
                    pd.Series(
                        truncated_y_t + projected_trend + projected_seasonality,
                        index=truncated_y_t.index,
                    ),
                )
            y_series = pd.concat([y_in_sample, y_out_of_sample])
            # If it is a single series time series, return tuple[pd.DataFrame, pd.Series]
            if len(y_t.columns) <= 1:
                y_series.index = original_index
                return y_series

            y.append(y_series)
        y_df = pd.DataFrame(y).T
        y_df.index = original_index
        y_df.columns = y_t.columns
        return y_df

    def get_trend_dataframe(self, X, y):
        """Return a list of dataframes with 4 columns: signal, trend, seasonality, residual.

        Args:
            X (pd.DataFrame): Input data with time series data in index.
            y (pd.Series or pd.DataFrame): Target variable data provided as a Series for univariate problems or
                a DataFrame for multivariate problems.

        Returns:
            (Single series) list of pd.DataFrame: Each DataFrame contains the columns "signal", "trend", "seasonality" and "residual,"
                with the latter 3 column values being the decomposed elements of the target data.  The "signal" column
                is simply the input target signal but reindexed with a datetime index to match the input features.
            (Multi series) dictionary of lists: Series id maps to a list of pd.DataFrames that each contain the columns "signal", "trend", "seasonality" and "residual"

        Raises:
            TypeError: If X does not have time-series data in the index.
            ValueError: If time series index of X does not have an inferred frequency.
            ValueError: If the forecaster associated with the detrender has not been fit yet.
            TypeError: If y is not provided as a pandas Series or DataFrame.

        """
        X = infer_feature_types(X)
        if not isinstance(X.index, pd.DatetimeIndex) and not isinstance(
            y.index,
            pd.DatetimeIndex,
        ):
            raise TypeError("Provided X or y should have datetimes in the index.")
        # Change the y index to a matching datetimeindex or else we get a failure
        # in ForecastingHorizon during decomposition.
        if not isinstance(y.index, pd.DatetimeIndex):
            y = self._set_time_index(X, y)
        if not isinstance(X.index, pd.DatetimeIndex):
            X.index = y.index
        self._check_oos_past(y)

        def _decompose_target(X, y, fh):
            """Function to generate a single DataFrame with trend, seasonality and residual components."""
            if isinstance(y, pd.Series):
                y = y.to_frame()
            if all(
                len(y.index) == len(self.trends[id].index)
                and all(
                    y.index == self.trends[id].index,
                )
                for id in y.columns
            ):
                # TODO: Do a better job cloning.
                decomposer = STLDecomposer(
                    seasonal_smoother=self.seasonal_smoother,
                    periods=self.periods,
                )
                decomposer.fit(X, y)
                trend = decomposer.trends
                seasonal = decomposer.seasonals
                residual = decomposer.residuals
            else:
                trend = self.trends
                seasonal = self.seasonals
                residual = self.residuals
            result_dict = {}
            for id in y.columns:
                df = pd.DataFrame(
                    {
                        "signal": y[id],
                        "trend": trend[id],
                        "seasonality": seasonal[id],
                        "residual": residual[id],
                    },
                )
                if len(y.columns) == 1:
                    return [df]
                else:
                    result_dict[id] = [df]
            return result_dict

        return _decompose_target(X, y, None)

    def get_trend_prediction_intervals(self, y, coverage=None):
        """Calculate the prediction intervals for the trend data.

        Args:
            y (pd.Series or pd.DataFrame): Target data.
            coverage (list[float]): A list of floats between the values 0 and 1 that the upper and lower bounds of the
                prediction interval should be calculated for.

        Returns:
            (Single series) dict of pd.Series: Prediction intervals, keys are in the format {coverage}_lower or {coverage}_upper.
            (Multi series) dict of dict of pd.Series: Each series id maps to a dictionary of prediction intervals
        """
        if isinstance(y, pd.Series):
            y = y.to_frame()

        if coverage is None:
            coverage = [0.95]

        self._check_oos_past(y)
        series_results = {}
        for id in y.columns:
            y_series = y[id]

            alphas = [1 - val for val in coverage]

            if len(y.columns) > 1:
                trend = self.trends[id]
                seasonality = self.seasonalities[id]
                period = self.periods[id]
            else:
                trend = list(self.trends.values())[0]
                seasonality = list(self.seasonalities.values())[0]
                period = list(self.periods.values())[0]
            if not self.forecast_summary or len(y_series) != len(
                self.forecast_summary.predicted_mean,
            ):
                self._project_trend_and_seasonality(
                    y_series,
                    trend,
                    seasonality,
                    period,
                )

            prediction_interval_result = {}
            for i, alpha in enumerate(alphas):
                result = self.forecast_summary.summary_frame(alpha=alpha)
                overlapping_ind = [ind for ind in y_series.index if ind in result.index]
                intervals = pd.DataFrame(
                    {
                        "lower": result["mean_ci_lower"] - result["mean"],
                        "upper": result["mean_ci_upper"] - result["mean"],
                    },
                )
                if len(overlapping_ind) > 0:  # y.index is datetime
                    intervals = intervals.loc[overlapping_ind]
                else:  # y.index is not datetime (e.g. int)
                    intervals = intervals[-len(y_series) :]
                    intervals.index = y_series.index
                prediction_interval_result[f"{coverage[i]}_lower"] = intervals["lower"]
                prediction_interval_result[f"{coverage[i]}_upper"] = intervals["upper"]
            series_results[id] = prediction_interval_result

            # only return the dictionary if single series
            if len(y.columns) <= 1:
                return prediction_interval_result
        return series_results
