"""Component that removes trends and seasonality from time series using STL."""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
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
        seasonal_smoother: int = 7,
        random_seed: int = 0,
        is_multiseries: bool = False,
        **kwargs,
    ):
        self.logger = logging.getLogger(__name__)
        self.series_id = series_id
        self.is_multiseries = is_multiseries
        # Programmatically adjust seasonal_smoother to fit underlying STL requirements,
        # that seasonal_smoother must be odd.
        if seasonal_smoother % 2 == 0:
            self.logger.warning(
                f"STLDecomposer provided with an even period of {seasonal_smoother}"
                f"Changing seasonal period to {seasonal_smoother+1}",
            )
            seasonal_smoother += 1

        self.forecast_summary = None
        super().__init__(
            component_obj=None,
            random_seed=random_seed,
            degree=degree,
            period=period,
            seasonal_smoother=seasonal_smoother,
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
        elif isinstance(y.index, RangeIndex) or y.index.is_numeric():
            units_forward = int(y.index[-1] - index[-1])

        # Model the trend and project it forward
        stlf = STLForecast(
            self.trend,
            ARIMA,
            model_kwargs=dict(order=(1, 1, 0), trend="t"),
            period=self.period,
        )
        stlf = stlf.fit()
        forecast = stlf.forecast(units_forward)

        # Store forecast summary for use in calculating trend prediction intervals.
        self.forecast_summary = stlf.get_prediction(
            len(self.trend),
            len(self.trend) + units_forward - 1,
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

    def _project_trend_and_seasonality(self, y):
        """Function to project both trend and seasonality forward into the future."""
        projected_trend = self._project_trend(y)

        projected_seasonality = self._project_seasonal(
            y,
            self.seasonality,
            self.period,
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
        # Warn for poor decomposition use with higher seasonal smoothers
        if self.seasonal_smoother > 14:
            self.logger.warning(
                f"STLDecomposer may perform poorly on data with a high seasonal smoother ({self.seasonal_smoother}).",
            )

        # If there is not a series_id, give them one series id with the value 0
        if self.series_id is None:
            self.series_id = "series_id"
            X[self.series_id] = 0
        else:
            self.is_multiseries = True

        # Initialize the new "series_id" column in Woodwork
        X.ww.init()

        # Group the data by series_id
        grouped_X = X.groupby(self.series_id)
        # Iterate through each id group
        self.decompositions = {}
        for id, series_X in grouped_X:

            if y is None:
                series_y = None
            elif isinstance(series_X.index, pd.DatetimeIndex):
                series_y = y[(series_X.reset_index(drop=True).index)]
            else:
                series_y = y[series_X.index]
            self.original_index = series_y.index if series_y is not None else None

            series_X, series_y = self._check_target(series_X, series_y)

            self._map_dt_to_integer(self.original_index, series_y.index)

            # Save the frequency of the fitted series for checking against transform data.
            self.frequency = series_y.index.freqstr or pd.infer_freq(series_y.index)
            # Determine the period of the seasonal component
            self.set_period(series_X, series_y)

            stl = STL(series_y, seasonal=self.seasonal_smoother, period=self.period)
            res = stl.fit()
            self.seasonal = res.seasonal
            self.period = stl.period
            dist = len(series_y) % self.period
            self.seasonality = (
                self.seasonal[-(dist + self.period) : -dist]
                if dist > 0
                else self.seasonal[-self.period :]
            )

            self.trend = res.trend
            self.residual = res.resid

            if self.is_multiseries:
                self.decompositions[id] = {
                    "seasonal": self.seasonal,
                    "seasonality": self.seasonality,
                    "trend": self.trend,
                    "residual": self.resid,
                    "period": self.period,
                }

        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
    ):
        """Transforms the target data by removing the STL trend and seasonality.

        Uses an ARIMA model to project forward the addititve trend and removes it. Then, utilizes the first period's
        worth of seasonal data determined in the .fit() function to extrapolate the seasonal signal of the data to be
        transformed.  This seasonal signal is also assumed to be additive and is removed.

        Args:
            X (pd.DataFrame, optional): Conditionally used to build datetime index.
            y (pd.Series): Target variable to detrend and deseasonalize.

        Returns:
            (Single series) pd.DataFrame, pd.Series: The list of input features are returned without modification. The target
                variable y is detrended and deseasonalized.
            (Multi series) pd.DataFrame, pd.DataFrame: The list of input features are returned without modification. The target
                variable y is detrended and deseasonalized.

        Raises:
            ValueError: If target data doesn't have DatetimeIndex AND no Datetime features in features data
        """
        if y is None:
            return X, y

        if not self.is_multiseries and X is not None:
            self.series_id = "series_id"
            X[self.series_id] = 0
        # If X is None, create a series with id=0 and series_X=None
        grouped_X = {0: X}.items() if X is None else X.groupby(self.series_id)

        features_list = []
        detrending_list = []
        for id, series_X in grouped_X:
            if self.is_multiseries:
                self.seasonality = self.decompositions[id]["seasonality"]
                self.trend = self.decompositions[id]["trend"]
                self.seasonal = self.decompositions[id]["seasonal"]
                self.residual = self.decompositions[id]["residual"]
                self.period = self.decompositions[id]["period"]
                if isinstance(series_X.index, pd.DatetimeIndex):
                    series_y = y[(series_X.reset_index(drop=True).index)]
                else:
                    series_y = y[series_X.index]
            else:
                series_y = y

            if series_y is None:
                return series_X, series_y
            original_index = series_y.index
            series_X, series_y = self._check_target(series_X, series_y)
            self._check_oos_past(series_y)

            y_in_sample = pd.Series([])
            y_out_of_sample = pd.Series([])

            # For partially and wholly in-sample data, retrieve stored results.
            if self.trend.index[0] <= series_y.index[0] <= self.trend.index[-1]:
                y_in_sample = self.residual[series_y.index[0] : series_y.index[-1]]

            # For out of sample data....
            if series_y.index[-1] > self.trend.index[-1]:
                try:
                    # ...that is partially out of sample and partially in sample.
                    truncated_y = series_y[
                        series_y.index.get_loc(self.trend.index[-1]) + 1 :
                    ]
                except KeyError:
                    # ...that is entirely out of sample.
                    truncated_y = series_y

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

            # If it is a single series time series, return tuple[pd.DataFrame, pd.Series]
            if not self.is_multiseries:
                return series_X, y_t

            features_list.append({id: series_X})
            detrending_list.append({id: y_t})

        # Convert the list to a DataFrame
        # For multiseries, return tuple[pd.DataFrame, pd.Dataframe] where each column is a series_id
        features_df = pd.DataFrame(features_list)
        detrending_df = pd.DataFrame(detrending_list)
        return features_df, detrending_df

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
                if isinstance(y_t.index, pd.RangeIndex) or y_t.index.is_numeric()
                else y_t.index[-1] + 1 * y_t.index.freq
            )
            trend = (
                self.trend.reset_index(drop=True)[left_index:right_index]
                if isinstance(y_t.index, pd.RangeIndex) or y_t.index.is_numeric()
                else self.trend[left_index:right_index]
            )
            seasonal = (
                self.seasonal.reset_index(
                    drop=True,
                )[left_index:right_index]
                if isinstance(y_t.index, pd.RangeIndex) or y_t.index.is_numeric()
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
                decomposer = STLDecomposer(
                    seasonal_smoother=self.seasonal_smoother,
                    period=self.period,
                )
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

    def get_trend_prediction_intervals(self, y, coverage=None):
        """Calculate the prediction intervals for the trend data.

        Args:
            y (pd.Series): Target data.
            coverage (list[float]): A list of floats between the values 0 and 1 that the upper and lower bounds of the
                prediction interval should be calculated for.

        Returns:
            dict of pd.Series: Prediction intervals, keys are in the format {coverage}_lower or {coverage}_upper.
        """
        if coverage is None:
            coverage = [0.95]

        self._check_oos_past(y)
        alphas = [1 - val for val in coverage]

        if not self.forecast_summary or len(y) != len(
            self.forecast_summary.predicted_mean,
        ):
            self._project_trend_and_seasonality(y)

        prediction_interval_result = {}
        for i, alpha in enumerate(alphas):
            result = self.forecast_summary.summary_frame(alpha=alpha)
            overlapping_ind = [ind for ind in y.index if ind in result.index]
            intervals = pd.DataFrame(
                {
                    "lower": result["mean_ci_lower"] - result["mean"],
                    "upper": result["mean_ci_upper"] - result["mean"],
                },
            )
            if len(overlapping_ind) > 0:  # y.index is datetime
                intervals = intervals.loc[overlapping_ind]
            else:  # y.index is not datetime (e.g. int)
                intervals = intervals[-len(y) :]
                intervals.index = y.index
            prediction_interval_result[f"{coverage[i]}_lower"] = intervals["lower"]
            prediction_interval_result[f"{coverage[i]}_upper"] = intervals["upper"]

        return prediction_interval_result

    # Overload the plot_decomposition fucntion to be able to plot multiple decompositions for multiseries
    def plot_decomposition(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        show: bool = False,
    ):
        """Plots the decomposition of the target signal.

        Args:
            X (pd.DataFrame): Input data with time series data in index.
            y (pd.Series or pd.DataFrame): Target variable data provided as a Series for univariate problems or
                a DataFrame for multivariate problems.
            show (bool): Whether to display the plot or not. Defaults to False.

        Returns:
            (Single series) matplotlib.pyplot.Figure, list[matplotlib.pyplot.Axes]: The figure and axes that have the decompositions
                plotted on them
            (Multi series) dict[matplotlib.pyplot.Figure, list[matplotlib.pyplot.Axes]]: A dictionary that maps the series id to
                the figure and axes that have the decompositions plotted on them


        """
        # Group the data by series_id
        grouped_X = X.groupby(self.series_id)

        # Iterate through each series id
        plot_info = {}
        for id, series_X in grouped_X:
            if self.is_multiseries:
                self.seasonality = self.decompositions[id]["seasonality"]
                self.seasonal = self.decompositions[id]["seasonal"]
                self.trend = self.decompositions[id]["trend"]
                self.residual = self.decompositions[id]["residual"]
                self.period = self.decompositions[id]["period"]

            if isinstance(series_X.index, pd.DatetimeIndex):
                series_y = y[(series_X.reset_index(drop=True).index)]
            else:
                series_y = y[series_X.index]

            if self.is_multiseries:
                series_X.index = pd.DatetimeIndex(
                    series_X[self.time_index],
                    freq=self.frequency,
                )

            decomposition_results = self.get_trend_dataframe(series_X, series_y)

            fig, axs = plt.subplots(4)
            fig.set_size_inches(18.5, 14.5)
            axs[0].plot(decomposition_results[0]["signal"], "r")
            axs[0].set_title("signal")
            axs[1].plot(decomposition_results[0]["trend"], "b")
            axs[1].set_title("trend")
            axs[2].plot(decomposition_results[0]["seasonality"], "g")
            axs[2].set_title("seasonality")
            axs[3].plot(decomposition_results[0]["residual"], "y")
            axs[3].set_title("residual")

            if self.is_multiseries:
                fig.suptitle("Decomposition for Series {}".format(id))
                plot_info[id] = (fig, axs)
            else:
                plot_info = (fig, axs)

            if show:  # pragma: no cover
                plt.show()

        return plot_info
