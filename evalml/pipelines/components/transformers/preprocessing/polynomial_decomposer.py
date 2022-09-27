"""Component that removes trends from time series by fitting a polynomial to the data."""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.exceptions
from skopt.space import Integer
from sktime.forecasting.base._fh import ForecastingHorizon
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.tsatools import freq_to_period

from evalml.pipelines.components.transformers.preprocessing import Decomposer
from evalml.utils import import_or_raise, infer_feature_types


class PolynomialDecomposer(Decomposer):
    """Removes trends and seasonality from time series by fitting a polynomial and moving average to the data.

    Scikit-learn's PolynomialForecaster is used to generate the additive trend portion of the target data. A polynomial
        will be fit to the data during fit.  That additive polynomial trend will be removed during fit so that statsmodel's
        seasonal_decompose can determine the addititve seasonality of the data by using rolling averages over the series'
        inferred periodicity.

        For example, daily time series data will generate rolling averages over the first week of data, normalize
        out the mean and return those 7 averages repeated over the entire length of the given series.  Those seven
        averages, repeated as many times as necessary to match the length of the given target data, will be used
        as the seasonal signal of the data.

    Args:
        time_index (str): Specifies the name of the column in X that provides the datetime objects. Defaults to None.
        degree (int): Degree for the polynomial. If 1, linear model is fit to the data.
            If 2, quadratic model is fit, etc. Defaults to 1.
        seasonal_period (int): The number of entries in the time series data that corresponds to one period of a
            cyclic signal.  For instance, if data is known to possess a weekly seasonal signal, and if the data
            is daily data, seasonal_period should be 7.  For daily data with a yearly seasonal signal, seasonal_period
            should be 365.  Defaults to -1, which uses the statsmodels libarary's freq_to_period function.
            https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/tsatools.py
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Polynomial Decomposer"
    hyperparameter_ranges = {"degree": Integer(1, 3)}
    """{
        "degree": Integer(1, 3)
    }"""
    modifies_features = False
    modifies_target = True

    def __init__(
        self,
        time_index: str = None,
        degree: int = 1,
        seasonal_period: int = -1,
        random_seed: int = 0,
        **kwargs,
    ):
        def raise_typeerror_if_not_int(var_name, var_value):
            if not isinstance(var_value, int):
                if isinstance(var_value, float) and var_value.is_integer():
                    var_value = int(var_value)
                else:
                    raise TypeError(
                        f"Parameter 'degree' must be an integer!: Received {type(degree).__name__}",
                    )
            return var_value

        self.logger = logging.getLogger(__name__)
        degree = raise_typeerror_if_not_int("degree", degree)
        self.seasonal_period = raise_typeerror_if_not_int(
            "seasonal_period",
            seasonal_period,
        )

        params = {"degree": degree, "seasonal_period": self.seasonal_period}
        params.update(kwargs)
        error_msg = "sktime is not installed. Please install using 'pip install sktime'"

        trend = import_or_raise("sktime.forecasting.trend", error_msg=error_msg)
        detrend = import_or_raise(
            "sktime.transformations.series.detrend",
            error_msg=error_msg,
        )

        decomposer = detrend.Detrender(trend.PolynomialTrendForecaster(degree=degree))

        self.time_index = time_index
        params["time_index"] = time_index

        super().__init__(
            parameters=params,
            component_obj=decomposer,
            random_seed=random_seed,
        )

    def _build_seasonal_signal(self, y, periodic_signal, periodicity, frequency):
        """Projects the cyclical, seasonal signal forward to cover the target data.

        Args:
            y (pandas.Series): Target data to be transformed
            periodic_signal (pandas.Series): Single period of the detected seasonal signal
            periodicity (int): Number of time units in a single cycle of the seasonal signal
            frequency (str): String representing the detected frequency of the time series data.
                Uses the same codes as the freqstr attribute of a pandas Series with DatetimeIndex.
                e.g. "D", "M", "Y" for day, month and year respectively
                See: https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
                See: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_timedelta.html

        Returns:
            pandas.Series: the seasonal signal extended to cover the target data to be transformed
        """
        # Determine where the seasonality starts
        first_index_diff = y.index[0] - periodic_signal.index[0]
        delta = pd.to_timedelta(1, frequency)
        period = pd.to_timedelta(periodicity, frequency)

        # Determine which index of the sample of seasonal data the transformed data starts at
        transform_first_ind = int((first_index_diff % period) / delta)

        # Cycle the sample of seasonal data so the transformed data's effective index is first
        rotated_seasonal_sample = np.roll(
            periodic_signal.T.values,
            -transform_first_ind,
        )

        # Repeat the single, rotated period of seasonal data to cover the entirety of the data
        # to be transformed.
        seasonal = np.tile(rotated_seasonal_sample, len(y) // periodicity + 1).T[
            : len(y)
        ]  # The extrapolated seasonal data will be too long, so truncate.

        # Add the date times back in.
        return pd.Series(seasonal, index=y.index)

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> PolynomialDecomposer:
        """Fits the PolynomialDecomposer and determine the seasonal signal.

        Currently only fits the polynomial detrender.  The seasonality is determined by removing
        the trend from the signal and using statsmodels' seasonal_decompose().  Both the trend
        and seasonality are currently assumed to be additive.

        Args:
            X (pd.DataFrame, optional): Conditionally used to build datetime index.
            y (pd.Series): Target variable to detrend and deseasonalize.

        Returns:
            self

        Raises:
            ValueError: If y is None.
            ValueError: If target data doesn't have DatetimeIndex AND no Datetime features in features data
        """
        if y is None:
            raise ValueError("y cannot be None for PolynomialDecomposer!")

        # Change the y index to a matching datetimeindex or else we get a failure
        # in ForecastingHorizon during decomposition.
        if not isinstance(y.index, pd.DatetimeIndex):
            y = self._set_time_index(X, y)

        # Copying y as we might modify it's index
        y_orig = infer_feature_types(y).copy()
        self._component_obj.fit(y_orig)

        y_detrended_with_time_index = self._component_obj.transform(y_orig)

        # Save the frequency of the fitted series for checking against transform data.
        self.frequency = y_detrended_with_time_index.index.freqstr

        # statsmodel's seasonal_decompose() repeats the seasonal signal over the length of
        # the given array.  We'll extract the first iteration and save it for use in .transform()
        # TODO: Resolve with https://github.com/alteryx/evalml/issues/3708
        if self.seasonal_period == -1:
            self.periodicity = freq_to_period(self.frequency)
            self.seasonal_period = self.periodicity
        else:
            self.periodicity = self.seasonal_period

        self.seasonality = seasonal_decompose(
            y_detrended_with_time_index,
            period=self.periodicity,
        ).seasonal[0 : self.periodicity]

        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Transforms the target data by removing the polynomial trend and rolling average seasonality.

        Applies the fit polynomial detrender to the target data, removing the additive polynomial trend. Then,
        utilizes the first period's worth of seasonal data determined in the .train() function to extrapolate
        the seasonal signal of the data to be transformed.  This seasonal signal is also assumed to be additive
        and is removed.

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

        # Give the internal target signal a datetime index built from X
        y = y.copy()
        if not isinstance(y.index, pd.DatetimeIndex):
            y = self._set_time_index(X, y)

        # Remove polynomial trend then seasonality of detrended signal
        y_ww = infer_feature_types(y)
        y_detrended = self._component_obj.transform(y_ww)

        if isinstance(y.index, pd.DatetimeIndex):
            # Repeat the seasonal signal over the target data
            seasonal = np.tile(
                self.seasonality.T,
                len(y_detrended) // self.periodicity + 1,
            ).T[: len(y_detrended)]

        y_t = pd.Series(y_detrended - seasonal)
        y_t.ww.init(logical_type="double")
        return X, y_t

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Removes fitted trend and seasonality from target variable.

        Args:
            X (pd.DataFrame, optional): Ignored.
            y (pd.Series): Target variable to detrend and deseasonalize.

        Returns:
            tuple of pd.DataFrame, pd.Series: The first element are the input features returned without modification.
                The second element is the target variable y with the fitted trend removed.
        """
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Adds back fitted trend and seasonality to target variable.

        The polynomial trend is added back into the signal, calling the detrender's inverse_transform().
        Then, the seasonality is projected forward to and added back into the signal.

        Args:
            y (pd.Series): Target variable.

        Returns:
            tuple of pd.DataFrame, pd.Series: The first element are the input features returned without modification.
                The second element is the target variable y with the trend and seasonality added back in.

        Raises:
            ValueError: If y is None.
        """
        if y is None:
            raise ValueError("y cannot be None for PolynomialDecomposer!")
        y_ww = infer_feature_types(y)

        # Add polynomial trend back to signal
        y_retrended = self._component_obj.inverse_transform(y_ww)

        seasonal = self._build_seasonal_signal(
            y_ww,
            self.seasonality,
            self.periodicity,
            self.frequency,
        )
        y_t = infer_feature_types(pd.Series(y_retrended + seasonal, index=y_ww.index))
        return y_t

    def get_trend_dataframe(self, X: pd.DataFrame, y: pd.Series) -> list[pd.DataFrame]:
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

        fh = ForecastingHorizon(X.index, is_relative=False)

        result_dfs = []

        def _decompose_target(X, y, fh):
            """Function to generate a single DataFrame with trend, seasonality and residual components."""
            forecaster = (
                self._component_obj.forecaster_
            )  # the .forecaster attribute is an unfitted version
            try:
                trend = forecaster.predict(fh=fh, X=y)
            except (sklearn.exceptions.NotFittedError, AttributeError):
                raise ValueError(
                    "PolynomialDecomposer has not been fit yet.  Please fit it and then build the decomposed dataframe.",
                )
            seasonality = seasonal_decompose(
                y - trend,
                period=self.seasonal_period,
            ).seasonal
            residual = y - trend - seasonality
            return pd.DataFrame(
                {
                    "signal": y,
                    "trend": trend,
                    "seasonality": seasonality,
                    "residual": residual,
                },
            )

        if isinstance(y, pd.Series):
            result_dfs.append(_decompose_target(X, y, fh))
        elif isinstance(y, pd.DataFrame):
            for colname in y.columns:
                result_dfs.append(_decompose_target(X, y[colname], fh))

        return result_dfs

    def plot_decomposition(self, X: pd.DataFrame, y: pd.Series, show=False):
        """Plots the decomposition of the target signal.

        Args:
            X (pd.DataFrame): Input data with time series data in index.
            y (pd.Series or pd.DataFrame): Target variable data provided as a Series for univariate problems or
                a DataFrame for multivariate problems.
            show (bool): Whether to display the plot or not. Defaults to False.

        Returns:
            matplotlib.pyplot.Figure, matplotlib.pyplot.Axes: The figure and axes that have the decompositions
                plotted on them

        """
        decomposition_results = self.get_trend_dataframe(X, y)
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
        if show:  # pragma: no cover
            plt.show()
        return fig, axs
