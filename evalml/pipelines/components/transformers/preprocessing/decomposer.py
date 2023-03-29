"""Component that removes trends from time series and returns the decomposed components."""
from __future__ import annotations

import re
from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.signal import argrelextrema

from evalml.pipelines.components.transformers.transformer import Transformer
from evalml.utils import get_time_index, infer_feature_types


class Decomposer(Transformer):
    """Component that removes trends and seasonality from time series and returns the decomposed components.

    Args:
        parameters (dict): Dictionary of parameters to pass to component object.
        component_obj (class) : Instance of a detrender/deseasonalizer class.
        random_seed (int): Seed for the random number generator. Defaults to 0.
        degree (int) : Currently the degree of the PolynomialDecomposer, not used for STLDecomposer.
        period (int) : The best guess, in units, for the period of the seasonal signal.
        seasonal_smoother (int): The seasonal smoothing parameter for STLDecomposer, not used for PolynomialDecomposer.
        time_index (str) : The column name of the feature matrix (X) that the datetime information
            should be pulled from.
    """

    name = "Decomposer"
    hyperparameter_ranges = None
    modifies_features = False
    modifies_target = True
    needs_fitting = True
    invalid_frequencies = []
    # Incompatibility: https://github.com/alteryx/evalml/issues/4103
    # TODO: Remove when support is added https://github.com/pandas-dev/pandas/issues/52127
    _integer_nullable_incompatibilities = ["y"]

    def __init__(
        self,
        component_obj=None,
        random_seed: int = 0,
        degree: int = 1,
        period: int = -1,
        seasonal_smoother: int = 7,
        time_index: str = None,
        **kwargs,
    ):
        degree = self._raise_typeerror_if_not_int("degree", degree)
        self.seasonal_smoother = self._raise_typeerror_if_not_int(
            "seasonal_smoother",
            seasonal_smoother,
        )
        self.period = period
        self.time_index = time_index
        parameters = {
            "degree": degree,
            "period": period,
            "seasonal_smoother": self.seasonal_smoother,
            "time_index": time_index,
        }
        parameters.update(kwargs)
        super().__init__(
            parameters=parameters,
            component_obj=component_obj,
            random_seed=random_seed,
            **kwargs,
        )

    def _raise_typeerror_if_not_int(self, var_name: str, var_value):
        if not isinstance(var_value, int):
            if isinstance(var_value, float) and var_value.is_integer():
                var_value = int(var_value)
            else:
                raise TypeError(
                    f"Parameter {var_name} must be an integer!: Received {type(var_value).__name__}",
                )
        return var_value

    def _set_time_index(self, X: pd.DataFrame, y: pd.Series):
        """Ensures that target data has a pandas.DatetimeIndex that matches feature data."""
        dt_df = infer_feature_types(X)
        time_index_name = self.time_index or self.parameters.get("time_index", None)
        time_index = get_time_index(dt_df, y, time_index_name)
        return y.set_axis(time_index)

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

    @classmethod
    def is_freq_valid(cls, freq: str):
        """Determines if the given string represents a valid frequency for this decomposer.

        Args:
            freq (str): A frequency to validate. See the pandas docs at https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases for options.

        Returns:
            boolean representing whether the frequency is valid or not.
        """
        match = re.match(r"(^\d+)?([A-Z]+)-?([A-Z]+)?", freq)
        _, freq, _ = match.groups()
        return freq not in cls.invalid_frequencies

    @abstractmethod
    def get_trend_dataframe(self, y: pd.Series):
        """Return a list of dataframes, each with 3 columns: trend, seasonality, residual."""

    @abstractmethod
    def inverse_transform(self, y: pd.Series):
        """Add the trend + seasonality back to y."""

    @classmethod
    def determine_periodicity(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        acf_threshold: float = 0.01,
        rel_max_order: int = 5,
    ):
        """Function that uses autocorrelative methods to determine the likely most signficant period of the seasonal signal.

        Args:
            X (pandas.DataFrame): The feature data of the time series problem.
            y (pandas.Series): The target data of a time series problem.
            acf_threshold (float) : The threshold for the autocorrelation function to determine the period. Any values below
                the threshold are considered to be 0 and will not be considered for the period. Defaults to 0.01.
            rel_max_order (int) : The order of the relative maximum to determine the period. Defaults to 5.

        Returns:
            int: The integer number of entries in time series data over which the seasonal part of the target data
                repeats.  If the time series data is in days, then this is the number of days that it takes the target's
                seasonal signal to repeat. Note: the target data can contain multiple seasonal signals.  This function
                will only return the stronger.  E.g. if the target has both weekly and yearly seasonality, the function
                may return either "7" or "365", depending on which seasonality is more strongly autocorrelated.  If no
                period is detected, returns None.

        """
        X, y = cls._handle_nullable_types(cls, X, y)

        def _get_rel_max_from_acf(y):
            """Determines the relative maxima of the target's autocorrelation."""
            acf = sm.tsa.acf(y, nlags=np.maximum(400, len(y)))
            # Filter out small values to avoid picking up noise
            filter_acf = [
                acf[i] if (acf[i] > acf_threshold) else 0 for i in range(len(acf))
            ]
            rel_max = argrelextrema(
                np.array(filter_acf),
                np.greater,
                order=rel_max_order,  # considers `order` points on either side to determine rel max
            )[0]
            if len(rel_max) == 0:
                return None
            max_acfs = [acf[i] for i in rel_max]
            return rel_max[np.argmax(max_acfs)]

        def _detrend_on_fly(X, y):
            """Uses a moving average to determine the target's trend and remove it."""
            # A larger moving average will be less likely to remove the seasonal signal
            # but we need to make sure we're not passing in a window that's larger than the data
            moving_avg = min(51, len(y) // 3)
            y_trend_estimate = y.rolling(moving_avg).mean().dropna()
            y_detrended = y - y_trend_estimate
            return round(
                y_detrended.dropna(),
                10,
            )  # round to 10 decimal places to avoid floating point errors

        # Make the data more stationary by detrending
        y_detrended = _detrend_on_fly(X, y)
        relative_maxima = _get_rel_max_from_acf(y_detrended)
        return relative_maxima

    def set_period(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        acf_threshold: float = 0.01,
        rel_max_order: int = 5,
    ):
        """Function to set the component's seasonal period based on the target's seasonality.

        Args:
            X (pandas.DataFrame): The feature data of the time series problem.
            y (pandas.Series): The target data of a time series problem.
            acf_threshold (float) : The threshold for the autocorrelation function to determine the period. Any values below
                the threshold are considered to be 0 and will not be considered for the period. Defaults to 0.01.
            rel_max_order (int) : The order of the relative maximum to determine the period. Defaults to 5.

        """
        self.period = self.determine_periodicity(X, y, acf_threshold, rel_max_order)
        self.update_parameters({"period": self.period})

    def _check_oos_past(self, y):
        """Function to check whether provided target data is out-of-sample and in the past."""
        index = self._choose_proper_index(y)

        if y.index[0] < index[0]:
            raise ValueError(
                f"STLDecomposer cannot transform/inverse transform data out of sample and before the data used"
                f"to fit the decomposer."
                f"\nRequested range: {str(y.index[0])}:{str(y.index[-1])}."
                f"\nSample range: {str(index[0])}:{str(index[-1])}.",
            )

    def _map_dt_to_integer(self, original_index, dt_index):
        """Function to generate an initial mapping of integer indices to datetime indices."""
        # Set an initial mapping of integers <-> datetimes at fit
        if isinstance(original_index, pd.DatetimeIndex):
            int_index = pd.RangeIndex(len(original_index))
        # Standardize the integer index as a RangeIndex and use existing integer indices
        elif isinstance(original_index, pd.RangeIndex) or original_index.is_numeric():
            int_index = pd.RangeIndex(
                start=original_index[0],
                stop=original_index[-1] + 1,
            )

        assert isinstance(dt_index, pd.DatetimeIndex)
        assert len(original_index) == len(dt_index)

        self.in_sample_integer_index = int_index
        self.in_sample_datetime_index = dt_index

    def _int_to_dt(self, integer_index_value):
        """Function to convert an integer index value to a datetime value based on the mapping made during fit."""
        try:
            dt = self.in_sample_datetime_index[
                self.in_sample_integer_index.get_loc(integer_index_value)
            ]
        except KeyError:
            more_than = integer_index_value - self.in_sample_integer_index[-1]
            dt = (
                self.in_sample_datetime_index.freq * more_than
                + self.in_sample_datetime_index[-1]
            )
        return dt

    def _convert_int_index_to_dt_index(self, integer_index):
        """Function to convert an entire index full of integers to datetimes."""
        dts = [self._int_to_dt(integer) for integer in integer_index]
        dt_index = pd.DatetimeIndex(dts, freq=self.frequency)
        return dt_index

    def _choose_proper_index(self, y):
        """Function that provides support for targets with integer and datetime indices."""
        if isinstance(y.index, pd.RangeIndex) or y.index.is_numeric():
            index = self.in_sample_integer_index
        elif isinstance(y.index, pd.DatetimeIndex):
            index = self.in_sample_datetime_index
        else:
            raise ValueError(
                f"Decomposer doesn't support target data with index of type ({type(y.index)})",
            )
        return index

    def _project_seasonal(
        self,
        y: pd.Series,
        periodic_signal: pd.Series,
        periodicity: pd.Series,
        frequency: str,
    ):
        """Projects the seasonal signal forward to cover the target data.

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
        index = self._choose_proper_index(y)

        # Determine where the seasonality starts
        if isinstance(y.index, pd.DatetimeIndex):
            transform_first_ind = (
                len(pd.date_range(start=index[0], end=y.index[0], freq=frequency))
                % periodicity
                - 1
            )
        elif isinstance(y.index, pd.RangeIndex) or y.index.is_numeric():
            first_index_diff = y.index[0] - index[0]
            transform_first_ind = first_index_diff % periodicity

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

    def plot_decomposition(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        show: bool = False,
    ) -> tuple[plt.Figure, list]:
        """Plots the decomposition of the target signal.

        Args:
            X (pd.DataFrame): Input data with time series data in index.
            y (pd.Series or pd.DataFrame): Target variable data provided as a Series for univariate problems or
                a DataFrame for multivariate problems.
            show (bool): Whether to display the plot or not. Defaults to False.

        Returns:
            matplotlib.pyplot.Figure, list[matplotlib.pyplot.Axes]: The figure and axes that have the decompositions
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

    def _check_target(self, X: pd.DataFrame, y: pd.Series):
        """Function to ensure target is not None and has a pandas.DatetimeIndex."""
        if y is None:
            raise ValueError("y cannot be None for Decomposer!")

        # Change the y index to a matching datetimeindex or else we get a failure
        # in ForecastingHorizon during decomposition.
        if not isinstance(y.index, pd.DatetimeIndex):
            y = self._set_time_index(X, y)

        return X, y
