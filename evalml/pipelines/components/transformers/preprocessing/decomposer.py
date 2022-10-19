"""Component that removes trends from time series and returns the decomposed components."""
from __future__ import annotations

from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.core.index import Int64Index
from scipy.signal import argrelextrema

from evalml.pipelines.components.transformers.transformer import Transformer
from evalml.utils import infer_feature_types


class Decomposer(Transformer):
    """Component that removes trends and seasonality from time series and returns the decomposed components.

    Args:
        parameters (dict): Dictionary of parameters to pass to component object.
        component_obj (class) : Instance of a detrender/deseasonalizer class.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Decomposer"
    hyperparameter_ranges = None
    modifies_features = False
    modifies_target = True
    needs_fitting = True

    def __init__(
        self,
        component_obj=None,
        random_seed: int = 0,
        degree: int = 1,
        seasonal_period: int = -1,
        time_index: str = None,
        **kwargs,
    ):
        degree = self._raise_typeerror_if_not_int("degree", degree)
        self.seasonal_period = self._raise_typeerror_if_not_int(
            "seasonal_period",
            seasonal_period,
        )
        self.time_index = time_index
        parameters = {
            "degree": degree,
            "seasonal_period": self.seasonal_period,
            "time_index": time_index,
        }
        parameters.update(kwargs)
        super().__init__(
            parameters=parameters,
            component_obj=component_obj,
            random_seed=random_seed,
            **kwargs,
        )

    def _raise_typeerror_if_not_int(self, var_name: str, var_value: str):
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

        # Prefer the user's provided time_index, if it exists
        if self.time_index and self.time_index in dt_df.columns:
            dt_col = dt_df[self.time_index]
        # If user's provided time_index doesn't exist, log it and find some datetimes to use
        elif (self.time_index is None) or self.time_index not in dt_df.columns:
            self.logger.warning(
                f"PolynomialDecomposer could not find requested time_index {self.time_index}",
            )
            # Use the feature data's index, preferentially
            num_datetime_features = dt_df.ww.select("Datetime").shape[1]
            if isinstance(dt_df.index, pd.DatetimeIndex):
                dt_col = pd.Series(dt_df.index)
            elif num_datetime_features == 0:
                raise ValueError(
                    "There are no Datetime features in the feature data and neither the feature nor the target data have a DateTime index.",
                )
            # Use a datetime column of the features if there's only one
            elif num_datetime_features == 1:
                dt_col = dt_df.ww.select("Datetime").squeeze()
            # With more than one datetime column, use the time_index parameter, if provided.
            elif num_datetime_features > 1:
                if self.parameters.get("time_index", None) is None:
                    raise ValueError(
                        "Too many Datetime features provided in data but no time_index column specified during __init__.",
                    )
                elif not self.parameters["time_index"] in X:
                    time_index_col = self.parameters["time_index"]
                    raise ValueError(
                        f"Too many Datetime features provided in data and provided time_index column {time_index_col} not present in data.",
                    )

        time_index = pd.DatetimeIndex(dt_col, freq=pd.infer_freq(dt_col)).rename(
            y.index.name,
        )
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

    @abstractmethod
    def get_trend_dataframe(self, y: pd.Series):
        """Return a list of dataframes, each with 3 columns: trend, seasonality, residual."""

    @abstractmethod
    def inverse_transform(self, y: pd.Series):
        """Add the trend + seasonality back to y."""

    def determine_periodicity(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "autocorrelation",
    ):
        """Function that uses autocorrelative methods to determine the first, signficant period of the seasonal signal.

        Args:
            X (pandas.DataFrame): The feature data of the time series problem.
            y (pandas.Series): The target data of a time series problem.
            method (str): Either "autocorrelation" or "partial-autocorrelation".  The method by which to determine the
                first period of the seasonal part of the target signal.  "partial-autocorrelation" should currently not
                be used.  Defaults to "autocorrelation".

        Returns:
            (list[int]): The integer numbers of entries in time series data over which the seasonal part of the target data
                repeats.  If the time series data is in days, then this is the number of days that it takes the target's
                seasonal signal to repeat. Note: the target data can contain multiple seasonal signals.  This function
                will only return the first, and thus, shortest period.  E.g. if the target has both weekly and yearly
                seasonality, the function will only return "7" and not return "365".  If no period is detected, returns [None].

        """

        def _get_rel_max_from_acf(y):
            """Determines the relative maxima of the target's autocorrelation."""
            acf = sm.tsa.acf(y, nlags=np.maximum(400, len(y)))
            filter_acf = [acf[i] if (acf[i] > 0) else 0 for i in range(len(acf))]
            rel_max = argrelextrema(
                np.array(filter_acf),
                np.greater,
                order=5,  # considers 5 points on either side to determine rel max
            )[0]
            max_acfs = [acf[i] for i in rel_max]
            if len(max_acfs) > 0:
                rel_max = np.array([filter_acf.index(max(max_acfs))])
            else:
                rel_max = []
            return rel_max

        def _get_rel_max_from_pacf(y):
            """Determines the relative maxima of the target's partial autocorrelation."""
            pacf = sm.tsa.pacf(y)
            return argrelextrema(pacf, np.greater)[0]

        def _detrend_on_fly(X, y):
            """Uses the underlying decomposer to determine the target's trend and remove it."""
            self.fit(X, y)
            res = self.get_trend_dataframe(X, y)
            y_time_index = self._set_time_index(X, y)
            y_detrended = y_time_index - res[0]["trend"]
            return y_detrended

        if method == "autocorrelation":
            _get_rel_max = _get_rel_max_from_acf
        elif method == "partial-autocorrelation":
            self.logger.warning(
                "Partial autocorrelations are not currently guaranteed to be accurate due to the need for continuing "
                "algorithmic work and should not be used at this time.",
            )
            _get_rel_max = _get_rel_max_from_pacf

        # Make the data more stationary by detrending
        y_detrended = _detrend_on_fly(X, y)
        relative_maxima = _get_rel_max(y_detrended)
        self.logger.info(
            f"Decomposer discovered {len(relative_maxima)} possible periods.",
        )

        if len(relative_maxima) == 0:
            self.logger.warning("No periodic signal could be detected in target data.")
            relative_maxima = [None]
        return relative_maxima[0]

    def set_seasonal_period(self, X: pd.DataFrame, y: pd.Series):
        """Function to set the component's seasonal period based on the target's seasonality.

        Args:
            X (pandas.DataFrame): The feature data of the time series problem.
            y (pandas.Series): The target data of a time series problem.

        """
        self.seasonal_period = self.determine_periodicity(X, y)
        self.parameters["seasonal_period"] = self.seasonal_period

    def _choose_proper_index(self, y):
        # TODO: Need to update this after we upgrade to Pandas 1.5+ and Int64Index is deprecated.
        if isinstance(y.index, (Int64Index, pd.RangeIndex)):
            index = self.original_index
        elif isinstance(y.index, pd.DatetimeIndex):
            index = self.trend.index
        else:
            raise ValueError(
                f"STLDecomposer doesn't support target data with index of type ({type(y.index)})",
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
                len(pd.date_range(start=index[0], end=y.index[0], freq=frequency)) % 12
                - 1
            )
        elif isinstance(y.index, Int64Index):
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
