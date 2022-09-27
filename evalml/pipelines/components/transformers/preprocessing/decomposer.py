"""Component that removes trends from time series and returns the decomposed components."""
from abc import abstractmethod

import numpy as np
import pandas as pd
import statsmodels.api as sm
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

    def __init__(self, parameters=None, component_obj=None, random_seed=0, **kwargs):
        super().__init__(
            parameters=parameters,
            component_obj=component_obj,
            random_seed=random_seed,
            **kwargs,
        )

    def _set_time_index(self, X, y):
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

    @abstractmethod
    def get_trend_dataframe(self, y):
        """Return a list of dataframes, each with 3 columns: trend, seasonality, residual."""

    @abstractmethod
    def inverse_transform(self, y):
        """Add the trend + seasonality back to y."""

    def determine_periodicity(self, X, y, method="autocorrelation"):
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

    def set_seasonal_period(self, X, y):
        """Function to set the component's seasonal period based on the target's seasonality.

        Args:
            X (pandas.DataFrame): The feature data of the time series problem.
            y (pandas.Series): The target data of a time series problem.

        """
        self.seasonal_period = self.determine_periodicity(X, y)
        self.parameters["seasonal_period"] = self.seasonal_period
