"""Component that removes trends from time series and returns the decomposed components."""
from abc import abstractmethod

import numpy as np
import statsmodels.api as sm
from scipy.signal import argrelextrema

from evalml.pipelines.components.transformers.transformer import Transformer


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

    @abstractmethod
    def get_trend_dataframe(self, y):
        """Return a list of dataframes, each with 3 columns: trend, seasonality, residual."""

    @abstractmethod
    def inverse_transform(self, y):
        """Add the trend + seasonality back to y."""

    def determine_periodicity(self, X, y, method="autocorrelation"):
        """Function that uses autocorrelative methods to determine the first, signficant period of the seasonal signal.

        Args:
            y (pandas.Series): The target data of a time series problem.
            method (str): Either "autocorrelation" or "partial-autocorrelation".  The method by which to determine the
                first period of the seasonal part of the target signal.  Defaults to "autocorrelation".

        Returns:
            (int): The integer number of entries in time series data over which the seasonal part of the target data
                repeats.  If the time series data is in days, then this is the number of days that it takes the target's
                seasonal signal to repeat.

                Note: the target data can contain multiple seasonal signals.  This function will only return the first,
                and thus, shortest period.  E.g. if the target has both weekly and yearly seasonality, the function will
                only return "7" and not return "365".
        """

        def _get_rel_max_from_acf(y):
            autocorrelation = sm.tsa.acf(y, nlags=400)
            relative_maxima = argrelextrema(autocorrelation, np.greater)[0]
            return relative_maxima

        if method == "autocorrelation":
            import matplotlib.pyplot as plt

            relative_maxima = _get_rel_max_from_acf(y)

            if len(relative_maxima) > 0:
                # Check that the distance between local maxima is about the same
                x = relative_maxima / relative_maxima[0]
                xx = np.arange(1, len(relative_maxima) + 1)
                if not all(x == xx):
                    raise Exception
            else:
                # Try detrending the data first to get better results.
                self.fit(X, y)
                res = self.get_trend_dataframe(X, y)
                y_time_index = self._set_time_index(X, y)
                y_detrended = y_time_index - res[0]["trend"]
                relative_maxima = _get_rel_max_from_acf(y_detrended)
                print("hi")

        elif method == "partial-autocorrelation":
            partial_autocorrelation = sm.tsa.pacf(y, nlags=400)
            relative_maxima = argrelextrema(partial_autocorrelation, np.greater)[0]

        # import matplotlib.pyplot as plt
        # plt.plot(autocorrelation, "bo")
        # plt.plot(partial_autocorrelation, "rx")
        # plt.show()
        return relative_maxima[0]
