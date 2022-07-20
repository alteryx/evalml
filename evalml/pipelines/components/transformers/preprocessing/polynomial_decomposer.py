"""Component that removes trends from time series by fitting a polynomial to the data."""
from datetime import timedelta

import numpy as np
import pandas as pd
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
        degree (int): Degree for the polynomial. If 1, linear model is fit to the data.
            If 2, quadratic model is fit, etc. Defaults to 1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Polynomial Decomposer"
    hyperparameter_ranges = {"degree": Integer(1, 3)}
    """{
        "degree": Integer(1, 3)
    }"""
    modifies_features = False
    modifies_target = True

    def __init__(self, degree=1, random_seed=0, **kwargs):
        if not isinstance(degree, int):
            if isinstance(degree, float) and degree.is_integer():
                degree = int(degree)
            else:
                raise TypeError(
                    f"Parameter Degree must be an integer!: Received {type(degree).__name__}",
                )

        params = {"degree": degree}
        params.update(kwargs)
        error_msg = "sktime is not installed. Please install using 'pip install sktime'"

        trend = import_or_raise("sktime.forecasting.trend", error_msg=error_msg)
        detrend = import_or_raise(
            "sktime.transformations.series.detrend",
            error_msg=error_msg,
        )

        decomposer = detrend.Detrender(trend.PolynomialTrendForecaster(degree=degree))

        super().__init__(
            parameters=params,
            component_obj=decomposer,
            random_seed=random_seed,
        )

    def _set_time_index(self, X, y):
        """Ensures that target data has a pandas.DatetimeIndex that matches feature data."""
        X = infer_feature_types(X)
        if X.ww.select("Datetime").shape[1] != 1:
            raise ValueError(
                "There are no Datetime features in the feature data and the target data doesn't have Datetime index.",
            )
        dt_col = X.ww.select("Datetime").squeeze()
        time_index = pd.DatetimeIndex(dt_col, freq=pd.infer_freq(dt_col)).rename(
            y.index.name,
        )
        return y.set_axis(time_index)

    def build_seasonal_signal(self, y_ww, periodic_signal, periodicity, frequency):
        # Determine where the seasonality starts
        first_index_diff = y_ww.index[0] - periodic_signal.index[0]
        # TODO: Write tests to test different time series frequencies.
        if frequency == "D":
            delta = timedelta(days=1)
            period = timedelta(days=periodicity)

        # Determine which index of the sample of seasonal data the transformed data starts at
        transform_first_ind = int((first_index_diff % period) / delta)

        # Cycle the sample of seasonal data so the transformed data's effective index is first
        rotated_seasonal_sample = np.roll(
            periodic_signal.T.values,
            -transform_first_ind,
        )

        # Repeat the single, rotated period of seasonal data to cover the entirety of the data
        # to be transformed.
        seasonal = np.tile(rotated_seasonal_sample, len(y_ww) // periodicity + 1).T[
            : len(y_ww)
        ]  # The extrapolated seasonal data will be too long, so truncate.

        # Add the date times back in.
        return pd.Series(seasonal, index=y_ww.index)

    def fit(self, X, y=None):
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

        # Copying y as we might modify it's index
        y_orig = infer_feature_types(y).copy()
        self._component_obj.fit(y_orig)

        y_detrended = self._component_obj.transform(y_orig)

        if not isinstance(y_detrended.index, pd.DatetimeIndex):
            y_detrended_with_time_index = self._set_time_index(X, y_detrended)
        else:
            y_detrended_with_time_index = y_detrended

        # Save the frequency of the fitted series for checking against transform data.
        self.frequency = y_detrended_with_time_index.index.freqstr

        # statsmodel's seasonal_decompose() repeats the seasonal signal over the length of
        # the given array.  We'll extract the first iteration and save it for use in .transform()
        self.periodicity = freq_to_period(self.frequency)

        self.seasonality = seasonal_decompose(y_detrended_with_time_index).seasonal[
            0 : self.periodicity
        ]

        return self

    def transform(self, X, y=None):
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

        # Remove polynomial trend then seasonality of detrended signal
        y_ww = infer_feature_types(y)
        y_detrended = self._component_obj.transform(y_ww)

        y = y.copy()
        if not isinstance(y.index, pd.DatetimeIndex):
            y = self._set_time_index(X, y)

        if isinstance(y.index, pd.DatetimeIndex):
            # Repeat the seasonal signal over the target data
            seasonal = np.tile(
                self.seasonality.T,
                len(y_detrended) // self.periodicity + 1,
            ).T[: len(y_detrended)]

        y_t = pd.Series(y_detrended - seasonal).set_axis(y.index)
        y_t.ww.init(logical_type="double")
        return X, y_t

    def fit_transform(self, X, y=None):
        """Removes fitted trend and seasonality from target variable.

        Args:
            X (pd.DataFrame, optional): Ignored.
            y (pd.Series): Target variable to detrend and deseasonalize.

        Returns:
            tuple of pd.DataFrame, pd.Series: The first element are the input features returned without modification.
                The second element is the target variable y with the fitted trend removed.
        """
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, y):
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

        seasonal = self.build_seasonal_signal(
            y_ww,
            self.seasonality,
            self.periodicity,
            self.frequency,
        )
        y_t = infer_feature_types(pd.Series(y_retrended + seasonal, index=y_ww.index))
        return y_t

    def get_trend_dataframe(self, X, y):
        """Return a list of dataframes with 3 columns: trend, seasonality, residual.

        Scikit-learn's PolynomialForecaster is used to generate the trend portion of the target data. statsmodel's
        seasonal_decompose is used to generate the seasonality of the data.

        Args:
            X (pd.DataFrame): Input data with time series data in index.
            y (pd.Series or pd.DataFrame): Target variable data provided as a Series for univariate problems or
                a DataFrame for multivariate problems.

        Returns:
            list of pd.DataFrame: Each DataFrame contains the columns "trend", "seasonality" and "residual,"
                with the column values being the decomposed elements of the target data.

        Raises:
            TypeError: If X does not have time-series data in the index.
            ValueError: If time series index of X does not have an inferred frequency.
            TypeError: If y is not provided as a pandas Series or DataFrame.

        """
        X = infer_feature_types(X)
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("Provided X should have datetimes in the index.")
        if X.index.freq is None:
            raise ValueError(
                "Provided DatetimeIndex of X should have an inferred frequency.",
            )
        fh = ForecastingHorizon(X.index, is_relative=False)

        result_dfs = []

        def _decompose_target(X, y, fh):
            """Function to generate a single DataFrame with trend, seasonality and residual components."""
            forecaster = self._component_obj.forecaster.clone()
            forecaster.fit(y=y, X=X)
            trend = forecaster.predict(fh=fh, X=y)
            seasonality = seasonal_decompose(y - trend).seasonal
            residual = y - trend - seasonality
            return pd.DataFrame(
                {
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
        else:
            raise TypeError("y must be pd.Series or pd.DataFrame!")

        return result_dfs
