import numpy as np
import statsmodels.api as sm
from pandas.core.nanops import nanmean as pd_nanmean
from statsmodels.tsa.filters._utils import _get_pandas_wrapper
from statsmodels.tsa.seasonal import STL


class STL:
    def __init__(self, period=365, lo_frac=0.6, lo_delta=0.01):
        self.period = period
        self.lo_frac = lo_frac
        self.lo_delta = lo_delta

        # use some existing pieces of statsmodels
        self.lowess = sm.nonparametric.lowess

    def fit(self, y):
        _pandas_wrapper = _get_pandas_wrapper(y)

        # get plain np array
        observed = np.asanyarray(y).squeeze()

        # calc trend, remove from observation
        self.trend = self.lowess(
            observed,
            [x for x in range(len(observed))],
            frac=self.lo_frac,
            delta=self.lo_delta * len(observed),
            return_sorted=False,
        )

        # period must not be larger than size of series to avoid introducing NaNs
        self.period = min(self.period, len(observed))

        detrended = observed - self.trend
        # calc one-period seasonality, remove tiled array from detrended
        period_averages = np.array(
            [pd_nanmean(detrended[i :: self.period]) for i in range(self.period)],
        )
        # 0-center the period avgs
        period_averages -= np.mean(period_averages)
        self.seasonal = np.tile(period_averages, len(observed) // self.period + 1)[
            : len(observed)
        ]
        return self

    def transform(self, y):
        _pandas_wrapper = _get_pandas_wrapper(y)

        # get plain np array
        observed = np.asanyarray(y).squeeze()
        self.detrended = observed - self.trend
        self.residual = self.detrended - self.seasonal
        return self.residual

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self):
        pass


class STLStats:
    def __init__(self, period=365, lo_frac=0.6, lo_delta=0.01):
        self.period = period
        self.lo_frac = lo_frac
        self.lo_delta = lo_delta

        # use some existing pieces of statsmodels
        self.lowess = sm.nonparametric.lowess

    def fit(self, y):
        _pandas_wrapper = _get_pandas_wrapper(y)

        # get plain np array
        observed = np.asanyarray(y).squeeze()

        # calc trend, remove from observation
        self.trend = self.lowess(
            observed,
            [x for x in range(len(observed))],
            frac=self.lo_frac,
            delta=self.lo_delta * len(observed),
            return_sorted=False,
        )

        # period must not be larger than size of series to avoid introducing NaNs
        self.period = min(self.period, len(observed))

        detrended = observed - self.trend
        # calc one-period seasonality, remove tiled array from detrended
        period_averages = np.array(
            [pd_nanmean(detrended[i :: self.period]) for i in range(self.period)],
        )
        # 0-center the period avgs
        period_averages -= np.mean(period_averages)
        self.seasonal = np.tile(period_averages, len(observed) // self.period + 1)[
            : len(observed)
        ]
        return self

    def transform(self, y):
        _pandas_wrapper = _get_pandas_wrapper(y)

        # get plain np array
        observed = np.asanyarray(y).squeeze()
        self.detrended = observed - self.trend
        self.residual = self.detrended - self.seasonal
        return self.residual

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self):
        pass
