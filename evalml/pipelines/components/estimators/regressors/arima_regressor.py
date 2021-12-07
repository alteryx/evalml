"""Autoregressive Integrated Moving Average Model. The three parameters (p, d, q) are the AR order, the degree of differencing, and the MA order. More information here: https://www.statsmodels.org/devel/generated/statsmodels.tsa.arima_model.ARIMA.html."""
import numpy as np
import pandas as pd
from skopt.space import Integer

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import import_or_raise, infer_feature_types


class ARIMARegressor(Estimator):
    """Autoregressive Integrated Moving Average Model. The three parameters (p, d, q) are the AR order, the degree of differencing, and the MA order. More information here: https://www.statsmodels.org/devel/generated/statsmodels.tsa.arima_model.ARIMA.html.

    Currently ARIMARegressor isn't supported via conda install. It's recommended that it be installed via PyPI.

    Args:
        date_index (str): Specifies the name of the column in X that provides the datetime objects. Defaults to None.
        trend (str): Controls the deterministic trend. Options are ['n', 'c', 't', 'ct'] where 'c' is a constant term,
            't' indicates a linear trend, and 'ct' is both. Can also be an iterable when defining a polynomial, such
            as [1, 1, 0, 1].
        start_p (int): Minimum Autoregressive order. Defaults to 2.
        d (int): Minimum Differencing degree. Defaults to 0.
        start_q (int): Minimum Moving Average order. Defaults to 2.
        max_p (int): Maximum Autoregressive order. Defaults to 5.
        max_d (int): Maximum Differencing degree. Defaults to 2.
        max_q (int): Maximum Moving Average order. Defaults to 5.
        seasonal (boolean): Whether to fit a seasonal model to ARIMA. Defaults to True.
        n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines. Defaults to -1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "ARIMA Regressor"
    hyperparameter_ranges = {
        "start_p": Integer(1, 3),
        "d": Integer(0, 2),
        "start_q": Integer(1, 3),
        "max_p": Integer(3, 10),
        "max_d": Integer(2, 5),
        "max_q": Integer(3, 10),
        "seasonal": [True, False],
    }
    """{
        "start_p": Integer(1, 3),
        "d": Integer(0, 2),
        "start_q": Integer(1, 3),
        "max_p": Integer(3, 10),
        "max_d": Integer(2, 5),
        "max_q": Integer(3, 10),
        "seasonal": [True, False],
    }"""
    model_family = ModelFamily.ARIMA
    """ModelFamily.ARIMA"""
    supported_problem_types = [ProblemTypes.TIME_SERIES_REGRESSION]
    """[ProblemTypes.TIME_SERIES_REGRESSION]"""

    def __init__(
        self,
        date_index=None,
        trend=None,
        start_p=2,
        d=0,
        start_q=2,
        max_p=5,
        max_d=2,
        max_q=5,
        seasonal=True,
        n_jobs=-1,
        random_seed=0,
        **kwargs,
    ):
        parameters = {
            "trend": trend,
            "start_p": start_p,
            "d": d,
            "start_q": start_q,
            "max_p": max_p,
            "max_d": max_d,
            "max_q": max_q,
            "seasonal": seasonal,
            "n_jobs": n_jobs,
            "date_index": date_index,
        }

        parameters.update(kwargs)

        arima_model_msg = (
            "sktime is not installed. Please install using `pip install sktime.`"
        )
        sktime_arima = import_or_raise(
            "sktime.forecasting.arima", error_msg=arima_model_msg
        )
        arima_model = sktime_arima.AutoARIMA(**parameters)

        super().__init__(
            parameters=parameters, component_obj=arima_model, random_seed=random_seed
        )

    def _remove_datetime(self, data, features=False):
        if data is None:
            return None
        data_no_dt = data.copy()
        if isinstance(
            data_no_dt.index, (pd.DatetimeIndex, pd.PeriodIndex, pd.IntervalIndex)
        ):
            data_no_dt = data_no_dt.reset_index(drop=True)
        if features:
            data_no_dt = data_no_dt.select_dtypes(exclude=["datetime64"])

        return data_no_dt

    def _match_indices(self, X, y):
        if X is not None:
            if X.index.equals(y.index):
                return X, y
            else:
                y.index = X.index
        return X, y

    def _set_forecast(self, X):
        from sktime.forecasting.base import ForecastingHorizon

        fh_ = ForecastingHorizon([i + 1 for i in range(len(X))], is_relative=True)
        return fh_

    def fit(self, X, y=None):
        """Fits ARIMA regressor to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series): The target training data of length [n_samples].

        Returns:
            self

        Raises:
            ValueError: If X was passed to `fit` but not passed in `predict`.
        """
        X, y = self._manage_woodwork(X, y)
        if y is None:
            raise ValueError("ARIMA Regressor requires y as input.")

        X = self._remove_datetime(X, features=True)
        y = self._remove_datetime(y)
        X, y = self._match_indices(X, y)

        if X is not None and not X.empty:
            self._component_obj.fit(y=y, X=X)
        else:
            self._component_obj.fit(y=y)
        return self

    def predict(self, X, y=None):
        """Make predictions using fitted ARIMA regressor.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].
            y (pd.Series): Target data.

        Returns:
            pd.Series: Predicted values.

        Raises:
            ValueError: If X was passed to `fit` but not passed in `predict`.
        """
        X, y = self._manage_woodwork(X, y)
        fh_ = self._set_forecast(X)
        X = X.select_dtypes(exclude=["datetime64"])

        if not X.empty:
            y_pred = self._component_obj.predict(fh=fh_, X=X)
        else:
            y_pred = self._component_obj.predict(fh=fh_)
        y_pred.index = X.index

        return infer_feature_types(y_pred)

    @property
    def feature_importance(self):
        """Returns array of 0's with a length of 1 as feature_importance is not defined for ARIMA regressor."""
        return np.zeros(1)
