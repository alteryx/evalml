"""Autoregressive Integrated Moving Average Model. The three parameters (p, d, q) are the AR order, the degree of differencing, and the MA order. More information here: https://www.statsmodels.org/devel/generated/statsmodels.tsa.arima.model.ARIMA.html."""
import numpy as np
import pandas as pd
from skopt.space import Integer
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import (
    downcast_int_nullable_to_double,
    import_or_raise,
    infer_feature_types,
)


class ARIMAStatsRegressor(Estimator):
    """Autoregressive Integrated Moving Average Model implemented by statsforecast.

    Args:
        time_index (str): Specifies the name of the column in X that provides the datetime objects. Defaults to None.
        sp (int or str): Period for seasonal differencing, specifically the number of periods in each season. If "detect", this
            model will automatically detect this parameter (given the time series is a standard frequency) and will fall
            back to 1 (no seasonality) if it cannot be detected. Defaults to 1.
        n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines. Defaults to -1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "ARIMA Statsforecast Regressor"
    hyperparameter_ranges = {}
    """{}"""
    model_family = ModelFamily.ARIMA
    """ModelFamily.ARIMA"""
    supported_problem_types = [ProblemTypes.TIME_SERIES_REGRESSION]
    """[ProblemTypes.TIME_SERIES_REGRESSION]"""

    def __init__(
        self,
        time_index=None,
        trend=None,
        sp="detect",
        n_jobs=-1,
        random_seed=0,
        **kwargs,
    ):
        parameters = {
            "sp": sp,
            "time_index": time_index,
            "n_jobs": n_jobs,
        }
        parameters.update(kwargs)

        super().__init__(
            parameters=parameters,
            component_obj=None,
            random_seed=random_seed,
        )

    def _get_sp(self, X, freq):
        if X is None:
            return 1
        freq_mappings = {
            "D": 7,
            "M": 12,
            "Q": 4,
        }
        time_index = self._parameters.get("time_index", None)
        sp = self.parameters.get("sp")
        if sp == "detect":
            inferred_freqs = X.ww.infer_temporal_frequencies()
            freq = inferred_freqs.get(time_index, None)
            sp = 1
            if freq is not None:
                sp = freq_mappings.get(freq[:1], 1)
        return sp

    def fit(self, X, y=None):
        """Fits ARIMA regressor to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series): The target training data of length [n_samples].

        Returns:
            self

        Raises:
            ValueError: If y was not passed in.
        """
        time_index = self.parameters.get("time_index")
        if X is not None:
            X = downcast_int_nullable_to_double(X)
            X = X.fillna(X.mean())
        if y is None:
            raise ValueError("ARIMA Regressor requires y as input.")
        X, y = self._manage_woodwork(X, y)

        inferred_freqs = X.ww.infer_temporal_frequencies()
        freq = inferred_freqs.get(time_index, None)
        sp = self._get_sp(X, freq)

        forecast_df = pd.DataFrame(
            {"ds": X[time_index], "y": y, "unique_id": [0] * len(y)},
        )
        models = [AutoARIMA(season_length=sp)]
        fcst = StatsForecast(
            df=forecast_df,
            models=models,
            freq=freq,
            n_jobs=self.parameters.get("n_jobs"),
        )
        self._component_obj = fcst

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
        y_pred = self._component_obj.forecast(len(X))
        y_pred = pd.Series(y_pred["AutoARIMA"])
        return infer_feature_types(y_pred)

    @property
    def feature_importance(self):
        """Returns array of 0's with a length of 1 as feature_importance is not defined for ARIMA regressor."""
        return np.zeros(1)
