"""Holt-Winters Exponential Smoothing Forecaster."""
import numpy as np
import pandas as pd
from skopt.space import Integer

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import import_or_raise, infer_feature_types


class ExponentialSmoothingRegressor(Estimator):
    """Holt-Winters Exponential Smoothing Forecaster.

    Currently ExponentialSmoothingRegressor isn't supported via conda install. It's recommended that it be installed via PyPI.

    Args:
        date_index (str): Specifies the name of the column in X that provides the datetime objects. Defaults to None.
        n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines. Defaults to -1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Exponential Smoothing Regressor"
    hyperparameter_ranges = {
    }
    """{
    }"""
    model_family = ModelFamily.EXPONENTIAL_SMOOTHING
    """ModelFamily.EXPONENTIAL_SMOOTHING"""
    supported_problem_types = [ProblemTypes.TIME_SERIES_REGRESSION]
    """[ProblemTypes.TIME_SERIES_REGRESSION]"""

    def __init__(
        self,
        date_index=None,
        forecast_horizon=None,
        n_jobs=-1,
        random_seed=0,
        **kwargs,
    ):
        parameters = {
            "date_index": date_index,
            "forecast_horizon": forecast_horizon,
        }
        parameters.update(kwargs)

        smoothing_model_msg = (
            "sktime is not installed. Please install using `pip install sktime.`"
        )
        sktime_smoothing = import_or_raise(
            "sktime.forecasting.exp_smoothing", error_msg=smoothing_model_msg
        )
        smoothing_model = sktime_smoothing.ExponentialSmoothing()

        super().__init__(
            parameters=parameters, component_obj=smoothing_model, random_seed=random_seed
        )

    def fit(self, X, y=None):
        """Fits Exponential Smoothing regressor to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series): The target training data of length [n_samples].

        Returns:
            self

        Raises:
            ValueError: If X was passed to `fit` but not passed in `predict`.
        """
        if y is None:
            raise ValueError("Exponential Smoothing Regressor requires y as input.")

        X, y = self._manage_woodwork(X, y)
        if X is not None and not X.empty:
            X = X.select_dtypes(exclude=["datetime64"])
            #self._component_obj.fit(y=y, X=X, fh=self.parameters["forecast_horizon"])
            self._component_obj.fit(y=y, X=X)
        else:
            #self._component_obj.fit(y=y, fh=self.parameters["forecast_horizon"])
            self._component_obj.fit(y=y)
        return self

    def predict(self, X, y=None):
        """Make predictions using fitted Exponential Smoothing regressor.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].
            y (pd.Series): Target data.

        Returns:
            pd.Series: Predicted values.

        Raises:
            ValueError: If X was passed to `fit` but not passed in `predict`.
        """
        X, y = self._manage_woodwork(X, y)
        if X is not None and not X.empty:
            X = X.select_dtypes(exclude=["datetime64"])
            y_pred = self._component_obj.predict(fh=self.parameters["forecast_horizon"], X=X)
        else:
            y_pred = self._component_obj.predict(fh=self.parameters["forecast_horizon"])
        return infer_feature_types(y_pred)

    @property
    def feature_importance(self):
        """Returns array of 0's with a length of 1 as feature_importance is not defined for Exponential Smoothing regressor."""
        return np.zeros(1)
