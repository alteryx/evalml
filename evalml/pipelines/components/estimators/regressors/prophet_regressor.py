"""Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well."""
import copy
from typing import Dict, List

import numpy as np
import pandas as pd
from skopt.space import Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import import_or_raise, infer_feature_types
from evalml.utils.gen_utils import classproperty


class ProphetRegressor(Estimator):
    """Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.

    More information here: https://facebook.github.io/prophet/
    """

    name = "Prophet Regressor"
    hyperparameter_ranges = {
        "changepoint_prior_scale": Real(0.001, 0.5),
        "seasonality_prior_scale": Real(0.01, 10),
        "holidays_prior_scale": Real(0.01, 10),
        "seasonality_mode": ["additive", "multiplicative"],
    }
    """{
        "changepoint_prior_scale": Real(0.001, 0.5),
        "seasonality_prior_scale": Real(0.01, 10),
        "holidays_prior_scale": Real(0.01, 10),
        "seasonality_mode": ["additive", "multiplicative"],
    }"""
    model_family = ModelFamily.PROPHET
    """ModelFamily.PROPHET"""
    supported_problem_types = [ProblemTypes.TIME_SERIES_REGRESSION]
    """[ProblemTypes.TIME_SERIES_REGRESSION]"""

    def __init__(
        self,
        time_index=None,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        holidays_prior_scale=10,
        seasonality_mode="additive",
        random_seed=0,
        stan_backend="CMDSTANPY",
        interval_width=0.95,
        **kwargs,
    ):
        parameters = {
            "changepoint_prior_scale": changepoint_prior_scale,
            "seasonality_prior_scale": seasonality_prior_scale,
            "holidays_prior_scale": holidays_prior_scale,
            "seasonality_mode": seasonality_mode,
            "stan_backend": stan_backend,
            "interval_width": interval_width,
        }

        parameters.update(kwargs)

        c_error_msg = (
            "cmdstanpy is not installed. Please install using `pip install cmdstanpy`."
        )
        _ = import_or_raise("cmdstanpy", error_msg=c_error_msg)

        p_error_msg = (
            "prophet is not installed. Please install using `pip install prophet`."
        )
        prophet = import_or_raise("prophet", error_msg=p_error_msg)

        prophet_regressor = prophet.Prophet(**parameters)
        parameters["time_index"] = time_index
        self.time_index = time_index

        super().__init__(
            parameters=parameters,
            component_obj=prophet_regressor,
            random_state=random_seed,
        )

    @staticmethod
    def build_prophet_df(X, y=None, time_index="ds"):
        """Build the Prophet data to pass fit and predict on."""
        X = copy.deepcopy(X)
        y = copy.deepcopy(y)
        if time_index is None:
            raise ValueError("time_index cannot be None!")

        if time_index in X.columns:
            date_column = X.pop(time_index)
        else:
            raise ValueError(f"Column {time_index} was not found in X!")

        prophet_df = X
        if y is not None:
            y.index = prophet_df.index
            prophet_df["y"] = y
        prophet_df["ds"] = date_column

        return prophet_df

    def fit(self, X, y=None):
        """Fits Prophet regressor component to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series): The target training data of length [n_samples].

        Returns:
            self
        """
        X, y = super()._manage_woodwork(X, y)

        prophet_df = ProphetRegressor.build_prophet_df(
            X=X,
            y=y,
            time_index=self.time_index,
        )

        self._component_obj.fit(prophet_df)
        return self

    def predict(self, X, y=None):
        """Make predictions using fitted Prophet regressor.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].
            y (pd.Series): Target data. Ignored.

        Returns:
            pd.Series: Predicted values.
        """
        X = infer_feature_types(X)

        prophet_df = ProphetRegressor.build_prophet_df(
            X=X,
            y=y,
            time_index=self.time_index,
        )

        prophet_output = self._component_obj.predict(prophet_df)
        predictions = prophet_output["yhat"]
        predictions = infer_feature_types(predictions)
        predictions = predictions.rename(None)
        predictions.index = X.index

        return predictions

    def get_prediction_intervals(
        self,
        X,
        y=None,
        coverage: List[float] = None,
    ) -> Dict[str, pd.Series]:
        """Find the prediction intervals using the fitted ProphetRegressor.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].
            y (pd.Series): Target data. Ignored.
            coverage (List[float]): A list of floats between the values 0 and 1 that the upper and lower bounds of the
                prediction interval should be calculated for.

        Returns:
            dict: Prediction intervals, keys are in the format {coverage}_lower or {coverage}_upper.
        """
        if coverage is None:
            coverage = [0.95]

        prediction_interval_result = {}
        for conf_int in coverage:
            self._component_obj.interval_width = conf_int
            X = infer_feature_types(X)

            prophet_df = ProphetRegressor.build_prophet_df(
                X=X,
                y=y,
                time_index=self.time_index,
            )
            prophet_output = self._component_obj.predict(prophet_df)
            prediction_interval_lower = prophet_output["yhat_lower"]
            prediction_interval_upper = prophet_output["yhat_upper"]
            prediction_interval_lower.index = X.index
            prediction_interval_upper.index = X.index
            prediction_interval_result[f"{conf_int}_lower"] = prediction_interval_lower
            prediction_interval_result[f"{conf_int}_upper"] = prediction_interval_upper
        return prediction_interval_result

    def get_params(self):
        """Get parameters for the Prophet regressor."""
        return self.__dict__["_parameters"]

    @property
    def feature_importance(self):
        """Returns array of 0's with len(1) as feature_importance is not defined for Prophet regressor."""
        return np.zeros(1)

    @classproperty
    def default_parameters(cls):
        """Returns the default parameters for this component.

        Returns:
            dict: Default parameters for this component.
        """
        parameters = {
            "changepoint_prior_scale": 0.05,
            "time_index": None,
            "seasonality_prior_scale": 10,
            "holidays_prior_scale": 10,
            "interval_width": 0.95,
            "seasonality_mode": "additive",
            "stan_backend": "CMDSTANPY",
        }
        return parameters
