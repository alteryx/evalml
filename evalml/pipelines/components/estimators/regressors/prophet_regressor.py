import copy

import numpy as np
import pandas as pd
from skopt.space import Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import import_or_raise, infer_feature_types
from evalml.utils.gen_utils import classproperty


class ProphetRegressor(Estimator):
    """
    Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.
    It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.

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
        date_column="ds",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        holidays_prior_scale=10,
        seasonality_mode="additive",
        random_seed=0,
        stan_backend="CMDSTANPY",
        **kwargs
    ):
        self.date_column = date_column

        parameters = {
            "changepoint_prior_scale": changepoint_prior_scale,
            "seasonality_prior_scale": seasonality_prior_scale,
            "holidays_prior_scale": holidays_prior_scale,
            "seasonality_mode": seasonality_mode,
            "stan_backend": stan_backend,
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
        super().__init__(
            parameters=parameters,
            component_obj=prophet_regressor,
            random_state=random_seed,
        )

    @staticmethod
    def build_prophet_df(X, y=None, date_column="ds"):
        if X is not None:
            X = copy.deepcopy(X)
        if y is not None:
            y = copy.deepcopy(y)

        if date_column in X.columns:
            date_col = X[date_column]
        elif isinstance(X.index, pd.DatetimeIndex):
            date_col = X.reset_index()
            date_col = date_col["index"]
        elif isinstance(y.index, pd.DatetimeIndex):
            date_col = y.reset_index()
            date_col = date_col["index"]
        else:
            msg = "Prophet estimator requires input data X to have a datetime column specified by the 'date_column' parameter. If it doesn't find one, it will look for the datetime column in the index of X or y."
            raise ValueError(msg)

        date_col = date_col.rename("ds")
        prophet_df = date_col.to_frame()
        if y is not None:
            y.index = prophet_df.index
            prophet_df["y"] = y
        return prophet_df

    def fit(self, X, y=None):
        if X is None:
            X = pd.DataFrame()

        X, y = super()._manage_woodwork(X, y)

        prophet_df = ProphetRegressor.build_prophet_df(
            X=X, y=y, date_column=self.date_column
        )

        self._component_obj.fit(prophet_df)
        return self

    def predict(self, X, y=None):
        if X is None:
            X = pd.DataFrame()

        X = infer_feature_types(X)

        prophet_df = ProphetRegressor.build_prophet_df(
            X=X, y=y, date_column=self.date_column
        )

        y_pred = self._component_obj.predict(prophet_df)["yhat"]
        return y_pred

    def get_params(self):
        return self.__dict__["_parameters"]

    @property
    def feature_importance(self):
        """
        Returns array of 0's with len(1) as feature_importance is not defined for Prophet regressor.
        """
        return np.zeros(1)

    @classproperty
    def default_parameters(cls):
        """Returns the default parameters for this component.

        Our convention is that Component.default_parameters == Component().parameters.

        Returns:
            dict: default parameters for this component.
        """

        parameters = {
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10,
            "holidays_prior_scale": 10,
            "seasonality_mode": "additive",
            "stan_backend": "CMDSTANPY",
        }

        return parameters
