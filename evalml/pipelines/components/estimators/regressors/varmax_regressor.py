"""Vector Autoregressive Moving Average with eXogenous regressors model. The two parameters (p, q) are the AR order and the MA order. More information here: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.varmax.VARMAX.html."""
from typing import Dict, Hashable, List, Optional, Union

import numpy as np
import pandas as pd
from skopt.space import Categorical, Integer
from sktime.forecasting.base import ForecastingHorizon

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.pipelines.components.utils import convert_bool_to_double, match_indices
from evalml.problem_types import ProblemTypes
from evalml.utils import (
    import_or_raise,
    infer_feature_types,
)


class VARMAXRegressor(Estimator):
    """Vector Autoregressive Moving Average with eXogenous regressors model. The two parameters (p, q) are the AR order and the MA order. More information here: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.varmax.VARMAX.html.

    Currently VARMAXRegressor isn't supported via conda install. It's recommended that it be installed via PyPI.

    Args:
        time_index (str): Specifies the name of the column in X that provides the datetime objects. Defaults to None.
        p (int): Maximum Autoregressive order. Defaults to 1.
        q (int): Maximum Moving Average order. Defaults to 0.
        trend (str): Controls the deterministic trend. Options are ['n', 'c', 't', 'ct'] where 'c' is a constant term,
            't' indicates a linear trend, and 'ct' is both. Can also be an iterable when defining a polynomial, such
            as [1, 1, 0, 1].
        random_seed (int): Seed for the random number generator. Defaults to 0.
        max_iter (int): Maximum number of iterations for solver. Defaults to 10.
        use_covariates (bool): If True, will pass exogenous variables in fit/predict methods. If False, forecasts will
            solely be based off of the datetimes and target values. Defaults to True.
    """

    name = "VARMAX Regressor"
    hyperparameter_ranges = {
        "p": Integer(0, 10),
        "q": Integer(0, 10),
        "trend": Categorical(["n", "c", "t", "ct"]),
    }
    """{
        "p": Integer(1, 10),
        "q": Integer(1, 10),
        "trend": Categorical(['n', 'c', 't', 'ct']),
    }"""
    model_family = ModelFamily.VARMAX
    is_multiseries = True
    """ModelFamily.VARMAX"""
    supported_problem_types = [ProblemTypes.TIME_SERIES_REGRESSION]
    """[ProblemTypes.TIME_SERIES_REGRESSION]"""

    def __init__(
        self,
        time_index: Optional[Hashable] = None,
        p: int = 1,
        q: int = 0,
        trend: Optional[str] = "c",
        random_seed: Union[int, float] = 0,
        maxiter: int = 10,
        use_covariates: bool = True,
        **kwargs,
    ):
        self.preds_95_upper = None
        self.preds_95_lower = None
        parameters = {
            "order": (p, q),
            "trend": trend,
            "maxiter": maxiter,
        }
        parameters.update(kwargs)

        varmax_model_msg = (
            "sktime is not installed. Please install using `pip install sktime.`"
        )
        sktime_varmax = import_or_raise(
            "sktime.forecasting.varmax",
            error_msg=varmax_model_msg,
        )
        varmax_model = sktime_varmax.VARMAX(**parameters)

        parameters["use_covariates"] = use_covariates
        parameters["time_index"] = time_index

        self.use_covariates = use_covariates
        self.time_index = time_index

        super().__init__(
            parameters=parameters,
            component_obj=varmax_model,
            random_seed=random_seed,
        )

    def _set_forecast_horizon(self, X: pd.DataFrame):
        # we can only calculate the difference if the indices are of the same type
        units_diff = 1
        if isinstance(X.index[0], type(self.last_X_index)):
            if isinstance(
                X.index,
                pd.DatetimeIndex,
            ):
                dates_diff = pd.date_range(
                    start=self.last_X_index,
                    end=X.index[0],
                    freq=X.index.freq,
                )
                units_diff = len(dates_diff) - 1
            elif X.index.is_numeric():
                units_diff = X.index[0] - self.last_X_index
        fh_ = ForecastingHorizon(
            [units_diff + i for i in range(len(X))],
            is_relative=True,
        )
        return fh_

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        """Fits VARMAX regressor to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.DataFrane): The target training data of shape [n_samples, n_series_id_values].

        Returns:
            self

        Raises:
            ValueError: If y was not passed in.
        """
        X, y = self._manage_woodwork(X, y)

        if y is None:
            raise ValueError("VARMAX Regressor requires y as input.")

        if X is not None and self.use_covariates:
            self.last_X_index = X.index[-1]
            X = X.ww.select(exclude=["Datetime"])

            X = convert_bool_to_double(X)
            y = convert_bool_to_double(y)
            X, y = match_indices(X, y)

            if not X.empty:
                self._component_obj.fit(y=y, X=X)
            else:
                self._component_obj.fit(y=y)
        else:
            self.last_X_index = y.index[-1]
            self._component_obj.fit(y=y)
        return self

    def _manage_types_and_forecast(self, X: pd.DataFrame) -> tuple:
        fh_ = self._set_forecast_horizon(X)
        X = X.ww.select(exclude=["Datetime"])
        X = convert_bool_to_double(X)
        return X, fh_

    def predict(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.Series:
        """Make predictions using fitted VARMAX regressor.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].
            y (pd.DataFrame): Target data of shape [n_samples, n_series_id_values].

        Returns:
            pd.Series: Predicted values.

        Raises:
            ValueError: If X was passed to `fit` but not passed in `predict`.
        """
        X, y = self._manage_woodwork(X, y)
        X, fh_ = self._manage_types_and_forecast(X=X)

        if not X.empty and self.use_covariates:
            if fh_[0] != 1:
                # statsmodels (which sktime uses under the hood) only forecasts off the training data
                # but sktime circumvents this by predicting everything from the end of training data to the date / periods requested
                # and only returning the values for dates / periods given to sktime. Because of this,
                # pmdarima requires the number of covariate rows to equal the length of the total number of periods (X.shape[0] == fh_[-1]) if covariates are used.
                # We circument this by adding arbitrary rows to the start of X since sktime discards these values when predicting.
                num_rows_diff = fh_[-1] - X.shape[0]
                filler = pd.DataFrame(
                    columns=X.columns,
                    index=range(num_rows_diff),
                ).fillna(0)
                X_ = pd.concat([filler, X], ignore_index=True)
                X_.ww.init(schema=X.ww.schema)
            else:
                X_ = X
            y_pred = self._component_obj.predict(
                fh=fh_,
                X=X_,
            )
        else:
            y_pred = self._component_obj.predict(
                fh=fh_,
            )
        y_pred.index = X.index
        return infer_feature_types(y_pred)

    def get_prediction_intervals(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None,
        coverage: List[float] = None,
        predictions: pd.Series = None,
    ) -> Dict[str, pd.Series]:
        """Find the prediction intervals using the fitted VARMAXRegressor.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].
            y (pd.DataFrame): Target data of shape [n_samples, n_series_id_values]. Optional.
            coverage (list[float]): A list of floats between the values 0 and 1 that the upper and lower bounds of the
                prediction interval should be calculated for.
            predictions (pd.Series): Not used for VARMAX regressor.

        Returns:
            dict: Prediction intervals, keys are in the format {coverage}_lower or {coverage}_upper.
        """
        raise NotImplementedError(
            "VARMAX does not have prediction intervals implemented yet.",
        )

    @property
    def feature_importance(self) -> np.ndarray:
        """Returns array of 0's with a length of 1 as feature_importance is not defined for VARMAX regressor."""
        return np.zeros(1)
