"""Vector Autoregressive Moving Average with eXogenous regressors model. The three parameters (p, q) are the AR order and the MA order. More information here: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.varmax.VARMAX.html#statsmodels.tsa.statespace.varmax.VARMAX."""
from typing import Dict, Hashable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype
from skopt.space import Categorical, Integer

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import (
    import_or_raise,
    infer_feature_types,
)


class VARMAXRegressor(Estimator):
    """Vector Autoregressive Moving Average with eXogenous regressors model. The three parameters (p, q) are the AR order and the MA order. More information here: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.varmax.VARMAX.html#statsmodels.tsa.statespace.varmax.VARMAX.

    Currently VARMAXRegressor isn't supported via conda install. It's recommended that it be installed via PyPI.

    Args:
        time_index (str): Specifies the name of the column in X that provides the datetime objects. Defaults to None.
        p (int): Maximum Autoregressive order. Defaults to 5.
        q (int): Maximum Moving Average order. Defaults to 5.
        trend (str): Controls the deterministic trend. Options are ['n', 'c', 't', 'ct'] where 'c' is a constant term,
            't' indicates a linear trend, and 'ct' is both. Can also be an iterable when defining a polynomial, such
            as [1, 1, 0, 1].
        random_seed (int): Seed for the random number generator. Defaults to 0.
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
        series_id: Optional[Hashable] = None,
        p: int = 1,
        q: int = 0,
        trend: Optional[str] = None,
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
        parameters["series_id"] = series_id

        self.use_covariates = use_covariates
        self.time_index = time_index
        self.series_id = series_id

        super().__init__(
            parameters=parameters,
            component_obj=varmax_model,
            random_seed=random_seed,
        )

    def _remove_datetime_and_series_id(
        self,
        data: pd.DataFrame,
        features: bool = False,
    ) -> pd.DataFrame:
        if data is None:
            return None
        data_no_dt_and_series_id = data.ww.copy()
        if isinstance(
            data_no_dt_and_series_id.index,
            (pd.DatetimeIndex, pd.PeriodIndex, pd.IntervalIndex),
        ):
            data_no_dt_and_series_id = data_no_dt_and_series_id.ww.reset_index(
                drop=True,
            )
        if features:
            data_no_dt_and_series_id = data_no_dt_and_series_id.ww.select(
                exclude=["Datetime"],
            )
        data_no_dt_and_series_id = self._drop_series_id_column(data_no_dt_and_series_id)
        return data_no_dt_and_series_id

    def _drop_series_id_column(self, data: pd.DataFrame):
        if data is None or not isinstance(data, pd.DataFrame):
            return None
        data_no_dt_and_series_id = data.ww.copy()
        if self.series_id in data_no_dt_and_series_id.columns:
            return data_no_dt_and_series_id.ww.drop(self.series_id)
        else:
            return data_no_dt_and_series_id

    def _match_indices(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if X is not None:
            if X.index.equals(y.index):
                return X, y
            else:
                y.index = X.index
        return X, y

    def _set_forecast(self, X: pd.DataFrame):
        from sktime.forecasting.base import ForecastingHorizon

        # we can only calculate the difference if the indices are of the same type
        units_diff = 1
        if isinstance(X.index[0], type(self.last_X_index)) and isinstance(
            X.index,
            pd.DatetimeIndex,
        ):
            dates_diff = pd.date_range(
                start=self.last_X_index,
                end=X.index[0],
                freq=X.index.freq,
            )
            units_diff = len(dates_diff) - 1
        elif is_integer_dtype(type(X.index[0])) and is_integer_dtype(
            type(self.last_X_index),
        ):
            units_diff = X.index[0] - self.last_X_index
        fh_ = ForecastingHorizon(
            [units_diff + i for i in range(len(X))],
            is_relative=True,
        )
        return fh_

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fits VARMAX regressor to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series): The target training data of length [n_samples].

        Returns:
            self

        Raises:
            ValueError: If y was not passed in.
        """
        X, y = self._manage_woodwork(X, y)
        if X is not None:
            X = X.ww.fillna(X.mean())
        if y is None:
            raise ValueError("VARMAX Regressor requires y as input.")

        y = self._unstack_target_data(X, y, self.time_index, self.series_id)
        y.ww.init()

        self.last_X_index = X.index[-1] if X is not None else y.index[-1]

        X = self._remove_datetime_and_series_id(X, features=True)

        if X is not None:
            X.ww.set_types(
                {
                    col: "Double"
                    for col in X.ww.select(["Boolean"], return_schema=True).columns
                },
            )
        y = self._remove_datetime_and_series_id(y)
        # X, y = self._match_indices(X, y)

        if X is not None and not X.empty and self.use_covariates:
            self._component_obj.fit(y=y, X=X)
        else:
            self._component_obj.fit(y=y)
        return self

    def _manage_types_and_forecast(self, X: pd.DataFrame) -> tuple:
        fh_ = self._set_forecast(X)
        X = X.ww.select(exclude=["Datetime"])
        X.ww.set_types(
            {
                col: "Double"
                for col in X.ww.select(["Boolean"], return_schema=True).columns
            },
        )
        X = self._drop_series_id_column(X)
        return X, fh_

    def predict(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.Series:
        """Make predictions using fitted VARMAX regressor.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].
            y (pd.Series): Target data.

        Returns:
            pd.Series: Predicted values.

        Raises:
            ValueError: If X was passed to `fit` but not passed in `predict`.
        """
        X, y = self._manage_woodwork(X, y)
        X, fh_ = self._manage_types_and_forecast(X=X)

        if not X.empty and self.use_covariates:
            # if fh_[0] != 1:
            #     # pmdarima (which sktime uses under the hood) only forecasts off the training data
            #     # but sktime circumvents this by predicting everything from the end of training data to the date / periods requested
            #     # and only returning the values for dates / periods given to sktime. Because of this,
            #     # pmdarima requires the number of covariate rows to equal the length of the total number of periods (X.shape[0] == fh_[-1]) if covariates are used.
            #     # We circument this by adding arbitrary rows to the start of X since sktime discards these values when predicting.
            #     num_rows_diff = fh_[-1] - X.shape[0]
            #     filler = pd.DataFrame(
            #         columns=X.columns,
            #         index=range(num_rows_diff),
            #     ).fillna(0)
            #     X_ = pd.concat([filler, X], ignore_index=True)
            #     X_.ww.init(schema=X.ww.schema)
            # else:
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
        y_pred = self._stack_target_data(y, self.time_index, self.series_id)

        return infer_feature_types(y_pred)

    def get_prediction_intervals(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        coverage: List[float] = None,
        predictions: pd.Series = None,
    ) -> Dict[str, pd.Series]:
        """Find the prediction intervals using the fitted VARMAXRegressor.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].
            y (pd.Series): Target data. Optional.
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

    @staticmethod
    def _unstack_target_data(X, y, time_index, series_id):
        if y is None or isinstance(y, pd.Series):
            return y
        target_name = y.name
        combo = pd.concat([X[series_id], y], axis=1)
        series_id_unique = combo[series_id].unique()
        target_cols = {}
        for col in series_id_unique:
            target_cols[col] = combo[combo[series_id] == col][target_name].reset_index(
                drop=True,
            )
        target_df = pd.concat(target_cols, axis=1)
        return target_df.set_index(X[time_index].unique())

    @staticmethod
    def _stack_target_data(data, time_index, series_id, cols_to_stack=None):
        if data is None or isinstance(data, pd.DataFrame):
            return data
        if cols_to_stack and not isinstance(cols_to_stack, list):
            raise ValueError("cols_to_stack needs to be a list of column names")
        if cols_to_stack:
            data = data[cols_to_stack]
        stacked_series = data.stack(0)
        columns = []
        for i in range(0, len(stacked_series.index[0])):
            columns.append(stacked_series.index.get_level_values(i))
        columns.append(stacked_series.values)
        stacked_target_df = pd.DataFrame(columns).T
        stacked_target_df.columns = [time_index, series_id, "target"]
        stacked_target_df = stacked_target_df.set_index(time_index)
        return stacked_target_df
