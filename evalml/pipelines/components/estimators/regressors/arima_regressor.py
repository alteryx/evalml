"""Autoregressive Integrated Moving Average Model. The three parameters (p, d, q) are the AR order, the degree of differencing, and the MA order. More information here: https://www.statsmodels.org/devel/generated/statsmodels.tsa.arima.model.ARIMA.html."""
from typing import Dict, Hashable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from skopt.space import Integer

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import (
    import_or_raise,
    infer_feature_types,
)


class ARIMARegressor(Estimator):
    """Autoregressive Integrated Moving Average Model. The three parameters (p, d, q) are the AR order, the degree of differencing, and the MA order. More information here: https://www.statsmodels.org/devel/generated/statsmodels.tsa.arima.model.ARIMA.html.

    Currently ARIMARegressor isn't supported via conda install. It's recommended that it be installed via PyPI.

    Args:
        time_index (str): Specifies the name of the column in X that provides the datetime objects. Defaults to None.
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
        sp (int or str): Period for seasonal differencing, specifically the number of periods in each season. If "detect", this
            model will automatically detect this parameter (given the time series is a standard frequency) and will fall
            back to 1 (no seasonality) if it cannot be detected. Defaults to 1.
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

    max_rows = 1000
    max_cols = 7

    def __init__(
        self,
        time_index: Optional[Hashable] = None,
        trend: Optional[str] = None,
        start_p: int = 2,
        d: int = 0,
        start_q: int = 2,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        seasonal: bool = True,
        sp: int = 1,
        n_jobs: int = -1,
        random_seed: Union[int, float] = 0,
        maxiter: int = 10,
        use_covariates: bool = True,
        **kwargs,
    ):
        self.preds_95_upper = None
        self.preds_95_lower = None
        parameters = {
            "trend": trend,
            "start_p": start_p,
            "d": d,
            "start_q": start_q,
            "max_p": max_p,
            "max_d": max_d,
            "max_q": max_q,
            "seasonal": seasonal,
            "maxiter": maxiter,
            "n_jobs": n_jobs,
        }
        parameters.update(kwargs)

        arima_model_msg = (
            "sktime is not installed. Please install using `pip install sktime.`"
        )
        sktime_arima = import_or_raise(
            "sktime.forecasting.arima",
            error_msg=arima_model_msg,
        )
        arima_model = sktime_arima.AutoARIMA(**parameters)

        parameters["use_covariates"] = use_covariates
        parameters["time_index"] = time_index

        self.sp = sp
        self.use_covariates = use_covariates

        super().__init__(
            parameters=parameters,
            component_obj=arima_model,
            random_seed=random_seed,
        )

    def _remove_datetime(
        self,
        data: pd.DataFrame,
        features: bool = False,
    ) -> pd.DataFrame:
        if data is None:
            return None
        data_no_dt = data.ww.copy()
        if isinstance(
            data_no_dt.index,
            (pd.DatetimeIndex, pd.PeriodIndex, pd.IntervalIndex),
        ):
            data_no_dt = data_no_dt.ww.reset_index(drop=True)
        if features:
            data_no_dt = data_no_dt.ww.select(exclude=["Datetime"])

        return data_no_dt

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
        if isinstance(X.index[0], type(self.last_X_index)):
            units_diff = X.index[0] - self.last_X_index
            if isinstance(X.index, pd.DatetimeIndex):
                dates_diff = pd.date_range(
                    start=self.last_X_index,
                    end=X.index[0],
                    freq=X.index.freq,
                )
                units_diff = len(dates_diff) - 1
            fh_ = ForecastingHorizon(
                [units_diff + i for i in range(len(X))],
                is_relative=True,
            )
        else:
            fh_ = ForecastingHorizon(
                [i + 1 for i in range(len(X))],
                is_relative=True,
            )
        return fh_

    def _get_sp(self, X: pd.DataFrame) -> int:
        if X is None:
            return 1
        freq_mappings = {
            "D": 7,
            "M": 12,
            "Q": 4,
        }
        time_index = self._parameters.get("time_index", None)
        sp = self.sp
        if sp == "detect":
            inferred_freqs = X.ww.infer_temporal_frequencies()
            freq = inferred_freqs.get(time_index, None)
            sp = 1
            if freq is not None:
                sp = freq_mappings.get(freq[:1], 1)
        return sp

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fits ARIMA regressor to data.

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
            raise ValueError("ARIMA Regressor requires y as input.")

        sp = self._get_sp(X)
        self._component_obj.sp = sp
        self.last_X_index = X.index[-1] if X is not None else y.index

        X = self._remove_datetime(X, features=True)

        if X is not None:
            X.ww.set_types(
                {
                    col: "Double"
                    for col in X.ww.select(["Boolean"], return_schema=True).columns
                },
            )
        y = self._remove_datetime(y)
        X, y = self._match_indices(X, y)

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
        return X, fh_

    @staticmethod
    def _parse_prediction_intervals(
        y_pred_intervals: pd.DataFrame,
        conf_int: float,
    ) -> Tuple[pd.Series, pd.Series]:
        preds_lower = y_pred_intervals.loc(axis=1)[("Coverage", conf_int, "lower")]
        preds_upper = y_pred_intervals.loc(axis=1)[("Coverage", conf_int, "upper")]
        preds_lower.name = None
        preds_upper.name = None
        return preds_lower, preds_upper

    def predict(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.Series:
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
        X, fh_ = self._manage_types_and_forecast(X=X)

        if not X.empty and self.use_covariates:
            if fh_[0] != 1:
                # pmdarima (which sktime uses under the hood) only forecasts off the training data
                # but sktime circumvents this by predicting everything from the end of training data to the date / periods requested
                # and only returning the values for dates / periods given to sktime. Because of this,
                # pmdarima requires the number of covariate rows to equal the length of the total number of periods (X.shape[0] == fh_[-1]) if covariates are used.
                # We circument this by adding arbitrary rows to the start of X since sktime discards these values when predicting.
                num_rows_diff = fh_[-1] - X.shape[0]
                X_ = pd.concat([X.head(num_rows_diff), X], ignore_index=True)
            else:
                X_ = X
            y_pred_intervals = self._component_obj.predict_interval(
                fh=fh_,
                X=X_,
                coverage=[0.95],
            )
        else:
            y_pred_intervals = self._component_obj.predict_interval(
                fh=fh_,
                coverage=[0.95],
            )
        y_pred_intervals.index = X.index

        (
            self.preds_95_lower,
            self.preds_95_upper,
        ) = ARIMARegressor._parse_prediction_intervals(y_pred_intervals, 0.95)

        y_pred = pd.concat((self.preds_95_lower, self.preds_95_upper), axis=1).mean(
            axis=1,
        )

        return infer_feature_types(y_pred)

    def get_prediction_intervals(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        coverage: List[float] = None,
        predictions: pd.Series = None,
    ) -> Dict[str, pd.Series]:
        """Find the prediction intervals using the fitted ARIMARegressor.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].
            y (pd.Series): Target data. Optional.
            coverage (list[float]): A list of floats between the values 0 and 1 that the upper and lower bounds of the
                prediction interval should be calculated for.
            predictions (pd.Series): Not used for ARIMA regressor.

        Returns:
            dict: Prediction intervals, keys are in the format {coverage}_lower or {coverage}_upper.
        """
        if coverage is None:
            coverage = [0.95]
        X, y = self._manage_woodwork(X, y)
        X, fh_ = self._manage_types_and_forecast(X=X)

        prediction_interval_result = {}

        if not X.empty and self.use_covariates:
            y_pred_intervals = self._component_obj.predict_interval(
                fh=fh_,
                X=X,
                coverage=coverage,
            )
        else:
            y_pred_intervals = self._component_obj.predict_interval(
                fh=fh_,
                coverage=coverage,
            )
        y_pred_intervals.index = X.index

        for conf_int in coverage:
            if (
                conf_int == 0.95
                and self.preds_95_lower is not None
                and self.preds_95_upper is not None
            ):
                prediction_interval_result[f"{conf_int}_lower"] = self.preds_95_lower
                prediction_interval_result[f"{conf_int}_upper"] = self.preds_95_upper
                continue
            preds_lower, preds_upper = ARIMARegressor._parse_prediction_intervals(
                y_pred_intervals,
                conf_int,
            )
            prediction_interval_result[f"{conf_int}_lower"] = preds_lower
            prediction_interval_result[f"{conf_int}_upper"] = preds_upper

        return prediction_interval_result

    @property
    def feature_importance(self) -> np.ndarray:
        """Returns array of 0's with a length of 1 as feature_importance is not defined for ARIMA regressor."""
        return np.zeros(1)
