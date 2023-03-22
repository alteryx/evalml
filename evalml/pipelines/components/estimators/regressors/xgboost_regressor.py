"""XGBoost Regressor."""
from typing import Dict, List, Optional, Union

import pandas as pd
from skopt.space import Integer, Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import _rename_column_names_to_numeric, import_or_raise


class XGBoostRegressor(Estimator):
    """XGBoost Regressor.

    Args:
        eta (float): Boosting learning rate. Defaults to 0.1.
        max_depth (int): Maximum tree depth for base learners. Defaults to 6.
        min_child_weight (float): Minimum sum of instance weight (hessian) needed in a child. Defaults to 1.0
        n_estimators (int): Number of gradient boosted trees. Equivalent to number of boosting rounds. Defaults to 100.
        random_seed (int): Seed for the random number generator. Defaults to 0.
        n_jobs (int): Number of parallel threads used to run xgboost. Note that creating thread contention will significantly slow down the algorithm. Defaults to 12.
    """

    name = "XGBoost Regressor"
    hyperparameter_ranges = {
        "eta": Real(0.000001, 1),
        "max_depth": Integer(1, 20),
        "min_child_weight": Real(1, 10),
        "n_estimators": Integer(1, 1000),
    }
    """{
        "eta": Real(0.000001, 1),
        "max_depth": Integer(1, 20),
        "min_child_weight": Real(1, 10),
        "n_estimators": Integer(1, 1000),
    }"""
    model_family = ModelFamily.XGBOOST
    """ModelFamily.XGBOOST"""
    supported_problem_types = [
        ProblemTypes.REGRESSION,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ]
    """[
        ProblemTypes.REGRESSION,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ]"""

    # xgboost supports seeds from -2**31 to 2**31 - 1 inclusive. these limits ensure the random seed generated below
    # is within that range.
    SEED_MIN = -(2**31)
    SEED_MAX = 2**31 - 1

    def __init__(
        self,
        eta: float = 0.1,
        max_depth: int = 6,
        min_child_weight: int = 1,
        n_estimators: int = 100,
        random_seed: Union[int, float] = 0,
        n_jobs: int = 12,
        **kwargs,
    ):
        parameters = {
            "eta": eta,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "n_estimators": n_estimators,
            "n_jobs": n_jobs,
        }
        parameters.update(kwargs)

        xgb_error_msg = (
            "XGBoost is not installed. Please install using `pip install xgboost.`"
        )
        xgb = import_or_raise("xgboost", error_msg=xgb_error_msg)
        xgb_regressor = xgb.XGBRegressor(random_state=random_seed, **parameters)
        super().__init__(
            parameters=parameters,
            component_obj=xgb_regressor,
            random_seed=random_seed,
        )

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fits XGBoost regressor component to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            self
        """
        X, y = super()._manage_woodwork(X, y)
        self.input_feature_names = list(X.columns)
        X = _rename_column_names_to_numeric(X)
        self._component_obj.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions using fitted XGBoost regressor.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].

        Returns:
            pd.Series: Predicted values.
        """
        X, _ = super()._manage_woodwork(X)
        X = _rename_column_names_to_numeric(X)
        return super().predict(X)

    def get_prediction_intervals(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        coverage: List[float] = None,
        predictions: pd.Series = None,
    ) -> Dict[str, pd.Series]:
        """Find the prediction intervals using the fitted XGBoostRegressor.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].
            y (pd.Series): Target data. Ignored.
            coverage (List[float]): A list of floats between the values 0 and 1 that the upper and lower bounds of the
                prediction interval should be calculated for.
            predictions (pd.Series): Optional list of predictions to use. If None, will generate predictions using `X`.

        Returns:
            dict: Prediction intervals, keys are in the format {coverage}_lower or {coverage}_upper.
        """
        X = _rename_column_names_to_numeric(X)
        prediction_interval_result = super().get_prediction_intervals(
            X=X,
            y=y,
            coverage=coverage,
            predictions=predictions,
        )
        return prediction_interval_result

    @property
    def feature_importance(self) -> pd.Series:
        """Feature importance of fitted XGBoost regressor."""
        return pd.Series(self._component_obj.feature_importances_)
