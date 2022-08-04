"""XGBoost Regressor."""
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
        eta=0.1,
        max_depth=6,
        min_child_weight=1,
        n_estimators=100,
        random_seed=0,
        n_jobs=12,
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

    def fit(self, X, y=None):
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

    def predict(self, X):
        """Make predictions using fitted XGBoost regressor.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].

        Returns:
            pd.Series: Predicted values.
        """
        X, _ = super()._manage_woodwork(X)
        X = _rename_column_names_to_numeric(X)
        return super().predict(X)

    @property
    def feature_importance(self):
        """Feature importance of fitted XGBoost regressor."""
        return pd.Series(self._component_obj.feature_importances_)
