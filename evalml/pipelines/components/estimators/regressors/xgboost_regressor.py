import pandas as pd
from skopt.space import Integer, Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import get_random_seed, import_or_raise
from evalml.utils.gen_utils import _rename_column_names_to_numeric


class XGBoostRegressor(Estimator):
    """XGBoost Regressor."""
    name = "XGBoost Regressor"
    hyperparameter_ranges = {
        "eta": Real(0.000001, 1),
        "max_depth": Integer(1, 20),
        "min_child_weight": Real(1, 10),
        "n_estimators": Integer(1, 1000),
    }
    model_family = ModelFamily.XGBOOST
    supported_problem_types = [ProblemTypes.REGRESSION]

    # xgboost supports seeds from -2**31 to 2**31 - 1 inclusive. these limits ensure the random seed generated below
    # is within that range.
    SEED_MIN = -2**31
    SEED_MAX = 2**31 - 1

    def __init__(self, eta=0.1, max_depth=6, min_child_weight=1, n_estimators=100, random_state=0, **kwargs):
        random_seed = get_random_seed(random_state, self.SEED_MIN, self.SEED_MAX)
        parameters = {"eta": eta,
                      "max_depth": max_depth,
                      "min_child_weight": min_child_weight,
                      "n_estimators": n_estimators}
        parameters.update(kwargs)

        xgb_error_msg = "XGBoost is not installed. Please install using `pip install xgboost.`"
        xgb = import_or_raise("xgboost", error_msg=xgb_error_msg)
        xgb_Regressor = xgb.XGBRegressor(random_state=random_seed,
                                         **parameters)
        super().__init__(parameters=parameters,
                         component_obj=xgb_Regressor,
                         random_state=random_state)

    def fit(self, X, y=None):
        # rename column names to column number if input is a pd.DataFrame in case it has column names that contain symbols ([, ], <) that XGBoost cannot properly handle
        if isinstance(X, pd.DataFrame):
            X = _rename_column_names_to_numeric(X)
        return super().fit(X, y)

    def predict(self, X):
        # rename column names to column number if input is a pd.DataFrame in case it has column names that contain symbols ([, ], <) that XGBoost cannot properly handle
        if isinstance(X, pd.DataFrame):
            X = _rename_column_names_to_numeric(X)
        predictions = super().predict(X)
        return predictions

    @property
    def feature_importance(self):
        return self._component_obj.feature_importances_
