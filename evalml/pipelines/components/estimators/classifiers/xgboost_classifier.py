from skopt.space import Integer, Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import (
    _rename_column_names_to_numeric,
    import_or_raise
)


class XGBoostClassifier(Estimator):
    """XGBoost Classifier."""
    name = "XGBoost Classifier"
    hyperparameter_ranges = {
        "eta": Real(0.000001, 1),
        "max_depth": Integer(1, 10),
        "min_child_weight": Real(1, 10),
        "n_estimators": Integer(1, 1000),
    }
    model_family = ModelFamily.XGBOOST
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS,
                               ProblemTypes.TIME_SERIES_BINARY, ProblemTypes.TIME_SERIES_MULTICLASS]

    # xgboost supports seeds from -2**31 to 2**31 - 1 inclusive. these limits ensure the random seed generated below
    # is within that range.
    SEED_MIN = -2**31
    SEED_MAX = 2**31 - 1

    def __init__(self, eta=0.1, max_depth=6, min_child_weight=1, n_estimators=100, random_seed=0, **kwargs):
        parameters = {"eta": eta,
                      "max_depth": max_depth,
                      "min_child_weight": min_child_weight,
                      "n_estimators": n_estimators}
        parameters.update(kwargs)
        xgb_error_msg = "XGBoost is not installed. Please install using `pip install xgboost.`"
        xgb = import_or_raise("xgboost", error_msg=xgb_error_msg)
        xgb_classifier = xgb.XGBClassifier(random_state=random_seed,
                                           **parameters)

        super().__init__(parameters=parameters,
                         component_obj=xgb_classifier,
                         random_seed=random_seed)

    def fit(self, X, y=None):
        X, y = super()._manage_woodwork(X, y)
        self.input_feature_names = list(X.columns)
        X = _rename_column_names_to_numeric(X, flatten_tuples=False)
        self._component_obj.fit(X, y)
        return self

    def predict(self, X):
        X = _rename_column_names_to_numeric(X, flatten_tuples=False)
        return super().predict(X)

    def predict_proba(self, X):
        X = _rename_column_names_to_numeric(X, flatten_tuples=False)
        return super().predict_proba(X)

    @property
    def feature_importance(self):
        return self._component_obj.feature_importances_
