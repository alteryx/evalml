from skopt.space import Integer, Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import (
    _rename_column_names_to_numeric,
    import_or_raise,
)


class XGBoostClassifier(Estimator):
    """
    XGBoost Classifier.

    Arguments:
        eta (float): Boosting learning rate. Defaults to 0.1.
        max_depth (int): Maximum tree depth for base learners. Defaults to 6.
        min_child_weight (float): Minimum sum of instance weight (hessian) needed in a child. Defaults to 1.0
        n_estimators (int): Number of gradient boosted trees. Equivalent to number of boosting rounds. Defaults to 100.
        random_seed (int): Seed for the random number generator. Defaults to 0.
        n_jobs (int): Number of parallel threads used to run xgboost. Note that creating thread contention will significantly slow down the algorithm. Defaults to -1.
    """

    name = "XGBoost Classifier"
    hyperparameter_ranges = {
        "eta": Real(0.000001, 1),
        "max_depth": Integer(1, 10),
        "min_child_weight": Real(1, 10),
        "n_estimators": Integer(1, 1000),
    }
    """{
        "eta": Real(0.000001, 1),
        "max_depth": Integer(1, 10),
        "min_child_weight": Real(1, 10),
        "n_estimators": Integer(1, 1000),
    }"""
    model_family = ModelFamily.XGBOOST
    """ModelFamily.XGBOOST"""
    supported_problem_types = [
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]
    """[
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]"""

    # xgboost supports seeds from -2**31 to 2**31 - 1 inclusive. these limits ensure the random seed generated below
    # is within that range.
    SEED_MIN = -(2 ** 31)
    SEED_MAX = 2 ** 31 - 1

    def __init__(
        self,
        eta=0.1,
        max_depth=6,
        min_child_weight=1,
        n_estimators=100,
        random_seed=0,
        n_jobs=-1,
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
        xgb_classifier = xgb.XGBClassifier(random_state=random_seed, **parameters)
        super().__init__(
            parameters=parameters, component_obj=xgb_classifier, random_seed=random_seed
        )

    @staticmethod
    def _convert_bool_to_int(X):
        return {
            col: "Integer" for col in X.ww.select("boolean", return_schema=True).columns
        }

    def fit(self, X, y=None):
        X, y = super()._manage_woodwork(X, y)
        X.ww.set_types(self._convert_bool_to_int(X))
        self.input_feature_names = list(X.columns)
        X = _rename_column_names_to_numeric(X, flatten_tuples=False)
        self._component_obj.fit(X, y)
        return self

    def predict(self, X):
        X, _ = super()._manage_woodwork(X)
        X.ww.set_types(self._convert_bool_to_int(X))
        X = _rename_column_names_to_numeric(X, flatten_tuples=False)
        return super().predict(X)

    def predict_proba(self, X):
        X, _ = super()._manage_woodwork(X)
        X.ww.set_types(self._convert_bool_to_int(X))
        X = _rename_column_names_to_numeric(X, flatten_tuples=False)
        return super().predict_proba(X)

    @property
    def feature_importance(self):
        return self._component_obj.feature_importances_
