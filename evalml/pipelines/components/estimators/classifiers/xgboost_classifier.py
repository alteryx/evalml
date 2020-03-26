from skopt.space import Integer, Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import import_or_raise


class XGBoostClassifier(Estimator):
    """XGBoost Classifier"""
    name = "XGBoost Classifier"
    hyperparameter_ranges = {
        "eta": Real(0, 1),
        "max_depth": Integer(1, 20),
        "min_child_weight": Real(1, 10),
        "n_estimators": Integer(1, 1000),
    }
    model_family = ModelFamily.XGBOOST
    problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def __init__(self, parameters={}, component_obj=None, random_state=0):
        xgb_error_msg = "XGBoost is not installed. Please install using `pip install xgboost.`"
        xgb = import_or_raise("xgboost", error_msg=xgb_error_msg)
        xgb_classifier = xgb.XGBClassifier(random_state=random_state,
                                           eta=parameters.get('eta', 0.1),
                                           max_depth=parameters.get('max_depth', 3),
                                           n_estimators=parameters.get('n_estimators', 100),
                                           min_child_weight=parameters.get('min_child_weight', 1))
        super().__init__(parameters=parameters,
                         component_obj=xgb_classifier,
                         random_state=random_state)

    @property
    def feature_importances(self):
        return self._component_obj.feature_importances_
