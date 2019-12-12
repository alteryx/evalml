from skopt.space import Integer, Real
from evalml.model_types import ModelTypes
from evalml.pipelines.components import ComponentTypes
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class XGBoostClassifier(Estimator):
    """XGBoost Classifier"""
    name = "XGBoost Classifier"
    component_type = ComponentTypes.CLASSIFIER
    _needs_fitting = True
    hyperparameter_ranges = {
        "eta": Real(0, 1),
        "max_depth": Integer(1, 20),
        "min_child_weight": Real(1, 10),
    }
    model_type = ModelTypes.XGBOOST
    problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def __init__(self, eta=0.1, max_depth=3, min_child_weight=1, random_state=0):
        parameters = {"eta": eta,
                      "max_depth": max_depth,
                      "min_child_weight": min_child_weight}
        xgb_error_msg = "XGBoost is not installed. Please install using `pip install xgboost.`"
        xgb = import_or_raise("xgboost", error_msg=xgb_error_msg)
        xgb_classifier = xgb.XGBClassifier(random_state=random_state,
                                           eta=eta,
                                           max_depth=max_depth,
                                           min_child_weight=min_child_weight)
        super().__init__(parameters=parameters,
                         component_obj=xgb_classifier,
                         random_state=random_state)

    @property
    def feature_importances(self):
        return self._component_obj.feature_importances_
