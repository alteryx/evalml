from skopt.space import Integer, Real
from xgboost import XGBClassifier

from evalml.model_types import ModelTypes
from evalml.pipelines.components import ComponentTypes
from evalml.pipelines.components.estimators import Estimator


class XGBoostClassifier(Estimator):
    """XGBoost Classifier"""
    hyperparameter_ranges = {
        "eta": Real(0, 1),
        "max_depth": Integer(1, 20),
        "min_child_weight": Real(1, 10),
    }
    model_type = ModelTypes.XGBOOST

    def __init__(self, eta, max_depth, min_child_weight, random_state=0):
        name = "XGBoost Classifier"
        component_type = ComponentTypes.CLASSIFIER
        parameters = {"eta": eta,
                      "max_depth": max_depth,
                      "min_child_weight": min_child_weight}
        xgb_classifier = XGBClassifier(random_state=random_state,
                                       eta=eta,
                                       max_depth=max_depth,
                                       min_child_weight=min_child_weight)
        super().__init__(name=name,
                         component_type=component_type,
                         parameters=parameters,
                         component_obj=xgb_classifier,
                         needs_fitting=True,
                         random_state=0)
