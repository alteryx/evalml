from skopt.space import Integer, Real
from xgboost import XGBClassifier

from evalml.model_types import ModelTypes
from evalml.pipelines.components import ComponentTypes
from evalml.pipelines.components.estimators import Estimator


class XGBoostClassifier(Estimator):
    """XGBoost Classifier"""
    name = "XGBoost Classifier"
    component_type = ComponentTypes.CLASSIFIER
    hyperparameter_ranges = {
        "eta": Real(0, 1),
        "max_depth": Integer(1, 20),
        "min_child_weight": Real(1, 10),
    }
    model_type = ModelTypes.XGBOOST

    def __init__(self, eta=0.1, max_depth=3, min_child_weight=1, random_state=0):
        parameters = {"eta": eta,
                      "max_depth": max_depth,
                      "min_child_weight": min_child_weight}
        xgb_classifier = XGBClassifier(random_state=random_state,
                                       eta=eta,
                                       max_depth=max_depth,
                                       min_child_weight=min_child_weight)
        super().__init__(name=self.name,
                         component_type=self.component_type,
                         parameters=parameters,
                         component_obj=xgb_classifier,
                         needs_fitting=True,
                         random_state=random_state)

    @property
    def feature_importances(self):
        return self._component_obj.feature_importances_
