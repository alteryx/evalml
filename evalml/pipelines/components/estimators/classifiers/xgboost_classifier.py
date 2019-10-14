from skopt.space import Integer, Real
from xgboost import XGBClassifier

from evalml.pipelines.components import ComponentTypes
from evalml.pipelines.components.estimator import Estimator


class XGBoostClassifier(Estimator):
    """XGBoost Classifier"""
    hyperparameters = {
        "eta": Real(0, 1),
        "max_depth": Integer(1, 20),
        "min_child_weight": Real(1, 10),
    }

    def __init__(self, eta, max_depth, min_child_weight, random_state=0, **kwargs):
        self.name = "XGBoost Classifier"
        self.component_type = ComponentTypes.CLASSIFIER
        self.eta = eta
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.random_state = random_state

        self.parameters = {"eta": self.eta, "max_depth": self.max_depth, "min_child_weight": self.min_child_weight}
        self._component_obj = XGBClassifier(random_state=self.random_state,
                                            eta=self.eta,
                                            max_depth=self.max_depth,
                                            min_child_weight=self.min_child_weight)
        super().__init__(name=self.name, component_type=self.component_type, parameters=self.parameters, component_obj=self._component_obj)
