import numpy as np
from catboost import CatBoostClassifier as CBClassifier
from evalml.model_types import ModelTypes
from evalml.pipelines.components import ComponentTypes
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class CatBoostClassifier(Estimator):
    """
    CatBoost Classifier
    """
    name = "CatBoost Classifier"
    component_type = ComponentTypes.CLASSIFIER
    _needs_fitting = True
    hyperparameter_ranges = {

    }
    model_type = ModelTypes.CATBOOST
    problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def __init__(self, n_jobs=-1, random_state=0):
        parameters = {}
        cb_classifier = CBClassifier(logging_level="Silent",
                                     random_state=random_state)
        super().__init__(parameters=parameters,
                         component_obj=cb_classifier,
                         random_state=random_state)

    @property
    def feature_importances(self):
        return self._component_obj.get_feature_importance()
    