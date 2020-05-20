from sklearn.ensemble import ExtraTreesClassifier as SKExtraTreesClassifier
from skopt.space import Integer

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class ExtraTreesClassifier(Estimator):
    """Extra Trees Classifier"""
    name = "Extra Trees Classifier"
    hyperparameter_ranges = {
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 32),
    }
    model_family = ModelFamily.EXTRA_TREES
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def __init__(self, n_estimators=10, max_depth=None, n_jobs=-1, random_state=0):
        parameters = {"n_estimators": n_estimators,
                      "max_depth": max_depth}
        et_classifier = SKExtraTreesClassifier(n_estimators=n_estimators,
                                               max_depth=max_depth,
                                               n_jobs=n_jobs,
                                               random_state=random_state)
        super().__init__(parameters=parameters,
                         component_obj=et_classifier,
                         random_state=random_state)

    @property
    def feature_importances(self):
        return self._component_obj.feature_importances_
