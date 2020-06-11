from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from skopt.space import Integer

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class RandomForestClassifier(Estimator):
    """Random Forest Classifier"""
    name = "Random Forest Classifier"
    hyperparameter_ranges = {
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 10),
    }
    model_family = ModelFamily.RANDOM_FOREST
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def __init__(self, n_estimators=100, max_depth=6, n_jobs=-1, random_state=0):
        parameters = {"n_estimators": n_estimators,
                      "max_depth": max_depth}
        rf_classifier = SKRandomForestClassifier(n_estimators=parameters['n_estimators'],
                                                 max_depth=parameters['max_depth'],
                                                 n_jobs=n_jobs,
                                                 random_state=random_state)
        super().__init__(parameters=parameters,
                         component_obj=rf_classifier,
                         random_state=random_state)

    @property
    def feature_importances(self):
        return self._component_obj.feature_importances_
