from sklearn.ensemble import AdaBoostClassifier as SKAdaBoostClassifier
from skopt.space import Integer, Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class AdaBoostClassifier(Estimator):
    """AdaBoost Classifier."""
    name = "AdaBoost Classifier"
    hyperparameter_ranges = {
        "n_estimators": Integer(10, 1000),
        "learning_rate": Real(0.000001, 1),
    }
    model_family = ModelFamily.ENSEMBLE
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS,
                               ProblemTypes.TIME_SERIES_BINARY, ProblemTypes.TIME_SERIES_MULTICLASS]

    def __init__(self, base_estimator=None, n_estimators=100, learning_rate=1.0, random_state=0, **kwargs):
        parameters = {"base_estimator": base_estimator,
                      "n_estimators": n_estimators,
                      "learning_rate": learning_rate}
        parameters.update(kwargs)

        adaboost_classifier = SKAdaBoostClassifier(random_state=random_state,
                                                   **parameters)
        super().__init__(parameters=parameters,
                         component_obj=adaboost_classifier,
                         random_state=random_state)
