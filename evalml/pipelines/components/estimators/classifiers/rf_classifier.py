from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from skopt.space import Integer

from evalml.model_types import ModelTypes
from evalml.pipelines.components import ComponentTypes
from evalml.pipelines.components.estimators import Estimator


class RandomForestClassifier(Estimator):
    """Random Forest Classifier"""

    hyperparameter_ranges = {
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 32),
    }
    model_type = ModelTypes.RANDOM_FOREST

    def __init__(self, n_estimators, max_depth=None, n_jobs=-1, random_state=0):
        name = "Random Forest Classifier"
        component_type = ComponentTypes.CLASSIFIER
        parameters = {"n_estimators": n_estimators,
                      "max_depth": max_depth}
        rf_classifier = SKRandomForestClassifier(random_state=random_state,
                                                 n_estimators=n_estimators,
                                                 max_depth=max_depth,
                                                 n_jobs=n_jobs)
        n_jobs = n_jobs
        random_state = random_state
        super().__init__(name=name,
                         component_type=component_type,
                         parameters=parameters,
                         component_obj=rf_classifier,
                         needs_fitting=True,
                         random_state=0)
