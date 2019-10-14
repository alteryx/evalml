from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from skopt.space import Integer

from evalml.pipelines.components import ComponentTypes
from evalml.pipelines.components.estimator import Estimator


class RandomForestClassifier(Estimator):
    """Random Forest Classifier"""

    hyperparameters = {
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 32),
    }

    def __init__(self, n_estimators, max_depth=None, n_jobs=-1, random_state=0):
        self.name = "Random Forest Classifier"
        self.component_type = ComponentTypes.CLASSIFIER
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._component_obj = SKRandomForestClassifier(random_state=self.random_state,
                                                       n_estimators=self.n_estimators,
                                                       max_depth=self.max_depth,
                                                       n_jobs=self.n_jobs)
        self.parameters = {"n_estimators": self.n_estimators, "max_depth": self.max_depth}
        super().__init__(name=self.name, component_type=self.component_type, parameters=self.parameters, component_obj=self._component_obj)
