from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor
from skopt.space import Integer

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class RandomForestRegressor(Estimator):
    """Random Forest Regressor"""
    name = "Random Forest Regressor"
    hyperparameter_ranges = {
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 32),
    }
    model_family = ModelFamily.RANDOM_FOREST
    supported_problem_types = [ProblemTypes.REGRESSION]

    def __init__(self, n_estimators=10, max_depth=None, n_jobs=-1, random_state=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        rf_regressor = SKRandomForestRegressor(n_estimators=n_estimators,
                                               max_depth=max_depth,
                                               n_jobs=n_jobs,
                                               random_state=random_state)
        super().__init__(component_obj=rf_regressor,
                         random_state=random_state)

    @property
    def feature_importances(self):
        return self._component_obj.feature_importances_
