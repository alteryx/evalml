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
    problem_types = [ProblemTypes.REGRESSION]

    def __init__(self, parameters={}, component_obj=None, random_state=0):
        rf_regressor = SKRandomForestRegressor(random_state=random_state,
                                               n_estimators=parameters.get('n_estimators', 10),
                                               max_depth=parameters.get('max_depth', None),
                                               n_jobs=parameters.get('n_jobs', -1))
        super().__init__(parameters=parameters,
                         component_obj=rf_regressor,
                         random_state=random_state)

    @property
    def feature_importances(self):
        return self._component_obj.feature_importances_
