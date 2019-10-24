from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor
from skopt.space import Integer

from evalml.model_types import ModelTypes
from evalml.pipelines.components import ComponentTypes
from evalml.pipelines.components.estimators import Estimator


class RandomForestRegressor(Estimator):
    """Random Forest Regressor"""
    name = "Random Forest Regressor"
    component_type = ComponentTypes.REGRESSOR
    hyperparameter_ranges = {
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 32),
    }
    model_type = ModelTypes.RANDOM_FOREST

    def __init__(self, n_estimators=10, max_depth=None, n_jobs=-1, random_state=0):
        parameters = {"n_estimators": n_estimators,
                      "max_depth": max_depth}
        rf_regressor = SKRandomForestRegressor(random_state=random_state,
                                               n_estimators=n_estimators,
                                               max_depth=max_depth,
                                               n_jobs=n_jobs)
        super().__init__(name=self.name,
                         component_type=self.component_type,
                         parameters=parameters,
                         component_obj=rf_regressor,
                         needs_fitting=True,
                         random_state=random_state)

    @property
    def feature_importances(self):
        return self._component_obj.feature_importances_
