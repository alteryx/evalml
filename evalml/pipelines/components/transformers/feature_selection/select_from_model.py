import numpy as np
from sklearn.feature_selection import SelectFromModel as SkSelect
from skopt.space import Real

from .feature_selector import FeatureSelector

from evalml.pipelines.components import ComponentTypes


class SelectFromModel(FeatureSelector):
    """Selects top features based on importance weights"""
    name = 'Select From Model'
    component_type = ComponentTypes.FEATURE_SELECTION
    hyperparameter_ranges = {
        "percent_features": Real(.01, 1),
        "threshold": ['mean', -np.inf]
    }

    def __init__(self, estimator, number_features, percent_features=0.5, threshold=-np.inf):
        max_features = max(1, int(percent_features * number_features))
        parameters = {"percent_features": percent_features,
                      "threshold": threshold}
        feature_selection = SkSelect(
            estimator=estimator,
            max_features=max_features,
            threshold=threshold
        )
        super().__init__(name=self.name,
                         component_type=self.component_type,
                         parameters=parameters,
                         component_obj=feature_selection,
                         needs_fitting=True,
                         random_state=0)
