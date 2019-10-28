import numpy as np
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from sklearn.feature_selection import SelectFromModel as SkSelect
from skopt.space import Real

from .feature_selector import FeatureSelector

from evalml.pipelines.components import ComponentTypes


class RFClassifierSelectFromModel(FeatureSelector):
    """Selects top features based on importance weights using a Random Forest classifier"""
    name = 'RF Select From Model'
    component_type = ComponentTypes.FEATURE_SELECTION_CLASSIFIER
    hyperparameter_ranges = {
        "percent_features": Real(.01, 1),
        "threshold": ['mean', -np.inf]
    }

    def __init__(self, number_features=None, n_estimators=10, max_depth=None,
                 percent_features=0.5, threshold=-np.inf, random_state=0):
        max_features = None
        if number_features:
            max_features = max(1, int(percent_features * number_features))
        parameters = {"percent_features": percent_features,
                      "threshold": threshold}
        estimator = SKRandomForestClassifier(random_state=random_state,
                                             n_estimators=n_estimators,
                                             max_depth=max_depth)
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
                         random_state=random_state)
