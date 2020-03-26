import numpy as np
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from sklearn.feature_selection import SelectFromModel as SkSelect
from skopt.space import Real

from .feature_selector import FeatureSelector


class RFClassifierSelectFromModel(FeatureSelector):
    """Selects top features based on importance weights using a Random Forest classifier"""
    name = 'RF Classifier Select From Model'
    hyperparameter_ranges = {
        "percent_features": Real(.01, 1),
        "threshold": ['mean', -np.inf]
    }

    def __init__(self, parameters={}, component_obj=None, random_state=0):
        max_features = parameters.get('number_features') and \
            max(1, int(parameters.get('percent_features', 0.5) * parameters.get('number_features')))
        estimator = SKRandomForestClassifier(random_state=random_state,
                                             n_estimators=parameters.get('n_estimators', 10),
                                             max_depth=parameters.get('max_depth', None),
                                             n_jobs=parameters.get('n_jobs', -1))
        feature_selection = SkSelect(estimator=estimator,
                                     max_features=max_features,
                                     threshold=parameters.get('threshold', -np.inf))
        super().__init__(parameters=parameters,
                         component_obj=feature_selection,
                         random_state=random_state)
