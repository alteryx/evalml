from .transformer import Transformer

import numpy as np
from sklearn.feature_selection import SelectFromModel as SkSelect
from skopt.space import Integer, Real

# TODO: AutoML must know to pass in the same estimator and number_features
class SelectFromModel(Transformer):
    def __init__(self, estimator, number_features, percent_features=0.5, threshold=-np.inf):
        name = 'Select From Model'
        component_type = 'feature_selection'
        hyperparameters = {
            "percent_features": Real(.01, 1),
            "threshold": ['mean', -np.inf]
        }

        feature_selection = SkSelect(
            estimator=estimator,
            max_features=max(1, int(percent_features * number_features)),
            threshold=threshold
        )

        super().__init__(name=name, component_type=component_type, hyperparameters=hyperparameters, needs_fitting=True, component_obj=feature_selection)

    def fit(self, X, objective_fit_size=.2):
        self.component_obj.fit(X)

    def transform(self, X):
        self.component_obj.transform(x)

    def fit_transform(self, X, y, objective_fit_size=.2):
        self.component_obj.fit_transform(x)
