import numpy as np
from sklearn.feature_selection import SelectFromModel as SkSelect
from skopt.space import Real

from .component_types import ComponentTypes
from .transformer import Transformer


# TODO: AutoML must know to pass in the same estimator and number_features
class SelectFromModel(Transformer):
    def __init__(self, estimator, number_features, percent_features=0.5, threshold=-np.inf):
        self.name = 'Select From Model'
        self.component_type = ComponentTypes.FEATURE_SELECTION
        self.hyperparameters = {
            "percent_features": Real(.01, 1),
            "threshold": ['mean', -np.inf]
        }
        self.number_features = number_features
        self.percent_features = percent_features
        self.max_features = max(1, int(percent_features * number_features))
        self.threshold = threshold
        self.feature_selection = SkSelect(
            estimator=estimator,
            max_features=self.max_features,
            threshold=self.threshold
        )

        self.parameters = {"percent_features": self.percent_features, "threshold": self.threshold}
        super().__init__(name=self.name, component_type=self.component_type, hyperparameters=self.hyperparameters, parameters=self.parameters, needs_fitting=True, component_obj=feature_selection)

    def fit_transform(self, X, y, objective_fit_size=.2):
        return self._component_obj.fit_transform(X, y)
