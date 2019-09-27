from sklearn.preprocessing import StandardScaler as SkScaler

from .component_types import ComponentTypes
from .transformer import Transformer


class StandardScaler(Transformer):
    def __init__(self):
        name = 'Standard Scaler'
        component_type = ComponentTypes.SCALER
        hyperparameters = None

        scaler = SkScaler()
        super().__init__(name=name, component_type=component_type, hyperparameters=hyperparameters, needs_fitting=True, component_obj=scaler)

    def fit(self, X, objective_fit_size=.2):
        self._component_obj.fit(X)

    def transform(self, X):
        self._component_obj.transform(X)

    def fit_transform(self, X, y, objective_fit_size=.2):
        self._component_obj.fit_transform(X)
