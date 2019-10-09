from .component_base import ComponentBase


class Transformer(ComponentBase):
    def __init__(self, name, component_type, hyperparameters={}, parameters={}, needs_fitting=False, component_obj=None):
        super().__init__(name=name, component_type=component_type, hyperparameters=hyperparameters, parameters=parameters, needs_fitting=needs_fitting, component_obj=component_obj)

    def fit(self, X, objective_fit_size=.2):
        self._component_obj.fit(X)

    def transform(self, X):
        return self._component_obj.transform(X)

    def fit_transform(self, X, objective_fit_size=.2):
        return self._component_obj.fit_transform(X)
