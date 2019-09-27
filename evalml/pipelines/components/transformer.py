from .component_base import ComponentBase


class Transformer(ComponentBase):
    def __init__(self, name, component_type, hyperparameters, needs_fitting=False, component_obj=None):
        super().__init__(name, component_type, hyperparameters, needs_fitting, component_obj)

    def fit(self, X, y, objective_fit_size=.2):
        pass

    def fit_transform(self, X, y, objective_fit_size=.2):
        pass
