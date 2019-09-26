from .component_base import ComponentBase


class Estimator(ComponentBase):
    def __init__(self, name, component_type, hyperparameters, needs_fitting=False, component_obj=None):
        super().__init__(name, component_type, hyperparameters, needs_fitting, component_obj)

    def fit(self, X, y, objective_fit_size=.2):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

    def score(self, X, y, other_objectives=None):
        pass
