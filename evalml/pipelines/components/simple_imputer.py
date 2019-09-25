from transformer import Transformer

from sklearn.impute import SimpleImputer

class SimpleImputer(Transformer):
    def __init__(self, impute_strategy="most_frequent", hyperparameters=None):
        name = 'Simple Imputer'
        component_type = 'imputer'
        potential_parameters = {"impute_strategy": ["mean", "median", "most_frequent"]}

        imputer = SimpleImputer(impute_strategy=impute_strategy)
        if hyperparameters:
            imputer = SimpleImputer(**hyperparameters)

        super().__init__(name=name, component_type=component_type, potential_parameters=potential_parameters, hyperparameters=hyperparameters, needs_fitting=True, component_obj=imputer)

    def fit(self, X, objective_fit_size=.2):
        self.component_obj.fit(X)

    def transform(self, X):
        self.component_obj.transform(x)

    def fit_transform(self, X, y, objective_fit_size=.2):
        self.component_obj.fit_transform(x)
