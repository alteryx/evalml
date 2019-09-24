from transformer import Transformer

from sklearn.impute import SimpleImputer

class SimpleImputer(Transformer):
    def __init__(self, impute_strategy="most_frequent", hyperparameters=None):
        self.potential_parameters = {"impute_strategy": ["mean", "median", "most_frequent"]}

        imputer = SimpleImputer(impute_strategy=impute_strategy)
        if hyperparameters:
            imputer = SimpleImputer(**hyperparameters)
        super().__init__(name='Simple Imputer', component_type='imputer', hyperparameters=hyperparameters, needs_fitting=True, component_obj=SimpleImputer(**hyperparameters))

    def fit(self, X, objective_fit_size=.2):
        self.component_obj.fit(X)

    def transform(self, X):
        self.component_obj.transform(x)

    def fit_transform(self, X, y, objective_fit_size=.2):
        self.component_obj.fit_transform(x)
