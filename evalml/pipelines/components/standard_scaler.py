from transformer import Transformer

from sklearn.preprocessing import StandardScaler

class OneHotEncoder(Transformer):
    def __init__(self, impute_strategy="most_frequent", hyperparameters=None):
        name = 'Standard Scaler'
        component_type = 'scaler'
        potential_parameters = Noneqq

        scaler = StandardScaler()
        if hyperparameters:
            scaler = StandardScaler()

        super().__init__(name=name, component_type=component_type, potential_parameters=potential_parameters, hyperparameters=hyperparameters, needs_fitting=True, component_obj=encoder)

    def fit(self, X, objective_fit_size=.2):
        self.component_obj.fit(X)

    def transform(self, X):
        self.component_obj.transform(x)

    def fit_transform(self, X, y, objective_fit_size=.2):
        self.component_obj.fit_transform(x)
