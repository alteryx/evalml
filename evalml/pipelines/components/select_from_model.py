from transformer import Transformer

from sklearn.feature_selection import SelectFromModel

class SelectFromModel(Transformer):
    def __init__(self):
        name = 'Select From Model'
        component_type = 'feature_selection'
        potential_parameters = None

        feature_selection = SelectFromModel(
            estimator=estimator,
            max_features=max(1, int(percent_features * number_features)),
            threshold=-np.inf
        )
        if hyperparameters:
            scaler = StandardScaler()

        super().__init__(name=name, component_type=component_type, hyper_parameters=hyper_parameters, needs_fitting=True, component_obj=encoder)

    def fit(self, X, objective_fit_size=.2):
        self.component_obj.fit(X)

    def transform(self, X):
        self.component_obj.transform(x)

    def fit_transform(self, X, y, objective_fit_size=.2):
        self.component_obj.fit_transform(x)
