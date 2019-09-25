from transformer import Transformer

import category_encoders as ce

class OneHotEncoder(Transformer):
    def __init__(self):
        name = 'One Hot Encoder'
        component_type = 'encoder'
        hyper_parameters = None

        encoder = ce.OneHotEncoder(use_cat_names=True, return_df=True)
        super().__init__(name=name, component_type=component_type, hyper_parameters=hyper_parameters, needs_fitting=True, component_obj=encoder)

    def fit(self, X, objective_fit_size=.2):
        self.component_obj.fit(X)

    def transform(self, X):
        self.component_obj.transform(x)

    def fit_transform(self, X, y, objective_fit_size=.2):
        self.component_obj.fit_transform(x)
