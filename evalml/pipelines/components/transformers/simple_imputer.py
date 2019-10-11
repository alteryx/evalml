from sklearn.impute import SimpleImputer as SkImputer

from .transformer import Transformer

from evalml.pipelines.components import ComponentTypes


class SimpleImputer(Transformer):
    """Imputes missing data with either mean, median and most_frequent for numerical data or most_frequent for categorical data"""
    hyperparameters = {"impute_strategy": ["mean", "median", "most_frequent"]}

    def __init__(self, impute_strategy="most_frequent"):
        self.name = 'Simple Imputer'
        self.component_type = ComponentTypes.IMPUTER
        self.impute_strategy = impute_strategy
        self.parameters = {"impute_strategy": self.impute_strategy}
        imputer = SkImputer(strategy=impute_strategy)
        super().__init__(name=self.name, component_type=self.component_type, parameters=self.parameters, needs_fitting=True, component_obj=imputer)
