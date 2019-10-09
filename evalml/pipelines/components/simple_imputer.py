from sklearn.impute import SimpleImputer as SkImputer

from .component_types import ComponentTypes
from .transformer import Transformer


class SimpleImputer(Transformer):
    "Imputes missing data with either mean, median and most_frequent for numerical data or most_frequent for categorical data"
    def __init__(self, impute_strategy="most_frequent"):
        self.name = 'Simple Imputer'
        self.component_type = ComponentTypes.IMPUTER
        self.impute_strategy = impute_strategy
        self.hyperparameters = {"impute_strategy": ["mean", "median", "most_frequent"]}
        self.parameters = {"impute_strategy": self.impute_strategy}
        imputer = SkImputer(strategy=impute_strategy)
        super().__init__(name=self.name, component_type=self.component_type, hyperparameters=self.hyperparameters, parameters=self.parameters, needs_fitting=True, component_obj=imputer)
