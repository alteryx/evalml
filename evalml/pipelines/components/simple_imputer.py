from sklearn.impute import SimpleImputer as SkImputer

from .component_types import ComponentTypes
from .transformer import Transformer


class SimpleImputer(Transformer):
    "Imputes missing data with either mean, median and most_frequent for numerical data or most_frequent for categorical data"
    def __init__(self, impute_strategy="most_frequent"):
        name = 'Simple Imputer'
        component_type = ComponentTypes.IMPUTER
        hyperparameters = {"impute_strategy": ["mean", "median", "most_frequent"]}

        imputer = SkImputer(strategy=impute_strategy)
        super().__init__(name=name, component_type=component_type, hyperparameters=hyperparameters, needs_fitting=True, component_obj=imputer)
