from sklearn.impute import SimpleImputer as SkImputer

from evalml.pipelines.components import ComponentTypes
from evalml.pipelines.components.transformers import Transformer


class SimpleImputer(Transformer):
    """Imputes missing data with either mean, median and most_frequent for numerical data or most_frequent for categorical data"""
    name = 'Simple Imputer'
    component_type = ComponentTypes.IMPUTER
    hyperparameter_ranges = {"impute_strategy": ["mean", "median", "most_frequent"]}

    def __init__(self, impute_strategy="most_frequent"):
        parameters = {"impute_strategy": impute_strategy}
        imputer = SkImputer(strategy=impute_strategy)
        super().__init__(name=self.name,
                         component_type=self.component_type,
                         parameters=parameters,
                         component_obj=imputer,
                         needs_fitting=True,
                         random_state=0)
