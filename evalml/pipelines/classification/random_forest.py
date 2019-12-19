import numpy as np

from evalml.pipelines import PipelineBase
from evalml.pipelines.components import (
    OneHotEncoder,
    RandomForestClassifier,
    RFClassifierSelectFromModel,
    SimpleImputer
)


class RFClassificationPipeline(PipelineBase):
    """Random Forest Pipeline for both binary and multiclass classification"""

    def __init__(self, objective, n_estimators, max_depth, impute_strategy,
                 percent_features, number_features, n_jobs=-1, random_state=0):

        imputer = SimpleImputer(impute_strategy=impute_strategy)
        enc = OneHotEncoder()
        estimator = RandomForestClassifier(n_estimators=n_estimators,
                                           max_depth=max_depth,
                                           n_jobs=n_jobs,
                                           random_state=random_state)
        feature_selection = RFClassifierSelectFromModel(n_estimators=n_estimators,
                                                        max_depth=max_depth,
                                                        number_features=number_features,
                                                        percent_features=percent_features,
                                                        threshold=-np.inf,
                                                        n_jobs=n_jobs,
                                                        random_state=random_state)

        super().__init__(objective=objective,
                         component_list=[enc, imputer, feature_selection, estimator],
                         n_jobs=n_jobs,
                         random_state=random_state)
