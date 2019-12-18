import numpy as np

from evalml.model_types import ModelTypes
from evalml.pipelines import PipelineBase
from evalml.pipelines.components import (
    OneHotEncoder,
    RFClassifierSelectFromModel,
    SimpleImputer,
    XGBoostClassifier
)
from evalml.problem_types import ProblemTypes


class XGBoostPipeline(PipelineBase):
    """XGBoost Pipeline for both binary and multiclass classification"""
    # model_type = ModelTypes.XGBOOST
    # problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def __init__(self, objective, eta, min_child_weight, max_depth, impute_strategy,
                 percent_features, number_features, n_estimators=10, n_jobs=-1, random_state=0):

        imputer = SimpleImputer(impute_strategy=impute_strategy)
        enc = OneHotEncoder()
        feature_selection = RFClassifierSelectFromModel(n_estimators=n_estimators,
                                                        max_depth=max_depth,
                                                        number_features=number_features,
                                                        percent_features=percent_features,
                                                        threshold=-np.inf,
                                                        n_jobs=n_jobs,
                                                        random_state=random_state)
        estimator = XGBoostClassifier(random_state=random_state,
                                      eta=eta,
                                      max_depth=max_depth,
                                      min_child_weight=min_child_weight)

        super().__init__(objective=objective,
                         component_list=[enc, imputer, feature_selection, estimator],
                         n_jobs=n_jobs,
                         random_state=random_state)
