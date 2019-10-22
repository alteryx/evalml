import numpy as np
from skopt.space import Integer, Real

from evalml.model_types import ModelTypes
from evalml.pipelines import PipelineBase
from evalml.pipelines.components import (
    OneHotEncoder,
    SelectFromModel,
    SimpleImputer,
    XGBoostClassifier
)
from evalml.problem_types import ProblemTypes


class XGBoostPipeline(PipelineBase):
    """XGBoost Pipeline for both binary and multiclass classification"""
    name = "XGBoost Classifier w/ One Hot Encoder + Simple Imputer + Select From Model"
    model_type = ModelTypes.XGBOOST
    problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    hyperparameters = {
        "eta": Real(0, 1),
        "min_child_weight": Real(1, 10),
        "max_depth": Integer(1, 20),
        "impute_strategy": ["mean", "median", "most_frequent"],
        "percent_features": Real(.01, 1)
    }

    def __init__(self, objective, eta, min_child_weight, max_depth, impute_strategy,
                 percent_features, number_features, n_jobs=1, random_state=0):

        imputer = SimpleImputer(impute_strategy=impute_strategy)
        enc = OneHotEncoder()
        estimator = XGBoostClassifier(
            random_state=random_state,
            eta=eta,
            max_depth=max_depth,
            min_child_weight=min_child_weight
        )
        feature_selection = SelectFromModel(
            estimator=estimator._component_obj,
            number_features=number_features,
            percent_features=percent_features,
            threshold=-np.inf
        )
        super().__init__(objective=objective, name=self.name, problem_type=self.problem_types, component_list=[enc, imputer, feature_selection, estimator])
