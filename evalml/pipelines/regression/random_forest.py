import numpy as np
from skopt.space import Integer, Real

from evalml.model_types import ModelTypes
from evalml.pipelines import PipelineBase
from evalml.pipelines.components import (
    OneHotEncoder,
    RandomForestRegressor,
    RFRegressorSelectFromModel,
    SimpleImputer
)
from evalml.problem_types import ProblemTypes


class RFRegressionPipeline(PipelineBase):
    """Random Forest Pipeline for regression problems"""
    name = "Random Forest Regressor w/ One Hot Encoder + Simple Imputer + RF Regressor Select From Model"
    model_type = ModelTypes.RANDOM_FOREST
    problem_types = [ProblemTypes.REGRESSION]

    hyperparameters = {
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 32),
        "impute_strategy": ["mean", "median", "most_frequent"],
        "percent_features": Real(.01, 1)
    }

    def __init__(self, objective, n_estimators, max_depth, impute_strategy, percent_features,
                 number_features, n_jobs=-1, random_state=0):

        imputer = SimpleImputer(impute_strategy=impute_strategy)
        enc = OneHotEncoder()
        feature_selection = RFRegressorSelectFromModel(n_estimators=n_estimators,
                                                       max_depth=max_depth,
                                                       number_features=number_features,
                                                       percent_features=percent_features,
                                                       threshold=-np.inf,
                                                       n_jobs=n_jobs,
                                                       random_state=0)
        estimator = RandomForestRegressor(random_state=0,
                                          n_estimators=n_estimators,
                                          max_depth=max_depth,
                                          n_jobs=n_jobs)

        super().__init__(objective=objective,
                         component_list=[enc, imputer, feature_selection, estimator],
                         n_jobs=n_jobs,
                         random_state=random_state)
