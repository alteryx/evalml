from skopt.space import Integer, Real

from evalml.model_types import ModelTypes
from evalml.pipelines import PipelineBase


class CatBoostRegressionPipeline(PipelineBase):
    """
    CatBoost Pipeline for regression problems.
    CatBoost is an open-source library and natively supports categorical features.

    For more information, check out https://catboost.ai/

    Note: impute_strategy must support both string and numeric data
    """
    name = "CatBoost Regressor w/ Simple Imputer"
    model_type = ModelTypes.CATBOOST
    component_graph = ['Simple Imputer', 'CatBoost Regressor']
    problem_types = ['regression']
    hyperparameters = {
        "impute_strategy": ["most_frequent"],
        "n_estimators": Integer(10, 1000),
        "eta": Real(0, 1),
        "max_depth": Integer(1, 8),
    }

    def __init__(self, objective, parameters):
        super().__init__(objective=objective,
                         parameters=parameters,
                         component_graph=self.__class__.component_graph,
                         problem_types=self.__class__.problem_types)
